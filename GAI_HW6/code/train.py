#!/usr/bin/env python
# coding=utf-8
import logging
import math
import os
import yaml
from pathlib import Path
import datasets
import lpips
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import DeepSpeedPlugin, ProjectConfiguration, set_seed
from transformers import CLIPTextModel, CLIPTokenizer
import diffusers
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers import AutoencoderKL, DDPMPipeline, DDPMScheduler, UNet2DConditionModel
from tqdm.auto import tqdm
from PIL import Image
from glob import glob
import json
import random
from torch.utils.data import Dataset
from torchvision import transforms


class ConsumerDataset(Dataset):
    def __init__(self, data_root, caption_file, tokenizer, size=256):
        self.data_root = data_root
        self.tokenizer = tokenizer
        with open(caption_file, 'r') as f:
            self.captions = json.load(f)
        self.image_files = glob(os.path.join(data_root, "*.png"))
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_file = self.image_files[idx]

        ## Load image
        image = Image.open(img_file).convert("RGB")
        image = self.transform(image)

        ## Load text prompt
        # Image file name: "mosterID_ACTION_frameID.png"
        # key in "train_ingo.json": "mosterID_ACTION"
        key = img_file.split("/")[-1].split(".")[0]
        key = "_".join(key.split("_")[:-1])

        # Sample caption =  moster description + action description
        given_descriptions = self.captions[key]['given_description']
        given_description = random.choice(given_descriptions)
        caption = f"{given_description} {self.captions[key]['action_description']}"
        caption = "" if random.random() < 0.1 else caption
        inputs = self.tokenizer(caption, return_tensors="pt", padding="max_length", truncation=True, max_length=77).input_ids
        
        return {
            "pixel_values": image,
            "input_ids": inputs.squeeze(0).long(),
        }

def train(args, logger):
    # Read the config
    with open(args.config_path, "r") as fp:
        config = yaml.safe_load(fp)

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir,
        logging_dir=logging_dir,
        total_limit=args.checkpoints_total_limit
    )
    
    accelerator = Accelerator(
        deepspeed_plugin=DeepSpeedPlugin(
            hf_ds_config=args.deepspeed
        ) if args.deepspeed is not None else None,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_dir=logging_dir,
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    
                                                
    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # Load from a pretrained checkpoint

    logger.info("Constructing model from provided config.")

    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_text_encoder_name_or_path)
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_text_encoder_name_or_path).eval()
    text_encoder.requires_grad_(False)

    vae = AutoencoderKL.from_pretrained(args.pretrained_vision_encoder_name_or_path, subfolder="vae")
    vae.requires_grad_(False)

    model = UNet2DConditionModel(
        sample_size=config["model"]["unet"]["sample_size"],
        in_channels=config["model"]["unet"]["in_channels"],
        out_channels=config["model"]["unet"]["out_channels"],
        layers_per_block=config["model"]["unet"]["layers_per_block"],
        block_out_channels=config["model"]["unet"]["block_out_channels"],
        down_block_types=config["model"]["unet"]["down_block_types"],
        up_block_types=config["model"]["unet"]["up_block_types"],
        cross_attention_dim=config["model"]["unet"]["cross_attention_dim"],
        mid_block_type=config["model"]["unet"]["mid_block_type"],
        norm_num_groups=config["model"]["unet"]["norm_num_groups"], 
        attention_head_dim=config["model"]["unet"]["attention_head_dim"],
    )

    if args.use_ema:
        ema_model = EMAModel(
            model.parameters(),
            decay=config["model"]["ema"]["ema_max_decay"],
            use_ema_warmup=config["model"]["ema"]["use_ema_warmup"],
            inv_gamma=config["model"]["ema"]["inv_gamma"],
            power=config["model"]["ema"]["power"],
            model_cls=UNet2DConditionModel,
            model_config=model.config,
        )

    noise_scheduler_config = config["model"]['noise_scheduler']
    noise_scheduler = DDPMScheduler(
            num_train_timesteps=noise_scheduler_config['num_train_timesteps'],
            beta_schedule=noise_scheduler_config['beta_schedule'],
            clip_sample=noise_scheduler_config['clip_sample'],
        )

    # Optimizer creation
    params_to_optimize = model.parameters()
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Dataloader
    train_dataset =  ConsumerDataset(args.img_path, args.dataset_path, tokenizer)

    # DataLoader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        persistent_workers=True
    )
    
    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    lpips_fn = lpips.LPIPS(net='vgg').to(accelerator.device)
    lpips_fn.eval()

    if args.use_ema:
        ema_model.to(accelerator.device)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("GAI", config=vars(args))

    # Train
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0
    
    # Load from a pretrained checkpoint
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the mos recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            logger.info("Loading from a pretrained checkpoint.")
            accelerator.load_state(os.path.join(args.output_dir, path))
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.learning_rate

            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    loss_for_log = {}
    for epoch in range(first_epoch, args.num_train_epochs):
        logger.info(f"==============================  training {epoch+1} ========================================")
        
        model.train()
        
        # Set the progress_bar to correct position
        if args.resume_from_checkpoint and epoch == first_epoch:
            progress_bar.update(resume_step // args.gradient_accumulation_steps)
        
        # Forward and backward...
        for batch in train_dataloader:
       
            with accelerator.accumulate(model):
                input_ids = batch["input_ids"].to(device=accelerator.device).long()
                pixel_values = batch["pixel_values"].to(device=accelerator.device, dtype=weight_dtype)
                if args.debug_model:
                    print(f"input_ids dtype: {input_ids.dtype}")
                    print(f"pixel_values dtype: {pixel_values.dtype}")

                with torch.no_grad():
                    encoder_hidden_states = text_encoder(input_ids)[0]
                    latents = vae.encode(pixel_values).latent_dist.sample()
                    latents = latents * 0.18215  # Scaling
                
                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=accelerator.device).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                noise_pred = model(noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states).sample
                
                loss = F.mse_loss(noise_pred.float(), noise.float())

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    params_to_clip = model.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                optimizer.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)            

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_model.step(model.parameters())
                progress_bar.update(1)
                global_step += 1

                if global_step % args.checkpointing_period == 0:
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    if args.use_ema:
                        ema_save_path = os.path.join(save_path, f"ema")
                        os.makedirs(ema_save_path, exist_ok=True)
                        ema_model.save_pretrained(ema_save_path)
                    logger.info(f"Saved state to {save_path}")

            logs = {"loss": loss.detach().item(), "lr": optimizer.param_groups[0]['lr']}

            if args.use_ema:
                logs["ema_decay"] = ema_model.cur_decay_value

            progress_bar.set_postfix(**logs)
            logs.update(loss_for_log)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        accelerator.unwrap_model(model).save_pretrained(args.output_dir)

        if args.use_ema:
            ema_save_path = os.path.join(args.output_dir, f"ema")
            accelerator.save_model(ema_model, ema_save_path)
            ema_model.store(model.parameters())
            ema_model.copy_to(model.parameters())
        
        pipeline = DDPMPipeline(
            unet=model,
            scheduler=noise_scheduler,
        )

        pipeline.save_pretrained(args.output_dir)

        if args.use_ema:
            ema_model.restore(model.parameters())
    
        logger.info(f"Saved Model to {args.output_dir}")

            
    accelerator.end_training()

# #!/usr/bin/env python
# # coding=utf-8
# import logging
# import math
# import os
# import yaml
# from pathlib import Path
# import datasets
# import lpips
# import torch
# from torch.utils.data import DataLoader
# import torch.nn.functional as F
# from accelerate import Accelerator
# from accelerate.utils import DeepSpeedPlugin, ProjectConfiguration, set_seed
# from transformers import CLIPTextModel, CLIPTokenizer
# import diffusers
# from diffusers.optimization import get_scheduler
# from diffusers.training_utils import EMAModel
# from diffusers import AutoencoderKL, DDPMPipeline, DDPMScheduler, UNet2DConditionModel
# from tqdm.auto import tqdm
# from PIL import Image
# import os
# from glob import glob
# import json
# import random
# from torch.utils.data import Dataset
# from torchvision import transforms


# class ConsumerDataset(Dataset):
#     def __init__(self, data_root, caption_file, tokenizer, size=256):
#         self.data_root = data_root
#         self.tokenizer = tokenizer
#         with open(caption_file, 'r') as f:
#             self.captions = json.load(f)
#         self.image_files = glob(os.path.join(data_root, "*.png"))
#         self.transform = transforms.Compose([
#             transforms.Resize((size, size)),
#             transforms.ToTensor(),
#             transforms.RandomHorizontalFlip(),
#             transforms.Normalize([0.5], [0.5])
#         ])

#     def __len__(self):
#         return len(self.image_files)

#     def __getitem__(self, idx):
#         img_file = self.image_files[idx]

#         ## Load image
#         image = Image.open(img_file).convert("RGB")
#         image = self.transform(image)

#         ## Load text prompt
#         # Image file name: "mosterID_ACTION_frameID.png"
#         # key in "train_ingo.json": "mosterID_ACTION"
#         key = img_file.split("/")[-1].split(".")[0]
#         key = "_".join(key.split("_")[:-1])

#         # Sample caption =  moster description + action description
#         given_descriptions = self.captions[key]['given_description']
#         given_description = random.choice(given_descriptions)
#         caption = f"{given_description} {self.captions[key]['action_description']}"
#         caption = "" if random.random() < 0.1 else caption
#         inputs = self.tokenizer(caption, return_tensors="pt", padding="max_length", truncation=True, max_length=77).input_ids
        
#         return {
#             "pixel_values": image,
#             "input_ids": inputs.squeeze(0).long(),
#         }

# def train(args, logger):
#     # Read the config
#     with open(args.config_path, "r") as fp:
#         config = yaml.safe_load(fp)

#     logging_dir = Path(args.output_dir, args.logging_dir)

#     accelerator_project_config = ProjectConfiguration(
#         project_dir=args.output_dir,
#         logging_dir=logging_dir,
#         total_limit=args.checkpoints_total_limit
#     )
    
#     accelerator = Accelerator(
#         deepspeed_plugin=DeepSpeedPlugin(
#             hf_ds_config=args.deepspeed
#         ) if args.deepspeed is not None else None,
#         gradient_accumulation_steps=args.gradient_accumulation_steps,
#         mixed_precision=args.mixed_precision,
#         log_with=args.report_to,
#         project_dir=logging_dir,
#         project_config=accelerator_project_config,
#     )

#     # Make one log on every process with the configuration for debugging.
#     logging.basicConfig(
#         format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
#         datefmt="%m/%d/%Y %H:%M:%S",
#         level=logging.INFO,
#     )
#     logger.info(accelerator.state, main_process_only=False)
#     if accelerator.is_local_main_process:
#         datasets.utils.logging.set_verbosity_warning()
#         diffusers.utils.logging.set_verbosity_info()
#     else:
#         datasets.utils.logging.set_verbosity_error()
#         diffusers.utils.logging.set_verbosity_error()

#     # If passed along, set the training seed now.
#     if args.seed is not None:
#         set_seed(args.seed)

#     # Handle the repository creation
#     if accelerator.is_main_process:
#         if args.output_dir is not None:
#             os.makedirs(args.output_dir, exist_ok=True)

#     # For mixed precision training we cast the text_encoder and vae weights to half-precision
#     # as these models are only used for inference, keeping weights in full precision is not required.
#     weight_dtype = torch.float32
#     if accelerator.mixed_precision == "fp16":
#         weight_dtype = torch.float16
#     elif accelerator.mixed_precision == "bf16":
#         weight_dtype = torch.bfloat16
    
                                                
#     # Enable TF32 for faster training on Ampere GPUs,
#     # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
#     if args.allow_tf32:
#         torch.backends.cuda.matmul.allow_tf32 = True

#     # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
#     if args.use_8bit_adam:
#         try:
#             import bitsandbytes as bnb
#         except ImportError:
#             raise ImportError(
#                 "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
#             )

#         optimizer_class = bnb.optim.AdamW8bit
#     else:
#         optimizer_class = torch.optim.AdamW

#     # Load from a pretrained checkpoint

#     logger.info("Constructing model from provided config.")

#     tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_text_encoder_name_or_path)
#     text_encoder = CLIPTextModel.from_pretrained(args.pretrained_text_encoder_name_or_path).eval()
#     text_encoder.requires_grad_(False)

#     vae = AutoencoderKL.from_pretrained(args.pretrained_vision_encoder_name_or_path, subfolder="vae")
#     vae.requires_grad_(False)

#     model = UNet2DConditionModel(
#         sample_size=config["model"]["unet"]["sample_size"],
#         in_channels=config["model"]["unet"]["in_channels"],
#         out_channels=config["model"]["unet"]["out_channels"],
#         layers_per_block=config["model"]["unet"]["layers_per_block"],
#         block_out_channels=config["model"]["unet"]["block_out_channels"],
#         down_block_types=config["model"]["unet"]["down_block_types"],
#         up_block_types=config["model"]["unet"]["up_block_types"],
#         cross_attention_dim=config["model"]["unet"]["cross_attention_dim"],
#         mid_block_type=config["model"]["unet"]["mid_block_type"],
#         norm_num_groups=config["model"]["unet"]["norm_num_groups"], 
#         attention_head_dim=config["model"]["unet"]["attention_head_dim"],
#     )

#     if args.use_ema:
#         ema_model = EMAModel(
#             model.parameters(),
#             decay=config["model"]["ema"]["ema_max_decay"],
#             use_ema_warmup=config["model"]["ema"]["use_ema_warmup"],
#             inv_gamma=config["model"]["ema"]["inv_gamma"],
#             power=config["model"]["ema"]["power"],
#             model_cls=UNet2DConditionModel,
#             model_config=model.config,
#         )

#     noise_scheduler_config = config["model"]['noise_scheduler']
#     noise_scheduler = DDPMScheduler(
#             num_train_timesteps=noise_scheduler_config['num_train_timesteps'],
#             beta_schedule=noise_scheduler_config['beta_schedule'],
#             clip_sample=noise_scheduler_config['clip_sample'],
#         )

#     # Optimizer creation
#     params_to_optimize = model.parameters()
#     optimizer = optimizer_class(
#         params_to_optimize,
#         lr=args.learning_rate,
#         betas=(args.adam_beta1, args.adam_beta2),
#         weight_decay=args.adam_weight_decay,
#         eps=args.adam_epsilon,
#     )

#     lr_scheduler = get_scheduler(
#         args.lr_scheduler,
#         optimizer=optimizer,
#         num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
#         num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
#         num_cycles=args.lr_num_cycles,
#         power=args.lr_power,
#     )

#     # Dataloader
#     train_dataset =  ConsumerDataset(args.img_path, args.dataset_path, tokenizer)

#     # DataLoader
#     train_dataloader = DataLoader(
#         train_dataset,
#         batch_size=args.train_batch_size,
#         shuffle=True,
#         num_workers=args.dataloader_num_workers,
#         pin_memory=True,
#         persistent_workers=True
#     )
    
#     # Scheduler and math around the number of training steps.
#     overrode_max_train_steps = False
#     num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
#     if args.max_train_steps is None:
#         args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
#         overrode_max_train_steps = True

#     # Prepare everything with our `accelerator`.
#     model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
#         model, optimizer, train_dataloader, lr_scheduler
#     )
#     vae.to(accelerator.device, dtype=weight_dtype)
#     text_encoder.to(accelerator.device, dtype=weight_dtype)
#     lpips_fn = lpips.LPIPS(net='vgg').to(accelerator.device)
#     lpips_fn.eval()

#     if args.use_ema:
#         ema_model.to(accelerator.device)

#     # We need to recalculate our total training steps as the size of the training dataloader may have changed.
#     num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
#     if overrode_max_train_steps:
#         args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
#     # Afterwards we recalculate our number of training epochs
#     args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

#     # We need to initialize the trackers we use, and also store our configuration.
#     # The trackers initializes automatically on the main process.
#     if accelerator.is_main_process:
#         accelerator.init_trackers("GAI_HW6", config=vars(args))

#     # Train
#     total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

#     logger.info("***** Running training *****")
#     logger.info(f"  Num examples = {len(train_dataset)}")
#     logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
#     logger.info(f"  Num Epochs = {args.num_train_epochs}")
#     logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
#     logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
#     logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
#     logger.info(f"  Total optimization steps = {args.max_train_steps}")
#     global_step = 0
#     first_epoch = 0
    
#     # Load from a pretrained checkpoint
#     if args.resume_from_checkpoint:
#         if args.resume_from_checkpoint != "latest":
#             path = os.path.basename(args.resume_from_checkpoint)
#         else:
#             # Get the mos recent checkpoint
#             dirs = os.listdir(args.output_dir)
#             dirs = [d for d in dirs if d.startswith("checkpoint")]
#             dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
#             path = dirs[-1] if len(dirs) > 0 else None

#         if path is None:
#             accelerator.print(
#                 f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
#             )
#             args.resume_from_checkpoint = None
#         else:
#             logger.info("Loading from a pretrained checkpoint.")
#             accelerator.load_state(os.path.join(args.output_dir, path))
#             for param_group in optimizer.param_groups:
#                 param_group['lr'] = args.learning_rate

#             global_step = int(path.split("-")[1])

#             resume_global_step = global_step * args.gradient_accumulation_steps
#             first_epoch = global_step // num_update_steps_per_epoch
#             resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)

#     # Only show the progress bar once on each machine.
#     progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
#     progress_bar.set_description("Steps")

#     loss_for_log = {}
#     for epoch in range(first_epoch, args.num_train_epochs):
#         logger.info(f"==============================  training {epoch+1} ========================================")
        
#         model.train()
        
#         # Set the progress_bar to correct position
#         if args.resume_from_checkpoint and epoch == first_epoch:
#             progress_bar.update(resume_step // args.gradient_accumulation_steps)
        
#         # Forward and backward...
#         for batch in train_dataloader:
       
#             with accelerator.accumulate(model):
#                 input_ids = batch["input_ids"].to(device=accelerator.device).long()
#                 pixel_values = batch["pixel_values"].to(device=accelerator.device, dtype=weight_dtype)
#                 if args.debug_model:
#                     print(f"input_ids dtype: {input_ids.dtype}")
#                     print(f"pixel_values dtype: {pixel_values.dtype}")

#                 with torch.no_grad():
#                     encoder_hidden_states = text_encoder(input_ids)[0]
#                     latents = vae.encode(pixel_values).latent_dist.sample()
#                     latents = latents * 0.18215  # Scaling
                
#                 noise = torch.randn_like(latents)
#                 timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=accelerator.device).long()
#                 noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
#                 noise_pred = model(noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states).sample
                
#                 loss = F.mse_loss(noise_pred.float(), noise.float())

#                 accelerator.backward(loss)

#                 if accelerator.sync_gradients:
#                     params_to_clip = model.parameters()
#                     accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

#                 optimizer.step()
#                 optimizer.zero_grad(set_to_none=args.set_grads_to_none)            

#             # Checks if the accelerator has performed an optimization step behind the scenes
#             if accelerator.sync_gradients:
#                 if args.use_ema:
#                     ema_model.step(model.parameters())
#                 progress_bar.update(1)
#                 global_step += 1

#                 if global_step % args.checkpointing_period == 0:
#                     save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
#                     accelerator.save_state(save_path)
#                     if args.use_ema:
#                         ema_save_path = os.path.join(save_path, f"ema")
#                         os.makedirs(ema_save_path, exist_ok=True)
#                         ema_model.save_pretrained(ema_save_path)
#                     logger.info(f"Saved state to {save_path}")

#             logs = {"loss": loss.detach().item(), "lr": optimizer.param_groups[0]['lr']}

#             if args.use_ema:
#                 logs["ema_decay"] = ema_model.cur_decay_value

#             progress_bar.set_postfix(**logs)
#             logs.update(loss_for_log)
#             accelerator.log(logs, step=global_step)

#             if global_step >= args.max_train_steps:
#                 break

#     # Create the pipeline using using the trained modules and save it.
#     accelerator.wait_for_everyone()
#     if accelerator.is_main_process:
#         accelerator.unwrap_model(model).save_pretrained(args.output_dir)

#         if args.use_ema:
#             ema_save_path = os.path.join(args.output_dir, f"ema")
#             accelerator.save_model(ema_model, ema_save_path)
#             ema_model.store(model.parameters())
#             ema_model.copy_to(model.parameters())
        
#         pipeline = DDPMPipeline(
#             unet=model,
#             scheduler=noise_scheduler,
#         )

#         pipeline.save_pretrained(args.output_dir)

#         if args.use_ema:
#             ema_model.restore(model.parameters())
    
#         logger.info(f"Saved Model to {args.output_dir}")

            
#     accelerator.end_training()
