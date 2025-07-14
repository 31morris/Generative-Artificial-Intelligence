#!/usr/bin/env python
# coding=utf-8
import logging
import math
import os
import yaml
from pathlib import Path
import datasets
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from accelerate import Accelerator
from accelerate.utils import DeepSpeedPlugin, ProjectConfiguration, set_seed

import diffusers
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel

from tqdm.auto import tqdm
from safetensors.torch import load_model

from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image, UnidentifiedImageError
import torch


class ConsumerDataset(Dataset):
    def __init__(self, data_dir, image_size=64):
        self.data_dir = data_dir
        self.image_paths = []
        for fname in os.listdir(data_dir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                path = os.path.join(data_dir, fname)
                try:
                    with Image.open(path) as img:
                        img.verify()
                    self.image_paths.append(path)
                except (OSError, UnidentifiedImageError):
                    print(f"[WARN] Skipping corrupted image: {path}")

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),  # Converts to [0,1]
            transforms.RandomHorizontalFlip(),
            transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        while idx < len(self.image_paths):
            path = self.image_paths[idx]
            try:
                image = Image.open(path).convert("RGB")
                return self.transform(image)
            except (OSError, UnidentifiedImageError) as e:
                print(f"[ERROR] Skipping corrupted image at {path}, error: {e}")
                idx += 1 
        raise IndexError("No valid image found after current index.")

class DataCollatorForConsumerDataset:
    def __call__(self, batch):
        return {
            "images": torch.stack(batch)
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

    model = UNet2DModel(
        sample_size=config["model"]["unet"]["sample_size"],
        in_channels=config["model"]["unet"]["in_channels"],
        out_channels=config["model"]["unet"]["out_channels"],
        layers_per_block=config["model"]["unet"]["layers_per_block"],
        block_out_channels=config["model"]["unet"]["block_out_channels"],
        down_block_types=config["model"]["unet"]["down_block_types"],
        up_block_types=config["model"]["unet"]["up_block_types"],
    )

    if args.use_ema:
        ema_model = EMAModel(
            model.parameters(),
            decay=config["model"]["ema"]["ema_max_decay"],
            use_ema_warmup=config["model"]["ema"]["use_ema_warmup"],
            inv_gamma=config["model"]["ema"]["inv_gamma"],
            power=config["model"]["ema"]["power"],
            model_cls=UNet2DModel,
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
    train_dataset = ConsumerDataset(args.img_path, image_size=64)

    data_collator = DataCollatorForConsumerDataset()

    # DataLoader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=data_collator,
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
        accelerator.init_trackers("GAI_HW5", config=vars(args))

    # Train!
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
            if args.use_ema:
                load_model(ema_model, os.path.join(args.output_dir, path, "ema", "model.safetensors"))
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
        logger.info(f"============================== start training {epoch+1} ========================================")
        
        model.train()
        
        # Set the progress_bar to correct position
        if args.resume_from_checkpoint and epoch == first_epoch:
            progress_bar.update(resume_step // args.gradient_accumulation_steps)
        
        # Forward and backward...
        for batch in train_dataloader:
       
            with accelerator.accumulate(model):
                img_gt = batch["images"].to(dtype=weight_dtype, device=accelerator.device)
                batch_size = img_gt.shape[0]
                
                noise = torch.randn_like(img_gt)
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (batch_size,), device=accelerator.device).long()
                noisy_latents = noise_scheduler.add_noise(img_gt, noise, timesteps)
        
                noise_pred = model(noisy_latents, timesteps).sample

                loss = F.mse_loss(noise_pred.float(), noise.float())


                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    params_to_clip = model.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
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
                    ema_save_path = os.path.join(save_path, f"ema")
                    accelerator.save_model(ema_model, ema_save_path)
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
