#!/usr/bin/env python
# coding=utf-8
import copy
import logging
import math
import os
from pathlib import Path
import transformers
import yaml
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, get_scheduler
from peft import LoraConfig, get_peft_model, PeftModel
from accelerate import Accelerator
from accelerate.utils import DeepSpeedPlugin, ProjectConfiguration, set_seed
from tqdm.auto import tqdm
from train.dataset import DataCollatorForConsumerDataset, ConsumerDataset

from safetensors.torch import load_model

def train(args, logger):
    # Set up logging
    logging_dir = Path(args.output_dir, args.logging_dir)
    
    # Set up accelerator
    accelerator_project_config = ProjectConfiguration(total_limit=args.checkpoint_total_limit)
    accelerator = Accelerator(
        deepspeed_plugin=DeepSpeedPlugin(hf_ds_config=args.deepspeed) if args.deepspeed is not None else None,
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
        transformers.utils.logging.set_verbosity_warning()
    else:
        transformers.utils.logging.set_verbosity_error()
    
    # Set seed before initializing model.
    if args.seed is not None:
        set_seed(args.seed)

    # create output directory if needed
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # set the weights dtype
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # # configure LoRA
    lora_config = LoraConfig(
        r=8,  # Low-rank的大小
        lora_alpha=32,  # LoRA的比例系数
        lora_dropout=0.1,  # Dropout比例
        target_modules=["q", "v"],  # 针对Transformer中哪些模块应用LoRA
    )

    # Load model from checkpoint if provided
    logger.info("Constructing model from checkpoint.")
    tokenizer = T5Tokenizer.from_pretrained(args.pretrained_model_name_or_path)
    model = T5ForConditionalGeneration.from_pretrained(args.pretrained_model_name_or_path)
    peft_model = get_peft_model(model, lora_config)

    # Enable TF32 for faster training on Ampere GPUs(GTX30 up or A100)
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # Use 8-bit Adam Optimizer for lower memory (16GB) usage
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

    params_to_optimize = peft_model.parameters()
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    
    # Dataloader
    train_dataset = ConsumerDataset(
        data_path=args.dataset_path,
        tokenizer=tokenizer,
    )
    # the method to batch the data
    data_collator = DataCollatorForConsumerDataset()

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=data_collator,
        num_workers=args.dataloader_num_workers,
        pin_memory=False,
        persistent_workers=True
    )
    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # Load from a pretrained checkpoint
    if (                                        #check the condition if we need to load from a pretrained checkpoint
        args.resume_from_checkpoint is None 
        and args.pretrained_model_name_or_path is not None
        and os.path.isfile(args.pretrained_model_name_or_path)
    ):
            logger.info("Loading from a pretrained checkpoint.")
            checkpoint = torch.load(args.pretrained_model_name_or_path)
            peft_model.module.load_state_dict(checkpoint["module"])

    global_step = 0
    first_epoch = 0

    # Resume training state from a checkpoint
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
        # Get the recent checkpoint
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
            accelerator.print(f"Resuming from checkpoint {path}")
            try:
                accelerator.load_state(os.path.join(args.output_dir, path)) # load_module_strict=False
            except:
                # load deepspeed's state_dict
                logger.info("Resuming training state failed. Attempting to only load from model checkpoint.")
                # checkpoint = torch.load(os.path.join(args.output_dir, path, "pytorch_model", "mp_rank_00_model_states.pt"))
                # peft_model.module.load_state_dict(checkpoint["module"])
                logger.info("Resuming training state failed. Attempting to only load from model checkpoint.")
                checkpoint = torch.load(os.path.join(args.output_dir, path))
                peft_model.module.load_state_dict(checkpoint)
                # peft_model = PeftModel.from_pretrained(model ,os.path.join(args.output_dir, path),is_trainable=True)

                
            global_step = int(path.split("-")[1])
            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)

    # Prepare everything with accelerator.
    peft_model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        peft_model, optimizer, train_dataloader, lr_scheduler                   
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # initialize the trackers we use
    if accelerator.is_main_process:
        accelerator.init_trackers("GAI_HW2", config=vars(args))

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  train from global step {global_step}")

    # show the progress bar only on the main process
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    for epoch in range(first_epoch, int(args.num_train_epochs)):
        logger.info(f"==============================start training {epoch+1}========================================")
        peft_model.train()
        if args.resume_from_checkpoint and epoch == first_epoch:
            progress_bar.update(resume_step// args.gradient_accumulation_steps)
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(peft_model):
                id = batch["ids"]
                target = batch["target"].to(dtype=weight_dtype)
                target = target.to(dtype=torch.long)
                input = batch["input"].to(dtype=weight_dtype)
                input = input.to(dtype=torch.long)
                logger.info(f"get Data {id}")

                logger.info(f"  Doing forward pass")
                # logger.info(f"input_ids shape: {input.shape}")
                # logger.info(f"attention_mask shape: {mask.shape}")
                # logger.info(f"labels shape: {target.shape}")
                loss= peft_model(input_ids=input, labels=target).loss
                # loss = outputs.loss
                # calculate the gradients and update the parameters
                logger.info(f"  Doing backward pass")
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = peft_model.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                # update the parameters
                optimizer.step()
                # update the learning rate
                lr_scheduler.step()
                # reset the gradients
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            if accelerator.sync_gradients:
                # update the progress bar and train step
                progress_bar.update(1)
                global_step += 1
                if global_step % args.gradient_update_period == 0:
                    if accelerator.is_main_process:
                        # save the model to the output directory
                        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.unwrap_model(peft_model).save_pretrained(checkpoint_dir)
                        checkpoint = {
                                        "model_state_dict": peft_model.state_dict(),
                                        "optimizer_state_dict": optimizer.state_dict(),
                                        "lr_scheduler_state_dict": lr_scheduler.state_dict(),
                                    }
                        torch.save(checkpoint, os.path.join(checkpoint_dir, "checkpoint.pth"))
                        # logger.info(f"Saving model and state checkpoint to {checkpoint_dir}")
            # show the loss and learning rate on the progress bar
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            # log the loss and learning rate
            accelerator.log(logs, step=global_step)
            if global_step >= args.max_train_steps:
                break

    # check all processes have finished
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        # save the model to the output directory
        accelerator.unwrap_model(peft_model).save_pretrained(args.output_dir)

        logger.info(f"Saving model checkpoint to {args.output_dir}")
    accelerator.end_training()

