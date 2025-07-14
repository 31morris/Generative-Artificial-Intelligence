#!/bin/bash
export OUTPUT_DIR="./checkpoints/0617_1"
export CFLAGS="-I/usr/include"
export LDFLAGS="-L/usr/lib/x86_64-linux-gnu"
export CUTLASS_PATH="/home/user/cutlass"
export TORCH_USE_CUDA_DSA=1

if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir "$OUTPUT_DIR"
    echo "Folder '$OUTPUT_DIR' created"
else
    echo "Folder '$OUTPUT_DIR' already exists"
fi

deepspeed --hostfile=hostfile.txt main.py \
    --deepspeed="./configs/zero2.json" \
    --dataset_path="./public_data/train_info.json" \
    --img_path="./public_data/train" \
    --pretrained_text_encoder_name_or_path="openai/clip-vit-base-patch32" \
    --pretrained_vision_encoder_name_or_path="CompVis/stable-diffusion-v1-4" \
    --output_dir=$OUTPUT_DIR \
    --train_batch_size=32 \
    --max_train_steps=300000 \
    --checkpointing_period=500 \
    --checkpoints_total_limit=10 \
    --learning_rate=1e-6 \
    --dataloader_num_workers=8 \
    --gradient_accumulation_steps=4 \
    --lr_scheduler="constant" \
    --lr_warmup_steps=2000 \
    --lr_num_cycles=1 \
    --lr_power=1.0 \
    --max_grad_norm=1.0 \
    --allow_tf32 \
    --mixed_precision="bf16" \
    --resume_from_checkpoint="latest"
    # --use_ema \