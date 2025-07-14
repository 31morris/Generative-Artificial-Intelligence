#!/bin/bash
export OUTPUT_DIR="../model"
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

deepspeed --hostfile=hostfile.txt args.py \
    --deepspeed="./configs/zero0.json" \
    --img_path="../data" \
    --output_dir=$OUTPUT_DIR \
    --train_batch_size=128 \
    --max_train_steps=200000 \
    --checkpointing_period=500 \
    --checkpoints_total_limit=10 \
    --learning_rate=5e-6 \
    --dataloader_num_workers=8 \
    --gradient_accumulation_steps=1 \
    --lr_scheduler="constant" \
    # --resume_from_checkpoint="latest"