export NCCL_IB_DISABLE=1
export NCCL_NVLS_ENABLE=0
export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=error

export OUTPUT_DIR="./checkpoints/0327_3"
export CFLAGS="-I/usr/include"
export LDFLAGS="-L/usr/lib/x86_64-linux-gnu"
export CUTLASS_PATH="/home/user/cutlass"

if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir "$OUTPUT_DIR"
    echo "Folder '$OUTPUT_DIR' created"
else
    echo "Folder '$OUTPUT_DIR' already exists"
fi

deepspeed --hostfile=hostfile.txt main.py \
    --dataset_path="./data/train.json" \
    --output_dir=$OUTPUT_DIR \
    --deepspeed="./configs/zero3.json" \
    --pretrained_model_name_or_path="google/flan-t5-base" \
    --train_batch_size=2 \
    --max_train_steps=200000 \
    --learning_rate=1e-4 \
    --checkpoint_total_limit=40 \
    --gradient_update_period=100 \
    --mixed_precision="no" \
    --lr_scheduler="polynomial" \
    --dataloader_num_workers=4 \
    --gradient_accumulation_steps=8 \
    --use_8bit_adam \
    --set_grads_to_none \
    --scale_lr \
    --resume_from_checkpoint="latest" 