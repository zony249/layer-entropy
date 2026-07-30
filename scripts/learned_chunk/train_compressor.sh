#!/bin/bash
#SBATCH --account=aip-lilimou
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=l40s:4
#SBATCH --mem=64G
#SBATCH --time=0-03:00
#SBATCH --job-name=qwen-cr-5-frozen-chunker
#SBATCH --output=logs/%j--%x-a-%a.log


export NAMES=("pos-control" "neg-control" "unif-cr-5")
export MASK_MODES=("full" "contextless" "hard")

nvidia-smi 

export HF_HOME=$SCRATCH
export WANDB_PROJECT="learned-chunks"
export WANDB_NAME="qwen-unif-cr-5-soft"

    # --config_file="accel_config/fsdp2.yaml" \
accelerate launch \
    --config_file="accel_config/fsdp2.yaml" \
    train_compressor.py \
        --chunking_model=/home/zonglin1/scratch/runs/qwen-chunker-layers-8-lower-lr/checkpoint-4000 \
        --model=Qwen/Qwen3-0.6B \
        --output_dir=$SCRATCH/runs/qwen-cr-5-frozen-chunker \
        --lr=2e-5 \
        --chunk_lr=0 \
        --epochs=3 \
        --per_device_batch_size=4 \
        --gradient_accumulation_steps=4 \
        --eval_steps=500 \
        --compression_rate=5 \
        --mask_mode="soft" \
        --alpha_unif=0 \
        