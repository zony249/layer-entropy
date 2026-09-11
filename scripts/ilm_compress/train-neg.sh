#!/bin/bash
#SBATCH --account=aip-lilimou
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=l40s:4
#SBATCH --mem=64G
#SBATCH --time=0-03:00
#SBATCH --job-name=qwen-neg
#SBATCH --output=logs/%j--%x-a-%a.log


nvidia-smi 

export CR=5
export WANDB_PROJECT=ilm-compress
export WANDB_NAME="qwen-neg"

accelerate launch \
    --config_file=accel_config/fsdp2.yaml \
    train.py \
        --gradient_accumulation_steps=4 \
        --output_dir=$SCRATCH/runs/qwen-neg \
        --add_gist \
        --compression_rate=$CR \
        --gist_scheme="dispersed" \
        --attention_mask_mode="contextless" \
        --compression_mode="none" \ 
