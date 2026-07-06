#!/bin/bash 
#SBATCH --account=aip-lilimou 
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=l40s:4
#SBATCH --mem=80G
#SBATCH --time=1-12:00
#SBATCH --job-name=qwen-0.6b-fourier-compress
#SBATCH --output=logs/%j--qwen-fourier-compress.log

# export CUDA_VISIBLE_DEVICES=4,5
export HF_HOME=$SCRATCH 
# export DEBUG_MODE=1
export WANDB_PROJECT="fourier-compression"
export WANDB_NAME="qwen-fourier--cr-5"
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

accelerate launch \
    --config_file="accel_config/fsdp2.yaml" \
        train.py \
        --output_dir="runs/qwen-fourier--cr-5" \
        --eval_steps=500 \
        --lr=1e-5 \
        --gradient_accumulation_steps=4 \
        --attention_mask_mode="compression" \
        --add_gist \
        --compression_mode="fourier" \
        --gist_scheme="end" \
        --compression_rate=5 \