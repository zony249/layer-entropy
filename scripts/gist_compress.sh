#!/bin/bash 
#SBATCH --account=aip-lilimou 
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=l40s:1
#SBATCH --mem=64G
#SBATCH --time=2-00:00
#SBATCH --job-name=qwen-0.6b-gist-compress
#SBATCH --output=logs/%j--qwen-gist-compress.log

# export CUDA_VISIBLE_DEVICES=4,5
export HF_HOME=$SCRATCH 
# export DEBUG_MODE=1
export WANDB_PROJECT="fourier-compression"
export WANDB_NAME="qwen-gist-compress-cr-20"
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

    # --config_file="accel_config/fsdp2.yaml" \
accelerate launch \
        distillation.py \
        --output_dir="runs/qwen-gist-compress-cr-20" \
        --eval_steps=20 \
        --gradient_accumulation_steps=4 \
        --attention_mask_mode="compression" \
        --add_gist \
        --compression_mode="none" \
        --gist_scheme="end" \
        --compression_rate=20 \