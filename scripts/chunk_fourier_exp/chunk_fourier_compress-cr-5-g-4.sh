#!/bin/bash 
#SBATCH --account=aip-lilimou 
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=l40s:1
#SBATCH --mem=64G
#SBATCH --time=2-12:00
#SBATCH --job-name=qwen-0.6b-fourier-compress
#SBATCH --output=logs/%j--qwen-fourier-compress-cr-5-g-4.log

# export CUDA_VISIBLE_DEVICES=4,5
export HF_HOME=$SCRATCH 
# export DEBUG_MODE=1
export WANDB_PROJECT="fourier-compression"
export WANDB_NAME="qwen-chunk-fourier-cr-5-g-4"
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

nvidia-smi

    # --config_file="accel_config/fsdp2.yaml" \
accelerate launch \
        train.py \
        --output_dir="runs/qwen-chunk-fourier-cr-5-g-4" \
        --eval_steps=500 \
        --gradient_accumulation_steps=4 \
        --attention_mask_mode="compression" \
        --add_gist \
        --compression_mode="fourier" \
        --gist_scheme="dispersed" \
        --compression_rate=5 \
        --gist_granularity=4 \