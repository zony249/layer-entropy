#!/bin/bash 
#SBATCH --account=aip-lilimou 
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=l40s:4
#SBATCH --mem=128G
#SBATCH --time=0-12:00
#SBATCH --job-name=qwen-0.6b-neg-control
#SBATCH --output=logs/%j--qwen-neg-control.log

# export CUDA_VISIBLE_DEVICES=4,5
export HF_HOME=$SCRATCH 
# export DEBUG_MODE=1
export WANDB_PROJECT="fourier-compression"
export WANDB_NAME="qwen-neg-control"
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

nvidia-smi

    # --config_file="accel_config/fsdp2.yaml" \
accelerate launch \
    --config_file="accel_config/fsdp2.yaml" \
        train.py \
        --output_dir=$SCRATCH/runs/qwen-neg-control \
        --eval_steps=500 \
        --gradient_accumulation_steps=4 \
        --lr=1e-6 \
        --attention_mask_mode="contextless" \
        --add_gist \