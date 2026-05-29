#!/bin/bash 
#SBATCH --account=aip-lilimou 
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-node=l40s:2
#SBATCH --mem=64G
#SBATCH --time=2-00:00
#SBATCH --job-name=qwen-0.6b-neg-control
#SBATCH --output=logs/%j--qwen-neg-control.log

# export CUDA_VISIBLE_DEVICES=4,5
export HF_HOME=$SCRATCH 
# export DEBUG_MODE=1
export WANDB_PROJECT="fourier-compression"
export WANDB_NAME="qwen-neg-control-cr-1"
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

    # --config_file="accel_config/fsdp2.yaml" \
accelerate launch \
    --config_file="accel_config/fsdp2.yaml" \
        train.py \
        --output_dir="runs/qwen-neg-control" \
        --eval_steps=500 \
        --gradient_accumulation_steps=4 \
        --attention_mask_mode="contextless" \
        --add_gist \