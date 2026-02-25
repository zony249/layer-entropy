#!/bin/bash 
#SBATCH --account=aip-lilimou 
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=l40s:1
#SBATCH --mem=160G
#SBATCH --time=0-12:00
#SBATCH --job-name=qwen-0.6b-finetune-squad
#SBATCH --output=logs/%j--qwen-0.6b-finetune-squad.log

export CUDA_VISIBLE_DEVICES=4,5
# export HF_HOME=$SCRATCH 
# export DEBUG_MODE=1
export WANDB_PROJECT="fourier-compression"
export WANDB_NAME="fsdp-test"
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

accelerate launch \
    --config_file="accel_config/fsdp2.yaml" \
        distillation.py \
        --output_dir="runs/tests" \
        --eval_steps=10 \
        --gradient_accumulation_steps=8 \