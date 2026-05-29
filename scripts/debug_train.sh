#!/bin/bash 
#SBATCH --account=aip-lilimou 
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=l40s:1
#SBATCH --mem=128G
#SBATCH --time=2-00:00
#SBATCH --job-name=qwen-0.6b-finetune-squad
#SBATCH --output=logs/%j--qwen-0.6b-debug-train.log

# export CUDA_VISIBLE_DEVICES=4,5
export HF_HOME=$SCRATCH 
export DEBUG_MODE=1
export WANDB_PROJECT="fourier-compression"
export WANDB_NAME="uniform-compress-dev"
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

nvidia-smi

    # --config_file="accel_config/fsdp2.yaml" \
accelerate launch \
    --num_processes=1 \
        train.py \
        --output_dir=$SCRATCH/runs/compress-dev \
        --eval_steps=500 \
        --gradient_accumulation_steps=4 \
        --attention_mask_mode="compression" \
        --add_gist \
        --compression_mode="none" \
        --gist_scheme="dispersed" \
        --compression_rate=5 \
        --act_guided_chunking=normdiff \
        --chunking_model=Qwen/Qwen3-0.6B \