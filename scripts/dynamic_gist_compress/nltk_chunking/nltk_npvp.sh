#!/bin/bash 
#SBATCH --account=aip-lilimou 
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=l40s:4
#SBATCH --mem=128G
#SBATCH --time=0-12:00
#SBATCH --job-name=qwen-0.6b-dynamic-nltk-npvp-1e-5
#SBATCH --output=logs/%j--%x.log

# export CUDA_VISIBLE_DEVICES=4,5
export HF_HOME=$SCRATCH 
# export DEBUG_MODE=1
export WANDB_PROJECT="fourier-compression"
export WANDB_NAME="qwen-dynamic-nltk-npvp-1e-5"
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

nvidia-smi

    # --config_file="accel_config/fsdp2.yaml" \
accelerate launch \
    --config_file="accel_config/fsdp2.yaml" \
        train.py \
        --output_dir=$SCRATCH/runs/qwen-nltk-npvp-1e-5 \
        --eval_steps=500 \
        --gradient_accumulation_steps=4 \
        --lr=1e-5 \
        --attention_mask_mode="compression" \
        --add_gist \
        --compression_mode="none" \
        --gist_scheme="dispersed" \
        --nltk_chunker np vp \