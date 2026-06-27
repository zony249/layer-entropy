#!/bin/bash
#SBATCH --account=aip-lilimou
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=l40s:4
#SBATCH --mem=96G
#SBATCH --time=0-16:00
#SBATCH --job-name=qwen-0.6b-unif-compress
#SBATCH --output=logs/%j--%x.log

# export CUDA_VISIBLE_DEVICES=4,5
export HF_HOME=$SCRATCH
# export DEBUG_MODE=1
export WANDB_PROJECT="fourier-compression"
export WANDB_NAME="qwen-unif-compress-cr-5"
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

nvidia-smi

    # --config_file="accel_config/fsdp2.yaml" \
accelerate launch \
    --config_file="accel_config/fsdp2.yaml" \
    train.py \
        --output_dir=$SCRATCH/runs/qwen-unif-compress-cr-5 \
        --eval_steps=500 \
        --gradient_accumulation_steps=4 \
        --lr=1e-5 \
        --add_gist \
        --compression_rate=5 \
        --attention_mask_mode="compression" \
        --compression_mode="none" \
        --gist_scheme="dispersed" \
        # --entropy_model=Qwen/Qwen3-0.6B \
        # --surprise_mode="ce" \
        # --entropy_model_temp=1 \
