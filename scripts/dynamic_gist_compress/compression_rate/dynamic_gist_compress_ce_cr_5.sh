#!/bin/bash 
#SBATCH --account=aip-lilimou 
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=l40s:1
#SBATCH --mem=40G
#SBATCH --time=2-00:00
#SBATCH --job-name=qwen-0.6b-dyna-compress
#SBATCH --output=logs/%j--qwen-dyna-compress-ce-cr-5.log

# export CUDA_VISIBLE_DEVICES=4,5
export HF_HOME=$SCRATCH 
# export DEBUG_MODE=1
export WANDB_PROJECT="fourier-compression"
export WANDB_NAME="qwen-dyna-compress-ce-cr-5"
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

nvidia-smi

    # --config_file="accel_config/fsdp2.yaml" \
accelerate launch \
        distillation.py \
        --output_dir=$SCRATCH/runs/qwen-dyna-compress-ce-cr-5 \
        --eval_steps=500 \
        --gradient_accumulation_steps=4 \
        --attention_mask_mode="compression" \
        --add_gist \
        --compression_mode="average" \
        --gist_scheme="dispersed" \
        --compression_rate=5 \
        --entropy_model=Qwen/Qwen3-0.6B \
        --surprise_mode="ce" \
        --entropy_model_temp=1 \