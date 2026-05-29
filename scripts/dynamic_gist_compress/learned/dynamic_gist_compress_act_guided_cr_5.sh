#!/bin/bash 
#SBATCH --account=aip-lilimou 
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=l40s:1
#SBATCH --mem=40G
#SBATCH --time=2-00:00
#SBATCH --job-name=qwen-0.6b-act-guided-compress-averaging
#SBATCH --output=logs/%j--qwen-act-guided-compress-averaging.log

# export CUDA_VISIBLE_DEVICES=4,5
export HF_HOME=$SCRATCH 
# export DEBUG_MODE=1
export WANDB_PROJECT="fourier-compression"
export WANDB_NAME="qwen-act-guided-compress-cr-5-averaging"
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

nvidia-smi

    # --config_file="accel_config/fsdp2.yaml" \
accelerate launch \
        train.py \
        --output_dir=$SCRATCH/runs/qwen-act-guided-compress-averaging-cr-5 \
        --eval_steps=500 \
        --gradient_accumulation_steps=4 \
        --add_gist \
        --compression_rate=5 \
        --attention_mask_mode="compression" \
        --compression_mode="average" \
        --gist_scheme="dispersed" \
        --act_guided_chunking=normdiff \
        --chunking_model=Qwen/Qwen3-0.6B \