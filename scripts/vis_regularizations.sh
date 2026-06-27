#!/bin/bash
#SBATCH --account=aip-lilimou
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=l40s:1
#SBATCH --mem=64G
#SBATCH --time=0-05:00
#SBATCH --job-name=qwen-0.6b-eval
#SBATCH --output=logs/%j--qwen-0.6b-eval-average-cr-1.log

# export DEBUG_MODE=1
# export CUDA_VISIBLE_DEVICES=4
export HF_HOME=$SCRATCH

accelerate launch \
    --num_processes 1 \
    --num_machines 1 \
    --mixed_precision=bf16 \
        vis_regularization.py \
        --model /home/zonglin1/scratch/runs/qwen-uniform-cr-7.7/best_tfmr \
        --add_gist \
        --compression_rate=5 \
        --attention_mask_mode=compression \
        --compression_mode=average \
        --gist_scheme=dispersed \
        --gist_granularity=1 \
        --act_guided_chunking=reg_cosine \
        --chunking_model=Qwen/Qwen3-0.6B \
