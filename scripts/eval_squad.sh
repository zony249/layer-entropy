#!/bin/bash
#SBATCH --account=aip-lilimou 
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=l40s:1
#SBATCH --mem=32G
#SBATCH --time=0-05:00
#SBATCH --job-name=average-compress
#SBATCH --output=logs/%j--average-compress-cr-5.log

# export DEBUG_MODE=1
# export CUDA_VISIBLE_DEVICES=4
export HF_HOME=$SCRATCH

accelerate launch \
    --num_processes 1 \
    --num_machines 1 \
    --mixed_precision=bf16 \
        eval_squad.py \
        --hf_model_name_or_path $SCRATCH/runs/qwen-average-compress-cr-5/best_tfmr \
        --setup="average" \
        --compression_rate=5 \
        --gist_scheme="dispersed" \
        --gist_granularity=1 \
        # --entropy_model=Qwen/Qwen3-0.6B \
        # --surprise_mode=entropy
