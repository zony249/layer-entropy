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
        visualizations.py \
        --model_1 "Qwen/Qwen3-0.6B" \
        --model_2 "runs/qwen-fourier-compress-cr-10/best_tfmr" \
        --setup="fourier" \
        --compression_rate=10 \
