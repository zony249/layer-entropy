#!/bin/bash
#SBATCH --account=aip-lilimou
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=l40s:1
#SBATCH --mem=32G
#SBATCH --time=0-05:00
#SBATCH --job-name=reg_cosine_cr-5
#SBATCH --output=logs/%j--%x--a-%a.log
#SBATCH --array=0-10

# export DEBUG_MODE=1
# export CUDA_VISIBLE_DEVICES=4
export HF_HOME=$SCRATCH

# export ALPHA=$( awk "BEGIN {print $SLURM_ARRAY_TASK_ID / 10}" )
# echo $ALPHA

accelerate launch \
    --num_processes 1 \
    --num_machines 1 \
    --mixed_precision=bf16 \
        eval_squad.py \
        --hf_model_name_or_path /home/zonglin1/scratch/runs/qwen-neg/best_tfmr \
        --add_gist \
        --attention_mask_mode=contextless \
        --compression_mode=none \
        --compression_rate 5 \
        --gist_scheme=dispersed \
        --gist_granularity=1 \
