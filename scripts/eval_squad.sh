#!/bin/bash
#SBATCH --account=aip-lilimou
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=l40s:1
#SBATCH --mem=32G
#SBATCH --time=0-05:00
#SBATCH --job-name=min_chunk_diff_a-0_cr-5
#SBATCH --output=logs/%j--%x.log

# export DEBUG_MODE=1
# export CUDA_VISIBLE_DEVICES=4
export HF_HOME=$SCRATCH

accelerate launch \
    --num_processes 1 \
    --num_machines 1 \
    --mixed_precision=bf16 \
        eval_squad.py \
        --hf_model_name_or_path /scratch/zonglin1/runs/qwen-min_chunk_diff-a-0-cr-5/best_tfmr \
        --add_gist \
        --attention_mask_mode=compression \
        --compression_mode=none \
        --compression_rate 5 \
        --gist_scheme=dispersed \
        --gist_granularity=1 \
        --act_guided_chunking=min_chunk_diff \
        --chunking_model=Qwen/Qwen3-0.6B \
        --use_layers 1 \
        --alpha_unif 0 \
        # --entropy_model=Qwen/Qwen3-0.6B \
        # --surprise_mode=ce \
        # --entropy_model_temp=1 \
