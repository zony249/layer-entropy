#!/bin/bash
#SBATCH --account=aip-lilimou
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=l40s:1
#SBATCH --mem=32G
#SBATCH --time=0-02:00
#SBATCH --job-name=eval_squad
#SBATCH --output=logs/%j--%x--a-%a.log
#SBATCH --array=0-3

# export DEBUG_MODE=1
# export CUDA_VISIBLE_DEVICES=4
# export SLURM_ARRAY_TASK_ID=0
export HF_HOME=$SCRATCH
# export CRS=("10" "15" "20" "30") 
# export CR=${CRS[$SLURM_ARRAY_TASK_ID]}
export CR=30

# export ALPHA=$( awk "BEGIN {print $SLURM_ARRAY_TASK_ID / 10}" )
# echo $ALPHA

accelerate launch \
    --mixed_precision=bf16 \
        eval_squad_2.py \
        --model /home/zonglin1/scratch/runs/qwen-cr-$CR-learned-gumbel-chunker/checkpoint-4000 \
        --chunking_model /home/zonglin1/scratch/runs/qwen-cr-$CR-learned-gumbel-chunker/checkpoint-4000/chunker \
        --compression_rate $CR \
        --mask_mode="soft"
