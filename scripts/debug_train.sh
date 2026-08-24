#!/bin/bash
#SBATCH --account=aip-lilimou
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=l40s:4
#SBATCH --mem=64G
#SBATCH --time=0-03:00
#SBATCH --job-name=qwen-cr-5-gridsearch
#SBATCH --array=0-8
#SBATCH --output=logs/%j--%x-a-%a.log


export LR=("1e-5" "1e-5" "1e-5" "3e-5" "3e-5" "3e-5" "5e-5" "5e-5" "5e-5")
export C_LR=("1e-5" "3e-5" "5e-5" "1e-5" "3e-5" "5e-5" "1e-5" "3e-5" "5e-5")

nvidia-smi 

export SLURM_ARRAY_TASK_ID=1
export UNIF_ALPHA=("0" "1" "3" "5" "10" "15" "30")

export HF_HOME=$SCRATCH
export WANDB_PROJECT="learned-chunks"
export WANDB_NAME="dev"

    # --config_file="accel_config/fsdp2.yaml" \
accelerate launch \
    --config_file="accel_config/fsdp2.yaml" \
    train_compressor.py \
        --chunking_model=/home/zonglin1/scratch/runs/qwen-gumbel-chunker-layers-8-reg-1e-3/checkpoint-4000 \
        --model=Qwen/Qwen3-0.6B \
        --output_dir=$SCRATCH/runs/dev \
        --lr=1e-5 \
        --chunk_lr=1e-5 \
        --epochs=3 \
        --per_device_batch_size=4 \
        --gradient_accumulation_steps=4 \
        --eval_steps=200 \
        --compression_rate=5 \
        --mask_mode="soft" \
        --alpha_unif=1 \
        