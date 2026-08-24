#!/bin/bash
#SBATCH --account=aip-lilimou
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=l40s:4
#SBATCH --mem=64G
#SBATCH --time=0-03:00
#SBATCH --job-name=qwen-pos-control
#SBATCH --output=logs/%j--%x-a-%a.log


export LR=("3e-5" "3e-5" "3e-5")
export C_LR=("1e-5" "3e-5" "5e-5")

nvidia-smi 

# export SLURM_ARRAY_TASK_ID=1
# export UNIF_ALPHA=("0" "1" "3" "5" "10" "15" "30")

export HF_HOME=$SCRATCH
export WANDB_PROJECT="learned-chunks"
export WANDB_JOB_TYPE="compressor"
export WANDB_NAME="qwen-pos-control"

    # --config_file="accel_config/fsdp2.yaml" \
accelerate launch \
    --config_file="accel_config/fsdp2.yaml" \
    train_compressor.py \
        --model=Qwen/Qwen3-0.6B \
        --output_dir=$SCRATCH/runs/qwen-pos-control \
        --lr=1e-5 \
        --epochs=3 \
        --per_device_batch_size=4 \
        --gradient_accumulation_steps=4 \
        --eval_steps=200 \
        --compression_rate=5 \
        --mask_mode="full" \
        --alpha_unif=1 \
        