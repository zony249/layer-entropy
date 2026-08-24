#!/bin/bash
#SBATCH --account=aip-lilimou
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=l40s:4
#SBATCH --mem=64G
#SBATCH --time=0-03:00
#SBATCH --job-name=qwen-cr-x-unif-hard
#SBATCH --array=0-3
#SBATCH --output=logs/%j--%x-a-%a.log



nvidia-smi 

# export SLURM_ARRAY_TASK_ID=1
export CRS=("10" "15" "20" "30") 
export CR=${CRS[$SLURM_ARRAY_TASK_ID]}
export CR=5

export HF_HOME=$SCRATCH
export WANDB_JOB_TYPE="compressor"
export WANDB_PROJECT="learned-chunks"
export WANDB_NAME="qwen-cr-$CR-unif-hard-lr-3e-5"

    # --config_file="accel_config/fsdp2.yaml" \
accelerate launch \
    --config_file="accel_config/fsdp2.yaml" \
    train_compressor.py \
        --model=Qwen/Qwen3-0.6B \
        --output_dir=$SCRATCH/runs/qwen-cr-$CR-unif-hard \
        --lr=3e-5 \
        --epochs=3 \
        --per_device_batch_size=4 \
        --gradient_accumulation_steps=4 \
        --eval_steps=500 \
        --compression_rate=$CR \
        --mask_mode="hard" \
        --add_sink \
        