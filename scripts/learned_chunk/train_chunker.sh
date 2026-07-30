#!/bin/bash
#SBATCH --account=aip-lilimou
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=l40s:4
#SBATCH --mem=50G
#SBATCH --time=0-00:40
#SBATCH --job-name=qwen-chunker-cr-5
#SBATCH --output=logs/%j--%x-a-%a.log



nvidia-smi 

export SLURM_ARRAY_TASK_ID=8

export HF_HOME=$SCRATCH
export WANDB_PROJECT="learned-chunks"
export WANDB_NAME="qwen-chunker-layers-$SLURM_ARRAY_TASK_ID"-reg-1e-4

    # --config_file="accel_config/fsdp2.yaml" \
accelerate launch \
    --num_processes=4 \
    --num_machines=1 \
    --mixed_precision bf16 \
    train_chunker.py \
        --chunking_model=Qwen/Qwen3-0.6B \
        --output_dir=$SCRATCH/runs/qwen-chunker-layers-$SLURM_ARRAY_TASK_ID-reg-1e-4 \
        --lr=1e-4 \
        --epochs=5 \
        --per_device_batch_size=4 \
        --gradient_accumulation_steps=8 \
        --eval_steps=500 \
        --compression_rate=5 \
        --num_layers=$SLURM_ARRAY_TASK_ID \
        