#!/bin/bash
#SBATCH --account=aip-lilimou
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=l40s:4
#SBATCH --mem=40G
#SBATCH --time=0-03:00
#SBATCH --job-name=qwen-chunker-cr-5-30
#SBATCH --array=0-4
#SBATCH --output=logs/%j--%x-a-%a.log



nvidia-smi 

# export SLURM_ARRAY_TASK_ID=0
export CRS=("5" "10" "15" "20" "30")
export CR=${CRS[$SLURM_ARRAY_TASK_ID]}
export LAYERS=8
export REG=1e-4

export HF_HOME=$SCRATCH
export WANDB_PROJECT="learned-chunks"
export WANDB_JOB_TYPE="chunker"
export WANDB_NAME="gumbel-chunker-cr-$CR-layers-$LAYERS-reg-$REG-temp"

    # --config_file="accel_config/fsdp2.yaml" \
accelerate launch \
    --num_processes=4 \
    --num_machines=1 \
    --mixed_precision bf16 \
    train_chunker.py \
        --chunking_model=Qwen/Qwen3-0.6B \
        --output_dir=$SCRATCH/runs/qwen-gumbel-chunker-cr-$CR-layers-$LAYERS-reg-$REG-v3 \
        --lr=1e-4 \
        --l2=$REG \
        --epochs=3 \
        --per_device_batch_size=4 \
        --gradient_accumulation_steps=8 \
        --eval_steps=200 \
        --compression_rate=$CR \
        --num_layers=$LAYERS \
        