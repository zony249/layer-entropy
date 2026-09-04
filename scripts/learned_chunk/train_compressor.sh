#!/bin/bash
#SBATCH --account=aip-lilimou
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=l40s:4
#SBATCH --mem=64G
#SBATCH --time=0-04:00
#SBATCH --job-name=qwen-cr-compression_experiments
#SBATCH --output=logs/%j--%x-a-%a.log



nvidia-smi 

# export SLURM_ARRAY_TASK_ID=1
export CRS=("5" "10" "15" "20" "30") 
export CR=${CRS[$SLURM_ARRAY_TASK_ID]}
export CLR=3e-5

export HF_HOME=$SCRATCH
export WANDB_PROJECT="learned-chunks"
export WANDB_JOB_TYPE="compressor"
export WANDB_NAME="qwen-cr-$CR-learned-chunk-clr-$CLR-alpha-1-temp-0.2"

    # --config_file="accel_config/fsdp2.yaml" \
accelerate launch \
    --config_file="accel_config/fsdp2.yaml" \
    train_compressor.py \
        --chunking_model=/home/zonglin1/scratch/runs/qwen-gumbel-chunker-cr-$CR-layers-8-reg-1e-4-v3/checkpoint-3057 \
        --chunker_layers=8 \
        --model=Qwen/Qwen3-0.6B \
        --output_dir=$SCRATCH/runs/qwen-cr-$CR-learned-gumbel-chunker-v2-$CLR-alpha-1-temp-0.2 \
        --lr=3e-5 \
        --chunk_lr=$CLR \
        --epochs=3 \
        --per_device_batch_size=4 \
        --gradient_accumulation_steps=4 \
        --eval_steps=200 \
        --compression_rate=$CR \
        --mask_mode="soft" \
        --alpha_unif=1 \
        