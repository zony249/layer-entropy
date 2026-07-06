#!/bin/bash
#SBATCH --account=aip-lilimou
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=l40s:4
#SBATCH --mem=80G
#SBATCH --time=1-12:00
#SBATCH --job-name=qwen-0.6b-reg_cosine-cr-5
#SBATCH --array=0-10
#SBATCH --output=logs/%j--%x-a-%a.log

# export CUDA_VISIBLE_DEVICES=4,5
export HF_HOME=$SCRATCH
# export DEBUG_MODE=1
export WANDB_PROJECT="fourier-compression"
export WANDB_NAME="qwen-reg_cosine-a-0.$SLURM_ARRAY_TASK_ID-cr-5"
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export ALPHA=$( awk "BEGIN {print $SLURM_ARRAY_TASK_ID / 10}" )
echo $ALPHA

nvidia-smi

    # --config_file="accel_config/fsdp2.yaml" \
accelerate launch \
    --config_file="accel_config/fsdp2.yaml" \
    train.py \
        --output_dir=$SCRATCH/runs/qwen-reg_cos-a-0.$SLURM_ARRAY_TASK_ID-cr-5 \
        --eval_steps=500 \
        --gradient_accumulation_steps=4 \
        --lr=1e-5 \
        --add_gist \
        --compression_rate=5 \
        --attention_mask_mode="compression" \
        --compression_mode="none" \
        --gist_scheme="dispersed" \
        --act_guided_chunking="reg_cosine" \
        --chunking_model=Qwen/Qwen3-0.6B \
        --use_layers 1 \
        --alpha_unif $ALPHA \
