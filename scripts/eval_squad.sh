#!/bin/bash

# export DEBUG_MODE=1
# export CUDA_VISIBLE_DEVICES=4
export HF_HOME=$SCRATCH

accelerate launch \
    --num_processes 1 \
    --num_machines 1 \
    --mixed_precision=bf16 \
        eval_squad.py \
        --hf_model_name_or_path "runs/qwen-pos-control/best_tfmr" \
        --setup="pos_control" \
