#!/bin/bash

# export DEBUG_MODE=1

accelerate launch \
    --num_processes 1 \
    --num_machines 1 \
    --mixed_precision=bf16 \
        eval_squad.py \
