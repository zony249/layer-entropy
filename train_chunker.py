import os 
import sys 
from copy import deepcopy 
from typing import Dict, List, Tuple, Any, Optional 
import shutil 
from argparse import ArgumentParser, Namespace

import torch 
from torch import nn 
from torch.nn import functional as F 
from torch.optim import AdamW

from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    AutoTokenizer, 
    TrainingArguments
)
from accelerate import Accelerator
import wandb

from dynam_compress_trainer import ChunkerTrainer, ChunkerTrainingArguments
from models.modeling_qwen3 import (
    Qwen3ForCausalLM, 
    Qwen3Chunker, 
)
from data_utils.squad import Squad 
from exp_args import parse_chunk_exp_args, join_args
from utils import ChunkCollator


if Accelerator().is_main_process:
    wandb.init()



if __name__ == "__main__": 


    # if Accelerator().is_main_process:
    #     import debugpy
    #     debugpy.listen(("0.0.0.0", 5678))
    #     print("Waiting for debugger attach...")
    #     debugpy.wait_for_client()

    parser = ArgumentParser() 
    parser.add_argument("--output_dir", type=str, default="runs/experiment")
    parser.add_argument("--lr", type=float, default=1e-5) 
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--per_device_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--num_layers", type=int, default=5)
    parser.add_argument("--l2", type=float, default=0)

    training_specific_args, unknown = parser.parse_known_args() 
    general_args, unknown = parse_chunk_exp_args(unknown)
    args = join_args(training_specific_args, general_args)

    chunking_model = Qwen3Chunker.from_pretrained(
        args.chunking_model,
        num_layers=args.num_layers,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.chunking_model)

    if Accelerator().is_main_process: 
        print(chunking_model)

    squad_task = Squad(
        list_splits=["train", "validation"],
        batch_size=args.per_device_batch_size, 
        preprocess_validation=True
    ) 
    dsets = squad_task.get_datasets(list_splits=["train", "validation"])
    trainset = dsets["train"]
    valset = dsets["validation"]

    collator = ChunkCollator(
        tokenizer=tokenizer,
        validation_mode=False,
        collator_args=args
    )

    training_args = ChunkerTrainingArguments(
        output_dir=args.output_dir, 
        per_device_train_batch_size=args.per_device_batch_size, 
        num_train_epochs=args.epochs, 
        learning_rate=args.lr, 
        gradient_accumulation_steps=args.gradient_accumulation_steps, 
        bf16=True, 
        bf16_full_eval=True, 
        logging_strategy="steps",
        logging_steps=10, 
        report_to="wandb", 
        eval_strategy="steps", 
        eval_steps=args.eval_steps, 
        # eval_delay=500, 
        save_strategy="steps", 
        save_steps=args.eval_steps, 
        remove_unused_columns=False, 
        t_high=0.8, 
        t_low=0.2
    )

    trainer = ChunkerTrainer(
        model=chunking_model,
        args=training_args,
        data_collator=collator,
        train_dataset=trainset,
        eval_dataset=valset, 
        processing_class=tokenizer, 
        optimizer_cls_and_kwargs=(AdamW, {"params": chunking_model.parameters(), "lr": args.lr, "weight_decay": args.l2})
    )

    trainer.train()

    pass