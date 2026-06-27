import os
import shutil
from typing import Any, Tuple, List, Union, Dict, Optional
from argparse import Namespace, ArgumentParser
from functools import partial
from copy import deepcopy

import numpy as np
import torch
from torch import nn
from torch.optim import (
    AdamW,
    Adafactor
)

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer,
    GenerationConfig,
    TrainingArguments
)
from nltk.tokenize import word_tokenize
from transformers.trainer_utils import EvalPrediction
import evaluate

# from trainer import Trainer, TrainingArguments
from gist_trainer import GistTrainer
from accelerate import Accelerator

from utils import GistDataCollator
from data_utils.squad import Squad

from models.modeling_qwen3 import Qwen3ForCausalLM, CompQwen3ForCausalLM
from exp_args import parse_exp_args, join_args


def parse_training_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--eval_steps", default=500, type=int, help="number of training steps until eval")
    parser.add_argument("--lr", type=float, default=1e-5, help="learning rate")
    parser.add_argument("--gradient_accumulation_steps", default=2, type=int, help="gradient accumulation steps")
    parser.add_argument("--output_dir", type=str, default="runs/debug")

    args, unknown = parser.parse_known_args()
    return args


if __name__ == "__main__":

    # try:
    # debug_mode = int(os.environ["DEBUG_MODE"])
    # if debug_mode > 0:
    # print("starting debugger")
    # if Accelerator().is_main_process:
    #     import debugpy
    #     debugpy.listen(("0.0.0.0", 5678))
    #     print("Waiting for debugger attach...")
    #     debugpy.wait_for_client()
    # except:
    #     pass

    os.environ["WANDB_PROJECT"]="fourier-compression"


    general_args = parse_training_args()
    experiment_args = parse_exp_args()
    args = join_args(general_args, experiment_args)

    assert not (args.gist_scheme == "end" and args.compression_mode == "average"), f"Averaging-based compression only supports dispersed gist tokens"


    model = CompQwen3ForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

    tokenizer = model.enable_compression(tokenizer)
    model.set_attention_mask_mode(args.attention_mask_mode)
    model.set_intermediate_transform(mode=args.compression_mode,
                                     gist_scheme=args.gist_scheme)

    # for chunking model only
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    inference_device = torch.device(f"cuda:{local_rank}")

    # TODO: Deprecate this
    entropy_model = None
    if args.entropy_model is not None:
        if args.entropy_model == "self":
            entropy_model = model
        else:
            entropy_model = AutoModelForCausalLM.from_pretrained(args.entropy_model, torch_dtype=torch.bfloat16).to(inference_device)
            entropy_model.eval()

    chunking_model = None
    if args.chunking_model is not None:
        if args.chunking_model == "self":
            chunking_model = model
        else:
            chunking_model = AutoModelForCausalLM.from_pretrained(args.chunking_model, torch_dtype=torch.bfloat16).to(inference_device)
            chunking_model.eval()

    nltk_chunker = None
    if args.nltk_chunker is not None:
        from nltk.chunk.regexp import RegexpChunkParser, RegexpChunkRule
        rules = []
        if "np" in args.nltk_chunker:
            rules += [RegexpChunkRule.fromstring("{<DT>?<JJ.*>*<NN.*>+<IN>?<NN.*>+<.*>?}")]
        if "vp" in args.nltk_chunker:
            rules += [RegexpChunkRule.fromstring("{<MD>?<RB.*>*<V.*>+}")]
        nltk_chunker = RegexpChunkParser(rules=rules)


    collator_args = deepcopy(args)
    collator_args.chunking_model = chunking_model
    collator_args.entropy_model = entropy_model
    collator_args.nltk_chunker = nltk_chunker

    collator = GistDataCollator(tokenizer,
                                gist_token_id=tokenizer.convert_tokens_to_ids("<GIST>"),
                                collate_args=collator_args)


    task = Squad(
        list_splits=["train", "validation"],
        local_dir="squad-local",
        load_local=False
    )

    trainset = task.get_dataset("train")
    valset = task.get_dataset("validation")


    # optim = AdamW(model.parameters(), 1e-7)


    training_args = TrainingArguments(
        output_dir=args.output_dir,
        # overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        do_predict=False,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=10,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        bf16=True,
        bf16_full_eval=True,
        eval_steps=args.eval_steps,
        save_steps=args.eval_steps,
        eval_strategy="steps",
        eval_delay=0,
        save_strategy="best",
        num_train_epochs=3,
        remove_unused_columns=False,
        greater_is_better=False,
        metric_for_best_model="loss",
        report_to="wandb",
        logging_steps=10,
        train_sampling_strategy="random"
        # torch_empty_cache_steps=4,
        # eval_accumulation_steps=4,
        # predict_with_generate=True,
        # generation_config=GenerationConfig(
        #     num_beams=1,
        #     do_sample=False,
        #     use_cache=True,
        #     max_new_tokens=128
        # )
    )


    trainer = GistTrainer(
        model=model,
        data_collator=collator,
        train_dataset=trainset,
        eval_dataset=valset,
        processing_class=tokenizer,
        compute_metrics=None, #partial(compute_metrics, tokenizer=tokenizer),
        # optimizers=(optim, None),
        optimizer_cls_and_kwargs=(AdamW,{"params": model.parameters(), "lr": args.lr}),
        args=training_args
    )
    trainer.train()
