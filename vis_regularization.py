import os
from typing import Tuple, List, Dict, Union, Any, Optional
from functools import partial
from argparse import ArgumentParser, Namespace
import shutil
import re
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
import seaborn as sns


import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer,
    BatchEncoding
)

from datasets import load_dataset
import evaluate

from accelerate import (
    Accelerator
)

from models.modeling_qwen3 import CompQwen3ForCausalLM
from utils import (
    find_tok_pos,
    apply_dispersed_gist,
    apply_gist,
    apply_end_gist,
    align_special_tokens,
)
from exp_args import parse_exp_args, join_args
from eval_squad import collate_fn
from models.compression_utils import find_context_start




def get_tokens_and_magnitude(input_ids: torch.LongTensor,
                            references: Dict[str, Any],
                            hidden_states: List[torch.FloatTensor],
                            tok: PreTrainedTokenizer):
    num_layers = len(hidden_states) - 1



    batch_token_norms = []
    batch_tokens = []
    for b in range(len(input_ids)):

        token_ids = input_ids[b]

        context_start = find_context_start(token_ids[None, :],
                            tok.pad_token_id,
                            pad_is_bos=tok.pad_token_id==tok.bos_token_id)[0]

        tokens = tok.convert_ids_to_tokens(token_ids[context_start:])
        tokens_stripped = [x.lstrip("Ġ") for x in tokens]

        layer_accum_token_norms = []

        for i in range(1, num_layers):
            seq = hidden_states[i][b]
            assert seq.shape[0] == token_ids.shape[0], \
                f"input_ids length does not match hidden states seq length for sample {b}. input_ids length = {token_ids.shape[0]}, hidden_states[{i}] length = {seq.shape[0]}"

            norms = seq[context_start:].norm(dim=-1)
            layer_accum_token_norms.append(norms)


        mean_token_norms = torch.stack(layer_accum_token_norms, dim=0).mean(dim=0)

        batch_tokens.append(tokens_stripped)
        batch_token_norms.append(mean_token_norms)



    return batch_tokens, batch_token_norms

def setup_model_and_env(model_name, args):
    model = CompQwen3ForCausalLM.from_pretrained(model_name)
    tok = AutoTokenizer.from_pretrained(model_name)
    model.enable_compression(tok)
    model.set_attention_mask_mode(args.attention_mask_mode)
    model.set_intermediate_transform(args.compression_mode, args.gist_scheme)

    return model, tok, args.gist_scheme





if __name__ == "__main__":

    # try:
    #     debug_mode = int(os.environ["DEBUG_MODE"])
    #     if debug_mode > 0:
    # print("starting debugger")
    # import debugpy
    # debugpy.listen(("0.0.0.0", 5678))
    # print("Waiting for debugger attach...")
    # debugpy.wait_for_client()
    # except:
    #     pass

    parser = ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="")
    # parser.add_argument("--setup", type=str, choices=["pos_control", "neg_control", "fourier", "gist", "average"])
    # parser.add_argument("--compression_rate", type=float, default=5.)
    parser.add_argument("--output_dir", type=str, default="runs/visualizations")
    # parser.add_argumemt("")
    vis_args, _ = parser.parse_known_args()
    experim_args = parse_exp_args()
    args = join_args(experim_args, vis_args)

    model_1, tok_1, gist_scheme = setup_model_and_env(model_name=args.model, args=args)

    chunking_model = None
    if args.act_guided_chunking is not None:
        chunking_model = AutoModelForCausalLM.from_pretrained(args.chunking_model)
        align_special_tokens(processing_class=tok_1, model=chunking_model)



    # define EOS, BOS, and PADDING tokens if not defined already
    align_special_tokens(processing_class=tok_1, model=model_1)

    # prepare output folder
    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir)

    valset = load_dataset("rajpurkar/squad_v2")["validation"]#.select(range(100))



    alpha = torch.linspace(0, 1, 200)
    accelerator = Accelerator()

    mask_rows = []
    for a in alpha:
        if chunking_model is not None:
            chunking_model = accelerator.prepare(chunking_model)

        collator_args = deepcopy(args)
        collator_args.chunking_model = chunking_model
        collator_args.alpha_unif = a

        dloader = DataLoader(valset,
                            batch_size=12,
                            shuffle=False,
                            collate_fn=partial(collate_fn,
                                                tokenizer=tok_1,
                                                add_thinking_tags=True,
                                                collator_args=collator_args
                                                ))


        tbar = tqdm(dloader, desc="Evaluating")
        batch, reference = next(iter(tbar))
        batch.pop("labels")

        sample = batch["input_ids"][0]

        # first, decode and strip
        decoded_sample = tok_1.decode(sample, skip_special_tokens=False)
        decoded_sample = decoded_sample.split(tok_1.pad_token)[-1]
        decoded_sample = decoded_sample.split("\n\nQue")[0]
        # print(decoded_sample)

        # then, re-encode
        retoked = tok_1.encode(decoded_sample)
        gist_mask = torch.tensor(retoked) == tok_1.convert_tokens_to_ids("<GIST>")
        # print(gist_mask)
        mask_rows.append(gist_mask)

    masked_rows_map = pad_sequence(mask_rows, batch_first=True, padding_value=0, padding_side="left").numpy()

    print(masked_rows_map)

    fig, ax = plt.subplots()
    ax.imshow(masked_rows_map, extent=[0, masked_rows_map.shape[1], alpha.max(), alpha.min()], aspect="auto")
    ax.set_ylabel("Alpha")
    ax.set_xlabel("Sequence Position")
    plt.savefig(os.path.join(args.output_dir, "mask.png"), dpi=300)
