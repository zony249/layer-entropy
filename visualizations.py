import os 
from typing import Tuple, List, Dict, Union, Any, Optional 
from functools import partial
from argparse import ArgumentParser, Namespace
import shutil
import re
import matplotlib.pyplot as plt
import numpy as np


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
from models.compression_utils import find_context_start



def collate_fn(batch, 
               tokenizer: PreTrainedTokenizer, 
               add_thinking_tags: Optional[bool] = True, 
               add_gist: Optional[bool] = False, 
               compression_rate: Optional[float] = 5., 
               gist_scheme="end", 
               ) -> Tuple[BatchEncoding, List[Dict[str, Any]]]: 
    """
    this is a simplified version of GistDataCollator used for evaluations.
    It does not append the answer to the end of the context + question, since the 
    LM should generate that.
    """

    contexts = []
    contexts_attn_mask = []
    questions = []
    questions_attn_mask = []
    # answers = []
    # answers_attn_mask = []
    input_ids = []
    attention_mask = []
    label_ids = []

    references = []

    gist_token_id = tokenizer.convert_tokens_to_ids("<GIST>")
    if tokenizer.bos_token is None: 
        tokenizer.bos_token = tokenizer.pad_token

    for example in batch: 
        ctx = tokenizer.bos_token + "Context: " + example["context"]
        que = "\n\nQuestion: " + example["question"] 
        que += "\n\n<think>\n\n</think>" if add_thinking_tags else ""
        que += "\n\nAnswer: "
        ans = example["answers"]["text"][0]  + tokenizer.eos_token if len(example["answers"]["text"]) > 0  else "" 

        ctx_tok = tokenizer(ctx)["input_ids"]
        que_tok = tokenizer(que, return_tensors="pt")["input_ids"][0]
        ans_tok = tokenizer(ans, return_tensors="pt")["input_ids"][0]

        references.append({"answers": example["answers"], "id": example["id"]})
        
        ctx_tok = apply_gist(ctx_tok, 
                             gist_token_id=gist_token_id, 
                             compression_rate=compression_rate, 
                             gist_scheme=gist_scheme) if add_gist else torch.tensor(ctx_tok)

        inputs = torch.cat([ctx_tok, que_tok], dim=0)            
        input_ids.append(inputs) 
        attention_mask.append(torch.ones_like(inputs))

        # labels = torch.cat([torch.ones_like(ctx_tok) * -100, 
        #                     torch.ones_like(que_tok) * -100, 
        #                     ans_tok], dim=0) 
        labels = ans_tok
        label_ids.append(labels) 
        # assert len(inputs) == len(labels)
    
    
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id, padding_side="left")
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0, padding_side="left")

    label_ids = pad_sequence(label_ids, batch_first=True, padding_value=-100, padding_side="left")

    gist_positions = find_tok_pos(input_ids, gist_token_id)
    
    return BatchEncoding({
        "input_ids": input_ids, 
        "labels": label_ids, 
        "attention_mask": attention_mask, 
    }), references

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

def setup_model_and_env(model_name): 
    if args.setup in ["pos_control", "neg_control"]:
        model = CompQwen3ForCausalLM.from_pretrained(model_name)
        tok = AutoTokenizer.from_pretrained(model_name)

        model.enable_compression(tok)
        if args.setup == "pos_control":
            model.set_attention_mask_mode("full")
        elif args.setup == "neg_control": 
            model.set_attention_mask_mode("contextless")
        gist_scheme = "end"
    elif args.setup in ["fourier", "gist", "average"]: 
        model = CompQwen3ForCausalLM.from_pretrained(model_name)
        tok = AutoTokenizer.from_pretrained(model_name)
        model.enable_compression(tok)

        if args.setup == "fourier": 
            model.set_attention_mask_mode("compression")
            model.set_intermediate_transform("fourier")
            gist_scheme = "end"
        elif args.setup == "gist": 
            model.set_attention_mask_mode("compression")
            gist_scheme = "end"
        elif args.setup == "average": 
            model.set_attention_mask_mode("compression")
            model.set_intermediate_transform("average")
            gist_scheme = "dispersed"
    else:
        raise NotImplementedError

    return model, tok, gist_scheme

if __name__ == "__main__": 
    
    try: 
        debug_mode = int(os.environ["DEBUG_MODE"]) 
        if debug_mode > 0: 
            print("starting debugger")
            import debugpy
            debugpy.listen(("172.26.93.9", 5678))
            print("Waiting for debugger attach...")
            debugpy.wait_for_client()
    except: 
        pass

    parser = ArgumentParser() 
    parser.add_argument("--model_1", type=str, required=True, help="")
    parser.add_argument("--model_2", type=str, required=True, help="")
    parser.add_argument("--setup", type=str, choices=["pos_control", "neg_control", "fourier", "gist", "average"])
    parser.add_argument("--compression_rate", type=float, default=5.)
    parser.add_argument("--output_dir", type=str, default="runs/visualizations")
    # parser.add_argumemt("")
    args = parser.parse_args() 

    model_1, tok_1, gist_scheme = setup_model_and_env(model_name=args.model_1)
    model_2, tok_2, gist_scheme = setup_model_and_env(model_name=args.model_2)
    
    # define EOS, BOS, and PADDING tokens if not defined already
    align_special_tokens(processing_class=tok_1, model=model_1)
    align_special_tokens(processing_class=tok_2, model=model_2)

    # prepare output folder
    if os.path.exists(args.output_dir): 
        shutil.rmtree(args.output_dir) 
    os.makedirs(args.output_dir)   

    valset = load_dataset("rajpurkar/squad_v2")["validation"]#.select(range(100))
    dloader = DataLoader(valset, 
                         batch_size=12, 
                         shuffle=False, 
                         collate_fn=partial(collate_fn, 
                                            tokenizer=tok_1, 
                                            add_thinking_tags=True, 
                                            add_gist=True, 
                                            compression_rate=args.compression_rate, 
                                            gist_scheme=gist_scheme))

    accelerator = Accelerator()
    model_1, tok_1, dloader = accelerator.prepare(model_1, tok_1, dloader)
    model_1.eval()
    model_2, tok_2 = accelerator.prepare(model_2, tok_2)
    model_2.eval()

    gen_config = {
        "num_beams": 1, 
        "do_sample": True, 
        "top_k": 20, 
        "top_p": 0.95, 
        "max_new_tokens": 128, 
        "output_hidden_states": True, 
        "return_dict_in_generate": True
    }

    all_preds = []
    all_references = []

    tbar = tqdm(dloader, desc="Evaluating")


    sample_counter = 0
    for batch, reference in tbar: 
        batch.pop("labels")
        outputs_1 = model_1(**batch, output_hidden_states=True)

        tokens_1, token_norms_1 = get_tokens_and_magnitude(
            batch["input_ids"], 
            reference, 
            outputs_1["hidden_states"], 
            tok_1
        )

        outputs_2 = model_2(**batch, output_hidden_states=True)

        tokens_2, token_norms_2 = get_tokens_and_magnitude(
            batch["input_ids"], 
            reference, 
            outputs_2["hidden_states"], 
            tok_2
        )

        #tokens is [batch, seq_len]
        for b in range(len(tokens_1)): 
            csv_name = f"{sample_counter:05d}.csv"
            output_file_name = os.path.join(args.output_dir, csv_name)
            t1_list = list(tokens_1[b]) 
            n1_list = list(token_norms_1[b])
            t2_list = list(tokens_2[b]) 
            n2_list = list(token_norms_2[b])
            ref = reference[b]

            with open(output_file_name, "w") as f:
                answers = ref["answers"]["text"][0] if len(ref["answers"]["text"]) > 0 else "No Answer."
                answers_start = ref["answers"]["answer_start"][0] if len(ref["answers"]["text"]) > 0 else ""
                f.write(f"idx,tokens,model 1 norms,model 2 norms,answers:{answers},answers_start:{answers_start}\n")
                idx = 0
                for t, n1, n2 in zip(t1_list, n1_list, n2_list): 
                    t = "<comma>" if t == "," else t 
                    f.write(f"{idx},{t},{n1.item():.4f},{n2.item():.4f}\n")
                    idx += 1
            # fig, ax = plt.subplots()

            # x = np.arange(len(tokens_1))


            sample_counter+=1

        break


