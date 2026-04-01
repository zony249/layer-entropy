import os 
from typing import Tuple, List, Dict, Union, Any, Optional 
from functools import partial
from argparse import ArgumentParser, Namespace
import shutil


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
    apply_end_gist
)



def collate_fn(batch, 
               tokenizer: PreTrainedTokenizer, 
               add_thinking_tags: Optional[bool] = True, 
               add_gist: Optional[bool] = False, 
               compression_rate: Optional[float] = 5., 
               gist_scheme="end", 
               gist_granularity: Optional[int] = 1
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
                             gist_scheme=gist_scheme, 
                             gist_granularity=gist_granularity) if add_gist else torch.tensor(ctx_tok)

        # contexts.append(ctx_tok)
        # contexts_attn_mask.append(torch.ones_like(ctx_tok))

        # questions.append(que_tok)
        # questions_attn_mask.append(torch.ones_like(que_tok))

        # answers.append(ans_tok)
        # answers_attn_mask.append(torch.ones_like(ans_tok))


        inputs = torch.cat([ctx_tok, que_tok], dim=0)            
        input_ids.append(inputs) 
        attention_mask.append(torch.ones_like(inputs))

        # labels = torch.cat([torch.ones_like(ctx_tok) * -100, 
        #                     torch.ones_like(que_tok) * -100, 
        #                     ans_tok], dim=0) 
        labels = ans_tok
        label_ids.append(labels) 
        # assert len(inputs) == len(labels)
    
    #TODO: Collate them
    # contexts = pad_sequence(contexts, batch_first=True, padding_value=tokenizer.pad_token_id, padding_side="left")
    # contexts_attn_mask = pad_sequence(contexts_attn_mask, batch_first=True, padding_value=0, padding_side="left")

    # questions = pad_sequence(questions, batch_first=True, padding_value=tokenizer.pad_token_id, padding_side="left")
    # questions_attn_mask = pad_sequence(questions_attn_mask, batch_first=True, padding_value=0, padding_side="left")

    # answers = pad_sequence(answers, batch_first=True, padding_value=tokenizer.pad_token_id, padding_side="left")
    # answers_attn_mask = pad_sequence(answers_attn_mask, batch_first=True, padding_value=0, padding_side="left")
    
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id, padding_side="left")
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0, padding_side="left")

    label_ids = pad_sequence(label_ids, batch_first=True, padding_value=-100, padding_side="left")

    gist_positions = find_tok_pos(input_ids, gist_token_id)
    
    return BatchEncoding({
        "input_ids": input_ids, 
        "labels": label_ids, 
        "attention_mask": attention_mask, 
    }), references


if __name__ == "__main__": 
    
    try: 
        debug_mode = int(os.environ["DEBUG_MODE"]) 
        if debug_mode > 0: 
            print("starting debugger")
            import debugpy
            debugpy.listen(("172.26.93.138", 5678))
            print("Waiting for debugger attach...")
            debugpy.wait_for_client()
    except: 
        pass

    parser = ArgumentParser() 
    parser.add_argument("--hf_model_name_or_path", type=str, required=True, help="")
    parser.add_argument("--setup", type=str, choices=["pos_control", "neg_control", "fourier", "gist", "average"])
    parser.add_argument("--compression_rate", type=float, default=5.)
    parser.add_argument("--gist_scheme", type=str, default="end", choices=["end", "dispersed"])
    parser.add_argument("--gist_granularity", type=int, default=1)
    # parser.add_argumemt("")
    args = parser.parse_args() 

    if args.setup in ["pos_control", "neg_control"]:
        model = CompQwen3ForCausalLM.from_pretrained(args.hf_model_name_or_path)
        tok = AutoTokenizer.from_pretrained(args.hf_model_name_or_path)

        model.enable_compression(tok)
        if args.setup == "pos_control":
            model.set_attention_mask_mode("full")
        elif args.setup == "neg_control": 
            model.set_attention_mask_mode("contextless")
        gist_scheme = args.gist_scheme
    elif args.setup in ["fourier", "gist", "average"]: 
        model = CompQwen3ForCausalLM.from_pretrained(args.hf_model_name_or_path)
        tok = AutoTokenizer.from_pretrained(args.hf_model_name_or_path)
        model.enable_compression(tok)

        if args.setup == "fourier": 
            model.set_attention_mask_mode("compression")
            model.set_intermediate_transform("fourier", gist_scheme=args.gist_scheme)
        elif args.setup == "gist": 
            model.set_attention_mask_mode("compression")
        elif args.setup == "average": 
            model.set_attention_mask_mode("compression")
            model.set_intermediate_transform("average", gist_scheme=args.gist_scheme)
        gist_scheme = args.gist_scheme
    else:
        raise NotImplementedError
    


    valset = load_dataset("rajpurkar/squad_v2")["validation"]#.select(range(100))
    dloader = DataLoader(valset, 
                         batch_size=12, 
                         shuffle=False, 
                         collate_fn=partial(collate_fn, 
                                            tokenizer=tok, 
                                            add_thinking_tags=True, 
                                            add_gist=True, 
                                            compression_rate=args.compression_rate, 
                                            gist_scheme=gist_scheme))

    accelerator = Accelerator()
    model, tok, dloader = accelerator.prepare(model, tok, dloader)
    model.eval()

    gen_config = {
        "num_beams": 1, 
        "do_sample": True, 
        "top_k": 20, 
        "top_p": 0.95, 
        "max_new_tokens": 128
    }

    all_preds = []
    all_references = []

    tbar = tqdm(dloader, desc="Evaluating")

    for batch, reference in tbar: 
        batch.pop("labels")
        preds = model.generate(**batch, **gen_config)     
        preds[:, :batch["input_ids"].shape[1]] = -100

        preds_decoded = tok.batch_decode(torch.where(preds == -100, tok.pad_token_id, preds), skip_special_tokens=True)
        preds_decoded = [p.strip() for p in preds_decoded]

        preds_dict = [{"prediction_text": p, "id": r["id"], "no_answer_probability": 1. if p == "Not enough information." else 0.} for p, r in zip(preds_decoded, reference)]
        all_preds += preds_dict 
        all_references += reference 

    all_preds = accelerator.gather_for_metrics(all_preds, True)
    all_references = accelerator.gather_for_metrics(all_references, True)

    if os.path.exists("runs/eval"): 
        shutil.rmtree("runs/eval")
    os.makedirs("runs/eval", exist_ok=True)
    with open("runs/eval/preds.txt", "w") as f: 
        for p in all_preds: 
            f.write(p["prediction_text"] + "\n")
    with open("runs/eval/ref.txt", "w") as f: 
        for r in all_references: 
            write_out = r["answers"]["text"][0] + "\n" if len(r["answers"]["text"]) > 0 else "Not enough info.\n"
            f.write(write_out)

    squad_v2_metric = evaluate.load("squad_v2")
    results = squad_v2_metric.compute(predictions=all_preds, references=all_references)
    print(results)