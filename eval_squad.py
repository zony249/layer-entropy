import os 
from typing import Tuple, List, Dict, Union, Any, Optional 
from functools import partial

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

def apply_gist(context: List[int], 
               gist_token_id: int,  
               compression_rate: float, 
               gist_scheme: Optional[str] = "end") -> torch.Tensor: 
    """
    context: [seq_len]
        post-tokenization sequence tensor
    """
    if gist_scheme == "dispersed": 
        return _apply_dispersed_gist(context, 
                                     gist_token_id, 
                                     compression_rate)
    elif gist_scheme == "end": 
        return _apply_end_gist(context, 
                               gist_token_id, 
                               compression_rate)
    else: 
        raise NotImplementedError() 

def _apply_dispersed_gist(context: List[int], 
                          gist_token_id: int, 
                          compression_rate: float) -> torch.Tensor: 
    if len(context) == 0:
        return torch.tensor([gist_token_id])
    context.reverse()
    i = 0
    while i < len(context): 
        context.insert(i, gist_token_id)
        i += int(compression_rate+1)
    context.reverse() 
    return torch.tensor(context)

def _apply_end_gist(context: List[int],
                    gist_token_id: int, 
                    compression_rate: float) -> torch.Tensor: 
    num_gist_tokens = len(context) // compression_rate
    context = context + [gist_token_id for _ in range(num_gist_tokens)] 
    return torch.tensor(context)

def find_tok_pos(batched_tensor:torch.LongTensor, token_id: int) -> List[torch.Tensor]: 
    positions = [] 
    for i in range(len(batched_tensor)): 
        ids = (batched_tensor[i] == token_id).nonzero(as_tuple=False).flatten()
        positions.append(ids)
    return positions


def collate_fn(batch, 
               tokenizer: PreTrainedTokenizer, 
               add_thinking_tags: Optional[bool] = True, 
               add_gist: Optional[bool] = False, 
               compression_rate: Optional[float] = 5., 
               ): 

    contexts = []
    contexts_attn_mask = []
    questions = []
    questions_attn_mask = []
    answers = []
    answers_attn_mask = []
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
                             compression_rate=compression_rate) if add_gist else torch.tensor(ctx_tok)

        contexts.append(ctx_tok)
        contexts_attn_mask.append(torch.ones_like(ctx_tok))

        questions.append(que_tok)
        questions_attn_mask.append(torch.ones_like(que_tok))

        answers.append(ans_tok)
        answers_attn_mask.append(torch.ones_like(ans_tok))


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
    contexts = pad_sequence(contexts, batch_first=True, padding_value=tokenizer.pad_token_id, padding_side="left")
    contexts_attn_mask = pad_sequence(contexts_attn_mask, batch_first=True, padding_value=0, padding_side="left")

    questions = pad_sequence(questions, batch_first=True, padding_value=tokenizer.pad_token_id, padding_side="left")
    questions_attn_mask = pad_sequence(questions_attn_mask, batch_first=True, padding_value=0, padding_side="left")

    answers = pad_sequence(answers, batch_first=True, padding_value=tokenizer.pad_token_id, padding_side="left")
    answers_attn_mask = pad_sequence(answers_attn_mask, batch_first=True, padding_value=0, padding_side="left")
    
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
            debugpy.listen(("172.26.93.228", 5678))
            print("Waiting for debugger attach...")
            debugpy.wait_for_client()
    except: 
        pass

    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

    # if your tokenizer already supports Gist tokens, this should do nothing
    special_tokens = {"additional_special_tokens": ["<GIST>"]} 
    tok.add_special_tokens(special_tokens) 
    model.resize_token_embeddings(len(tok))


    valset = load_dataset("rajpurkar/squad_v2")["validation"].select(range(100))
    dloader = DataLoader(valset, 
                         batch_size=2, 
                         shuffle=False, 
                         collate_fn=partial(collate_fn, 
                                            tokenizer=tok, 
                                            add_thinking_tags=True, 
                                            add_gist=False, 
                                            compression_rate=5.))

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

    squad_v2_metric = evaluate.load("squad_v2")
    results = squad_v2_metric.compute(predictions=all_preds, references=all_references)
    print(results)