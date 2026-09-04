import os
from typing import Tuple, List, Dict, Union, Any, Optional
from functools import partial
from argparse import ArgumentParser, Namespace
import shutil
from copy import deepcopy


import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import matplotlib.pyplot as plt

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer,
    BatchEncoding,
    PreTrainedModel,
)

from nltk.chunk.api import ChunkParserI

from datasets import load_dataset
import evaluate

from accelerate import (
    Accelerator
)

from models.modeling_qwen3 import ZipQwen3ForCausalLM, Qwen3ForCausalLM, Qwen3Chunker
from utils import (
    find_tok_pos,
    apply_dispersed_gist,
    add_gist_str_using_linguistic_chunker,
    apply_gist,
    apply_end_gist,
    compute_surprise,
    align_special_tokens,
    compute_norm_of_diffs
)
from exp_args import parse_chunk_exp_args, join_args
from utils import DefaultArgs, ChunkCollator, compute_soft_compression_rate


def collate_fn(batch,
               tokenizer: PreTrainedTokenizer,
               add_thinking_tags: Optional[bool] = True,
               # add_gist: Optional[bool] = False,
               # compression_rate: Optional[float] = 5.,
               # gist_scheme="end",
               # gist_granularity: Optional[int] = 1,
               # entropy_model: Optional[PreTrainedModel] = None, # Deprecated
               # surprise_mode: Optional[str] = "entropy", # Deprecated
               # act_guided_chunking: Optional[str] = None,
               # chunking_model: Optional[PreTrainedModel] = None,
               # nltk_chunker: Optional[ChunkParserI] = None,
               collator_args: Namespace | None = None
               ) -> Tuple[BatchEncoding, List[Dict[str, Any]]]:
    """
    this is a simplified version of GistDataCollator used for evaluations.
    It does not append the answer to the end of the context + question, since the
    LM should generate that.
    """
    args = collator_args if collator_args is not None else DefaultArgs()

    add_gist = args.add_gist
    compression_rate = args.compression_rate
    gist_scheme = args.gist_scheme
    gist_granularity = args.gist_granularity
    entropy_model = args.entropy_model
    temp = args.entropy_model_temp
    surprise_mode = args.surprise_mode
    act_guided_chunking = None if args.act_guided_chunking == "none" else args.act_guided_chunking
    chunking_model = args.chunking_model
    nltk_chunker = args.nltk_chunker
    layer_selection = [1] if args.use_layers is None else args.use_layers
    alpha_unif = args.alpha_unif

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if tokenizer.bos_token is None:
        tokenizer.bos_token = tokenizer.pad_token

    act_guided_chunking = None if act_guided_chunking == "none" else act_guided_chunking

    gist_token_id = tokenizer.convert_tokens_to_ids("<GIST>")

    contexts = [example["context"] for example in batch]
    tok_outputs = tokenizer(contexts, return_tensors="pt", padding=True, padding_side="left")
    tok_outputs_unpadded = tokenizer(contexts)

    if False:
        if entropy_model is not None:
            tok_outputs = tok_outputs.to(entropy_model.device)
            inputs = {"input_ids": tok_outputs["input_ids"],
                        "labels": tok_outputs["input_ids"],
                        "attention_mask":  tok_outputs["attention_mask"]}
            surprises = compute_surprise(inputs, entropy_model, tokenizer,
                                            surprise_mode=surprise_mode).cpu()

    act_signals = None
    if act_guided_chunking is not None:
        tok_outputs = tok_outputs.to(chunking_model.device)
        inputs = {"input_ids": tok_outputs["input_ids"],
                    "labels": tok_outputs["input_ids"],
                    "attention_mask":  tok_outputs["attention_mask"]}
        act_signals = compute_norm_of_diffs(inputs, chunking_model, None, act_guided_chunking)

    if act_guided_chunking is not None:
        if act_guided_chunking == "min_chunk_diff":
            assert chunking_model is not None, "Chunking model must be defined"
            tok_outputs = tok_outputs.to(chunking_model.device)
            inputs = {"input_ids": tok_outputs["input_ids"],
                        "labels": tok_outputs["input_ids"],
                        "attention_mask":  tok_outputs["attention_mask"]}
            with torch.no_grad():
                outputs = chunking_model(**inputs, output_hidden_states=True)
                act_signals = torch.stack([outputs.hidden_states[i] for i in layer_selection], dim=0).mean(dim=0)
                del outputs
                # act_signals = torch.stack(outputs.hidden_states, dim=0).mean(dim=0)

    if act_guided_chunking is not None:
        if act_guided_chunking == "reg_cosine":
            assert chunking_model is not None, "Chunking model must be defined"
            tok_outputs = tok_outputs.to(chunking_model.device)
            inputs = {"input_ids": tok_outputs["input_ids"],
                        "labels": tok_outputs["input_ids"],
                        "attention_mask":  tok_outputs["attention_mask"]}
            with torch.no_grad():
                outputs = chunking_model(**inputs, output_hidden_states=True)
                act_signals = torch.stack([outputs.hidden_states[i] for i in layer_selection], dim=0).mean(dim=0)
                del outputs

    if nltk_chunker is not None:
        contexts = [add_gist_str_using_linguistic_chunker(
            context=ctx,
            nltk_chunker=nltk_chunker,
            gist_token=tokenizer.convert_ids_to_tokens(gist_token_id)
        ) for ctx in contexts]
        add_gist = False # turns off further addition of gist tokens, since it's already added

    # potentially insert gist, then de-pad
    for i in range(len(batch)):
        if nltk_chunker is not None:
            batch[i]["context"] = contexts[i]
        else:
            #shorten surprise sequence, as we don't care about padding
            len_seq = len(tok_outputs_unpadded["input_ids"][i])
            unpadded_len = len(tok_outputs["input_ids"][i])
            unpadded_seq = tok_outputs["input_ids"][i][unpadded_len - len_seq:]
            surprise = None # surprises[i][unpadded_len - len_seq:] if entropy_model is not None else None
            act_sig = act_signals[i][unpadded_len - len_seq:] if act_signals is not None else None

            padded_seq_with_gist = apply_gist(unpadded_seq.tolist(),
                gist_scheme=gist_scheme,
                gist_token_id=gist_token_id,
                compression_rate=compression_rate,
                gist_granularity=gist_granularity,
                surprises=surprise,
                act_guided_chunking=act_guided_chunking,
                act_signal=act_sig,
                alpha_unif=alpha_unif) if add_gist else unpadded_seq


            batch[i]["context"] = tokenizer.decode(padded_seq_with_gist)
            pass


    contexts = []
    input_ids = []
    attention_mask = []
    label_ids = []

    references = []

    if tokenizer.bos_token is None:
        tokenizer.bos_token = tokenizer.pad_token

    for example in batch:
        ctx = tokenizer.bos_token + "Context: " + example["context"]
        que = "\n\nQuestion: " + example["question"]
        que += "\n\n<think>\n\n</think>" if add_thinking_tags else ""
        que += "\n\nAnswer: "
        ans = example["answers"]["text"][0]  + tokenizer.eos_token if len(example["answers"]["text"]) > 0  else ""

        ctx_tok = tokenizer(ctx, return_tensors="pt")["input_ids"][0]
        que_tok = tokenizer(que, return_tensors="pt")["input_ids"][0]
        ans_tok = tokenizer(ans, return_tensors="pt")["input_ids"][0]

        references.append({"answers": example["answers"], "id": example["id"]})


        inputs = torch.cat([ctx_tok, que_tok], dim=0)
        input_ids.append(inputs)
        attention_mask.append(torch.ones_like(inputs))

        labels = ans_tok
        label_ids.append(labels)


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
    parser.add_argument("--output_dir", type=str, default="runs/eval")
    general_args, unknown = parser.parse_known_args()
    experiment_args, _ = parse_chunk_exp_args(unknown)
    args = join_args(experiment_args, general_args)


    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    # max_memory_mapping = {
    #     0: "8GiB",   
    #     1: "30GiB",   
    # }

    model = ZipQwen3ForCausalLM.from_pretrained(args.model, device_map="auto").to(torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")

    align_special_tokens(tok, model)



    entropy_model = None

    chunking_model = Qwen3Chunker.from_pretrained(args.chunking_model, device_map="auto").to(torch.bfloat16) if args.chunking_model is not None else None
    valset = load_dataset("rajpurkar/squad_v2")["validation"]#.select(range(100))


    collator_args = deepcopy(args)
    collator = ChunkCollator(tok, validation_mode=True, collator_args=collator_args)

    dloader = DataLoader(valset,
                         batch_size=12,
                         shuffle=False,
                         collate_fn=collator)

    accelerator = Accelerator()
    # model, tok, dloader = accelerator.prepare(model, tok, dloader)
    # chunking_model = accelerator.prepare_model(chunking_model, evaluation_mode=True) if chunking_model is not None else None
    model.eval()
    if chunking_model is not None: 
        chunking_model.eval()

    gen_config = {
        "num_beams": 1,
        "do_sample": True,
        "top_k": 20,
        "top_p": 0.95,
        "max_new_tokens": 128
    }

    all_preds = []
    all_references = []
    all_comp_rates = []

    tbar = tqdm(dloader, desc="Evaluating")

    idx=0
    for batch, reference in tbar:
        device_type = chunking_model.device
        batch = batch.to(device_type)
        batch.pop("labels")
        token_type_ids = batch.pop("token_type_ids")

        if args.mask_mode == "hard": 
            chunk_signal = F.one_hot(token_type_ids, num_classes=2).float()
        elif args.mask_mode == "soft": 
            chunker_inputs = deepcopy(batch) 
            # chunker_inputs["labels"] = token_type_ids
            with torch.no_grad():
                logits = chunking_model(**chunker_inputs)["logits"]
            chunk_signal = F.gumbel_softmax(logits, hard=True)
        elif args.mask_mode == "contextless":
            modded_ids_lst = []
            for j in range(batch["attention_mask"].shape[0]):
                ctx_end = (token_type_ids[j] == 0).nonzero(as_tuple=False)[-1].item() + 1
                mod_ids = token_type_ids[j] 
                mod_ids[:ctx_end] = 0
                modded_ids_lst.append(mod_ids)
            modded_ids = torch.stack(modded_ids_lst, dim=0)
            chunk_signal = F.one_hot(modded_ids, num_classes=2)
        elif args.mask_mode == "full":
            token_types = torch.ones_like(batch["attention_mask"])
            chunk_signal = F.one_hot(token_types, num_classes=2)
        
        batch["chunk_signal"] = chunk_signal 

        compression_rates = compute_soft_compression_rate(chunk_signal, batch["attention_mask"], token_type_ids, compute_mean=False)
        all_comp_rates.append(compression_rates)

        device_type = model.device
        batch = batch.to(device_type)

        with torch.no_grad():
            preds = model.generate(**batch, **gen_config)
        preds[:, :batch["input_ids"].shape[1]] = -100

        preds_decoded = tok.batch_decode(torch.where(preds == -100, tok.pad_token_id, preds), skip_special_tokens=True)
        preds_decoded = [p.strip() for p in preds_decoded]

        preds_dict = [{"prediction_text": p if p != "Not enough information." else "" , "id": r["id"], "no_answer_probability": 1. if p == "Not enough information." else 0.} for p, r in zip(preds_decoded, reference)]
        [print(p, "||", r["answers"]["text"][0] if len(r["answers"]["text"]) > 0 else "Not enough information.") for p, r in zip(preds_decoded, reference)]
        all_preds += preds_dict
        all_references += reference
        if idx > 1000: 
            break
        if idx%10 == 0:
            softmask = model.model.compute_soft_chunk_mask(chunk_signal, batch["attention_mask"].shape[1], batch["attention_mask"].shape[1])
            softmask = softmask.detach().cpu()[0, 0] # isolate just the first one of the batch 
            softmask *= batch["attention_mask"][0][:, None].cpu() * batch["attention_mask"][0][None, :].cpu() # mask out padding tokens
            plt.imshow(softmask.numpy())
            plt.savefig(os.path.join(args.output_dir, f"sample_{idx:04}_mask.png"), dpi=300)
            plt.close()


            ctx_begin = (batch["attention_mask"][0] == 1).nonzero(as_tuple=False)[0].item()
            token_type_labels = (chunk_signal[0][ctx_begin:, 1] >= 0.5)
            tokens = tok.convert_ids_to_tokens(batch["input_ids"][0, ctx_begin:])

            assert len(tokens) == len(token_type_labels), f"each token should be associated with a token type"
            with open(os.path.join(args.output_dir, f"sample_{idx:04}_tokens.tsv"), "w") as f: 
                f.write("token\ttype\n")
                for k in range(len(tokens)): 
                    f.write(f"{tokens[k]}\t{token_type_labels[k]}\n")

                f.write("pred\treference\n") 
                f.write(f'{preds_decoded[0]}\t{reference[0]["answers"]["text"][0] if len(reference[0]["answers"]["text"]) > 0 else "Not enough information."}\n')
            

            pass

        idx+=1 



    all_preds = accelerator.gather_for_metrics(all_preds, True)
    all_references = accelerator.gather_for_metrics(all_references, True)
    all_comp_rates = accelerator.gather_for_metrics(all_comp_rates, True)
    mean_comp_rate = torch.cat(all_comp_rates, dim=-1).mean().item()
    std_comp_rate = torch.std(torch.cat(all_comp_rates, dim=-1).flatten()).item()

    with open("runs/eval/preds.txt", "w") as f:
        for p in all_preds:
            f.write(p["prediction_text"] + "\n")
    with open("runs/eval/ref.txt", "w") as f:
        for r in all_references:
            write_out = r["answers"]["text"][0] + "\n" if len(r["answers"]["text"]) > 0 else "Not enough information.\n"
            f.write(write_out)

    squad_v2_metric = evaluate.load("squad_v2")
    results = squad_v2_metric.compute(predictions=all_preds, references=all_references)
    results["compression_rate"] = mean_comp_rate
    results["std_comp_rate"] = std_comp_rate
    print(results)
