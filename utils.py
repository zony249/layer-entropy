import os
import re
from typing import List, Dict, Tuple, Union, Optional, Any
from dataclasses import dataclass
from copy import deepcopy
import gc
from argparse import Namespace

import numpy as np
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.tag import pos_tag
from nltk.tree.tree import Tree
from nltk.chunk.api import ChunkParserI
import torch
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    PreTrainedTokenizer,
    PreTrainedTokenizerBase,
    PreTrainedModel,
    ProcessorMixin,
    BatchEncoding
)
from transformers.data.data_collator import (
    DataCollatorMixin
)
from transformers.utils import logging



logger = logging.get_logger(__name__)

class DefaultArgs(Namespace):
    gist_scheme: str = "dispersed"
    add_thinking_tags: bool = True
    compression_rate: int | None = None
    add_gist: bool = True
    gist_granularity: int = 1
    entropy_model: PreTrainedModel | None = None
    surprise_mode: str | None = None
    entropy_model_temp: float = 1.0
    act_guided_chunking: str | None = None
    attention_guided_chunking: str | None = None
    chunking_model: PreTrainedModel | None = None
    nltk_chunker: Any | None = None
    use_layers: List[int] | None = None
    alpha_unif: float = 0




@dataclass
class GistDataCollator(DataCollatorMixin):
    """
    Very simple data collator that simply collates batches of dict-like objects and performs special handling for
    potential keys named:

        - `label`: handles a single value (int or float) per object
        - `label_ids`: handles a list of values per object

    Does not do any additional preprocessing: property names of the input object will be used as corresponding inputs
    to the model. See glue and ner for example of how it's useful.

    This is an object (like other data collators) rather than a pure function like default_data_collator. This can be
    helpful if you need to set a return_tensors value at initialization.

    Args:
        return_tensors (`str`, *optional*, defaults to `"pt"`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    return_tensors: str = "pt"

    def __init__(self,
                 tokenizer: PreTrainedTokenizer,
                 gist_token_id: int,
                 add_thinking_tags: bool = True,
                 # gist_scheme: Optional[str] = None,
                 # compression_rate: Optional[float] = None,
                 # add_thinking_tags: Optional[bool] = False,
                 # add_gist: Optional[bool] = True,
                 # gist_granularity: Optional[int] = 1,
                 # entropy_model: Optional[PreTrainedModel] = None,
                 # surprise_mode: Optional[str] = None,
                 # temp: Optional[float] = 1,
                 # act_guided_chunking: Optional[str] = None,
                 # chunking_model: Optional[PreTrainedModel] = None,
                 # nltk_chunker: Optional[ChunkParserI] = None,
                 collate_args: Namespace | None = None,
                 eval_mode: bool = False,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.args = collate_args if collate_args is not None else DefaultArgs()
        self.tokenizer = tokenizer
        self.gist_token_id = gist_token_id
        self.gist_scheme = "end" if self.args.gist_scheme is None else self.args.gist_scheme
        self.add_thinking_tags = add_thinking_tags
        self.compression_rate = self.args.compression_rate
        self.add_gist = self.args.add_gist
        self.gist_granularity = self.args.gist_granularity
        self.entropy_model = self.args.entropy_model
        self.surprise_mode = self.args.surprise_mode
        self.temp = self.args.entropy_model_temp
        self.act_guided_chunking = None if self.args.act_guided_chunking == "none" else self.args.act_guided_chunking
        self.attention_guided_chunking = self.args.attention_guided_chunking
        self.chunking_model = self.args.chunking_model
        self.nltk_chunker = self.args.nltk_chunker
        self.layer_selection = [1] if self.args.use_layers is None else self.args.use_layers
        self.alpha_unif = self.args.alpha_unif
        self.eval_mode = eval_mode

        # logging
        self.seq_count = 0
        self.rate = 0

        self.chunk_diff_norm = []
        self.uniform_chunk_diff_norm = []
        self.multi_token_chunk_diff_norm = []

        if self.add_gist:
            if self.act_guided_chunking is not None or self.surprise_mode is not None:
                assert self.compression_rate is not None, f"compression_rate must be specified if gist tokens are to be added"

        if self.act_guided_chunking is not None:
            assert self.chunking_model is not None, "act_guided_chunking requires a chunking_model"



    def __call__(self, batch: List[Dict[str, Any]]) -> BatchEncoding:
        """
        features: [(context, question, answers), ...]
        """
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if self.tokenizer.bos_token is None:
            self.tokenizer.bos_token = self.tokenizer.pad_token

        if self.attention_guided_chunking: 
            pass
            inputs = [example["context"] + "\n\nQuestion: " + example["question"] for example in batch]
            contexts = [example["context"] for example in batch]
            questions = ["\n\nQuestion: " + example["question"] for example in batch]
            tok_outputs = self.tokenizer(inputs, return_tensors="pt", padding=True, padding_side="left")
            contexts_unpadded: List = self.tokenizer(contexts)
            questions_unpadded: List = self.tokenizer(questions)
            tok_outputs_unpadded = contexts_unpadded
        else:
            contexts = [example["context"] for example in batch]
            tok_outputs = self.tokenizer(contexts, return_tensors="pt", padding=True, padding_side="left")
            tok_outputs_unpadded = self.tokenizer(contexts)
        

        ####################################################
        ##### post-tokenization-based gist assignment: #####
        ####################################################

        if self.entropy_model is not None:
            logger.warning("Entropy model is deprecated. This will do nothing for now, and will be removed in the future.")
            if False:
                tok_outputs = tok_outputs.to(self.entropy_model.device)
                inputs = {"input_ids": tok_outputs["input_ids"],
                            "labels": tok_outputs["input_ids"],
                            "attention_mask":  tok_outputs["attention_mask"]}
                surprises = compute_surprise(inputs,
                                            self.entropy_model,
                                            self.tokenizer,
                                            surprise_mode=self.surprise_mode,
                                            temp=self.temp).cpu()

        act_signals: torch.Tensor | None = None
        if self.act_guided_chunking is not None:
            if self.act_guided_chunking == "normdiff":
                tok_outputs = tok_outputs.to(self.chunking_model.device)
                inputs = {"input_ids": tok_outputs["input_ids"],
                            "labels": tok_outputs["input_ids"],
                            "attention_mask":  tok_outputs["attention_mask"]}
                if self.chunking_model is not None:
                    act_signals = compute_norm_of_diffs(inputs, self.chunking_model, None, self.act_guided_chunking)
                else:
                    raise ValueError("Chunking model not assigned")

        if self.act_guided_chunking is not None:
            if self.act_guided_chunking == "min_chunk_diff":
                assert self.chunking_model is not None, "Chunking model must be defined"
                tok_outputs = tok_outputs.to(self.chunking_model.device)
                inputs = {"input_ids": tok_outputs["input_ids"],
                            "labels": tok_outputs["input_ids"],
                            "attention_mask":  tok_outputs["attention_mask"]}
                with torch.no_grad():
                    outputs = self.chunking_model(**inputs, output_hidden_states=True)
                    act_signals = torch.stack([outputs.hidden_states[i] for i in self.layer_selection], dim=0).mean(dim=0)
                    del outputs

        if self.act_guided_chunking is not None:
            if self.act_guided_chunking == "reg_cosine":
                assert self.chunking_model is not None, "Chunking model must be defined"
                tok_outputs = tok_outputs.to(self.chunking_model.device)
                inputs = {"input_ids": tok_outputs["input_ids"],
                            "labels": tok_outputs["input_ids"],
                            "attention_mask":  tok_outputs["attention_mask"]}
                with torch.no_grad():
                    outputs = self.chunking_model(**inputs, output_hidden_states=True)
                    act_signals = torch.stack([outputs.hidden_states[i] for i in self.layer_selection], dim=0).mean(dim=0)
                    del outputs
                    # act_signals = torch.stack(outputs.hidden_states, dim=0).mean(dim=0)
        batch_attn_states = None 
        if self.attention_guided_chunking is not None: 
            if self.attention_guided_chunking == "q-wise": 
                assert self.chunking_model is not None, "Attention guided chunking requires a chunking model" 
                batch_attn_states = compute_attention_states(
                    model=self.chunking_model, 
                    inputs=tok_outputs, 
                    unpadded_contexts=contexts_unpadded, 
                    unpadded_questions=questions_unpadded
                )
                pass

        ################################
        ##### NLTK GIST ASSIGNMENT #####
        ################################

        if self.nltk_chunker is not None:
            contexts = [add_gist_str_using_linguistic_chunker(
                context=ctx,
                nltk_chunker=self.nltk_chunker,
                gist_token=self.tokenizer.convert_ids_to_tokens(self.gist_token_id)
            ) for ctx in contexts]
            self.add_gist = False # turns off further addition of gist tokens, since it's already added


        uniform_gist = []
        # potentially insert gist, then de-pad
        for i in range(len(batch)):
            if self.nltk_chunker is not None:
                batch[i]["context"] = contexts[i]
            else:
                #shorten surprise sequence, as we don't care about padding
                len_seq = len(tok_outputs_unpadded["input_ids"][i])
                padded_len = len(tok_outputs["input_ids"][i])
                unpadded_seq: List[int] =  contexts_unpadded["input_ids"][i] #tok_outputs["input_ids"][i][padded_len - len_seq:].tolist()
                surprise = None # surprises[i][unpadded_len - len_seq:] if self.entropy_model is not None else None
                act_sig: torch.Tensor | None = act_signals[i][padded_len - len_seq:] if act_signals is not None else None
                attn_sig: torch.Tensor | None = batch_attn_states[i] if batch_attn_states is not None else None


                padded_seq_with_gist = apply_gist(deepcopy(unpadded_seq),
                    gist_scheme=self.gist_scheme,
                    gist_token_id=self.gist_token_id,
                    compression_rate=self.compression_rate,
                    gist_granularity=self.gist_granularity,
                    surprises=surprise,
                    act_guided_chunking=self.act_guided_chunking,
                    act_signal=act_sig,
                    attention_guided_chunking=self.attention_guided_chunking, 
                    attn_signal=attn_sig,
                    alpha_unif=self.alpha_unif) if self.add_gist else deepcopy(unpadded_seq)

                unif = apply_gist(deepcopy(unpadded_seq),
                    gist_scheme=self.gist_scheme,
                    gist_token_id=self.gist_token_id,
                    compression_rate=self.compression_rate,
                    gist_granularity=self.gist_granularity,
                ) if self.add_gist else deepcopy(unpadded_seq)

                batch[i]["context"] = self.tokenizer.decode(padded_seq_with_gist)
                uniform_gist.append(self.tokenizer.decode(unif))

                if self.act_guided_chunking == "min_chunk_diff":
                    self.log_chunk_diff(deepcopy(unpadded_seq), act_sig, self.compression_rate)
                    pass

        if self.act_guided_chunking is not None:
            del act_signals
            gc.collect()
            torch.cuda.empty_cache()

        contexts = []
        input_ids = []
        attention_mask = []
        label_ids = []


        references = []

        for example in batch:
            ctx = self.tokenizer.bos_token + "Context: " + example["context"]
            que = "\n\nQuestion: " + example["question"]
            que += "\n\n<think>\n\n</think>" if self.add_thinking_tags else ""
            que += "\n\nAnswer: "
            if self.eval_mode: 
                ans = example["answers"]["text"][0]  + self.tokenizer.eos_token if len(example["answers"]["text"]) > 0  else ""
            else: 
                ans = example["answers"] + self.tokenizer.eos_token
            

            ctx_tok = self.tokenizer(ctx, return_tensors="pt")["input_ids"][0]
            que_tok = self.tokenizer(que, return_tensors="pt")["input_ids"][0]
            ans_tok = self.tokenizer(ans, return_tensors="pt")["input_ids"][0]

            if self.eval_mode: 
                references.append({"answers": example["answers"], "id": example["id"]})
                inputs = torch.cat([ctx_tok, que_tok], dim=0)
                labels = ans_tok
                label_ids.append(labels)
            else:
                inputs = torch.cat([ctx_tok, que_tok, ans_tok], dim=0)
                labels = torch.cat([torch.ones_like(ctx_tok) * -100,
                                    torch.ones_like(que_tok) * -100,
                                    ans_tok], dim=0)
                label_ids.append(labels)
                assert len(inputs) == len(labels)

            input_ids.append(inputs)
            attention_mask.append(torch.ones_like(inputs))


            # log compression_rate
            self.log_compression_rate(ctx_tok, gist_token_id=self.gist_token_id)


        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id, padding_side="left")
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0, padding_side="left")

        label_ids = pad_sequence(label_ids, batch_first=True, padding_value=-100, padding_side="left")

        gist_positions = find_tok_pos(input_ids, self.gist_token_id)

        if self.eval_mode: 
            return BatchEncoding({
                "input_ids": input_ids,
                "labels": label_ids,
                "attention_mask": attention_mask,
                # "gist_positions": gist_positions
            }), references

        return BatchEncoding({
            "input_ids": input_ids,
            "labels": label_ids,
            "attention_mask": attention_mask,
            # "gist_positions": gist_positions
        })

    def log_compression_rate(self,
                             ctx: torch.LongTensor,
                             gist_token_id: int):
        num_gist = (ctx == gist_token_id).sum()
        num_non_gist = (ctx != gist_token_id).sum()
        rate = num_non_gist / num_gist
        self.seq_count += 1
        self.rate = (self.seq_count-1)/self.seq_count * self.rate + 1/self.seq_count * rate


    def log_chunk_diff(self,
        context: List[int],
        hidden_states: torch.FloatTensor,
        compression_rate: float):
        splits = int(len(context) // compression_rate)
        merge_mapping = chunk_from_minumum_vector_distance(hidden_states, splits)

        merge_mapping_uniform = {}
        chunk_size = int(len(context) // splits)
        for i in range(splits):
            merge_mapping_uniform[i*chunk_size] = list(range(i*chunk_size, (i+1)*chunk_size))

        guided_chunking = np.array(compute_chunk_diff(hidden_states, merge_mapping))
        uniform_chunking = np.array(compute_chunk_diff(hidden_states, merge_mapping_uniform))

        chunk_diff_norm = np.mean(guided_chunking)
        uniform_chunk_diff_norm = np.mean(uniform_chunking)
        multi_token_chunk_diff_norm = guided_chunking[guided_chunking > 0].mean()


        self.chunk_diff_norm.append(chunk_diff_norm)
        self.uniform_chunk_diff_norm.append(uniform_chunk_diff_norm)
        self.multi_token_chunk_diff_norm.append(multi_token_chunk_diff_norm)

    def get_chunk_diff(self):
        return np.mean(self.chunk_diff_norm)

    def get_uniform_chunk_diff(self):
        return np.mean(self.uniform_chunk_diff_norm)

    def get_multi_tok_chunk_diff(self):
        return np.mean(self.multi_token_chunk_diff_norm)


@dataclass
class ChunkCollator(DataCollatorMixin): 

    def __init__(self, 
                 tokenizer: PreTrainedTokenizer, 
                 validation_mode: bool = False, 
                 collator_args: Namespace | None = None, 
                 *args, 
                 **kwargs): 
        super().__init__() 
        self.args = collator_args if collator_args is not None else DefaultArgs()

        self.tokenizer = tokenizer
        self.tokenizer.bos_token = self.tokenizer.eos_token 
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.validation_mode = validation_mode

        self.compression_rate = self.args.compression_rate
        self.add_sink = self.args.add_sink
    
    def __call__(self, examples: Any) -> BatchEncoding: 
        """
        examples: [{"context": str, 
                    "question": str, 
                    "answer": str}, ...]
        """
        input_ids_list: List = []
        labels_list: List = []
        attention_mask_list: List = []
        token_type_ids_list: List = []
        references = [] if self.validation_mode else None

        for sample in examples: 
            if not self.validation_mode:
                _input_ids, _labels, _token_type_ids = self.process_train_sample(sample) 
            else:      
                _input_ids, _labels, _token_type_ids = self.process_val_sample(sample)
                references.append({"answers": sample["answers"], "id": sample["id"]})

            input_ids_list.append(torch.tensor(_input_ids))
            labels_list.append(torch.tensor(_labels))
            attention_mask_list.append(torch.tensor([1 for _ in range(len(_input_ids))]))
            token_type_ids_list.append(torch.tensor(_token_type_ids))
            

        
        input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=self.tokenizer.pad_token_id, padding_side="left")
        labels = pad_sequence(labels_list, batch_first=True, padding_value=-100, padding_side="left")
        attention_mask = pad_sequence(attention_mask_list, batch_first=True, padding_value=0, padding_side="left").long()
        token_type_ids = pad_sequence(token_type_ids_list, batch_first=True, padding_value=0, padding_side="left").long() 
        batch_enc = BatchEncoding({
            "input_ids": input_ids, 
            "labels": labels, 
            "attention_mask": attention_mask, 
            "token_type_ids": token_type_ids
        })
        if self.validation_mode: 
            return batch_enc, references 
        return batch_enc

    def process_train_sample(self, sample: Dict[str, str]) -> Tuple[List, List, List]: 
        ctx = "Context: " + sample["context"]
        que = "\n\nQuestion: " + sample["question"] + "\n\n<think>\n\n</think>" + "\n\nAnswer: "
        ans = sample["answers"] + self.tokenizer.eos_token
        
        ctx_ids = self.tokenizer.encode(self.tokenizer.bos_token + ctx)
        que_ids = self.tokenizer.encode(que)
        ans_ids = self.tokenizer.encode(ans)
        input_ids = ctx_ids + que_ids + ans_ids

        que_start = len(ctx_ids)
        interval = int(np.round(self.compression_rate))
        token_type_ids_rev = []
        i = 0
        while i < que_start: 
            if i % interval == 0: 
                token_type_ids_rev.append(1) 
            else: 
                token_type_ids_rev.append(0) 
            i+=1 

        if self.add_sink: 
            i = len(token_type_ids_rev) - 1 
            count_ones = 0
            while i > 0: 
                if token_type_ids_rev[i] == 1: 
                    count_ones += 1 
                    token_type_ids_rev.pop(i) 
                if count_ones >= 3: 
                    break 
                i -= 1
            token_type_ids_rev += [1 for _ in range(count_ones)]


        
        labels = torch.cat([torch.ones(len(ctx_ids)) * -100, 
                            torch.ones(len(que_ids)) * -100, 
                            torch.tensor(ans_ids)]).long().tolist()
        
        token_type_ids_rev = [1 for _ in range(len(que_ids) + len(ans_ids))] + token_type_ids_rev 
        token_type_ids = token_type_ids_rev[::-1]

        return input_ids, labels, token_type_ids
    

    def process_val_sample(self, sample: Any) -> Tuple[List, List, List]: 
        pass
        ctx = "Context: " + sample["context"]
        que = "\n\nQuestion: " + sample["question"] + "\n\n<think>\n\n</think>" + "\n\nAnswer: "
        ans = sample["answers"]["text"][0]  + self.tokenizer.eos_token if len(sample["answers"]["text"]) > 0  else ""
        
        ctx_ids = self.tokenizer.encode(self.tokenizer.bos_token + ctx)
        que_ids = self.tokenizer.encode(que)
        ans_ids = self.tokenizer.encode(ans)

        input_ids = ctx_ids + que_ids 
        labels = ans_ids 

        que_start = len(ctx_ids)
        interval = int(np.round(self.compression_rate))
        token_type_ids_rev = []
        i = 0
        while i < que_start: 
            if i % interval == 0: 
                token_type_ids_rev.append(1) 
            else: 
                token_type_ids_rev.append(0) 
            i+=1 

        if self.add_sink: 
            i = len(token_type_ids_rev) - 1 
            count_ones = 0
            while i > 0: 
                if token_type_ids_rev[i] == 1: 
                    count_ones += 1 
                    token_type_ids_rev.pop(i) 
                if count_ones >= 3: 
                    break 
                i -= 1
            token_type_ids_rev += [1 for _ in range(count_ones)]



        token_type_ids_rev = [1 for _ in range(len(que_ids))] + token_type_ids_rev 
        token_type_ids = token_type_ids_rev[::-1]
        return input_ids, labels, token_type_ids







def compute_surprise(inputs: Any,
                     entropy_model: PreTrainedModel,
                     tokenizer: PreTrainedTokenizer,
                     surprise_mode: Optional[str]=None,
                     temp: Optional[float] = 1):
    with torch.no_grad():
        outputs = entropy_model(**inputs)
    logits = outputs.logits / temp
    b,s,c = outputs.logits.shape
    labels_shift_left = torch.cat([inputs["labels"][:, 1:], torch.ones_like(inputs["labels"][:, 0:1]) * -100], dim=1)

    if surprise_mode == "ce":
        # nll
        surprise = F.cross_entropy(logits.view(-1, c), labels_shift_left.view(-1), reduction="none").view(b, s)
    else:
        # entropy
        surprise = (-F.softmax(outputs.logits, dim=-1) * F.log_softmax(outputs.logits, dim=-1)).sum(dim=-1)
    return surprise


def compute_norm_of_diffs(inputs: BatchEncoding,
                          chunking_model: PreTrainedModel,
                          tokenizer: PreTrainedTokenizer,
                          chunking_mode) -> torch.Tensor:
    with torch.no_grad():
        outputs = chunking_model(**inputs, output_hidden_states=True)

    hidden_states = outputs.hidden_states[-1]
    b, s, h = hidden_states.shape
    diff = hidden_states[:, 1:] - hidden_states[:, :-1]
    norms = diff.norm(dim=-1)
    return norms # [b, s-1]

def compute_attention_states(
    model: PreTrainedModel, 
    inputs: BatchEncoding,  
    unpadded_contexts: List[List[int]], 
    unpadded_questions: List[List[int]], 
) -> List[torch.Tensor]: 
    assert model.config._attn_implementation == "eager", "Attention implementation must be set to eager to output attentions"
    inputs = inputs.to(model.device) 
    with torch.no_grad():
        attn_states = torch.stack(model(**inputs, output_attentions=True).attentions, dim=0).mean(dim=(0, 2))
    ctx_lens = [len(x) for x in unpadded_contexts["input_ids"]]
    question_lens = [len(x) for x in unpadded_questions["input_ids"]]
    input_lens = [x + y for x, y in zip(ctx_lens, question_lens)]
    question_to_context_attn_states = [] 
    for i in range(len(attn_states)): 
        context_start = attn_states[i].shape[1] - input_lens[i]
        question_start = context_start + ctx_lens[i]
        question_to_context_attn_states.append(attn_states[i][question_start:, context_start:question_start])
    return question_to_context_attn_states


def apply_gist(context: List[int],
               gist_scheme: str,
               gist_token_id: int,
               compression_rate: float | None,
               gist_granularity: Optional[int] = 1,           # Deprecated
               surprises: Optional[torch.FloatTensor] = None, # Deprecated
               act_guided_chunking:  Optional[str] = None,
               attention_guided_chunking: str | None = None, 
               act_signal: Optional[torch.Tensor] = None,
               attn_signal: torch.Tensor = None, 
               nltk_chunker: Optional[ChunkParserI] = None,
               alpha_unif: float | None = None) -> torch.Tensor | List[int]:
    """
    context: [seq_len]
        post-tokenization sequence tensor
    """

    if gist_scheme == "dispersed":
        assert compression_rate is not None, "compression_rate must be specified"
        if surprises is not None:
            if False:
                return apply_dispersed_gist_with_surprise_guidance(context,
                                                                gist_token_id,
                                                                compression_rate,
                                                                surprises)
        if act_guided_chunking is not None:
            if act_guided_chunking == "normdiff":
                assert act_signal is not None, "norm_diff requires activation signal"
                return apply_dispersed_gist_with_norm_diff_guidance(context,
                                                             gist_token_id,
                                                             compression_rate,
                                                             normdiffs = act_signal)
            elif act_guided_chunking == "min_chunk_diff":
                assert act_signal is not None, "min_chunk_diff requires activation signal"
                alpha_unif = alpha_unif if alpha_unif is not None else 0
                return apply_dispersed_gist_with_min_chunk_diff_objective(
                    context=context,
                    gist_token_id=gist_token_id,
                    compression_rate=compression_rate,
                    hidden_states=act_signal,
                    alpha=alpha_unif
                )
            elif act_guided_chunking == "reg_cosine":
                assert act_signal is not None, "regularized cosine chunking requires activation signal"
                alpha = alpha_unif if alpha_unif is not None else 0
                return apply_gist_with_reg_cosine_chunking(
                    context=context,
                    gist_token_id=gist_token_id,
                    compression_rate=compression_rate,
                    hidden_states=act_signal,
                    alpha=alpha
                )
            else:
                raise NotImplementedError 
        if attention_guided_chunking is not None: 
            if attention_guided_chunking == "q-wise": 
                assert attn_signal is not None, "q-wise attention_guided_chunking requires attention signal"
                return apply_gist_with_q_wise_attention_guidance(
                    contexts=context, 
                    attention_signal=attn_signal, 
                    compression_rate=compression_rate, 
                    gist_token_id=gist_token_id
                )

        if nltk_chunker is not None:
            pass

        return apply_dispersed_gist(context,
                                    gist_token_id,
                                    compression_rate,
                                    gist_granularity)
    elif gist_scheme == "end":
        return apply_end_gist(context,
                              gist_token_id,
                              compression_rate)
    else:
        raise NotImplementedError()

def apply_dispersed_gist(context: List[int],
                         gist_token_id: int,
                         compression_rate: float,
                         gist_granularity: int = 1) -> torch.Tensor:
    if len(context) == 0:
        return torch.tensor([gist_token_id])
    context.reverse()
    baseline = int(len(context) // gist_granularity)
    i = 0
    while i < len(context):
        for _ in range(min(baseline, gist_granularity)):
            context.insert(i, gist_token_id)

        i += int(gist_granularity * (compression_rate + 1))
    context.reverse()
    return torch.tensor(context)


def apply_dispersed_gist_with_surprise_guidance(
                        context: List[int],
                         gist_token_id: int,
                         compression_rate: float,
                         surprises: torch.FloatTensor) -> torch.Tensor:
    """
    context: List[int]: assumes no padding
    surprises: torch.FloatTensor[seq_len]: Token-by-token loss
    """
    if len(context) == 0:
        return torch.tensor([gist_token_id])

    num_gist_tokens = int(np.ceil(len(context) / compression_rate))
    context = context[::-1]

    total_surprise = surprises.sum()
    surprises_r = surprises.flip(dims=(0,))
    surprises_cs = torch.cumsum(surprises_r, dim=0)
    split_interval = total_surprise / num_gist_tokens


    interp_surprises_cs = F.interpolate(surprises_cs[None, None, :], 5* len(surprises_cs), mode="linear")[0, 0]
    scale = 1/5

    idx = []
    si = split_interval.item()
    for i, x in enumerate(interp_surprises_cs):
        if x > si:
            idx.append(i)
            si += split_interval
        if len(idx) == num_gist_tokens - 1:
            break

    idx = torch.tensor(idx, dtype=float, device=surprises.device)
    idx = torch.round(idx * scale).int()


    idx = torch.flip(idx, dims=(0,))
    for x in idx:
        context.insert(x+1, gist_token_id)
    if context[0] != gist_token_id:
        context.insert(0, gist_token_id)
    context = context[::-1]

    return torch.tensor(context, device=surprises.device)



def apply_dispersed_gist_with_norm_diff_guidance(context: List[int],
                                                 gist_token_id: int,
                                                 compression_rate: float,
                                                 normdiffs: torch.FloatTensor) -> List[int]:
    pass
    topk = len(context) // compression_rate
    idx = torch.argsort(-normdiffs)[:topk-1]
    sorted_idx = torch.sort(idx, descending=True).values

    for i in sorted_idx:
        context.insert(i, gist_token_id)
    context.append(gist_token_id)
    return context

def apply_gist_with_reg_cosine_chunking(
    context: List[int],
    gist_token_id:int,
    compression_rate: float,
    hidden_states: torch.Tensor,
    alpha: float,
):
    splits = int(np.ceil(len(context) / compression_rate))
    chunk_map = compute_reg_cosine_chunking(hidden_states, splits=splits, alpha=alpha)
    chunk_heads = sorted(chunk_map.keys())[::-1]
    for head in chunk_heads:
        # print(chunk_map[head])
        gist_idx = chunk_map[head][-1]
        context.insert(gist_idx, gist_token_id)

    return context


def apply_dispersed_gist_with_min_chunk_diff_objective(
    context: List[int],
    gist_token_id: int,
    compression_rate: float,
    hidden_states: torch.Tensor,
    alpha: float) -> List[int]:

    splits = int(np.ceil(len(context) / compression_rate))
    merge_mapping = chunk_from_minumum_vector_distance(hidden_states, splits, alpha)
    chunk_heads = sorted(merge_mapping.keys())[::-1]
    for head in chunk_heads:
        gist_idx = merge_mapping[head][-1]
        context.insert(gist_idx, gist_token_id)

    return context


def apply_end_gist(context: List[int],
                   gist_token_id: int,
                   compression_rate: float) -> torch.Tensor:
    num_gist_tokens = len(context) // compression_rate
    if num_gist_tokens == 0:
        num_gist_tokens = 1
    context = context + [gist_token_id for _ in range(int(num_gist_tokens))]
    return torch.tensor(context)

def find_tok_pos(batched_tensor:torch.LongTensor, token_id: int) -> List[torch.Tensor]:
    positions = []
    for i in range(len(batched_tensor)):
        ids = (batched_tensor[i] == token_id).nonzero(as_tuple=False).flatten()
        positions.append(ids)
    return positions


def add_gist_str_using_linguistic_chunker(context: str,
                                      nltk_chunker: ChunkParserI,
                                      gist_token: str):
    """
    This function expects contexts and gist tokens to be in str form (i.e., untokenized)
    """

    output_string: List[str] = []
    pre_tokenize: List[str] = sent_tokenize(context)

    for subseq in pre_tokenize:
        tagged = pos_tag(word_tokenize(subseq))
        tree: Tree = nltk_chunker.parse(tagged)
        for node in tree:
            if isinstance(node, Tree):
                # output_string.append(gist_token)
                for subnode in node.flatten():
                    output_string.append(subnode[0])
                output_string.append(gist_token)
            else:
                output_string.append(node[0])
        # output_string.append("." + gist_token)
    if output_string[-1] != gist_token:
        output_string.append(gist_token)

    pre_out = TreebankWordDetokenizer().detokenize(output_string)
    compact_gist = re.sub(f'\s{gist_token}\s?', gist_token, pre_out)
    output = re.sub(f'\s\.', '.', compact_gist)
    return output


def apply_gist_with_q_wise_attention_guidance(
    contexts: List[int], 
    attention_signal: torch.Tensor, 
    compression_rate: float, 
    gist_token_id: int, 
    ignore_first: bool = True
):  
    splits: int = int(np.ceil(len(contexts) / compression_rate))
    sum_of_output_weights = attention_signal.sum(dim=0)
    idx = torch.argsort(sum_of_output_weights, descending=True)
    if ignore_first: 
        idx = idx[idx != 0]
    top_k = sorted(idx[:splits].tolist(), reverse=True) 
    for x in top_k: 
        contexts.insert(x+1, gist_token_id)
    # contexts.append(gist_token_id)
    return contexts 







def chunk_from_minumum_vector_distance(
    hidden_states: torch.Tensor,
    splits: int,
    alpha: float = 0
    ) -> Dict[int, List[int]]:
    """
    Applies chunking based on minimum vector distance. Greedy algorithm
    hidden_states: [seq_len, dim]
    splits: int
    """
    seq_len, dim = hidden_states.shape
    hidden_states = hidden_states - hidden_states.mean(dim=-1, keepdim=True)

    # reg_vectors = compute_uniform_regularization(hidden_states, splits)
    # hidden_states = hidden_states + alpha * reg_vectors

    with torch.no_grad():
        indexify = [(a, i) for i, a in enumerate(hidden_states)]
        merge_mapping = {i: [i] for i in range(seq_len)}

        while len(indexify) > splits:
            # compute adjacent state distance
            arrs = torch.stack([x[0] for x in indexify], dim=0)
            diffs = (arrs[1:] - arrs[:-1]).norm(dim=-1)
            # get smallest dist idx
            merge_idx = torch.argmin(diffs)  # this points to the position in indexify
            # merge:
            base = indexify[merge_idx]
            merge_with = indexify[merge_idx + 1]

            if merge_with[1] in merge_mapping:
                append = merge_mapping[merge_with[1]]
                merge_mapping.pop(merge_with[1], None)
            else:
                append = [merge_with[1]]
            merge_mapping[base[1]] += append

            compound = (
                (base[0] * (len(merge_mapping[base[1]]) - 1) + merge_with[0])
                / len(merge_mapping[base[1]]),
                base[1],
            )
            indexify[merge_idx] = compound
            indexify.pop(merge_idx + 1)

        #        merge_mapping[merge_idx] = ()
        # split_size = hidden_states.shape[0] // splits
        # merge_mapping_uniform = {}
        # for i in range(splits):
        #     merge_mapping_uniform[i * split_size] = np.arange(
        #         i * split_size, (i + 1) * split_size
        #     )

    merge_mapping = compute_uniform_regularization(merge_mapping, splits, seq_len, alpha)

    return merge_mapping


def compute_uniform_regularization(
    merge_mapping: Dict[int, List[int]],
    splits:int,
    seq_len:int,
    alpha: float = 0,
        ) -> Dict[int, List[int]]:
    """
    applies uniform regularization weighed by alpha, and adjusts the chunk boundaries in merge_mapping
    merge_mapping: Dict[chunk_head: [chunk_elements, ...]]
    alpha: float between 0 and 1
    """

    raw_chunk_heads:torch.Tensor = torch.tensor(sorted(list(merge_mapping.keys())))
    buffer:int = int(seq_len / splits)
    unif_bounds:torch.Tensor = torch.round(torch.linspace(0, seq_len-buffer, splits)).int()
    # print(buffer, seq_len, splits, unif_bounds)

    reg_chunk_boundaries:torch.Tensor = (1-alpha) * raw_chunk_heads.float() + alpha * unif_bounds.float()
    reg_bound_round = torch.round(reg_chunk_boundaries).int()
    reg_mapping = {}
    for i in range(len(reg_bound_round)-1):
        reg_mapping[reg_bound_round[i]] = list(range(reg_bound_round[i], reg_bound_round[i+1]))
    reg_mapping[reg_bound_round[-1]] = list(range(reg_bound_round[-1], seq_len))
    return reg_mapping



def compute_chunk_diff(
    hidden_states: torch.FloatTensor,
    merge_mapping: Dict[int, List[int]]):
    """
    Computes the maximum hidden state diff within a chunk.
    hidden_states: [seq_len, dim]
    merge_mapping: Dict[chunk_start: rest of the chunk indices]
    """
    chunk_max_diffs = []
    for k in merge_mapping:
        chunk_idxs = merge_mapping[k]
        chunk_vecs = hidden_states[chunk_idxs]
        s, d = chunk_vecs.shape
        diffnorm_map = (chunk_vecs[:, None, :] - chunk_vecs[None, :, :]).norm(dim=-1)

        chunk_max_diffs.append(diffnorm_map.max().detach().item())
    return chunk_max_diffs


def compute_reg_cosine_chunking(
    hidden_states: torch.Tensor,
    splits:int,
    alpha: float = 0
) -> Dict[int, List[int]]:
    """
    hidden_states: [seq_len, dim]
    """
    # first obtain transition scores
    seq_len, dim = hidden_states.shape
    u = hidden_states[1:]
    v = hidden_states[:-1]
    cos = ((u*v).sum(dim=-1, keepdim=True) /(u.norm(dim=-1, keepdim=True) *  v.norm(dim=-1, keepdim=True))).flatten()
    dissim = 1 - cos
    dissim = dissim.detach()

    # get uniform chunk boundaries:
    buffer:int = int(seq_len / splits)
    unif_bounds:torch.Tensor = torch.round(torch.linspace(0, seq_len-buffer, splits)).int()
    new_candidate_chunks: List[int] = []
    for j in range(len(unif_bounds)):
        low = int(unif_bounds[j])
        try:
            high = int(unif_bounds[j+1])
        except IndexError:
            high = seq_len

        # within each uniform boundary, compute maximum dissimilarity and its index
        in_chunk_max_dissim=-1
        max_idx = high
        for i in range(low, high-1):
            if dissim[i] > in_chunk_max_dissim:
                in_chunk_max_dissim = dissim[i]
                max_idx = i
        new_candidate_chunks.append(max_idx)

    k: int = int(alpha * splits)
    dissim_at_candidate_chunk = torch.tensor([dissim[i] if i < len(dissim) else 0 for i in new_candidate_chunks])
    top_k = torch.argsort(dissim_at_candidate_chunk, descending=True)[:k]
    chosen_ones = [new_candidate_chunks[i] for i in top_k]

    new_bounds = deepcopy(unif_bounds)
    for i in chosen_ones:
        mod_idx = 0
        # find the uniform bound that is just greater than i
        for idx, j in enumerate(unif_bounds):
            if j >= i:
                mod_idx = idx
                new_bounds[mod_idx] = i
                break


    merge_dict = {}
    for i in range(len(new_bounds)-1):
        merge_dict[new_bounds[i]] = list(range(new_bounds[i], new_bounds[i+1]))

    merge_dict[new_bounds[-1]] = list(range(new_bounds[-1], seq_len))
    return merge_dict


def compute_soft_compression_rate(
    probs: torch.Tensor | None,
    attention_mask: torch.Tensor, 
    token_type_ids: torch.Tensor, 
    compute_mean: bool = True, 
) -> torch.Tensor: 
    """
    args: 
        probs: torch.Tensor[batch, sequence, 2]
            2-way softmax probabilities
        attention_mask: torch.Tensor[batch, sequence] 
            boolean attention mask
        token_type_ids: torch.Tensor[batch, sequence]
            token type (either 0 or 1)
    """
    if probs is None: 
        token_types = torch.ones_like(attention_mask)
        probs = F.one_hot(token_types, num_classes=2)

    compression_rates = []
    for i in range(probs.shape[0]): 
        ctx_start = (attention_mask[i] == 1).nonzero(as_tuple=False)[0].item()
        try:
            ctx_end = (token_type_ids[i] == 0).nonzero(as_tuple=False)[-1].item() + 1
        except IndexError:
            ctx_end = len(attention_mask[i])
        # ctx_end = inputs["attention_mask"][i].shape[0] - ans_len 
        comp_mass = probs[i][ctx_start:ctx_end, 1].sum().item()
        total_mass = ctx_end - ctx_start
        compression_rates.append((total_mass + 1) / (comp_mass + 1))
    comp_rates = torch.tensor(compression_rates, device=probs.device)

    if compute_mean: 
        mean_comp_rate = comp_rates.mean()
        return mean_comp_rate
    return comp_rates.flatten()







def align_special_tokens(processing_class: PreTrainedTokenizer,
                         model: PreTrainedModel):
    """
    Aligns the special tokens of the tokenizer with the model configs.

    A new tokens may be defined in the tokenizer for fine-tuning purposes, e.g. an "end of turn" token may be
    added on chat models. In that case, we want the model configs to be aligned with the tokenizer, so that all
    downstream uses work as expected. This alignment should happen before training, to ensure the prediction step
    uses the new tokens as well.


    THIS IS OBTAINED FROM HUGGINGFACE TRAINER
    """
    if isinstance(processing_class, ProcessorMixin):
        tokenizer: PreTrainedTokenizerBase = processing_class.tokenizer
    else:
        tokenizer = processing_class
    model_has_generation_config = (
        hasattr(model, "generation_config") and model.generation_config is not None
    )
    updated_tokens = {}

    # 1 - Align EOS token. EOS is more complex than the others, as `generation_config` may hold more than one EOS
    # token.
    tokenizer_has_new_eos = tokenizer.eos_token_id != model.config.eos_token_id
    if model_has_generation_config:
        # `generation_config.eos_token_id` is None: direct comparison
        if model.generation_config.eos_token_id is None:
            tokenizer_has_new_eos |= tokenizer.eos_token_id != model.generation_config.eos_token_id
        else:
            # `generation_config.eos_token_id` is an `int`: convert it to list (and continue below)
            if isinstance(model.generation_config.eos_token_id, int):
                model.generation_config.eos_token_id = [model.generation_config.eos_token_id]
            # `generation_config.eos_token_id` is a `list`: check if the tokenizer's EOS token is in the list
            tokenizer_has_new_eos |= tokenizer.eos_token_id not in model.generation_config.eos_token_id

    if tokenizer_has_new_eos:
        updated_tokens["eos_token_id"] = tokenizer.eos_token_id
        model.config.eos_token_id = tokenizer.eos_token_id
        # The generation config may hold more than one EOS token. We preserve the original EOS tokens: any of the
        # EOS tokens defined here will halt generation.
        if model_has_generation_config:
            all_eos_tokens = [tokenizer.eos_token_id]
            if model.generation_config.eos_token_id is not None:
                all_eos_tokens += list(model.generation_config.eos_token_id)
            model.generation_config.eos_token_id = [token for token in all_eos_tokens if token is not None]

    # 2 - Align BOS
    tokenizer_has_new_bos = tokenizer.bos_token_id != model.config.bos_token_id
    if model_has_generation_config:
        tokenizer_has_new_bos |= tokenizer.bos_token_id != model.generation_config.bos_token_id

    if tokenizer_has_new_bos:
        updated_tokens["bos_token_id"] = tokenizer.bos_token_id
        model.config.bos_token_id = tokenizer.bos_token_id
        if model_has_generation_config:
            model.generation_config.bos_token_id = tokenizer.bos_token_id

    # 3 - Align PAD
    tokenizer_has_new_pad = tokenizer.pad_token_id != model.config.pad_token_id
    if model_has_generation_config:
        tokenizer_has_new_pad |= tokenizer.pad_token_id != model.generation_config.pad_token_id

    if tokenizer_has_new_pad:
        updated_tokens["pad_token_id"] = tokenizer.pad_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
        if model_has_generation_config:
            model.generation_config.pad_token_id = tokenizer.pad_token_id

    # 4 - Warn users about the changes
    if len(updated_tokens) > 0:
        logger.warning(
            "The tokenizer has new PAD/BOS/EOS tokens that differ from the model config and generation config. "
            "The model config and generation config were aligned accordingly, being updated with the tokenizer's "
            f"values. Updated tokens: {updated_tokens}."
        )






if __name__ == "__main__":


    from datasets import load_dataset
    from torch.utils.data import DataLoader
    from data_utils.squad import Squad
    from transformers import AutoTokenizer, AutoModelForCausalLM


    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    # ent_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", attn_implementation="eager").cuda()
    # special_tokens = {"additional_special_tokens": ["<GIST>"]}
    # tokenizer.add_special_tokens(special_tokens)


    # import debugpy
    # debugpy.listen(("0.0.0.0", 5678))
    # print("Waiting for debugger attach...")
    # debugpy.wait_for_client()



    args = DefaultArgs()
    args.add_gist = True
    # args.act_guided_chunking = "min_chunk_diff"
    args.attention_guided_chunking = "q-wise"
    # args.chunking_model = ent_model
    args.use_layers = [1]
    args.compression_rate = 3
    args.gist_scheme = "dispersed"


    # collator = GistDataCollator(
    #     tokenizer,
    #     gist_token_id = tokenizer.convert_tokens_to_ids("<GIST>"),
    #     add_thinking_tags = True,
    #     collate_args = args
    # )

    test_sample = [
        {"context": "something wrong with this tokenizer, maybe it's the vocab size",
        "question": "why does this tokenizer suck?" ,
        "answers": "because of vocab size."},
        {"context": "I fixed the tokenizer. I realized that the tokenizer was not working properly because of one token not accounted for in the tokenizer's vocab.",
        "question": "what did I do?" ,
        "answers": "fixed the tokenizer"},
    ]

    # output = collator(test_sample)
    # pass


    # hidden_states = torch.randn((10, 3))
    # splits = 8



    # #
    # context_seq = test_sample[1]["context"]
    # toked = tokenizer([context_seq], return_tensors="pt").to("cuda")
    # hids = ent_model(**toked, output_hidden_states=True).hidden_states[10]
    # hid = hids[0]

    # compute_reg_cosine_chunking(hid, splits=8, alpha=0.5)

    # x = torch.tensor([[1, 2, 3, 4, 5, 1, 2, 3, 4], [5, 4, 3, 2, 4, 4, 5, 6, 4]])
    # poses = find_tok_pos(x, 3)

    # SANITY CHECK: DOES CAUSAL GIST MASK FUNCTIONS WORK?
    ### ANSWER: YES
    # from models.compression_utils import (
    #     create_causal_gist_mask_for_generation,
    #     create_causal_gist_mask,
    #     find_idx,
    # )

    # gist_idx = find_idx(output["input_ids"], tokenizer.convert_tokens_to_ids("<GIST>"))

    # gen_mask = create_causal_gist_mask_for_generation(output["attention_mask"],
    #                                                   num_new_tokens=3,
    #                                                   gist_idx=gist_idx)

    # pref_mask = create_causal_gist_mask(output["attention_mask"],
    #                                     gist_idx=gist_idx)

    # pass

    collator = ChunkCollator(tokenizer=tokenizer, collator_args=args)
    collator(test_sample)