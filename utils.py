import os 
from typing import List, Dict, Tuple, Union, Optional, Any
from dataclasses import dataclass 


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
                 gist_scheme: Optional[str] = None,
                 compression_rate: Optional[float] = None, 
                 add_thinking_tags: Optional[bool] = False, 
                 add_gist: Optional[bool] = True, 
                 gist_granularity: Optional[int] = 1, 
                 *args, **kwargs): 
        super().__init__(*args, **kwargs)
        self.tokenizer = tokenizer
        self.gist_token_id = gist_token_id 
        self.gist_scheme = "end" if gist_scheme is None else gist_scheme
        self.add_thinking_tags = add_thinking_tags
        self.add_gist = add_gist
        self.gist_granularity = gist_granularity

        if self.gist_scheme == "dispersed": 
            assert compression_rate is not None, f"compression rate cannot be None if gist tokens are dispersed"
        self.compression_rate = compression_rate

    

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        features: [(context, question, answers), ...]
        """
        contexts = []
        contexts_attn_mask = []
        questions = []
        questions_attn_mask = []
        answers = []
        answers_attn_mask = []
        input_ids = []
        attention_mask = []
        label_ids = []

        if self.tokenizer.pad_token is None: 
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if self.tokenizer.bos_token is None: 
            self.tokenizer.bos_token = self.tokenizer.pad_token

        for example in batch: 
            ctx = self.tokenizer.bos_token + "Context: " + example["context"]
            que = "\n\nQuestion: " + example["question"] 
            que += "\n\n<think>\n\n</think>" if self.add_thinking_tags else ""
            que += "\n\nAnswer: "
            ans = example["answers"] + self.tokenizer.eos_token

            ctx_tok = self.tokenizer(ctx)["input_ids"]
            que_tok = self.tokenizer(que, return_tensors="pt")["input_ids"][0]
            ans_tok = self.tokenizer(ans, return_tensors="pt")["input_ids"][0]

            
            ctx_tok = apply_gist(ctx_tok, 
                                 gist_scheme=self.gist_scheme, 
                                 gist_token_id=self.gist_token_id, 
                                 compression_rate=self.compression_rate, 
                                 gist_granularity=self.gist_granularity) if self.add_gist else torch.tensor(ctx_tok)

            contexts.append(ctx_tok)
            contexts_attn_mask.append(torch.ones_like(ctx_tok))

            questions.append(que_tok)
            questions_attn_mask.append(torch.ones_like(que_tok))

            answers.append(ans_tok)
            answers_attn_mask.append(torch.ones_like(ans_tok))


            inputs = torch.cat([ctx_tok, que_tok, ans_tok], dim=0)            
            input_ids.append(inputs) 
            attention_mask.append(torch.ones_like(inputs))

            labels = torch.cat([torch.ones_like(ctx_tok) * -100, 
                                torch.ones_like(que_tok) * -100, 
                                ans_tok], dim=0) 
            label_ids.append(labels) 
            assert len(inputs) == len(labels)
        
        #TODO: Collate them
        contexts = pad_sequence(contexts, batch_first=True, padding_value=self.tokenizer.pad_token_id, padding_side="left")
        contexts_attn_mask = pad_sequence(contexts_attn_mask, batch_first=True, padding_value=0, padding_side="left")

        questions = pad_sequence(questions, batch_first=True, padding_value=self.tokenizer.pad_token_id, padding_side="left")
        questions_attn_mask = pad_sequence(questions_attn_mask, batch_first=True, padding_value=0, padding_side="left")

        answers = pad_sequence(answers, batch_first=True, padding_value=self.tokenizer.pad_token_id, padding_side="left")
        answers_attn_mask = pad_sequence(answers_attn_mask, batch_first=True, padding_value=0, padding_side="left")
        
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id, padding_side="left")
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0, padding_side="left")

        label_ids = pad_sequence(label_ids, batch_first=True, padding_value=-100, padding_side="left")

        gist_positions = find_tok_pos(input_ids, self.gist_token_id)
        
        return BatchEncoding({
            "input_ids": input_ids, 
            "labels": label_ids, 
            "attention_mask": attention_mask, 
            # "gist_positions": gist_positions
        })



def apply_gist(context: List[int], 
               gist_scheme: str, 
               gist_token_id: int, 
               compression_rate: float, 
               gist_granularity: Optional[int] = 1) -> torch.Tensor: 
    """
    context: [seq_len]
        post-tokenization sequence tensor
    """

    if gist_scheme == "dispersed": 
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
                         gist_granularity: Optional[int] = 1) -> torch.Tensor: 
    if len(context) == 0:
        return torch.tensor([gist_token_id])
    context.reverse()
    i = 0
    while i < len(context): 
        for _ in range(gist_granularity):
            context.insert(i, gist_token_id)
            
        i += int(gist_granularity * (compression_rate + 1))
    context.reverse() 
    return torch.tensor(context)

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
    from transformers import AutoTokenizer 
    
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B") 
    special_tokens = {"additional_special_tokens": ["<GIST>"]}
    tokenizer.add_special_tokens(special_tokens)

    collator = GistDataCollator(
        tokenizer, 
        tokenizer.convert_tokens_to_ids("<GIST>"), 
        compression_rate=3, 
        gist_scheme="dispersed", 
        gist_granularity=3) 

    test_sample = [
        {"context": "something wrong with this tokenizer, maybe it's the vocab size", 
        "question": "why does this tokenizer suck?" ,
        "answers": "because of vocab size."}, 
        {"context": "I fixed the tokenizer. I realized that the tokenizer was not working properly because of one token not accounted for in the tokenizer's vocab.", 
        "question": "what did I do?" ,
        "answers": "fixed the tokenizer"}, 
    ]

    output = collator(test_sample)

    # x = torch.tensor([[1, 2, 3, 4, 5, 1, 2, 3, 4], [5, 4, 3, 2, 4, 4, 5, 6, 4]])
    # poses = find_tok_pos(x, 3)

    # SANITY CHECK: DOES CAUSAL GIST MASK FUNCTIONS WORK?
    ### ANSWER: YES
    from models.compression_utils import (
        create_causal_gist_mask_for_generation, 
        create_causal_gist_mask, 
        find_idx, 
    )
    
    gist_idx = find_idx(output["input_ids"], tokenizer.convert_tokens_to_ids("<GIST>"))

    gen_mask = create_causal_gist_mask_for_generation(output["attention_mask"],
                                                      num_new_tokens=3, 
                                                      gist_idx=gist_idx)
    
    pref_mask = create_causal_gist_mask(output["attention_mask"], 
                                        gist_idx=gist_idx)

    pass