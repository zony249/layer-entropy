import os 
from typing import List, Dict, Tuple, Union, Optional, Any
from dataclasses import dataclass 

import torch 
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    PreTrainedTokenizer, 
    BatchEncoding
)
from transformers.data.data_collator import (
    DataCollatorMixin
)



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
                 *args, **kwargs): 
        super().__init__(*args, **kwargs)
        self.tokenizer = tokenizer
        self.gist_token_id = gist_token_id 
        self.gist_scheme = "end" if gist_scheme is None else gist_scheme
        self.add_thinking_tags = add_thinking_tags
        self.add_gist = add_gist

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

            
            ctx_tok = self.apply_gist(ctx_tok) if self.add_gist else torch.tensor(ctx_tok)

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

        gist_positions = self.find_tok_pos(input_ids, self.gist_token_id)
        
        return BatchEncoding({
            "input_ids": input_ids, 
            "labels": label_ids, 
            "attention_mask": attention_mask, 
            # "gist_positions": gist_positions
        })



    def apply_gist(self, context: List[int]) -> torch.Tensor: 
        """
        context: [seq_len]
            post-tokenization sequence tensor
        """

        if self.gist_scheme == "dispersed": 
            return self._apply_dispersed_gist(context)
        elif self.gist_scheme == "end": 
            return self._apply_end_gist(context)
        else: 
            raise NotImplementedError() 

    def _apply_dispersed_gist(self, context: List[int]) -> torch.Tensor: 
        if len(context) == 0:
            return torch.tensor([self.gist_token_id])
        context.reverse()
        i = 0
        while i < len(context): 
            context.insert(i, self.gist_token_id)
            i += int(self.compression_rate+1)
        context.reverse() 
        return torch.tensor(context)
    def _apply_end_gist(self, context: List[int]) -> torch.Tensor: 
        num_gist_tokens = len(context) // self.compression_rate
        context = context + [self.gist_token_id for _ in range(num_gist_tokens)] 
        return torch.tensor(context)

    def find_tok_pos(self, batched_tensor:torch.LongTensor, token_id: int) -> List[torch.Tensor]: 
        positions = [] 
        for i in range(len(batched_tensor)): 
            ids = (batched_tensor[i] == token_id).nonzero(as_tuple=False).flatten()
            positions.append(ids)
        return positions

if __name__ == "__main__": 


    from datasets import load_dataset
    from torch.utils.data import DataLoader
    from data_utils.squad import Squad
    from transformers import AutoTokenizer 
    
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B") 
    special_tokens = {"extra_special_tokens": ["<GIST>"]}
    tokenizer.add_special_tokens(special_tokens)

    collator = GistDataCollator(tokenizer, tokenizer.convert_tokens_to_ids("<GIST>"), compression_rate=3, gist_scheme="dispersed") 

    test_sample = [
        {"context": "something wrong with this tokenizer, maybe it's the vocab size", 
        "question": "why does this tokenizer suck?" ,
        "answers": "because of vocab size."}, 
        {"context": "I fixed the tokenizer. it was dumb.", 
        "question": "what did I do?" ,
        "answers": "fixed the tokenizer"}, 
    ]

    collator(test_sample)

    x = torch.tensor([[1, 2, 3, 4, 5, 1, 2, 3, 4], [5, 4, 3, 2, 4, 4, 5, 6, 4]])
    poses = collator.find_tok_pos(x, 3)
    pass