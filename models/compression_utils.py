import os 
from typing import List, Dict, Tuple, Union, Optional
from copy import deepcopy

import torch 
import numpy as np 
from torch import nn 
from torch.nn import functional as F 


def find_idx(input_ids: torch.LongTensor, 
             token_id: int) -> List[torch.LongTensor]: 
    assert token_id is not None, "token_id cannot be None"
    ids = [] 
    for seq in input_ids: 
        pos = torch.nonzero(seq == token_id).squeeze() 
        if pos.ndim == 0: 
            pos = pos.unsqueeze(dim=0)
        ids.append(pos)
    return ids 

def find_context_start(input_ids: torch.LongTensor, 
                       pad_token_id: int, 
                       pad_is_bos: bool = False): 
    assert pad_token_id is not None
    non_pad_mask = input_ids != pad_token_id 
    cumsum_non_pad = torch.cumsum(non_pad_mask, dim=1) 
    idxs = find_idx(cumsum_non_pad, token_id=1)
    if pad_is_bos: 
        idxs = [x-1 for x in idxs]
    return idxs


def create_causal_gist_mask(attention_mask: torch.LongTensor, 
                            gist_idx: List[torch.LongTensor]) -> torch.BoolTensor: 
    """
    gist_idx: List of tensors of indices
    """
    causal_gist_mask = [] 

    for i, attn_seq in enumerate(attention_mask): 
        positions = gist_idx[i] 
        causal_mask = torch.tril(torch.ones_like(attn_seq[:, None] * attn_seq[None, :]))
        if len(positions) == 0: 
            causal_gist_mask.append(causal_mask * (attn_seq[:, None] * attn_seq[None, :]))
            continue
        question_start = torch.max(positions) + 1 
        positions = positions.sort().values

        # last_pos = -1
        # last_block_pos = -1
        # for j in range(question_start): 
        #     if not(j in positions and j == last_pos + 1):
        #         causal_mask[j, :last_pos+1] = 0 
        #         last_block_pos = j if torch.sum(causal_mask[j])==1 else last_block_pos 
        #     else: 
        #         causal_mask[j, :last_block_pos] = 0 
        #     if j in positions: 
        #         last_pos = j 
        last_pos = -1
        last_block_pos = -1
        set_last_block = False
        for j in range(question_start): 
            if j in positions: 
                set_last_block = True
                # if not (last_pos == j-1):
                if last_block_pos > 0:
                    causal_mask[j, :last_block_pos] = 0
                last_pos = j 
            else: 
                if set_last_block: 
                    set_last_block = False
                    last_block_pos = j 
                for k in positions: 
                    if k > j:
                        break
                    causal_mask[j, k] = 0
            

        for j in range(question_start, len(attn_seq)):
            causal_mask[j, :question_start] = 0 
            causal_mask[torch.ones_like(positions, dtype=torch.long)*j, positions] = 1 

        causal_mask *= attn_seq[:, None] * attn_seq[None, :] 
        causal_gist_mask.append(causal_mask)

    return torch.stack(causal_gist_mask, dim=0)[:, None, ...].bool() 

def create_causal_gist_mask_for_generation(attention_mask: torch.LongTensor, 
                                           num_new_tokens: int, 
                                           gist_idx: List[torch.LongTensor]): 

    causal_gist_mask = []
    for i, attn_seq in enumerate(attention_mask): 
        mask = torch.stack([torch.ones_like(attn_seq) for _ in range(num_new_tokens)], dim=0)
        causal_mask = torch.tril(mask, diagonal=mask.shape[1]-mask.shape[0])
        pass
        positions = gist_idx[i] 
        if len(positions) == 0: 
            causal_gist_mask.append(causal_mask * attn_seq[None, :])
            continue 
        question_start = torch.max(positions) + 1

        for j in range(num_new_tokens):
            causal_mask[j, :question_start] = 0 
            causal_mask[torch.ones_like(positions, dtype=torch.long)*j, positions] = 1 

        causal_mask *= attn_seq[None, :] 
        causal_gist_mask.append(causal_mask)
    
    return torch.stack(causal_gist_mask, dim=0)[:, None, ...].bool() 
    

def create_contextless_mask(attention_mask: torch.LongTensor, 
                            gist_idx: List[torch.LongTensor]) -> torch.BoolTensor: 
    causal_contextless_masks = [] 
    for i, attn_seq in enumerate(attention_mask): 
        positions = gist_idx[i] 

        if len(positions) == 0: 
            causal_contextless_masks.append(attn_seq[:, None] * attn_seq[None, :])
            continue 

        questions_start = torch.max(positions) + 1
        mask = torch.tril(attn_seq[:, None] * attn_seq[None, :])
        mask[:, :questions_start] = 0
        causal_contextless_masks.append(mask)
    
    return torch.stack(causal_contextless_masks, dim=0)[:, None, ...].bool() 

def create_contextless_mask_for_generation(attention_mask: torch.LongTensor, 
                                           num_new_tokens: int, 
                                           gist_idx: List[torch.LongTensor]) -> torch.BoolTensor: 
    causal_contextless_masks = [] 
    for i, attn_seq in enumerate(attention_mask): 
        mask = torch.stack([attn_seq for _ in range(num_new_tokens)])
        causal_mask = torch.tril(mask, diagonal=mask.shape[1]-mask.shape[0])
        positions = gist_idx[i] 
        if len(positions) == 0: 
            causal_contextless_masks.append(causal_mask * attn_seq[None, :])
            continue 
        question_start = torch.max(positions) + 1
        causal_mask[:, :question_start] = 0 
        causal_contextless_masks.append(causal_mask)
    
    return torch.stack(causal_contextless_masks, dim=0)[:, None, ...].bool() 
        


def fourier_transform_compress(hidden_states: torch.Tensor, 
                               gist_idx: List[torch.LongTensor], 
                               context_start_list: List[torch.LongTensor]): 
    """
    hidden_states: [batch, seq, hidden]
    gist_idx: List[batch, [num_gist]]
    context_start: List[batch, [num_gist]]
    """

    output_hidden_states = []
    for i, sequence in enumerate(hidden_states): 
        # 1. separate out the non-gist tokens from the gist
        positions = gist_idx[i] 
        if len(positions) == 0: 
            # If no gist tokens, then cannot compress
            output_hidden_states.append(sequence)
            continue
        # positions = positions.sort().values
        context_start = context_start_list[i]
        gist_start = torch.min(positions)
        question_start = torch.max(positions) + 1
        # context_vec = torch.stack([v for j, v in enumerate(sequence) if j not in positions and j < question_start], dim=0)
        context_vec = sequence[context_start:gist_start]

        # 2. perform fourier transform of the non-gist
        orig_dtype = context_vec.dtype
        freqs = torch.fft.rfft(context_vec.float(), dim=0)
        k = int(positions.numel() // 2) + 1

        freqs[k:] = 0
        compressed_context = torch.fft.irfft(freqs, dim=0).to(dtype=orig_dtype)
        num_gist = positions.numel() 


        # 3. interpolate if necessary
        if compressed_context.shape[0] != num_gist: 
            reshape = compressed_context.permute(1, 0)[None, ...]
            reshape = F.interpolate(reshape, size=(num_gist,), mode="linear")
            compressed_context = reshape.permute(0, 2, 1)[0] 

        # for i, vec in enumerate(compressed_context): 
        #     sequence[positions[i]] = vec 
        # assert(context_vec.shape[0] + compressed_context.shape[0]  == sequence.shape[0]) 
        sequence = torch.cat((sequence[:context_start], context_vec, compressed_context, sequence[question_start:]), dim=0) 

        

        output_hidden_states.append(sequence)
    return torch.stack(output_hidden_states, dim=0)
        


def average_compress(hidden_states: torch.Tensor, 
                     gist_idx: List[torch.LongTensor], 
                     context_start_list: List[torch.LongTensor]):

    output_hidden_states = []
    for i, sequence in enumerate(hidden_states): 
        positions = gist_idx[i].sort().values 

        if len(positions)==0: 
            output_hidden_states.append(sequence)
            continue 

        seq_with_compressed_gist = []
        prev_j = context_start_list[i]
        if prev_j > 0: 
            seq_with_compressed_gist.append(sequence[:prev_j])
        for j in positions: 
            if prev_j == j:
                continue
            preceding_vectors = sequence[prev_j:j] 
            seq_with_compressed_gist.append(preceding_vectors)
            gist = preceding_vectors.mean(dim=0, keepdim=True)
            seq_with_compressed_gist.append(gist)
            prev_j = j+1
        seq_with_compressed_gist.append(sequence[prev_j:])
        seq_with_compressed_gist = torch.cat(seq_with_compressed_gist, dim=0)
    
        output_hidden_states.append(seq_with_compressed_gist)
    return torch.stack(output_hidden_states, dim=0)


def fourier_transform_chunk_compress(hidden_states: torch.Tensor, 
                                     gist_idx: List[torch.LongTensor], 
                                     context_start_list: List[torch.LongTensor]): 
    """
    hidden_states: [batch, seq_len, dim]
    gist_idx: List[batch, num_gist]
    context_start_list: List[batch, 1]
    """

    output_sequences = []

    for i, seq in enumerate(hidden_states): 
        context_start = context_start_list[i] 
        positions = gist_idx[i]
        if len(positions) == 0: 
            # TODO:save sequence 
            output_sequences.append(seq)
            continue

        #just a random default initial val
        start_chunk_gist_idx = positions[0]
        start_chunk_context_idx = context_start

        seq_pieces = [seq[:context_start]]

        for j, idx in enumerate(positions): 
            try: 
                next_idx = positions[j+1] 
                end_chunk = next_idx - idx > 1 
            except IndexError: 
                # its the last index 
                next_idx = positions[-1]
                end_chunk = True 
            if end_chunk: 
                # do compression 
                first_gist_idx = start_chunk_gist_idx 
                last_gist_idx = idx 
                first_context_idx = start_chunk_context_idx 
                num_compress_toks = last_gist_idx - first_gist_idx + 1
                
                context_chunk = seq[first_context_idx:first_gist_idx]
                orig_dtype = context_chunk.dtype

                freqs = torch.fft.rfft(context_chunk.float(), dim=0)
                k = int(num_compress_toks // 2) + 1

                freqs[k:] = 0
                compressed_chunk = torch.fft.irfft(freqs, n=len(context_chunk), dim=0).to(dtype=orig_dtype)

                if compressed_chunk.shape[0] != num_compress_toks: 
                    reshape = compressed_chunk.permute(1, 0)[None, ...]
                    reshape = F.interpolate(reshape, size=(num_compress_toks,), mode="linear")
                    compressed_chunk = reshape.permute(0, 2, 1)[0] 

                seq_pieces.append(context_chunk)
                seq_pieces.append(compressed_chunk)

                start_chunk_gist_idx = next_idx
                start_chunk_context_idx = idx + 1
        seq_pieces.append(seq[positions[-1]+1:])
        comp_seq = torch.cat(seq_pieces, dim=0)
        assert comp_seq.shape[0] == seq.shape[0]
        output_sequences.append(comp_seq)
    return torch.stack(output_sequences, dim=0)

def disperse_position_ids(position_ids: torch.LongTensor, 
                          gist_idx: List[torch.LongTensor], 
                          context_start_list: List[torch.LongTensor]): 
    """
    gist_idx: list of tensors representing positions of gist tokens
    context_start_list: list of positions of context start. each list element is a scalar.
    """
    modified_pos_ids = []
    for i in range(len(context_start_list)): 
        if len(position_ids) == len(context_start_list): 
            pos_ids = deepcopy(position_ids[i])
        else: 
            pos_ids = deepcopy(position_ids[0])

        context_start = context_start_list[i] 
        gist_start = gist_idx[i].min() 
        gist_end = gist_idx[i].max() + 1
        

        range_context = gist_start - context_start 
        num_gist = gist_end - gist_start 

        gist_position_ids = torch.linspace(context_start, gist_start, num_gist, device=position_ids.device).round().long() + num_gist - 1
        pos_ids[gist_start:gist_end] = gist_position_ids 

        modified_pos_ids.append(pos_ids) 
    return torch.stack(modified_pos_ids) 







if __name__ == "__main__": 

    input_ids= torch.tensor([[0, 1, 2, 9, 3, 9, 4, 9, 5, 9, 6, 9, 7, 8, 8, 8], 
                             [0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 8, 8], 
                             [0, 0, 0, 9, 9, 0, 0, 0, 9, 9, 0, 9, 0, 0, 0, 0]], dtype=torch.long)
    
    idx = find_idx(input_ids, token_id=9)
    attention_mask = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
                                   [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
                                   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], dtype=torch.long)

    context_start = find_context_start(input_ids, pad_token_id=0)
    pass

    # position_ids = torch.arange(0, input_ids.shape[1])[None, :]

    # mod_pos_id = disperse_position_ids(position_ids, idx, context_start)

    # create_causal_gist_mask(attention_mask, idx)
    # create_causal_gist_mask_for_generation(attention_mask, 3, idx)
    # create_contextless_mask(attention_mask, idx)
    # create_contextless_mask_for_generation(attention_mask, 3, idx)

    hidden_states = torch.randn((2, 16, 128))
    hidden_states[[0, 0, 0, 0, 0, 0], [3, 4, 8, 9, 13, 14]] = -100
    hidden_states[[1, 1, 1], [4, 8, 12]] = -100
    gist_idx = [torch.tensor([3, 4, 8, 9, 13, 14]), torch.tensor([4, 8, 12])] 
    # hidden_states[[0, 0, 0], [4, 8, 12]] = -100
    # gist_idx = [torch.tensor([4, 8, 12]), torch.empty((0))] 
    context_start = [torch.tensor([0]), torch.tensor([0])]
    # hidden_compressed = fourier_transform_compress(hidden_states, gist_idx=gist_idx)
    # hidden_compressed = average_compress(hidden_states, gist_idx)

    outputs = fourier_transform_chunk_compress(hidden_states, gist_idx, context_start)

    pass