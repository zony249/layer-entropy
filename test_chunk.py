import asyncio
import os
import subprocess
import sys
import zipfile
from typing import Any, Dict, List, Optional, Self, Tuple, Union

import huggingface_hub
import nltk
import numpy as np

#
import torch
from nltk import Tree
from nltk.chunk.api import ChunkParserI
from nltk.chunk.regexp import *
from nltk.chunk.regexp import RegexpChunkParser, RegexpChunkRule
from nltk.chunk.util import *
from nltk.parse.corenlp import CoreNLPParser, CoreNLPServer
from nltk.tag import pos_tag
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer

nltk.download("punkt")
nltk.download("averaged_perceptron_tagger_eng")
port = 9000
filename = "stanford-corenlp-latest.zip"
local_dir = "parsing/corenlp"

corenlp_file = huggingface_hub.hf_hub_download(
    "stanfordnlp/CoreNLP", filename=filename, local_dir=local_dir
)

with zipfile.ZipFile(os.path.join(local_dir, filename), "r") as zip_ref:
    zip_ref.extractall(local_dir)
    program_name = zip_ref.namelist()[0].split("/")[0]


def add_gist_str_using_linguistic_chunker(
    context: str, nltk_chunker: ChunkParserI, gist_token: str
):
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
    compact_gist = re.sub(f"\s{gist_token}\s?", gist_token, pre_out)
    output = re.sub(f"\s\.", ".", compact_gist)
    return output






def chunk_from_minumum_vector_distance(
    hidden_states: torch.FloatTensor,
    splits: int) -> Dict[int, List[int, ...]]:
    """
    Applies chunking based on minimum vector distance. Greedy algorithm
    hidden_states: [seq_len, dim]
    splits: int
    """
    seq_len, dim = hidden_states.shape

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
    split_size = hidden_states.shape[0] // splits
    merge_mapping_uniform = {}
    for i in range(splits):
        merge_mapping_uniform[i * split_size] = np.arange(
            i * split_size, (i + 1) * split_size
        )
    return merge_mapping



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








if __name__ == "__main__":
    text = "My master's thesis is on the topic of Knowledge Distillation. From the looks of it, it will contain a lot of filler..."

    # tagged = pos_tag(word_tokenize(text))

    # np = "{<DT>?<JJ.*>*<NN.*>+<IN>?<NN.*>+<.*>?}"
    # vp = "{<MD>?<RB.*>*<V.*>+}"
    # rules = [RegexpChunkRule.fromstring(np)]
    # rules += [RegexpChunkRule.fromstring(vp)]
    # chunker = RegexpChunkParser(rules=rules, chunk_label="CH")

    # tree: Tree = chunker.parse(tagged)
    # output = add_gist_str_using_linguistic_chunker(context=text,
    #                                                 nltk_chunker=chunker,
    #                                                 gist_token="<GIST>")
    #

    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B").cuda()
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    toked = tokenizer([text], return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model(**toked, output_hidden_states=True)

    toked_vis = [(t, i) for t, i in zip(tokenizer.tokenize(text), list(range(len(tokenizer.tokenize(text)))))]
    print(toked_vis)
    splits = 10

    # act_chunk_sizes = [2, 4, 7, 5, 2]
    # arr = np.concatenate(
    #     [
    #         10 * np.random.rand(1, 100) + 0.5 * np.random.rand(i, 100)
    #         for i in act_chunk_sizes
    #     ],
    #     axis=0,
    # )
    hidden_states = outputs.hidden_states[-1][0]  # expected to be [s, dims]


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

    # print(merge_mapping)
    # compute_chunk_diff(hidden_states, merge_mapping)
    # print("")
    # print(merge_mapping_uniform)
    # compute_chunk_diff(hidden_states, merge_mapping_uniform)
    # pass
