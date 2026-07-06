import os 
import sys 
from copy import deepcopy 
from typing import List, Dict, Tuple, Optional, Any
import matplotlib.pyplot as plt
from tqdm import tqdm


import numpy as np
import torch 
from torch.nn import functional as F 



if __name__ == "__main__": 

    token_annotation = torch.tensor([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 2, 2, 2, 2, 2, 2])
    # token_annotation = []
    # for i in range(50): 
    #     token_annotation.append(torch.tensor([0, 0, 0, 0, 0])) 
    #     token_annotation.append(torch.tensor([1]))
    # token_annotation.append(torch.tensor([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]))
    # token_annotation = torch.cat(token_annotation, dim=0) 

    seq_len = token_annotation.shape[0]
    alpha = 0.05
    z = F.one_hot(token_annotation, num_classes=3).float()
    P = z * (1-alpha) + alpha / 3

    P_same = torch.zeros((seq_len, seq_len))

    tbar = tqdm(range(4), desc="Running") 
    for _ in tbar:
        for j in range(seq_len):
            inside = 1
            for i in range(j+1): 
                val = torch.prod(P[i:j, 0]) 
                if i == j: 
                    val *= (P[j, 0] + P[j, 1])
                P_same[j, i] = val

        M = torch.zeros((seq_len, seq_len)) 
        for j in range(seq_len): 
            for i in range(j+1): 
                M[j, i] = P_same[j, i] + (1 - P_same[j, i]) * (P[i, 1] * P[j, 2] + P[i, 2] * P[j, 2])
    
    tbar = tqdm(range(1000), desc="Running") 
    for _ in tbar:
        log_P = torch.log(P * (1-1e-8) + 1e-8) 
        clog_P_j = torch.cumsum(log_P[:, 0], dim=0) - log_P[:, 0] + (torch.log(P[:, 0] + P[:, 1]))
        clog_P_i = torch.cumsum(log_P[:, 0], dim=0) - log_P[:, 0]
        clog_P_ji = clog_P_j[:, None] - clog_P_i[None, :] 
        P_same = torch.tril(torch.exp(clog_P_ji))
        M = torch.zeros((seq_len, seq_len))
        Pji_bound_downstream = torch.tril(P[:, 2][:, None] * P[:, 1][None, :])
        Pji_downstream = torch.tril(P[:, 2][:, None] * P[:, 2][None, :])
        M = P_same + (1-P_same) * (Pji_bound_downstream + Pji_downstream)



    
    plt.imshow(M.numpy()) 
    plt.savefig("runs/softmask.png", dpi=300)