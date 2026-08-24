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


    torch.autograd.set_detect_anomaly(True)

    # token_annotation = torch.tensor([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 2, 2, 2, 2, 2, 2])
    token_annotation = torch.tensor([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1])

    token_annotation = []
    for i in range(50): 
        token_annotation.append(torch.tensor([0, 0, 0, 0, 0])) 
        token_annotation.append(torch.tensor([1]))
    token_annotation.append(torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]))
    token_annotation = torch.cat(token_annotation, dim=0) 

    seq_len = token_annotation.shape[0]
    alpha = 0.1
    P = F.one_hot(token_annotation, num_classes=2).float()
    # P = token_annotation.float()
    P.requires_grad = True
    P_same = torch.zeros((seq_len, seq_len))

    
    tbar = tqdm(range(1000), desc="Running") 
    for _ in tbar:

        P_dtype = P.dtype
        P = P.float()
        log_P = torch.log(P[:, 0].clamp(min=1e-8)) 
        clog_P = torch.cumsum(log_P, dim=0) 
        clog_P_ij = clog_P[:, None] - clog_P[None, :] + log_P[None, :] - log_P[:, None] 
        causal_mask = torch.tril(torch.ones(clog_P_ij.shape, dtype=bool, device=P.device))
        log_P_same = torch.where(causal_mask, clog_P_ij, torch.finfo(P.dtype).min * 1e-2) 
        P_same = torch.exp(log_P_same)
        P_end = torch.tril(P[:, 1][:, None] * P[:, 1][None, :])
        M = P_same  + (1-P_same) * P_end
        M = M.to(P_dtype)




    M.sum().backward()


    print(M.min(), M.max())
    
    plt.imshow(M.detach().cpu().numpy()) 
    plt.savefig("runs/softmask.png", dpi=300)