import torch
import torch.nn as nn
import math

def src_mask(src, pad_token_id = 0):
    return (src != pad_token_id).unsqueeze(1).unsqueeze(2)

def tgt_mask(tgt, pad_token_id = 0):
    B, T = tgt.shape
    mask = (tgt != pad_token_id).unsqueeze(1).unsqueeze(2)
    future_mask = torch.tril(torch.ones(T, T)).bool()
    future_mask = future_mask.unsqueeze(1).unsqueeze(2)
    return mask & future_mask
