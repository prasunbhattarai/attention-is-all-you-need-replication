import torch
import torch.nn as nn
import math

def src_mask(src, pad_token_id = 0):
    return (src != pad_token_id).unsqueeze(1).unsqueeze(2)
