import torch
import torch.nn as nn
import math

class InputEmbedding(nn.Module):
    def __init__(self, vocal_size, d_model):
        super().__init__()
        self.vocal_size = vocal_size
        self.d_model = d_model
        self.embedding = nn.Embedding(vocal_size, d_model)
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model) 

class PositionalEncoding(nn.Module):
    def __init__(self,d_model, max_len):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len


        self.embedding = torch.zeros(max_len, d_model)
        self.pos = torch.arange(0 , max_len).float().unsqueeze(1)
        self.two_i = torch.arange(0, d_model, 2).float()
                
        self.embedding[:, 0::2] = torch.sin((self.pos)/(10000**(self.two_i / d_model)))
        self.embedding[:, 1::2] = torch.cos((self.pos)/(10000**(self.two_i / d_model)))
    
    def forward (self, x):
        
        postional_encoding = self.embedding[: x.size(1), :].unsqueeze(0)
        return x + postional_encoding
