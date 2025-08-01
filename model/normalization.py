import torch
import torch.nn as nn
import math

class LayerNormalization(nn.Module):
    def __init__(self, d_model, e = 1e-6 ):
        super().__init__()
        self.d_model = d_model
        self.e = e
        self.alpha = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))

    def forward(self,x):
        mean = x.mean(dim = -1, keepdim = True)
        var = x.var(dim = -1, keepdim = True)
        norm_x = (x - mean)/ torch.sqrt(var + self.e)
        return self.alpha * norm_x + self.bias


class ResidualConnection(nn.Module):
    def __init__(self, dropout, d_model):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm_layer = LayerNormalization(d_model)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm_layer(x)))
