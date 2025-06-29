import torch
import torch.nn as nn
from attention import MultiHeadAttention
from normalization import LayerNormalization , ResidualConnection
from feed_forward import PositionalFeedForward


class EncoderBlock(nn.Module):
    def __init__(self, dropout, d_model ,no_head, dff):
        super().__init__()
        self.self_attention  = MultiHeadAttention(d_model, no_head)
        self.feed_forward = PositionalFeedForward(d_model, dff, dropout)
        self.residualconnection = nn.ModuleList([ResidualConnection(dropout, d_model) for i in range (2)])

    def forward(self, x, src_mask ):
        x = self.residualconnection[0](x ,lambda x: self.self_attention(x,x,x, src_mask))
        x = self.residualconnection[1](x, self.feed_forward)
        return x
    
class Encoder(nn.Module):
    def __init__(self, layer,N,d_model):
        super().__init__()
        self.layer = nn.ModuleList([layer for i in range (N)]) 
        self.norm = LayerNormalization(d_model)
    
    def forward(self, x, mask):
        for layer in self.layer:
            x = layer(x, mask)
        return self.norm(x)
        