import torch
import torch.nn as nn
from attention import MultiHeadAttention
from normalization import LayerNormalization , ResidualConnection
from feed_forward import PositionalFeedForward


class DecoderBlock(nn.Module):
    def __init__(self, dropout, d_model,no_head,dff):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, no_head)
        self.cross_attention = MultiHeadAttention(d_model, no_head)
        self.feed_forward = PositionalFeedForward(d_model, dff, dropout)
        self.residual_connection = nn.ModuleList([ResidualConnection(dropout, d_model) for i in range(3)])
    
    def forward (self, x ,encoder_output, src_marks, target_mask):
        x = self.residual_connection[0](x, lambda x: self.attention(x, x, x, target_mask))
        x = self.residual_connection[1](x, lambda x: self.cross_attention(x, encoder_output, encoder_output, src_marks))
        x = self.residual_connection[2](x, self.feed_forward)
        return x

class Decoder(nn.Module):
    def __init__(self, layer: nn.ModuleList ,d_model):
        super().__init__()
        self.layer = layer
        self.norm = LayerNormalization(d_model)
    def forward(self,x, encoder_output, mask, target_mask):
        for layer in self.layer:
            x = layer(x, encoder_output, mask, target_mask)
        return self.norm(x)

