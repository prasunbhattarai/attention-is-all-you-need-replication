import torch
import torch.nn as nn
import math
from embedding  import InputEmbedding, PositionalEncoding
from encoder import Encoder
from decoder import Decoder
from final_layer import FinalLayer

class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, max_len, encoder: Encoder, decoder: Decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = InputEmbedding(vocab_size, d_model)
        self.tgt_embed = InputEmbedding(vocab_size, d_model)
        
        self.src_pos = PositionalEncoding(d_model, max_len)
        self.tgt_pos = PositionalEncoding(d_model, max_len)
        
        self.linear_layer = FinalLayer(d_model, vocab_size)

    def encode(self, x, scr_mask):
        out = self.src_embed(x)
        out = self.src_pos(out)
        return self.encoder(out, scr_mask)
    
    def decode(self,x, encoder_output, mask, target_mask):
        out = self.tgt_embed(x)
        out = self.tgt_pos(out)
        return self.decoder(out, encoder_output, mask, target_mask)
    
    def linear(self, x):
        return self.linear_layer(x)
