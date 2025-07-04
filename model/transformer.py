import torch.nn as nn
from embedding  import InputEmbedding, PositionalEncoding
from feed_forward import PositionalFeedForward
from attention import MultiHeadAttention
from encoder import Encoder, EncoderBlock
from decoder import Decoder, DecoderBlock
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
    
    def forward(self,src,tgt, src_mask, tgt_mask):
        enc_out = self.encode(src , src_mask)
        dec_out = self.decode(tgt, enc_out, src_mask, tgt_mask)
        return self.linear(dec_out)


def transformer_block(d_model, no_head,dff, dropout, vocab_size, max_len, N):


    encoder_block = [EncoderBlock(dropout, d_model, no_head, dff) for i in range(N)]
    encoder = Encoder(encoder_block , d_model)

    decoder_block = [DecoderBlock(dropout, d_model, no_head, dff) for i in range(N)]
    decoder = Decoder(decoder_block, d_model)

    model = Transformer(vocab_size, d_model, max_len,encoder, decoder)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model
