import torch
import torch.nn as nn
import math



class FinalLayer(nn.Module):
    def __init__(self, d_model , vocab_size):
        super().__init__()
        self.linear = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        return torch.log_softmax(self.linear(x), dim = -1)

    