import torch
import torch.nn as nn
import math


class PositionalFeedForward(nn.Module):
    def __init__(self, d_model, dff, dropout = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dff)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(dff, d_model)
    def forward(self,x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x