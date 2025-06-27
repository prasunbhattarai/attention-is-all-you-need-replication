import torch
import torch.nn as nn
import math

class ScaledDotProduct(nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax(dim =-1)
    
    def forward(self, q, k, v):
        k_transpose = torch.transpose(k,-2,-1)
        numerator = q @ k_transpose  #torch.matmul(q, k_transpose)
        score = numerator/ math.sqrt(k.size(-1))
        attention_weight = self.softmax(score)
        output = attention_weight @ v #torch.matmul( attention_weight,v)

        return output

class MultiHeadAttention(nn.Module):
    def __init__(self,d_model, no_head):
        super().__init__()
        self.no_head = no_head 
        self.d_model = d_model
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.d_k = d_model // no_head
        self.attention = ScaledDotProduct()
    def forward(self, q, k, v):
        batch_size = q.size(0)
        seq_len = q.size(1)

        q_prime = self.w_q(q)
        k_prime = self.w_k(k)
        v_prime = self.w_v(v)
        
        
        q_head = q_prime.view(batch_size, seq_len, self.no_head, self.d_k).transpose(1,2)
        k_head = k_prime.view(batch_size, seq_len, self.no_head, self.d_k).transpose(1,2)
        v_head = v_prime.view(batch_size, seq_len, self.no_head, self.d_k).transpose(1,2)

        attention_output = self.attention(q_head,k_head,v_head).transpose(1,2)
        attention_output = attention_output.contiguous().view(batch_size, seq_len, self.d_model)

        return self.w_o(attention_output)


        