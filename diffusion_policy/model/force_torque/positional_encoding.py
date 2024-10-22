import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, model_dim, max_len=5000, learn_embedding = False, dropout: float = 0.1,):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, model_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, model_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / model_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, model_dim)
        self.register_buffer('pe', pe)

        self.learn_embedding = learn_embedding
        
        self.embedding = nn.Parameter(torch.randn(1, model_dim), requires_grad=True)
        
 
    def forward(self, x):
        if self.learn_embedding:
            x = x + self.embedding
        x = x + self.pe[:, :x.size(1), :]
        return x
