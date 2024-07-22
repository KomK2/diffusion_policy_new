import torch
import torch.nn as nn


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, max_len=5000, model_dim=64):
        super(LearnablePositionalEncoding, self).__init__()
        self.model_dim = model_dim
        self.pe = nn.Embedding(max_len, model_dim)
        nn.init.normal_(self.pe.weight, mean=0, std=0.02)  # Initialize with values from a normal distribution

    def forward(self, x):
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(x.size(0), -1)
        pos_embeddings = self.pe(positions)
        x = x + pos_embeddings
        return x

class LPositionalEmbedding(nn.Module):
    def __init__(self, model_dim, max_len=5000):
        super(LPositionalEmbedding, self).__init__()
        self.model_dim = model_dim
        self.pe = nn.Parameter(torch.zeros(1, max_len, model_dim))
        nn.init.normal_(self.pe, mean=0, std=0.02)  # Initialize with values from a normal distribution

    def forward(self, x):
        seq_len = x.size(1) 
        x = x + self.pe[:, :seq_len, :]
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, model_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, model_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, model_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / model_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

class ForceTorqueEncoder(nn.Module):
    def __init__(self, input_dim=6, model_dim=64, num_heads=8, num_layers=2, dropout=0.1, max_len=5000):
        super(ForceTorqueEncoder, self).__init__()
        self.input_linear = nn.Linear(input_dim, model_dim)
        self.get_positional_encoding("lpositional", model_dim, max_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_linear = nn.Linear(model_dim, model_dim)

    def forward(self, x):
        x = self.input_linear(x)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)
        x = self.output_linear(x)
        return x
    
    def get_positional_encoding(self, x, model_dim, max_len ):
        if x == "learnable":
            self.positional_encoding =  LearnablePositionalEncoding(max_len, model_dim)
        elif x == "lpositional":
            self.positional_encoding =  LPositionalEmbedding(model_dim, max_len)
        elif x == "positional":
            self.positional_encoding =  PositionalEncoding(model_dim, max_len)
        else:
            raise ValueError(f"Unknown positional encoding type: {x}")