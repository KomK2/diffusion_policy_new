from typing import Dict, Tuple, Union
import copy
import torch
import torch.nn as nn
import torchvision

from module_attr_mixin import ModuleAttrMixin

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


class ForceTorqueEncoder(ModuleAttrMixin):
    def __init__(self,
            ft_data_shape: Tuple[int,int],
            is_transformer: bool=True,
            d_model: int=128,
            nhead: int=8,
            num_encoder_layers: int=3,
        ):
        super().__init__()
        self.ft_data_shape = ft_data_shape
        self.is_transformer = is_transformer

        # input embedding layer
        self.embedding_ft = nn.Linear(6, d_model)
        # positional encoding
        self.positional_encoding_ft = PositionalEncoding(d_model, max_len=ft_data_shape[0], learn_embedding=False)
        # transformer encoder
        self.transformer_encoder_ft = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model, nhead, batch_first=True),
                num_layers=num_encoder_layers
            )
        
        self.fc = nn.Linear(d_model, 256)
        
        
    def forward(self, ft_data):
        print("ft_data shape is ", ft_data.shape)   
        ft_data = self.embedding_ft(ft_data)
        print("ft_data shape after input embedding is ", ft_data.shape)
        ft_data = self.positional_encoding_ft(ft_data)
        print("ft_data shape after positional encoding is ", ft_data.shape)
        ft_data = self.transformer_encoder_ft(ft_data)
        print("ft_data shape after transformer encoder is ", ft_data.shape)

        ft_data = self.fc(ft_data)

        return ft_data

if __name__ == "__main__":
    ft_data_shape = (10,6)
    encoder = ForceTorqueEncoder(ft_data_shape)

    ft_data = torch.randn((32,10,6))

    output = encoder(ft_data)