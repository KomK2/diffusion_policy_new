import torch
import torch.nn as nn

from diffusion_policy.model.force_torque.positional_encoding import PositionalEncoding
 
 
class ForceTorqueEncoder(nn.Module):
 
    def __init__(self, ft_seq_len, d_model=256, nhead=8, num_encoder_layers=3):
        super(ForceTorqueEncoder, self).__init__()
 
        self.ft_seq_len = ft_seq_len

        # input embedding layer
        self.embedding_ft = nn.Linear(6, d_model)   # batch_size, seq_len, 6 -> batch_size, seq_len, d_model

        self.positional_encoding_ft = PositionalEncoding(d_model, max_len=ft_seq_len, learn_embedding=True) # batch_size, seq_len, d_model -> batch_size, seq_len, d_model

        self.transformer_encoder_ft = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, batch_first=True ),
            num_layers=num_encoder_layers
        ) 

        self.layer_norm = nn.LayerNorm(d_model) 

        self.fc = nn.Linear(d_model * ft_seq_len, d_model) 
 
    
    def forward(self, ft_data):
        
        ft_data = self.embedding_ft(ft_data)
        ft_data = self.positional_encoding_ft(ft_data)
        ft_data = self.transformer_encoder_ft(ft_data)

        ft_data = self.layer_norm(ft_data)
        
        output = ft_data.view(ft_data.size(0), -1)  
        output = self.fc(output) 

        return output