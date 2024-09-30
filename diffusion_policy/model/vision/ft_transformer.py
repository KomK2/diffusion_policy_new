import torch
import torch.nn as nn
 
class PositionalEncoding(nn.Module):
    def __init__(self, model_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, model_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, model_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / model_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, model_dim)
        self.register_buffer('pe', pe)
 
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x
 
 
 
class ForceTorqueEncoder(nn.Module):
 
    def __init__(self, ft_seq_len, d_model=126, nhead=6, num_encoder_layers=3):
        super(ForceTorqueEncoder, self).__init__()
 
        self.ft_seq_len = ft_seq_len
        self.embedding_ft = nn.Linear(6, d_model)  # Embedding for each force/torque vector
        self.embedding_pose = nn.Linear(6, d_model)  # Embedding for pose
 
        self.positional_encoding_ft = PositionalEncoding(d_model, max_len=ft_seq_len)
 
        
        self.transformer_encoder_ft = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, batch_first=True ),
            num_layers=num_encoder_layers
        )
 
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
 
 
        self.fc = nn.Linear(d_model, d_model)
 
    
    def forward(self, ft_data, pose):
 
        ft_data = self.embedding_ft(ft_data)
        ft_data = self.positional_encoding_ft(ft_data)
        ft_data = self.transformer_encoder_ft(ft_data)
 
        pose = pose.view(-1, 1, 6)
        pose_embedding = self.embedding_pose(pose)
 
        # print(f"ft_data shape: {ft_data.shape}")  
        # print(f"pose_embedding shape: {pose_embedding.shape}")
 
        
        # Q:pose, K,V : force/torque sequence
        attn_output, _ = self.multihead_attn(pose_embedding, ft_data, ft_data)
 
        
        combined_features = attn_output + pose_embedding
 
        # Mean pooling across the sequence dimension (dim=1) and pass through a fully connected layer
        output = self.fc(combined_features.mean(dim=1))  
 
 
        return output