import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class TransformerEncoder(nn.Module):
    def __init__(self, vocabulary_size, hidden_size, num_heads=4, num_layers=2, dim_feedforward=128):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer_encoder(x)
        x, _ = torch.max(x, dim=1)
        return self.fc(x).unsqueeze(dim=2)
