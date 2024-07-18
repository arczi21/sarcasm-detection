import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x, hidden):
        x, hidden = self.lstm(x, hidden)
        x = self.fc(x)
        return x, hidden