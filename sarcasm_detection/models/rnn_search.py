import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class RNNSearch(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers):
        super(RNNSearch, self).__init__()

        self.encoder = nn.RNN(input_size, hidden_size, n_layers, bidirectional=True, batch_first=True)
        self.W = nn.Linear(2 * hidden_size, 1)
        self.alignment_model = nn.Linear(hidden_size, 1)
        self.U = nn.Linear(2 * hidden_size, 1)

    def forward(self, x):
        embeddings, _ = self.encoder(x)
        e = self.W(embeddings)
        alignment = F.softmax(e, dim=1)
        x = self.U(alignment * embeddings)
        return x
