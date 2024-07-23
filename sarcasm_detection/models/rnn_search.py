import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class RNNSearchDecoder(nn.Module):
    def __init__(self, hidden_size):
        super(RNNSearchDecoder, self).__init__()

        # alignment model
        self.W = nn.Linear(2 * hidden_size, 1)
        self.alignment_model = nn.Linear(hidden_size, 1)
        self.U = nn.Linear(2 * hidden_size, 1)

    def forward(self, h):
        e = self.W(h)
        alignment = F.softmax(e, dim=1)
        weighted_average = torch.sum(h * alignment, dim=1)
        return self.U(weighted_average)
