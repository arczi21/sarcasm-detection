import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_heads, num_layers, dim_feedforward, output_size=1):
        super(TransformerDecoder, self).__init__()

        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        decoder_layer = nn.TransformerDecoderLayer(hidden_size, num_heads, dim_feedforward, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        # self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, tgt_key_padding_mask=None):
        x = self.embedding(x) * math.sqrt(self.hidden_size)

        n = x.size(1)
        tgt_mask = torch.zeros(n, n).to(x.device)
        upper_triangular_mask = torch.triu(torch.ones((n, n)), diagonal=1)
        tgt_mask[upper_triangular_mask == 1] = float('-inf')

        x = self.transformer_decoder(x, x)
        # x = self.fc(x)
        return x
