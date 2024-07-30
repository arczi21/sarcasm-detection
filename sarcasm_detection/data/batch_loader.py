import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sarcasm_detection.data.data_loader import DataLoader
from sarcasm_detection.preprocessing.encoder import Encoder
from torch.nn.utils.rnn import pad_sequence


class BatchLoader:
    def __init__(self, data_loader: DataLoader, encoder: Encoder, batch_size=32):
        self.data_loader = data_loader
        self.encoder = encoder
        self.batch_size = batch_size
        self.idx = 0

    def n_batches(self):
        return math.ceil(len(self.data_loader) / self.batch_size)

    def get_batch(self, begin, end):
        text_batch, label_batch = self.data_loader.get_data(begin, end)
        text_batch_tensor = self.encoder.encode_text_list(text_batch)
        return text_batch_tensor, torch.tensor(label_batch)

    def get_all_data(self):
        return self.get_batch(0, len(self.data_loader))

    def iterate_batches(self):
        for i in range(self.n_batches()):
            yield self.get_batch(i*self.batch_size, min((i+1)*self.batch_size, len(self.data_loader)))
