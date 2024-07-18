import torch
import torch.nn as nn
import torch.nn.functional as F
import nltk
from nltk.tokenize import TreebankWordTokenizer, TreebankWordDetokenizer
from nltk.probability import FreqDist
from sarcasm_detection.data.data_loader import SarcasticDataLoader
from torch.nn.utils.rnn import pad_sequence


class SarcasticEncoder:
    def __init__(self, dataset, max_tokens=None):
        nltk.download('punkt', quiet=True)

        self.tokenizer = TreebankWordTokenizer()

        tokenized_dataset = [self.tokenizer.tokenize(data) for data in dataset]
        tokens = [token for data in tokenized_dataset for token in data]

        self.freq_dist = FreqDist(tokens)
        if max_tokens is not None:
            most_common_tokens = self.freq_dist.most_common(max_tokens - 2)
            vocabulary = [token for token, freq in most_common_tokens]
        else:
            vocabulary = list(self.freq_dist.keys())
        self.token_dict = {token: idx for idx, token in enumerate(vocabulary, start=1)}
        self.token_dict["<UNK>"] = len(self.token_dict) + 1

    def __len__(self):
        return len(self.token_dict) + 1

    def encode_text(self, text):
        tokens = self.tokenizer.tokenize(text.lower())
        return [self.token_dict.get(token, self.token_dict["<UNK>"]) for token in tokens]

    def encode_text_tensor(self, text):
        tokens = self.encode_text(text)
        one_hot = F.one_hot(torch.tensor(tokens), len(self))
        return one_hot.to(torch.float32)

    def encode_text_tensor_batch(self, text_list):
        batch = []
        for text in text_list:
            batch.append(torch.tensor(self.encode_text(text)))
        batch = pad_sequence(batch, batch_first=True, padding_value=0)
        batch = F.one_hot(batch)
        return batch.to(torch.float32)



