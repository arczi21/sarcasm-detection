import pandas as pd
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sarcasm_detection.data import SarcasticDataLoader, BatchLoader
from sarcasm_detection.preprocessing import SarcasticEncoder
from sarcasm_detection.models import LSTM
from sarcasm_detection.training.lstm.config import CONFIG

def train(n_epochs=2, hidden_size=512, n_layers=3, batch_size=256, max_tokens=3000, lr=0.0001, log_every=100,
          wandb_log=True, df_train=None, df_test=None):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if df_test is None or df_test is None:
        df = pd.read_json('./data/Sarcasm_Headlines_Dataset.json', lines=True)
        df_train, df_test = train_test_split(df, train_size=0.95)
        df_train = df_train.reset_index()
        df_test = df_test.reset_index()

    data_loader_train = SarcasticDataLoader(df_train)
    data_loader_test = SarcasticDataLoader(df_test)

    headlines = data_loader_train.get_all_headlines()
    encoder = SarcasticEncoder(headlines, max_tokens=max_tokens)
    if max_tokens is None:
        max_tokens = len(encoder)

    batch_loader_train = BatchLoader(data_loader_train, encoder, batch_size)
    batch_loader_test = BatchLoader(data_loader_test, encoder)

    config = {
        'n_epochs': n_epochs,
        'hidden_size': hidden_size,
        'n_layers': n_layers,
        'batch_size': batch_size,
        'max_tokens': max_tokens,
        'lr': lr,
        'log_every': log_every,
    }

    lstm = LSTM(max_tokens, hidden_size, n_layers).to(device)

    optimizer = optim.Adam(lstm.parameters(), lr=lr)

    step = 0

    if wandb_log:
        wandb.init(project="NLP-sarcasm-lstm", config=config)

    def calculate_output(batch):
        batch = batch.to(device)
        hidden = (torch.zeros(n_layers, len(batch), hidden_size).to(device),
                  torch.zeros(n_layers, len(batch), hidden_size).to(device))

        output, hidden = lstm(batch, hidden)
        return output, hidden

    def test_model():
        batch, labels = batch_loader_test.get_all_data()
        output, _ = calculate_output(batch)
        labels = labels.to(device)
        accuracy = torch.sum((F.sigmoid(output[:, -1, 0]) > 0.5) == labels) / len(batch)
        loss = torch.mean(labels * torch.log(1 + torch.exp(-output[:, -1, 0])) + (1 - labels) * (
                output[:, -1, 0] + torch.log(1 + torch.exp(-output[:, -1, 0]))))
        return accuracy.item(), loss.item()

    for epoch in range(n_epochs):

        for batch, labels in batch_loader_train.iterate_batches():
            output, _ = calculate_output(batch)
            labels = labels.to(device)

            loss = labels * torch.log(1 + torch.exp(-output[:, -1, 0])) + (1 - labels) * (
                        output[:, -1, 0] + torch.log(1 + torch.exp(-output[:, -1, 0])))
            loss = loss.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if wandb_log:
                wandb.log({"loss": loss.item(), "step": step})
                wandb.log({"epoch": epoch + 1, "step": step})
                if step % log_every == 0:
                    accuracy, test_loss = test_model()
                    wandb.log({"test_accuracy": accuracy, "step": step})
                    wandb.log({"test_loss": test_loss, "step": step})
            step += 1

    accuracy, test_loss = test_model()
    if wandb_log:
        wandb.finish()
    return lstm, accuracy, test_loss


if __name__ == '__main__':
    train(n_epochs=5, hidden_size=CONFIG['hidden_size'], n_layers=CONFIG['n_layers'], batch_size=CONFIG['batch_size'],
          max_tokens=CONFIG['max_tokens'], lr=CONFIG['lr'], log_every=10, wandb_log=True)
