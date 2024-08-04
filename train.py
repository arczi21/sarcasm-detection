import wandb
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sarcasm_detection.data import DataLoader, BatchLoader
from sarcasm_detection.preprocessing import Encoder


def get_metrics(model, batch, labels, device):
    batch = batch.to(device)
    labels = labels.to(device)

    output = model(batch)[:, -1, 0]

    accuracy = torch.sum((F.sigmoid(output) > 0.5) == labels) / len(batch)
    loss = torch.mean(labels * torch.log(1 + torch.exp(-output)) + (1 - labels) * (
            output + torch.log(1 + torch.exp(-output))))

    return loss, accuracy


def train(data_loader_train, data_loader_test, model, encoder, n_epochs=5, batch_size=256, lr=0.0001,
          wandb_log=False, log_every=100, **kwargs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # print(device)

    # data_loader_train = DataLoader(df_train)
    # data_loader_test = DataLoader(df_valid)

    # headlines = data_loader_train.get_all_headlines()
    # encoder = Encoder(headlines, max_tokens=max_tokens)

    batch_loader_train = BatchLoader(data_loader_train, encoder, batch_size)
    batch_loader_test = BatchLoader(data_loader_test, encoder)

    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    step = 0
    test_loss = []
    test_accuracy = []

    if wandb_log:
        wandb.init(project="NLP-sarcasm")

    for epoch in range(n_epochs):

        for batch, labels in batch_loader_train.iterate_batches():

            loss, _ = get_metrics(model, batch, labels, device)

            loss = loss.mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()


            if wandb_log:
                wandb.log({"loss": loss.item(), "step": step})
                wandb.log({"epoch": epoch + 1, "step": step})
                if step % log_every == 0:
                    batch, labels = batch_loader_test.get_all_data()

                    loss, accuracy = get_metrics(model, batch, labels, device)

                    wandb.log({"test_accuracy": accuracy.item(), "step": step})
                    wandb.log({"test_loss": loss.item(), "step": step})

            step += 1

        batch, labels = batch_loader_test.get_all_data()
        with torch.no_grad():
            loss, accuracy = get_metrics(model, batch, labels, device)
        test_loss.append(loss.item())
        test_accuracy.append(accuracy.item())

    if wandb_log:
        wandb.finish()
    return model, {'loss': test_loss, 'accuracy': test_accuracy}