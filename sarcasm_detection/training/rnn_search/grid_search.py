from itertools import product
import pandas as pd
from sklearn.model_selection import train_test_split
from sarcasm_detection.training.rnn_search import train

param_grid = {
    'hidden_size': [256, 512, 1024],
    'n_layers': [1, 2],
    'batch_size': [32],
    'max_tokens': [5000],
    'lr': [0.001, 0.0001]
}

best_params = None
best_loss = float('inf')

keys = param_grid.keys()
values = param_grid.values()

combinations = [dict(zip(keys, combination)) for combination in product(*values)]

if __name__ == '__main__':
    df_train = pd.read_csv('data/processed/train.csv')
    df_valid = pd.read_csv('data/processed/validation.csv')

    for params in combinations:
        _, accuracy, test_loss = train(df_train, df_valid, n_epochs=4, wandb_log=False, **params)
        print('params: %s | accuracy: %s | loss: %s' % (params, accuracy, test_loss))
        if test_loss < best_loss:
            best_loss = test_loss
            best_params = params

    print(best_params, best_loss)