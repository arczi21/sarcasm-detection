import itertools
import pandas as pd
from sklearn.model_selection import train_test_split
from sarcasm_detection.training.lstm import train

param_grid = {
    'hidden_size': [64, 512],
    'n_layers': [3],
    'batch_size': [32, 256],
    'max_tokens': [1000, 3000],
    'lr': [0.001, 0.0001]
}

best_params = None
best_loss = float('inf')

param_combinations = list(itertools.product(
    param_grid['hidden_size'],
    param_grid['n_layers'],
    param_grid['batch_size'],
    param_grid['max_tokens'],
    param_grid['lr']
))

if __name__ == '__main__':
    df = pd.read_json('./data/Sarcasm_Headlines_Dataset.json', lines=True)
    df_train, df_test = train_test_split(df, train_size=0.95)
    df_train = df_train.reset_index()
    df_test = df_test.reset_index()
    for params in param_combinations:
        hidden_size, n_layers, batch_size, max_tokens, lr = params
        _, accuracy, test_loss = train(n_epochs=2, hidden_size=hidden_size, n_layers=n_layers, batch_size=batch_size,
                                        max_tokens=max_tokens, lr=lr, wandb_log=False, df_train=df_train,
                                        df_test=df_test)
        print('params: %s | accuracy: %s | loss: %s' % (params, accuracy, test_loss))
        if test_loss < best_loss:
            best_loss = test_loss
            best_params = params

    print(best_params, best_loss)