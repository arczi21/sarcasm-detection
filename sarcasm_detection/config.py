CONFIG = {
    'lstm': {
        'hidden_size': 512,
        'n_layers': 1,
        'batch_size': 32,
        'max_tokens': 5000,
        'lr': 0.001
    },
    'rnn_search': {
        'hidden_size': 512,
        'n_layers': 3,
        'batch_size': 64,
        'max_tokens': 5000,
        'lr': 0.0001
    },
    'transformer': {
        'hidden_size': 256,
        'dim_feedforward': 256,
        'num_heads': 2,
        'num_layers': 2,
        'batch_size': 32,
        'max_tokens': 5000,
        'lr': 0.0001
    },
    'transformer_decoder': {
        'hidden_size': 256,
        'dim_feedforward': 256,
        'num_heads': 4,
        'num_layers': 2,
        'batch_size': 32,
        'max_tokens': 5000,
        'lr': 0.0001
    }
}