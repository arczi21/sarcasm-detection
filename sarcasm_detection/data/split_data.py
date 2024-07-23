import pandas as pd
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    df = pd.read_json('data/raw/Sarcasm_Headlines_Dataset.json', lines=True)
    df_train, df_valid_test = train_test_split(df, train_size=0.9)
    df_valid, df_test = train_test_split(df_valid_test, train_size=0.5)

    df_train.reset_index(drop=True, inplace=True)
    df_valid.reset_index(drop=True, inplace=True)
    df_test.reset_index(drop=True, inplace=True)

    df_train.to_csv('data/processed/train.csv', index=False)
    df_valid.to_csv('data/processed/validation.csv', index=False)
    df_test.to_csv('data/processed/test.csv', index=False)
