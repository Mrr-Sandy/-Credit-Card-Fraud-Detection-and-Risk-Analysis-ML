import pandas as pd


def load_train(path="data/train.csv"):
    return pd.read_csv(path)


if __name__ == "__main__":
    train = load_train()
    print("train shape:", train.shape)
    print(train["is_fraud"].value_counts().to_dict())
