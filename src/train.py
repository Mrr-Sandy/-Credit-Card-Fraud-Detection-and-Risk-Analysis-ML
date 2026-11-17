import pandas as pd


def load_train(path="data/train.csv"):
    return pd.read_csv(path)


def split_xy(df):
    # is_fraud is the target, everything else is a feature
    return df.drop(columns="is_fraud"), df["is_fraud"]


if __name__ == "__main__":
    X_train, y_train = split_xy(load_train())
    print("features:", X_train.columns.tolist())
    print("train shape:", X_train.shape, "fraud share:", round(y_train.mean(), 2))
