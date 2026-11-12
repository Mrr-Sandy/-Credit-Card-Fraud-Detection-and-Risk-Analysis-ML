import numpy as np
import pandas as pd

from features import build_features


def split_data(df, test_size=0.2, seed=42):
    # split fraud and normal rows separately so both sets keep the same fraud share
    rng = np.random.default_rng(seed)
    train_parts, test_parts = [], []

    for label in [0, 1]:
        part = df[df["is_fraud"] == label]
        order = rng.permutation(len(part))
        n_test = int(len(part) * test_size)
        test_parts.append(part.iloc[order[:n_test]])
        train_parts.append(part.iloc[order[n_test:]])

    train = pd.concat(train_parts).sample(frac=1, random_state=seed)
    test = pd.concat(test_parts).sample(frac=1, random_state=seed)
    return train.reset_index(drop=True), test.reset_index(drop=True)


SCALE_COLS = ["amount", "hour", "txn_count_1h", "avg_spending", "amount_ratio"]


def scale_data(train, test):
    # mean and std come from train only, otherwise test data leaks into training
    train, test = train.copy(), test.copy()
    mean = train[SCALE_COLS].mean()
    std = train[SCALE_COLS].std()
    train[SCALE_COLS] = (train[SCALE_COLS] - mean) / std
    test[SCALE_COLS] = (test[SCALE_COLS] - mean) / std
    return train, test, mean, std


if __name__ == "__main__":
    data = build_features(pd.read_csv("data/transactions.csv"))
    train, test = split_data(data)
    train, test, mean, std = scale_data(train, test)
    print("train:", train.shape, "fraud:", train["is_fraud"].sum())
    print("test: ", test.shape, "fraud:", test["is_fraud"].sum())
    print(train[SCALE_COLS].mean().round(3).to_dict())
