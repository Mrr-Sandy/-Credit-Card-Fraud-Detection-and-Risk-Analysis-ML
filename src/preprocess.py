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


if __name__ == "__main__":
    data = build_features(pd.read_csv("data/transactions.csv"))
    train, test = split_data(data)
    print("train:", train.shape, "fraud:", train["is_fraud"].sum())
    print("test: ", test.shape, "fraud:", test["is_fraud"].sum())
