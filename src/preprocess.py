import pandas as pd


def load_data(path="data/transactions.csv"):
    return pd.read_csv(path)


def add_features(df):
    df = df.copy()
    # how big is this transaction compared to what the card usually spends
    df["amount_ratio"] = (df["amount"] / df["avg_spending"]).round(3)
    # late night (11pm to 5am) flag
    df["is_night"] = df["hour"].isin([23, 0, 1, 2, 3, 4]).astype(int)
    return df


def encode_txn_type(df):
    # one-hot encode online / pos / atm
    return pd.get_dummies(df, columns=["txn_type"], dtype=int)


if __name__ == "__main__":
    df = encode_txn_type(add_features(load_data()))
    print(df.head())
    print(df.columns.tolist())
