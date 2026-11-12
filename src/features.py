import pandas as pd


def add_amount_ratio(df):
    # how big is this transaction compared to what the card normally spends
    df = df.copy()
    df["amount_ratio"] = (df["amount"] / df["avg_spending"]).round(3)
    return df


def add_night_flag(df):
    # transactions between 11 PM and 5 AM are treated as unusual hours
    df = df.copy()
    df["is_night"] = ((df["hour"] >= 23) | (df["hour"] <= 4)).astype(int)
    return df


def encode_txn_type(df):
    # models need numbers, so online/pos/atm becomes three 0/1 columns
    return pd.get_dummies(df, columns=["txn_type"], prefix="type", dtype=int)


def build_features(df):
    df = add_amount_ratio(df)
    df = add_night_flag(df)
    df = encode_txn_type(df)
    return df


if __name__ == "__main__":
    data = build_features(pd.read_csv("data/transactions.csv"))
    print(data.columns.tolist())
    print(data.head(3))
