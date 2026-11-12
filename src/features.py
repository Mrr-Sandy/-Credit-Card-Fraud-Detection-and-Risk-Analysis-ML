import pandas as pd


def add_amount_ratio(df):
    # how big is this transaction compared to what the card normally spends
    df = df.copy()
    df["amount_ratio"] = (df["amount"] / df["avg_spending"]).round(3)
    return df


if __name__ == "__main__":
    data = pd.read_csv("data/transactions.csv")
    data = add_amount_ratio(data)
    print(data.groupby("is_fraud")["amount_ratio"].mean())
