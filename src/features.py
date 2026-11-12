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


if __name__ == "__main__":
    data = pd.read_csv("data/transactions.csv")
    data = add_amount_ratio(data)
    data = add_night_flag(data)
    print(data.groupby("is_fraud")[["amount_ratio", "is_night"]].mean())
