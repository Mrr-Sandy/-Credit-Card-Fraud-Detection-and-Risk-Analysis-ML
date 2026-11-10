import numpy as np
import pandas as pd

rng = np.random.default_rng(42)


def make_normal(n):
    # normal customers spend a few hundred rupees, mostly in the daytime
    avg_spending = rng.integers(300, 1500, size=n)
    amount = avg_spending * rng.uniform(0.3, 1.6, size=n)

    data = pd.DataFrame({
        "amount": amount.round(2),
        "hour": rng.integers(8, 23, size=n),
        "txn_count_1h": rng.choice([1, 1, 1, 2, 2, 3], size=n),
        "avg_spending": avg_spending,
        "new_location": rng.choice([0, 1], size=n, p=[0.95, 0.05]),
        "txn_type": rng.choice(["online", "pos", "atm"], size=n, p=[0.4, 0.5, 0.1]),
        "is_fraud": 0,
    })
    return data


def make_fraud(n):
    # fraud: big amount, late night, many transactions quickly, new place
    avg_spending = rng.integers(300, 1500, size=n)
    amount = rng.integers(20000, 50000, size=n)

    data = pd.DataFrame({
        "amount": amount.astype(float),
        "hour": rng.choice([0, 1, 2, 3, 4, 23], size=n),
        "txn_count_1h": rng.integers(4, 9, size=n),
        "avg_spending": avg_spending,
        "new_location": rng.choice([0, 1], size=n, p=[0.1, 0.9]),
        "txn_type": rng.choice(["online", "pos", "atm"], size=n, p=[0.7, 0.1, 0.2]),
        "is_fraud": 1,
    })
    return data


def make_dataset(total=5000, fraud_share=0.05):
    n_fraud = int(total * fraud_share)
    n_normal = total - n_fraud

    df = pd.concat([make_normal(n_normal), make_fraud(n_fraud)])
    # shuffle so fraud rows are not all at the bottom
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df


if __name__ == "__main__":
    df = make_dataset()
    print(df.head())
    print(df["is_fraud"].value_counts())
