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


if __name__ == "__main__":
    normal = make_normal(5)
    print(normal)
