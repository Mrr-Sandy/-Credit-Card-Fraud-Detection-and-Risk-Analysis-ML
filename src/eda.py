import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

FIG_DIR = "reports/figures"


def load_data(path="data/transactions.csv"):
    return pd.read_csv(path)


def basic_info(df):
    print("shape:", df.shape)
    print()
    print(df.dtypes)
    print()
    print(df.describe().round(2))


if __name__ == "__main__":
    df = load_data()
    basic_info(df)
