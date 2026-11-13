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


def plot_class_distribution(df):
    counts = df["is_fraud"].value_counts().sort_index()
    plt.figure(figsize=(5, 4))
    plt.bar(["Normal", "Fraud"], counts.values, color=["steelblue", "indianred"])
    for i, v in enumerate(counts.values):
        plt.text(i, v, str(v), ha="center", va="bottom")
    plt.title("Class distribution")
    plt.ylabel("Transactions")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/class_distribution.png", dpi=120)
    plt.close()


if __name__ == "__main__":
    df = load_data()
    basic_info(df)
    plot_class_distribution(df)
