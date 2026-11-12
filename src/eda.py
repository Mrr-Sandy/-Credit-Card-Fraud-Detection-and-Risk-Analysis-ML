import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

os.makedirs("reports/figures", exist_ok=True)
df = pd.read_csv("data/transactions.csv")


def plot_class_balance():
    counts = df["is_fraud"].value_counts().sort_index()
    plt.figure(figsize=(5, 4))
    plt.bar(["Normal", "Fraud"], counts.values, color=["steelblue", "indianred"])
    for i, v in enumerate(counts.values):
        plt.text(i, v + 50, str(v), ha="center")
    plt.title("Normal vs Fraud transactions")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig("reports/figures/class_balance.png")
    plt.close()


def plot_amount():
    # log scale because fraud amounts are much bigger than normal ones
    plt.figure(figsize=(6, 4))
    plt.hist(df[df["is_fraud"] == 0]["amount"], bins=40, alpha=0.7, label="Normal")
    plt.hist(df[df["is_fraud"] == 1]["amount"], bins=40, alpha=0.7, label="Fraud")
    plt.xscale("log")
    plt.xlabel("amount (log scale)")
    plt.ylabel("count")
    plt.title("Transaction amount by class")
    plt.legend()
    plt.tight_layout()
    plt.savefig("reports/figures/amount_by_class.png")
    plt.close()


if __name__ == "__main__":
    plot_class_balance()
    plot_amount()
    print("fraud share: {:.1%}".format(df["is_fraud"].mean()))
