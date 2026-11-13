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


def plot_amount(df):
    plt.figure(figsize=(6, 4))
    plt.hist(df[df.is_fraud == 0]["amount"], bins=40, alpha=0.7, label="Normal", color="steelblue")
    plt.hist(df[df.is_fraud == 1]["amount"], bins=40, alpha=0.7, label="Fraud", color="indianred")
    plt.yscale("log")
    plt.xlabel("Amount (Rs)")
    plt.ylabel("Transactions (log scale)")
    plt.title("Transaction amount by class")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/amount_by_class_logy.png", dpi=120)
    plt.close()


def plot_hour(df):
    rate = df.groupby("hour")["is_fraud"].mean() * 100
    plt.figure(figsize=(7, 4))
    plt.bar(rate.index, rate.values, color="indianred")
    plt.xlabel("Hour of day")
    plt.ylabel("Fraud rate (%)")
    plt.title("Fraud rate by hour")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/fraud_rate_by_hour.png", dpi=120)
    plt.close()


def plot_behaviour(df):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    df.groupby("txn_count_1h")["is_fraud"].mean().mul(100).plot.bar(ax=axes[0], color="indianred")
    axes[0].set_title("Fraud rate by transactions in last hour")
    axes[0].set_ylabel("Fraud rate (%)")
    df.groupby("new_location")["is_fraud"].mean().mul(100).plot.bar(ax=axes[1], color="indianred")
    axes[1].set_title("Fraud rate by new location")
    axes[1].set_ylabel("Fraud rate (%)")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/behaviour_features.png", dpi=120)
    plt.close()


def plot_txn_type(df):
    rate = df.groupby("txn_type")["is_fraud"].mean() * 100
    plt.figure(figsize=(5, 4))
    plt.bar(rate.index, rate.values, color="indianred")
    plt.xlabel("Transaction type")
    plt.ylabel("Fraud rate (%)")
    plt.title("Fraud rate by transaction type")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/fraud_rate_by_txn_type.png", dpi=120)
    plt.close()


def plot_correlation(df):
    corr = df.drop(columns="txn_type").corr()
    plt.figure(figsize=(6, 5))
    plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar()
    plt.xticks(range(len(corr)), corr.columns, rotation=45, ha="right")
    plt.yticks(range(len(corr)), corr.columns)
    for i in range(len(corr)):
        for j in range(len(corr)):
            plt.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    plt.title("Correlation matrix")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/correlation.png", dpi=120)
    plt.close()


if __name__ == "__main__":
    df = load_data()
    basic_info(df)
    plot_class_distribution(df)
    plot_amount(df)
    plot_hour(df)
    plot_behaviour(df)
    plot_txn_type(df)
    plot_correlation(df)
