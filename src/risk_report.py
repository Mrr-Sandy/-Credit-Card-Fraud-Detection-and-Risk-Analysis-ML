import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from predict import load_model, risk_level

LEVELS = ["Low", "Medium", "High"]


def risk_table(model, path="data/test.csv"):
    # test.csv is already feature-engineered and scaled by preprocess.py
    df = pd.read_csv(path)
    proba = model.predict_proba(df[model.feature_names_in_])[:, 1]
    out = pd.DataFrame({"risk_level": [risk_level(p) for p in proba], "is_fraud": df["is_fraud"]})
    table = out.groupby("risk_level")["is_fraud"].agg(transactions="count", fraud="sum")
    return table.reindex(LEVELS, fill_value=0)


def plot_risk(table, path="reports/figures/risk_levels.png"):
    plt.figure(figsize=(5.5, 4))
    x = range(len(table))
    plt.bar(x, table["transactions"] - table["fraud"], label="Normal", color="steelblue")
    plt.bar(x, table["fraud"], bottom=table["transactions"] - table["fraud"], label="Fraud", color="indianred")
    plt.xticks(x, table.index)
    plt.ylabel("Transactions")
    plt.title("Test transactions by risk level")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


if __name__ == "__main__":
    table = risk_table(load_model())
    print(table)
    table.to_csv("reports/risk_summary.csv")
    plot_risk(table)
    print("saved reports/risk_summary.csv and reports/figures/risk_levels.png")
