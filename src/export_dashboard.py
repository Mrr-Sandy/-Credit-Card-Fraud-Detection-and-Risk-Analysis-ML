"""Export the trained Random Forest and the numbers the web dashboard needs.

Writes:
  dashboard/api/_model.json    trees + scaler, used by the /api/predict function
  dashboard/public/data.json   statistics and metrics shown on the dashboard
"""
import json

import numpy as np
import pandas as pd

from predict import load_model, load_scaler

LOW, HIGH = 0.30, 0.70


def export_model(model, scaler):
    trees = []
    for est in model.estimators_:
        t = est.tree_
        value = t.value[:, 0, :]
        prob = value[:, 1] / value.sum(axis=1)
        leaf = t.children_left == -1
        trees.append({
            "f": np.where(leaf, -1, t.feature).tolist(),
            # internal node: split threshold, leaf: fraud probability
            "t": [round(float(p if lf else th), 6) for p, th, lf in zip(prob, t.threshold, leaf)],
            "l": t.children_left.tolist(),
            "r": t.children_right.tolist(),
        })
    return {
        "features": list(model.feature_names_in_),
        "scale": {c: {"mean": float(scaler.loc[c, "mean"]), "std": float(scaler.loc[c, "std"])}
                  for c in scaler.index},
        "trees": trees,
    }


def hist_log(series, edges):
    counts, _ = np.histogram(series, bins=edges)
    return counts.tolist()


def build_stats(model):
    raw = pd.read_csv("data/transactions.csv")
    test = pd.read_csv("data/test.csv")
    train = pd.read_csv("data/train.csv")
    fraud = raw[raw.is_fraud == 1]
    normal = raw[raw.is_fraud == 0]

    edges = np.logspace(np.log10(raw.amount.min()), np.log10(raw.amount.max()), 21)
    proba = model.predict_proba(test[model.feature_names_in_])[:, 1]
    level = np.where(proba < LOW, "Low", np.where(proba < HIGH, "Medium", "High"))
    risk = {k: {"normal": int(((level == k) & (test.is_fraud == 0)).sum()),
                "fraud": int(((level == k) & (test.is_fraud == 1)).sum())}
            for k in ["Low", "Medium", "High"]}

    metrics = pd.read_csv("reports/model_comparison.csv", index_col=0)
    pred = model.predict(test[model.feature_names_in_])
    tn = int(((pred == 0) & (test.is_fraud == 0)).sum())
    fp = int(((pred == 1) & (test.is_fraud == 0)).sum())
    fn = int(((pred == 0) & (test.is_fraud == 1)).sum())
    tp = int(((pred == 1) & (test.is_fraud == 1)).sum())

    imp = sorted(zip(model.feature_names_in_, model.feature_importances_), key=lambda x: -x[1])
    return {
        "dataset": {
            "rows": int(len(raw)), "fraud": int(len(fraud)), "normal": int(len(normal)),
            "fraud_rate": round(len(fraud) / len(raw), 4),
            "train_rows": int(len(train)), "test_rows": int(len(test)),
            "test_fraud": int(test.is_fraud.sum()),
        },
        "hour": {"normal": normal.hour.value_counts().reindex(range(24), fill_value=0).tolist(),
                 "fraud": fraud.hour.value_counts().reindex(range(24), fill_value=0).tolist()},
        "amount": {"edges": [round(float(e), 1) for e in edges],
                   "normal_max": round(float(normal.amount.max()), 2),
                   "fraud_min": round(float(fraud.amount.min()), 2),
                   "normal": hist_log(normal.amount, edges), "fraud": hist_log(fraud.amount, edges)},
        "txn_count": {"normal": normal.txn_count_1h.value_counts().reindex(range(1, 9), fill_value=0).tolist(),
                      "fraud": fraud.txn_count_1h.value_counts().reindex(range(1, 9), fill_value=0).tolist()},
        "new_location": {"normal": round(float(normal.new_location.mean()), 3),
                         "fraud": round(float(fraud.new_location.mean()), 3)},
        "txn_type": {t: {"normal": int((normal.txn_type == t).sum()), "fraud": int((fraud.txn_type == t).sum())}
                     for t in ["online", "pos", "atm"]},
        "importance": [{"feature": f, "value": round(float(v), 4)} for f, v in imp],
        "metrics": {name: {k: float(v) for k, v in row.items()} for name, row in metrics.iterrows()},
        "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "risk": risk,
        "thresholds": {"low": LOW, "high": HIGH},
    }


if __name__ == "__main__":
    model, scaler = load_model(), load_scaler()
    with open("dashboard/api/_model.json", "w") as f:
        json.dump(export_model(model, scaler), f, separators=(",", ":"))
    with open("dashboard/public/data.json", "w") as f:
        json.dump(build_stats(model), f, separators=(",", ":"))
    print("wrote dashboard/api/_model.json and dashboard/public/data.json")
