import os

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

MODEL_FILES = {
    "Logistic Regression": "models/logistic_regression.joblib",
    "Decision Tree": "models/decision_tree.joblib",
    "Random Forest": "models/random_forest.joblib",
}


def load_test(path="data/test.csv"):
    df = pd.read_csv(path)
    return df.drop(columns="is_fraud"), df["is_fraud"]


def load_models():
    return {name: joblib.load(path) for name, path in MODEL_FILES.items()}


def get_predictions(model, X):
    # hard label (0/1) and fraud probability for every transaction
    return model.predict(X), model.predict_proba(X)[:, 1]


def compute_metrics(y_true, pred, proba):
    # accuracy is left out on purpose, it is misleading when fraud is rare
    return {
        "Precision": precision_score(y_true, pred),
        "Recall": recall_score(y_true, pred),
        "F1": f1_score(y_true, pred),
        "ROC-AUC": roc_auc_score(y_true, proba),
        "PR-AUC": average_precision_score(y_true, proba),
    }


def build_table(models, X, y):
    rows = {}
    for name, model in models.items():
        pred, proba = get_predictions(model, X)
        rows[name] = compute_metrics(y, pred, proba)
    return pd.DataFrame(rows).T.round(3)


FIG_DIR = "reports/figures"


def plot_confusion_matrices(models, X, y):
    fig, axes = plt.subplots(1, len(models), figsize=(4 * len(models), 3.6))
    for ax, (name, model) in zip(axes, models.items()):
        cm = confusion_matrix(y, model.predict(X))
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=12)
        ax.set_xticks([0, 1], ["Normal", "Fraud"])
        ax.set_yticks([0, 1], ["Normal", "Fraud"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(name)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/confusion_matrices.png", dpi=120)
    plt.close()


if __name__ == "__main__":
    X_test, y_test = load_test()
    models = load_models()
    print("test shape:", X_test.shape, "fraud rows:", int(y_test.sum()))
    print("models loaded:", list(models))

    table = build_table(models, X_test, y_test)
    print(table)

    os.makedirs("reports", exist_ok=True)
    table.to_csv("reports/model_comparison.csv")
    print("saved reports/model_comparison.csv")

    plot_confusion_matrices(models, X_test, y_test)
