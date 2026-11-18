import joblib
import pandas as pd
from sklearn.metrics import (
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


if __name__ == "__main__":
    X_test, y_test = load_test()
    models = load_models()
    print("test shape:", X_test.shape, "fraud rows:", int(y_test.sum()))
    print("models loaded:", list(models))

    for name, model in models.items():
        pred, proba = get_predictions(model, X_test)
        print(name, {k: round(v, 3) for k, v in compute_metrics(y_test, pred, proba).items()})
