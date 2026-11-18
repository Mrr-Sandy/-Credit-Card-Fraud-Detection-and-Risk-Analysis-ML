import joblib
import pandas as pd

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


if __name__ == "__main__":
    X_test, y_test = load_test()
    models = load_models()
    print("test shape:", X_test.shape, "fraud rows:", int(y_test.sum()))
    print("models loaded:", list(models))

    for name, model in models.items():
        pred, proba = get_predictions(model, X_test)
        print(name, "- predicted fraud:", int(pred.sum()), "- max proba:", round(proba.max(), 3))
