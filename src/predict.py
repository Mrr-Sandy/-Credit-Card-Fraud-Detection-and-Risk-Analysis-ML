import joblib
import pandas as pd

from features import build_features

# probability below LOW_LIMIT is Low, from HIGH_LIMIT upwards is High, the rest is Medium
LOW_LIMIT = 0.30
HIGH_LIMIT = 0.70
MODEL_PATH = "models/random_forest.joblib"
SCALER_PATH = "models/scaler_values.csv"


def load_model(path=MODEL_PATH):
    return joblib.load(path)


def load_scaler(path=SCALER_PATH):
    # mean and std that were saved from the train set in preprocess.py
    return pd.read_csv(path, index_col=0)


def prepare_input(transaction, model):
    # one raw transaction (dict) -> same feature columns the model was trained on
    df = build_features(pd.DataFrame([transaction]))
    # a single row only creates one type_* column, so add the missing ones as 0
    return df.reindex(columns=model.feature_names_in_, fill_value=0)


def scale_input(df, scaler):
    # use the train mean/std, never recompute them on new data
    df = df.copy()
    cols = list(scaler.index)
    df[cols] = (df[cols] - scaler["mean"]) / scaler["std"]
    return df


def fraud_probability(transaction, model, scaler):
    X = scale_input(prepare_input(transaction, model), scaler)
    return float(model.predict_proba(X)[0, 1])


def risk_level(probability):
    if probability < LOW_LIMIT:
        return "Low"
    if probability < HIGH_LIMIT:
        return "Medium"
    return "High"


if __name__ == "__main__":
    model = load_model()
    scaler = load_scaler()
    print("model features:", list(model.feature_names_in_))
    print("scaled columns:", list(scaler.index))

    example = {"amount": 450, "hour": 14, "txn_count_1h": 1, "avg_spending": 700,
               "new_location": 0, "txn_type": "pos"}
    prob = fraud_probability(example, model, scaler)
    print("fraud probability:", round(prob, 3), "->", risk_level(prob), "risk")
