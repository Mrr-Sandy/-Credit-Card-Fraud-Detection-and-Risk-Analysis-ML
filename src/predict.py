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


def predict_transaction(transaction, model=None, scaler=None):
    # main function: raw transaction in, probability and risk level out
    model = model or load_model()
    scaler = scaler if scaler is not None else load_scaler()
    prob = fraud_probability(transaction, model, scaler)
    return {"fraud_probability": round(prob, 3), "risk_level": risk_level(prob)}


EXAMPLES = {
    "usual daytime purchase": {"amount": 450, "hour": 14, "txn_count_1h": 1, "avg_spending": 700,
                               "new_location": 0, "txn_type": "pos"},
    "big amount, but daytime and usual place": {"amount": 9000, "hour": 15, "txn_count_1h": 1,
                                                "avg_spending": 800, "new_location": 0, "txn_type": "online"},
    "big amount, night, new place, many txns": {"amount": 35000, "hour": 2, "txn_count_1h": 6,
                                                "avg_spending": 600, "new_location": 1, "txn_type": "online"},
}


if __name__ == "__main__":
    model, scaler = load_model(), load_scaler()
    for name, txn in EXAMPLES.items():
        print(f"{name}: {predict_transaction(txn, model, scaler)}")
