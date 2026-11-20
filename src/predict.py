import joblib
import pandas as pd

MODEL_PATH = "models/random_forest.joblib"
SCALER_PATH = "models/scaler_values.csv"


def load_model(path=MODEL_PATH):
    return joblib.load(path)


def load_scaler(path=SCALER_PATH):
    # mean and std that were saved from the train set in preprocess.py
    return pd.read_csv(path, index_col=0)


if __name__ == "__main__":
    model = load_model()
    scaler = load_scaler()
    print("model features:", list(model.feature_names_in_))
    print("scaled columns:", list(scaler.index))
