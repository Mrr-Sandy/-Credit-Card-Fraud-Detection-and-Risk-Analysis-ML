import pandas as pd
from sklearn.linear_model import LogisticRegression


def load_train(path="data/train.csv"):
    return pd.read_csv(path)


def split_xy(df):
    # is_fraud is the target, everything else is a feature
    return df.drop(columns="is_fraud"), df["is_fraud"]


def train_logistic(X, y):
    # simple baseline model
    model = LogisticRegression(max_iter=1000, random_state=42)
    return model.fit(X, y)


if __name__ == "__main__":
    X_train, y_train = split_xy(load_train())
    print("features:", X_train.columns.tolist())
    print("train shape:", X_train.shape, "fraud share:", round(y_train.mean(), 2))

    log_reg = train_logistic(X_train, y_train)
    print("logistic regression trained, train accuracy:", round(log_reg.score(X_train, y_train), 4))
