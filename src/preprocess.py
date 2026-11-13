import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SCALE_COLS = ["amount", "avg_spending", "amount_ratio", "txn_count_1h", "hour"]


def load_data(path="data/transactions.csv"):
    return pd.read_csv(path)


def add_features(df):
    df = df.copy()
    # how big is this transaction compared to what the card usually spends
    df["amount_ratio"] = (df["amount"] / df["avg_spending"]).round(3)
    # late night (11pm to 5am) flag
    df["is_night"] = df["hour"].isin([23, 0, 1, 2, 3, 4]).astype(int)
    return df


def encode_txn_type(df):
    # one-hot encode online / pos / atm
    return pd.get_dummies(df, columns=["txn_type"], dtype=int)


def split_and_scale(df, test_size=0.2, seed=42):
    X = df.drop(columns="is_fraud")
    y = df["is_fraud"]

    # stratify keeps the 5% fraud share the same in train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    # fit the scaler on train only so test data does not leak into it
    scaler = StandardScaler()
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train[SCALE_COLS] = scaler.fit_transform(X_train[SCALE_COLS])
    X_test[SCALE_COLS] = scaler.transform(X_test[SCALE_COLS])
    return X_train, X_test, y_train, y_test, scaler


if __name__ == "__main__":
    df = encode_txn_type(add_features(load_data()))
    X_train, X_test, y_train, y_test, scaler = split_and_scale(df)
    print("train:", X_train.shape, "fraud share:", round(y_train.mean(), 3))
    print("test: ", X_test.shape, "fraud share:", round(y_test.mean(), 3))

    # save processed data and the scaler for the modeling step
    X_train.assign(is_fraud=y_train).to_csv("data/train.csv", index=False)
    X_test.assign(is_fraud=y_test).to_csv("data/test.csv", index=False)
    joblib.dump(scaler, "models/scaler.joblib")
    print("saved data/train.csv, data/test.csv, models/scaler.joblib")
