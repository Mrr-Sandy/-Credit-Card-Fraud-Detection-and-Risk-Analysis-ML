import joblib
import pandas as pd
from sklearn.metrics import precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


def load_train(path="data/train.csv"):
    return pd.read_csv(path)


def split_xy(df):
    # is_fraud is the target, everything else is a feature
    return df.drop(columns="is_fraud"), df["is_fraud"]


def train_logistic(X, y):
    # simple baseline model
    model = LogisticRegression(max_iter=1000, random_state=42)
    return model.fit(X, y)


def train_tree(X, y):
    # depth is limited so the tree does not just memorise the oversampled fraud rows
    model = DecisionTreeClassifier(max_depth=5, random_state=42)
    return model.fit(X, y)


def train_forest(X, y):
    model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
    return model.fit(X, y)


def quick_check(models, path="data/test.csv"):
    # only a sanity check on the untouched test set, full evaluation comes in the next step
    X_test, y_test = split_xy(pd.read_csv(path))
    for name, model in models.items():
        pred = model.predict(X_test)
        print(f"{name}: recall={recall_score(y_test, pred):.3f} precision={precision_score(y_test, pred):.3f}")


if __name__ == "__main__":
    X_train, y_train = split_xy(load_train())
    print("features:", X_train.columns.tolist())
    print("train shape:", X_train.shape, "fraud share:", round(y_train.mean(), 2))

    log_reg = train_logistic(X_train, y_train)
    print("logistic regression trained, train accuracy:", round(log_reg.score(X_train, y_train), 4))

    tree = train_tree(X_train, y_train)
    print("decision tree trained, train accuracy:", round(tree.score(X_train, y_train), 4))

    forest = train_forest(X_train, y_train)
    print("random forest trained, train accuracy:", round(forest.score(X_train, y_train), 4))

    # save the models so evaluation and prediction can load them later
    joblib.dump(log_reg, "models/logistic_regression.joblib")
    joblib.dump(tree, "models/decision_tree.joblib")
    joblib.dump(forest, "models/random_forest.joblib")
    print("saved 3 models in models/")

    quick_check({"logistic regression": log_reg, "decision tree": tree, "random forest": forest})
