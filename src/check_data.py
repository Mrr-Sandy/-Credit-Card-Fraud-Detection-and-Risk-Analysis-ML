import pandas as pd

df = pd.read_csv("data/transactions.csv")

print("shape:", df.shape)
print()
print("missing values:")
print(df.isnull().sum())
print()
print("normal vs fraud:")
print(df["is_fraud"].value_counts())
print()
print("average values by class:")
print(df.groupby("is_fraud")[["amount", "hour", "txn_count_1h", "new_location"]].mean().round(2))
