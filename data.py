import pandas as pd

df = pd.read_csv("data/fraud_test_preprocessed.csv")
print(df.columns)
print(df.head())
print(df.info())
