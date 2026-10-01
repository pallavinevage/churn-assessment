from pathlib import Path
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
df = pd.read_csv(DATA)

print("Rows, columns:", df.shape)
print("\nColumn types:\n", df.dtypes)
print("\nMissing values:\n", df.isna().sum().sum())
print("\nBlank TotalCharges rows:", (df["TotalCharges"].str.strip() == "").sum())
print("\nChurn share:\n", df["Churn"].value_counts(normalize=True))
print("\nChurn rate by contract:\n", df.groupby("Contract")["Churn"].apply(lambda s: (s == "Yes").mean()))