
import pandas as pd

path = "data/lmvd_manifest.csv"
df = pd.read_csv(path)

print("Columns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head().to_string(index=False))

print("\nDuplicate sample IDs:")
print(df["sample_id"].duplicated().sum())

print("\nLabels:")
print(df["label"].value_counts().sort_index())
