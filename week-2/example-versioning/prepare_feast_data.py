from sklearn.datasets import load_iris
import pandas as pd

# Load iris dataset
iris = load_iris(as_frame=True)

# Create dataframe
df = iris.frame.copy()

# Rename target column
df.rename(columns={"target": "species"}, inplace=True)

# Add unique ID
df["sample_id"] = range(len(df))

# Feast requires timestamp
df["event_timestamp"] = pd.Timestamp("2024-01-01")

# Create output directory if it doesn't exist
import os
os.makedirs("feature_repo/feature_repo/data", exist_ok=True)

# Save parquet
output_path = "feature_repo/feature_repo/data/iris_features.parquet"

df.to_parquet(output_path, index=False)

print(df.head())
print()
print("Saved to:", output_path)