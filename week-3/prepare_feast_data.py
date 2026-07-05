import pandas as pd
import os

# ==========================================================
# IITM Feast Dataset Path
# ==========================================================

DATA_PATH = "feature_repo/feature_repo/data/iris_data_adapted_for_feast.csv"

# ==========================================================
# Verify Dataset Exists
# ==========================================================

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Dataset not found:\n{DATA_PATH}"
    )

# ==========================================================
# Load Dataset
# ==========================================================

df = pd.read_csv(DATA_PATH)

# ==========================================================
# Convert Timestamp Columns
# ==========================================================

df["event_timestamp"] = pd.to_datetime(df["event_timestamp"])
df["created_timestamp"] = pd.to_datetime(df["created_timestamp"])

# ==========================================================
# Display Dataset Information
# ==========================================================

print("\n==============================")
print("IITM Feast Dataset Loaded")
print("==============================\n")

print("Dataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst Five Rows:")
print(df.head())

print("\nData Types:")
print(df.dtypes)

print("\nUnique Iris IDs:")
print(df["iris_id"].unique())

print("\nSpecies:")
print(df["species"].unique())

print("\nDataset is ready for Feast.")