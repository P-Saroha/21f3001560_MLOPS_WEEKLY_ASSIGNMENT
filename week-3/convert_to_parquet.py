import pandas as pd

df = pd.read_csv("feature_repo/feature_repo/data/iris_data_adapted_for_feast.csv")

df["event_timestamp"] = pd.to_datetime(df["event_timestamp"])
df["created_timestamp"] = pd.to_datetime(df["created_timestamp"])

df.to_parquet(
    "feature_repo/feature_repo/data/iris_data_adapted_for_feast.parquet",
    index=False,
)

print("Parquet file created successfully.")