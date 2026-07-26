from pathlib import Path

import pandas as pd

# ==========================================================
# Paths
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "feature_repo" / "feature_repo" / "data" / "iris_data_adapted_for_feast.csv"


def load_dataset():

    df = pd.read_csv(DATA_PATH)

    df["event_timestamp"] = pd.to_datetime(
        df["event_timestamp"]
    )

    df["created_timestamp"] = pd.to_datetime(
        df["created_timestamp"]
    )

    return df


def main():

    df = load_dataset()

    print("\n==============================")
    print("IITM Feast Dataset Loaded")
    print("==============================")

    print("\nShape")
    print(df.shape)

    print("\nColumns")
    print(df.columns.tolist())

    print("\nData Types")
    print(df.dtypes)

    print("\nFirst Five Rows")
    print(df.head())

    print("\nUnique Iris IDs")
    print(df["iris_id"].unique())

    print("\nSpecies")
    print(df["species"].unique())

    print("\nDataset Ready For Feast")


if __name__ == "__main__":
    main()