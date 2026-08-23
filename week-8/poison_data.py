from pathlib import Path

import numpy as np
import pandas as pd


# ==========================================================
# Paths
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_PATH = BASE_DIR / "data" / "iris_clean.csv"
OUTPUT_DIR = BASE_DIR / "data"


# ==========================================================
# Configuration
# ==========================================================

FEATURE_COLUMNS = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
]

TARGET_COLUMN = "species"

POISONING_LEVELS = {
    "5": 0.05,
    "10": 0.10,
    "50": 0.50,
}

RANDOM_SEED = 42


# ==========================================================
# Poisoning Function
# ==========================================================

def create_poisoned_dataset(df, poisoning_rate, seed):
    rng = np.random.default_rng(seed)

    poisoned_df = df.copy()

    number_of_samples = len(poisoned_df)

    number_to_poison = round(
        number_of_samples * poisoning_rate
    )

    poisoned_indices = rng.choice(
        number_of_samples,
        size=number_to_poison,
        replace=False,
    )

    # ------------------------------------------------------
    # Replace all four features with random values
    # ------------------------------------------------------

    for feature in FEATURE_COLUMNS:

        minimum = df[feature].min()
        maximum = df[feature].max()

        poisoned_df.loc[
            poisoned_indices,
            feature
        ] = rng.uniform(
            minimum,
            maximum,
            size=number_to_poison,
        )

    # ------------------------------------------------------
    # Replace labels with random classes
    # ------------------------------------------------------

    classes = df[TARGET_COLUMN].unique()

    poisoned_df.loc[
        poisoned_indices,
        TARGET_COLUMN
    ] = rng.choice(
        classes,
        size=number_to_poison,
    )

    return poisoned_df, poisoned_indices


# ==========================================================
# Main
# ==========================================================

def main():

    df = pd.read_csv(INPUT_PATH)

    print("=" * 60)
    print("Original Dataset")
    print("=" * 60)
    print(f"Rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print("\nOriginal class distribution:")
    print(df[TARGET_COLUMN].value_counts())

    # ------------------------------------------------------
    # Generate poisoned datasets
    # ------------------------------------------------------

    for level, poisoning_rate in POISONING_LEVELS.items():

        poisoned_df, poisoned_indices = create_poisoned_dataset(
            df,
            poisoning_rate,
            RANDOM_SEED,
        )

        output_path = (
            OUTPUT_DIR /
            f"iris_poisoned_{level}.csv"
        )

        poisoned_df.to_csv(
            output_path,
            index=False,
        )

        print("\n" + "=" * 60)
        print(f"Poisoning Level: {level}%")
        print("=" * 60)

        print(
            f"Samples poisoned: "
            f"{len(poisoned_indices)}"
        )

        print(
            f"Dataset saved to: "
            f"{output_path}"
        )

        print("\nClass distribution:")
        print(
            poisoned_df[TARGET_COLUMN]
            .value_counts()
        )

    print("\n" + "=" * 60)
    print("Poisoning completed successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()
