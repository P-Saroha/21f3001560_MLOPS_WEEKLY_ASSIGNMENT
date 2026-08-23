from pathlib import Path

import numpy as np
import pandas as pd


# ==========================================================
# Paths
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"

INPUT_FILE = DATA_DIR / "iris_clean.csv"
OUTPUT_FILE = DATA_DIR / "iris_with_location.csv"


# ==========================================================
# Configuration
# ==========================================================

RANDOM_STATE = 42


# ==========================================================
# Load Dataset
# ==========================================================

df = pd.read_csv(INPUT_FILE)

print("Original Dataset:")
print(df.head())

print("\nOriginal Shape:")
print(df.shape)


# ==========================================================
# Add Sensitive Attribute
# ==========================================================

rng = np.random.default_rng(RANDOM_STATE)

df["location"] = rng.integers(
    0,
    2,
    size=len(df),
)


# ==========================================================
# Save Dataset
# ==========================================================

df.to_csv(
    OUTPUT_FILE,
    index=False,
)


# ==========================================================
# Verification
# ==========================================================

print("\nDataset With Location:")
print(df.head())

print("\nLocation Distribution:")
print(
    df["location"]
    .value_counts()
    .sort_index()
)

print("\nFinal Shape:")
print(df.shape)

print(f"\nSaved to: {OUTPUT_FILE}")