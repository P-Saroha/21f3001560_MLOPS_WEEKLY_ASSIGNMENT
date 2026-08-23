import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# Configuration
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

SOURCE_PATH = (
    BASE_DIR.parent
    / "week-8"
    / "data"
    / "iris_clean.csv"
)

DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Load IRIS dataset
# ============================================================

df = pd.read_csv(SOURCE_PATH)

REQUIRED_COLUMNS = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
    "species",
]

missing_columns = [
    column for column in REQUIRED_COLUMNS
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

df = df[REQUIRED_COLUMNS].copy()


# ============================================================
# Create ONE fixed split
#
# 80% train      = 120
# 10% validation = 15
# 10% test       = 15
#
# Stratified by species
# ============================================================

train_df, temp_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=df["species"],
)

validation_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=42,
    stratify=temp_df["species"],
)

train_df = train_df.reset_index(drop=True)
validation_df = validation_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)


# ============================================================
# V1 — Raw Feature Representation
# ============================================================

def create_v1_record(row):
    input_text = (
        f"sepal_length: {row['sepal_length']}, "
        f"sepal_width: {row['sepal_width']}, "
        f"petal_length: {row['petal_length']}, "
        f"petal_width: {row['petal_width']}"
    )

    return {
        "input_text": input_text,
        "output_text": row["species"],
    }


# ============================================================
# V2 — Natural Language Representation
# ============================================================

def create_v2_record(row):
    input_text = (
        f"A flower specimen has a sepal length of "
        f"{row['sepal_length']} cm, "
        f"sepal width of {row['sepal_width']} cm, "
        f"petal length of {row['petal_length']} cm, and "
        f"petal width of {row['petal_width']} cm. "
        f"Identify the iris species."
    )

    output_text = f"This is Iris {row['species']}."

    return {
        "input_text": input_text,
        "output_text": output_text,
    }


# ============================================================
# Write JSONL
# ============================================================

def write_jsonl(dataframe, output_path, record_function):
    with open(output_path, "w", encoding="utf-8") as f:
        for _, row in dataframe.iterrows():
            record = record_function(row)
            f.write(
                json.dumps(record, ensure_ascii=False)
                + "\n"
            )


# V1
write_jsonl(
    train_df,
    DATA_DIR / "v1_train.jsonl",
    create_v1_record,
)

write_jsonl(
    validation_df,
    DATA_DIR / "v1_validation.jsonl",
    create_v1_record,
)

# V2
write_jsonl(
    train_df,
    DATA_DIR / "v2_train.jsonl",
    create_v2_record,
)

write_jsonl(
    validation_df,
    DATA_DIR / "v2_validation.jsonl",
    create_v2_record,
)


# ============================================================
# Save held-out test set
# ============================================================

test_df.to_csv(
    DATA_DIR / "test.csv",
    index=False,
)


# ============================================================
# Summary
# ============================================================

print("=" * 65)
print("WEEK 10 - IRIS LLMOPS DATA PREPARATION")
print("=" * 65)

print(f"\nSource dataset:")
print(SOURCE_PATH)

print(f"\nTotal samples:      {len(df)}")
print(f"Training samples:   {len(train_df)}")
print(f"Validation samples: {len(validation_df)}")
print(f"Test samples:       {len(test_df)}")

print("\nTraining distribution:")
print(train_df["species"].value_counts().sort_index())

print("\nValidation distribution:")
print(validation_df["species"].value_counts().sort_index())

print("\nTest distribution:")
print(test_df["species"].value_counts().sort_index())

print("\nCreated files:")

for filename in [
    "v1_train.jsonl",
    "v1_validation.jsonl",
    "v2_train.jsonl",
    "v2_validation.jsonl",
    "test.csv",
]:
    print(DATA_DIR / filename)

print("\nV1 and V2 use the SAME fixed train/validation/test split.")
print("=" * 65)
