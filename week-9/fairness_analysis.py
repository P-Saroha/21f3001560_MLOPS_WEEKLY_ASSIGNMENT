from pathlib import Path

import pandas as pd

from fairlearn.metrics import MetricFrame
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# ==========================================================
# Paths
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_FILE = BASE_DIR / "data" / "iris_with_location.csv"


# ==========================================================
# Dataset Configuration
# ==========================================================

FEATURE_COLUMNS = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
]

TARGET_COLUMN = "species"

SENSITIVE_COLUMN = "location"

RANDOM_STATE = 42


# ==========================================================
# Load Dataset
# ==========================================================

df = pd.read_csv(DATA_FILE)

print("=" * 70)
print("DATASET")
print("=" * 70)

print(df.head())

print(f"\nDataset Shape: {df.shape}")


# ==========================================================
# Encode Target
# ==========================================================

label_encoder = LabelEncoder()

df[TARGET_COLUMN] = label_encoder.fit_transform(
    df[TARGET_COLUMN]
)


# ==========================================================
# Define Features, Target and Sensitive Attribute
# ==========================================================

X = df[FEATURE_COLUMNS]

y = df[TARGET_COLUMN]

sensitive_features = df[SENSITIVE_COLUMN]


# ==========================================================
# Train/Test Split
# ==========================================================

X_train, X_test, y_train, y_test, sensitive_train, sensitive_test = (
    train_test_split(
        X,
        y,
        sensitive_features,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )
)


# ==========================================================
# Verify Sensitive Attribute Is NOT a Training Feature
# ==========================================================

print("\n" + "=" * 70)
print("FEATURE CONFIGURATION")
print("=" * 70)

print("Training Features:")
print(FEATURE_COLUMNS)

print("\nSensitive Attribute:")
print(SENSITIVE_COLUMN)

print("\nLocation included in training features?")
print(SENSITIVE_COLUMN in FEATURE_COLUMNS)


# ==========================================================
# Train Random Forest
# ==========================================================

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=RANDOM_STATE,
)

model.fit(
    X_train,
    y_train,
)


# ==========================================================
# Predictions
# ==========================================================

y_pred = model.predict(X_test)


# ==========================================================
# Overall Metrics
# ==========================================================

overall_accuracy = accuracy_score(
    y_test,
    y_pred,
)

overall_precision = precision_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0,
)

overall_recall = recall_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0,
)


# ==========================================================
# Fairlearn MetricFrame
# ==========================================================

metrics = {
    "accuracy": accuracy_score,
    "precision": lambda y_true, y_pred: precision_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    ),
    "recall": lambda y_true, y_pred: recall_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    ),
}


metric_frame = MetricFrame(
    metrics=metrics,
    y_true=y_test,
    y_pred=y_pred,
    sensitive_features=sensitive_test,
)


# ==========================================================
# Display Results
# ==========================================================

print("\n" + "=" * 70)
print("OVERALL MODEL PERFORMANCE")
print("=" * 70)

print(f"Accuracy  : {overall_accuracy:.4f}")
print(f"Precision : {overall_precision:.4f}")
print(f"Recall    : {overall_recall:.4f}")


print("\n" + "=" * 70)
print("FAIRNESS ANALYSIS — METRICFRAME")
print("=" * 70)

print("\nMetrics by Location:")
print(metric_frame.by_group)


print("\n" + "=" * 70)
print("FAIRNESS METRIC DIFFERENCES")
print("=" * 70)

print("\nDifference between worst and best group:")
print(metric_frame.difference())


print("\n" + "=" * 70)
print("FAIRNESS METRIC RANGES")
print("=" * 70)

print(metric_frame.ratio())


# ==========================================================
# Group Sizes
# ==========================================================

print("\n" + "=" * 70)
print("TEST SET GROUP DISTRIBUTION")
print("=" * 70)

print(
    sensitive_test.value_counts()
    .sort_index()
)