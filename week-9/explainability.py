from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import shap

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder


# ==========================================================
# Paths
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_FILE = BASE_DIR / "data" / "iris_with_location.csv"

PLOTS_DIR = BASE_DIR / "plots"


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

RANDOM_STATE = 42


# ==========================================================
# Load Dataset
# ==========================================================

df = pd.read_csv(DATA_FILE)

print("=" * 70)
print("SHAP EXPLAINABILITY")
print("=" * 70)

print(f"Dataset shape: {df.shape}")


# ==========================================================
# Prepare Features and Target
# ==========================================================

X = df[FEATURE_COLUMNS]

label_encoder = LabelEncoder()

y = label_encoder.fit_transform(
    df[TARGET_COLUMN]
)

class_names = list(label_encoder.classes_)

print("\nClasses:")
for index, class_name in enumerate(class_names):
    print(f"{index}: {class_name}")

print("\nFeatures used for model:")
print(FEATURE_COLUMNS)

print("\nLocation excluded from model features:")
print("location")


# ==========================================================
# Train Random Forest
# ==========================================================

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=RANDOM_STATE,
)

model.fit(
    X,
    y,
)

print("\nRandom Forest trained successfully.")


# ==========================================================
# Create SHAP Tree Explainer
# ==========================================================

explainer = shap.TreeExplainer(model)

print("\nSHAP TreeExplainer created.")


# ==========================================================
# Generate SHAP Values for FULL DATASET
# ==========================================================

shap_explanation = explainer(X)

print("\nSHAP values generated using the FULL dataset.")

print(f"SHAP values shape: {shap_explanation.values.shape}")


# ==========================================================
# Generate Summary Plot for Each Class
# ==========================================================

for class_index, class_name in enumerate(class_names):

    print("\n" + "=" * 70)
    print(f"Generating SHAP plot for: {class_name}")
    print("=" * 70)

    # ------------------------------------------------------
    # Extract SHAP values for this class
    # ------------------------------------------------------

    class_shap_values = shap_explanation.values[:, :, class_index]

    # ------------------------------------------------------
    # Create summary plot
    # ------------------------------------------------------

    plt.figure()

    shap.summary_plot(
        class_shap_values,
        X,
        show=False,
    )

    plt.title(
        f"SHAP Summary Plot - {class_name}"
    )

    plt.tight_layout()

    # ------------------------------------------------------
    # Save plot
    # ------------------------------------------------------

    output_file = (
        PLOTS_DIR
        / f"shap_{class_name}.png"
    )

    plt.savefig(
        output_file,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Saved: {output_file}")


# ==========================================================
# Virginica Feature Importance
# ==========================================================

virginica_index = class_names.index("virginica")

virginica_shap_values = (
    shap_explanation.values[
        :,
        :,
        virginica_index
    ]
)

mean_abs_shap = (
    abs(virginica_shap_values)
    .mean(axis=0)
)

importance_df = pd.DataFrame(
    {
        "feature": FEATURE_COLUMNS,
        "mean_absolute_shap": mean_abs_shap,
    }
).sort_values(
    "mean_absolute_shap",
    ascending=False,
)


# ==========================================================
# Display Virginica Importance
# ==========================================================

print("\n" + "=" * 70)
print("VIRGINICA FEATURE IMPORTANCE")
print("=" * 70)

print(
    importance_df.to_string(
        index=False
    )
)


# ==========================================================
# Final Output
# ==========================================================

print("\n" + "=" * 70)
print("SHAP ANALYSIS COMPLETE")
print("=" * 70)

print("\nGenerated plots:")

for class_name in class_names:
    print(
        PLOTS_DIR
        / f"shap_{class_name}.png"
    )