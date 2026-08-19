from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import ks_2samp


# ==========================================================
# Paths
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"

REFERENCE_FILE = DATA_DIR / "iris_clean.csv"
PRODUCTION_FILE = DATA_DIR / "iris_production.csv"

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

RANDOM_STATE = 42


# ==========================================================
# Load Reference / Training Dataset
# ==========================================================

reference_df = pd.read_csv(
    REFERENCE_FILE
)

print("=" * 70)
print("DATA DRIFT DETECTION")
print("=" * 70)

print(
    f"Reference dataset shape: {reference_df.shape}"
)


# ==========================================================
# Create Simulated Production Dataset
# ==========================================================

production_df = reference_df.copy()


# ----------------------------------------------------------
# Introduce controlled distribution shifts
# ----------------------------------------------------------

production_df["petal_length"] = (
    production_df["petal_length"] + 1.0
)

production_df["petal_width"] = (
    production_df["petal_width"] + 0.5
)


# ----------------------------------------------------------
# Save production dataset
# ----------------------------------------------------------

production_df.to_csv(
    PRODUCTION_FILE,
    index=False,
)

print(
    f"Production dataset saved to: {PRODUCTION_FILE}"
)


# ==========================================================
# Display Feature Statistics
# ==========================================================

print("\n" + "=" * 70)
print("FEATURE DISTRIBUTION COMPARISON")
print("=" * 70)

statistics = []

for feature in FEATURE_COLUMNS:

    reference_mean = reference_df[feature].mean()
    production_mean = production_df[feature].mean()

    reference_std = reference_df[feature].std()
    production_std = production_df[feature].std()

    statistics.append(
        {
            "feature": feature,
            "reference_mean": reference_mean,
            "production_mean": production_mean,
            "mean_shift": (
                production_mean
                - reference_mean
            ),
            "reference_std": reference_std,
            "production_std": production_std,
        }
    )

statistics_df = pd.DataFrame(
    statistics
)

print(
    statistics_df.to_string(
        index=False
    )
)


# ==========================================================
# Kolmogorov-Smirnov Tests
# ==========================================================

print("\n" + "=" * 70)
print("KOLMOGOROV-SMIRNOV DRIFT TEST")
print("=" * 70)

drift_results = []

for feature in FEATURE_COLUMNS:

    reference_values = reference_df[feature]

    production_values = production_df[feature]

    ks_statistic, p_value = ks_2samp(
        reference_values,
        production_values,
    )

    # ------------------------------------------------------
    # Drift decision
    # ------------------------------------------------------

    drift_detected = p_value < 0.05

    drift_results.append(
        {
            "feature": feature,
            "ks_statistic": ks_statistic,
            "p_value": p_value,
            "drift_detected": drift_detected,
        }
    )

    print(f"\nFeature: {feature}")
    print(
        f"KS Statistic : {ks_statistic:.4f}"
    )
    print(
        f"P-value      : {p_value:.6f}"
    )

    if drift_detected:
        print(
            "Result       : DRIFT DETECTED"
        )
    else:
        print(
            "Result       : NO SIGNIFICANT DRIFT"
        )


drift_results_df = pd.DataFrame(
    drift_results
)


# ==========================================================
# Distribution Plots
# ==========================================================

print("\n" + "=" * 70)
print("GENERATING DISTRIBUTION PLOTS")
print("=" * 70)

for feature in FEATURE_COLUMNS:

    plt.figure(figsize=(8, 5))

    plt.hist(
        reference_df[feature],
        bins=15,
        alpha=0.5,
        label="Training / Reference",
    )

    plt.hist(
        production_df[feature],
        bins=15,
        alpha=0.5,
        label="Production",
    )

    plt.xlabel(feature)
    plt.ylabel("Frequency")

    plt.title(
        f"Training vs Production - {feature}"
    )

    plt.legend()

    plt.tight_layout()

    output_file = (
        PLOTS_DIR
        / f"drift_{feature}.png"
    )

    plt.savefig(
        output_file,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Saved: {output_file}"
    )


# ==========================================================
# Final Summary
# ==========================================================

print("\n" + "=" * 70)
print("DRIFT SUMMARY")
print("=" * 70)

print(
    drift_results_df.to_string(
        index=False
    )
)

print("\nData drift analysis complete.")