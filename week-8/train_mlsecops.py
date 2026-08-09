from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# ==========================================================
# Paths
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"


# ==========================================================
# Dataset Configuration
# ==========================================================

DATASETS = {
    "Clean_0%": DATA_DIR / "iris_clean.csv",
    "Poisoned_5%": DATA_DIR / "iris_poisoned_5.csv",
    "Poisoned_10%": DATA_DIR / "iris_poisoned_10.csv",
    "Poisoned_50%": DATA_DIR / "iris_poisoned_50.csv",
}


FEATURE_COLUMNS = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
]

TARGET_COLUMN = "species"


# ==========================================================
# MLflow Configuration
# ==========================================================

MLFLOW_DB = BASE_DIR / "mlflow.db"

mlflow.set_tracking_uri(
    f"sqlite:///{MLFLOW_DB}"
)

mlflow.set_experiment(
    "iris_mlsecops_poisoning"
)


# ==========================================================
# Load Clean Test Set
# ==========================================================

clean_df = pd.read_csv(
    DATA_DIR / "iris_clean.csv"
)

label_encoder = LabelEncoder()

clean_df[TARGET_COLUMN] = label_encoder.fit_transform(
    clean_df[TARGET_COLUMN]
)

X_clean = clean_df[FEATURE_COLUMNS]
y_clean = clean_df[TARGET_COLUMN]


# ==========================================================
# Create ONE fixed clean test set
# ==========================================================

X_train_clean, X_test, y_train_clean, y_test = train_test_split(
    X_clean,
    y_clean,
    test_size=0.20,
    random_state=42,
    stratify=y_clean,
)


# ==========================================================
# Training Function
# ==========================================================

def train_and_log(run_name, dataset_path, poisoning_level):

    print("\n" + "=" * 70)
    print(f"RUN: {run_name}")
    print("=" * 70)

    df = pd.read_csv(dataset_path)

    # ------------------------------------------------------
    # Encode labels using the SAME encoder
    # ------------------------------------------------------

    df[TARGET_COLUMN] = label_encoder.transform(
        df[TARGET_COLUMN]
    )

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    # ------------------------------------------------------
    # Use the same test indices as the clean dataset
    #
    # We create the poisoned training data separately.
    # The test set always comes from clean data.
    # ------------------------------------------------------

    train_indices, test_indices = train_test_split(
        range(len(clean_df)),
        test_size=0.20,
        random_state=42,
        stratify=clean_df[TARGET_COLUMN],
    )

    X_train = X.iloc[list(train_indices)]
    y_train = y.iloc[list(train_indices)]

    # ------------------------------------------------------
    # Model
    # ------------------------------------------------------

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42,
    )

    # ------------------------------------------------------
    # Train
    # ------------------------------------------------------

    model.fit(
        X_train,
        y_train,
    )

    # ------------------------------------------------------
    # Evaluate on CLEAN test data
    # ------------------------------------------------------

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    precision = precision_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0,
    )

    # ------------------------------------------------------
    # MLflow
    # ------------------------------------------------------

    with mlflow.start_run(
        run_name=run_name
    ):

        # Parameters

        mlflow.log_param(
            "poisoning_level",
            poisoning_level,
        )

        mlflow.log_param(
            "n_estimators",
            100,
        )

        mlflow.log_param(
            "max_depth",
            5,
        )

        mlflow.log_param(
            "random_state",
            42,
        )

        mlflow.log_param(
            "test_set",
            "clean",
        )

        # Metrics

        mlflow.log_metric(
            "accuracy",
            accuracy,
        )

        mlflow.log_metric(
            "precision",
            precision,
        )

        mlflow.log_metric(
            "recall",
            recall,
        )

        mlflow.log_metric(
            "f1_score",
            f1,
        )

        # Model

        mlflow.sklearn.log_model(
            model,
            name="model",
        )

    # ------------------------------------------------------
    # Console output
    # ------------------------------------------------------

    print(f"Poisoning Level : {poisoning_level}")
    print(f"Accuracy        : {accuracy:.4f}")
    print(f"Precision       : {precision:.4f}")
    print(f"Recall          : {recall:.4f}")
    print(f"F1 Score        : {f1:.4f}")

    return {
        "run": run_name,
        "poisoning_level": poisoning_level,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }


# ==========================================================
# Main
# ==========================================================

def main():

    results = []

    for run_name, dataset_path in DATASETS.items():

        if run_name == "Clean_0%":
            poisoning_level = "0%"
        elif run_name == "Poisoned_5%":
            poisoning_level = "5%"
        elif run_name == "Poisoned_10%":
            poisoning_level = "10%"
        else:
            poisoning_level = "50%"

        result = train_and_log(
            run_name,
            dataset_path,
            poisoning_level,
        )

        results.append(result)

    # ------------------------------------------------------
    # Summary
    # ------------------------------------------------------

    results_df = pd.DataFrame(results)

    print("\n\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    print(
        results_df.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()