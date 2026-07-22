from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from feast import FeatureStore

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

DATA_PATH = (
    BASE_DIR
    / "feature_repo"
    / "feature_repo"
    / "data"
    / "iris_data_adapted_for_feast.csv"
)

MODEL_PATH = BASE_DIR / "iris_model.pkl"

ENCODER_PATH = BASE_DIR / "label_encoder.pkl"

FEATURE_REPO = BASE_DIR / "feature_repo" / "feature_repo"

# ==========================================================
# MLflow Configuration
# ==========================================================

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("iris_feast_experiments")


def train_model():

    # ==========================================================
    # Connect to Feast
    # ==========================================================

    store = FeatureStore(repo_path=str(FEATURE_REPO))

    # ==========================================================
    # Load Dataset
    # ==========================================================

    raw_df = pd.read_csv(DATA_PATH)

    raw_df["event_timestamp"] = pd.to_datetime(
        raw_df["event_timestamp"]
    )

    entity_df = raw_df[
        [
            "iris_id",
            "event_timestamp",
        ]
    ]

    # ==========================================================
    # Retrieve Historical Features
    # ==========================================================

    training_df = store.get_historical_features(
        entity_df=entity_df,
        features=[
            "iris_features:sepal_length",
            "iris_features:sepal_width",
            "iris_features:petal_length",
            "iris_features:petal_width",
            "iris_features:species",
        ],
    ).to_df()

    print("\nHistorical Features Retrieved From Feast\n")
    print(training_df.head())

    # ==========================================================
    # Encode Labels
    # ==========================================================

    label_encoder = LabelEncoder()

    training_df["species"] = label_encoder.fit_transform(
        training_df["species"]
    )

    joblib.dump(label_encoder, ENCODER_PATH)

    # ==========================================================
    # Features & Labels
    # ==========================================================

    X = training_df[
        [
            "sepal_length",
            "sepal_width",
            "petal_length",
            "petal_width",
        ]
    ]

    y = training_df["species"]

    # ==========================================================
    # Train/Test Split
    # ==========================================================

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    # ==========================================================
    # Hyperparameter Grid
    # ==========================================================

    max_depth_values = [3, 5, 10]
    n_estimators_values = [50, 100]

    best_accuracy = 0.0
    best_model = None
    best_params = None

    experiment = 1

    # ==========================================================
    # Hyperparameter Tuning + MLflow Logging
    # ==========================================================

    for max_depth in max_depth_values:

        for n_estimators in n_estimators_values:

            with mlflow.start_run(run_name=f"Experiment_{experiment}"):

                print("\n" + "=" * 60)
                print(f"Experiment {experiment}")

                model = RandomForestClassifier(
                    max_depth=max_depth,
                    n_estimators=n_estimators,
                    random_state=42,
                )

                model.fit(X_train, y_train)

                predictions = model.predict(X_test)

                # --------------------------------------
                # Metrics
                # --------------------------------------

                accuracy = accuracy_score(
                    y_test,
                    predictions,
                )

                precision = precision_score(
                    y_test,
                    predictions,
                    average="weighted",
                )

                recall = recall_score(
                    y_test,
                    predictions,
                    average="weighted",
                )

                f1 = f1_score(
                    y_test,
                    predictions,
                    average="weighted",
                )

                # --------------------------------------
                # Console Output
                # --------------------------------------

                print(f"max_depth      : {max_depth}")
                print(f"n_estimators   : {n_estimators}")
                print(f"Accuracy       : {accuracy:.4f}")
                print(f"Precision      : {precision:.4f}")
                print(f"Recall         : {recall:.4f}")
                print(f"F1 Score       : {f1:.4f}")

                # --------------------------------------
                # MLflow Parameters
                # --------------------------------------

                mlflow.log_param(
                    "max_depth",
                    max_depth,
                )

                mlflow.log_param(
                    "n_estimators",
                    n_estimators,
                )

                mlflow.log_param(
                    "random_state",
                    42,
                )

                # --------------------------------------
                # MLflow Metrics
                # --------------------------------------

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

                # --------------------------------------
                # Log Model
                # --------------------------------------

                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path="model",
                    registered_model_name="iris_random_forest",
                )

                # --------------------------------------
                # Best Model Selection
                # --------------------------------------

                if accuracy > best_accuracy:

                    best_accuracy = accuracy
                    best_model = model

                    best_params = {
                        "max_depth": max_depth,
                        "n_estimators": n_estimators,
                    }

                experiment += 1

    # ==========================================================
    # Save Best Model
    # ==========================================================

    joblib.dump(
        best_model,
        MODEL_PATH,
    )

    print("\n" + "=" * 60)
    print("Best Model Selected")
    print(f"Accuracy        : {best_accuracy:.4f}")
    print(f"Parameters      : {best_params}")
    print("=" * 60)

    print("\nBest model saved successfully.")
    print("Label encoder saved successfully.")

    return best_accuracy


def main():

    train_model()


if __name__ == "__main__":
    main()