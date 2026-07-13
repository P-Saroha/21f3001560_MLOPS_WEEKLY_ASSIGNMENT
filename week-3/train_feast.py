from pathlib import Path

import pandas as pd
import joblib

from feast import FeatureStore

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# ==========================================================
# Paths
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "feature_repo" / "feature_repo" / "data" / "iris_data_adapted_for_feast.csv"

MODEL_PATH = BASE_DIR / "iris_model.pkl"

ENCODER_PATH = BASE_DIR / "label_encoder.pkl"

FEATURE_REPO = BASE_DIR / "feature_repo" / "feature_repo"


def train_model():

    # ==========================================================
    # Connect to Feast
    # ==========================================================

    store = FeatureStore(
        repo_path=str(FEATURE_REPO)
    )

    # ==========================================================
    # Read Dataset
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
    # Historical Features
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
    # Train
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

    model = RandomForestClassifier(
        random_state=42
    )

    model.fit(X, y)

    predictions = model.predict(X)

    accuracy = accuracy_score(
        y,
        predictions
    )

    print(f"\nTraining Accuracy : {accuracy:.4f}")

    joblib.dump(
        model,
        MODEL_PATH
    )

    print("\nModel saved.")
    print("Label encoder saved.")

    return accuracy


def main():
    train_model()


if __name__ == "__main__":
    main()