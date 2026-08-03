from pathlib import Path

import joblib
import pandas as pd
import mlflow.pyfunc

from mlflow import MlflowClient

from feast import FeatureStore


import mlflow
import mlflow.pyfunc
from mlflow import MlflowClient

# ==========================================================
# MLflow Configuration
# ==========================================================

TRACKING_URI = "sqlite:///mlflow.db"
MODEL_NAME = "iris_random_forest"

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_registry_uri(TRACKING_URI)

# ==========================================================
# Paths
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent

ENCODER_PATH = BASE_DIR / "label_encoder.pkl"

DATA_PATH = (
    BASE_DIR
    / "feature_repo"
    / "feature_repo"
    / "data"
    / "iris_data_adapted_for_feast.csv"
)

FEATURE_REPO = (
    BASE_DIR
    / "feature_repo"
    / "feature_repo"
)

# ==========================================================
# MLflow
# ==========================================================

TRACKING_URI = "sqlite:///mlflow.db"

MODEL_NAME = "iris_random_forest"


def load_latest_model():

    client = MlflowClient(
        tracking_uri=TRACKING_URI,
        registry_uri=TRACKING_URI,
    )

    versions = client.search_model_versions(
        f"name='{MODEL_NAME}'"
    )

    latest_version = max(
        versions,
        key=lambda v: int(v.version)
    )

    print("=" * 60)
    print("Loading Model From MLflow Registry")
    print(f"Model Name : {MODEL_NAME}")
    print(f"Version    : {latest_version.version}")
    print("=" * 60)

    model_uri = f"models:/{MODEL_NAME}/{latest_version.version}"

    model = mlflow.pyfunc.load_model(model_uri)

    return model


def predict_sample():

    model = load_latest_model()

    label_encoder = joblib.load(
        ENCODER_PATH
    )

    store = FeatureStore(
        repo_path=str(FEATURE_REPO)
    )

    raw_df = pd.read_csv(
        DATA_PATH
    )

    sample = raw_df.iloc[0]

    iris_id = int(
        sample["iris_id"]
    )

    online_features = store.get_online_features(

        features=[
            "iris_features:sepal_length",
            "iris_features:sepal_width",
            "iris_features:petal_length",
            "iris_features:petal_width",
        ],

        entity_rows=[
            {
                "iris_id": iris_id
            }
        ],

    ).to_dict()

    X = pd.DataFrame(
        {
            "sepal_length": [
                online_features["sepal_length"][0]
            ],
            "sepal_width": [
                online_features["sepal_width"][0]
            ],
            "petal_length": [
                online_features["petal_length"][0]
            ],
            "petal_width": [
                online_features["petal_width"][0]
            ],
        }
    )

    prediction = model.predict(
        X
    )[0]

    predicted_species = label_encoder.inverse_transform(
        [int(prediction)]
    )[0]

    actual_species = sample["species"]

    return predicted_species, actual_species


def main():

    predicted, actual = predict_sample()

    print("\n" + "=" * 60)

    print("Prediction :", predicted)

    print("Actual     :", actual)

    print("Match      :", predicted == actual)

    print("=" * 60)


if __name__ == "__main__":
    main()