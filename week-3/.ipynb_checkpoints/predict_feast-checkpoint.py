from pathlib import Path

import pandas as pd
import joblib

from feast import FeatureStore

# ==========================================================
# Paths
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "iris_model.pkl"

ENCODER_PATH = BASE_DIR / "label_encoder.pkl"

DATA_PATH = BASE_DIR / "feature_repo" / "feature_repo" / "data" / "iris_data_adapted_for_feast.csv"

FEATURE_REPO = BASE_DIR / "feature_repo" / "feature_repo"


def predict_sample():

    model = joblib.load(MODEL_PATH)

    label_encoder = joblib.load(
        ENCODER_PATH
    )

    store = FeatureStore(
        repo_path=str(FEATURE_REPO)
    )

    raw_df = pd.read_csv(DATA_PATH)

    sample = raw_df.iloc[0]

    iris_id = int(sample["iris_id"])

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

    prediction = model.predict(X)[0]

    predicted_species = label_encoder.inverse_transform(
        [prediction]
    )[0]

    actual_species = sample["species"]

    return predicted_species, actual_species


def main():

    predicted, actual = predict_sample()

    print("=" * 50)

    print("Prediction :", predicted)

    print("Actual     :", actual)

    print("Match      :", predicted == actual)

    print("=" * 50)


if __name__ == "__main__":
    main()