import joblib
import pandas as pd

from feast import FeatureStore

# -----------------------------
# Load trained model
# -----------------------------
model = joblib.load("iris_model.pkl")

# -----------------------------
# Connect to Feast
# -----------------------------
store = FeatureStore(
    repo_path="feature_repo/feature_repo"
)

# -----------------------------
# Fetch features from Feast Online Store
# -----------------------------
sample_id = 0

features = store.get_online_features(
    features=[
        "iris_features:sepal length (cm)",
        "iris_features:sepal width (cm)",
        "iris_features:petal length (cm)",
        "iris_features:petal width (cm)",
    ],
    entity_rows=[
        {"sample_id": sample_id},
    ],
).to_dict()

# Create dataframe for prediction
feast_df = pd.DataFrame({
    "sepal length (cm)": [features["sepal length (cm)"][0]],
    "sepal width (cm)": [features["sepal width (cm)"][0]],
    "petal length (cm)": [features["petal length (cm)"][0]],
    "petal width (cm)": [features["petal width (cm)"][0]],
})

# Prediction using Feast features
feast_prediction = model.predict(feast_df)

# -----------------------------
# Load raw data
# -----------------------------
raw_df = pd.read_parquet(
    "feature_repo/feature_repo/data/iris_features.parquet"
)

raw_sample = raw_df[raw_df["sample_id"] == sample_id]

raw_X = raw_sample[
    [
        "sepal length (cm)",
        "sepal width (cm)",
        "petal length (cm)",
        "petal width (cm)",
    ]
]

# Prediction using raw data
raw_prediction = model.predict(raw_X)

species = {
    0: "Setosa",
    1: "Versicolor",
    2: "Virginica",
}

print("=" * 50)
print("Sample ID:", sample_id)
print("=" * 50)

print("\nPrediction using Feast Features:")
print(species[int(feast_prediction[0])])

print("\nPrediction using Raw Data:")
print(species[int(raw_prediction[0])])

print("\nPredictions Match:", feast_prediction[0] == raw_prediction[0])