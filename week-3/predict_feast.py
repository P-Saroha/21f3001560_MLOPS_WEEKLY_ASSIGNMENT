import pandas as pd
import joblib

from feast import FeatureStore

# ==========================================================
# Load Model and Label Encoder
# ==========================================================

model = joblib.load("iris_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# ==========================================================
# Connect to Feast Repository
# ==========================================================

store = FeatureStore(
    repo_path="feature_repo/feature_repo"
)

# ==========================================================
# Read Original Dataset
# ==========================================================

raw_df = pd.read_csv(
    "feature_repo/feature_repo/data/iris_data_adapted_for_feast.csv"
)

# ----------------------------------------------------------
# Choose one entity for inference
# ----------------------------------------------------------

sample = raw_df.iloc[0]

iris_id = int(sample["iris_id"])

# ==========================================================
# Retrieve Online Features from Feast
# ==========================================================

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

print("=" * 50)
print(f"Iris ID : {iris_id}")
print("=" * 50)

print("\nRetrieved Features From Feast\n")
print(online_features)

# ==========================================================
# Prepare Input for Prediction
# ==========================================================

X = pd.DataFrame(
    {
        "sepal_length": [online_features["sepal_length"][0]],
        "sepal_width": [online_features["sepal_width"][0]],
        "petal_length": [online_features["petal_length"][0]],
        "petal_width": [online_features["petal_width"][0]],
    }
)

# ==========================================================
# Prediction
# ==========================================================

prediction = model.predict(X)[0]

predicted_species = label_encoder.inverse_transform([prediction])[0]

actual_species = sample["species"]

# ==========================================================
# Results
# ==========================================================

print("\nPrediction using Feast Features")
print(predicted_species)

print("\nActual Species from Dataset")
print(actual_species)

print("\nPrediction Match:", predicted_species == actual_species)