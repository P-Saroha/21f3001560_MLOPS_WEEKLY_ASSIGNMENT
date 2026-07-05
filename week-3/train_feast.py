import pandas as pd
import joblib

from feast import FeatureStore

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score


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

# Convert timestamp column
raw_df["event_timestamp"] = pd.to_datetime(raw_df["event_timestamp"])

# ==========================================================
# Entity DataFrame
# ==========================================================

entity_df = raw_df[["iris_id", "event_timestamp"]]

# ==========================================================
# Retrieve Historical Features from Feast
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
# Encode Target Labels
# ==========================================================

label_encoder = LabelEncoder()

training_df["species"] = label_encoder.fit_transform(
    training_df["species"]
)

# Save Label Encoder
joblib.dump(label_encoder, "label_encoder.pkl")

# ==========================================================
# Prepare Training Data
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
# Train Model
# ==========================================================

model = RandomForestClassifier(
    random_state=42
)

model.fit(X, y)

predictions = model.predict(X)

accuracy = accuracy_score(y, predictions)

print("\nTraining Accuracy:", accuracy)

# ==========================================================
# Save Model
# ==========================================================

joblib.dump(model, "iris_model.pkl")

print("\nModel saved as iris_model.pkl")
print("Label Encoder saved as label_encoder.pkl")