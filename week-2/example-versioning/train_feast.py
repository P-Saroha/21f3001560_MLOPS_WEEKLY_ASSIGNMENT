import pandas as pd
import joblib

from feast import FeatureStore
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Connect to Feast repository
store = FeatureStore(repo_path="feature_repo/feature_repo")

# Entity dataframe
entity_df = pd.DataFrame({
    "sample_id": range(150),
    "event_timestamp": pd.Timestamp("2024-01-01"),
})

# Fetch historical features
training_df = store.get_historical_features(
    entity_df=entity_df,
    features=[
        "iris_features:sepal length (cm)",
        "iris_features:sepal width (cm)",
        "iris_features:petal length (cm)",
        "iris_features:petal width (cm)",
    ],
).to_df()

print(training_df.head())

# Features
X = training_df[
    [
        "sepal length (cm)",
        "sepal width (cm)",
        "petal length (cm)",
        "petal width (cm)",
    ]
]

# Target
raw_df = pd.read_parquet(
    "feature_repo/feature_repo/data/iris_features.parquet"
)

y = raw_df["species"]

# Train model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X, y)

pred = model.predict(X)

print("Training Accuracy:", accuracy_score(y, pred))

# Save model
joblib.dump(model, "iris_model.pkl")

print("Model saved as iris_model.pkl")
