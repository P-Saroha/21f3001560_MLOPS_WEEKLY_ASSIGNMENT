import os
import json
import joblib
from datetime import datetime

import pandas as pd

from google.cloud import storage

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


PROJECT_BUCKET = "21f3001560-iris-data"
ARTIFACT_BUCKET = "21f3001560-iris-artifacts"

DATA_VERSION = "v2"


def download_blob(bucket_name, source_blob_name, destination_file_name):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)


def upload_file(bucket_name, source_file_name, destination_blob_name):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)


timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

os.makedirs("artifacts", exist_ok=True)

local_data = "artifacts/data.csv"

download_blob(
    PROJECT_BUCKET,
    f"data/{DATA_VERSION}/data.csv",
    local_data
)

df = pd.read_csv(local_data)

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

model = DecisionTreeClassifier(max_depth=3)

model.fit(X_train, y_train)

preds = model.predict(X_test)

accuracy = accuracy_score(y_test, preds)

model_path = "artifacts/model.joblib"

joblib.dump(model, model_path)

metrics = {
    "accuracy": float(accuracy)
}

metrics_path = "artifacts/metrics.json"

with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=4)

upload_file(
    ARTIFACT_BUCKET,
    model_path,
    f"{timestamp}/model.joblib"
)

upload_file(
    ARTIFACT_BUCKET,
    metrics_path,
    f"{timestamp}/metrics.json"
)

print(f"Training Complete")
print(f"Accuracy: {accuracy}")
print(f"Timestamp: {timestamp}")