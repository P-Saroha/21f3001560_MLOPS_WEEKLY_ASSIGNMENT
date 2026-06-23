import os
import joblib
import pandas as pd

from google.cloud import storage
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


DATA_BUCKET = "21f3001560-iris-data"
ARTIFACT_BUCKET = "21f3001560-iris-artifacts"


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


os.makedirs("artifacts", exist_ok=True)

download_blob(
    ARTIFACT_BUCKET,
    "latest_timestamp.txt",
    "artifacts/latest_timestamp.txt"
)

with open("artifacts/latest_timestamp.txt") as f:
    timestamp = f.read().strip()

download_blob(
    ARTIFACT_BUCKET,
    f"{timestamp}/model.joblib",
    "artifacts/model.joblib"
)

download_blob(
    DATA_BUCKET,
    "data/v1/data.csv",
    "artifacts/data.csv"
)

df = pd.read_csv("artifacts/data.csv")

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

model = joblib.load("artifacts/model.joblib")

preds = model.predict(X_test)

acc = accuracy_score(y_test, preds)

pred_df = pd.DataFrame({
    "actual": y_test.values,
    "prediction": preds
})

pred_file = "artifacts/predictions.csv"

pred_df.to_csv(pred_file, index=False)

upload_file(
    ARTIFACT_BUCKET,
    pred_file,
    f"{timestamp}/predictions.csv"
)

print(f"Inference Complete")
print(f"Accuracy = {acc}")
print(f"Predictions uploaded to {timestamp}/predictions.csv")