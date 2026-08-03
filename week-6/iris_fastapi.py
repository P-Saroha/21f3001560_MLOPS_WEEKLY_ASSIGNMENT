from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Create FastAPI application
app = FastAPI(
    title="Iris Classification API",
    description="FastAPI service for predicting Iris flower species",
    version="1.0",
    root_path="/proxy/8200"
)

# Load trained model
model = joblib.load("iris_model.pkl")

# Load label encoder
label_encoder = joblib.load("label_encoder.pkl")


# Input schema
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


@app.get("/")
def home():
    return {
        "message": "IRIS Classification API is running successfully!"
    }


@app.post("/predict")
def predict(data: IrisInput):

    features = pd.DataFrame(
        [[
            data.sepal_length,
            data.sepal_width,
            data.petal_length,
            data.petal_width
        ]],
        columns=[
            "sepal_length",
            "sepal_width",
            "petal_length",
            "petal_width"
        ]
    )

    prediction = model.predict(features)

    species = label_encoder.inverse_transform(prediction)

    return {
        "prediction": species[0]
    }