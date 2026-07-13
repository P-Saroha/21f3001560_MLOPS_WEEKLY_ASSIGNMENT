from pathlib import Path

import joblib

BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_PATH = BASE_DIR / "week-3" / "iris_model.pkl"

ENCODER_PATH = BASE_DIR / "week-3" / "label_encoder.pkl"


def test_model_exists():

    assert MODEL_PATH.exists()


def test_label_encoder_exists():

    assert ENCODER_PATH.exists()


def test_model_load():

    model = joblib.load(MODEL_PATH)

    assert model is not None


def test_label_encoder_load():

    encoder = joblib.load(ENCODER_PATH)

    assert encoder is not None


def test_model_predict_method():

    model = joblib.load(MODEL_PATH)

    assert hasattr(model, "predict")