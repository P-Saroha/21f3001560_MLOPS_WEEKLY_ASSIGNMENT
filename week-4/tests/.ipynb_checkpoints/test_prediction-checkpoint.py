from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parents[2]

sys.path.append(str(BASE_DIR / "week-3"))

from predict_feast import predict_sample


def test_prediction_matches():

    predicted, actual = predict_sample()

    assert predicted == actual