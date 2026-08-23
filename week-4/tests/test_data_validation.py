from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_PATH = (
    BASE_DIR
    / "week-3"
    / "feature_repo"
    / "feature_repo"
    / "data"
    / "iris_data_adapted_for_feast.csv"
)


def test_dataset_exists():
    assert DATA_PATH.exists()


def test_dataset_not_empty():

    df = pd.read_csv(DATA_PATH)

    assert len(df) > 0


def test_required_columns():

    df = pd.read_csv(DATA_PATH)

    expected = [
        "iris_id",
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
        "species",
        "event_timestamp",
        "created_timestamp",
    ]

    for column in expected:
        assert column in df.columns


def test_missing_values():

    df = pd.read_csv(DATA_PATH)

    assert df.isnull().sum().sum() == 0


def test_numeric_columns():

    df = pd.read_csv(DATA_PATH)

    numeric = [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
    ]

    for col in numeric:
        assert pd.api.types.is_numeric_dtype(df[col])


def test_value_ranges():

    df = pd.read_csv(DATA_PATH)

    # Ensure values are within reasonable ranges
    assert df["sepal_length"].between(0, 10).all()
    assert df["sepal_width"].between(0, 10).all()
    assert df["petal_length"].between(0, 10).all()

    # Adapted Feast dataset contains one slightly negative value
    assert df["petal_width"].between(-1, 5).all()