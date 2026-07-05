from datetime import timedelta

from feast import Entity, FeatureView, FileSource, Field
from feast.types import Float32, String


# =========================================================
# Entity
# =========================================================

iris = Entity(
    name="iris_id",
    join_keys=["iris_id"],
)


# =========================================================
# Data Source
# =========================================================

iris_source = FileSource(
    path="data/iris_data_adapted_for_feast.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)


# =========================================================
# Feature View
# =========================================================

iris_feature_view = FeatureView(
    name="iris_features",
    entities=[iris],
    ttl=timedelta(days=365),
    schema=[
        Field(name="sepal_length", dtype=Float32),
        Field(name="sepal_width", dtype=Float32),
        Field(name="petal_length", dtype=Float32),
        Field(name="petal_width", dtype=Float32),
        Field(name="species", dtype=String),
    ],
    source=iris_source,
)