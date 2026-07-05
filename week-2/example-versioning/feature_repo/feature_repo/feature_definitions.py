from datetime import timedelta

from feast import Entity, FeatureView, FileSource
from feast.types import Float32, Int64
from feast import Field

# ----------------------------
# Entity
# ----------------------------
iris = Entity(
    name="sample_id",
    join_keys=["sample_id"],
)
# ----------------------------
# Data Source
# ----------------------------
iris_source = FileSource(
    path="data/iris_features.parquet",
    timestamp_field="event_timestamp",
)

# ----------------------------
# Feature View
# ----------------------------
iris_feature_view = FeatureView(
    name="iris_features",
    entities=[iris],
    ttl=timedelta(days=3650),
    schema=[
    Field(name="sepal length (cm)", dtype=Float32),
    Field(name="sepal width (cm)", dtype=Float32),
    Field(name="petal length (cm)", dtype=Float32),
    Field(name="petal width (cm)", dtype=Float32),
    ],
    source=iris_source,
)
