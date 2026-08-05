from feast import FeatureStore, Entity, FeatureView, Field, FileSource
from feast.types import Float32, Int64, String
from datetime import timedelta
from src.config.settings import settings

store = FeatureStore(repo_path="data/features")

customer_entity = Entity(name="customer_id", join_keys=["customer_id"])

customer_source = FileSource(
    path="data/features/customer_features.parquet",
    timestamp_field="event_timestamp",
)

customer_features = FeatureView(
    name="customer_features",
    entities=[customer_entity],
    ttl=timedelta(days=90),
    schema=[
        Field(name="lifetime_value", dtype=Float32),
        Field(name="ticket_count_30d", dtype=Int64),
        Field(name="tier", dtype=String),
        Field(name="churn_risk_score", dtype=Float32),
        Field(name="avg_response_time_hrs", dtype=Float32),
    ],
    source=customer_source,
    online=True,
)

def get_online_features(customer_ids: list[str]) -> dict:
    return store.get_online_features(
        features=[
            "customer_features:lifetime_value",
            "customer_features:ticket_count_30d",
            "customer_features:tier",
            "customer_features:churn_risk_score",
        ],
        entity_rows=[{"customer_id": cid} for cid in customer_ids],
    ).to_dict()

def materialize_incremental():
    store.materialize_incremental(end_date=None)
