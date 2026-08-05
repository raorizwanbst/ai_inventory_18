from prefect import flow, task
from prefect.task_runners import ConcurrentTaskRunner
import mlflow
from src.ml.train import run_training, load_hyperparams
from src.rag.retriever import KnowledgeBase
from src.config.settings import settings
from datasets import load_dataset
import dvc.api

@task
def load_versioned_dataset(version: str = "v1.4.2"):
    path = dvc.api.get_url(
        "data/datasets/customer_support_v3.parquet",
        repo=".",
        rev=version,
    )
    return load_dataset("parquet", data_files=path)

@task
def validate_data(dataset):
    assert len(dataset["train"]) > 1000, "Training set too small"
    return True

@task
def train_model():
    return run_training()

@task
def evaluate_and_register(run_id: str):
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = mlflow.tracking.MlflowClient()
    client.transition_model_version_stage(
        name="rag-classifier",
        version=client.get_latest_versions("rag-classifier")[0].version,
        stage="Staging",
    )
    return "Staging"

@task
def rebuild_vector_index():
    kb = KnowledgeBase()
    return "index_rebuilt"

@flow(name="rag-finetune-pipeline", task_runner=ConcurrentTaskRunner())
def ml_pipeline(data_version: str = "v1.4.2"):
    dataset = load_versioned_dataset(data_version)
    validate_data(dataset)
    result = train_model()
    stage = evaluate_and_register(result)
    rebuild_vector_index()
    return {"status": "success", "stage": stage}

if __name__ == "__main__":
    ml_pipeline()
