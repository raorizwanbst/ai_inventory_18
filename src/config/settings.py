import os
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    azure_openai_api_key: str = os.getenv("AZURE_OPENAI_API_KEY", "")
    azure_openai_endpoint: str = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    model_name: str = os.getenv("MODEL_NAME", "gpt-4o-2024-08-06")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
    pinecone_api_key: str = os.getenv("PINECONE_API_KEY", "")
    vector_store_uri: str = os.getenv("VECTOR_STORE_URI", "")
    mcp_server_url: str = os.getenv("MCP_SERVER_URL", "http://localhost:8080/mcp")
    a2a_agent_endpoint: str = os.getenv("A2A_AGENT_ENDPOINT", "")
    mlflow_tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    wandb_project: str = "enterprise-rag-agent"
    feast_registry: str = os.getenv("FEAST_REGISTRY_PATH", "data/features/registry.pb")
    dvc_remote: str = os.getenv("DVC_REMOTE", "")
    langfuse_public_key: str = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    langfuse_secret_key: str = os.getenv("LANGFUSE_SECRET_KEY", "")
    huggingface_token: str = os.getenv("HUGGINGFACE_TOKEN", "")
    guardrail_config_path: str = "src/guardrails/rails_config.yml"
    skill_manifest_path: str = "src/skills/manifest.json"
    knowledge_base_path: str = "data/knowledge/corporate_kb"
    feature_store_project: str = "contoso_features"
    experiment_name: str = "rag-finetune-v2"
    model_registry_uri: str = "models:/rag-classifier/Production"
    data_version: str = "v1.4.2"
    training_dataset: str = "data/datasets/customer_support_v3.parquet"
    hyperparams_path: str = "src/ml/hyperparams.yaml"

settings = Settings()
