# Enterprise RAG Agent Platform (AIBOM Sample)

Dummy but realistic multi-agent RAG platform used to exercise Cisco AI BOM scanners.

## Covered AIBOM component types

| Type | Where it appears |
|------|------------------|
| model | src/ml/train.py, src/agents/*.py, hyperparams.yaml (Llama-3.1-8B, gpt-4o) |
| llm_endpoint | src/api/endpoints.py, settings, .env (OpenAI, Anthropic, Azure) |
| model_endpoint | src/api/endpoints.py, infra/terraform (SageMaker, vLLM) |
| agent | src/agents/customer_support_agent.py, research_agent.py |
| agent_proxy | src/agents/agent_proxy.py (A2A remote proxies) |
| tool | src/tools/*.py (CRM, search, code interpreter, document loader) |
| mcp_server | src/mcp/server.py |
| mcp_client | src/mcp/client.py |
| mcp_gateway | src/mcp/client.py (MCPGateway) |
| embedding | src/rag/embeddings.py |
| vector_store | src/rag/retriever.py (Chroma, FAISS, Pinecone) |
| dataset | data/datasets/, dvc.yaml, train.py |
| retriever | src/rag/retriever.py (HybridRetriever) |
| knowledge_base | src/rag/retriever.py (KnowledgeBase class) |
| feature_store | src/ml/feature_store.py, data/features/ |
| memory | src/memory/conversation_store.py |
| prompt | src/prompts/templates.py |
| training_run | src/ml/train.py (mlflow.start_run) |
| hyperparameter | src/ml/hyperparams.yaml |
| model_artifact | data/artifacts/, model_card.md |
| experiment_tracker | src/ml/train.py, notebooks/ (MLflow + W&B) |
| model_registry | src/ml/train.py, pipeline.py (mlflow.register) |
| data_versioning | data/dvc.yaml, pipeline.py (dvc.api) |
| ml_pipeline | src/ml/pipeline.py (Prefect flow) |
| guardrail | src/guardrails/ (NeMo + Guardrails AI) |
| skill | src/skills/manifest.json, ticket_triage.py |
| observability | src/observability/tracing.py (Langfuse) |
| secret | .env, k8s secrets, settings.py |
| dependency | requirements.txt, pyproject.toml |

## Quick scan

```bash
cisco-aibom analyze . -o json -O report.json --llm-model gpt-4o --llm-api-key $OPENAI_API_KEY
```
