FROM python:3.12-slim AS base

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY data/knowledge/ ./data/knowledge/
COPY src/guardrails/rails_config.yml ./src/guardrails/
COPY src/skills/manifest.json ./src/skills/

ENV MODEL_NAME=gpt-4o-2024-08-06
ENV EMBEDDING_MODEL=text-embedding-3-large
ENV MCP_SERVER_URL=http://mcp:8080/mcp
ENV MLFLOW_TRACKING_URI=http://mlflow:5000

EXPOSE 8000 8080

CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
