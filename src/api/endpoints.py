from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.agents.customer_support_agent import CustomerSupportAgent
from src.agents.research_agent import ResearchAgent
from src.agents.agent_proxy import remote_research_proxy
from src.config.settings import settings
import httpx

app = FastAPI(title="Contoso RAG Agent Platform", version="2.4.1")

support_agent = CustomerSupportAgent()
research_agent = ResearchAgent()

class ChatRequest(BaseModel):
    user_id: str
    message: str

class ChatResponse(BaseModel):
    response: str
    sources: list = []

@app.post("/v1/chat/support", response_model=ChatResponse)
async def chat_support(req: ChatRequest):
    result = await support_agent.handle(req.user_id, req.message)
    return ChatResponse(**result)

@app.post("/v1/research")
async def research(query: str):
    return research_agent.run(query)

@app.post("/v1/proxy/research")
async def proxy_research(task: str):
    return await remote_research_proxy.invoke(task)

@app.get("/v1/models")
async def list_models():
    return {
        "llm_endpoints": [
            {"name": "gpt-4o-2024-08-06", "provider": "openai", "type": "llm_endpoint"},
            {"name": "claude-3-5-sonnet-20241022", "provider": "anthropic", "type": "llm_endpoint"},
            {"name": "azure-gpt-4o", "provider": "azure", "endpoint": settings.azure_openai_endpoint, "type": "llm_endpoint"},
        ],
        "model_endpoints": [
            {"name": "rag-classifier-endpoint", "provider": "sagemaker", "type": "model_endpoint",
             "uri": "https://runtime.sagemaker.us-east-1.amazonaws.com/endpoints/rag-classifier-endpoint"},
            {"name": "local-vllm", "provider": "vllm", "uri": "http://localhost:8001/v1", "type": "model_endpoint"},
        ],
        "embeddings": [
            {"name": "text-embedding-3-large", "provider": "openai", "type": "embedding"},
            {"name": "all-MiniLM-L6-v2", "provider": "huggingface", "type": "embedding"},
        ],
    }

@app.get("/health")
async def health():
    return {"status": "ok", "model": settings.model_name}
