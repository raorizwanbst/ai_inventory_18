from mcp.server.fastmcp import FastMCP
from mcp.types import Tool, TextContent
import httpx
from typing import Any

mcp = FastMCP("contoso-enterprise-tools")

@mcp.tool()
def get_weather(city: str, units: str = "metric") -> dict[str, Any]:
    """Fetch current weather for a city. Used by field-service agents."""
    resp = httpx.get(
        f"https://api.openweathermap.org/data/2.5/weather",
        params={"q": city, "units": units, "appid": "${OPENWEATHER_API_KEY}"},
        timeout=5.0,
    )
    data = resp.json()
    return {
        "city": city,
        "temp": data["main"]["temp"],
        "conditions": data["weather"][0]["description"],
        "humidity": data["main"]["humidity"],
    }

@mcp.tool()
def academic_search(query: str, limit: int = 10) -> dict[str, Any]:
    """Search academic papers via Semantic Scholar API."""
    resp = httpx.get(
        "https://api.semanticscholar.org/graph/v1/paper/search",
        params={"query": query, "limit": limit, "fields": "title,abstract,year,authors"},
        timeout=15.0,
    )
    return {"results": resp.json().get("data", [])}

@mcp.tool()
def internal_kb_query(query: str, collection: str = "policy") -> list[dict]:
    """Query the internal knowledge base collections: policy, product, compliance."""
    from src.rag.retriever import HybridRetriever
    from src.config.settings import settings
    retriever = HybridRetriever(
        embedding_model=settings.embedding_model,
        vector_store_uri=settings.vector_store_uri,
    )
    docs = retriever.retrieve(query, top_k=5, filter={"collection": collection})
    return [{"text": d.page_content, "meta": d.metadata} for d in docs]

@mcp.resource("config://agent-settings")
def get_agent_settings() -> str:
    return """{
        "max_iterations": 8,
        "temperature": 0.1,
        "model": "gpt-4o-2024-08-06",
        "guardrails_enabled": true
    }"""

if __name__ == "__main__":
    mcp.run(transport="sse", host="0.0.0.0", port=8080)
