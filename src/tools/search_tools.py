from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from src.rag.retriever import HybridRetriever
from src.config.settings import settings

retriever = HybridRetriever(
    embedding_model=settings.embedding_model,
    vector_store_uri=settings.vector_store_uri,
)

@tool
def semantic_search(query: str, top_k: int = 5) -> list[dict]:
    """Perform semantic search over the corporate knowledge base and product docs."""
    results = retriever.retrieve(query, top_k=top_k)
    return [
        {
            "content": r.page_content,
            "score": r.metadata.get("score", 0.0),
            "source": r.metadata.get("source", "unknown"),
            "chunk_id": r.metadata.get("chunk_id"),
        }
        for r in results
    ]

@tool
def web_search(query: str) -> str:
    """Search the public web for up-to-date information using DuckDuckGo."""
    search = DuckDuckGoSearchRun()
    return search.run(query)

@tool
def product_catalog_search(sku: str | None = None, category: str | None = None) -> list[dict]:
    """Search the internal product catalog by SKU or category."""
    import httpx
    params = {}
    if sku:
        params["sku"] = sku
    if category:
        params["category"] = category
    resp = httpx.get(
        "https://catalog.internal.contoso.com/api/v1/products",
        params=params,
        timeout=8.0,
    )
    resp.raise_for_status()
    return resp.json().get("items", [])
