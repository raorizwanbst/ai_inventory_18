from langchain_core.tools import tool
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from typing import List

@tool
def load_pdf(path: str) -> list[str]:
    """Load and extract text from a local PDF file."""
    loader = PyPDFLoader(path)
    docs = loader.load()
    return [d.page_content for d in docs]

@tool
def load_url(url: str) -> str:
    """Fetch and extract main text content from a public URL."""
    loader = WebBaseLoader(url)
    docs = loader.load()
    return "\n\n".join(d.page_content for d in docs)
