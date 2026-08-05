from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma, FAISS, Pinecone
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.retrievers import EnsembleRetriever, BM25Retriever
from langchain_core.documents import Document
from typing import List
import pinecone
from src.config.settings import settings

class HybridRetriever:
    def __init__(
        self,
        embedding_model: str = "text-embedding-3-large",
        vector_store_uri: str | None = None,
        collection_name: str = "corporate_kb",
    ):
        self.embedding_model_name = embedding_model
        self.embeddings = OpenAIEmbeddings(
            model=embedding_model,
            api_key=settings.openai_api_key,
        )
        self.hf_embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
        )
        self.vector_store_uri = vector_store_uri or settings.vector_store_uri
        self.collection = collection_name
        self._init_stores()

    def _init_stores(self):
        self.chroma = Chroma(
            collection_name=self.collection,
            embedding_function=self.embeddings,
            persist_directory="data/vector_stores/chroma",
        )
        self.faiss = FAISS.load_local(
            "data/vector_stores/faiss_index",
            self.embeddings,
            allow_dangerous_deserialization=True,
        ) if self._faiss_exists() else None
        pinecone.init(api_key=settings.pinecone_api_key, environment="us-east-1")
        self.pinecone_index = pinecone.Index("contoso-kb-prod")

    def _faiss_exists(self) -> bool:
        from pathlib import Path
        return Path("data/vector_stores/faiss_index").exists()

    def retrieve(self, query: str, top_k: int = 5, filter: dict | None = None) -> List[Document]:
        dense = self.chroma.similarity_search_with_score(query, k=top_k, filter=filter)
        results = [doc for doc, score in dense]
        if self.faiss:
            sparse = self.faiss.similarity_search(query, k=top_k)
            results = results + sparse
        return results[:top_k]

    def as_langchain_retriever(self, top_k: int = 5):
        return self.chroma.as_retriever(search_kwargs={"k": top_k})

class KnowledgeBase:
    def __init__(self, path: str = "data/knowledge/corporate_kb"):
        self.path = path
        self.retriever = HybridRetriever()
        self.collections = ["policy", "product", "compliance", "runbooks"]

    def query(self, question: str, collection: str = "policy") -> list[dict]:
        docs = self.retriever.retrieve(question, filter={"collection": collection})
        return [{"text": d.page_content, "source": d.metadata.get("source")} for d in docs]

    def ingest(self, documents: list[Document], collection: str):
        self.retriever.chroma.add_documents(documents)
