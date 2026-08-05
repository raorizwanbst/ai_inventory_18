from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings, BedrockEmbeddings
from sentence_transformers import SentenceTransformer
from src.config.settings import settings

openai_embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    dimensions=3072,
    api_key=settings.openai_api_key,
)

openai_small = OpenAIEmbeddings(
    model="text-embedding-3-small",
    dimensions=1536,
    api_key=settings.openai_api_key,
)

hf_minilm = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

hf_mpnet = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
)

local_e5 = SentenceTransformer("intfloat/e5-large-v2")

bedrock_embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v2:0",
    region_name="us-east-1",
)

EMBEDDING_REGISTRY = {
    "text-embedding-3-large": openai_embeddings,
    "text-embedding-3-small": openai_small,
    "all-MiniLM-L6-v2": hf_minilm,
    "all-mpnet-base-v2": hf_mpnet,
    "e5-large-v2": local_e5,
    "titan-embed-v2": bedrock_embeddings,
}

def get_embedding_model(name: str = "text-embedding-3-large"):
    return EMBEDDING_REGISTRY.get(name, openai_embeddings)
