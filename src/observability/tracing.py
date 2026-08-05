from langfuse import Langfuse
from langfuse.callback import CallbackHandler
from src.config.settings import settings
import os

langfuse = Langfuse(
    public_key=settings.langfuse_public_key,
    secret_key=settings.langfuse_secret_key,
    host="https://cloud.langfuse.com",
)

def get_langfuse_handler() -> CallbackHandler:
    return CallbackHandler(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        host="https://cloud.langfuse.com",
    )

def log_generation(name: str, input_text: str, output_text: str, model: str, metadata: dict | None = None):
    generation = langfuse.generation(
        name=name,
        model=model,
        input=input_text,
        output=output_text,
        metadata=metadata or {},
    )
    generation.end()
    return generation

def log_span(name: str, metadata: dict | None = None):
    return langfuse.span(name=name, metadata=metadata or {})
