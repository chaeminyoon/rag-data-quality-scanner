"""
Vector store backends.

Use get_vector_store() to obtain the configured backend.
PineconeClient is imported lazily so the pinecone package (and an API key)
is only required when VECTOR_BACKEND="pinecone".
"""

from typing import Optional

from config import get_settings

from .base import VectorStore
from .local_store import LocalVectorStore


def get_vector_store(backend: Optional[str] = None) -> VectorStore:
    """
    Factory for the configured vector store backend.

    Args:
        backend: "local" or "pinecone" (defaults to settings.VECTOR_BACKEND)

    Returns:
        VectorStore instance
    """
    backend = backend or get_settings().VECTOR_BACKEND

    if backend == "pinecone":
        from .pinecone_client import PineconeClient

        return PineconeClient()
    return LocalVectorStore()


def __getattr__(name):
    # Lazy re-export for backward compatibility: `from src.vectordb import PineconeClient`
    if name == "PineconeClient":
        from .pinecone_client import PineconeClient

        return PineconeClient
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "VectorStore",
    "LocalVectorStore",
    "PineconeClient",
    "get_vector_store",
]
