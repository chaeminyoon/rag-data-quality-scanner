"""
Embedding providers.

Use get_embedding_provider() to obtain the configured backend.
CohereClient is imported lazily so the cohere package (and an API key)
is only required when EMBEDDING_BACKEND="cohere".
"""

from typing import Optional

from config import get_settings

from .base import EmbeddingProvider
from .local_client import LocalEmbeddingClient


def get_embedding_provider(backend: Optional[str] = None) -> EmbeddingProvider:
    """
    Factory for the configured embedding backend.

    Args:
        backend: "local" or "cohere" (defaults to settings.EMBEDDING_BACKEND)

    Returns:
        EmbeddingProvider instance
    """
    backend = backend or get_settings().EMBEDDING_BACKEND

    if backend == "cohere":
        from .cohere_client import CohereClient

        return CohereClient()
    return LocalEmbeddingClient()


def __getattr__(name):
    # Lazy re-export for backward compatibility: `from src.embeddings import CohereClient`
    if name == "CohereClient":
        from .cohere_client import CohereClient

        return CohereClient
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "EmbeddingProvider",
    "LocalEmbeddingClient",
    "CohereClient",
    "get_embedding_provider",
]
