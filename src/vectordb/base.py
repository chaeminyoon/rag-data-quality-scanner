"""
Vector store interface.

All vector search backends (in-memory local store, Pinecone, ...)
implement this interface. Result format matches the previous Pinecone
client so existing callers keep working:
    {"id": str, "score": float, "metadata": dict | None}
"""

from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional

import numpy as np


class VectorStore(ABC):
    """Backend-agnostic vector store with namespace support."""

    @abstractmethod
    def create_index(self, dimension: int) -> None:
        """Create (or validate) the index for the given dimension."""

    @abstractmethod
    def upsert_documents(
        self,
        documents: List[Dict],
        embeddings: np.ndarray,
        namespace: str = "default",
        text_key: str = "text",
        id_key: str = "id",
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> None:
        """Insert or update documents with their embeddings."""

    @abstractmethod
    def query(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        namespace: str = "default",
    ) -> List[Dict]:
        """Return top-k matches as {"id", "score", "metadata"} dicts."""

    @abstractmethod
    def delete_namespace(self, namespace: str) -> None:
        """Delete all vectors in a namespace (no-op if absent)."""
