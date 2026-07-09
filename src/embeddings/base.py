"""
Embedding provider interface.

All embedding backends (local sentence-transformers, Cohere API, ...)
implement this interface so the rest of the pipeline is backend-agnostic.
"""

from abc import ABC, abstractmethod
from typing import Callable, List, Optional

import numpy as np


class EmbeddingProvider(ABC):
    """Backend-agnostic embedding interface."""

    #: Human-readable backend name (e.g. "local:intfloat/multilingual-e5-small")
    name: str = "unknown"

    @abstractmethod
    def embed_documents(
        self,
        texts: List[str],
        input_type: str = "search_document",
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> np.ndarray:
        """
        Embed documents for indexing.

        Args:
            texts: List of document texts
            input_type: "search_document" for indexing, "search_query" for queries
            progress_callback: Optional callback for progress updates (0.0 to 1.0)

        Returns:
            np.ndarray of shape (len(texts), dimension)
        """

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query for retrieval."""
        return self.embed_documents([query], input_type="search_query")[0]

    def embed_queries(
        self,
        queries: List[str],
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> np.ndarray:
        """Embed multiple queries for retrieval."""
        return self.embed_documents(
            queries, input_type="search_query", progress_callback=progress_callback
        )

    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """Return the embedding dimension of this provider."""

    @property
    def recommended_duplicate_threshold(self) -> float:
        """
        Cosine-similarity threshold above which two documents should be
        treated as near-duplicates FOR THIS MODEL.

        Similarity scales are not portable across embedding models: e.g.
        e5-family models map even unrelated passages to ~0.80 cosine, so a
        threshold calibrated for Cohere embed-v3 (0.92) catastrophically
        over-merges with e5 (measured: paraphrases ~0.93, hard distractors
        with different answers ~0.90, true near-dups ~0.99+).
        """
        return 0.92  # Cohere embed-v3 scale (historical default)
