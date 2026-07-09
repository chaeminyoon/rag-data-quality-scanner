"""
Local embedding provider using sentence-transformers.

Runs fully offline (after the first model download) — no API keys required.
Default model is multilingual (handles Korean + English), unlike the previous
Cohere setup which used an English-only model on Korean sample data.
"""

from typing import Callable, List, Optional

import numpy as np

from config import get_logger, get_settings
from .base import EmbeddingProvider

logger = get_logger("embeddings.local_client")

# E5-family models are trained with instruction prefixes; omitting them
# measurably degrades retrieval quality.
_E5_PREFIXES = {"search_document": "passage: ", "search_query": "query: "}


class LocalEmbeddingClient(EmbeddingProvider):
    """
    sentence-transformers embedding provider.

    Features:
    - Multilingual by default (intfloat/multilingual-e5-small)
    - L2-normalized embeddings (cosine == dot product downstream)
    - Automatic device selection (CUDA / Apple MPS / CPU)
    - Lazy model loading (first embed call)
    """

    BATCH_SIZE = 64

    def __init__(self, model_name: Optional[str] = None):
        settings = get_settings()
        self.model_name = model_name or settings.LOCAL_EMBED_MODEL
        self.name = f"local:{self.model_name}"
        self._model = None  # lazy-loaded
        self._is_e5 = "e5" in self.model_name.lower()

    @property
    def model(self):
        if self._model is None:
            # Import here so the package is only required when this backend is used
            from sentence_transformers import SentenceTransformer

            logger.info(f"Loading local embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def _apply_prefix(self, texts: List[str], input_type: str) -> List[str]:
        if not self._is_e5:
            return texts
        prefix = _E5_PREFIXES.get(input_type, "")
        return [prefix + t for t in texts]

    def embed_documents(
        self,
        texts: List[str],
        input_type: str = "search_document",
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> np.ndarray:
        if not texts:
            return np.array([])

        prefixed = self._apply_prefix(texts, input_type)

        all_embeddings = []
        for start in range(0, len(prefixed), self.BATCH_SIZE):
            batch = prefixed[start : start + self.BATCH_SIZE]
            embeddings = self.model.encode(
                batch,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            all_embeddings.append(embeddings)

            if progress_callback:
                progress_callback(min(1.0, (start + len(batch)) / len(prefixed)))

        return np.vstack(all_embeddings)

    def get_embedding_dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    @property
    def recommended_duplicate_threshold(self) -> float:
        # e5-family similarity is compressed into a high range (unrelated
        # passages ≈ 0.80): true near-dups sit at ≈0.99, paraphrases ≈0.93.
        # Measured on the controlled eval corpus — see docs/RESEARCH_NOTES.md.
        if self._is_e5:
            return 0.985
        return 0.95

    def validate_connection(self) -> bool:
        """Local models have no connection; verify the model loads."""
        try:
            _ = self.model
            return True
        except Exception as e:
            logger.error(f"Failed to load local embedding model: {e}")
            return False
