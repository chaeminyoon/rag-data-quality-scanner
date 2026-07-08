"""
In-memory vector store — exact cosine search with numpy.

No external service required. Exact (non-approximate) search is the right
default at data-quality-scanner scale (≤ ~100k docs): it removes ANN recall
as a confound when benchmarking data cleaning effects.
"""

from typing import Callable, Dict, List, Optional

import numpy as np

from config import get_logger
from .base import VectorStore

logger = get_logger("vectordb.local_store")


class LocalVectorStore(VectorStore):
    """
    Exact cosine-similarity search over normalized in-memory matrices.

    Namespaces isolate document sets (e.g. "original" vs "cleaned")
    exactly like Pinecone namespaces did.
    """

    def __init__(self):
        # namespace -> {"ids": [...], "matrix": np.ndarray, "metadata": [...]}
        self._namespaces: Dict[str, Dict] = {}
        self._dimension: Optional[int] = None

    def create_index(self, dimension: int) -> None:
        if self._dimension is not None and self._dimension != dimension:
            logger.warning(
                f"Dimension change {self._dimension} -> {dimension}; clearing store"
            )
            self._namespaces.clear()
        self._dimension = dimension

    @staticmethod
    def _normalize(matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms

    def upsert_documents(
        self,
        documents: List[Dict],
        embeddings: np.ndarray,
        namespace: str = "default",
        text_key: str = "text",
        id_key: str = "id",
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> None:
        if len(documents) != len(embeddings):
            raise ValueError("documents and embeddings count mismatch")

        embeddings = np.asarray(embeddings, dtype=np.float32)
        if self._dimension is None:
            self._dimension = embeddings.shape[1]

        ids = [doc.get(id_key, str(i)) for i, doc in enumerate(documents)]
        metadata = [
            {**doc.get("metadata", {}), text_key: doc.get(text_key, "")}
            for doc in documents
        ]

        ns = self._namespaces.setdefault(
            namespace, {"ids": [], "matrix": None, "metadata": []}
        )

        # Upsert semantics: replace existing ids, append new ones
        existing = {doc_id: idx for idx, doc_id in enumerate(ns["ids"])}
        normalized = self._normalize(embeddings)

        new_rows, new_ids, new_meta = [], [], []
        for doc_id, row, meta in zip(ids, normalized, metadata):
            if doc_id in existing:
                ns["matrix"][existing[doc_id]] = row
                ns["metadata"][existing[doc_id]] = meta
            else:
                new_rows.append(row)
                new_ids.append(doc_id)
                new_meta.append(meta)

        if new_rows:
            new_matrix = np.array(new_rows, dtype=np.float32)
            ns["matrix"] = (
                new_matrix
                if ns["matrix"] is None
                else np.vstack([ns["matrix"], new_matrix])
            )
            ns["ids"].extend(new_ids)
            ns["metadata"].extend(new_meta)

        if progress_callback:
            progress_callback(1.0)

        logger.info(
            f"Upserted {len(documents)} docs to namespace '{namespace}' "
            f"(total {len(ns['ids'])})"
        )

    def query(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        namespace: str = "default",
    ) -> List[Dict]:
        ns = self._namespaces.get(namespace)
        if not ns or ns["matrix"] is None or len(ns["ids"]) == 0:
            return []

        q = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
        q_norm = np.linalg.norm(q)
        if q_norm > 0:
            q = q / q_norm

        scores = ns["matrix"] @ q
        top_k = min(top_k, len(scores))
        top_idx = np.argpartition(-scores, top_k - 1)[:top_k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]

        return [
            {
                "id": ns["ids"][i],
                "score": float(scores[i]),
                "metadata": ns["metadata"][i],
            }
            for i in top_idx
        ]

    def query_batch(
        self,
        query_embeddings: np.ndarray,
        top_k: int = 10,
        namespace: str = "default",
    ) -> List[List[Dict]]:
        return [
            self.query(embedding, top_k=top_k, namespace=namespace)
            for embedding in query_embeddings
        ]

    def delete_namespace(self, namespace: str) -> None:
        self._namespaces.pop(namespace, None)

    def namespace_size(self, namespace: str) -> int:
        ns = self._namespaces.get(namespace)
        return len(ns["ids"]) if ns else 0
