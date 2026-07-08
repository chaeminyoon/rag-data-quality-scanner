"""
Rerankers: reorder initial retrieval results by query-document relevance.

Two backends implement the same interface:
- LocalReranker: sentence-transformers cross-encoder (offline, no API key)
- CohereReranker: Cohere Rerank API

Cross-encoder reranking is consistently reported as one of the highest-impact
single components in RAG retrieval pipelines.
"""

from abc import ABC, abstractmethod
from typing import List, Optional

from config import get_logger, get_settings

logger = get_logger("evaluator.reranker")


class BaseReranker(ABC):
    """Shared reranking interface and retrieval-result plumbing."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        documents: List[str],
        top_n: Optional[int] = None,
    ) -> List[dict]:
        """
        Rerank document texts for a query.

        Returns:
            List of dicts with 'index' (into the input list) and
            'relevance_score', sorted by descending relevance.
        """

    def rerank_retrieval_results(
        self,
        query: str,
        retrieval_results: List[dict],
        text_key: str = "text",
        id_key: str = "id",
        score_key: str = "score",
        top_n: Optional[int] = None,
    ) -> List[dict]:
        """
        Rerank results from vector retrieval.

        Args:
            query: Search query
            retrieval_results: Results from vector search
                ({"id", "score", "metadata": {"text": ...}} format)
            text_key: Key for document text
            id_key: Key for document ID
            score_key: Key for original retrieval score
            top_n: Number of results to return

        Returns:
            Reranked results with original and rerank scores
        """
        if not retrieval_results:
            return []

        # Get texts for reranking (metadata-nested or top-level)
        texts = []
        for result in retrieval_results:
            if "metadata" in result and result["metadata"] and text_key in result["metadata"]:
                texts.append(result["metadata"][text_key])
            elif text_key in result:
                texts.append(result[text_key])
            else:
                texts.append("")

        rerank_results = self.rerank(
            query=query,
            documents=texts,
            top_n=top_n or len(retrieval_results),
        )

        reranked = []
        for rr in rerank_results:
            original = retrieval_results[rr["index"]]

            doc_id = original.get(id_key)
            if doc_id is None and "metadata" in original:
                doc_id = original["metadata"].get(id_key, rr["index"])

            reranked.append({
                "id": doc_id,
                "text": texts[rr["index"]],
                "original_score": original.get(score_key, 0.0),
                "rerank_score": rr["relevance_score"],
                "metadata": original.get("metadata", {}),
            })

        return reranked


class LocalReranker(BaseReranker):
    """
    Cross-encoder reranker via sentence-transformers (offline).

    Default model is multilingual (mMARCO-trained), suitable for
    Korean + English corpora.
    """

    def __init__(self, model_name: Optional[str] = None):
        settings = get_settings()
        self.model_name = model_name or settings.LOCAL_RERANK_MODEL
        self._model = None  # lazy-loaded
        logger.info(f"Initialized LocalReranker with model={self.model_name}")

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            logger.info(f"Loading local rerank model: {self.model_name}")
            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_n: Optional[int] = None,
    ) -> List[dict]:
        if not documents:
            return []

        scores = self.model.predict([(query, doc) for doc in documents])
        order = sorted(range(len(documents)), key=lambda i: -float(scores[i]))
        top_n = top_n or len(documents)

        return [
            {"index": i, "relevance_score": float(scores[i])}
            for i in order[:top_n]
        ]


class CohereReranker(BaseReranker):
    """Cohere Rerank API backend (requires COHERE_API_KEY)."""

    def __init__(self, cohere_client=None):
        if cohere_client is None:
            from src.embeddings.cohere_client import CohereClient

            cohere_client = CohereClient()
        self.cohere = cohere_client
        logger.info("Initialized CohereReranker")

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_n: Optional[int] = None,
    ) -> List[dict]:
        return self.cohere.rerank(query=query, documents=documents, top_n=top_n)

    def rerank_with_ids(
        self,
        query: str,
        documents: List[dict],
        text_key: str = "text",
        id_key: str = "id",
        top_n: Optional[int] = None,
    ) -> List[dict]:
        return self.cohere.rerank_with_ids(
            query=query,
            documents=documents,
            text_key=text_key,
            id_key=id_key,
            top_n=top_n,
        )


def get_reranker(backend: Optional[str] = None) -> Optional[BaseReranker]:
    """
    Factory for the configured rerank backend.

    Args:
        backend: "local", "cohere", or "none" (defaults to settings.RERANK_BACKEND)

    Returns:
        BaseReranker instance, or None when backend == "none"
    """
    backend = backend or get_settings().RERANK_BACKEND

    if backend == "none":
        return None
    if backend == "cohere":
        return CohereReranker()
    return LocalReranker()
