"""
Cohere Rerank integration for result reranking.
"""

from typing import List, Optional

from config import get_logger
from src.embeddings import CohereClient

logger = get_logger("evaluator.reranker")


class CohereReranker:
    """
    Wrapper for Cohere Rerank 3.5 functionality.

    Improves retrieval quality by semantically reordering
    initial retrieval results.
    """

    def __init__(self, cohere_client: Optional[CohereClient] = None):
        """
        Initialize reranker.

        Args:
            cohere_client: Cohere client (created if not provided)
        """
        self.cohere = cohere_client or CohereClient()
        logger.info("Initialized CohereReranker")

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_n: Optional[int] = None,
    ) -> List[dict]:
        """
        Rerank documents for a query.

        Args:
            query: Search query
            documents: List of document texts
            top_n: Number of results to return (default: all)

        Returns:
            List of dicts with 'index' and 'relevance_score'
        """
        return self.cohere.rerank(
            query=query,
            documents=documents,
            top_n=top_n,
        )

    def rerank_with_ids(
        self,
        query: str,
        documents: List[dict],
        text_key: str = "text",
        id_key: str = "id",
        top_n: Optional[int] = None,
    ) -> List[dict]:
        """
        Rerank documents preserving their IDs.

        Args:
            query: Search query
            documents: List of dicts with text and ID
            text_key: Key for document text
            id_key: Key for document ID
            top_n: Number of results to return

        Returns:
            List of dicts with 'id', 'text', 'relevance_score'
        """
        return self.cohere.rerank_with_ids(
            query=query,
            documents=documents,
            text_key=text_key,
            id_key=id_key,
            top_n=top_n,
        )

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

        Combines original retrieval score with rerank score.

        Args:
            query: Search query
            retrieval_results: Results from vector search
            text_key: Key for document text
            id_key: Key for document ID
            score_key: Key for original retrieval score
            top_n: Number of results to return

        Returns:
            Reranked results with combined scores
        """
        if not retrieval_results:
            return []

        # Get texts for reranking
        texts = []
        for result in retrieval_results:
            # Handle nested metadata structure
            if "metadata" in result and text_key in result["metadata"]:
                texts.append(result["metadata"][text_key])
            elif text_key in result:
                texts.append(result[text_key])
            else:
                texts.append("")

        # Rerank
        rerank_results = self.cohere.rerank(
            query=query,
            documents=texts,
            top_n=top_n or len(retrieval_results),
        )

        # Build reranked list
        reranked = []
        for rr in rerank_results:
            original = retrieval_results[rr["index"]]
            original_score = original.get(score_key, 0.0)

            # Get ID from either top level or metadata
            doc_id = original.get(id_key)
            if doc_id is None and "metadata" in original:
                doc_id = original["metadata"].get(id_key, rr["index"])

            # Get text from either top level or metadata
            doc_text = texts[rr["index"]]

            reranked.append({
                "id": doc_id,
                "text": doc_text,
                "original_score": original_score,
                "rerank_score": rr["relevance_score"],
                "metadata": original.get("metadata", {}),
            })

        return reranked
