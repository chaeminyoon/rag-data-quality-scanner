"""
Cohere API client wrapper with retry logic and batching.
Supports Embed v3 and Rerank 3.5.
"""

from typing import Callable, List, Optional

import cohere
from cohere.core.api_error import ApiError
import numpy as np
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from config import get_logger, get_settings

logger = get_logger("embeddings.cohere_client")


class CohereClient:
    """
    Cohere API client optimized for RAG applications.

    Features:
    - Embed v3 for document and query embeddings
    - Rerank 3.5 for result reranking
    - Automatic batching with progress callbacks
    - Retry logic with exponential backoff
    """

    # API limits
    MAX_EMBED_BATCH_SIZE = 96
    MAX_RERANK_DOCS = 1000

    def __init__(
        self,
        api_key: Optional[str] = None,
        embed_model: Optional[str] = None,
        rerank_model: Optional[str] = None,
    ):
        """
        Initialize Cohere client.

        Args:
            api_key: Cohere API key (uses settings if not provided)
            embed_model: Embedding model name
            rerank_model: Rerank model name
        """
        settings = get_settings()

        self.api_key = api_key or settings.COHERE_API_KEY.get_secret_value()
        self.embed_model = embed_model or settings.COHERE_EMBED_MODEL
        self.rerank_model = rerank_model or settings.COHERE_RERANK_MODEL

        self.client = cohere.Client(api_key=self.api_key)

        logger.info(f"Initialized Cohere client with embed={self.embed_model}, rerank={self.rerank_model}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((ApiError,)),
        before_sleep=lambda retry_state: logger.warning(
            f"Rate limited, retrying in {retry_state.next_action.sleep} seconds..."
        ),
    )
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
        if not texts:
            return np.array([])

        all_embeddings = []
        total_batches = (len(texts) + self.MAX_EMBED_BATCH_SIZE - 1) // self.MAX_EMBED_BATCH_SIZE

        logger.info(f"Embedding {len(texts)} documents in {total_batches} batches")

        for batch_idx in range(0, len(texts), self.MAX_EMBED_BATCH_SIZE):
            batch = texts[batch_idx : batch_idx + self.MAX_EMBED_BATCH_SIZE]

            try:
                response = self.client.embed(
                    texts=batch,
                    model=self.embed_model,
                    input_type=input_type,
                )
                all_embeddings.extend(response.embeddings)

            except ApiError as e:
                logger.error(f"API error: {e}")
                # Return zero vectors for failed batch
                dimension = len(all_embeddings[0]) if all_embeddings else 1024
                all_embeddings.extend([[0.0] * dimension] * len(batch))

            # Progress callback
            if progress_callback:
                progress = (batch_idx + len(batch)) / len(texts)
                progress_callback(progress)

        logger.info(f"Successfully embedded {len(all_embeddings)} documents")
        return np.array(all_embeddings)

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query for retrieval.

        Args:
            query: Query text

        Returns:
            np.ndarray of shape (dimension,)
        """
        embeddings = self.embed_documents([query], input_type="search_query")
        return embeddings[0]

    def embed_queries(
        self,
        queries: List[str],
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> np.ndarray:
        """
        Embed multiple queries for retrieval.

        Args:
            queries: List of query texts
            progress_callback: Optional callback for progress updates

        Returns:
            np.ndarray of shape (len(queries), dimension)
        """
        return self.embed_documents(queries, input_type="search_query", progress_callback=progress_callback)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((ApiError,)),
    )
    def rerank(
        self,
        query: str,
        documents: List[str],
        top_n: Optional[int] = None,
        return_documents: bool = False,
    ) -> List[dict]:
        """
        Rerank documents for relevance to query.

        Args:
            query: Search query
            documents: List of document texts (max 1000)
            top_n: Number of top results to return (default: all)
            return_documents: Include document text in response

        Returns:
            List of dicts with 'index', 'relevance_score', and optionally 'document'
        """
        if not documents:
            return []

        if len(documents) > self.MAX_RERANK_DOCS:
            logger.warning(f"Truncating to {self.MAX_RERANK_DOCS} documents for reranking")
            documents = documents[: self.MAX_RERANK_DOCS]

        top_n = top_n or len(documents)

        logger.info(f"Reranking {len(documents)} documents for query")

        response = self.client.rerank(
            query=query,
            documents=documents,
            model=self.rerank_model,
            top_n=top_n,
            return_documents=return_documents,
        )

        results = []
        for result in response.results:
            item = {
                "index": result.index,
                "relevance_score": result.relevance_score,
            }
            if return_documents:
                item["document"] = result.document.text if hasattr(result.document, 'text') else str(result.document)
            results.append(item)

        return results

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
            top_n: Number of top results to return

        Returns:
            List of dicts with 'id', 'text', and 'relevance_score'
        """
        texts = [doc[text_key] for doc in documents]
        rerank_results = self.rerank(query, texts, top_n=top_n)

        results = []
        for result in rerank_results:
            original_doc = documents[result["index"]]
            results.append({
                "id": original_doc[id_key],
                "text": original_doc[text_key],
                "relevance_score": result["relevance_score"],
            })

        return results

    def get_embedding_dimension(self) -> int:
        """
        Get the embedding dimension for the current model.

        Returns:
            Embedding dimension
        """
        # Embed a test text to get dimension
        test_embedding = self.embed_documents(["test"], input_type="search_document")
        return test_embedding.shape[1]

    def validate_connection(self) -> bool:
        """
        Validate API connection is working.

        Returns:
            True if connection is valid
        """
        try:
            self.client.embed(
                texts=["test"],
                model=self.embed_model,
                input_type="search_document",
            )
            logger.info("Cohere connection validated successfully")
            return True
        except Exception as e:
            logger.error(f"Cohere connection validation failed: {e}")
            return False
