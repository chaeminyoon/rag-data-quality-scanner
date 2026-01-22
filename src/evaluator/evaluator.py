"""
RAG benchmark evaluation orchestrator.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np

from config import get_logger, get_settings
from src.embeddings import CohereClient
from src.vectordb import PineconeClient
from .metrics import MetricsCalculator, EvaluationMetrics
from .reranker import CohereReranker

logger = get_logger("evaluator.evaluator")


@dataclass
class QueryResult:
    """Result for a single query evaluation."""

    query_id: str
    query: str
    retrieved_ids: List[str]
    relevant_ids: Set[str]
    metrics: EvaluationMetrics


@dataclass
class EvaluationResult:
    """Complete evaluation results."""

    total_queries: int
    metrics: EvaluationMetrics
    query_results: List[QueryResult]
    with_rerank: bool = False

    @property
    def summary(self) -> Dict:
        return {
            "total_queries": self.total_queries,
            "with_rerank": self.with_rerank,
            **self.metrics.summary,
        }


@dataclass
class ComparisonResult:
    """Comparison between original and cleaned data."""

    original_metrics: EvaluationMetrics
    cleaned_metrics: EvaluationMetrics
    cleaned_with_rerank_metrics: Optional[EvaluationMetrics] = None
    improvement: Dict = field(default_factory=dict)
    per_query_comparison: List[Dict] = field(default_factory=list)

    def __post_init__(self):
        """Calculate improvement percentages."""
        self.improvement = self._calculate_improvement()

    def _calculate_improvement(self) -> Dict:
        """Calculate percentage improvement for each metric."""
        def calc_pct(old: float, new: float) -> float:
            if old == 0:
                return 100.0 if new > 0 else 0.0
            return ((new - old) / old) * 100

        # Compare original to cleaned
        improvement = {
            "ndcg_improvement": calc_pct(
                self.original_metrics.ndcg_at_k,
                self.cleaned_metrics.ndcg_at_k,
            ),
            "hit_rate_improvement": calc_pct(
                self.original_metrics.hit_rate_at_k,
                self.cleaned_metrics.hit_rate_at_k,
            ),
            "mrr_improvement": calc_pct(
                self.original_metrics.mrr,
                self.cleaned_metrics.mrr,
            ),
        }

        # Add rerank improvement if available
        if self.cleaned_with_rerank_metrics:
            improvement["ndcg_with_rerank"] = self.cleaned_with_rerank_metrics.ndcg_at_k
            improvement["ndcg_rerank_improvement"] = calc_pct(
                self.original_metrics.ndcg_at_k,
                self.cleaned_with_rerank_metrics.ndcg_at_k,
            )

        return improvement

    @property
    def summary(self) -> Dict:
        summary = {
            "original": self.original_metrics.summary,
            "cleaned": self.cleaned_metrics.summary,
            "improvement": {
                k: f"{v:+.1f}%" for k, v in self.improvement.items()
                if "improvement" in k
            },
        }

        if self.cleaned_with_rerank_metrics:
            summary["cleaned_with_rerank"] = self.cleaned_with_rerank_metrics.summary

        return summary


class RAGEvaluator:
    """
    Evaluate RAG performance with before/after comparison.

    Orchestrates:
    1. Document indexing to Pinecone
    2. Query evaluation
    3. Reranking integration
    4. Metrics calculation
    """

    def __init__(
        self,
        cohere_client: Optional[CohereClient] = None,
        pinecone_client: Optional[PineconeClient] = None,
        reranker: Optional[CohereReranker] = None,
        k: int = None,
    ):
        """
        Initialize evaluator.

        Args:
            cohere_client: Cohere client
            pinecone_client: Pinecone client
            reranker: Cohere reranker
            k: Default k for @k metrics
        """
        settings = get_settings()

        self.cohere = cohere_client or CohereClient()
        self.pinecone = pinecone_client or PineconeClient()
        self.reranker = reranker or CohereReranker(self.cohere)
        self.metrics = MetricsCalculator(k=k or settings.TOP_K)
        self.k = k or settings.TOP_K

        logger.info(f"Initialized RAGEvaluator with k={self.k}")

    def setup_index(
        self,
        documents: List[Dict],
        embeddings: np.ndarray,
        namespace: str,
        text_key: str = "text",
        id_key: str = "id",
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> None:
        """
        Index documents to Pinecone.

        Args:
            documents: List of documents
            embeddings: Document embeddings
            namespace: Pinecone namespace
            text_key: Key for document text
            id_key: Key for document ID
            progress_callback: Progress callback
        """
        logger.info(f"Indexing {len(documents)} documents to namespace '{namespace}'")

        # Ensure index exists
        dimension = embeddings.shape[1] if len(embeddings.shape) > 1 else len(embeddings[0])
        self.pinecone.create_index(dimension=dimension)

        # Clear existing data in namespace
        self.pinecone.delete_namespace(namespace)

        # Upsert documents
        self.pinecone.upsert_documents(
            documents=documents,
            embeddings=embeddings,
            namespace=namespace,
            progress_callback=progress_callback,
        )

    def evaluate(
        self,
        queries: List[Dict],
        namespace: str,
        query_key: str = "query",
        relevant_ids_key: str = "relevant_doc_ids",
        query_id_key: str = "query_id",
        use_rerank: bool = False,
        top_k_retrieval: int = 100,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> EvaluationResult:
        """
        Evaluate retrieval performance.

        Args:
            queries: List of query dicts with ground truth
            namespace: Pinecone namespace to query
            query_key: Key for query text
            relevant_ids_key: Key for relevant document IDs
            query_id_key: Key for query ID
            use_rerank: Apply Cohere Rerank
            top_k_retrieval: Initial retrieval count
            progress_callback: Progress callback

        Returns:
            EvaluationResult with metrics
        """
        logger.info(f"Evaluating {len(queries)} queries on namespace '{namespace}'")

        query_results = []
        all_retrieved = []
        all_relevant = []

        for i, query_data in enumerate(queries):
            query_text = query_data.get(query_key, "")
            query_id = query_data.get(query_id_key, f"query_{i}")
            relevant_ids = set(query_data.get(relevant_ids_key, []))

            # Embed query
            query_embedding = self.cohere.embed_query(query_text)

            # Retrieve from Pinecone
            results = self.pinecone.query(
                query_embedding=query_embedding,
                top_k=top_k_retrieval,
                namespace=namespace,
            )

            # Apply reranking if requested
            if use_rerank and results:
                reranked = self.reranker.rerank_retrieval_results(
                    query=query_text,
                    retrieval_results=results,
                    top_n=self.k,
                )
                retrieved_ids = [r["id"] for r in reranked]
            else:
                retrieved_ids = [r["id"] for r in results[:self.k]]

            # Compute metrics for this query
            query_metrics = self.metrics.compute_all(retrieved_ids, relevant_ids, self.k)

            query_results.append(QueryResult(
                query_id=query_id,
                query=query_text,
                retrieved_ids=retrieved_ids,
                relevant_ids=relevant_ids,
                metrics=query_metrics,
            ))

            all_retrieved.append(retrieved_ids)
            all_relevant.append(relevant_ids)

            if progress_callback:
                progress_callback((i + 1) / len(queries))

        # Compute average metrics
        avg_metrics = self.metrics.compute_average(all_retrieved, all_relevant, self.k)

        return EvaluationResult(
            total_queries=len(queries),
            metrics=avg_metrics,
            query_results=query_results,
            with_rerank=use_rerank,
        )

    def compare(
        self,
        queries: List[Dict],
        original_documents: List[Dict],
        original_embeddings: np.ndarray,
        cleaned_documents: List[Dict],
        cleaned_embeddings: np.ndarray,
        query_key: str = "query",
        relevant_ids_key: str = "relevant_doc_ids",
        text_key: str = "text",
        id_key: str = "id",
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> ComparisonResult:
        """
        Compare performance between original and cleaned data.

        Args:
            queries: List of query dicts with ground truth
            original_documents: Original documents
            original_embeddings: Original embeddings
            cleaned_documents: Cleaned documents
            cleaned_embeddings: Cleaned embeddings
            query_key: Key for query text
            relevant_ids_key: Key for relevant document IDs
            text_key: Key for document text
            id_key: Key for document ID
            progress_callback: Callback (stage, progress)

        Returns:
            ComparisonResult with before/after metrics
        """
        def emit_progress(stage: str, progress: float):
            if progress_callback:
                progress_callback(stage, progress)

        # Index original data
        emit_progress("indexing_original", 0.0)
        self.setup_index(
            documents=original_documents,
            embeddings=original_embeddings,
            namespace="original",
            text_key=text_key,
            id_key=id_key,
            progress_callback=lambda p: emit_progress("indexing_original", p),
        )

        # Index cleaned data
        emit_progress("indexing_cleaned", 0.0)
        self.setup_index(
            documents=cleaned_documents,
            embeddings=cleaned_embeddings,
            namespace="cleaned",
            text_key=text_key,
            id_key=id_key,
            progress_callback=lambda p: emit_progress("indexing_cleaned", p),
        )

        # Evaluate original
        emit_progress("evaluating_original", 0.0)
        original_result = self.evaluate(
            queries=queries,
            namespace="original",
            query_key=query_key,
            relevant_ids_key=relevant_ids_key,
            use_rerank=False,
            progress_callback=lambda p: emit_progress("evaluating_original", p),
        )

        # Evaluate cleaned
        emit_progress("evaluating_cleaned", 0.0)
        cleaned_result = self.evaluate(
            queries=queries,
            namespace="cleaned",
            query_key=query_key,
            relevant_ids_key=relevant_ids_key,
            use_rerank=False,
            progress_callback=lambda p: emit_progress("evaluating_cleaned", p),
        )

        # Evaluate cleaned with rerank
        emit_progress("evaluating_rerank", 0.0)
        cleaned_rerank_result = self.evaluate(
            queries=queries,
            namespace="cleaned",
            query_key=query_key,
            relevant_ids_key=relevant_ids_key,
            use_rerank=True,
            progress_callback=lambda p: emit_progress("evaluating_rerank", p),
        )

        # Build per-query comparison
        per_query = []
        for orig, clean, rerank in zip(
            original_result.query_results,
            cleaned_result.query_results,
            cleaned_rerank_result.query_results,
        ):
            per_query.append({
                "query_id": orig.query_id,
                "query": orig.query[:100],
                "original_ndcg": orig.metrics.ndcg_at_k,
                "cleaned_ndcg": clean.metrics.ndcg_at_k,
                "rerank_ndcg": rerank.metrics.ndcg_at_k,
            })

        return ComparisonResult(
            original_metrics=original_result.metrics,
            cleaned_metrics=cleaned_result.metrics,
            cleaned_with_rerank_metrics=cleaned_rerank_result.metrics,
            per_query_comparison=per_query,
        )

    def cleanup(self, namespaces: List[str] = None) -> None:
        """
        Clean up Pinecone namespaces.

        Args:
            namespaces: Namespaces to delete (default: original, cleaned)
        """
        namespaces = namespaces or ["original", "cleaned"]

        for ns in namespaces:
            try:
                self.pinecone.delete_namespace(ns)
                logger.info(f"Deleted namespace: {ns}")
            except Exception as e:
                logger.warning(f"Failed to delete namespace {ns}: {e}")
