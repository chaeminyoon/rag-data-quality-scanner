"""
Information retrieval evaluation metrics.
"""

from dataclasses import dataclass
from typing import Dict, List, Set

import numpy as np
from sklearn.metrics import ndcg_score

from config import get_logger

logger = get_logger("evaluator.metrics")


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""

    ndcg_at_k: float
    hit_rate_at_k: float
    mrr: float
    precision_at_k: float
    recall_at_k: float
    k: int

    @property
    def summary(self) -> Dict:
        return {
            f"NDCG@{self.k}": round(self.ndcg_at_k, 4),
            f"Hit Rate@{self.k}": round(self.hit_rate_at_k, 4),
            "MRR": round(self.mrr, 4),
            f"Precision@{self.k}": round(self.precision_at_k, 4),
            f"Recall@{self.k}": round(self.recall_at_k, 4),
        }


class MetricsCalculator:
    """
    Calculate information retrieval metrics.

    Supports:
    - NDCG@k (Normalized Discounted Cumulative Gain)
    - Hit Rate@k
    - MRR (Mean Reciprocal Rank)
    - Precision@k
    - Recall@k
    """

    def __init__(self, k: int = 10):
        """
        Initialize metrics calculator.

        Args:
            k: Default cutoff position for @k metrics
        """
        self.k = k

    def ndcg_at_k(
        self,
        retrieved_ids: List[str],
        relevant_ids: Set[str],
        k: int = None,
    ) -> float:
        """
        Calculate NDCG@k.

        NDCG measures ranking quality, accounting for position of relevant items.

        Args:
            retrieved_ids: Ordered list of retrieved document IDs
            relevant_ids: Set of relevant document IDs
            k: Cutoff position (uses default if None)

        Returns:
            NDCG@k score (0 to 1)
        """
        k = k or self.k
        retrieved = retrieved_ids[:k]

        if not relevant_ids or not retrieved:
            return 0.0

        # Create relevance scores (1 for relevant, 0 for not)
        y_true = [[1 if doc_id in relevant_ids else 0 for doc_id in retrieved]]

        # Create predicted scores (decreasing by position)
        y_score = [[len(retrieved) - i for i in range(len(retrieved))]]

        try:
            return ndcg_score(y_true, y_score, k=k)
        except ValueError:
            return 0.0

    def hit_rate_at_k(
        self,
        retrieved_ids: List[str],
        relevant_ids: Set[str],
        k: int = None,
    ) -> float:
        """
        Calculate Hit Rate@k (binary recall).

        Hit Rate = 1 if any relevant document is in top-k results, else 0.

        Args:
            retrieved_ids: Ordered list of retrieved document IDs
            relevant_ids: Set of relevant document IDs
            k: Cutoff position

        Returns:
            1.0 if hit, 0.0 otherwise
        """
        k = k or self.k
        retrieved = set(retrieved_ids[:k])

        if not relevant_ids:
            return 0.0

        return 1.0 if retrieved & relevant_ids else 0.0

    def mrr(
        self,
        retrieved_ids: List[str],
        relevant_ids: Set[str],
    ) -> float:
        """
        Calculate Mean Reciprocal Rank.

        MRR = 1/rank of first relevant document.

        Args:
            retrieved_ids: Ordered list of retrieved document IDs
            relevant_ids: Set of relevant document IDs

        Returns:
            Reciprocal rank (0 to 1)
        """
        if not relevant_ids:
            return 0.0

        for rank, doc_id in enumerate(retrieved_ids, start=1):
            if doc_id in relevant_ids:
                return 1.0 / rank

        return 0.0

    def precision_at_k(
        self,
        retrieved_ids: List[str],
        relevant_ids: Set[str],
        k: int = None,
    ) -> float:
        """
        Calculate Precision@k.

        Precision = (relevant items in top-k) / k

        Args:
            retrieved_ids: Ordered list of retrieved document IDs
            relevant_ids: Set of relevant document IDs
            k: Cutoff position

        Returns:
            Precision score (0 to 1)
        """
        k = k or self.k
        retrieved = retrieved_ids[:k]

        if not retrieved:
            return 0.0

        relevant_retrieved = sum(1 for doc_id in retrieved if doc_id in relevant_ids)
        return relevant_retrieved / len(retrieved)

    def recall_at_k(
        self,
        retrieved_ids: List[str],
        relevant_ids: Set[str],
        k: int = None,
    ) -> float:
        """
        Calculate Recall@k.

        Recall = (relevant items in top-k) / (total relevant items)

        Args:
            retrieved_ids: Ordered list of retrieved document IDs
            relevant_ids: Set of relevant document IDs
            k: Cutoff position

        Returns:
            Recall score (0 to 1)
        """
        k = k or self.k
        retrieved = retrieved_ids[:k]

        if not relevant_ids:
            return 0.0

        relevant_retrieved = sum(1 for doc_id in retrieved if doc_id in relevant_ids)
        return relevant_retrieved / len(relevant_ids)

    def compute_all(
        self,
        retrieved_ids: List[str],
        relevant_ids: Set[str],
        k: int = None,
    ) -> EvaluationMetrics:
        """
        Compute all metrics at once.

        Args:
            retrieved_ids: Ordered list of retrieved document IDs
            relevant_ids: Set of relevant document IDs
            k: Cutoff position

        Returns:
            EvaluationMetrics with all scores
        """
        k = k or self.k

        return EvaluationMetrics(
            ndcg_at_k=self.ndcg_at_k(retrieved_ids, relevant_ids, k),
            hit_rate_at_k=self.hit_rate_at_k(retrieved_ids, relevant_ids, k),
            mrr=self.mrr(retrieved_ids, relevant_ids),
            precision_at_k=self.precision_at_k(retrieved_ids, relevant_ids, k),
            recall_at_k=self.recall_at_k(retrieved_ids, relevant_ids, k),
            k=k,
        )

    def compute_average(
        self,
        all_retrieved: List[List[str]],
        all_relevant: List[Set[str]],
        k: int = None,
    ) -> EvaluationMetrics:
        """
        Compute average metrics across multiple queries.

        Args:
            all_retrieved: List of retrieved ID lists (per query)
            all_relevant: List of relevant ID sets (per query)
            k: Cutoff position

        Returns:
            EvaluationMetrics with averaged scores
        """
        k = k or self.k

        if len(all_retrieved) != len(all_relevant):
            raise ValueError("Mismatched number of queries and ground truth")

        if not all_retrieved:
            return EvaluationMetrics(
                ndcg_at_k=0.0,
                hit_rate_at_k=0.0,
                mrr=0.0,
                precision_at_k=0.0,
                recall_at_k=0.0,
                k=k,
            )

        # Compute per-query metrics
        ndcg_scores = []
        hit_rates = []
        mrr_scores = []
        precision_scores = []
        recall_scores = []

        for retrieved, relevant in zip(all_retrieved, all_relevant):
            ndcg_scores.append(self.ndcg_at_k(retrieved, relevant, k))
            hit_rates.append(self.hit_rate_at_k(retrieved, relevant, k))
            mrr_scores.append(self.mrr(retrieved, relevant))
            precision_scores.append(self.precision_at_k(retrieved, relevant, k))
            recall_scores.append(self.recall_at_k(retrieved, relevant, k))

        return EvaluationMetrics(
            ndcg_at_k=np.mean(ndcg_scores),
            hit_rate_at_k=np.mean(hit_rates),
            mrr=np.mean(mrr_scores),
            precision_at_k=np.mean(precision_scores),
            recall_at_k=np.mean(recall_scores),
            k=k,
        )

    def compute_metrics_at_multiple_k(
        self,
        all_retrieved: List[List[str]],
        all_relevant: List[Set[str]],
        k_values: List[int] = None,
    ) -> Dict[int, EvaluationMetrics]:
        """
        Compute metrics at multiple k values.

        Args:
            all_retrieved: List of retrieved ID lists (per query)
            all_relevant: List of relevant ID sets (per query)
            k_values: List of k values to evaluate

        Returns:
            Dict mapping k to EvaluationMetrics
        """
        k_values = k_values or [1, 3, 5, 10, 20]

        results = {}
        for k in k_values:
            results[k] = self.compute_average(all_retrieved, all_relevant, k)

        return results
