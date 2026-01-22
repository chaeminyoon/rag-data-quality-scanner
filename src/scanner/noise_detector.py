"""
Duplicate and noise detection using embeddings and cosine similarity.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from config import get_logger, get_settings

logger = get_logger("scanner.noise_detector")


@dataclass
class DuplicateCluster:
    """A cluster of duplicate or near-duplicate documents."""

    cluster_id: int
    document_ids: List[str]
    similarity_scores: Dict[Tuple[str, str], float] = field(default_factory=dict)
    representative_id: Optional[str] = None

    @property
    def size(self) -> int:
        return len(self.document_ids)

    @property
    def avg_similarity(self) -> float:
        if not self.similarity_scores:
            return 0.0
        return np.mean(list(self.similarity_scores.values()))


@dataclass
class NoiseReport:
    """Report of detected noise in the dataset."""

    total_documents: int
    duplicate_clusters: List[DuplicateCluster]
    similarity_matrix: np.ndarray
    noise_scores: np.ndarray
    document_ids: List[str]

    @property
    def total_duplicates(self) -> int:
        """Total number of documents involved in duplicates."""
        return sum(cluster.size for cluster in self.duplicate_clusters)

    @property
    def unique_duplicates(self) -> int:
        """Number of documents that should be removed (keeping one per cluster)."""
        return sum(cluster.size - 1 for cluster in self.duplicate_clusters)

    @property
    def duplicate_percentage(self) -> float:
        """Percentage of documents that are duplicates."""
        if self.total_documents == 0:
            return 0.0
        return (self.total_duplicates / self.total_documents) * 100


class NoiseDetector:
    """
    Detect duplicate and near-duplicate content using embeddings.

    Uses cosine similarity to identify clusters of similar documents
    that may represent duplicates or near-duplicates.
    """

    def __init__(
        self,
        threshold: Optional[float] = None,
    ):
        """
        Initialize noise detector.

        Args:
            threshold: Cosine similarity threshold for duplicates (default: from settings)
        """
        settings = get_settings()
        self.threshold = threshold or settings.DUPLICATE_THRESHOLD

        logger.info(f"Initialized NoiseDetector with threshold={self.threshold}")

    def detect_duplicates(
        self,
        document_ids: List[str],
        embeddings: np.ndarray,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> NoiseReport:
        """
        Detect duplicate and near-duplicate documents.

        Args:
            document_ids: List of document identifiers
            embeddings: np.ndarray of shape (n_docs, dimension)
            progress_callback: Optional callback for progress updates

        Returns:
            NoiseReport with duplicate clusters and similarity matrix
        """
        if len(document_ids) != len(embeddings):
            raise ValueError("Document IDs and embeddings count mismatch")

        n_docs = len(document_ids)
        logger.info(f"Detecting duplicates in {n_docs} documents with threshold={self.threshold}")

        if progress_callback:
            progress_callback(0.1)

        # Compute pairwise cosine similarity
        similarity_matrix = cosine_similarity(embeddings)

        if progress_callback:
            progress_callback(0.5)

        # Find duplicate clusters using Union-Find
        clusters = self._find_duplicate_clusters(
            document_ids, similarity_matrix
        )

        if progress_callback:
            progress_callback(0.8)

        # Compute noise scores
        noise_scores = self._compute_noise_scores(similarity_matrix)

        if progress_callback:
            progress_callback(1.0)

        report = NoiseReport(
            total_documents=n_docs,
            duplicate_clusters=clusters,
            similarity_matrix=similarity_matrix,
            noise_scores=noise_scores,
            document_ids=document_ids,
        )

        logger.info(
            f"Found {len(clusters)} duplicate clusters, "
            f"{report.unique_duplicates} documents to remove "
            f"({report.duplicate_percentage:.1f}%)"
        )

        return report

    def _find_duplicate_clusters(
        self,
        document_ids: List[str],
        similarity_matrix: np.ndarray,
    ) -> List[DuplicateCluster]:
        """
        Find clusters of duplicate documents using Union-Find algorithm.

        Args:
            document_ids: List of document identifiers
            similarity_matrix: Pairwise cosine similarity matrix

        Returns:
            List of DuplicateCluster objects
        """
        n = len(document_ids)

        # Union-Find data structure
        parent = list(range(n))
        rank = [0] * n

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            px, py = find(x), find(y)
            if px == py:
                return
            if rank[px] < rank[py]:
                px, py = py, px
            parent[py] = px
            if rank[px] == rank[py]:
                rank[px] += 1

        # Find pairs above threshold and union them
        pairs_above_threshold: Dict[Tuple[int, int], float] = {}
        for i in range(n):
            for j in range(i + 1, n):
                if similarity_matrix[i, j] >= self.threshold:
                    union(i, j)
                    pairs_above_threshold[(i, j)] = similarity_matrix[i, j]

        # Group documents by cluster
        cluster_map: Dict[int, List[int]] = {}
        for i in range(n):
            root = find(i)
            if root not in cluster_map:
                cluster_map[root] = []
            cluster_map[root].append(i)

        # Create DuplicateCluster objects for clusters with more than one document
        clusters = []
        cluster_id = 0
        for root, indices in cluster_map.items():
            if len(indices) > 1:
                doc_ids = [document_ids[i] for i in indices]

                # Get similarity scores for pairs in this cluster
                similarity_scores = {}
                for i in range(len(indices)):
                    for j in range(i + 1, len(indices)):
                        idx_i, idx_j = indices[i], indices[j]
                        pair = (min(idx_i, idx_j), max(idx_i, idx_j))
                        if pair in pairs_above_threshold:
                            doc_pair = (document_ids[idx_i], document_ids[idx_j])
                            similarity_scores[doc_pair] = pairs_above_threshold[pair]

                # Select representative (first document in sorted order)
                representative = doc_ids[0]

                clusters.append(
                    DuplicateCluster(
                        cluster_id=cluster_id,
                        document_ids=doc_ids,
                        similarity_scores=similarity_scores,
                        representative_id=representative,
                    )
                )
                cluster_id += 1

        return clusters

    def _compute_noise_scores(self, similarity_matrix: np.ndarray) -> np.ndarray:
        """
        Compute per-document noise score.

        A document has high noise score if it has many high-similarity
        neighbors (potential duplicates).

        Args:
            similarity_matrix: Pairwise cosine similarity matrix

        Returns:
            Array of noise scores (0-1) for each document
        """
        n = len(similarity_matrix)
        noise_scores = np.zeros(n)

        for i in range(n):
            # Count neighbors above threshold (excluding self)
            similarities = similarity_matrix[i].copy()
            similarities[i] = 0  # Exclude self-similarity

            # Noise score = proportion of documents above threshold
            above_threshold = np.sum(similarities >= self.threshold)
            noise_scores[i] = above_threshold / (n - 1) if n > 1 else 0

        return noise_scores

    def get_similarity_heatmap_data(
        self,
        similarity_matrix: np.ndarray,
        document_ids: List[str],
        max_docs: int = 100,
    ) -> Dict:
        """
        Prepare data for similarity heatmap visualization.

        Args:
            similarity_matrix: Pairwise cosine similarity matrix
            document_ids: List of document identifiers
            max_docs: Maximum documents to include (for performance)

        Returns:
            Dict with heatmap data for Plotly
        """
        n = len(similarity_matrix)

        # Subsample if too many documents
        if n > max_docs:
            indices = np.linspace(0, n - 1, max_docs, dtype=int)
            matrix = similarity_matrix[np.ix_(indices, indices)]
            labels = [document_ids[i][:30] for i in indices]
        else:
            matrix = similarity_matrix
            labels = [doc_id[:30] for doc_id in document_ids]

        return {
            "z": matrix.tolist(),
            "x": labels,
            "y": labels,
            "colorscale": "RdBu",
            "zmin": 0,
            "zmax": 1,
        }
