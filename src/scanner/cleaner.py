"""
Data cleaning and deduplication logic.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Set

import numpy as np

from config import get_logger, get_settings
from .noise_detector import DuplicateCluster, NoiseReport
from .text_analyzer import TextAnalysisResult, QualityIssueType

logger = get_logger("scanner.cleaner")


class CleaningStrategy(Enum):
    """Cleaning aggressiveness levels."""

    CONSERVATIVE = "conservative"  # Only exact duplicates
    MODERATE = "moderate"          # Near-duplicates + short sentences
    AGGRESSIVE = "aggressive"      # All quality issues


@dataclass
class CleanedDataResult:
    """Results of data cleaning operation."""

    original_count: int
    cleaned_count: int
    removed_count: int
    cleaned_documents: List[Dict]
    removed_documents: List[Dict]
    removal_reasons: Dict[str, List[str]]  # reason -> list of doc_ids

    @property
    def removal_percentage(self) -> float:
        if self.original_count == 0:
            return 0.0
        return (self.removed_count / self.original_count) * 100

    @property
    def summary(self) -> Dict:
        return {
            "original_count": self.original_count,
            "cleaned_count": self.cleaned_count,
            "removed_count": self.removed_count,
            "removal_percentage": f"{self.removal_percentage:.1f}%",
            "removal_breakdown": {
                reason: len(doc_ids) for reason, doc_ids in self.removal_reasons.items()
            },
        }


class DataCleaner:
    """
    Clean and deduplicate documents based on quality analysis.

    Supports multiple cleaning strategies from conservative to aggressive.
    """

    def __init__(
        self,
        min_length: Optional[int] = None,
        duplicate_threshold: Optional[float] = None,
    ):
        """
        Initialize data cleaner.

        Args:
            min_length: Minimum document length to keep
            duplicate_threshold: Similarity threshold for duplicates
        """
        settings = get_settings()

        self.min_length = min_length or settings.MIN_SENTENCE_LENGTH
        self.duplicate_threshold = duplicate_threshold or settings.DUPLICATE_THRESHOLD

        logger.info(
            f"Initialized DataCleaner with min_length={self.min_length}, "
            f"threshold={self.duplicate_threshold}"
        )

    def clean(
        self,
        documents: List[Dict],
        noise_report: Optional[NoiseReport] = None,
        text_analysis: Optional[TextAnalysisResult] = None,
        strategy: CleaningStrategy = CleaningStrategy.MODERATE,
        text_key: str = "text",
        id_key: str = "id",
    ) -> CleanedDataResult:
        """
        Clean documents based on quality analysis results.

        Args:
            documents: List of documents to clean
            noise_report: Results from noise detection
            text_analysis: Results from text analysis
            strategy: Cleaning aggressiveness level
            text_key: Key for document text
            id_key: Key for document ID

        Returns:
            CleanedDataResult with cleaned and removed documents
        """
        logger.info(f"Cleaning {len(documents)} documents with strategy={strategy.value}")

        # Build document lookup
        doc_lookup = {doc.get(id_key, str(i)): doc for i, doc in enumerate(documents)}

        # Track documents to remove and reasons
        to_remove: Dict[str, str] = {}  # doc_id -> reason

        # Apply cleaning based on strategy
        if noise_report and strategy in [CleaningStrategy.MODERATE, CleaningStrategy.AGGRESSIVE]:
            # Remove duplicates (keep representative from each cluster)
            for cluster in noise_report.duplicate_clusters:
                for doc_id in cluster.document_ids:
                    if doc_id != cluster.representative_id:
                        to_remove[doc_id] = "duplicate"

        if text_analysis:
            # Remove short documents
            if strategy in [CleaningStrategy.MODERATE, CleaningStrategy.AGGRESSIVE]:
                for doc_id in text_analysis.short_documents:
                    if doc_id not in to_remove:
                        to_remove[doc_id] = "too_short"

            # Remove documents with other issues (aggressive only)
            if strategy == CleaningStrategy.AGGRESSIVE:
                for issue in text_analysis.quality_issues:
                    if issue.issue_type in [
                        QualityIssueType.LOW_WORD_COUNT,
                        QualityIssueType.HIGH_SPECIAL_CHAR_RATIO,
                    ]:
                        if issue.document_id not in to_remove:
                            to_remove[issue.document_id] = issue.issue_type.value

        # Separate cleaned and removed documents
        cleaned_documents = []
        removed_documents = []
        removal_reasons: Dict[str, List[str]] = {}

        for doc_id, doc in doc_lookup.items():
            if doc_id in to_remove:
                removed_documents.append(doc)
                reason = to_remove[doc_id]
                if reason not in removal_reasons:
                    removal_reasons[reason] = []
                removal_reasons[reason].append(doc_id)
            else:
                cleaned_documents.append(doc)

        result = CleanedDataResult(
            original_count=len(documents),
            cleaned_count=len(cleaned_documents),
            removed_count=len(removed_documents),
            cleaned_documents=cleaned_documents,
            removed_documents=removed_documents,
            removal_reasons=removal_reasons,
        )

        logger.info(
            f"Cleaning complete: removed {result.removed_count} documents "
            f"({result.removal_percentage:.1f}%)"
        )

        return result

    def deduplicate_only(
        self,
        documents: List[Dict],
        duplicate_clusters: List[DuplicateCluster],
        id_key: str = "id",
        select_strategy: str = "first",
    ) -> CleanedDataResult:
        """
        Remove only duplicates, keeping one representative per cluster.

        Args:
            documents: List of documents
            duplicate_clusters: Clusters from noise detection
            id_key: Key for document ID
            select_strategy: How to select representative ("first", "longest")

        Returns:
            CleanedDataResult with deduplicated documents
        """
        doc_lookup = {doc.get(id_key, str(i)): doc for i, doc in enumerate(documents)}
        to_remove: Set[str] = set()

        for cluster in duplicate_clusters:
            if select_strategy == "longest":
                # Select longest document as representative
                cluster_docs = [
                    (doc_id, doc_lookup.get(doc_id))
                    for doc_id in cluster.document_ids
                    if doc_id in doc_lookup
                ]
                if cluster_docs:
                    sorted_docs = sorted(
                        cluster_docs,
                        key=lambda x: len(x[1].get("text", "")) if x[1] else 0,
                        reverse=True,
                    )
                    # Mark all except longest for removal
                    for doc_id, _ in sorted_docs[1:]:
                        to_remove.add(doc_id)
            else:
                # Default: keep first (representative_id)
                for doc_id in cluster.document_ids:
                    if doc_id != cluster.representative_id:
                        to_remove.add(doc_id)

        cleaned = [doc for doc in documents if doc.get(id_key) not in to_remove]
        removed = [doc for doc in documents if doc.get(id_key) in to_remove]

        return CleanedDataResult(
            original_count=len(documents),
            cleaned_count=len(cleaned),
            removed_count=len(removed),
            cleaned_documents=cleaned,
            removed_documents=removed,
            removal_reasons={"duplicate": list(to_remove)},
        )

    def filter_by_length(
        self,
        documents: List[Dict],
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        text_key: str = "text",
        id_key: str = "id",
    ) -> CleanedDataResult:
        """
        Filter documents by text length.

        Args:
            documents: List of documents
            min_length: Minimum length (uses default if None)
            max_length: Maximum length (no limit if None)
            text_key: Key for document text
            id_key: Key for document ID

        Returns:
            CleanedDataResult with filtered documents
        """
        min_len = min_length if min_length is not None else self.min_length

        cleaned = []
        removed = []
        removal_reasons: Dict[str, List[str]] = {"too_short": [], "too_long": []}

        for doc in documents:
            text = doc.get(text_key, "")
            doc_id = doc.get(id_key, "unknown")
            text_length = len(text)

            if text_length < min_len:
                removed.append(doc)
                removal_reasons["too_short"].append(doc_id)
            elif max_length and text_length > max_length:
                removed.append(doc)
                removal_reasons["too_long"].append(doc_id)
            else:
                cleaned.append(doc)

        # Clean up empty reasons
        removal_reasons = {k: v for k, v in removal_reasons.items() if v}

        return CleanedDataResult(
            original_count=len(documents),
            cleaned_count=len(cleaned),
            removed_count=len(removed),
            cleaned_documents=cleaned,
            removed_documents=removed,
            removal_reasons=removal_reasons,
        )

    def filter_by_quality_score(
        self,
        documents: List[Dict],
        text_analysis: TextAnalysisResult,
        min_quality_score: float = 0.5,
        id_key: str = "id",
    ) -> CleanedDataResult:
        """
        Filter documents by quality score.

        Args:
            documents: List of documents
            text_analysis: Results from text analysis
            min_quality_score: Minimum quality score to keep
            id_key: Key for document ID

        Returns:
            CleanedDataResult with filtered documents
        """
        # Build quality score lookup
        score_lookup = {
            stat.document_id: stat.quality_score
            for stat in text_analysis.document_stats
        }

        cleaned = []
        removed = []
        low_quality_ids = []

        for doc in documents:
            doc_id = doc.get(id_key, "unknown")
            score = score_lookup.get(doc_id, 1.0)

            if score < min_quality_score:
                removed.append(doc)
                low_quality_ids.append(doc_id)
            else:
                cleaned.append(doc)

        return CleanedDataResult(
            original_count=len(documents),
            cleaned_count=len(cleaned),
            removed_count=len(removed),
            cleaned_documents=cleaned,
            removed_documents=removed,
            removal_reasons={"low_quality": low_quality_ids} if low_quality_ids else {},
        )
