"""
Text quality analysis for documents.
"""

import re
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import numpy as np

from config import get_logger, get_settings

logger = get_logger("scanner.text_analyzer")


class QualityIssueType(Enum):
    """Types of text quality issues."""

    TOO_SHORT = "too_short"
    TOO_LONG = "too_long"
    LOW_WORD_COUNT = "low_word_count"
    HIGH_SPECIAL_CHAR_RATIO = "high_special_char_ratio"
    MISSING_PUNCTUATION = "missing_punctuation"
    REPETITIVE_CONTENT = "repetitive_content"


@dataclass
class QualityIssue:
    """A detected quality issue in a document."""

    document_id: str
    issue_type: QualityIssueType
    message: str
    severity: str = "warning"  # "warning" or "error"
    metadata: Dict = field(default_factory=dict)


@dataclass
class DocumentStats:
    """Statistics for a single document."""

    document_id: str
    char_count: int
    word_count: int
    sentence_count: int
    avg_word_length: float
    special_char_ratio: float
    quality_score: float


@dataclass
class TextAnalysisResult:
    """Results of text quality analysis."""

    total_documents: int
    document_stats: List[DocumentStats]
    quality_issues: List[QualityIssue]
    length_distribution: Dict[str, int]
    avg_quality_score: float

    @property
    def total_issues(self) -> int:
        return len(self.quality_issues)

    @property
    def issues_by_type(self) -> Dict[QualityIssueType, int]:
        counter = Counter(issue.issue_type for issue in self.quality_issues)
        return dict(counter)

    @property
    def short_documents(self) -> List[str]:
        return [
            issue.document_id
            for issue in self.quality_issues
            if issue.issue_type == QualityIssueType.TOO_SHORT
        ]

    @property
    def long_documents(self) -> List[str]:
        return [
            issue.document_id
            for issue in self.quality_issues
            if issue.issue_type == QualityIssueType.TOO_LONG
        ]


class TextAnalyzer:
    """
    Analyze text quality characteristics.

    Evaluates:
    - Document length distribution
    - Word count and structure
    - Special character ratios
    - Overall quality scoring
    """

    # Sentence-ending punctuation
    SENTENCE_PATTERN = re.compile(r"[.!?]+")

    # Special characters (excluding common punctuation)
    SPECIAL_CHAR_PATTERN = re.compile(r"[^\w\s.,!?;:\'\"-]")

    def __init__(
        self,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        min_word_count: int = 3,
        max_special_char_ratio: float = 0.1,
    ):
        """
        Initialize text analyzer.

        Args:
            min_length: Minimum document length in characters
            max_length: Maximum document length in characters
            min_word_count: Minimum word count
            max_special_char_ratio: Maximum ratio of special characters
        """
        settings = get_settings()

        self.min_length = min_length or settings.MIN_SENTENCE_LENGTH
        self.max_length = max_length or settings.MAX_SENTENCE_LENGTH
        self.min_word_count = min_word_count
        self.max_special_char_ratio = max_special_char_ratio

        logger.info(
            f"Initialized TextAnalyzer with min_length={self.min_length}, "
            f"max_length={self.max_length}"
        )

    def analyze(
        self,
        documents: List[Dict],
        text_key: str = "text",
        id_key: str = "id",
    ) -> TextAnalysisResult:
        """
        Analyze text quality of documents.

        Args:
            documents: List of documents with text content
            text_key: Key for document text
            id_key: Key for document ID

        Returns:
            TextAnalysisResult with analysis details
        """
        logger.info(f"Analyzing text quality for {len(documents)} documents")

        document_stats = []
        quality_issues = []
        length_bins = {
            "very_short": 0,  # < 50 chars
            "short": 0,       # 50-200 chars
            "medium": 0,      # 200-500 chars
            "long": 0,        # 500-1000 chars
            "very_long": 0,   # > 1000 chars
        }

        for doc in documents:
            doc_id = doc.get(id_key, "unknown")
            text = doc.get(text_key, "")

            # Compute statistics
            stats = self._compute_document_stats(doc_id, text)
            document_stats.append(stats)

            # Categorize by length
            char_count = stats.char_count
            if char_count < 50:
                length_bins["very_short"] += 1
            elif char_count < 200:
                length_bins["short"] += 1
            elif char_count < 500:
                length_bins["medium"] += 1
            elif char_count < 1000:
                length_bins["long"] += 1
            else:
                length_bins["very_long"] += 1

            # Detect quality issues
            issues = self._detect_issues(doc_id, text, stats)
            quality_issues.extend(issues)

        # Calculate average quality score
        avg_quality = np.mean([s.quality_score for s in document_stats]) if document_stats else 0.0

        result = TextAnalysisResult(
            total_documents=len(documents),
            document_stats=document_stats,
            quality_issues=quality_issues,
            length_distribution=length_bins,
            avg_quality_score=avg_quality,
        )

        logger.info(
            f"Analysis complete: avg_quality={avg_quality:.2f}, "
            f"issues={len(quality_issues)}"
        )

        return result

    def _compute_document_stats(self, doc_id: str, text: str) -> DocumentStats:
        """Compute statistics for a single document."""
        char_count = len(text)
        words = text.split()
        word_count = len(words)

        # Sentence count
        sentences = self.SENTENCE_PATTERN.split(text)
        sentence_count = len([s for s in sentences if s.strip()])

        # Average word length
        avg_word_length = np.mean([len(w) for w in words]) if words else 0

        # Special character ratio
        special_chars = self.SPECIAL_CHAR_PATTERN.findall(text)
        special_char_ratio = len(special_chars) / char_count if char_count > 0 else 0

        # Quality score (0-1)
        quality_score = self._compute_quality_score(
            char_count, word_count, sentence_count, special_char_ratio
        )

        return DocumentStats(
            document_id=doc_id,
            char_count=char_count,
            word_count=word_count,
            sentence_count=sentence_count,
            avg_word_length=avg_word_length,
            special_char_ratio=special_char_ratio,
            quality_score=quality_score,
        )

    def _compute_quality_score(
        self,
        char_count: int,
        word_count: int,
        sentence_count: int,
        special_char_ratio: float,
    ) -> float:
        """
        Compute quality score (0-1) based on various factors.

        Higher score = better quality.
        """
        score = 1.0

        # Length penalty
        if char_count < self.min_length:
            score *= 0.3
        elif char_count > self.max_length:
            score *= 0.7

        # Word count penalty
        if word_count < self.min_word_count:
            score *= 0.4

        # Special character penalty
        if special_char_ratio > self.max_special_char_ratio:
            penalty = min(1.0, special_char_ratio / self.max_special_char_ratio)
            score *= (1.0 - penalty * 0.5)

        # Sentence structure bonus
        if sentence_count > 0 and word_count > 0:
            words_per_sentence = word_count / sentence_count
            if 5 <= words_per_sentence <= 25:
                score *= 1.1  # Good sentence structure

        return min(1.0, max(0.0, score))

    def _detect_issues(
        self,
        doc_id: str,
        text: str,
        stats: DocumentStats,
    ) -> List[QualityIssue]:
        """Detect quality issues in a document."""
        issues = []

        # Too short
        if stats.char_count < self.min_length:
            issues.append(
                QualityIssue(
                    document_id=doc_id,
                    issue_type=QualityIssueType.TOO_SHORT,
                    message=f"Document too short ({stats.char_count} chars < {self.min_length})",
                    severity="error",
                    metadata={"char_count": stats.char_count},
                )
            )

        # Too long
        if stats.char_count > self.max_length:
            issues.append(
                QualityIssue(
                    document_id=doc_id,
                    issue_type=QualityIssueType.TOO_LONG,
                    message=f"Document too long ({stats.char_count} chars > {self.max_length})",
                    severity="warning",
                    metadata={"char_count": stats.char_count},
                )
            )

        # Low word count
        if stats.word_count < self.min_word_count:
            issues.append(
                QualityIssue(
                    document_id=doc_id,
                    issue_type=QualityIssueType.LOW_WORD_COUNT,
                    message=f"Low word count ({stats.word_count} < {self.min_word_count})",
                    severity="error",
                    metadata={"word_count": stats.word_count},
                )
            )

        # High special character ratio
        if stats.special_char_ratio > self.max_special_char_ratio:
            issues.append(
                QualityIssue(
                    document_id=doc_id,
                    issue_type=QualityIssueType.HIGH_SPECIAL_CHAR_RATIO,
                    message=f"High special char ratio ({stats.special_char_ratio:.2%})",
                    severity="warning",
                    metadata={"special_char_ratio": stats.special_char_ratio},
                )
            )

        return issues

    def get_length_histogram_data(
        self,
        document_stats: List[DocumentStats],
        bins: int = 20,
    ) -> Dict:
        """
        Prepare data for length histogram visualization.

        Args:
            document_stats: List of document statistics
            bins: Number of histogram bins

        Returns:
            Dict with histogram data for Plotly
        """
        lengths = [s.char_count for s in document_stats]

        if not lengths:
            return {"x": [], "counts": []}

        hist, bin_edges = np.histogram(lengths, bins=bins)

        return {
            "x": [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(len(hist))],
            "counts": hist.tolist(),
            "bin_edges": bin_edges.tolist(),
        }

    def get_quality_distribution_data(
        self,
        document_stats: List[DocumentStats],
    ) -> Dict:
        """
        Prepare data for quality score distribution visualization.

        Args:
            document_stats: List of document statistics

        Returns:
            Dict with distribution data
        """
        scores = [s.quality_score for s in document_stats]

        if not scores:
            return {"low": 0, "medium": 0, "high": 0}

        return {
            "low": sum(1 for s in scores if s < 0.5),
            "medium": sum(1 for s in scores if 0.5 <= s < 0.8),
            "high": sum(1 for s in scores if s >= 0.8),
        }
