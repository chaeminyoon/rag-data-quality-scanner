"""
Main data quality scanner orchestrator.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from config import get_logger
from src.embeddings import CohereClient
from .noise_detector import NoiseDetector, NoiseReport
from .text_analyzer import TextAnalyzer, TextAnalysisResult
from .cleaner import DataCleaner, CleaningStrategy, CleanedDataResult


logger = get_logger("scanner.scanner")


@dataclass
class ScanResult:
    """Complete results of data quality scan."""

    total_documents: int
    embeddings: np.ndarray
    noise_report: NoiseReport
    text_analysis: TextAnalysisResult
    overall_quality_score: float

    @property
    def summary(self) -> Dict:
        return {
            "total_documents": self.total_documents,
            "duplicate_clusters": len(self.noise_report.duplicate_clusters),
            "documents_in_duplicates": self.noise_report.total_duplicates,
            "duplicate_percentage": f"{self.noise_report.duplicate_percentage:.1f}%",
            "quality_issues": self.text_analysis.total_issues,
            "avg_text_quality": f"{self.text_analysis.avg_quality_score:.2f}",
            "overall_quality_score": f"{self.overall_quality_score:.2f}",
        }

    @property
    def issues_breakdown(self) -> Dict:
        breakdown = {
            "duplicates": self.noise_report.unique_duplicates,
        }
        for issue_type, count in self.text_analysis.issues_by_type.items():
            breakdown[issue_type.value] = count
        return breakdown


class DataQualityScanner:
    """
    Orchestrate complete data quality scanning pipeline.

    Coordinates:
    1. Embedding generation with Cohere
    2. Duplicate/noise detection
    3. Text quality analysis
    4. Data cleaning

    This is the main entry point for the scanning functionality.
    """

    def __init__(
        self,
        cohere_client: Optional[CohereClient] = None,
        noise_detector: Optional[NoiseDetector] = None,
        text_analyzer: Optional[TextAnalyzer] = None,
        cleaner: Optional[DataCleaner] = None,
    ):
        """
        Initialize the scanner with optional custom components.

        Args:
            cohere_client: Cohere API client (created if not provided)
            noise_detector: Noise detection component
            text_analyzer: Text analysis component
            cleaner: Data cleaning component
        """
        self.cohere = cohere_client or CohereClient()
        self.noise_detector = noise_detector or NoiseDetector()
        self.text_analyzer = text_analyzer or TextAnalyzer()
        self.cleaner = cleaner or DataCleaner()

        logger.info("Initialized DataQualityScanner")

    def scan(
        self,
        documents: List[Dict],
        text_key: str = "text",
        id_key: str = "id",
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> ScanResult:
        """
        Execute full quality scan on documents.

        Args:
            documents: List of documents with text and ID
            text_key: Key for document text
            id_key: Key for document ID
            progress_callback: Optional callback (stage: str, progress: float)

        Returns:
            ScanResult with complete analysis
        """
        n_docs = len(documents)
        logger.info(f"Starting quality scan for {n_docs} documents")

        def emit_progress(stage: str, progress: float):
            if progress_callback:
                progress_callback(stage, progress)

        # Stage 1: Generate embeddings
        emit_progress("embedding", 0.0)
        texts = [doc.get(text_key, "") for doc in documents]
        doc_ids = [doc.get(id_key, str(i)) for i, doc in enumerate(documents)]

        embeddings = self.cohere.embed_documents(
            texts,
            progress_callback=lambda p: emit_progress("embedding", p),
        )
        emit_progress("embedding", 1.0)

        # Stage 2: Detect duplicates/noise
        emit_progress("noise_detection", 0.0)
        noise_report = self.noise_detector.detect_duplicates(
            document_ids=doc_ids,
            embeddings=embeddings,
            progress_callback=lambda p: emit_progress("noise_detection", p),
        )
        emit_progress("noise_detection", 1.0)

        # Stage 3: Analyze text quality
        emit_progress("text_analysis", 0.0)
        text_analysis = self.text_analyzer.analyze(
            documents,
            text_key=text_key,
            id_key=id_key,
        )
        emit_progress("text_analysis", 1.0)

        # Calculate overall quality score
        overall_score = self._calculate_overall_quality(noise_report, text_analysis)

        result = ScanResult(
            total_documents=n_docs,
            embeddings=embeddings,
            noise_report=noise_report,
            text_analysis=text_analysis,
            overall_quality_score=overall_score,
        )

        logger.info(f"Scan complete: overall_quality={overall_score:.2f}")

        return result

    def scan_and_clean(
        self,
        documents: List[Dict],
        strategy: CleaningStrategy = CleaningStrategy.MODERATE,
        text_key: str = "text",
        id_key: str = "id",
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> Tuple[ScanResult, CleanedDataResult]:
        """
        Scan and automatically clean documents.

        Args:
            documents: List of documents
            strategy: Cleaning aggressiveness level
            text_key: Key for document text
            id_key: Key for document ID
            progress_callback: Optional callback for progress

        Returns:
            Tuple of (ScanResult, CleanedDataResult)
        """
        # Run scan
        scan_result = self.scan(
            documents,
            text_key=text_key,
            id_key=id_key,
            progress_callback=progress_callback,
        )

        # Clean based on scan results
        if progress_callback:
            progress_callback("cleaning", 0.0)

        cleaned_result = self.cleaner.clean(
            documents=documents,
            noise_report=scan_result.noise_report,
            text_analysis=scan_result.text_analysis,
            strategy=strategy,
            text_key=text_key,
            id_key=id_key,
        )

        if progress_callback:
            progress_callback("cleaning", 1.0)

        return scan_result, cleaned_result

    def _calculate_overall_quality(
        self,
        noise_report: NoiseReport,
        text_analysis: TextAnalysisResult,
    ) -> float:
        """
        Calculate overall data quality score (0-1).

        Considers:
        - Duplicate percentage (40% weight)
        - Text quality average (40% weight)
        - Issue density (20% weight)
        """
        # Duplicate penalty
        dup_score = max(0, 1 - (noise_report.duplicate_percentage / 100) * 2)

        # Text quality score
        text_score = text_analysis.avg_quality_score

        # Issue density penalty
        if text_analysis.total_documents > 0:
            issue_ratio = text_analysis.total_issues / text_analysis.total_documents
            issue_score = max(0, 1 - issue_ratio)
        else:
            issue_score = 1.0

        # Weighted average
        overall = (0.4 * dup_score) + (0.4 * text_score) + (0.2 * issue_score)

        return round(overall, 3)

    def get_visualization_data(self, scan_result: ScanResult) -> Dict:
        """
        Prepare data for dashboard visualizations.

        Args:
            scan_result: Results from scan

        Returns:
            Dict with data for various charts
        """
        return {
            "similarity_heatmap": self.noise_detector.get_similarity_heatmap_data(
                scan_result.noise_report.similarity_matrix,
                scan_result.noise_report.document_ids,
            ),
            "length_histogram": self.text_analyzer.get_length_histogram_data(
                scan_result.text_analysis.document_stats,
            ),
            "quality_distribution": self.text_analyzer.get_quality_distribution_data(
                scan_result.text_analysis.document_stats,
            ),
            "issues_breakdown": scan_result.issues_breakdown,
            "summary": scan_result.summary,
        }

    def quick_scan(
        self,
        documents: List[Dict],
        text_key: str = "text",
        id_key: str = "id",
    ) -> Dict:
        """
        Perform quick scan without embeddings (text analysis only).

        Useful for initial assessment before full scan.

        Args:
            documents: List of documents
            text_key: Key for document text
            id_key: Key for document ID

        Returns:
            Quick analysis summary
        """
        text_analysis = self.text_analyzer.analyze(
            documents, text_key=text_key, id_key=id_key
        )

        return {
            "total_documents": len(documents),
            "avg_quality_score": text_analysis.avg_quality_score,
            "total_issues": text_analysis.total_issues,
            "issues_by_type": {
                k.value: v for k, v in text_analysis.issues_by_type.items()
            },
            "length_distribution": text_analysis.length_distribution,
            "recommendation": self._get_recommendation(text_analysis),
        }

    def _get_recommendation(self, text_analysis: TextAnalysisResult) -> str:
        """Generate recommendation based on text analysis."""
        issues = text_analysis.total_issues
        total = text_analysis.total_documents
        quality = text_analysis.avg_quality_score

        if issues == 0 and quality >= 0.8:
            return "Data quality looks good. Consider running full scan with embeddings for duplicate detection."
        elif issues / total > 0.2:
            return "High number of quality issues detected. Recommend AGGRESSIVE cleaning strategy."
        elif quality < 0.5:
            return "Low average quality score. Review short documents and special characters."
        else:
            return "Some issues detected. Recommend MODERATE cleaning strategy."
