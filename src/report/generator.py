"""
PDF report generation using HTML templates.
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from jinja2 import Environment, FileSystemLoader

from config import get_logger

logger = get_logger("report.generator")

# Template directory
TEMPLATE_DIR = Path(__file__).parent / "templates"


@dataclass
class ReportData:
    """Data container for report generation."""

    # Meta
    generated_at: str
    customer_name: Optional[str] = None
    ticket_id: Optional[str] = None

    # Quality metrics
    total_documents: int = 0
    cleaned_documents: int = 0
    duplicates_removed: int = 0
    quality_issues: int = 0
    quality_score: float = 0.0

    # Performance metrics
    original_ndcg: float = 0.0
    cleaned_ndcg: float = 0.0
    rerank_ndcg: float = 0.0

    # Derived metrics
    ndcg_improvement: float = 0.0
    rerank_improvement: float = 0.0
    cleaning_percentage: float = 0.0
    duplicate_percentage: float = 0.0
    quality_issue_percentage: float = 0.0

    # Breakdowns
    issues_breakdown: Dict[str, int] = None
    per_query_results: List[Dict] = None

    def __post_init__(self):
        self.issues_breakdown = self.issues_breakdown or {}
        self.per_query_results = self.per_query_results or []

        # Calculate derived metrics
        if self.total_documents > 0:
            self.cleaning_percentage = (self.cleaned_documents / self.total_documents) * 100
            self.duplicate_percentage = (self.duplicates_removed / self.total_documents) * 100
            self.quality_issue_percentage = f"{(self.quality_issues / self.total_documents) * 100:.1f}"

        if self.original_ndcg > 0:
            self.ndcg_improvement = ((self.cleaned_ndcg - self.original_ndcg) / self.original_ndcg) * 100
            self.rerank_improvement = ((self.rerank_ndcg - self.original_ndcg) / self.original_ndcg) * 100


class ReportGenerator:
    """
    Generate PDF reports from scan and benchmark results.

    Uses Jinja2 HTML templates with optional WeasyPrint PDF conversion.
    """

    def __init__(self, template_dir: Path = None):
        """
        Initialize report generator.

        Args:
            template_dir: Directory containing HTML templates
        """
        self.template_dir = template_dir or TEMPLATE_DIR

        self.env = Environment(
            loader=FileSystemLoader(str(self.template_dir)),
            autoescape=True,
        )

        logger.info(f"Initialized ReportGenerator with templates from {self.template_dir}")

    def generate_html(
        self,
        scan_result,
        cleaned_result=None,
        comparison_result=None,
        customer_name: str = None,
        ticket_id: str = None,
    ) -> str:
        """
        Generate HTML report from results.

        Args:
            scan_result: Results from quality scan
            cleaned_result: Results from cleaning (optional)
            comparison_result: Results from benchmark comparison (optional)
            customer_name: Customer name for report
            ticket_id: Ticket ID for report

        Returns:
            HTML string
        """
        # Prepare report data
        data = ReportData(
            generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            customer_name=customer_name,
            ticket_id=ticket_id,
            total_documents=scan_result.total_documents,
            quality_score=scan_result.overall_quality_score,
            quality_issues=scan_result.text_analysis.total_issues,
            duplicates_removed=scan_result.noise_report.unique_duplicates,
            issues_breakdown=scan_result.issues_breakdown,
        )

        if cleaned_result:
            data.cleaned_documents = cleaned_result.cleaned_count

        if comparison_result:
            data.original_ndcg = comparison_result.original_metrics.ndcg_at_k
            data.cleaned_ndcg = comparison_result.cleaned_metrics.ndcg_at_k
            if comparison_result.cleaned_with_rerank_metrics:
                data.rerank_ndcg = comparison_result.cleaned_with_rerank_metrics.ndcg_at_k
            data.per_query_results = comparison_result.per_query_comparison

        # Recalculate derived metrics
        data.__post_init__()

        # Render template
        template = self.env.get_template("ticket_resolution.html")
        html = template.render(**data.__dict__)

        logger.info("Generated HTML report")
        return html

    def generate_pdf(
        self,
        scan_result,
        cleaned_result=None,
        comparison_result=None,
        customer_name: str = None,
        ticket_id: str = None,
    ) -> bytes:
        """
        Generate PDF report from results.

        Requires WeasyPrint to be installed.

        Args:
            scan_result: Results from quality scan
            cleaned_result: Results from cleaning (optional)
            comparison_result: Results from benchmark comparison (optional)
            customer_name: Customer name for report
            ticket_id: Ticket ID for report

        Returns:
            PDF bytes
        """
        try:
            from weasyprint import HTML
        except ImportError:
            logger.error("WeasyPrint not installed. Install with: pip install weasyprint")
            raise ImportError("WeasyPrint is required for PDF generation")

        html = self.generate_html(
            scan_result=scan_result,
            cleaned_result=cleaned_result,
            comparison_result=comparison_result,
            customer_name=customer_name,
            ticket_id=ticket_id,
        )

        pdf = HTML(string=html).write_pdf()

        logger.info("Generated PDF report")
        return pdf

    def generate_markdown(
        self,
        scan_result,
        cleaned_result=None,
        comparison_result=None,
        customer_name: str = None,
        ticket_id: str = None,
    ) -> str:
        """
        Generate Markdown report from results.

        Args:
            scan_result: Results from quality scan
            cleaned_result: Results from cleaning (optional)
            comparison_result: Results from benchmark comparison (optional)
            customer_name: Customer name for report
            ticket_id: Ticket ID for report

        Returns:
            Markdown string
        """
        generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        md = f"""# TICKET RESOLUTION REPORT

**Generated:** {generated_at}
**Customer:** {customer_name or "N/A"}
**Ticket ID:** {ticket_id or "N/A"}

---

## Executive Summary

| Metric | Value |
|--------|-------|
| Total Documents | {scan_result.total_documents} |
| Quality Score | {scan_result.overall_quality_score:.0%} |
| Duplicates Found | {scan_result.noise_report.unique_duplicates} |
| Quality Issues | {scan_result.text_analysis.total_issues} |
"""

        if cleaned_result:
            md += f"""| Documents After Cleaning | {cleaned_result.cleaned_count} |
| Documents Removed | {cleaned_result.removed_count} ({cleaned_result.removal_percentage:.1f}%) |
"""

        if comparison_result:
            md += f"""
---

## RAG Performance Comparison

| Configuration | NDCG@10 | Hit Rate@10 |
|---------------|---------|-------------|
| Original | {comparison_result.original_metrics.ndcg_at_k:.2%} | {comparison_result.original_metrics.hit_rate_at_k:.2%} |
| Cleaned | {comparison_result.cleaned_metrics.ndcg_at_k:.2%} | {comparison_result.cleaned_metrics.hit_rate_at_k:.2%} |
"""
            if comparison_result.cleaned_with_rerank_metrics:
                md += f"""| Cleaned + Rerank | {comparison_result.cleaned_with_rerank_metrics.ndcg_at_k:.2%} | {comparison_result.cleaned_with_rerank_metrics.hit_rate_at_k:.2%} |
"""
            md += f"""
**Improvement:** +{comparison_result.improvement.get("ndcg_rerank_improvement", 0):.1f}%
"""

        md += """
---

## Root Cause Analysis

### Issues Detected:
"""
        for issue_type, count in scan_result.issues_breakdown.items():
            md += f"- **{issue_type.replace('_', ' ').title()}**: {count} documents\n"

        md += """
---

## Recommendations

1. **Implement deduplication** in data ingestion pipeline
2. **Add minimum length validation** (10+ characters)
3. **Use Cohere Rerank 3.5** in production retrieval
4. **Schedule monthly data quality audits**

---

*Report generated by Embed Data Quality Scanner*
*Powered by Cohere Embed v3 & Rerank 3.5*
"""

        logger.info("Generated Markdown report")
        return md
