"""
Embed Data Quality Scanner - Streamlit Application

A CSE ticket resolution tool for diagnosing and resolving RAG accuracy issues.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Optional
import io
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_settings, setup_logging, get_logger
from src.ingest import PDFParser, CSVLoader, TextChunker, ChunkStrategy
from src.scanner import DataQualityScanner, CleaningStrategy
from src.embeddings import CohereClient
from src.vectordb import PineconeClient
from src.evaluator import RAGEvaluator

# Setup logging
setup_logging()
logger = get_logger("main")

# Page configuration
st.set_page_config(
    page_title="Embed Data Quality Scanner",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E3A5F;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #6B7280;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
    }
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    .improvement-positive {
        color: #10B981;
        font-weight: bold;
    }
    .improvement-negative {
        color: #EF4444;
        font-weight: bold;
    }
    .stProgress > div > div > div > div {
        background-color: #667eea;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables."""
    defaults = {
        "documents": None,
        "ground_truth": None,
        "scan_result": None,
        "cleaned_result": None,
        "comparison_result": None,
        "current_page": "upload",
        "cohere_connected": False,
        "pinecone_connected": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_sidebar():
    """Render sidebar with navigation and settings."""
    with st.sidebar:
        st.markdown("## 🔍 Data Quality Scanner")
        st.markdown("*CSE Ticket Resolution Tool*")
        st.divider()

        # Navigation
        st.markdown("### Navigation")
        pages = {
            "upload": "📤 1. Upload Data",
            "scan": "🔍 2. Quality Scan",
            "benchmark": "📊 3. RAG Benchmark",
            "report": "📄 4. Generate Report",
        }

        for page_id, page_name in pages.items():
            if st.button(page_name, key=f"nav_{page_id}", use_container_width=True):
                st.session_state.current_page = page_id
                st.rerun()

        st.divider()

        # Session info
        st.markdown("### Session Info")
        if st.session_state.documents:
            st.info(f"Documents: {len(st.session_state.documents)}")
        else:
            st.info("No documents loaded")

        if st.session_state.scan_result:
            st.success("Scan: Complete ✓")
        if st.session_state.cleaned_result:
            st.success("Cleaning: Complete ✓")
        if st.session_state.comparison_result:
            st.success("Benchmark: Complete ✓")

        st.divider()

        # Settings
        st.markdown("### Settings")
        duplicate_threshold = st.slider(
            "Duplicate Threshold",
            min_value=0.80,
            max_value=0.99,
            value=0.92,
            step=0.01,
            help="Cosine similarity threshold for duplicate detection"
        )
        st.session_state.duplicate_threshold = duplicate_threshold

        cleaning_strategy = st.selectbox(
            "Cleaning Strategy",
            options=["CONSERVATIVE", "MODERATE", "AGGRESSIVE"],
            index=1,
            help="How aggressively to clean the data"
        )
        st.session_state.cleaning_strategy = CleaningStrategy[cleaning_strategy]

        st.divider()

        # API Status
        st.markdown("### API Status")
        try:
            settings = get_settings()
            st.success("Cohere: Configured ✓")
            st.success("Pinecone: Configured ✓")
        except Exception as e:
            st.error("API keys not configured")
            st.caption("Set keys in .env file")


def render_upload_page():
    """Render data upload page."""
    st.markdown('<p class="main-header">📤 Upload Your Data</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Upload PDF or CSV files containing your document corpus</p>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns([2, 1])

    with col1:
        # File uploader
        uploaded_files = st.file_uploader(
            "Upload documents",
            type=["pdf", "csv"],
            accept_multiple_files=True,
            help="Supported formats: PDF, CSV"
        )

        if uploaded_files:
            st.success(f"Uploaded {len(uploaded_files)} file(s)")

            # Process files
            documents = []
            for file in uploaded_files:
                if file.name.endswith(".pdf"):
                    parser = PDFParser()
                    docs = parser.parse(file, filename=file.name)
                    documents.extend([{"id": d.id, "text": d.text, "metadata": d.metadata} for d in docs])
                elif file.name.endswith(".csv"):
                    # Show column mapping
                    st.markdown("#### CSV Column Mapping")
                    loader = CSVLoader()
                    columns = loader.get_columns(file)
                    file.seek(0)

                    text_col = st.selectbox("Text Column", columns, key=f"text_{file.name}")
                    id_col = st.selectbox("ID Column (optional)", ["None"] + columns, key=f"id_{file.name}")
                    id_col = None if id_col == "None" else id_col

                    docs = loader.load_documents(file, text_column=text_col, id_column=id_col)
                    documents.extend([{"id": d.id, "text": d.text, "metadata": d.metadata} for d in docs])

            if documents:
                st.session_state.documents = documents
                st.success(f"Loaded {len(documents)} documents")

                # Preview
                with st.expander("Preview Documents"):
                    preview_df = pd.DataFrame([
                        {"ID": d["id"][:30], "Text": d["text"][:200] + "..."}
                        for d in documents[:10]
                    ])
                    st.dataframe(preview_df, use_container_width=True)

    with col2:
        # Ground truth upload
        st.markdown("#### Ground Truth (Optional)")
        st.markdown("Upload for benchmarking")

        gt_file = st.file_uploader(
            "Upload queries CSV",
            type=["csv"],
            help="CSV with 'query' and 'relevant_doc_ids' columns",
            key="gt_upload"
        )

        if gt_file:
            loader = CSVLoader()
            try:
                ground_truth = loader.load_ground_truth(gt_file)
                st.session_state.ground_truth = [
                    {"query_id": g.query_id, "query": g.query, "relevant_doc_ids": g.relevant_doc_ids}
                    for g in ground_truth
                ]
                st.success(f"Loaded {len(ground_truth)} queries")
            except Exception as e:
                st.error(f"Error: {e}")

    # Navigation button
    st.divider()
    if st.session_state.documents:
        if st.button("Proceed to Quality Scan →", type="primary", use_container_width=True):
            st.session_state.current_page = "scan"
            st.rerun()


def render_scan_page():
    """Render quality scan page."""
    st.markdown('<p class="main-header">🔍 Data Quality Analysis</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Analyzing your documents for duplicates, noise, and quality issues</p>',
        unsafe_allow_html=True
    )

    if not st.session_state.documents:
        st.warning("Please upload documents first")
        return

    # Run scan button
    if st.button("Run Quality Scan", type="primary", use_container_width=True):
        with st.spinner("Scanning documents..."):
            progress_bar = st.progress(0)
            status_text = st.empty()

            def progress_callback(stage: str, progress: float):
                stage_names = {
                    "embedding": "Generating embeddings",
                    "noise_detection": "Detecting duplicates",
                    "text_analysis": "Analyzing text quality",
                    "cleaning": "Cleaning data",
                }
                status_text.text(f"{stage_names.get(stage, stage)}...")
                # Weighted progress across stages
                stage_weights = {"embedding": 0.5, "noise_detection": 0.3, "text_analysis": 0.2}
                base_progress = sum(
                    stage_weights.get(s, 0) for s in stage_weights if s < stage
                )
                total_progress = base_progress + (stage_weights.get(stage, 0.2) * progress)
                progress_bar.progress(min(total_progress, 1.0))

            try:
                scanner = DataQualityScanner()
                scan_result, cleaned_result = scanner.scan_and_clean(
                    documents=st.session_state.documents,
                    strategy=st.session_state.get("cleaning_strategy", CleaningStrategy.MODERATE),
                    progress_callback=progress_callback,
                )
                st.session_state.scan_result = scan_result
                st.session_state.cleaned_result = cleaned_result
                progress_bar.progress(1.0)
                status_text.text("Scan complete!")
                st.success("Quality scan completed successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Scan failed: {e}")
                logger.exception("Scan failed")

    # Display results if available
    if st.session_state.scan_result:
        scan_result = st.session_state.scan_result
        cleaned_result = st.session_state.cleaned_result

        # Metrics row
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Quality Score",
                f"{scan_result.overall_quality_score:.0%}",
                help="Overall data quality score"
            )
        with col2:
            st.metric(
                "Total Documents",
                scan_result.total_documents,
            )
        with col3:
            st.metric(
                "Duplicates Found",
                scan_result.noise_report.unique_duplicates,
                f"-{scan_result.noise_report.duplicate_percentage:.1f}%"
            )
        with col4:
            st.metric(
                "Quality Issues",
                scan_result.text_analysis.total_issues,
            )

        st.divider()

        # Charts row
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Issue Breakdown")
            issues = scan_result.issues_breakdown
            if issues:
                fig = px.pie(
                    names=list(issues.keys()),
                    values=list(issues.values()),
                    color_discrete_sequence=px.colors.qualitative.Set2,
                )
                fig.update_layout(margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No issues detected!")

        with col2:
            st.markdown("#### Document Length Distribution")
            length_data = scan_result.text_analysis.length_distribution
            fig = px.bar(
                x=list(length_data.keys()),
                y=list(length_data.values()),
                labels={"x": "Length Category", "y": "Count"},
                color_discrete_sequence=["#667eea"],
            )
            fig.update_layout(margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig, use_container_width=True)

        # Similarity heatmap
        st.markdown("#### Similarity Heatmap")
        st.caption("Red areas indicate potential duplicates (high similarity)")

        heatmap_data = scan_result.noise_report.similarity_matrix
        # Subsample for display
        max_display = 50
        if len(heatmap_data) > max_display:
            indices = np.linspace(0, len(heatmap_data) - 1, max_display, dtype=int)
            display_matrix = heatmap_data[np.ix_(indices, indices)]
        else:
            display_matrix = heatmap_data

        fig = px.imshow(
            display_matrix,
            color_continuous_scale="RdBu_r",
            zmin=0,
            zmax=1,
        )
        fig.update_layout(
            margin=dict(l=20, r=20, t=30, b=20),
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Cleaning summary
        if cleaned_result:
            st.divider()
            st.markdown("#### Cleaning Summary")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Original", cleaned_result.original_count)
            with col2:
                st.metric("Cleaned", cleaned_result.cleaned_count)
            with col3:
                st.metric(
                    "Removed",
                    cleaned_result.removed_count,
                    f"-{cleaned_result.removal_percentage:.1f}%"
                )

            if cleaned_result.removal_reasons:
                with st.expander("Removal Breakdown"):
                    for reason, doc_ids in cleaned_result.removal_reasons.items():
                        st.write(f"**{reason}**: {len(doc_ids)} documents")

        # Navigation
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            if st.button("← Back to Upload", use_container_width=True):
                st.session_state.current_page = "upload"
                st.rerun()
        with col2:
            if st.session_state.ground_truth and st.button("Proceed to Benchmark →", type="primary", use_container_width=True):
                st.session_state.current_page = "benchmark"
                st.rerun()
            elif not st.session_state.ground_truth:
                st.info("Upload ground truth queries for benchmarking")


def render_benchmark_page():
    """Render RAG benchmark page."""
    st.markdown('<p class="main-header">📊 RAG Performance Benchmark</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Comparing retrieval performance before and after data cleaning</p>',
        unsafe_allow_html=True
    )

    if not st.session_state.scan_result:
        st.warning("Please run quality scan first")
        return

    if not st.session_state.ground_truth:
        st.warning("Please upload ground truth queries for benchmarking")
        return

    # Run benchmark button
    if st.button("Run Benchmark", type="primary", use_container_width=True):
        with st.spinner("Running benchmark..."):
            progress_bar = st.progress(0)
            status_text = st.empty()

            def progress_callback(stage: str, progress: float):
                stage_names = {
                    "indexing_original": "Indexing original data",
                    "indexing_cleaned": "Indexing cleaned data",
                    "evaluating_original": "Evaluating original",
                    "evaluating_cleaned": "Evaluating cleaned",
                    "evaluating_rerank": "Evaluating with rerank",
                }
                status_text.text(f"{stage_names.get(stage, stage)}...")
                stages = list(stage_names.keys())
                if stage in stages:
                    base = stages.index(stage) / len(stages)
                    progress_bar.progress(base + (progress / len(stages)))

            try:
                scan_result = st.session_state.scan_result
                cleaned_result = st.session_state.cleaned_result

                # Re-embed cleaned documents
                cohere = CohereClient()
                cleaned_texts = [d["text"] for d in cleaned_result.cleaned_documents]
                cleaned_embeddings = cohere.embed_documents(cleaned_texts)

                evaluator = RAGEvaluator()
                comparison = evaluator.compare(
                    queries=st.session_state.ground_truth,
                    original_documents=st.session_state.documents,
                    original_embeddings=scan_result.embeddings,
                    cleaned_documents=cleaned_result.cleaned_documents,
                    cleaned_embeddings=cleaned_embeddings,
                    progress_callback=progress_callback,
                )

                st.session_state.comparison_result = comparison
                progress_bar.progress(1.0)
                status_text.text("Benchmark complete!")
                st.success("Benchmark completed successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Benchmark failed: {e}")
                logger.exception("Benchmark failed")

    # Display results if available
    if st.session_state.comparison_result:
        comparison = st.session_state.comparison_result

        # Main metrics comparison
        st.markdown("#### Performance Comparison")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Original**")
            st.metric(
                f"NDCG@{comparison.original_metrics.k}",
                f"{comparison.original_metrics.ndcg_at_k:.2%}"
            )
            st.metric(
                f"Hit Rate@{comparison.original_metrics.k}",
                f"{comparison.original_metrics.hit_rate_at_k:.2%}"
            )

        with col2:
            st.markdown("**Cleaned**")
            ndcg_delta = comparison.improvement.get("ndcg_improvement", 0)
            hr_delta = comparison.improvement.get("hit_rate_improvement", 0)
            st.metric(
                f"NDCG@{comparison.cleaned_metrics.k}",
                f"{comparison.cleaned_metrics.ndcg_at_k:.2%}",
                f"+{ndcg_delta:.1f}%"
            )
            st.metric(
                f"Hit Rate@{comparison.cleaned_metrics.k}",
                f"{comparison.cleaned_metrics.hit_rate_at_k:.2%}",
                f"+{hr_delta:.1f}%"
            )

        with col3:
            if comparison.cleaned_with_rerank_metrics:
                st.markdown("**Cleaned + Rerank**")
                rerank_improvement = comparison.improvement.get("ndcg_rerank_improvement", 0)
                st.metric(
                    f"NDCG@{comparison.cleaned_with_rerank_metrics.k}",
                    f"{comparison.cleaned_with_rerank_metrics.ndcg_at_k:.2%}",
                    f"+{rerank_improvement:.1f}%"
                )
                st.metric(
                    f"Hit Rate@{comparison.cleaned_with_rerank_metrics.k}",
                    f"{comparison.cleaned_with_rerank_metrics.hit_rate_at_k:.2%}"
                )

        st.divider()

        # Bar chart comparison
        st.markdown("#### NDCG Comparison")

        metrics_data = {
            "Configuration": ["Original", "Cleaned", "Cleaned + Rerank"],
            "NDCG@10": [
                comparison.original_metrics.ndcg_at_k,
                comparison.cleaned_metrics.ndcg_at_k,
                comparison.cleaned_with_rerank_metrics.ndcg_at_k if comparison.cleaned_with_rerank_metrics else 0,
            ],
        }

        fig = px.bar(
            x=metrics_data["Configuration"],
            y=metrics_data["NDCG@10"],
            color=metrics_data["Configuration"],
            color_discrete_sequence=["#EF4444", "#10B981", "#3B82F6"],
        )
        fig.update_layout(
            showlegend=False,
            yaxis_title="NDCG@10",
            yaxis_range=[0, 1],
            margin=dict(l=20, r=20, t=30, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Per-query breakdown
        if comparison.per_query_comparison:
            st.markdown("#### Per-Query Breakdown")

            df = pd.DataFrame(comparison.per_query_comparison)
            df["improvement"] = (
                (df["rerank_ndcg"] - df["original_ndcg"]) / df["original_ndcg"] * 100
            ).round(1)
            df["improvement"] = df["improvement"].apply(lambda x: f"+{x:.1f}%" if x > 0 else f"{x:.1f}%")

            st.dataframe(
                df[["query", "original_ndcg", "cleaned_ndcg", "rerank_ndcg", "improvement"]].rename(
                    columns={
                        "query": "Query",
                        "original_ndcg": "Original",
                        "cleaned_ndcg": "Cleaned",
                        "rerank_ndcg": "With Rerank",
                        "improvement": "Improvement",
                    }
                ),
                use_container_width=True,
            )

        # Navigation
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            if st.button("← Back to Scan", use_container_width=True):
                st.session_state.current_page = "scan"
                st.rerun()
        with col2:
            if st.button("Generate Report →", type="primary", use_container_width=True):
                st.session_state.current_page = "report"
                st.rerun()


def render_report_page():
    """Render report generation page."""
    st.markdown('<p class="main-header">📄 Ticket Resolution Report</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Generate a professional report for ticket documentation</p>',
        unsafe_allow_html=True
    )

    if not st.session_state.scan_result:
        st.warning("Please complete the quality scan first")
        return

    # Report options
    col1, col2 = st.columns(2)

    with col1:
        customer_name = st.text_input("Customer Name", placeholder="ACME Corp")
        ticket_id = st.text_input("Ticket ID", placeholder="P1-2025-0001")

    with col2:
        st.markdown("#### Include in Report")
        include_duplicates = st.checkbox("Duplicate List", value=True)
        include_query_breakdown = st.checkbox("Query Breakdown", value=True)
        include_recommendations = st.checkbox("Recommendations", value=True)

    st.divider()

    # Report preview
    st.markdown("### Report Preview")

    scan_result = st.session_state.scan_result
    cleaned_result = st.session_state.cleaned_result
    comparison = st.session_state.comparison_result

    # Build report content
    report_content = f"""
# TICKET RESOLUTION REPORT

**Generated:** {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}
**Customer:** {customer_name or "N/A"}
**Ticket ID:** {ticket_id or "N/A"}

---

## EXECUTIVE SUMMARY

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Data Quality Score | {scan_result.overall_quality_score:.0%} | - | - |
| Total Documents | {scan_result.total_documents} | {cleaned_result.cleaned_count if cleaned_result else "N/A"} | {f"-{cleaned_result.removed_count}" if cleaned_result else "N/A"} |
"""

    if comparison:
        report_content += f"""| NDCG@10 | {comparison.original_metrics.ndcg_at_k:.2%} | {comparison.cleaned_with_rerank_metrics.ndcg_at_k:.2%} | +{comparison.improvement.get("ndcg_rerank_improvement", 0):.1f}% |
| Hit Rate@10 | {comparison.original_metrics.hit_rate_at_k:.2%} | {comparison.cleaned_with_rerank_metrics.hit_rate_at_k:.2%} | - |
"""

    report_content += """
---

## ROOT CAUSE ANALYSIS

### Data Quality Issues Detected:
"""

    for issue_type, count in scan_result.issues_breakdown.items():
        report_content += f"- **{issue_type}**: {count} documents\n"

    if include_recommendations:
        report_content += """
---

## RECOMMENDATIONS

1. **Implement deduplication** in data ingestion pipeline to prevent future duplicates
2. **Add minimum length validation** (10+ characters) to filter meaningless content
3. **Use Cohere Rerank 3.5** in production retrieval for optimal accuracy
4. **Schedule monthly data quality audits** using this scanner tool
"""

    report_content += """
---

*Report generated by Embed Data Quality Scanner*
*Powered by Cohere Embed v3 & Rerank 3.5*
"""

    st.markdown(report_content)

    st.divider()

    # Download buttons
    col1, col2 = st.columns(2)

    with col1:
        st.download_button(
            label="📥 Download Report (Markdown)",
            data=report_content,
            file_name=f"ticket_resolution_report_{ticket_id or 'report'}.md",
            mime="text/markdown",
            use_container_width=True,
        )

    with col2:
        # CSV export of scan results
        if cleaned_result:
            removed_df = pd.DataFrame([
                {"id": d["id"], "reason": "quality_issue", "text": d["text"][:100]}
                for d in cleaned_result.removed_documents
            ])
            csv_buffer = io.StringIO()
            removed_df.to_csv(csv_buffer, index=False)

            st.download_button(
                label="📥 Download Removed Documents (CSV)",
                data=csv_buffer.getvalue(),
                file_name=f"removed_documents_{ticket_id or 'export'}.csv",
                mime="text/csv",
                use_container_width=True,
            )

    # Navigation
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Back to Benchmark", use_container_width=True):
            st.session_state.current_page = "benchmark"
            st.rerun()
    with col2:
        if st.button("Start New Session", type="primary", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()


def main():
    """Main application entry point."""
    init_session_state()
    render_sidebar()

    # Route to current page
    page = st.session_state.get("current_page", "upload")

    if page == "upload":
        render_upload_page()
    elif page == "scan":
        render_scan_page()
    elif page == "benchmark":
        render_benchmark_page()
    elif page == "report":
        render_report_page()


if __name__ == "__main__":
    main()
