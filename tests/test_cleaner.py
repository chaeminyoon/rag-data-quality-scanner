"""Tests for DataCleaner strategies."""

import pytest

from src.scanner.cleaner import CleaningStrategy, DataCleaner
from src.scanner.noise_detector import DuplicateCluster, NoiseReport
from src.scanner.text_analyzer import TextAnalyzer

import numpy as np


@pytest.fixture
def cleaner():
    return DataCleaner(min_length=10, duplicate_threshold=0.92)


def make_noise_report(clusters, doc_ids):
    return NoiseReport(
        total_documents=len(doc_ids),
        duplicate_clusters=clusters,
        similarity_matrix=np.eye(len(doc_ids)),
        noise_scores=np.zeros(len(doc_ids)),
        document_ids=doc_ids,
    )


class TestConservative:
    def test_removes_exact_duplicates(self, cleaner):
        docs = [
            {"id": "d1", "text": "동일한 문서 내용입니다."},
            {"id": "d2", "text": "동일한  문서   내용입니다."},  # whitespace-only diff
            {"id": "d3", "text": "완전히 다른 문서입니다."},
        ]
        result = cleaner.clean(docs, strategy=CleaningStrategy.CONSERVATIVE)
        kept_ids = {d["id"] for d in result.cleaned_documents}
        assert kept_ids == {"d1", "d3"}
        assert result.removal_reasons["exact_duplicate"] == ["d2"]

    def test_does_not_remove_near_duplicates(self, cleaner):
        # Near-duplicate (one word differs) must survive CONSERVATIVE
        docs = [
            {"id": "d1", "text": "RAG는 검색 증강 생성 기술입니다."},
            {"id": "d2", "text": "RAG는 검색 증강 생성 기법입니다."},
        ]
        cluster = DuplicateCluster(
            cluster_id=0, document_ids=["d1", "d2"], representative_id="d1"
        )
        report = make_noise_report([cluster], ["d1", "d2"])
        result = cleaner.clean(
            docs, noise_report=report, strategy=CleaningStrategy.CONSERVATIVE
        )
        assert result.removed_count == 0


class TestModerate:
    def test_keeps_clean_original_over_perturbed_copy(self, cleaner):
        # 교란된 사본(이중 공백·구두점 앞 공백)이 원본보다 길어도 원본을 유지
        docs = [
            {"id": "d1", "text": "RAG는 검색 증강 생성 기술입니다."},
            {"id": "d2", "text": "참고: RAG는  검색 증강 생성 기술입니다 ."},
        ]
        cluster = DuplicateCluster(
            cluster_id=0, document_ids=["d1", "d2"], representative_id="d2"
        )
        report = make_noise_report([cluster], ["d1", "d2"])
        result = cleaner.clean(
            docs, noise_report=report, strategy=CleaningStrategy.MODERATE
        )
        kept_ids = {d["id"] for d in result.cleaned_documents}
        assert kept_ids == {"d1"}

    def test_tie_breaks_to_shorter_copy(self, cleaner):
        # 인공물이 없으면 더 짧은(군더더기 적은) 사본 유지
        docs = [
            {"id": "d1", "text": "RAG는 검색 증강 생성입니다."},
            {"id": "d2", "text": "RAG는 검색 증강 생성입니다. 추가 안내는 문서를 참고하세요."},
        ]
        cluster = DuplicateCluster(
            cluster_id=0, document_ids=["d1", "d2"], representative_id="d2"
        )
        report = make_noise_report([cluster], ["d1", "d2"])
        result = cleaner.clean(
            docs, noise_report=report, strategy=CleaningStrategy.MODERATE
        )
        kept_ids = {d["id"] for d in result.cleaned_documents}
        assert kept_ids == {"d1"}

    def test_removes_short_documents(self, cleaner):
        docs = [
            {"id": "d1", "text": "충분히 긴 정상적인 문서 내용입니다."},
            {"id": "d2", "text": "ㅎㅎ"},
        ]
        analyzer = TextAnalyzer(min_length=10)
        analysis = analyzer.analyze(docs)
        result = cleaner.clean(
            docs, text_analysis=analysis, strategy=CleaningStrategy.MODERATE
        )
        kept_ids = {d["id"] for d in result.cleaned_documents}
        assert kept_ids == {"d1"}
        assert "too_short" in result.removal_reasons


class TestAggressive:
    def test_removes_special_char_documents(self, cleaner):
        docs = [
            {"id": "d1", "text": "정상적인 문서 내용이 여기에 있습니다."},
            {"id": "d2", "text": "!!!@#$%^&*()@#$%^&*()"},
        ]
        analyzer = TextAnalyzer(min_length=5)
        analysis = analyzer.analyze(docs)
        result = cleaner.clean(
            docs, text_analysis=analysis, strategy=CleaningStrategy.AGGRESSIVE
        )
        kept_ids = {d["id"] for d in result.cleaned_documents}
        assert "d2" not in kept_ids


class TestResultAccounting:
    def test_counts_are_consistent(self, cleaner):
        docs = [{"id": f"d{i}", "text": f"문서 {i} 내용은 서로 전부 다릅니다 " * 3} for i in range(5)]
        result = cleaner.clean(docs, strategy=CleaningStrategy.MODERATE)
        assert result.original_count == 5
        assert result.cleaned_count + result.removed_count == result.original_count
