"""Tests for hard-distractor analysis and failure classification."""

from dataclasses import dataclass, field
from typing import List, Set

import pytest

from src.scanner.distractor_analyzer import (
    DistractorAnalyzer,
    classify_retrieval_failures,
)


@dataclass
class FakeQueryResult:
    query_id: str
    retrieved_ids: List[str]
    relevant_ids: Set[str]


@dataclass
class FakeEvalResult:
    query_results: List[FakeQueryResult]


class FakeEvaluator:
    """Returns pre-baked retrieval results."""

    def __init__(self, query_results):
        self._result = FakeEvalResult(query_results)

    def evaluate(self, **kwargs):
        return self._result


class TestDistractorAnalyzer:
    def test_displacing_doc_is_flagged(self):
        # distractor가 1~2위, 정답이 3위 -> distractor 2건 기록
        qr = FakeQueryResult("q1", ["bad1", "bad2", "gold1"], {"gold1"})
        analyzer = DistractorAnalyzer(FakeEvaluator([qr]), k=10)
        report = analyzer.analyze([{"query_id": "q1"}], namespace="ns")

        assert report.queries_with_displacement == 1
        assert set(report.distractors.keys()) == {"bad1", "bad2"}
        assert report.distractors["bad1"]["queries"] == ["q1"]

    def test_answer_at_rank1_no_displacement(self):
        qr = FakeQueryResult("q1", ["gold1", "bad1"], {"gold1"})
        report = DistractorAnalyzer(FakeEvaluator([qr]), k=10).analyze(
            [{"query_id": "q1"}], namespace="ns"
        )
        assert report.queries_with_displacement == 0
        assert report.distractors == {}

    def test_repeat_offender_ranked_first(self):
        qrs = [
            FakeQueryResult("q1", ["bad", "gold1"], {"gold1"}),
            FakeQueryResult("q2", ["bad", "gold2"], {"gold2"}),
            FakeQueryResult("q3", ["other", "gold3"], {"gold3"}),
        ]
        report = DistractorAnalyzer(FakeEvaluator(qrs), k=10).analyze(
            [{"query_id": f"q{i}"} for i in range(3)], namespace="ns"
        )
        top = report.top_distractors
        assert top[0]["doc_id"] == "bad"
        assert top[0]["displacements"] == 2

    def test_answer_not_retrieved_all_topk_are_distractors(self):
        qr = FakeQueryResult("q1", ["a", "b"], {"gold1"})  # 정답 미회수
        report = DistractorAnalyzer(FakeEvaluator([qr]), k=10).analyze(
            [{"query_id": "q1"}], namespace="ns"
        )
        assert set(report.distractors.keys()) == {"a", "b"}


class TestFailureClassification:
    def test_fp1_missing_content(self):
        # 정답 문서가 코퍼스에 아예 없음 (과도한 클리닝 시나리오)
        qr = FakeQueryResult("q1", ["x"], {"gold1"})
        out = classify_retrieval_failures([qr], corpus_doc_ids={"x", "y"}, k=10)
        assert out["fp1_missing_content"] == ["q1"]

    def test_fp2_missed_top_ranked(self):
        qr = FakeQueryResult("q1", ["x", "y"], {"gold1"})
        out = classify_retrieval_failures(
            [qr], corpus_doc_ids={"x", "y", "gold1"}, k=10
        )
        assert out["fp2_missed_top_ranked"] == ["q1"]

    def test_ok(self):
        qr = FakeQueryResult("q1", ["gold1"], {"gold1"})
        out = classify_retrieval_failures([qr], corpus_doc_ids={"gold1"}, k=10)
        assert out["ok"] == ["q1"]

    def test_full_ground_truth_overrides_filtered(self):
        # arm 필터로 relevant가 비어도, 전체 GT로 FP1을 판정
        qr = FakeQueryResult("q1", ["x"], set())
        out = classify_retrieval_failures(
            [qr],
            corpus_doc_ids={"x"},
            k=10,
            full_ground_truth={"q1": {"gold1"}},
        )
        assert out["fp1_missing_content"] == ["q1"]
