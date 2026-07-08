"""Tests for retrieval evaluation metrics (standard-definition correctness)."""

import numpy as np
import pytest

from src.evaluator.metrics import MetricsCalculator


@pytest.fixture
def calc():
    return MetricsCalculator(k=10)


class TestNDCG:
    def test_perfect_ranking(self, calc):
        # All 3 relevant docs at the top -> NDCG = 1.0
        retrieved = ["a", "b", "c", "x", "y"]
        relevant = {"a", "b", "c"}
        assert calc.ndcg_at_k(retrieved, relevant) == pytest.approx(1.0)

    def test_partial_retrieval_is_penalized(self, calc):
        # Only 1 of 3 relevant docs retrieved (at rank 1).
        # Standard NDCG: DCG = 1/log2(2) = 1.0
        # IDCG = 1/log2(2) + 1/log2(3) + 1/log2(4) ≈ 2.1309
        # -> NDCG ≈ 0.4693. The old (buggy) implementation returned 1.0.
        retrieved = ["a", "x", "y", "z"]
        relevant = {"a", "b", "c"}
        expected = 1.0 / (1.0 / np.log2(2) + 1.0 / np.log2(3) + 1.0 / np.log2(4))
        assert calc.ndcg_at_k(retrieved, relevant) == pytest.approx(expected, abs=1e-6)
        assert calc.ndcg_at_k(retrieved, relevant) < 0.5

    def test_rank_position_matters(self, calc):
        relevant = {"a"}
        top = calc.ndcg_at_k(["a", "x", "y"], relevant)
        bottom = calc.ndcg_at_k(["x", "y", "a"], relevant)
        assert top > bottom > 0.0

    def test_no_relevant_retrieved(self, calc):
        assert calc.ndcg_at_k(["x", "y"], {"a"}) == 0.0

    def test_empty_inputs(self, calc):
        assert calc.ndcg_at_k([], {"a"}) == 0.0
        assert calc.ndcg_at_k(["a"], set()) == 0.0

    def test_k_cutoff(self, calc):
        # Relevant doc at rank 3 is invisible at k=2
        assert calc.ndcg_at_k(["x", "y", "a"], {"a"}, k=2) == 0.0


class TestMRR:
    def test_first_position(self, calc):
        assert calc.mrr(["a", "x"], {"a"}) == 1.0

    def test_third_position(self, calc):
        assert calc.mrr(["x", "y", "a"], {"a"}) == pytest.approx(1 / 3)

    def test_not_found(self, calc):
        assert calc.mrr(["x", "y"], {"a"}) == 0.0


class TestPrecisionRecall:
    def test_precision(self, calc):
        # 2 relevant in top-4 -> P@4 = 0.5
        assert calc.precision_at_k(["a", "x", "b", "y"], {"a", "b"}, k=4) == 0.5

    def test_recall(self, calc):
        # 2 of 4 relevant retrieved -> R = 0.5
        assert calc.recall_at_k(["a", "b", "x"], {"a", "b", "c", "d"}, k=10) == 0.5

    def test_hit_rate(self, calc):
        assert calc.hit_rate_at_k(["x", "a"], {"a"}, k=2) == 1.0
        assert calc.hit_rate_at_k(["x", "y"], {"a"}, k=2) == 0.0


class TestAverage:
    def test_compute_average(self, calc):
        all_retrieved = [["a", "x"], ["y", "b"]]
        all_relevant = [{"a"}, {"b"}]
        m = calc.compute_average(all_retrieved, all_relevant, k=2)
        assert m.hit_rate_at_k == 1.0
        assert m.mrr == pytest.approx((1.0 + 0.5) / 2)

    def test_mismatched_lengths_raise(self, calc):
        with pytest.raises(ValueError):
            calc.compute_average([["a"]], [])
