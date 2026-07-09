"""Tests for BM25 retrieval, RRF fusion, and MinHash candidate generation."""

import pytest

from src.retrieval import BM25Retriever, reciprocal_rank_fusion
from src.retrieval.bm25 import tokenize
from src.scanner.minhash_dedup import char_shingles, find_candidate_pairs


class TestTokenizer:
    def test_korean_bigrams_included(self):
        tokens = tokenize("클라우드포트의 저장 용량")
        assert "클라우드포트의" in tokens
        assert "클라" in tokens  # 문자 바이그램
        assert "용량" in tokens

    def test_particle_robustness(self):
        # 조사가 붙어도 바이그램으로 겹침이 생겨 매칭 가능
        with_particle = set(tokenize("클라우드포트의"))
        without = set(tokenize("클라우드포트"))
        assert with_particle & without


class TestBM25Retriever:
    @pytest.fixture
    def retriever(self):
        docs = [
            {"id": "d1", "text": "클라우드포트의 API 요청 한도는 분당 100회입니다."},
            {"id": "d2", "text": "메일허브의 저장 용량은 50GB입니다."},
            {"id": "d3", "text": "김치찌개 레시피는 돼지고기를 먼저 볶는 것이 핵심입니다."},
        ]
        return BM25Retriever(docs)

    def test_exact_term_match_ranks_first(self, retriever):
        results = retriever.query("클라우드포트 API 한도", top_k=3)
        assert results[0]["id"] == "d1"

    def test_result_format(self, retriever):
        results = retriever.query("메일허브 용량", top_k=1)
        assert set(results[0].keys()) == {"id", "score", "metadata"}
        assert "text" in results[0]["metadata"]

    def test_empty_corpus(self):
        r = BM25Retriever([])
        assert r.query("아무거나") == []


class TestRRF:
    def test_doc_in_both_lists_wins(self):
        dense = [
            {"id": "a", "score": 0.9, "metadata": {}},
            {"id": "b", "score": 0.8, "metadata": {}},
        ]
        sparse = [
            {"id": "b", "score": 5.0, "metadata": {}},
            {"id": "c", "score": 4.0, "metadata": {}},
        ]
        fused = reciprocal_rank_fusion([dense, sparse], top_k=3)
        assert fused[0]["id"] == "b"  # 양쪽 모두 등장 -> RRF 최고점

    def test_top_k_respected(self):
        lists = [[{"id": str(i), "score": 1.0, "metadata": {}} for i in range(10)]]
        assert len(reciprocal_rank_fusion(lists, top_k=3)) == 3

    def test_empty_lists(self):
        assert reciprocal_rank_fusion([[], []]) == []


class TestMinHash:
    def test_shingles(self):
        s = char_shingles("중복 문서 탐지", k=4)
        assert "중복 문" in s

    def test_near_duplicates_are_candidates(self):
        texts = [
            "클라우드포트의 API 요청 한도는 분당 100회입니다. 초과하면 제한됩니다.",
            "참고: 클라우드포트의 API 요청 한도는 분당 100회입니다. 초과하면  제한됩니다.",
            "메일허브의 저장 용량은 50GB이며 추가 구매가 가능합니다.",
        ]
        pairs = find_candidate_pairs(texts, jaccard_threshold=0.5)
        assert (0, 1) in pairs
        assert (0, 2) not in pairs

    def test_same_template_different_fact_not_candidates(self):
        # 같은 문장 구조지만 핵심 값(서비스명·수치)이 다른 문서 —
        # 임베딩은 혼동하지만 자카드는 낮아 후보로 제안되지 않아야 함
        texts = [
            "알림봇의 세션 만료 시간은 30분이며, 만료 후 재인증이 필요합니다. 자세한 내용은 문서를 참고하세요.",
            "폼빌더의 백업 주기는 매일 자정이며, 백업본은 30일 보관됩니다. 정책은 변경될 수 있습니다.",
        ]
        pairs = find_candidate_pairs(texts, jaccard_threshold=0.5)
        assert pairs == {}
