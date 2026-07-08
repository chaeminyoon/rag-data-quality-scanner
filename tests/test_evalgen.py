"""Tests for the controlled evaluation dataset generator."""

import pytest

from src.evalgen import DocClass, EvalDatasetGenerator
from src.evalgen.generator import ANSWER_BEARING


@pytest.fixture(scope="module")
def dataset():
    return EvalDatasetGenerator(seed=42).generate()


class TestDeterminism:
    def test_same_seed_same_dataset(self):
        a = EvalDatasetGenerator(seed=7).generate()
        b = EvalDatasetGenerator(seed=7).generate()
        assert [d["text"] for d in a.documents] == [d["text"] for d in b.documents]
        assert a.queries == b.queries

    def test_different_seed_different_dataset(self):
        a = EvalDatasetGenerator(seed=1).generate()
        b = EvalDatasetGenerator(seed=2).generate()
        assert [d["text"] for d in a.documents] != [d["text"] for d in b.documents]


class TestStructure:
    def test_class_counts_match_config(self, dataset):
        counts = dataset.class_counts
        assert counts["gold"] == 40
        assert counts["relevant"] == 40
        assert counts["related"] == 80
        assert counts["exact_dup"] == 15
        assert counts["near_dup"] == 15

    def test_every_doc_has_label(self, dataset):
        doc_ids = {d["id"] for d in dataset.documents}
        assert doc_ids == set(dataset.labels.keys())

    def test_unique_doc_ids(self, dataset):
        ids = [d["id"] for d in dataset.documents]
        assert len(ids) == len(set(ids))


class TestGroundTruth:
    def test_relevant_ids_exist_and_answer_bearing(self, dataset):
        doc_ids = {d["id"] for d in dataset.documents}
        for q in dataset.queries:
            for rid in q["relevant_doc_ids"]:
                assert rid in doc_ids
                assert dataset.labels[rid]["answer_bearing"] is True

    def test_gold_always_relevant(self, dataset):
        for q in dataset.queries:
            classes = {dataset.labels[rid]["doc_class"] for rid in q["relevant_doc_ids"]}
            assert "gold" in classes

    def test_attribute_queries_include_paraphrase(self, dataset):
        # 속성형(q_) 쿼리는 gold+paraphrase 둘 다 정답에 포함
        # (코드 조회형 qc_ 쿼리는 코드가 gold에만 있으므로 제외)
        for q in dataset.queries:
            if not q["query_id"].startswith("q_"):
                continue
            classes = {dataset.labels[rid]["doc_class"] for rid in q["relevant_doc_ids"]}
            assert "relevant" in classes

    def test_related_docs_never_in_ground_truth(self, dataset):
        all_relevant = {rid for q in dataset.queries for rid in q["relevant_doc_ids"]}
        for doc_id, meta in dataset.labels.items():
            if meta["doc_class"] == DocClass.RELATED.value:
                assert doc_id not in all_relevant


class TestDuplicates:
    def test_exact_dup_matches_gold_text(self, dataset):
        texts = {d["id"]: d["text"] for d in dataset.documents}
        for doc_id, meta in dataset.labels.items():
            if meta["doc_class"] == DocClass.EXACT_DUP.value:
                topic_golds = [
                    other_id
                    for other_id, m in dataset.labels.items()
                    if m["topic_id"] == meta["topic_id"] and m["doc_class"] == "gold"
                ]
                assert any(texts[doc_id] == texts[g] for g in topic_golds)

    def test_near_dup_differs_but_same_topic(self, dataset):
        texts = {d["id"]: d["text"] for d in dataset.documents}
        for doc_id, meta in dataset.labels.items():
            if meta["doc_class"] == DocClass.NEAR_DUP.value:
                topic_golds = [
                    other_id
                    for other_id, m in dataset.labels.items()
                    if m["topic_id"] == meta["topic_id"] and m["doc_class"] == "gold"
                ]
                assert topic_golds
                # 원문과 다르지만(교란 적용) 핵심 정답 값은 유지
                assert all(texts[doc_id] != texts[g] for g in topic_golds)
