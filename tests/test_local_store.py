"""Tests for the in-memory LocalVectorStore."""

import numpy as np
import pytest

from src.vectordb.local_store import LocalVectorStore


@pytest.fixture
def store():
    s = LocalVectorStore()
    s.create_index(dimension=3)
    return s


def docs_and_embeddings():
    documents = [
        {"id": "d1", "text": "첫 번째 문서"},
        {"id": "d2", "text": "두 번째 문서"},
        {"id": "d3", "text": "세 번째 문서"},
    ]
    embeddings = np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.7, 0.7, 0.0]], dtype=np.float32
    )
    return documents, embeddings


class TestUpsertAndQuery:
    def test_query_returns_ranked_matches(self, store):
        documents, embeddings = docs_and_embeddings()
        store.upsert_documents(documents, embeddings, namespace="ns")

        results = store.query(np.array([1.0, 0.0, 0.0]), top_k=3, namespace="ns")
        assert [r["id"] for r in results] == ["d1", "d3", "d2"]
        assert results[0]["score"] == pytest.approx(1.0)
        assert results[0]["metadata"]["text"] == "첫 번째 문서"

    def test_top_k_limits_results(self, store):
        documents, embeddings = docs_and_embeddings()
        store.upsert_documents(documents, embeddings, namespace="ns")
        assert len(store.query(np.array([1.0, 0.0, 0.0]), top_k=2, namespace="ns")) == 2

    def test_upsert_replaces_existing_id(self, store):
        documents, embeddings = docs_and_embeddings()
        store.upsert_documents(documents, embeddings, namespace="ns")

        # Re-upsert d1 pointing in a new direction
        store.upsert_documents(
            [{"id": "d1", "text": "갱신된 문서"}],
            np.array([[0.0, 0.0, 1.0]], dtype=np.float32),
            namespace="ns",
        )
        assert store.namespace_size("ns") == 3
        results = store.query(np.array([0.0, 0.0, 1.0]), top_k=1, namespace="ns")
        assert results[0]["id"] == "d1"
        assert results[0]["metadata"]["text"] == "갱신된 문서"

    def test_mismatched_counts_raise(self, store):
        with pytest.raises(ValueError):
            store.upsert_documents([{"id": "d1", "text": "x"}], np.zeros((2, 3)))


class TestNamespaces:
    def test_namespaces_are_isolated(self, store):
        documents, embeddings = docs_and_embeddings()
        store.upsert_documents(documents[:1], embeddings[:1], namespace="a")
        store.upsert_documents(documents[1:], embeddings[1:], namespace="b")

        assert store.namespace_size("a") == 1
        assert store.namespace_size("b") == 2
        results = store.query(np.array([0.0, 1.0, 0.0]), top_k=5, namespace="a")
        assert [r["id"] for r in results] == ["d1"]

    def test_delete_namespace(self, store):
        documents, embeddings = docs_and_embeddings()
        store.upsert_documents(documents, embeddings, namespace="ns")
        store.delete_namespace("ns")
        assert store.query(np.array([1.0, 0.0, 0.0]), namespace="ns") == []

    def test_query_empty_namespace(self, store):
        assert store.query(np.array([1.0, 0.0, 0.0]), namespace="missing") == []


class TestNormalization:
    def test_unnormalized_input_is_cosine_ranked(self, store):
        # Same direction, different magnitudes -> same cosine score
        documents = [{"id": "d1", "text": "a"}, {"id": "d2", "text": "b"}]
        embeddings = np.array([[10.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
        store.upsert_documents(documents, embeddings, namespace="ns")

        results = store.query(np.array([5.0, 0.0, 0.0]), top_k=2, namespace="ns")
        assert results[0]["score"] == pytest.approx(results[1]["score"], abs=1e-6)
