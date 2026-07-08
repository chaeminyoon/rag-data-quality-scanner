"""Tests for embedding-based duplicate detection."""

import numpy as np
import pytest

from src.scanner.noise_detector import NoiseDetector


def unit(v):
    v = np.array(v, dtype=float)
    return v / np.linalg.norm(v)


@pytest.fixture
def detector():
    return NoiseDetector(threshold=0.92)


class TestDetectDuplicates:
    def test_identical_embeddings_clustered(self, detector):
        base = unit([1.0, 0.2, 0.1])
        embeddings = np.array([base, base, unit([0.0, 1.0, 0.0])])
        report = detector.detect_duplicates(["d1", "d2", "d3"], embeddings)

        assert len(report.duplicate_clusters) == 1
        assert set(report.duplicate_clusters[0].document_ids) == {"d1", "d2"}

    def test_orthogonal_embeddings_not_clustered(self, detector):
        embeddings = np.array(
            [unit([1, 0, 0]), unit([0, 1, 0]), unit([0, 0, 1])]
        )
        report = detector.detect_duplicates(["d1", "d2", "d3"], embeddings)
        assert report.duplicate_clusters == []
        assert report.duplicate_percentage == 0.0

    def test_transitive_clustering(self, detector):
        # a~b similar, b~c similar -> union-find merges {a, b, c}
        a = unit([1.0, 0.0, 0.0])
        b = unit([1.0, 0.35, 0.0])   # cos(a,b) ≈ 0.944
        c = unit([1.0, 0.7, 0.0])    # cos(b,c) ≈ 0.987, cos(a,c) ≈ 0.819
        embeddings = np.array([a, b, c])
        report = detector.detect_duplicates(["a", "b", "c"], embeddings)

        assert len(report.duplicate_clusters) == 1
        assert set(report.duplicate_clusters[0].document_ids) == {"a", "b", "c"}

    def test_mismatched_lengths_raise(self, detector):
        with pytest.raises(ValueError):
            detector.detect_duplicates(["d1"], np.zeros((2, 3)))

    def test_report_accounting(self, detector):
        base = unit([1.0, 0.0])
        embeddings = np.array([base, base, base, unit([0.0, 1.0])])
        report = detector.detect_duplicates(["a", "b", "c", "d"], embeddings)

        assert report.total_documents == 4
        assert report.total_duplicates == 3       # docs involved in duplication
        assert report.unique_duplicates == 2      # removable (keep 1 per cluster)


class TestThreshold:
    def test_higher_threshold_fewer_clusters(self):
        a = unit([1.0, 0.0])
        b = unit([1.0, 0.3])  # cos ≈ 0.958
        embeddings = np.array([a, b])

        loose = NoiseDetector(threshold=0.95)
        strict = NoiseDetector(threshold=0.99)

        assert len(loose.detect_duplicates(["a", "b"], embeddings).duplicate_clusters) == 1
        assert len(strict.detect_duplicates(["a", "b"], embeddings).duplicate_clusters) == 0
