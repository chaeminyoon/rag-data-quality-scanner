"""
Hard-distractor analysis — the document class that hurts RAG the most.

Cuconasu et al. (SIGIR 2024, "The Power of Noise") showed that "related"
documents — semantically similar to a query but NOT containing the answer —
degrade RAG accuracy far more than random noise (one related document:
up to -25%). Plain duplicate detection cannot see them: they are not
duplicates of anything, they are answer-less lookalikes.

Given a query set with ground truth, this analyzer identifies the documents
that systematically outrank answer-bearing documents, quantifying exactly
which corpus entries push answers out of the retrieved top-k.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from config import get_logger

logger = get_logger("scanner.distractor_analyzer")


@dataclass
class DistractorReport:
    """Corpus-level hard-distractor analysis."""

    #: doc_id -> {"displacements": int, "queries": [query_id, ...]}
    distractors: Dict[str, Dict]
    total_queries: int
    queries_with_displacement: int

    @property
    def top_distractors(self) -> List[Dict]:
        """Documents sorted by how many queries they displaced answers in."""
        return sorted(
            (
                {"doc_id": doc_id, **info}
                for doc_id, info in self.distractors.items()
            ),
            key=lambda x: -x["displacements"],
        )

    @property
    def summary(self) -> Dict:
        return {
            "total_queries": self.total_queries,
            "queries_with_displacement": self.queries_with_displacement,
            "distractor_documents": len(self.distractors),
            "worst_offenders": [
                (d["doc_id"], d["displacements"]) for d in self.top_distractors[:5]
            ],
        }


class DistractorAnalyzer:
    """
    Identify documents that outrank answer-bearing documents in retrieval.

    A document earns a "displacement" for a query when it is retrieved at a
    higher rank than the best-ranked relevant document. Documents that do
    this across many queries are hard distractors: strong candidates for
    manual review, downranking, or answer-enrichment.
    """

    def __init__(self, evaluator, k: int = 10):
        """
        Args:
            evaluator: RAGEvaluator with an already-indexed namespace
            k: retrieval depth to analyze
        """
        self.evaluator = evaluator
        self.k = k

    def analyze(
        self,
        queries: List[Dict],
        namespace: str,
        retrieval_mode: str = "dense",
        query_key: str = "query",
        relevant_ids_key: str = "relevant_doc_ids",
        query_id_key: str = "query_id",
    ) -> DistractorReport:
        """
        Run retrieval for each query and attribute answer displacement.

        Returns:
            DistractorReport ranking corpus documents by displacement count
        """
        result = self.evaluator.evaluate(
            queries=queries,
            namespace=namespace,
            retrieval_mode=retrieval_mode,
            use_rerank=False,
            top_k_retrieval=self.k,
        )

        distractors: Dict[str, Dict] = {}
        displaced_queries = 0

        for qr in result.query_results:
            relevant: Set[str] = set(qr.relevant_ids)
            if not relevant:
                continue
            retrieved = qr.retrieved_ids[: self.k]

            # 정답 중 가장 높은 순위 (미회수면 k와 동일하게 취급)
            best_relevant_rank = next(
                (i for i, d in enumerate(retrieved) if d in relevant),
                len(retrieved),
            )
            if best_relevant_rank == 0:
                continue  # 정답이 1위 — 밀려난 것 없음

            displaced_queries += 1
            for doc_id in retrieved[:best_relevant_rank]:
                if doc_id in relevant:
                    continue
                info = distractors.setdefault(
                    doc_id, {"displacements": 0, "queries": []}
                )
                info["displacements"] += 1
                info["queries"].append(qr.query_id)

        logger.info(
            f"Distractor analysis: {len(distractors)} documents displaced answers "
            f"in {displaced_queries}/{len(queries)} queries"
        )
        return DistractorReport(
            distractors=distractors,
            total_queries=len(queries),
            queries_with_displacement=displaced_queries,
        )


def classify_retrieval_failures(
    query_results,
    corpus_doc_ids: Set[str],
    k: int = 10,
    full_ground_truth: Optional[Dict[str, Set[str]]] = None,
) -> Dict[str, List[str]]:
    """
    Classify per-query retrieval outcomes using the Barnett et al. (CAIN 2024)
    failure-point taxonomy — the two retrieval-stage failure modes:

        FP1 "missing content"    — no answer-bearing document exists in the
                                   corpus (e.g. over-aggressive cleaning
                                   deleted every copy of the answer)
        FP2 "missed top ranked"  — answers exist in the corpus but none was
                                   retrieved into the top-k
        ok                       — at least one answer retrieved in top-k

    Returns:
        {"fp1_missing_content": [...], "fp2_missed_top_ranked": [...], "ok": [...]}
    """
    out: Dict[str, List[str]] = {
        "fp1_missing_content": [],
        "fp2_missed_top_ranked": [],
        "ok": [],
    }
    full_ground_truth = full_ground_truth or {}
    for qr in query_results:
        # FP1 판정은 클리닝 전 전체 정답군 기준이어야 한다
        # (벤치마크는 arm별로 정답을 존재 문서로 필터하므로)
        relevant = set(full_ground_truth.get(qr.query_id, qr.relevant_ids))
        present = relevant & corpus_doc_ids
        if not present:
            out["fp1_missing_content"].append(qr.query_id)
        elif not (set(qr.retrieved_ids[:k]) & present):
            out["fp2_missed_top_ranked"].append(qr.query_id)
        else:
            out["ok"].append(qr.query_id)
    return out
