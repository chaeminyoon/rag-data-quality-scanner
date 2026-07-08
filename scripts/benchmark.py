"""
Fair 2x2 ablation benchmark: {original, cleaned} x {dense, dense+rerank}.

Fixes the evaluation-design flaw of the previous pipeline, which applied
reranking only to the cleaned corpus and thus conflated cleaning gains with
reranking gains. All four cells share the same queries and metric code;
bootstrap 95% CIs quantify whether differences exceed sampling noise.

Usage:
    python scripts/benchmark.py [--data data/eval] [--strategy moderate]
        [--k 10] [--rerank-candidates 50] [--out reports]
"""

import argparse
import csv
import json
import os
import sys
import time
from typing import Dict, List, Optional, Set

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import get_logger
from src.scanner import DataQualityScanner, CleaningStrategy
from src.evaluator import RAGEvaluator

logger = get_logger("scripts.benchmark")


def load_data(data_dir: str):
    with open(f"{data_dir}/documents.csv") as f:
        documents = [
            {"id": r["id"], "text": r["text"], "metadata": {"category": r.get("category", "")}}
            for r in csv.DictReader(f)
        ]
    with open(f"{data_dir}/ground_truth.csv") as f:
        queries = [
            {
                "query_id": r["query_id"],
                "query": r["query"],
                "relevant_doc_ids": [x.strip() for x in r["relevant_doc_ids"].split(",")],
            }
            for r in csv.DictReader(f)
        ]
    labels = {}
    labels_path = f"{data_dir}/doc_labels.csv"
    if os.path.exists(labels_path):
        with open(labels_path) as f:
            labels = {r["doc_id"]: r for r in csv.DictReader(f)}
    return documents, queries, labels


def fact_recall_at_k(
    retrieved: List[str], relevant: Set[str], labels: Dict, k: int
) -> Optional[float]:
    """
    Group-aware recall: duplicates of the same fact count once.

    Standard Recall@k penalizes a cleaned corpus for having removed duplicate
    copies of an answer, even though the fact itself is still retrievable.
    Here every (topic) fact group is covered if ANY of its answer-bearing
    docs is retrieved. Requires doc_labels.csv; returns None without it.
    """
    if not labels:
        return None
    relevant_topics = {labels[d]["topic_id"] for d in relevant if d in labels}
    relevant_topics.discard("none")
    if not relevant_topics:
        return None
    covered = {
        labels[d]["topic_id"]
        for d in retrieved[:k]
        if d in labels
        and labels[d]["topic_id"] in relevant_topics
        and labels[d].get("answer_bearing") == "True"
    }
    return len(covered) / len(relevant_topics)


def cleaning_attribution(cleaned_result, labels: Dict) -> Optional[Dict]:
    """
    Label-based cleaning quality: was what we removed actually a duplicate?

    dedup_precision — of docs removed as duplicates, fraction whose true
                      class is exact_dup/near_dup
    dedup_recall    — of all injected duplicates, fraction removed
    collateral      — removed docs by true class (what cleaning destroyed)
    """
    if not labels:
        return None
    removed_ids = {d["id"] for d in cleaned_result.removed_documents}
    dup_removed = {
        doc_id
        for reason_ids in [cleaned_result.removal_reasons.get("duplicate", []),
                           cleaned_result.removal_reasons.get("exact_duplicate", [])]
        for doc_id in reason_ids
    }
    true_dups = {d for d, m in labels.items() if m["doc_class"] in ("exact_dup", "near_dup")}

    # TP 판정: 주입한 중복 클래스이거나, 남아 있는 문서와 텍스트가 동일한 사본
    # (저품질 상용구처럼 우연히 동일한 문서의 병합도 올바른 제거)
    removed_docs = {d["id"]: d for d in cleaned_result.removed_documents}
    kept_norm_texts = {
        " ".join(d.get("text", "").split())
        for d in cleaned_result.cleaned_documents
    }
    def is_correct_removal(doc_id: str) -> bool:
        if doc_id in true_dups:
            return True
        doc = removed_docs.get(doc_id)
        if doc is None:
            return False
        return " ".join(doc.get("text", "").split()) in kept_norm_texts

    tp_precision = sum(1 for d in dup_removed if is_correct_removal(d))
    tp_recall = len(dup_removed & true_dups)
    collateral = {}
    for doc_id in removed_ids:
        c = labels.get(doc_id, {}).get("doc_class", "unknown")
        collateral[c] = collateral.get(c, 0) + 1
    return {
        "dedup_precision": tp_precision / len(dup_removed) if dup_removed else None,
        "dedup_recall": tp_recall / len(true_dups) if true_dups else None,
        "removed_by_true_class": collateral,
    }


def bootstrap_ci(values: List[float], n_boot: int = 2000, seed: int = 0):
    """Percentile bootstrap 95% CI of the mean."""
    rng = np.random.RandomState(seed)
    values = np.array(values, dtype=float)
    means = [
        values[rng.randint(0, len(values), len(values))].mean() for _ in range(n_boot)
    ]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def evaluate_arm(
    evaluator: RAGEvaluator,
    queries: List[Dict],
    namespace: str,
    use_rerank: bool,
    labels: Dict,
    k: int,
    rerank_candidates: int,
    full_ground_truth: Optional[Dict[str, Set[str]]] = None,
) -> Dict:
    """Evaluate one ablation cell; returns per-query and aggregate metrics."""
    result = evaluator.evaluate(
        queries=queries,
        namespace=namespace,
        use_rerank=use_rerank,
        top_k_retrieval=rerank_candidates if use_rerank else k,
    )

    full_ground_truth = full_ground_truth or {}
    per_query = {
        "ndcg": [], "mrr": [], "hit_rate": [], "fact_recall": [],
    }
    for qr in result.query_results:
        per_query["ndcg"].append(qr.metrics.ndcg_at_k)
        per_query["mrr"].append(qr.metrics.mrr)
        per_query["hit_rate"].append(qr.metrics.hit_rate_at_k)
        full_relevant = full_ground_truth.get(qr.query_id, qr.relevant_ids)
        fr = fact_recall_at_k(qr.retrieved_ids, full_relevant, labels, k)
        if fr is not None:
            per_query["fact_recall"].append(fr)

    agg = {}
    for name, vals in per_query.items():
        if not vals:
            continue
        lo, hi = bootstrap_ci(vals)
        agg[name] = {"mean": float(np.mean(vals)), "ci95": [lo, hi]}
    return {"aggregate": agg, "per_query": per_query, "query_results": result.query_results}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/eval")
    parser.add_argument("--strategy", default="moderate",
                        choices=["conservative", "moderate", "aggressive"])
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--rerank-candidates", type=int, default=50)
    parser.add_argument("--out", default="reports")
    args = parser.parse_args()

    documents, queries, labels = load_data(args.data)
    print(f"문서 {len(documents)}개 / 쿼리 {len(queries)}개 로드")

    # 1) 스캔 + 클리닝
    t0 = time.time()
    scanner = DataQualityScanner()
    scan_result, cleaned_result = scanner.scan_and_clean(
        documents=documents, strategy=CleaningStrategy(args.strategy)
    )
    print(f"[스캔+클리닝] {time.time()-t0:.1f}s — "
          f"{cleaned_result.original_count} -> {cleaned_result.cleaned_count}개 "
          f"(제거 {cleaned_result.removed_count})")
    print("제거 사유:", {k: len(v) for k, v in cleaned_result.removal_reasons.items()})

    # 2) 임베딩 + 인덱싱 (재사용: 클린 문서의 임베딩은 스캔 결과에서 추출)
    embedder = scanner.embedder
    id_to_idx = {d["id"]: i for i, d in enumerate(documents)}
    cleaned_idx = [id_to_idx[d["id"]] for d in cleaned_result.cleaned_documents]
    cleaned_embeddings = scan_result.embeddings[cleaned_idx]

    evaluator = RAGEvaluator(embedding_provider=embedder, k=args.k)
    evaluator.setup_index(documents, scan_result.embeddings, namespace="original")
    evaluator.setup_index(cleaned_result.cleaned_documents, cleaned_embeddings,
                          namespace="cleaned")

    # 3) 2x2 ablation
    # NDCG/MRR은 "해당 코퍼스에 존재하는 정답" 기준으로 평가한다.
    # 평면(고정) GT로 평가하면 중복 사본 제거가 '정답 미회수'로 부당 감점되기
    # 때문 (중복이 지표를 왜곡하는 전형적 사례). 코퍼스 간 재현율 비교는
    # 사실(fact) 단위 fact_recall이 담당한다.
    full_gt = {q["query_id"]: set(q["relevant_doc_ids"]) for q in queries}
    corpus_ids = {
        "original": {d["id"] for d in documents},
        "cleaned": {d["id"] for d in cleaned_result.cleaned_documents},
    }
    destroyed = {}
    arm_queries = {}
    for corpus, ids in corpus_ids.items():
        qs = []
        lost = 0
        for q in queries:
            present = [r for r in q["relevant_doc_ids"] if r in ids]
            if not present:
                lost += 1  # 클리닝이 이 쿼리의 정답을 전부 삭제 (FP1 유발)
            qs.append({**q, "relevant_doc_ids": present})
        arm_queries[corpus] = qs
        destroyed[corpus] = lost
    if destroyed["cleaned"]:
        print(f"경고: 클리닝으로 정답이 전부 삭제된 쿼리 {destroyed['cleaned']}개")

    cells = {}
    for corpus in ["original", "cleaned"]:
        for rerank in [False, True]:
            name = f"{corpus}{'+rerank' if rerank else ''}"
            t0 = time.time()
            cells[name] = evaluate_arm(
                evaluator, arm_queries[corpus], corpus, rerank, labels,
                k=args.k, rerank_candidates=args.rerank_candidates,
                full_ground_truth=full_gt,
            )
            a = cells[name]["aggregate"]
            print(f"[{name:>18}] {time.time()-t0:.1f}s | " + " | ".join(
                f"{m}: {v['mean']:.3f} [{v['ci95'][0]:.3f},{v['ci95'][1]:.3f}]"
                for m, v in a.items()
            ))

    # 4) 리포트 저장
    os.makedirs(args.out, exist_ok=True)
    attribution = cleaning_attribution(cleaned_result, labels)
    if attribution:
        print("클리닝 귀속:", json.dumps(attribution, ensure_ascii=False))

    report = {
        "config": vars(args),
        "cleaning_attribution": attribution,
        "answers_destroyed_by_cleaning": destroyed["cleaned"],
        "corpus": {
            "total": len(documents),
            "cleaned": cleaned_result.cleaned_count,
            "removed": cleaned_result.removed_count,
            "removal_reasons": {k: len(v) for k, v in cleaned_result.removal_reasons.items()},
        },
        "results": {
            name: cell["aggregate"] for name, cell in cells.items()
        },
    }
    out_path = f"{args.out}/benchmark_{args.strategy}_k{args.k}.json"
    with open(out_path, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n리포트 저장: {out_path}")


if __name__ == "__main__":
    main()
