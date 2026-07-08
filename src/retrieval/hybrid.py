"""
Reciprocal Rank Fusion (RRF) for combining dense and sparse result lists.

RRF is rank-based, so it needs no score normalization across retrievers
whose score scales are incomparable (cosine vs BM25) — the same
scale-portability problem we hit with duplicate thresholds.

    RRF(d) = sum over result lists L of 1 / (k + rank_L(d))

k=60 is the standard constant from Cormack et al. (2009).
"""

from typing import Dict, List

RRF_K = 60


def reciprocal_rank_fusion(
    result_lists: List[List[Dict]],
    top_k: int = 10,
    k: int = RRF_K,
) -> List[Dict]:
    """
    Fuse ranked result lists by reciprocal rank.

    Args:
        result_lists: lists of {"id", "score", "metadata"} results,
            each sorted by descending relevance
        top_k: number of fused results to return
        k: RRF dampening constant

    Returns:
        Fused results sorted by RRF score; "score" holds the RRF score and
        "metadata" comes from whichever list saw the document first.
    """
    fused: Dict[str, Dict] = {}
    for results in result_lists:
        for rank, item in enumerate(results, start=1):
            doc_id = item["id"]
            entry = fused.setdefault(
                doc_id,
                {"id": doc_id, "score": 0.0, "metadata": item.get("metadata", {})},
            )
            entry["score"] += 1.0 / (k + rank)

    return sorted(fused.values(), key=lambda x: -x["score"])[:top_k]
