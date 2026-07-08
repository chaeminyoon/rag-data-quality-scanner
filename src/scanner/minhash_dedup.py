"""
MinHash/LSH candidate generation for two-stage duplicate detection.

Stage 1 (this module): character-shingle MinHash + LSH proposes candidate
pairs in ~O(n) instead of the O(n²) full similarity matrix.

Stage 2 (NoiseDetector): candidates are verified with embedding cosine
similarity at the model-calibrated threshold.

The lexical stage doubles as a PRECISION guard: template-heavy corpora
(helpdesk KBs, tickets) contain documents that are semantically near-identical
to an embedding ("same template, different fact") yet lexically distinct.
Those never become candidates here, so they cannot be wrongly merged —
the failure mode that collapsed 196/222 docs in the embedding-only run.
"""

import re
from typing import Dict, List, Set, Tuple

from datasketch import MinHash, MinHashLSH

from config import get_logger

logger = get_logger("scanner.minhash_dedup")

_WS_RE = re.compile(r"\s+")


def char_shingles(text: str, k: int = 4) -> Set[str]:
    """Character k-gram shingles over whitespace-normalized lowercase text."""
    normalized = _WS_RE.sub(" ", text.lower()).strip()
    if len(normalized) < k:
        return {normalized} if normalized else set()
    return {normalized[i : i + k] for i in range(len(normalized) - k + 1)}


def find_candidate_pairs(
    texts: List[str],
    jaccard_threshold: float = 0.5,
    num_perm: int = 128,
    shingle_k: int = 4,
) -> Dict[Tuple[int, int], float]:
    """
    Propose near-duplicate candidate pairs via MinHash LSH.

    Args:
        texts: document texts (index-aligned with the caller's ids)
        jaccard_threshold: approximate Jaccard similarity for LSH banding
        num_perm: MinHash permutations (accuracy/speed tradeoff)
        shingle_k: character shingle size

    Returns:
        {(i, j): estimated_jaccard} with i < j
    """
    lsh = MinHashLSH(threshold=jaccard_threshold, num_perm=num_perm)
    minhashes: List[MinHash] = []

    for idx, text in enumerate(texts):
        m = MinHash(num_perm=num_perm)
        for sh in char_shingles(text, k=shingle_k):
            m.update(sh.encode("utf-8"))
        minhashes.append(m)
        lsh.insert(str(idx), m)

    candidates: Dict[Tuple[int, int], float] = {}
    for idx in range(len(texts)):
        for other in lsh.query(minhashes[idx]):
            j = int(other)
            if j == idx:
                continue
            pair = (min(idx, j), max(idx, j))
            if pair not in candidates:
                candidates[pair] = float(minhashes[pair[0]].jaccard(minhashes[pair[1]]))

    logger.info(
        f"MinHash LSH proposed {len(candidates)} candidate pairs "
        f"from {len(texts)} documents"
    )
    return candidates
