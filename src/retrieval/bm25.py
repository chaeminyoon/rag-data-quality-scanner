"""
BM25 sparse retriever with a Korean-friendly tokenizer.

Sparse retrieval complements dense embeddings: it matches exact terms
(service names, error codes, version numbers) that embeddings smooth over —
the primary reason hybrid search outperforms dense-only retrieval on
identifier-heavy corpora like helpdesk tickets.

Korean note: without a morphological analyzer, whitespace tokens carry
attached particles (조사) — "클라우드포트의" won't match "클라우드포트".
We therefore index both whitespace tokens AND their character bigrams,
which is a standard lightweight approach for Korean sparse retrieval.
"""

import re
from typing import Dict, List

from rank_bm25 import BM25Okapi

from config import get_logger

logger = get_logger("retrieval.bm25")

_TOKEN_RE = re.compile(r"[0-9a-zA-Z가-힣]+")


def tokenize(text: str) -> List[str]:
    """Whitespace-ish tokens + character bigrams (Korean particle robustness)."""
    tokens = _TOKEN_RE.findall(text.lower())
    out: List[str] = []
    for tok in tokens:
        out.append(tok)
        if len(tok) >= 3 and re.search(r"[가-힣]", tok):
            out.extend(tok[i : i + 2] for i in range(len(tok) - 1))
    return out


class BM25Retriever:
    """
    In-memory BM25 index over a document set (one per namespace).

    Result format matches VectorStore.query:
        {"id": str, "score": float, "metadata": {"text": ...}}
    """

    def __init__(self, documents: List[Dict], text_key: str = "text", id_key: str = "id"):
        self.ids = [doc.get(id_key, str(i)) for i, doc in enumerate(documents)]
        self.texts = [doc.get(text_key, "") for doc in documents]
        self.metadata = [
            {**doc.get("metadata", {}), text_key: doc.get(text_key, "")}
            for doc in documents
        ]
        corpus_tokens = [tokenize(t) for t in self.texts]
        # rank_bm25 divides by zero on fully-empty corpora; guard.
        if not any(corpus_tokens):
            corpus_tokens = [["<empty>"] for _ in self.texts] or [["<empty>"]]
        self._bm25 = BM25Okapi(corpus_tokens)
        logger.info(f"Built BM25 index over {len(self.ids)} documents")

    def query(self, query_text: str, top_k: int = 10) -> List[Dict]:
        if not self.ids:
            return []
        scores = self._bm25.get_scores(tokenize(query_text))
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]
        return [
            {"id": self.ids[i], "score": float(scores[i]), "metadata": self.metadata[i]}
            for i in order
        ]
