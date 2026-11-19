"""Reranking: re-score retrieved candidates with a cross-encoder.

Why a second stage?
    The embedding model (a bi-encoder) embeds the question and each chunk
    separately and compares the two vectors. That is fast enough to search
    the whole corpus, but the model never sees the question and the chunk
    together. A cross-encoder reads the (question, chunk) pair in a single
    pass, so it can judge whether this chunk actually answers this question.
    It is far more accurate but too slow to run over every chunk, so we use
    it only on the short candidate list that first-stage retrieval returns.
"""
from functools import lru_cache

from sentence_transformers import CrossEncoder

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@lru_cache(maxsize=1)
def get_reranker():
    """Load the cross-encoder once and reuse it."""
    return CrossEncoder(RERANK_MODEL)


def rerank(question, chunks, top_n):
    """Re-score every candidate against the question; return the best top_n.

    Each returned chunk keeps its first-stage score as "retrieval_score" and
    its first-stage position as "retrieval_rank"; "score" becomes the
    cross-encoder relevance score (higher is better, can be negative).
    """
    if not chunks:
        return []
    pair_scores = get_reranker().predict([(question, c["text"]) for c in chunks])
    rescored = [
        {**c, "retrieval_score": c["score"], "retrieval_rank": rank, "score": float(s)}
        for rank, (c, s) in enumerate(zip(chunks, pair_scores), start=1)
    ]
    return sorted(rescored, key=lambda c: -c["score"])[:top_n]
