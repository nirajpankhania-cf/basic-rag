"""Retrieval: turn a question into a ranked list of chunks.

Three first-stage strategies:
  vector_search   - semantic similarity between question and chunk embeddings
  keyword_search  - exact term matching against the chunk text
  hybrid_search   - both, fused with Reciprocal Rank Fusion (RRF)

retrieve() optionally adds a second stage: a cross-encoder reranks the
first-stage candidates (see rag/rerank.py).
"""
import math
import re

from rag.config import TOP_K
from rag.index import embed, get_collection
from rag.rerank import rerank

# How many candidates each retriever contributes before fusion, and how many
# fused candidates are handed to the reranker.
CANDIDATES = 10
# Standard RRF constant: dampens the gap between rank 1 and rank 2 so that
# no single retriever dominates the fused ranking.
RRF_K = 60

STOPWORDS = {
    "a", "an", "and", "are", "about", "at", "be", "can", "do", "does", "for", "from",
    "get", "have", "how", "i", "in", "is", "it", "many", "much", "my", "of", "on",
    "or", "say", "says", "section", "tell", "that", "the", "to", "what", "when",
    "where", "which", "who", "with", "me", "we", "our", "under", "if",
}


def vector_search(question, k=TOP_K, collection=None):
    """Embed the question and return the k most similar chunks.

    Each result is a dict: id, text, metadata, score (cosine similarity,
    higher is better).
    """
    collection = collection or get_collection()
    res = collection.query(query_embeddings=embed([question]), n_results=k)
    return [
        {"id": id_, "text": text, "metadata": meta, "score": 1 - dist}
        for id_, text, meta, dist in zip(
            res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
        )
    ]


def query_terms(question):
    """Lowercased query terms worth matching: keeps codes like 'exp-6' intact."""
    tokens = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", question.lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


def keyword_search(question, k=TOP_K, collection=None):
    """Grep-style search: score each chunk by which query terms it contains.

    A term counts if it appears in the chunk as a whole word (so 'exp-6'
    does not match 'exp-60'). Each matched term is weighted by how rare it
    is across the corpus, log(N / chunks_containing_term), so an exact
    section code that appears in one chunk outweighs a common word like
    'policy' that appears in all of them.
    """
    collection = collection or get_collection()
    stored = collection.get(include=["documents", "metadatas"])
    texts = [t.lower() for t in stored["documents"]]
    n = len(texts)

    scores = [0.0] * n
    for term in set(query_terms(question)):
        pattern = re.compile(rf"(?<![a-z0-9-]){re.escape(term)}(?![a-z0-9-])")
        hits = [i for i, text in enumerate(texts) if pattern.search(text)]
        if not hits:
            continue
        weight = math.log(n / len(hits)) + 0.1  # +0.1 so a term in every chunk still counts a little
        for i in hits:
            scores[i] += weight

    ranked = sorted((i for i in range(n) if scores[i] > 0), key=lambda i: -scores[i])
    return [
        {
            "id": stored["ids"][i],
            "text": stored["documents"][i],
            "metadata": stored["metadatas"][i],
            "score": scores[i],
        }
        for i in ranked[:k]
    ]


def hybrid_search(question, k=TOP_K, collection=None):
    """Run vector and keyword search, then fuse the two rankings with RRF.

    The two retrievers produce scores on different scales (cosine similarity
    vs. summed term weights), so we ignore the raw scores and combine ranks:

        fused_score(chunk) = sum over retrievers of 1 / (RRF_K + rank)

    A chunk ranked highly by either retriever does well; a chunk ranked
    highly by both does best.
    """
    collection = collection or get_collection()
    vector_hits = vector_search(question, k=CANDIDATES, collection=collection)
    keyword_hits = keyword_search(question, k=CANDIDATES, collection=collection)

    fused = {}
    for name, hits in (("vector", vector_hits), ("keyword", keyword_hits)):
        for rank, hit in enumerate(hits, start=1):
            entry = fused.setdefault(hit["id"], {**hit, "score": 0.0, "found_by": []})
            entry["score"] += 1 / (RRF_K + rank)
            entry["found_by"].append(f"{name}#{rank}")

    return sorted(fused.values(), key=lambda h: -h["score"])[:k]


def retrieve(question, mode="hybrid", k=TOP_K, use_rerank=True, collection=None):
    """Single entry point used by ask.py and the evaluation harness.

    With use_rerank=True (the default) this is a two-stage pipeline:
      1. first-stage search returns CANDIDATES chunks (wide net, cheap)
      2. the cross-encoder re-scores them and the best k are kept
    With use_rerank=False the first-stage top k is returned as is.
    """
    search = {"vector": vector_search, "keyword": keyword_search, "hybrid": hybrid_search}[mode]
    if not use_rerank:
        return search(question, k=k, collection=collection)
    candidates = search(question, k=CANDIDATES, collection=collection)
    return rerank(question, candidates, top_n=k)
