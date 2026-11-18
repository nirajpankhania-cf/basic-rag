"""Retrieval: turn a question into a ranked list of chunks."""
from rag.config import TOP_K
from rag.index import embed, get_collection


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
