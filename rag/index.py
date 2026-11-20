"""Embed chunks with sentence-transformers and store them in chromadb."""
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import chromadb
from sentence_transformers import SentenceTransformer

from rag.chunking import chunk_all
from rag.config import CHROMA_DIR, COLLECTION_NAME, DOCS_DIR, EMBEDDING_MODEL


@lru_cache(maxsize=1)
def get_embedder():
    """Load the embedding model once and reuse it."""
    return SentenceTransformer(EMBEDDING_MODEL)


def embed(texts):
    """Return one normalized embedding (list of floats) per input text."""
    return get_embedder().encode(list(texts), normalize_embeddings=True).tolist()


def build_index(docs_dir=DOCS_DIR, chroma_dir=CHROMA_DIR):
    """Chunk every document, embed the chunks, and (re)create the collection.

    The collection is dropped and rebuilt on every run so the index always
    mirrors exactly what is in the documents folder.
    """
    client = chromadb.PersistentClient(path=str(chroma_dir))
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass  # first run: nothing to delete
    collection = client.create_collection(
        name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )

    chunks = chunk_all(docs_dir)
    collection.add(
        ids=[c["id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        metadatas=[c["metadata"] for c in chunks],
        embeddings=embed(c["text"] for c in chunks),
    )
    return collection


def get_collection(chroma_dir=CHROMA_DIR):
    """Open the existing collection, building it first if it does not exist."""
    client = chromadb.PersistentClient(path=str(chroma_dir))
    try:
        collection = client.get_collection(COLLECTION_NAME)
        if collection.count() > 0:
            return collection
    except Exception:
        pass
    return build_index(chroma_dir=chroma_dir)


@dataclass
class StoredChunks:
    """Every chunk in the collection, as parallel lists.

    Fields are never None: an empty collection gives empty lists.
    """

    ids: list[str] = field(default_factory=list)
    documents: list[str] = field(default_factory=list)
    metadatas: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class QueryHits:
    """Nearest neighbours for one query embedding, as parallel lists.

    Fields are never None: no matches gives empty lists.
    """

    ids: list[str] = field(default_factory=list)
    documents: list[str] = field(default_factory=list)
    metadatas: list[dict[str, Any]] = field(default_factory=list)
    distances: list[float] = field(default_factory=list)


def _first(rows):
    """First row of a per-query result, or [] if chroma did not return the field."""
    return rows[0] if rows else []


def get_all(collection) -> StoredChunks:
    """Fetch every stored chunk with its text and metadata.

    chromadb types each optional field as Optional[...] because it is None
    when left out of `include`; this is the one place that is dealt with.
    """
    res = collection.get(include=["documents", "metadatas"])
    return StoredChunks(
        ids=list(res["ids"]),
        documents=list(res["documents"] or []),
        metadatas=[dict(m) for m in res["metadatas"] or []],
    )


def query(collection, embedding, k) -> QueryHits:
    """Return the k chunks nearest to one query embedding."""
    res = collection.query(query_embeddings=[embedding], n_results=k)
    return QueryHits(
        ids=list(_first(res["ids"])),
        documents=list(_first(res["documents"])),
        metadatas=[dict(m) for m in _first(res["metadatas"])],
        distances=[float(d) for d in _first(res["distances"])],
    )
