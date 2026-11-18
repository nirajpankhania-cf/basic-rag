"""Embed chunks with sentence-transformers and store them in chromadb."""
from functools import lru_cache

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
