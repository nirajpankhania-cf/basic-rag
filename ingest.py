"""Chunk, embed, and store every policy document. Run this after changing documents."""
from collections import Counter

from rag.index import build_index

if __name__ == "__main__":
    collection = build_index()
    stored = collection.get(include=["metadatas"])
    per_file = Counter(m["source"] for m in stored["metadatas"])
    print(f"Stored {collection.count()} chunks in collection '{collection.name}':")
    for source, n in sorted(per_file.items()):
        print(f"  {source:<28} {n} chunks")
