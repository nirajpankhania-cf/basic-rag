"""Minimal embed -> store -> retrieve loop (Suggested Approach steps 2-3).

Proves the core loop works on two known pieces of text before any
chunking, hybrid search, or reranking is layered on top.
"""
import chromadb
from sentence_transformers import SentenceTransformer

# Small, fast, CPU-friendly embedding model: 384-dimensional vectors.
model = SentenceTransformer("all-MiniLM-L6-v2")

# In-memory client: nothing is written to disk, so every run starts clean.
client = chromadb.Client()
collection = client.create_collection(
    name="minimal_loop",
    metadata={"hnsw:space": "cosine"},  # cosine distance: 0 = identical
)

texts = {
    "pto": "Full-time employees accrue 20 days of paid time off per year.",
    "expenses": "Meal expenses are reimbursed up to 60 dollars per day while travelling.",
}

# --- Step 2: embed and store ---
ids = list(texts.keys())
docs = list(texts.values())
embeddings = model.encode(docs).tolist()
collection.add(ids=ids, documents=docs, embeddings=embeddings)

print(f"Embedding dimension: {len(embeddings[0])}")
print(f"Chunks stored in collection: {collection.count()}")

# --- Step 3: query and confirm the more relevant text comes back first ---
question = "How many vacation days do I get each year?"
query_embedding = model.encode([question]).tolist()
results = collection.query(query_embeddings=query_embedding, n_results=2)

print(f"\nQuestion: {question}")
for rank, (doc_id, doc, dist) in enumerate(
    zip(results["ids"][0], results["documents"][0], results["distances"][0]), start=1
):
    print(f"  {rank}. [{doc_id}] distance={dist:.4f}  {doc}")

top_id = results["ids"][0][0]
assert top_id == "pto", f"Expected 'pto' to rank first, got '{top_id}'"
print("\nPASS: the PTO text ranked first for a vacation question.")
