"""Ask the policy assistant a question.

Usage:
    python ask.py "How much is the home office stipend?"
    python ask.py "How much is the home office stipend?" --retrieve-only
"""
import argparse

from rag.config import TOP_K
from rag.generation import generate_answer
from rag.retrieval import vector_search


def print_chunks(chunks):
    print("Retrieved chunks:")
    for i, c in enumerate(chunks, start=1):
        m = c["metadata"]
        print(f"  [{i}] score={c['score']:.3f}  {m['source']} | {m['section']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("-k", type=int, default=TOP_K, help="number of chunks to retrieve")
    parser.add_argument(
        "--retrieve-only", action="store_true", help="show retrieval, skip the LLM call"
    )
    args = parser.parse_args()

    print(f"Question: {args.question}\n")
    chunks = vector_search(args.question, k=args.k)
    print_chunks(chunks)
    if args.retrieve_only:
        return

    result = generate_answer(args.question, chunks)
    print(f"\nAnswer:\n{result['answer']}\n")
    print("Sources cited:")
    if not result["sources"]:
        print("  (the answer cited no sources)")
    for s in result["sources"]:
        print(f"  [{s['number']}] {s['source']} | {s['doc_title']} | {s['section']}")


if __name__ == "__main__":
    main()
