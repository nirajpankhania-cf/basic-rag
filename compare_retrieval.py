"""Show vector-only and hybrid retrieval side by side for one query.

Usage:
    python compare_retrieval.py "What does EXP-4 say?" --expect expense_policy::EXP-4
"""
import argparse

from rag.config import TOP_K
from rag.index import get_collection
from rag.retrieval import retrieve


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--expect", help="id of the chunk that should be retrieved")
    parser.add_argument("-k", type=int, default=TOP_K)
    args = parser.parse_args()

    collection = get_collection()
    print(f"Question: {args.question}")
    if args.expect:
        print(f"Expected chunk: {args.expect}")

    for mode in ("vector", "hybrid"):
        hits = retrieve(args.question, mode=mode, k=args.k, collection=collection)
        print(f"\n{mode.upper()} top {args.k}:")
        for rank, h in enumerate(hits, start=1):
            marker = "  <-- expected" if h["id"] == args.expect else ""
            found_by = f"  ({', '.join(h['found_by'])})" if "found_by" in h else ""
            print(f"  {rank}. {h['id']:<30} score={h['score']:.4f}{found_by}{marker}")
        if args.expect:
            hit = any(h["id"] == args.expect for h in hits)
            print(f"  => expected chunk {'RETRIEVED' if hit else 'MISSED'}")


if __name__ == "__main__":
    main()
