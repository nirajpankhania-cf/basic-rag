"""Show the hybrid candidate list before and after cross-encoder reranking.

Usage:
    python compare_rerank.py "Can I expense a gym session while travelling?"
"""
import argparse

from rag.config import TOP_K
from rag.index import get_collection
from rag.rerank import rerank
from rag.retrieval import CANDIDATES, hybrid_search


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("-k", type=int, default=TOP_K)
    args = parser.parse_args()

    candidates = hybrid_search(args.question, k=CANDIDATES, collection=get_collection())
    print(f"Question: {args.question}")
    print(f"\nBEFORE rerank (hybrid candidates, top {args.k} would be used):")
    for rank, c in enumerate(candidates, start=1):
        cut = "" if rank != args.k else "   <-- cut-off without reranking"
        print(f"  {rank:>2}. {c['id']:<30} rrf={c['score']:.4f}{cut}")

    print(f"\nAFTER rerank (cross-encoder, final top {args.k}):")
    for rank, c in enumerate(rerank(args.question, candidates, top_n=args.k), start=1):
        move = c["retrieval_rank"] - rank
        note = f"up {move}" if move > 0 else f"down {-move}" if move < 0 else "same"
        print(
            f"  {rank:>2}. {c['id']:<30} score={c['score']:>7.3f}  "
            f"(was #{c['retrieval_rank']}, {note})"
        )


if __name__ == "__main__":
    main()
