"""Two-question debugging for a bad answer: was it retrieval, or the source data?

Usage:
    python diagnose.py "How many PTO days do I accrue per year?" \
        --expect pto_policy::PTO-2 --suspect pto_policy_2023.md

Question 1 - Was it retrieval?
    Did the chunk that holds the correct answer reach the LLM?
Question 2 - Was it the source data?
    Do the retrieved chunks themselves disagree? Which files do they come
    from, and what version does each file claim to be?
Confirmation
    Re-run the identical pipeline on a copy of the corpus with the suspect
    file removed. If the answer becomes correct with no code change, the
    root cause is the source data.
"""
import argparse
import difflib
import os
import re
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path

from rag.config import DOCS_DIR, TOP_K
from rag.generation import generate_answer
from rag.index import build_index
from rag.retrieval import retrieve


def sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n", text) if s.strip()]


def show_retrieved(chunks, expect):
    for i, c in enumerate(chunks, start=1):
        marker = "  <-- expected" if c["id"] == expect else ""
        print(f"  [{i}] {c['metadata']['source']:<22} {c['metadata']['section']}{marker}")


def show_answer(question, chunks):
    if not os.environ.get("FOUNDRY_API_KEY"):
        print("  (FOUNDRY_API_KEY not set; skipping the LLM answer)")
        return
    result = generate_answer(question, chunks)
    print("  Answer: " + result["answer"].replace("\n", "\n          "))
    cited = ", ".join(f"{s['source']} | {s['section']}" for s in result["sources"])
    print(f"  Cited:  {cited or '(none)'}")


def find_conflicts(chunks):
    """Retrieved chunks that claim to be the same section but differ in text."""
    groups = defaultdict(list)
    for c in chunks:
        groups[(c["metadata"]["doc_title"], c["metadata"]["section"])].append(c)
    return {
        key: group
        for key, group in groups.items()
        if len({c["metadata"]["source"] for c in group}) > 1
        and len({c["text"] for c in group}) > 1
    }


def header_line(collection, source):
    """The version/effective-date line from a file's header chunk."""
    res = collection.get(where={"$and": [{"source": source}, {"section_code": "HEADER"}]})
    return res["documents"][0].split("\n", 1)[1] if res["documents"] else "(no header found)"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--expect", required=True, help="chunk id holding the correct answer")
    parser.add_argument("--suspect", help="source file to remove for the confirmation run")
    args = parser.parse_args()

    workdir = Path(tempfile.mkdtemp(prefix="diagnose_"))
    collection = build_index(chroma_dir=workdir / "full")
    chunks = retrieve(args.question, k=TOP_K, collection=collection)

    print(f"Question: {args.question}\n")
    print("=== The pipeline as it runs today ===")
    show_retrieved(chunks, args.expect)
    show_answer(args.question, chunks)

    # ---- Question 1: was it retrieval? ----
    print("\n=== Question 1: was it retrieval? ===")
    ids = [c["id"] for c in chunks]
    if args.expect in ids:
        print(f"  NO. The correct chunk ({args.expect}) was retrieved at rank "
              f"{ids.index(args.expect) + 1} of {len(ids)}.")
        print("  Retrieval gave the LLM what it needed.")
    else:
        print(f"  YES. The correct chunk ({args.expect}) was NOT retrieved.")
        print("  Fix retrieval before looking anywhere else.")

    # ---- Question 2: was it the source data? ----
    print("\n=== Question 2: was it the source data? ===")
    conflicts = find_conflicts(chunks)
    if not conflicts:
        print("  No conflicting chunks among the retrieved context.")
    for (doc_title, section), group in conflicts.items():
        print(f"  CONFLICT: {len(group)} retrieved chunks are both "
              f"'{doc_title} - {section}' but their text differs.")
        for c in group:
            source = c["metadata"]["source"]
            print(f"\n  {source}")
            print(f"    file header: {header_line(collection, source)}")
        a, b = group[0], group[1]
        print("\n  Sentences that differ:")
        for line in difflib.ndiff(sentences(a["text"]), sentences(b["text"])):
            if line.startswith("- "):
                print(f"    {a['metadata']['source']:<20} {line[2:]}")
            elif line.startswith("+ "):
                print(f"    {b['metadata']['source']:<20} {line[2:]}")
        print()
    if conflicts:
        print("  The LLM was handed contradictory passages with the same title and")
        print("  section, and nothing in the passages says which one is current.")

    # ---- Confirmation: same code, clean data ----
    if args.suspect:
        print(f"\n=== Confirmation: same pipeline, corpus without {args.suspect} ===")
        clean_docs = workdir / "docs"
        clean_docs.mkdir()
        for path in Path(DOCS_DIR).glob("*.md"):
            if path.name != args.suspect:
                shutil.copy(path, clean_docs / path.name)
        clean = build_index(docs_dir=clean_docs, chroma_dir=workdir / "clean")
        clean_chunks = retrieve(args.question, k=TOP_K, collection=clean)
        show_retrieved(clean_chunks, args.expect)
        show_answer(args.question, clean_chunks)
        print("\n  No retrieval or generation code changed between the two runs;")
        print("  only the source data did.")

    shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()
