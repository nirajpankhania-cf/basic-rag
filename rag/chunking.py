"""Section-aware chunking with source metadata.

Strategy
--------
1. Split each policy document on its "## " section headings. A policy
   section is one self-contained rule ("PTO-4: Carryover"), so a section
   boundary is the most meaningful place to cut: a chunk never mixes two
   unrelated rules, and every chunk can be cited by section.
2. If a section is longer than CHUNK_SIZE_WORDS, split it further with a
   sliding window that overlaps by CHUNK_OVERLAP_WORDS.

Chunk size: 150 words
    The embedding model (all-MiniLM-L6-v2) only reads the first 256 tokens
    of its input, roughly 190 English words; anything after that is silently
    ignored. 150 words leaves room for the title and heading we prepend, so
    the whole chunk is actually embedded. It is also small enough that one
    chunk is about one topic, which keeps the embedding focused. Our policy
    sections run 60-110 words, so today each section becomes exactly one chunk.

Overlap: 30 words (20%)
    Only applies when a long section has to be split. Without overlap, a
    sentence that straddles the cut is broken in half and neither chunk
    can answer a question about it. 30 words is about one to two sentences,
    enough to keep a boundary sentence whole in at least one chunk without
    storing much duplicate text.

Every chunk carries metadata (source file, document title, section) so
answers can be attributed to their source.
"""
import re
from pathlib import Path

from rag.config import CHUNK_OVERLAP_WORDS, CHUNK_SIZE_WORDS, DOCS_DIR


def split_words(text, size=CHUNK_SIZE_WORDS, overlap=CHUNK_OVERLAP_WORDS):
    """Split text into windows of `size` words, each overlapping the last by `overlap`."""
    if overlap >= size:
        raise ValueError("overlap must be smaller than chunk size")
    words = text.split()
    if len(words) <= size:
        return [" ".join(words)]
    pieces = []
    step = size - overlap
    for start in range(0, len(words), step):
        pieces.append(" ".join(words[start : start + size]))
        if start + size >= len(words):
            break
    return pieces


def chunk_document(path):
    """Turn one markdown policy file into a list of chunk dicts."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")

    title_match = re.search(r"^# (.+)$", text, flags=re.MULTILINE)
    doc_title = title_match.group(1).strip() if title_match else path.stem

    # parts = [preamble, heading1, body1, heading2, body2, ...]
    parts = re.split(r"^## (.+)$", text, flags=re.MULTILINE)

    sections = []
    preamble = re.sub(r"^# .+$", "", parts[0], flags=re.MULTILINE).strip()
    if preamble:
        sections.append(("Document header", preamble))
    for heading, body in zip(parts[1::2], parts[2::2]):
        sections.append((heading.strip(), body.strip()))

    chunks = []
    for heading, body in sections:
        # "PTO-4: Carryover" -> section_code "PTO-4"; header has no code.
        code_match = re.match(r"([A-Z]+-\d+)", heading)
        section_code = code_match.group(1) if code_match else "HEADER"
        pieces = split_words(body)
        for i, piece in enumerate(pieces):
            chunk_id = f"{path.stem}::{section_code}"
            if len(pieces) > 1:
                chunk_id += f"::{i}"
            chunks.append(
                {
                    "id": chunk_id,
                    # Title and heading are prepended so the chunk makes sense
                    # on its own, to both the embedding model and the LLM.
                    "text": f"{doc_title} - {heading}\n{piece}",
                    "metadata": {
                        "source": path.name,
                        "doc_title": doc_title,
                        "section": heading,
                        "section_code": section_code,
                        "part": i,
                    },
                }
            )
    return chunks


def chunk_all(docs_dir=DOCS_DIR):
    """Chunk every markdown file in the documents folder."""
    chunks = []
    for path in sorted(Path(docs_dir).glob("*.md")):
        chunks.extend(chunk_document(path))
    return chunks


if __name__ == "__main__":
    all_chunks = chunk_all()
    print(f"{len(all_chunks)} chunks")
    for c in all_chunks:
        print(f"  {c['id']:<32} {len(c['text'].split()):>4} words")
