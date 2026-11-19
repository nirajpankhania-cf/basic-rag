"""Unit tests for chunking and chunk metadata (no models needed)."""
from rag.chunking import chunk_all, split_words
from rag.config import CHUNK_OVERLAP_WORDS, CHUNK_SIZE_WORDS


def test_short_text_is_one_chunk():
    assert split_words("just a few words") == ["just a few words"]


def test_long_text_is_split_with_overlap():
    words = [str(i) for i in range(400)]
    pieces = split_words(" ".join(words))

    assert len(pieces) > 1
    assert all(len(p.split()) <= CHUNK_SIZE_WORDS for p in pieces)
    # The end of each piece is repeated at the start of the next one.
    for first, second in zip(pieces, pieces[1:]):
        assert first.split()[-CHUNK_OVERLAP_WORDS:] == second.split()[:CHUNK_OVERLAP_WORDS]
    # Nothing is lost: every word appears in some piece.
    assert {w for p in pieces for w in p.split()} == set(words)


def test_chunk_ids_are_unique():
    ids = [c["id"] for c in chunk_all()]
    assert len(ids) == len(set(ids))


def test_every_chunk_has_source_metadata():
    chunks = chunk_all()
    assert chunks, "no chunks produced; is data/policies empty?"
    for chunk in chunks:
        meta = chunk["metadata"]
        assert meta["source"].endswith(".md")
        assert meta["doc_title"]
        assert meta["section"]
