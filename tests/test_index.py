"""Unit tests for the chromadb boundary in rag/index.py (no models needed)."""
from rag.index import get_all, query
from tests.conftest import FakeCollection


def test_get_all_never_returns_none_fields():
    """chromadb gives None for fields it did not return; we give empty lists."""
    stored = get_all(FakeCollection(ids=[]))
    assert stored.ids == []
    assert stored.documents == []
    assert stored.metadatas == []


def test_get_all_returns_parallel_lists_of_plain_dicts():
    collection = FakeCollection(
        ids=["a", "b"], documents=["text a", "text b"], metadatas=[{"source": "x"}, {"source": "y"}]
    )
    stored = get_all(collection)
    assert stored.ids == ["a", "b"]
    assert stored.documents == ["text a", "text b"]
    assert stored.metadatas == [{"source": "x"}, {"source": "y"}]
    assert all(type(m) is dict for m in stored.metadatas)


def test_query_unwraps_the_single_result_row():
    collection = FakeCollection(
        ids=["a", "b", "c"],
        documents=["ta", "tb", "tc"],
        metadatas=[{}, {}, {}],
        distances=[0.1, 0.2, 0.3],
    )
    hits = query(collection, embedding=[0.0], k=2)
    assert hits.ids == ["a", "b"]
    assert hits.documents == ["ta", "tb"]
    assert hits.distances == [0.1, 0.2]

    empty = query(FakeCollection(ids=[]), embedding=[0.0], k=2)
    assert empty.ids == [] and empty.documents == [] and empty.distances == []
