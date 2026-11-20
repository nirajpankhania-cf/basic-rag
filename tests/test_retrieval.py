"""Retrieval tests.

The first group are unit tests for the individual search functions (no models
needed). The second group is the recall harness: did the chunk that contains
the answer get retrieved by each configuration?
"""
import pytest

import rag.retrieval as retrieval_module
from rag.retrieval import hybrid_search, keyword_search, query_terms
from tests.conftest import CONFIGS, FULL_PIPELINE, FakeCollection
from tests.eval_set import CASES, RECALL_THRESHOLD


# --- unit tests --------------------------------------------------------------


def test_query_terms_drops_stopwords_and_keeps_codes():
    assert query_terms("What does the EXP-6 section say about meals?") == ["exp-6", "meals"]


def test_keyword_search_matches_whole_words_only():
    collection = FakeCollection(
        ids=["exp6", "exp60", "other"],
        documents=["Rule EXP-6: meals.", "Rule EXP-60: hotels.", "Unrelated policy."],
        metadatas=[{"section": "EXP-6"}, {"section": "EXP-60"}, {"section": "X"}],
    )
    hits = keyword_search("what does EXP-6 say", k=3, collection=collection)

    assert [h["id"] for h in hits] == ["exp6"]
    assert hits[0]["text"] == "Rule EXP-6: meals."
    assert hits[0]["metadata"] == {"section": "EXP-6"}
    assert hits[0]["score"] > 0


def test_hybrid_search_ranks_chunks_found_by_both_retrievers_first(monkeypatch):
    def fake_vector(question, k, collection):
        return [{"id": "v_only", "score": 0.9}, {"id": "both", "score": 0.8}]

    def fake_keyword(question, k, collection):
        return [{"id": "k_only", "score": 3.0}, {"id": "both", "score": 1.0}]

    monkeypatch.setattr(retrieval_module, "vector_search", fake_vector)
    monkeypatch.setattr(retrieval_module, "keyword_search", fake_keyword)

    fused = hybrid_search("q", k=3, collection=object())

    assert [h["id"] for h in fused] == ["both", "v_only", "k_only"]
    assert fused[0]["found_by"] == ["vector#2", "keyword#2"]


# --- recall harness ----------------------------------------------------------


def is_hit(case, chunks):
    """True if any expected chunk id is among the retrieved chunks."""
    retrieved_ids = {c["id"] for c in chunks}
    return any(expected in retrieved_ids for expected in case["expected_chunks"])


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_expected_chunk_retrieved(case, retrieved):
    """Per question: the full pipeline must retrieve an expected chunk."""
    chunks = retrieved[FULL_PIPELINE][case["id"]]
    assert is_hit(case, chunks), (
        f"Expected one of {case['expected_chunks']} "
        f"but retrieved {[c['id'] for c in chunks]}"
    )


def test_recall_meets_threshold(retrieved, eval_results):
    """Aggregate: recall@k for every config; the full pipeline must meet the bar."""
    recall = {}
    for name in CONFIGS:
        hits = sum(is_hit(case, retrieved[name][case["id"]]) for case in CASES)
        recall[name] = (hits, len(CASES))
    eval_results["recall"] = recall

    hits, total = recall[FULL_PIPELINE]
    assert hits / total >= RECALL_THRESHOLD, (
        f"Recall {hits}/{total} is below the {RECALL_THRESHOLD:.0%} threshold"
    )


def test_hybrid_not_worse_than_vector(retrieved):
    """Adding keyword search must never lose recall compared to vector-only."""
    def hits(name):
        return sum(is_hit(case, retrieved[name][case["id"]]) for case in CASES)

    assert hits("hybrid") >= hits("vector only")
