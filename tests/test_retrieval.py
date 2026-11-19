"""Retrieval recall: did the chunk that contains the answer get retrieved?"""
import pytest

from tests.conftest import CONFIGS, FULL_PIPELINE
from tests.eval_set import CASES, RECALL_THRESHOLD


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
