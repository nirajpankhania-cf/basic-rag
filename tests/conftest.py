"""Shared fixtures and the end-of-run evaluation summary."""
import os

import pytest

from rag.config import TOP_K
from rag.index import build_index
from rag.retrieval import retrieve
from tests.eval_set import CASES

# Retrieval configurations compared in the summary. The last one is the
# full pipeline that ask.py uses and that the per-question tests assert on.
CONFIGS = {
    "vector only": {"mode": "vector", "use_rerank": False},
    "hybrid": {"mode": "hybrid", "use_rerank": False},
    "hybrid + rerank": {"mode": "hybrid", "use_rerank": True},
}
FULL_PIPELINE = "hybrid + rerank"

requires_llm = pytest.mark.skipif(
    not os.environ.get("FOUNDRY_API_KEY"),
    reason="FOUNDRY_API_KEY not set; answer-accuracy tests need the LLM",
)


class FakeCollection:
    """Stand-in for a chromadb collection, so unit tests need no models or disk.

    get() and query() return dicts shaped like chromadb's GetResult and
    QueryResult, including None for fields that were not stored.
    """

    def __init__(self, ids, documents=None, metadatas=None, distances=None):
        self.ids = ids
        self.documents = documents
        self.metadatas = metadatas
        self.distances = distances

    def get(self, include=None):
        return {"ids": self.ids, "documents": self.documents, "metadatas": self.metadatas}

    def query(self, query_embeddings, n_results):
        def row(values):
            return None if values is None else [values[:n_results]]

        return {
            "ids": [self.ids[:n_results]],
            "documents": row(self.documents),
            "metadatas": row(self.metadatas),
            "distances": row(self.distances),
        }


def pytest_configure(config):
    config.eval_results = {}  # filled in by tests, printed in the summary


@pytest.fixture(scope="session")
def collection(tmp_path_factory):
    """Build a fresh index from data/policies in a temp folder.

    The harness never depends on whatever is in the local chroma_db folder,
    so it gives the same result on a laptop and in CI.
    """
    return build_index(chroma_dir=tmp_path_factory.mktemp("chroma"))


@pytest.fixture(scope="session")
def retrieved(collection):
    """Retrieved chunks for every case under every config, computed once.

    retrieved[config_name][case_id] -> list of chunk dicts
    """
    return {
        name: {
            case["id"]: retrieve(case["question"], k=TOP_K, collection=collection, **cfg)
            for case in CASES
        }
        for name, cfg in CONFIGS.items()
    }


@pytest.fixture(scope="session")
def eval_results(request):
    return request.config.eval_results


def pytest_terminal_summary(terminalreporter, config):
    results = config.eval_results
    if not results:
        return
    tr = terminalreporter
    tr.section("RAG evaluation results")
    tr.write_line(f"Test set: {len(CASES)} questions, top-k = {TOP_K}")
    if "recall" in results:
        tr.write_line("")
        tr.write_line("Retrieval recall@k (expected chunk in retrieved top-k):")
        for name, (hits, total) in results["recall"].items():
            tr.write_line(f"  {name:<18} {hits}/{total} = {hits / total:.0%}")
    if "accuracy" in results:
        hits, total = results["accuracy"]
        tr.write_line("")
        tr.write_line("Answer accuracy (answer contains every expected key fact):")
        tr.write_line(f"  {FULL_PIPELINE:<18} {hits}/{total} = {hits / total:.0%}")
    if "citation" in results:
        hits, total = results["citation"]
        tr.write_line("")
        tr.write_line("Source attribution (answer cites the expected source chunk):")
        tr.write_line(f"  {FULL_PIPELINE:<18} {hits}/{total} = {hits / total:.0%}")
    if "accuracy" not in results:
        tr.write_line("")
        tr.write_line("Answer accuracy: SKIPPED (FOUNDRY_API_KEY not set)")
