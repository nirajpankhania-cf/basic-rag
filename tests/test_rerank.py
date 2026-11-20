"""Unit tests for cross-encoder reranking, with the model stubbed out."""
import rag.rerank as rerank_module
from rag.rerank import rerank


class FakeReranker:
    """Scores each (question, chunk) pair from a fixed lookup on the chunk text."""

    def __init__(self, scores):
        self.scores = scores

    def predict(self, pairs):
        return [self.scores[text] for _, text in pairs]


def test_rerank_reorders_and_keeps_first_stage_info(monkeypatch):
    chunks = [
        {"id": "first", "text": "weak", "score": 0.9},
        {"id": "second", "text": "strong", "score": 0.5},
        {"id": "third", "text": "medium", "score": 0.1},
    ]
    fake = FakeReranker({"weak": -1.0, "strong": 5.0, "medium": 2.0})
    monkeypatch.setattr(rerank_module, "get_reranker", lambda: fake)

    result = rerank("q", chunks, top_n=2)

    assert [c["id"] for c in result] == ["second", "third"]
    assert result[0]["score"] == 5.0
    assert result[0]["retrieval_score"] == 0.5
    assert result[0]["retrieval_rank"] == 2


def test_rerank_empty_input_returns_empty_without_loading_model(monkeypatch):
    def fail():
        raise AssertionError("model should not be loaded for empty input")

    monkeypatch.setattr(rerank_module, "get_reranker", fail)
    assert rerank("q", [], top_n=3) == []
