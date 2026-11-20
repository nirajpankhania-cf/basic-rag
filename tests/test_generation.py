"""Unit tests for prompt building and citation extraction, with the LLM stubbed out."""
from types import SimpleNamespace

from rag.generation import build_prompt, generate_answer

CHUNKS = [
    {"id": "pto::PTO-2", "text": "You accrue 20 days.", "metadata": {"source": "pto.md"}},
    {"id": "exp::EXP-4", "text": "Meals up to 60.", "metadata": {"source": "exp.md"}},
]


class FakeClient:
    """Mimics the OpenAI client just enough to return a fixed message content."""

    def __init__(self, content):
        completions = SimpleNamespace(create=lambda **kwargs: self._response(content))
        self.chat = SimpleNamespace(completions=completions)

    @staticmethod
    def _response(content):
        message = SimpleNamespace(content=content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def test_build_prompt_numbers_chunks_from_one():
    prompt = build_prompt("How many days?", CHUNKS)
    assert "[1] You accrue 20 days." in prompt
    assert "[2] Meals up to 60." in prompt
    assert prompt.endswith("Question: How many days?")


def test_generate_answer_maps_citations_to_sources():
    result = generate_answer("q", CHUNKS, client=FakeClient("20 days [1]. Ignore [7]."))
    assert result["answer"] == "20 days [1]. Ignore [7]."
    assert [s["id"] for s in result["sources"]] == ["pto::PTO-2"]
    assert result["sources"][0]["number"] == 1
    assert result["sources"][0]["source"] == "pto.md"
    assert result["retrieved"] == CHUNKS


def test_generate_answer_handles_missing_content():
    """The SDK types content as Optional; None must give an empty answer, not a crash."""
    result = generate_answer("q", CHUNKS, client=FakeClient(None))
    assert result["answer"] == ""
    assert result["sources"] == []
