"""Answer accuracy: does the generated answer contain the expected key facts?

These tests call the LLM, so they are skipped when FOUNDRY_API_KEY is not set.
"""
import re

import pytest

from rag.generation import generate_answer
from tests.conftest import FULL_PIPELINE, requires_llm
from tests.eval_set import ACCURACY_THRESHOLD, CASES

pytestmark = requires_llm


def normalize(text):
    """Lowercase and drop thousands separators so '1,500' matches '1500'."""
    return re.sub(r"(?<=\d),(?=\d)", "", text.lower())


def contains_fact(answer, alternatives):
    """True if any acceptable phrasing appears as a whole word/number.

    The boundary check stops '20' from matching inside '2026' or '120'.
    """
    answer = normalize(answer)
    return any(
        re.search(rf"(?<![a-z0-9]){re.escape(normalize(alt))}(?![a-z0-9])", answer)
        for alt in alternatives
    )


def missing_facts(case, answer):
    return [alts for alts in case["key_facts"] if not contains_fact(answer, alts)]


@pytest.fixture(scope="session")
def answers(retrieved):
    """Generate each answer once (lazily) and reuse it across tests."""
    cache = {}

    def get(case):
        if case["id"] not in cache:
            chunks = retrieved[FULL_PIPELINE][case["id"]]
            cache[case["id"]] = generate_answer(case["question"], chunks)
        return cache[case["id"]]

    return get


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_answer_contains_key_facts(case, answers):
    """Per question: every expected key fact must appear in the answer."""
    answer = answers(case)["answer"]
    missing = missing_facts(case, answer)
    assert not missing, f"Answer is missing {missing}.\nAnswer was: {answer}"


def test_accuracy_meets_threshold(answers, eval_results):
    """Aggregate: answer accuracy must meet the bar; also record citation rate."""
    correct = sum(not missing_facts(case, answers(case)["answer"]) for case in CASES)
    cited = sum(
        any(s["id"] in case["expected_chunks"] for s in answers(case)["sources"])
        for case in CASES
    )
    eval_results["accuracy"] = (correct, len(CASES))
    eval_results["citation"] = (cited, len(CASES))

    assert correct / len(CASES) >= ACCURACY_THRESHOLD, (
        f"Accuracy {correct}/{len(CASES)} is below the {ACCURACY_THRESHOLD:.0%} threshold"
    )
