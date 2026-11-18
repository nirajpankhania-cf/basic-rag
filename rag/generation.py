"""Generation: pass retrieved chunks to the LLM and return a cited answer."""
import os
import re

from openai import OpenAI

from rag.config import LLM_BASE_URL, LLM_MODEL

SYSTEM_PROMPT = (
    "You are an internal policy assistant for Larkspur Analytics. "
    "Answer the employee's question using ONLY the numbered context passages. "
    "Cite the passages you used with their numbers in square brackets, like [1]. "
    "If the context does not contain the answer, say you don't know. "
    "Be concise."
)


def get_client():
    api_key = os.environ.get("FOUNDRY_API_KEY")
    if not api_key:
        raise RuntimeError("Set the FOUNDRY_API_KEY environment variable first.")
    return OpenAI(base_url=LLM_BASE_URL, api_key=api_key)


def build_prompt(question, chunks):
    """Number each chunk so the model can cite it as [n]."""
    context = "\n\n".join(f"[{i}] {c['text']}" for i, c in enumerate(chunks, start=1))
    return f"Context passages:\n\n{context}\n\nQuestion: {question}"


def generate_answer(question, chunks, client=None):
    """Return {"answer": str, "sources": [...], "retrieved": [...]}.

    "sources" are the retrieved chunks the model actually cited. If the model
    cited nothing, sources is empty rather than guessed.
    """
    client = client or get_client()
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(question, chunks)},
        ],
    )
    answer = response.choices[0].message.content.strip()

    cited_numbers = sorted({int(n) for n in re.findall(r"\[(\d+)\]", answer)})
    sources = [
        {"number": n, **chunks[n - 1]["metadata"], "id": chunks[n - 1]["id"]}
        for n in cited_numbers
        if 1 <= n <= len(chunks)
    ]
    return {"answer": answer, "sources": sources, "retrieved": chunks}
