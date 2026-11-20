# Diagnosis: the PTO carryover question

## The question and the flawed answer

**Question:** "How many unused PTO days can I carry over into next year?"

**Correct answer** (current policy, `pto_policy.md`, version 3.0, section PTO-4):
up to 5 days, which must be used by March 31.

**What the system answered:**

> The passages conflict: [1] allows up to **10 days**, while [2] allows up to
> **5 days**. I don't know which limit applies because the context doesn't
> identify the current policy.
>
> Sources cited:
> [1] pto_policy_2023.md | Paid Time Off Policy | PTO-4: Carryover
> [2] pto_policy.md | Paid Time Off Policy | PTO-4: Carryover

**Why this is a failure:** the question has one definite answer and the
system could not give it. It presented a superseded limit (10 days) as
equally valid, and the superseded passage was ranked first. An employee
reading this could reasonably plan around 10 days and forfeit five of them.

## Question 1: was it retrieval?

**No.** Retrieved chunks for the question (hybrid + rerank, top 4):

| Rank | Source file | Section | Rerank score |
|---|---|---|---|
| 1 | pto_policy_2023.md | PTO-4: Carryover | 9.2003 |
| 2 | pto_policy.md | PTO-4: Carryover | 8.9233 |
| 3 | pto_policy.md | PTO-5: Sick Leave | 3.0180 |
| 4 | pto_policy_2023.md | PTO-5: Sick Leave | 2.9214 |

The chunk that holds the correct answer, `pto_policy::PTO-4`, was retrieved
at rank 2 of 4 and reached the LLM. Retrieval found the Carryover section,
which is exactly what it was asked to do, and the two Carryover chunks score
far above everything else.

The outdated chunk ranking first is not a retrieval bug either. The two
chunks are the same section with a few numbers changed, so they are almost
equally relevant to the question (9.20 vs 8.92 is a small gap). Retrieval and
reranking measure relevance; neither has any way to measure which document
is current.

## Question 2: was it the source data?

**Yes.** Two retrieved chunks claim to be the same thing, "Paid Time Off
Policy - PTO-4: Carryover", and contradict each other:

| | pto_policy_2023.md | pto_policy.md |
|---|---|---|
| File header | Version 1.0, effective January 1, 2023 | Version 3.0, effective January 1, 2026 |
| Carryover limit | up to 10 days | up to 5 days |
| Use-by date | June 30 | March 31 |

`pto_policy_2023.md` is an outdated duplicate: version 1.0 of the PTO policy,
superseded by version 3.0 but never removed from `data/policies/`. Ingestion
indexes every file in that folder, so both versions went into the vector store.

The version and effective date exist only in each file's header line. After
section-based chunking, that line is its own chunk, and the Carryover chunks
carry no version information. The LLM received two passages with identical
titles and no way to tell which was current.

## Was it generation?

**No.** The model did what the prompt told it to: answer only from the
context and say so when the context does not settle the question. It
reported the conflict accurately, invented nothing, and cited both sources
correctly. Given contradictory inputs, refusing to pick is the honest output.

## Confirmation: same code, clean data

`diagnose.py` rebuilt the index from a copy of the corpus without
`pto_policy_2023.md` and ran the same question through the same retrieval
and generation code:

| | With outdated duplicate | Without it |
|---|---|---|
| Retrieved | both versions of PTO-4 | only pto_policy.md PTO-4 (rank 1) |
| Answer | "The passages conflict... I don't know which limit applies" | "You can carry over up to **5 days**... use them by **March 31**" |
| Correct? | No | Yes |

No retrieval or generation code changed between the two runs. Only the
source data changed, and the answer went from wrong to right. The root cause
is the source data.

## Why the evaluation harness did not catch it

The 14-question harness scores 100% on recall and answer accuracy, because
none of its questions touch the PTO policy. A passing harness only vouches
for what its test set covers. `tests/test_data_quality.py` now checks the
corpus directly: no two files may supply the same section of the same
document. It fails on the current corpus and is marked as a known issue
(`xfail`) that points here.

## How to fix it

1. **Fix the data:** remove or archive `pto_policy_2023.md` so only the
   current version is ingested. This is the real fix, and the confirmation
   run shows it works.
2. **Add lineage metadata:** parse version, effective date, and status from
   each file's header at ingestion and attach them to every chunk. Then
   ingestion can refuse or filter superseded documents, and answers can cite
   the version they rely on.
3. **Keep the lineage check:** once the duplicate is gone, remove the `xfail`
   mark so the test fails the pipeline if a second version is ever added again.
