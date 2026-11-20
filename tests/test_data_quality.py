"""Data quality / lineage checks on the source corpus (no models needed)."""
from collections import defaultdict

import pytest

from rag.chunking import chunk_all


@pytest.mark.xfail(
    reason="Known data quality issue: pto_policy_2023.md is an outdated duplicate "
    "of pto_policy.md and is still in data/policies. See DIAGNOSIS.md.",
    strict=False,
)
def test_each_section_comes_from_one_source_file():
    """No two files should supply the same section of the same document.

    If they do, the corpus holds more than one version of a policy and the
    assistant cannot know which one is current.
    """
    sources = defaultdict(set)
    for chunk in chunk_all():
        meta = chunk["metadata"]
        sources[(meta["doc_title"], meta["section_code"])].add(meta["source"])

    duplicated = {key: files for key, files in sources.items() if len(files) > 1}
    assert not duplicated, f"Sections supplied by more than one file: {duplicated}"
