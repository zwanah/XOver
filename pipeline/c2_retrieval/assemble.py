"""Assembly stage: stitch the selected demos into a plain few-shot prompt.

Pure render. Takes the list of demo-pool **indices** produced by the selection
stage (:mod:`select`) and emits one prompt: a header, one ``#question`` /
``#OverpassQL`` block per demo, and the eval query at the tail.

This is the **plain** demonstration format — the methodology's main setting
(§4): no chain-of-thought reasoning fields. Each demo is rendered directly from
its :class:`~pipeline.c2_retrieval.pool_map.PoolEntry` (which already carries the
``nl_question`` and gold ``OverpassQL`` from the score-source files), so no CoT
corpus is required.
"""
from __future__ import annotations

from typing import List

from .pool_map import PoolEntry, PoolMap

HEADER = "/* Some OverpassQL examples are provided based on similar problems: */"
EXAMPLE_HEAD = "/* Example {i} */"
TAIL = "/* Answer the following: {nl} */"


def render_block(entry: PoolEntry, i: int) -> str:
    """Render one plain demo block (1-indexed): question + gold OverpassQL."""
    head = EXAMPLE_HEAD.format(i=i)
    question = f"#question: {entry.nl_question}"
    gold = f"#OverpassQL:\n{entry.overpassql}"
    return f"{head}\n{question}\n{gold}"


def assemble_prompt(selected: List[int], pool_map: PoolMap, eval_nl: str) -> str:
    """Stitch the selected demos' plain blocks + eval query into one prompt."""
    blocks = [
        render_block(pool_map.entry(idx), i)
        for i, idx in enumerate(selected, start=1)
    ]
    parts = [HEADER, *blocks, TAIL.format(nl=eval_nl)]
    return "\n\n".join(parts)
