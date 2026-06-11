"""Tests for plain few-shot prompt assembly (index-driven, no CoT)."""
from pipeline.c2_retrieval.assemble import HEADER, assemble_prompt, render_block
from pipeline.c2_retrieval.pool_map import PoolEntry


class StubPool:
    """Minimal PoolMap stand-in: pool index -> PoolEntry."""

    def __init__(self, golds):
        self._golds = golds

    def entry(self, i):
        return PoolEntry(i, "train", i, f"nl{i}", self._golds[i])


def test_render_block_plain():
    entry = PoolEntry(7, "train", 7, "nl7", "[out:json];q7;out;")
    block = render_block(entry, i=1)
    assert block.startswith("/* Example 1 */")
    assert "#question: nl7" in block
    assert "#OverpassQL:" in block
    assert block.rstrip().endswith("[out:json];q7;out;")
    # plain: NO reasoning fields
    assert "#reason" not in block and "#scope" not in block and "#Osm_Tag" not in block


def test_assemble_prompt_order_header_tail():
    golds = {i: f"[out:json];q{i};out;" for i in range(3)}
    prompt = assemble_prompt([0, 1, 2], StubPool(golds), "find cafes")
    assert prompt.startswith(HEADER)
    assert prompt.rstrip().endswith("/* Answer the following: find cafes */")
    # most-relevant first: Example 1 before Example 2
    assert prompt.index("/* Example 1 */") < prompt.index("/* Example 2 */")


def test_assemble_prompt_renders_each_selected_index():
    golds = {i: f"[out:json];q{i};out;" for i in range(5)}
    prompt = assemble_prompt([4, 2], StubPool(golds), "x")
    assert "#question: nl4" in prompt and "#question: nl2" in prompt
    assert "#question: nl0" not in prompt  # unselected demo absent
