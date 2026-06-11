"""Unit tests for DeepEye-style confidence selection (``confidence.py``).

Pure logic only — the LLM judge is a stub. Run: ``pytest tests/test_confidence.py``.

The load-bearing invariant: a comparator that always returns TIE (or one that is
never called because of the shortcut) must reduce confidence selection *exactly*
to ``selectors.nonempty_vote``.
"""
from __future__ import annotations

from pipeline.c3_selection import selectors
from pipeline.c3_selection.confidence import GroupRep, select_confidence


def _c(sig, ex):
    return {"sig": sig, "ex": int(ex)}


SET_A = ("set", "aaa")
SET_B = ("set", "bbb")
SET_C = ("set", "ccc")
EMPTY = ("empty",)
ERR = ("error", "boom")


def always_tie(a: GroupRep, b: GroupRep):
    return ["TIE", "TIE", "TIE"]


def prefer_b(a: GroupRep, b: GroupRep):
    # Judge always backs the lower-consistency candidate.
    return ["B", "B", "B"]


def prefer_a(a: GroupRep, b: GroupRep):
    return ["A", "A", "A"]


# --- shortcut / trivial routes ----------------------------------------------
def test_empty_pool_returns_minus_one():
    r = select_confidence([_c(ERR, 0), _c(ERR, 1)], always_tie)
    assert r.picked_index == -1 and r.route == "empty_pool"


def test_single_group_short_circuits():
    cands = [_c(SET_A, 1), _c(SET_A, 1), _c(SET_A, 1)]
    r = select_confidence(cands, always_tie)
    assert r.route == "single" and r.picked_index == 0


def test_shortcut_when_top1_consistency_high():
    # 3/4 agree -> consistency .75 >= .7 threshold -> no tournament.
    cands = [_c(SET_A, 1), _c(SET_A, 1), _c(SET_A, 1), _c(SET_B, 0)]
    r = select_confidence(cands, prefer_b, k=2, shortcut_threshold=0.7)
    assert r.route == "shortcut" and r.picked_index == 0


# --- the reduce-to-nonempty_vote invariant ----------------------------------
def _ex_of_pick(cands, r):
    return cands[r.picked_index]["ex"] if r.picked_index >= 0 else 0


def test_always_tie_reduces_to_nonempty_vote():
    # A judge that abstains (all TIE) must reproduce nonempty_vote's pick ex,
    # even on contested pools (threshold high enough to force the tournament).
    pools = [
        [_c(SET_A, 0), _c(SET_A, 0), _c(SET_B, 1), _c(SET_B, 1), _c(SET_C, 1)],
        [_c(SET_A, 1), _c(SET_B, 0), _c(SET_C, 0), _c(EMPTY, 0)],
        [_c(EMPTY, 1), _c(EMPTY, 1), _c(SET_A, 0)],         # nonempty pref
        [_c(ERR, 0), _c(SET_A, 1), _c(SET_B, 0)],
        [_c(SET_A, 0), _c(SET_B, 1)],                       # 1-1 split
    ]
    for cands in pools:
        r = select_confidence(cands, always_tie, k=4, shortcut_threshold=1.1)
        assert _ex_of_pick(cands, r) == selectors.nonempty_vote(cands)


def test_no_comparator_degrades_to_top1():
    cands = [_c(SET_A, 0), _c(SET_A, 0), _c(SET_B, 1), _c(SET_B, 1), _c(SET_C, 1)]
    r = select_confidence(cands, None, k=3, shortcut_threshold=1.1)
    # No judge -> top-1 consistency group (tie .4/.4 broken by earliest index A).
    assert _ex_of_pick(cands, r) == selectors.nonempty_vote(cands)


# --- the judge actually changes the pick ------------------------------------
def test_judge_can_flip_to_lower_consistency_group():
    # Two groups, A bigger (3) vs B (2). nonempty_vote -> A (ex 0).
    # A judge backing B must flip the pick to B (ex 1) when consistency
    # weighting does not veto it.
    cands = [_c(SET_A, 0), _c(SET_A, 0), _c(SET_A, 0),
             _c(SET_B, 1), _c(SET_B, 1)]
    assert selectors.nonempty_vote(cands) == 0
    r = select_confidence(cands, prefer_b, k=2, shortcut_threshold=1.1)
    assert r.route == "tournament"
    assert _ex_of_pick(cands, r) == 1  # judge flipped A->B


def test_judge_backing_a_keeps_top1():
    cands = [_c(SET_A, 1), _c(SET_A, 1), _c(SET_A, 1),
             _c(SET_B, 0), _c(SET_B, 0)]
    r = select_confidence(cands, prefer_a, k=2, shortcut_threshold=1.1)
    assert r.route == "tournament" and _ex_of_pick(cands, r) == 1


def test_consistency_weight_can_veto_a_thin_judge_margin():
    # A overwhelmingly consistent (5) vs B (1). Even if the judge mildly favors
    # B, the consistency weight should keep A. Judge splits 2A/1B per pair.
    cands = ([_c(SET_A, 1)] * 5) + [_c(SET_B, 0)]

    def split(a, b):
        return ["A", "A", "B"]
    r = select_confidence(cands, split, k=2, shortcut_threshold=1.1)
    assert _ex_of_pick(cands, r) == 1  # A retained


if __name__ == "__main__":
    import sys
    import pytest
    sys.exit(pytest.main([__file__, "-q"]))
