"""Candidate selectors for the diagnostic.

Each selector consumes a query's candidate pool and returns the EX correctness
(0/1) of the candidate it would pick, so selectors are directly comparable on
the same pools. A candidate is a dict with at least:

    {"sig": Signature, "ex": int}   # ex = 1 iff EX-correct vs gold (canonical)

All candidates sharing a denotation signature share the same ``ex`` by
construction (EX compares the candidate denotation to the gold denotation), so
a group's correctness is well-defined by any member.

Selectors implemented (the execution-only baselines all three SOTA text-to-SQL
systems rely on — OpenSearch-SQL/DeepEye-SQL/DPC-SQL):

* ``top1_greedy``      — the temp-0 greedy decode (= Pass@1, the shippable pick).
* ``oracle_passk``     — 1 iff ANY candidate is correct (the headroom ceiling).
* ``majority_vote``    — denotation self-consistency: pick the largest group.
* ``nonempty_vote``    — majority vote restricted to non-empty denotations first.
* ``random_expected``  — expected accuracy of a uniformly random pick (mean ex).
"""
from __future__ import annotations

from typing import Dict, List, Set

from .denotation import Signature, is_empty_sig, is_error_sig


def top1_greedy(greedy_ex: int) -> int:
    return int(greedy_ex)


def oracle_passk(cands: List[Dict]) -> int:
    return 1 if any(int(c["ex"]) == 1 for c in cands) else 0


def random_expected(cands: List[Dict]) -> float:
    if not cands:
        return 0.0
    return sum(int(c["ex"]) for c in cands) / len(cands)


def _group_and_pick(cands: List[Dict], votable_idx: Set[int]) -> int:
    """Pick the largest denotation group among the votable indices; return its ex.

    Deterministic tie-break: larger group first, then prefer non-empty over
    empty, then the group whose first member appears earliest in pool order.
    Returns 0 if there is nothing votable.
    """
    if not votable_idx:
        return 0
    groups: Dict[Signature, Dict] = {}
    for i, c in enumerate(cands):
        if i not in votable_idx:
            continue
        sig = c["sig"]
        g = groups.get(sig)
        if g is None:
            groups[sig] = {"count": 1, "first": i, "ex": int(c["ex"]),
                           "empty": is_empty_sig(sig)}
        else:
            g["count"] += 1
    best = min(
        groups.values(),
        key=lambda g: (-g["count"], g["empty"], g["first"]),
    )
    return int(best["ex"])


def majority_vote(cands: List[Dict]) -> int:
    """Denotation self-consistency over all non-error candidates."""
    votable = {i for i, c in enumerate(cands) if not is_error_sig(c["sig"])}
    return _group_and_pick(cands, votable)


def nonempty_vote(cands: List[Dict]) -> int:
    """Prefer the majority *non-empty* denotation; fall back to full vote."""
    nonempty = {i for i, c in enumerate(cands)
                if not is_error_sig(c["sig"]) and not is_empty_sig(c["sig"])}
    if nonempty:
        return _group_and_pick(cands, nonempty)
    return majority_vote(cands)
