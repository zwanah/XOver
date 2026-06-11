"""DeepEye-style confidence selection over a denotation-grouped candidate pool.

This is the contested-case re-ranker that sits on top of plain denotation
voting (``selectors.nonempty_vote``). "Confidence" combines two signals:

* **consistency confidence** — the fraction of valid candidates that agree on a
  denotation (free from the K-pool, same signal voting uses); and
* **LLM-judge confidence** — a pairwise tournament over the top-k denotation
  groups whose robust win-probability re-ranks the contested cases.

The pipeline (mirrors ``references/DeepEye-SQL`` ``SQLSelectionRunner``):

1. Group candidates by denotation signature; drop errors. With
   ``prefer_nonempty`` the vote pool is the non-empty candidates (falling back
   to all valid when none are non-empty) — identical pool to ``nonempty_vote``.
2. Rank groups by consistency; keep the top ``k`` distinct denotations.
3. **Shortcut:** if the top-1 consistency >= ``shortcut_threshold``, pick it —
   no LLM. High-agreement queries cost nothing and reduce to majority vote.
4. **Tournament:** otherwise run a pairwise LLM judge over the top-k group
   representatives, build a robust win matrix, weight each candidate's mean win
   probability by its (normalised) consistency, and pick the argmax.

This module is **network-free**: the LLM judge is injected as a ``comparator``
callable so the logic is unit-testable with a fake. With a comparator that
always returns ``TIE``, confidence selection reduces *exactly* to
``nonempty_vote`` (see tests) — the harness invariant.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from .denotation import Signature, is_empty_sig, is_error_sig

# A comparator judges an ordered pair of group representatives and returns a
# list of votes, one per sampled judge call. ``a`` is always the higher-
# consistency candidate (the A-biased "Top-1"); each vote is "A", "B" or "TIE".
Comparator = Callable[["GroupRep", "GroupRep"], List[str]]


@dataclass
class GroupRep:
    """Representative of one denotation group among the candidates.

    ``index`` points at the earliest pool member with this denotation, so the
    caller can map a pick back to a concrete candidate (its ex, or its text).
    """

    sig: Signature
    index: int
    count: int
    consistency: float
    ex: int
    empty: bool


@dataclass
class ConfidenceResult:
    """Pick plus diagnostics, so callers can attribute a null to its cause."""

    picked_index: int                      # candidate index, or -1 if no pool
    route: str                             # "empty_pool"|"single"|"shortcut"|"tournament"
    n_groups: int
    top1_consistency: float
    tournament_groups: List[GroupRep] = field(default_factory=list)
    win_prob: List[float] = field(default_factory=list)
    final_score: List[float] = field(default_factory=list)


def _rank_groups(cands: List[Dict], prefer_nonempty: bool) -> List[GroupRep]:
    """Group candidates by denotation and rank by consistency (desc).

    Pool = non-empty valid candidates (``prefer_nonempty``, falling back to all
    valid when none are non-empty) — exactly ``nonempty_vote``'s pool. Tie-break
    matches ``selectors._group_and_pick``: larger group, then non-empty over
    empty, then earliest pool position.
    """
    valid = [i for i, c in enumerate(cands) if not is_error_sig(c["sig"])]
    if not valid:
        return []
    nonempty = [i for i in valid if not is_empty_sig(cands[i]["sig"])]
    pool = nonempty if (prefer_nonempty and nonempty) else valid

    groups: Dict[Signature, Dict] = {}
    for i in pool:
        sig = cands[i]["sig"]
        g = groups.get(sig)
        if g is None:
            groups[sig] = {"count": 1, "first": i, "ex": int(cands[i]["ex"]),
                           "empty": is_empty_sig(sig)}
        else:
            g["count"] += 1
    n_pool = len(pool)
    reps = [
        GroupRep(sig=sig, index=g["first"], count=g["count"],
                 consistency=g["count"] / n_pool, ex=g["ex"], empty=g["empty"])
        for sig, g in groups.items()
    ]
    reps.sort(key=lambda r: (-r.count, r.empty, r.index))
    return reps


def _pairs(n: int) -> List[tuple]:
    return [(i, j) for i in range(n) for j in range(n) if i < j]


def select_confidence(
    cands: List[Dict],
    comparator: Optional[Comparator],
    *,
    k: int = 2,
    shortcut_threshold: float = 0.8,
    prefer_nonempty: bool = True,
) -> ConfidenceResult:
    """Pick a candidate index via consistency + (contested) LLM tournament.

    ``cands`` items need at least ``sig`` (frozen, hashable) and ``ex``;
    ``comparator`` is called only on contested queries and may be ``None`` when
    the caller knows no tournament will run (e.g. shortcut-only ablations).
    """
    reps = _rank_groups(cands, prefer_nonempty)
    if not reps:
        return ConfidenceResult(picked_index=-1, route="empty_pool",
                                n_groups=0, top1_consistency=0.0)

    top1 = reps[0]
    if len(reps) == 1:
        return ConfidenceResult(picked_index=top1.index, route="single",
                                n_groups=1, top1_consistency=top1.consistency)

    if top1.consistency >= shortcut_threshold:
        return ConfidenceResult(picked_index=top1.index, route="shortcut",
                                n_groups=len(reps),
                                top1_consistency=top1.consistency)

    top_k = reps[:max(1, k)]
    if len(top_k) == 1:
        return ConfidenceResult(picked_index=top_k[0].index, route="single",
                                n_groups=len(reps),
                                top1_consistency=top1.consistency)
    if comparator is None:
        # No judge available: degrade to top-1 consistency (== nonempty_vote).
        return ConfidenceResult(picked_index=top1.index, route="shortcut",
                                n_groups=len(reps),
                                top1_consistency=top1.consistency)

    m = len(top_k)
    # win_sum[i][j] accumulates fractional wins of i over j across votes.
    win_sum = [[0.0] * m for _ in range(m)]
    n_votes = [[0] * m for _ in range(m)]
    for i, j in _pairs(m):
        votes = comparator(top_k[i], top_k[j]) or []
        for v in votes:
            v = (v or "").strip().upper()
            if v == "A":
                win_sum[i][j] += 1.0
            elif v == "B":
                win_sum[j][i] += 1.0
            elif v == "TIE":
                win_sum[i][j] += 0.5
                win_sum[j][i] += 0.5
            else:
                continue  # unparseable vote contributes nothing
            n_votes[i][j] += 1
            n_votes[j][i] += 1

    # Robust win probability: average vote outcome per opponent, then mean over
    # opponents. Pairs with no parsed vote default to 0.5 (no information).
    win_prob = [0.0] * m
    for i in range(m):
        probs = []
        for j in range(m):
            if i == j:
                continue
            probs.append(win_sum[i][j] / n_votes[i][j] if n_votes[i][j] else 0.5)
        win_prob[i] = sum(probs) / len(probs) if probs else 0.0

    weights = [r.consistency for r in top_k]
    wsum = sum(weights) or 1.0
    final_score = [win_prob[i] * (weights[i] / wsum) for i in range(m)]

    best = max(range(m), key=lambda i: (final_score[i], -top_k[i].index))
    return ConfidenceResult(
        picked_index=top_k[best].index, route="tournament",
        n_groups=len(reps), top1_consistency=top1.consistency,
        tournament_groups=top_k, win_prob=win_prob, final_score=final_score,
    )
