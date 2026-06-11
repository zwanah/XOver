"""Load TACO's precomputed similarity + influence scores and blend them.

Each score pkl is a list of entries ``{query_index, scores[N], train_indices[N]}``
over the N=6180 demo pool, already shifted to [0, 1] (cosine via (cos+1)/2). We
reuse them verbatim — never recompute. Relevance is a plain weighted blend:

    relevance(q, e) = alpha * sim(q, e) + (1 - alpha) * influence(q, e)

and selection is descending top-k over that, with a deterministic tie-break.
No budget, no diversity, no efficiency term (spec §2).
"""
from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from typing import Dict, List, Protocol, Tuple, runtime_checkable

logger = logging.getLogger(__name__)

# Tolerance for the "already in [0, 1]" precondition (float round-trip slack).
_RANGE_EPS = 1e-4


@runtime_checkable
class ScoreFunction(Protocol):
    """Pluggable per-query relevance scorer over the demo pool (C2 seam).

    Any object exposing ``relevance(alpha)`` ((pool_index, score) pairs) and
    ``ranked(alpha)`` (pool indices, relevance-descending) satisfies this
    interface. The default implementation is :class:`QueryScores` (a blend of
    TACO's precomputed similarity + influence). A future *trained* retriever
    (the R1–R3 variants) can be dropped in here without touching the selection
    (:mod:`select`) or assembly (:mod:`assemble`) stages.
    """

    query_index: int

    def relevance(self, alpha: float) -> List[Tuple[int, float]]: ...

    def ranked(self, alpha: float) -> List[int]: ...


@dataclass(frozen=True)
class QueryScores:
    """Per-eval-query score vectors, aligned to ``train_indices`` (pool indices)."""

    query_index: int
    train_indices: Tuple[int, ...]
    sim: Tuple[float, ...]
    influence: Tuple[float, ...]

    def relevance(self, alpha: float) -> List[Tuple[int, float]]:
        """(pool_index, relevance) pairs in ``train_indices`` order."""
        return [
            (ti, alpha * s + (1.0 - alpha) * inf)
            for ti, s, inf in zip(self.train_indices, self.sim, self.influence)
        ]

    def ranked(self, alpha: float) -> List[int]:
        """Pool indices sorted by relevance descending; ties broken by pool index asc.

        The tie-break makes selection fully deterministic given (alpha, scores).
        """
        scored = self.relevance(alpha)
        scored.sort(key=lambda pr: (-pr[1], pr[0]))
        return [pool_idx for pool_idx, _ in scored]


def _load_pkl(path: str) -> Dict[int, Dict]:
    with open(path, "rb") as f:
        data = pickle.load(f)
    return {int(e["query_index"]): e for e in data}


def _assert_in_unit_range(name: str, query_index: int, vec: List[float]) -> None:
    lo, hi = min(vec), max(vec)
    if lo < -_RANGE_EPS or hi > 1.0 + _RANGE_EPS:
        raise ValueError(
            f"{name} scores for query {query_index} out of [0,1]: "
            f"min={lo:.4f} max={hi:.4f}. The blend assumes both signals are "
            f"already normalized; recheck the source pkl."
        )


class ScoreTable:
    """Aligned similarity + influence scores, keyed by eval ``query_index``."""

    def __init__(self, by_query: Dict[int, QueryScores], pool_size: int):
        self.by_query = by_query
        self.pool_size = pool_size

    @property
    def query_indices(self) -> List[int]:
        return sorted(self.by_query)

    @classmethod
    def load(cls, sim_path: str, influence_path: str, pool_size: int) -> "ScoreTable":
        """Load and align the two pkls; hard-fail on any structural mismatch."""
        sim_raw = _load_pkl(sim_path)
        inf_raw = _load_pkl(influence_path)

        if set(sim_raw) != set(inf_raw):
            raise ValueError(
                f"similarity / influence query_index sets differ: "
                f"sim={len(sim_raw)} inf={len(inf_raw)} entries"
            )

        by_query: Dict[int, QueryScores] = {}
        for qi in sim_raw:
            s_e, i_e = sim_raw[qi], inf_raw[qi]
            s_ti = [int(x) for x in s_e["train_indices"]]
            i_ti = [int(x) for x in i_e["train_indices"]]
            if len(s_ti) != pool_size:
                raise ValueError(
                    f"query {qi}: pool size {len(s_ti)} != expected {pool_size}"
                )
            if s_ti != i_ti:
                raise ValueError(
                    f"query {qi}: sim/influence train_indices misaligned — cannot blend"
                )
            sim_v = [float(x) for x in s_e["scores"]]
            inf_v = [float(x) for x in i_e["scores"]]
            _assert_in_unit_range("similarity", qi, sim_v)
            _assert_in_unit_range("influence", qi, inf_v)
            by_query[qi] = QueryScores(
                query_index=qi,
                train_indices=tuple(s_ti),
                sim=tuple(sim_v),
                influence=tuple(inf_v),
            )

        logger.info(
            "loaded scores: %d queries x %d pool (sim=%s, inf=%s)",
            len(by_query), pool_size, sim_path.split("/")[-1],
            influence_path.split("/")[-1],
        )
        return cls(by_query, pool_size)
