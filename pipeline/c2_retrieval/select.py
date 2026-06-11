"""Selection stage (separated from assembly): scores -> top-k demo-pool indices.

This is the **swappable** stage of C2. Scoring (:mod:`scores`) produces a
relevance ranking; this module turns that ranking into a concrete list of
demo-pool **indices**; assembly (:mod:`assemble`) then consumes only those
indices. Keeping the three stages decoupled means an alternative example-
selection method (diversity / MMR / budget-aware / a learned selector) can be
dropped in by implementing the :class:`Selector` protocol — without touching the
score function or the prompt assembler.

The default :class:`TopKSelector` walks the relevance ranking most-relevant-first
and applies one skip rule:

  * **dedup-by-gold-prefix** (W1 parity) — skip a candidate whose gold OverpassQL
    shares an ``dedup_prefix_len``-char prefix with an already-selected demo.

The CoT-era *missing-CoT backfill* skip is intentionally gone: in the plain
setting every pool entry is a valid ``(nl_question, OverpassQL)`` demo, so there
is nothing to quarantine. ``select()`` returns a :class:`SelectionResult` whose
``selected`` field is the ordered list of pool indices passed downstream.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Protocol

from .dpp import greedy_dpp_select
from .pool_map import PoolMap
from .scores import ScoreFunction

logger = logging.getLogger(__name__)


@dataclass
class SelectionResult:
    """Outcome of selecting demos for one eval query."""

    selected: List[int] = field(default_factory=list)     # pool indices, relevance/select order
    dedup_skips: List[int] = field(default_factory=list)  # pool indices skipped (prefix dup)


class Selector(Protocol):
    """Pluggable example-selection strategy: ranking -> ordered pool indices."""

    def select(self, scores: ScoreFunction, pool_map: PoolMap, k: int) -> SelectionResult: ...


def _candidate_pool(
    scores: ScoreFunction,
    pool_map: PoolMap,
    M: int,
    alpha: float,
    dedup_prefix_len: int,
    result: "SelectionResult",
) -> List[int]:
    """Top-M relevance-ordered candidates after dedup-by-gold-prefix (plain mode).

    Shared candidate construction for the rerankers so their pool is built identically
    to :class:`TopKSelector`'s walk (same dedup rule, same order). Plain mode has no
    missing-CoT skip — every pool entry is a valid ``(nl_question, OverpassQL)`` demo.
    """
    seen_prefix = set()
    cand: List[int] = []
    for idx in scores.ranked(alpha):
        if len(cand) >= M:
            break
        prefix = pool_map.entry(idx).overpassql[:dedup_prefix_len]
        if prefix in seen_prefix:
            result.dedup_skips.append(idx)
            continue
        seen_prefix.add(prefix)
        cand.append(idx)
    return cand


@dataclass(frozen=True)
class TopKSelector:
    """Relevance top-k with dedup-by-gold-prefix (the default plain-mode selector)."""

    alpha: float
    dedup_prefix_len: int

    def select(self, scores: ScoreFunction, pool_map: PoolMap, k: int) -> SelectionResult:
        result = SelectionResult()
        seen_prefix = set()
        for idx in scores.ranked(self.alpha):
            if len(result.selected) >= k:
                break
            prefix = pool_map.entry(idx).overpassql[: self.dedup_prefix_len]
            if prefix in seen_prefix:
                result.dedup_skips.append(idx)
                continue
            seen_prefix.add(prefix)
            result.selected.append(idx)

        if len(result.selected) < k:
            logger.warning(
                "query %d: only %d/%d demos filled (ranking exhausted; %d dedup)",
                scores.query_index, len(result.selected), k, len(result.dedup_skips),
            )
        return result


@dataclass(frozen=True)
class ProfileDppSelector:
    """Cross-lingual relevance-profile DPP diversity over the relevance top-M pool.

    The C2 diversity sub-claim (methodology diversity-pivot). Maximizes

        f_phi(S) = phi(sum_{e in S} R(q,e)) + lambda * logdet(I + K_S)

    via the fixed-k greedy MAP (:func:`greedy_dpp_select`), where ``K`` is the
    centered cross-lingual relevance-profile kernel (:class:`ProfileKernelProvider`):
    two demos are redundant evidence when their per-(language, query) influence
    profiles co-vary. ``lam == 0`` reduces EXACTLY to :class:`TopKSelector` (the
    candidate list is fed in relevance-descending order, ``phi`` is linear, and the
    greedy tie-break preserves that order) — regression-checked.

    The kernel is query-independent (the multilingual influence bank does not depend
    on the eval query), so one provider is shared across all queries; ``query_index``
    is passed through to ``gram`` only for interface parity.
    """

    alpha: float
    dedup_prefix_len: int
    M: int
    lam: float
    phi_mode: str
    kernel_provider: object  # ProfileKernelProvider (duck-typed: exposes .gram(indices, qi))

    def select(self, scores: ScoreFunction, pool_map: PoolMap, k: int) -> SelectionResult:
        result = SelectionResult()
        cand = _candidate_pool(
            scores, pool_map, self.M, self.alpha, self.dedup_prefix_len, result
        )
        if cand:
            rel_map = dict(scores.relevance(self.alpha))
            relevance = [rel_map[idx] for idx in cand]
            K = self.kernel_provider.gram(cand, scores.query_index)
            positions = greedy_dpp_select(relevance, K, k, self.lam, self.phi_mode)
            result.selected = [cand[p] for p in positions]

        if len(result.selected) < k:
            logger.warning(
                "query %d: profile-dpp filled only %d/%d demos (M=%d, %d dedup)",
                scores.query_index, len(result.selected), k, self.M,
                len(result.dedup_skips),
            )
        return result
