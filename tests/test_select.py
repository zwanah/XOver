"""Tests for the separated selection stage (TopKSelector + ProfileDppSelector)."""
import numpy as np

from pipeline.c2_retrieval.pool_map import PoolEntry
from pipeline.c2_retrieval.scores import QueryScores
from pipeline.c2_retrieval.select import ProfileDppSelector, TopKSelector


class StubPool:
    """Minimal PoolMap stand-in: pool index -> PoolEntry (gold drives dedup)."""

    def __init__(self, golds):
        self._golds = golds

    def entry(self, i):
        return PoolEntry(i, "train", i, f"nl{i}", self._golds[i])


def _qs_descending(n):
    """QueryScores whose ranked(alpha=1.0) == [0, 1, ..., n-1]."""
    sim = [1.0 - i * 0.01 for i in range(n)]
    return QueryScores(0, tuple(range(n)), tuple(sim), tuple([0.0] * n))


def test_topk_plain():
    n = 5
    golds = {i: f"[out:json];q{i};out;" for i in range(n)}
    sel = TopKSelector(alpha=1.0, dedup_prefix_len=80)
    res = sel.select(_qs_descending(n), StubPool(golds), k=3)
    assert res.selected == [0, 1, 2]
    assert res.dedup_skips == []


def test_dedup_by_prefix_skips_and_continues():
    # idx1's gold shares the 80-char prefix with idx0 -> skipped, selection continues.
    golds = {0: "[out:json];SAME;out;", 1: "[out:json];SAME;out;",
             2: "[out:json];q2;out;", 3: "[out:json];q3;out;", 4: "[out:json];q4;out;"}
    sel = TopKSelector(alpha=1.0, dedup_prefix_len=80)
    res = sel.select(_qs_descending(5), StubPool(golds), k=3)
    assert res.selected == [0, 2, 3]
    assert res.dedup_skips == [1]


def test_underfill_when_ranking_exhausted():
    # Only 2 distinct golds; k=3 -> underfilled, no crash.
    golds = {0: "[out:json];A;out;", 1: "[out:json];A;out;", 2: "[out:json];A;out;"}
    sel = TopKSelector(alpha=1.0, dedup_prefix_len=80)
    res = sel.select(_qs_descending(3), StubPool(golds), k=3)
    assert res.selected == [0]
    assert res.dedup_skips == [1, 2]


def test_alpha_blends_sim_and_influence():
    # sim ranks [0,1,2]; influence reversed. alpha=0 -> influence dominates -> [2,1,0].
    qs = QueryScores(0, (0, 1, 2), (0.9, 0.5, 0.1), (0.1, 0.5, 0.9))
    golds = {i: f"[out:json];q{i};out;" for i in range(3)}
    sel = TopKSelector(alpha=0.0, dedup_prefix_len=80)
    res = sel.select(qs, StubPool(golds), k=3)
    assert res.selected == [2, 1, 0]


# --- ProfileDppSelector -----------------------------------------------------------

class StubKernel:
    """Kernel provider stand-in: subsets a fixed pool kernel, forces unit diagonal."""

    def __init__(self, K):
        self._K = np.asarray(K, dtype=np.float64)

    def gram(self, indices, query_index=None):
        idx = np.asarray(indices, dtype=np.int64)
        sub = self._K[np.ix_(idx, idx)].astype(np.float64, copy=True)
        sub = 0.5 * (sub + sub.T)
        np.fill_diagonal(sub, 1.0)
        return sub


def test_profile_dpp_lambda0_equals_topk():
    # lambda=0 reduces EXACTLY to relevance top-k, regardless of the kernel.
    n = 5
    golds = {i: f"[out:json];q{i};out;" for i in range(n)}
    pool = StubPool(golds)
    qs = _qs_descending(n)
    # Any PSD-ish off-diagonal kernel; lambda=0 must ignore it.
    K = 0.3 * np.ones((n, n)) + 0.7 * np.eye(n)
    dpp = ProfileDppSelector(
        alpha=1.0, dedup_prefix_len=80, M=n, lam=0.0, phi_mode="linear",
        kernel_provider=StubKernel(K),
    )
    topk = TopKSelector(alpha=1.0, dedup_prefix_len=80)
    assert dpp.select(qs, pool, k=3).selected == topk.select(qs, pool, k=3).selected == [0, 1, 2]


def test_profile_dpp_diversifies_redundant_pair():
    # idx0 and idx1 are perfectly redundant (K=1); idx2 is orthogonal. With a large
    # lambda the reranker should NOT pick both 0 and 1 — it swaps in the distinct idx2.
    n = 3
    golds = {i: f"[out:json];q{i};out;" for i in range(n)}
    pool = StubPool(golds)
    qs = _qs_descending(n)  # relevance order 0 > 1 > 2
    K = np.array([[1.0, 1.0, 0.0],
                  [1.0, 1.0, 0.0],
                  [0.0, 0.0, 1.0]])
    dpp = ProfileDppSelector(
        alpha=1.0, dedup_prefix_len=80, M=n, lam=10.0, phi_mode="linear",
        kernel_provider=StubKernel(K),
    )
    res = dpp.select(qs, pool, k=2)
    assert set(res.selected) == {0, 2}  # distinct pair beats the redundant {0,1}


def test_profile_dpp_respects_dedup_and_M():
    # dedup-by-prefix still applies when building the top-M candidate pool.
    golds = {0: "[out:json];SAME;out;", 1: "[out:json];SAME;out;",
             2: "[out:json];q2;out;", 3: "[out:json];q3;out;"}
    pool = StubPool(golds)
    qs = _qs_descending(4)
    dpp = ProfileDppSelector(
        alpha=1.0, dedup_prefix_len=80, M=20, lam=0.0, phi_mode="linear",
        kernel_provider=StubKernel(np.eye(4)),
    )
    res = dpp.select(qs, pool, k=3)
    assert res.selected == [0, 2, 3]
    assert res.dedup_skips == [1]
