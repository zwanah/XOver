"""Query-independent *cross-lingual relevance-profile* kernel for the DPP selector.

Where the euclidean kernel (``KernelProvider``) diversifies over raw NL-embedding
similarity and the bridge kernel over per-query ``q-e`` bridge vectors, this kernel
diversifies over each demo's **cross-lingual relevance profile**: the vector of how
useful demo ``e`` is to a fixed bank of multilingual eval queries, read straight from
the multilingual influence model (ml_8x). For demo ``e``::

    p_e = [ inf_en(q_1, e), ..., inf_en(q_N, e),
            inf_zh(q_1, e), ..., inf_zh(q_N, e),
            inf_ja(q_1, e), ..., inf_ja(q_N, e) ]

Two demos are redundant evidence when they are jointly useful to the *same* queries
across languages — i.e. their profiles co-vary. We therefore measure redundancy by the
**cosine of mean-centered** profiles (Pearson correlation), mapped to ``[0, 1]``::

    K(e_i, e_j) = (1 + corr(p_ei, p_ej)) / 2

The centering is load-bearing, not cosmetic: raw influence is dominated by a demo-intrinsic
"generically useful" component, so *raw* cosine over profiles is near-rank-1 (empirically
off-diagonal mean 0.988, std 0.009 over top-20 candidate sets) — every demo looks redundant
with every other and ``lambda`` changes no picks. Per-column (per ``(lang, query)``)
mean-centering removes that shared baseline and exposes the demo-specific covariation that
*is* discriminative (off-diagonal mean 0.835, std 0.150). See
``docs/profile-dpp-results.md`` for the spread analysis.

``K = (J + ÛÛᵀ)/2`` is PSD with unit diagonal (sum of the all-ones ``J`` and a Gram matrix
of unit-norm rows), so it satisfies the greedy DPP contract (``div_gain >= 0``,
``schur ∈ [1, 1+K_ii]``) — same construction as ``BridgeKernelProvider``. Because the
centered/normalized profile matrix is fixed (the influence bank does not depend on the eval
query being served), the kernel is query-independent: ``gram`` subsets precomputed rows and
``query_index`` is accepted for interface parity only (advisor: leave-one-out is a red
herring — q*'s ~3 columns out of ~3000 are near-constant across the relevant candidates and
carry the same no-gold influence signal already in the quality term).
"""
from __future__ import annotations

import logging
import pickle
from typing import Dict, List, Sequence

import numpy as np

logger = logging.getLogger(__name__)


def _unit_rows(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0.0] = 1.0
    return x / n


class ProfileKernelProvider:
    """Holds the centered + row-normalized profile matrix; subsets candidate Gram blocks."""

    def __init__(self, unit_profiles: np.ndarray):
        # (pool, D) rows already mean-centered per column and L2-normalized.
        self._U = np.asarray(unit_profiles, dtype=np.float64)

    @classmethod
    def from_influence_pkls(
        cls,
        paths: Sequence[str],
        pool_size: int,
    ) -> "ProfileKernelProvider":
        """Build the profile matrix from per-language influence pkls.

        Each pkl is a list of ``{query_index, scores[pool_size], train_indices[pool_size]}``
        over the shared demo pool (the same format ``ScoreTable`` consumes). All languages
        must cover the SAME ``query_index`` set — same qi means the same (translated) sample,
        which is the precondition the profile alignment relies on. Columns are ordered
        ``(lang0, q0..qN), (lang1, q0..qN), ...`` with queries sorted ascending per language.
        """
        if not paths:
            raise ValueError("ProfileKernelProvider needs at least one influence pkl path")

        per_lang: List[Dict[int, dict]] = []
        for path in paths:
            with open(path, "rb") as f:
                raw = pickle.load(f)
            per_lang.append({int(e["query_index"]): e for e in raw})

        qsets = [set(d) for d in per_lang]
        common = qsets[0]
        for s in qsets[1:]:
            if s != common:
                raise ValueError(
                    "profile influence pkls have mismatched query_index sets "
                    f"({[len(s) for s in qsets]} entries) — same qi must mean the same "
                    "sample across languages; cannot align profiles"
                )
        queries = sorted(common)

        n_cols = len(per_lang) * len(queries)
        P = np.zeros((pool_size, n_cols), dtype=np.float64)
        col = 0
        for d in per_lang:
            for qi in queries:
                e = d[qi]
                ti = np.asarray(e["train_indices"], dtype=np.int64)
                sc = np.asarray(e["scores"], dtype=np.float64)
                if ti.shape[0] != pool_size:
                    raise ValueError(
                        f"profile pkl query {qi}: pool size {ti.shape[0]} != "
                        f"expected {pool_size}"
                    )
                P[ti, col] = sc
                col += 1

        # Per-column mean-center (remove the per-query global relevance baseline), then
        # row-normalize so the candidate Gram is a cosine of the demo-specific deviations.
        P -= P.mean(axis=0, keepdims=True)
        U = _unit_rows(P)
        logger.info(
            "loaded profile kernel: %d demos x %d cols (%d langs x %d queries)",
            pool_size, n_cols, len(per_lang), len(queries),
        )
        return cls(U)

    def gram(self, indices: list, query_index: int = None) -> np.ndarray:
        """Centered-profile Gram ``K=(1+corr)/2`` over candidate ``indices``.

        Symmetrized with an exact unit diagonal so the greedy's ``a = 1 + K_ii`` term is 2.
        ``query_index`` is accepted for interface parity with ``BridgeKernelProvider`` and
        ignored (this kernel is query-independent).
        """
        idx = np.asarray(indices, dtype=np.int64)
        Uc = self._U[idx]
        s = 0.5 * (1.0 + Uc @ Uc.T)
        s = 0.5 * (s + s.T)
        np.fill_diagonal(s, 1.0)
        return s
