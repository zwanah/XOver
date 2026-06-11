"""DPP / LogDet diversity demo selector (fixed-k, lambda-controlled).

Ports TACO's objective (``eval_backend/TACO/utils/budget_selection.py``)

    f_phi(S) = phi(sum_i rel_i) + lambda * logdet(I + K_S)

using the fast greedy MAP inference with an incremental Cholesky factor of
``A_S = I + K_S``. We take the **TACO_C unit-gain variant**: fixed budget of ``k``
demos, pick the argmax marginal gain each step, NO token-cost / efficiency division.

Dimensional contract (see plan ``relevance-dpp-iterative-hopcroft.md``):
  * ``relevance`` is the alpha-blend, already normalized to [0, 1].
  * With ``phi_mode='linear'`` the per-step ``rel_gain == rel_i`` (order-independent),
    so ``lambda == 0`` reduces EXACTLY to relevance top-k (the candidate list is fed in
    relevance-descending order, and the ``(-gain, position)`` tie-break keeps that order).
  * ``div_gain = ln(schur)`` is the log-volume (nats) item ``i`` adds orthogonal to the
    selected set; ``lambda`` is the exchange rate [relevance / nat] between the two terms.

For a PSD kernel with unit diagonal the Schur complement is always in ``[1, 1+K_ii]`` so
``div_gain >= 0`` and the ``schur <= eps`` guard never fires — it is kept only as a
NaN/Inf safety net for a degenerate (non-PSD) kernel.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

_EPS = 1e-12


def phi_concave(x: float, mode: str = "linear") -> float:
    """Concave transform of the cumulative relevance ``sum rel_i``.

    ``linear`` keeps lambda interpretable (rel_gain stays in [0, 1] units); ``log``
    returns bits (log2) and ``sqrt`` returns sqrt-relevance. All are monotone, so
    ``lambda == 0`` reduces to relevance top-k for every mode.
    """
    x = max(float(x), 0.0)
    if mode == "linear":
        return x
    if mode == "log":
        return float(np.log1p(x) / np.log(2.0))
    if mode == "sqrt":
        return float(np.sqrt(x))
    raise ValueError(f"unknown phi_mode {mode!r} (expected linear|log|sqrt)")


def chol_append(chol_L: Optional[np.ndarray], v: np.ndarray, new_diag: float) -> np.ndarray:
    """Extend the lower-triangular Cholesky factor of ``A_S = I + K_S`` by one element.

    ``L_new = [[L, 0], [v^T, new_diag]]`` where ``v = L^{-1} k`` and
    ``new_diag = sqrt(schur)``. O(t) per append.
    """
    if chol_L is None or chol_L.size == 0:
        return np.array([[new_diag]], dtype=np.float64)
    t = chol_L.shape[0]
    out = np.zeros((t + 1, t + 1), dtype=np.float64)
    out[:t, :t] = chol_L
    out[t, :t] = v
    out[t, t] = new_diag
    return out


def dpp_marginal_gain(
    idx: int,
    K: np.ndarray,
    relevance: np.ndarray,
    lam: float,
    sel_rows: Optional[np.ndarray],
    chol_L: Optional[np.ndarray],
    sum_rel: float,
    phi_mode: str,
    eps: float = _EPS,
) -> Tuple[float, np.ndarray, float]:
    """Marginal gain of adding ``idx`` to the current set.

    Returns ``(total_gain, v, new_diag)`` where ``v`` / ``new_diag`` feed ``chol_append``.
    ``total_gain == -inf`` signals an infeasible pick (degenerate Schur complement).
    """
    rel_gain = phi_concave(sum_rel + float(relevance[idx]), phi_mode) - phi_concave(sum_rel, phi_mode)
    a = 1.0 + float(K[idx, idx])  # diagonal of A_S = I + K_S

    if sel_rows is None or sel_rows.shape[0] == 0:
        schur = a
        v = np.empty((0,), dtype=np.float64)
    else:
        k_vec = sel_rows[:, idx].astype(np.float64, copy=False)  # K_{S, idx}
        v = np.linalg.solve(chol_L, k_vec)                       # L^{-1} k, O(t^2)
        schur = a - float(v @ v)                                 # Schur complement

    if not np.isfinite(schur) or schur <= eps:
        return -np.inf, v, 0.0

    div_gain = float(np.log(schur))
    new_diag = float(np.sqrt(schur))
    return rel_gain + lam * div_gain, v, new_diag


def greedy_dpp_select(
    relevance: List[float],
    K: np.ndarray,
    k: int,
    lam: float,
    phi_mode: str = "linear",
) -> List[int]:
    """Greedily pick up to ``k`` positions maximizing ``f_phi``.

    Args:
        relevance: per-candidate relevance, in candidate-list order (relevance desc).
        K: candidate-by-candidate PSD kernel (M x M), unit diagonal.
        k: number of demos to select (clamped to the candidate count).
        lam: diversity weight; ``0`` reproduces relevance top-k.
        phi_mode: concave transform of cumulative relevance.

    Returns:
        Selected POSITIONS into the candidate list, in selection order. Deterministic:
        ties broken by ``(-gain, position)`` so the lower (more relevant) position wins.
    """
    rel = np.asarray(relevance, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    n = K.shape[0]
    k = min(k, n)

    selected: List[int] = []
    sel_mask = np.zeros(n, dtype=bool)
    chol_L: Optional[np.ndarray] = None
    sel_rows: Optional[np.ndarray] = None
    sum_rel = 0.0

    while len(selected) < k:
        best_key: Optional[Tuple[float, int]] = None
        best: Optional[Tuple[int, np.ndarray, float]] = None
        for i in range(n):
            if sel_mask[i]:
                continue
            gain, v, new_diag = dpp_marginal_gain(
                i, K, rel, lam, sel_rows, chol_L, sum_rel, phi_mode
            )
            if not np.isfinite(gain):
                continue
            key = (-gain, i)
            if best_key is None or key < best_key:
                best_key, best = key, (i, v, new_diag)
        if best is None:
            break
        i, v, new_diag = best
        selected.append(i)
        sel_mask[i] = True
        chol_L = chol_append(chol_L, v, new_diag)
        sel_rows = K[np.asarray(selected, dtype=np.int64), :]
        sum_rel += float(rel[i])

    return selected
