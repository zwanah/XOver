"""Tests for the cross-lingual relevance-profile kernel (C2 diversity sub-claim).

Invariants the greedy DPP contract relies on: PSD with exact unit diagonal; demos
with identical (post-centering) profiles are maximally redundant (K -> 1); and a
mismatched per-language query_index set is rejected (profiles cannot be aligned).
"""
import pickle

import numpy as np
import pytest

from pipeline.c2_retrieval.profile_kernel import ProfileKernelProvider


def _write_pkl(path, entries):
    with open(path, "wb") as f:
        pickle.dump(entries, f)


def _lang_entries(pool_size, query_to_scores):
    """One language's pkl: [{query_index, train_indices[pool], scores[pool]}]."""
    return [
        {"query_index": qi, "train_indices": list(range(pool_size)), "scores": list(scores)}
        for qi, scores in query_to_scores.items()
    ]


def test_gram_is_psd_with_unit_diagonal(tmp_path):
    pool_size = 6
    rng = np.random.RandomState(0)
    paths = []
    for lang in ("en", "zh"):
        p = tmp_path / f"inf_{lang}.pkl"
        q2s = {qi: rng.rand(pool_size) for qi in (0, 1, 2)}
        _write_pkl(p, _lang_entries(pool_size, q2s))
        paths.append(str(p))

    prov = ProfileKernelProvider.from_influence_pkls(paths, pool_size)
    K = prov.gram([0, 1, 2, 3, 4, 5])
    assert np.allclose(np.diag(K), 1.0)              # exact unit diagonal
    assert np.allclose(K, K.T)                        # symmetric
    assert np.linalg.eigvalsh(K).min() > -1e-8        # PSD (K = (J + Gram)/2)
    assert (K >= -1e-9).all() and (K <= 1.0 + 1e-9).all()  # (1+corr)/2 in [0,1]


def test_identical_profiles_are_redundant(tmp_path):
    # Two demos with the SAME per-(lang,query) influence pattern (after centering they
    # point the same way) -> kernel entry ~1; a distinct demo -> lower.
    pool_size = 3
    # demos 0 and 1 share a profile; demo 2 differs. Use 2 queries x 1 lang.
    scores_q0 = [0.9, 0.9, 0.1]
    scores_q1 = [0.2, 0.2, 0.8]
    p = tmp_path / "inf_en.pkl"
    _write_pkl(p, _lang_entries(pool_size, {0: scores_q0, 1: scores_q1}))
    prov = ProfileKernelProvider.from_influence_pkls([str(p)], pool_size)
    K = prov.gram([0, 1, 2])
    assert K[0, 1] == pytest.approx(1.0, abs=1e-6)     # identical profiles -> redundant
    assert K[0, 2] < K[0, 1]                            # distinct demo less redundant


def test_mismatched_query_sets_rejected(tmp_path):
    pool_size = 4
    p_en = tmp_path / "inf_en.pkl"
    p_zh = tmp_path / "inf_zh.pkl"
    _write_pkl(p_en, _lang_entries(pool_size, {0: [0.1, 0.2, 0.3, 0.4], 1: [0.4, 0.3, 0.2, 0.1]}))
    _write_pkl(p_zh, _lang_entries(pool_size, {0: [0.1, 0.2, 0.3, 0.4], 2: [0.4, 0.3, 0.2, 0.1]}))
    with pytest.raises(ValueError, match="mismatched query_index"):
        ProfileKernelProvider.from_influence_pkls([str(p_en), str(p_zh)], pool_size)


def test_empty_paths_rejected():
    with pytest.raises(ValueError, match="at least one influence pkl"):
        ProfileKernelProvider.from_influence_pkls([], 6)
