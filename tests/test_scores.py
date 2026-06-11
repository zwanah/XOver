"""Tests for the relevance blend + top-k selection (no real pkls needed)."""
import pickle

import pytest

from pipeline.c2_retrieval.scores import QueryScores, ScoreTable


def _qs(train_indices, sim, influence, qi=0):
    return QueryScores(qi, tuple(train_indices), tuple(sim), tuple(influence))


def test_relevance_blend_math():
    qs = _qs([0, 1], sim=[1.0, 0.0], influence=[0.0, 1.0])
    # alpha=0.2 -> 0.2*sim + 0.8*influence
    rel = dict(qs.relevance(0.2))
    assert rel[0] == pytest.approx(0.2)
    assert rel[1] == pytest.approx(0.8)


def test_ranked_descending_and_deterministic_tiebreak():
    # pool 2 and 0 tie at relevance 0.5; tie-break = pool index ascending.
    qs = _qs([2, 0, 1], sim=[0.5, 0.5, 1.0], influence=[0.5, 0.5, 0.0])
    # alpha=1.0 -> relevance == sim: idx1=1.0, idx2=0.5, idx0=0.5
    order = qs.ranked(1.0)
    assert order[0] == 1
    assert order[1:] == [0, 2]  # tie broken by pool index asc


def test_alpha_extremes():
    qs = _qs([0, 1], sim=[0.9, 0.1], influence=[0.1, 0.9])
    assert qs.ranked(1.0)[0] == 0   # similarity-only picks idx0
    assert qs.ranked(0.0)[0] == 1   # influence-only picks idx1


def _write_pkl(path, qi, train_indices, scores):
    with open(path, "wb") as f:
        pickle.dump(
            [{"query_index": qi, "train_indices": list(train_indices),
              "scores": list(scores)}], f)


def test_scoretable_load_and_align(tmp_path):
    sim_p = tmp_path / "sim.pkl"
    inf_p = tmp_path / "inf.pkl"
    _write_pkl(sim_p, 0, [0, 1, 2], [0.1, 0.5, 0.9])
    _write_pkl(inf_p, 0, [0, 1, 2], [0.9, 0.5, 0.1])
    st = ScoreTable.load(str(sim_p), str(inf_p), pool_size=3)
    assert st.query_indices == [0]
    qs = st.by_query[0]
    assert qs.train_indices == (0, 1, 2)
    # alpha=0.5 -> all relevance 0.5; tie-break asc
    assert qs.ranked(0.5) == [0, 1, 2]


def test_scoretable_misaligned_train_indices_fails(tmp_path):
    sim_p, inf_p = tmp_path / "s.pkl", tmp_path / "i.pkl"
    _write_pkl(sim_p, 0, [0, 1, 2], [0.1, 0.2, 0.3])
    _write_pkl(inf_p, 0, [0, 2, 1], [0.1, 0.2, 0.3])  # reordered
    with pytest.raises(ValueError, match="misaligned"):
        ScoreTable.load(str(sim_p), str(inf_p), pool_size=3)


def test_scoretable_out_of_range_fails(tmp_path):
    sim_p, inf_p = tmp_path / "s.pkl", tmp_path / "i.pkl"
    _write_pkl(sim_p, 0, [0, 1], [0.1, 1.5])  # > 1
    _write_pkl(inf_p, 0, [0, 1], [0.1, 0.2])
    with pytest.raises(ValueError, match="out of"):
        ScoreTable.load(str(sim_p), str(inf_p), pool_size=2)


def test_scoretable_query_set_mismatch_fails(tmp_path):
    sim_p, inf_p = tmp_path / "s.pkl", tmp_path / "i.pkl"
    _write_pkl(sim_p, 0, [0, 1], [0.1, 0.2])
    _write_pkl(inf_p, 1, [0, 1], [0.1, 0.2])
    with pytest.raises(ValueError, match="query_index sets differ"):
        ScoreTable.load(str(sim_p), str(inf_p), pool_size=2)
