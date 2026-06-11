"""Unit tests for the candidate-selection diagnostic core (``src/diag/``).

Pure logic only — no execution. Run: ``pytest tests/test_diag.py``.
"""
from __future__ import annotations

from pipeline.c3_selection import analyze, selectors
from pipeline.c3_selection.denotation import (
    EMPTY,
    ERROR,
    freeze,
    is_empty_sig,
    is_error_sig,
    signature,
    to_jsonable,
)
from pipeline.c3_selection.strata import S0, S1, S2, S3, classify, n_denotation_classes


# --- denotation signatures --------------------------------------------------
def test_signature_error_empty_count_set():
    assert signature("Some error", -1)[0] == "error"
    assert signature([], 0) == EMPTY
    assert signature("count_42", 42) == ("scalar", "count_42")
    sig = signature(["n/1", "w/2"], 2)
    assert sig[0] == "set" and isinstance(sig[1], str)  # hashed, not raw list


def test_signature_equality_matches_list_equality():
    # Same contents (and order) ⇒ same signature, mirroring EX's `==`.
    a = signature(["n/1", "n/2"], 2)
    b = signature(["n/1", "n/2"], 2)
    c = signature(["n/2", "n/1"], 2)  # different order ⇒ different signature
    assert a == b
    assert a != c
    assert hash(a) == hash(b)  # hashable for grouping


def test_freeze_roundtrip_is_hashable():
    sig = signature(["n/1", "w/2"], 2)
    restored = freeze(to_jsonable(sig))
    assert restored == sig
    assert hash(restored) == hash(sig)
    {restored}  # would raise if unhashable


def test_sig_predicates():
    assert is_empty_sig(EMPTY)
    assert not is_empty_sig(("set", ("n/1",)))
    assert is_error_sig(ERROR)
    assert is_error_sig(signature("boom", -1))
    assert not is_error_sig(EMPTY)


# --- strata -----------------------------------------------------------------
def test_n_denotation_classes_excludes_errors():
    sigs = [signature(["a"], 1), signature(["a"], 1),
            signature("err", -1), EMPTY]
    # {("set",("a",)), EMPTY} ⇒ 2 (error not counted)
    assert n_denotation_classes(sigs) == 2


def test_classify_s0_gold_error():
    assert classify(gold_num=-1, gold_has_error=True, candidate_sigs=[]) == S0
    assert classify(gold_num=-1, gold_has_error=False, candidate_sigs=[]) == S0


def test_classify_s1_gold_empty():
    assert classify(0, False, [signature(["a"], 1)]) == S1


def test_classify_s2_single_class():
    sigs = [signature(["a"], 1), signature(["a"], 1)]
    assert classify(5, False, sigs) == S2


def test_classify_s3_competing_classes():
    sigs = [signature(["a"], 1), signature(["b"], 1), EMPTY]
    assert classify(5, False, sigs) == S3


def test_classify_s2_when_only_errors_plus_one():
    # gold non-empty, candidates: one real set + errors ⇒ 1 class ⇒ S2.
    sigs = [signature(["a"], 1), signature("err", -1), signature("err2", -1)]
    assert classify(5, False, sigs) == S2


# --- selectors --------------------------------------------------------------
def _c(sig, ex):
    return {"sig": sig, "ex": int(ex)}


def test_oracle_passk():
    assert selectors.oracle_passk([_c(EMPTY, 0), _c(("set", ("a",)), 1)]) == 1
    assert selectors.oracle_passk([_c(EMPTY, 0), _c(("set", ("a",)), 0)]) == 0


def test_random_expected_is_mean_ex():
    cands = [_c(("set", ("a",)), 1), _c(("set", ("b",)), 0),
             _c(EMPTY, 0), _c(("set", ("c",)), 1)]
    assert selectors.random_expected(cands) == 0.5


def test_majority_vote_picks_largest_group():
    # wrong denotation has 3 votes, correct has 2 ⇒ voting picks WRONG (ex=0).
    wrong = ("set", ("x",))
    right = ("set", ("gold",))
    cands = [_c(wrong, 0), _c(wrong, 0), _c(wrong, 0),
             _c(right, 1), _c(right, 1)]
    assert selectors.majority_vote(cands) == 0
    assert selectors.oracle_passk(cands) == 1  # the wedge: oracle would get it


def test_majority_vote_tiebreak_prefers_nonempty():
    # 2 empty vs 2 non-empty (correct). Tie broken toward non-empty ⇒ ex=1.
    right = ("set", ("gold",))
    cands = [_c(EMPTY, 0), _c(EMPTY, 0), _c(right, 1), _c(right, 1)]
    assert selectors.majority_vote(cands) == 1


def test_nonempty_vote_ignores_empty_majority():
    # empties dominate (3) but nonempty_vote restricts to non-empty groups.
    right = ("set", ("gold",))
    cands = [_c(EMPTY, 0), _c(EMPTY, 0), _c(EMPTY, 0), _c(right, 1)]
    assert selectors.majority_vote(cands) == 0      # plain vote ⇒ empty wins
    assert selectors.nonempty_vote(cands) == 1      # nonempty pref ⇒ correct


def test_vote_handles_all_error_pool():
    cands = [_c(signature("e", -1), 0), _c(signature("e2", -1), 0)]
    assert selectors.majority_vote(cands) == 0
    assert selectors.nonempty_vote(cands) == 0


# --- analyze ----------------------------------------------------------------
def _rec(qi, gold_num, greedy_ex, cand_pairs, gold_err=False):
    return {
        "query_index": qi,
        "gold_num": gold_num,
        "gold_has_error": gold_err,
        "greedy_ex": greedy_ex,
        "candidates": [{"sig": to_jsonable(s), "ex": e} for s, e in cand_pairs],
    }


def test_summarize_strata_sizes_and_gate():
    wrong, right = signature(["x"], 1), signature(["gold"], 1)
    records = [
        # S0: gold error
        _rec(0, -1, 0, [(EMPTY, 0)], gold_err=True),
        # S1: gold empty
        _rec(1, 0, 1, [(EMPTY, 1), (EMPTY, 1)]),
        # S2: gold non-empty, one denotation class
        _rec(2, 3, 1, [(right, 1), (right, 1)]),
        # S3: gold non-empty, competing classes; voting wrong, oracle right
        _rec(3, 3, 0, [(wrong, 0), (wrong, 0), (wrong, 0), (right, 1), (right, 1)]),
    ]
    s = analyze.summarize(records)
    assert s["strata_sizes"] == {"S0": 1, "S1": 1, "S2": 1, "S3": 1}
    assert s["usable_denominator"] == 3
    # On S3 the wedge shows: majority vote misses, oracle hits.
    s3 = s["selectors"]["S3"]
    assert s3["majority_vote"] == 0.0
    assert s3["oracle_passk"] == 1.0
    assert s3["pass1_greedy"] == 0.0
    # S1 is reported separately, not folded into the gate.
    assert "S1" in s["selectors"]
    assert s["diversity"]["S3"]["mean_classes"] == 2.0  # {wrong, right}
