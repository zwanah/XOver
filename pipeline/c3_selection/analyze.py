"""Aggregate executed candidate pools into the diagnostic report.

Input = a list of per-query records (produced by ``scripts/diag_execute.py``):

    {
      "query_index": int,
      "gold_num": int,          # -1 error, 0 empty, >0 element count
      "gold_has_error": bool,
      "greedy_ex": int,         # EX of the temp-0 greedy decode (= Pass@1)
      "candidates": [           # the K temp>0 samples
          {"sig": <jsonable signature>, "ex": int}, ...
      ],
    }

Signatures arrive as JSON lists; we ``freeze`` them back to hashable tuples.
Everything here is pure — no execution, no I/O.
"""
from __future__ import annotations

from typing import Dict, List

from . import selectors
from .denotation import freeze, is_error_sig
from .strata import classify, n_denotation_classes


def _prep(rec: Dict) -> Dict:
    """Normalize a raw record: freeze signatures, attach stratum + class count."""
    cands = [{"sig": freeze(c["sig"]), "ex": int(c["ex"])}
             for c in rec["candidates"]]
    sigs = [c["sig"] for c in cands]
    stratum = classify(rec["gold_num"], bool(rec["gold_has_error"]), sigs)
    return {
        "query_index": rec["query_index"],
        "stratum": stratum,
        "greedy_ex": int(rec.get("greedy_ex", 0)),
        "candidates": cands,
        "n_classes": n_denotation_classes(sigs),
        "n_votable": sum(1 for s in sigs if not is_error_sig(s)),
    }


def _selector_table(recs: List[Dict]) -> Dict[str, float]:
    """Mean accuracy of each selector over a set of prepared records."""
    n = len(recs)
    if n == 0:
        return {k: 0.0 for k in
                ("pass1_greedy", "random", "majority_vote",
                 "nonempty_vote", "oracle_passk")}
    return {
        "pass1_greedy": sum(selectors.top1_greedy(r["greedy_ex"]) for r in recs) / n,
        "random": sum(selectors.random_expected(r["candidates"]) for r in recs) / n,
        "majority_vote": sum(selectors.majority_vote(r["candidates"]) for r in recs) / n,
        "nonempty_vote": sum(selectors.nonempty_vote(r["candidates"]) for r in recs) / n,
        "oracle_passk": sum(selectors.oracle_passk(r["candidates"]) for r in recs) / n,
    }


def _diversity(recs: List[Dict]) -> Dict[str, float]:
    if not recs:
        return {"mean_classes": 0.0, "mean_votable": 0.0}
    return {
        "mean_classes": sum(r["n_classes"] for r in recs) / len(recs),
        "mean_votable": sum(r["n_votable"] for r in recs) / len(recs),
    }


def summarize(records: List[Dict]) -> Dict:
    """Build the full diagnostic summary.

    Headline selector table is reported on **S3** (the wedge) and on **S2∪S3**;
    S1 is shown separately and excluded from the headline (empty-match makes EX
    degenerate there); S0 is the dropped/unusable count.
    """
    prepped = [_prep(r) for r in records]
    by: Dict[str, List[Dict]] = {"S0": [], "S1": [], "S2": [], "S3": []}
    for r in prepped:
        by[r["stratum"]].append(r)

    s2s3 = by["S2"] + by["S3"]
    return {
        "n_total": len(prepped),
        "strata_sizes": {k: len(v) for k, v in by.items()},
        "usable_denominator": len(prepped) - len(by["S0"]),
        "selectors": {
            "S3": _selector_table(by["S3"]),       # the gate
            "S2_S3": _selector_table(s2s3),
            "S1": _selector_table(by["S1"]),        # reported, not gated
        },
        "diversity": {
            "S3": _diversity(by["S3"]),
            "S2_S3": _diversity(s2s3),
            "all_nonS0": _diversity([r for r in prepped if r["stratum"] != "S0"]),
        },
    }
