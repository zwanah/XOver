"""P1 glue: turn a C3 selector's pick into a runner-contract prediction file.

Reads the diagnostic's executed.json (per-candidate denotation sig + ex) and
candidates.json (the raw generated texts), applies one selector per query to
choose a candidate, and emits a prediction JSON
``[{query_index, query_target_ovq, generated_text}]`` that
``pipeline.c1_eval.runner`` (and ``pipeline.c4_revision.run``) consume.

Selectors mirror ``pipeline.c3_selection.selectors`` but return the chosen
candidate's TEXT (not just its EX), so the same vote that scores in the report
drives the prediction file fed downstream to C4 revision + EX eval.

Usage (from final/):
    python scripts/p1_select_to_predfile.py \
        --executed data/diag_p1_en/executed.json \
        --candidates data/diag_p1_en/candidates.json \
        --selector nonempty_vote \
        --output data/diag_p1_en/pred_nonempty_vote.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Set

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.c3_selection.denotation import (  # noqa: E402
    freeze, is_empty_sig, is_error_sig,
)


def _pick_index(cands: List[Dict], votable_idx: Set[int]) -> int:
    """Index of the largest denotation group among ``votable_idx`` (else -1).

    Same deterministic tie-break as ``selectors._group_and_pick`` (larger group,
    then non-empty over empty, then earliest pool position), but returns the
    representative candidate index instead of its ex.
    """
    if not votable_idx:
        return -1
    groups: Dict[tuple, Dict] = {}
    for i, c in enumerate(cands):
        if i not in votable_idx:
            continue
        sig = c["sig"]
        g = groups.get(sig)
        if g is None:
            groups[sig] = {"count": 1, "first": i, "empty": is_empty_sig(sig)}
        else:
            g["count"] += 1
    best = min(groups.values(), key=lambda g: (-g["count"], g["empty"], g["first"]))
    return int(best["first"])


def pick(cands: List[Dict], greedy_idx: int, selector: str) -> int:
    """Return the chosen candidate index for the given selector.

    ``greedy`` is special: it is the temp-0 decode, stored separately from the
    K-pool, so the caller maps -1 -> the greedy text.
    """
    if selector == "greedy":
        return -1  # sentinel: use the greedy text
    if selector == "majority_vote":
        votable = {i for i, c in enumerate(cands) if not is_error_sig(c["sig"])}
        return _pick_index(cands, votable)
    if selector == "nonempty_vote":
        nonempty = {i for i, c in enumerate(cands)
                    if not is_error_sig(c["sig"]) and not is_empty_sig(c["sig"])}
        if nonempty:
            return _pick_index(cands, nonempty)
        votable = {i for i, c in enumerate(cands) if not is_error_sig(c["sig"])}
        return _pick_index(cands, votable)
    raise ValueError(f"unknown selector {selector!r} (greedy|majority_vote|nonempty_vote)")


def main() -> None:
    p = argparse.ArgumentParser(description="C3 selector pick -> prediction file")
    p.add_argument("--executed", required=True)
    p.add_argument("--candidates", required=True)
    p.add_argument("--selector", default="nonempty_vote",
                   choices=["greedy", "majority_vote", "nonempty_vote"])
    p.add_argument("--output", required=True)
    opts = p.parse_args()

    executed = json.load(open(opts.executed))["records"]
    cand_raw = json.load(open(opts.candidates))
    cand_recs = cand_raw["records"] if isinstance(cand_raw, dict) and "records" in cand_raw else cand_raw
    texts_by_qi = {r["query_index"]: r for r in cand_recs}

    out = []
    n_fallback_greedy = 0
    for rec in executed:
        qi = rec["query_index"]
        ctext = texts_by_qi[qi]
        cands = [{"sig": freeze(c["sig"]), "ex": int(c["ex"])} for c in rec["candidates"]]
        idx = pick(cands, greedy_idx=-1, selector=opts.selector)
        if idx < 0:
            # greedy selector, or no votable group -> fall back to greedy decode
            text = ctext["greedy"]
            if opts.selector != "greedy":
                n_fallback_greedy += 1
        else:
            text = ctext["candidates"][idx]
        out.append({
            "query_index": qi,
            "query_target_ovq": ctext["OverpassQL"],
            "generated_text": text,
            # Carry the NL through so downstream C4 geo/syntax repair prompts
            # (and any regen) have the question; the source candidate records
            # always store it.
            "query_nl": ctext.get("nl_question", "") or ctext.get("query_nl", ""),
        })

    with open(opts.output, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"wrote {opts.output}  n={len(out)}  selector={opts.selector}  "
          f"greedy_fallback={n_fallback_greedy}")


if __name__ == "__main__":
    main()
