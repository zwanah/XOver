"""Diagnostic step D4 — aggregate executed pools into the report + GATE read.

Reads ``data/diag/executed.json``, runs the pure ``src/diag/analyze.summarize``,
and writes a human-readable ``data/diag/REPORT.md`` plus a JSON summary. The
headline is read on **S3** (gold non-empty, competing denotations) — the wedge.

Usage:  python scripts/diag_analyze.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pipeline.c3_selection.analyze import summarize  # noqa: E402

SELECTORS = ["pass1_greedy", "random", "majority_vote", "nonempty_vote", "oracle_passk"]


def _fmt_row(name: str, tbl: dict) -> str:
    cells = " | ".join(f"{tbl[s]:.3f}" for s in SELECTORS)
    return f"| {name} | {cells} |"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/diag/executed.json")
    p.add_argument("--out_md", default="data/diag/REPORT.md")
    p.add_argument("--out_json", default="data/diag/summary.json")
    opts = p.parse_args()

    with open(opts.input, encoding="utf-8") as f:
        data = json.load(f)
    s = summarize(data["records"])

    with open(opts.out_json, "w", encoding="utf-8") as f:
        json.dump(s, f, ensure_ascii=False, indent=2)

    sizes = s["strata_sizes"]
    div = s["diversity"]
    lines = [
        "# Candidate-Selection Diagnostic — Pass@K + denotation-voting",
        "",
        f"Generator/run args: `{json.dumps(data.get('args', {}))}`",
        "",
        "## Strata",
        "S0 gold-error (dropped) · S1 gold-empty (reported, **not** gated) · "
        "S2 gold-nonempty single denotation · **S3 gold-nonempty competing denotations = the gate**",
        "",
        f"- total queries: **{s['n_total']}**, usable (non-S0): **{s['usable_denominator']}**",
        f"- S0={sizes['S0']}  S1={sizes['S1']}  S2={sizes['S2']}  **S3={sizes['S3']}**",
        "",
        "## Selector accuracy (mean EX over stratum)",
        "",
        "| stratum | " + " | ".join(SELECTORS) + " |",
        "|" + "---|" * (len(SELECTORS) + 1),
        _fmt_row("**S3 (GATE)**", s["selectors"]["S3"]),
        _fmt_row("S2∪S3", s["selectors"]["S2_S3"]),
        _fmt_row("S1 (excl.)", s["selectors"]["S1"]),
        "",
        "## Diversity (candidate denotation classes per pool)",
        f"- S3: mean_classes={div['S3']['mean_classes']:.2f}, "
        f"mean_votable={div['S3']['mean_votable']:.2f}",
        f"- S2∪S3: mean_classes={div['S2_S3']['mean_classes']:.2f}",
        f"- all non-S0: mean_classes={div['all_nonS0']['mean_classes']:.2f}",
        "",
        "## How to read the gate",
        "- **oracle_passk ≫ pass1_greedy** on S3 ⇒ real headroom; a selector/verifier can help.",
        "- **majority_vote (and nonempty_vote) ≪ oracle_passk** on S3 ⇒ execution-denotation "
        "voting cannot capture the headroom — the wedge for a *learned* verifier.",
        "- If S3 is tiny or mean_classes≈1, low oracle is **inconclusive** (low diversity / "
        "generation artifact), not 'module dead'.",
    ]
    md = "\n".join(lines) + "\n"
    with open(opts.out_md, "w", encoding="utf-8") as f:
        f.write(md)
    print(md)


if __name__ == "__main__":
    main()
