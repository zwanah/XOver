"""Paired EX / EX_soft comparison of few-shot arms vs the plain baseline.

For the single-field ICL ablation (plain vs extract vs find) on one cell, this
reports per-arm EX / EX_soft / runtime-timeout counts and the *paired* tests vs
plain that the aggregate counts cannot give: McNemar's test on the binary EX
flips (with the b/c flip table) and a paired t-test on EX_soft.

Run from repo root after all arms have been EX-evaluated (and uniformly
``--retry_errors``-refreshed):

    python scripts/diag_ex_paired.py
"""
from __future__ import annotations

import json
import math
import os

ED = "outputs/predictions/ovq_fewshot/eval_result"
CELL = "deepseek-v4-flash_dev_k6_a0.2"
ARMS = ("plain", "extract", "find")


def _load(arm: str) -> dict:
    path = os.path.join(ED, f"{CELL}_{arm}_eval_results.json")
    data = json.load(open(path, encoding="utf-8"))["data"]
    return {r["query_index"]: r for r in data}


def _timeout_count(rows: dict) -> int:
    return sum(
        1 for r in rows.values()
        if "runtime_error_timeout" in str(r.get("execution_info", ""))
    )


def _mcnemar(plain: dict, arm: dict) -> dict:
    keys = sorted(set(plain) & set(arm))
    b = sum(1 for k in keys if plain[k]["EX"] == 1 and arm[k]["EX"] == 0)  # plain✓ arm✗
    c = sum(1 for k in keys if plain[k]["EX"] == 0 and arm[k]["EX"] == 1)  # plain✗ arm✓
    both = sum(1 for k in keys if plain[k]["EX"] == 1 and arm[k]["EX"] == 1)
    n = b + c
    # continuity-corrected McNemar chi-square (1 dof); exact-ish guard for small n
    chi2 = (abs(b - c) - 1) ** 2 / n if n > 0 else 0.0
    return {"b_hurt": b, "c_help": c, "both": both, "net": c - b, "churn": n, "chi2": chi2}


def _paired_t(plain: dict, arm: dict, field: str) -> dict:
    keys = sorted(set(plain) & set(arm))
    d = [arm[k][field] - plain[k][field] for k in keys]
    n = len(d)
    mean = sum(d) / n
    var = sum((x - mean) ** 2 for x in d) / (n - 1)
    se = math.sqrt(var / n)
    t = mean / se if se > 0 else 0.0
    return {"mean_pts": mean * 100, "t": t, "n": n}


def main() -> None:
    rows = {a: _load(a) for a in ARMS}
    plain = rows["plain"]

    print(f"cell: {CELL}  (dev/english, n=1000, deepseek-v4-flash non-thinking)\n")
    print(f"{'arm':<9} {'EX':>5} {'EX_soft':>8} {'timeouts':>9}")
    for a in ARMS:
        r = rows[a]
        ex = sum(x["EX"] for x in r.values())
        soft = sum(x["EX_soft"] for x in r.values()) / len(r)
        print(f"{a:<9} {ex:>5} {soft:>8.4f} {_timeout_count(r):>9}")

    print("\n--- paired vs plain ---")
    for a in ("extract", "find"):
        mc = _mcnemar(plain, rows[a])
        ts = _paired_t(plain, rows[a], "EX_soft")
        print(f"\n{a} vs plain:")
        print(f"  EX McNemar: HURT(plain✓{a}✗)={mc['b_hurt']} "
              f"HELP(plain✗{a}✓)={mc['c_help']} net={mc['net']:+d} "
              f"churn={mc['churn']} chi2={mc['chi2']:.2f} "
              f"({'sig' if mc['chi2'] > 3.84 else 'ns'} @0.05)")
        print(f"  EX_soft paired t: d={ts['mean_pts']:+.3f} pts  t={ts['t']:.2f} "
              f"({'sig' if abs(ts['t']) > 1.96 else 'ns'})")


if __name__ == "__main__":
    main()
