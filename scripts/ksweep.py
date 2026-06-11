"""K-sweep over the real P1 candidate pool (dev/english n=1000, 8 sampled @ temp0.8).

For each K in {1,2,4,8}, average each selector's EX over ALL C(8,K) subsets of a
query's 8 candidates, then average across queries — the exact expectation of
"draw K samples, then apply selector". greedy (Pass@1) is K-independent.
"""
import json
import sys
from itertools import combinations

sys.path.insert(0, "/path/to/XOver")
from pipeline.c3_selection import selectors as S
from pipeline.c3_selection.strata import classify
from pipeline.c3_selection.denotation import freeze

d = json.load(open("/path/to/XOver/data/diag_p1_en/executed.json"))
recs = d["records"]
# JSON loads signatures as lists; selectors group on them, so re-freeze to tuples.
for r in recs:
    for c in r["candidates"]:
        c["sig"] = freeze(c["sig"])
N = len(recs)

KS = [1, 2, 4, 8]
SEL = {
    "oracle": S.oracle_passk,
    "nonempty_vote": S.nonempty_vote,
    "majority_vote": S.majority_vote,
    "random": S.random_expected,
}


def exp_over_subsets(cands, k, fn):
    """Mean of fn over all C(len,k) subsets."""
    idx = range(len(cands))
    subs = list(combinations(idx, k))
    tot = 0.0
    for sub in subs:
        sublist = [cands[i] for i in sub]
        tot += fn(sublist)
    return tot / len(subs)


def run(records, label):
    n = len(records)
    greedy = sum(int(r["greedy_ex"]) for r in records) / n
    print(f"\n=== {label}  (n={n}) ===")
    print(f"{'selector':<16}" + "".join(f"K={k:<7}" for k in KS))
    print(f"{'greedy(Pass@1)':<16}" + f"{greedy*100:<8.1f}" * len(KS) +
          "  <- K-independent")
    rows = {}
    for name, fn in SEL.items():
        vals = []
        for k in KS:
            acc = sum(exp_over_subsets(r["candidates"], min(k, len(r["candidates"])), fn)
                      for r in records) / n
            vals.append(acc)
        rows[name] = vals
        print(f"{name:<16}" + "".join(f"{v*100:<8.1f}" for v in vals))
    # marginal gains for nonempty_vote
    nv = rows["nonempty_vote"]
    print("\nnonempty_vote marginal (pts):")
    for i in range(1, len(KS)):
        print(f"  K={KS[i-1]}->{KS[i]}: {(nv[i]-nv[i-1])*100:+.2f}")
    # voting-capture of oracle wedge at each K
    orc = rows["oracle"]
    print("voting-capture = (nonempty_vote - greedy)/(oracle - greedy):")
    for i, k in enumerate(KS):
        denom = orc[i] - greedy
        cap = (nv[i] - greedy) / denom if denom > 1e-9 else float("nan")
        print(f"  K={k}: vote+{(nv[i]-greedy)*100:+.2f}  oracle+{(orc[i]-greedy)*100:+.2f}  capture={cap*100:.0f}%")
    return rows


# Headline: all 1000
run(recs, "ALL (1000)")

# Stratify. classify needs candidate sigs; use the full 8-pool to assign stratum
# (stratum is a property of the query, computed once on the full pool).
strat = {}
for r in recs:
    sigs = [c["sig"] for c in r["candidates"]]
    s = classify(r["gold_num"], r["gold_has_error"], sigs)
    strat.setdefault(s, []).append(r)

print("\n\n#### strata sizes:", {k: len(v) for k, v in sorted(strat.items())})
for s in ["S3", "S2", "S1", "S0"]:
    if s in strat and strat[s]:
        run(strat[s], s)
