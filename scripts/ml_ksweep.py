"""Multilingual C3 K-sweep (K=1..8) over the dev candidate pools.

For each language and each K in 1..8, compute the exact expectation of
"draw K of the 8 temp-0.8 samples, then apply the selector" by averaging the
selector's EX over all C(8,K) subsets of a query's pool, then averaging across
queries. greedy (Pass@1) is K-independent and shown as the shippable baseline.

Primary selector is ``nonempty_vote`` (C3 denotation self-consistency). We also
report oracle (Pass@K ceiling), majority_vote, and random for context, plus the
per-K marginal gain and the per-language / global argmax K for picking the test
operating point.
"""
import argparse
import json
import os
import sys
from itertools import combinations

sys.path.insert(0, "/path/to/XOver")
from pipeline.c3_selection import selectors as S
from pipeline.c3_selection.denotation import freeze

LANGS = ["en", "zh", "yue", "fr", "de", "ja", "ko", "ru"]
KS = list(range(1, 9))
SEL = {
    "nonempty_vote": S.nonempty_vote,
    "majority_vote": S.majority_vote,
    "oracle": S.oracle_passk,
    "random": S.random_expected,
}
# split is set in main(); BASE is templated on it.
BASE = "/path/to/XOver/data/diag_ml_{split}/{lang}/executed_lam0.3_k8.json"
SPLIT = "dev"


def load(lang):
    d = json.load(open(BASE.format(split=SPLIT, lang=lang)))
    recs = d["records"]
    for r in recs:
        for c in r["candidates"]:
            c["sig"] = freeze(c["sig"])
    return recs


def exp_over_subsets(cands, k, fn):
    subs = list(combinations(range(len(cands)), k))
    return sum(fn([cands[i] for i in sub]) for sub in subs) / len(subs)


def sweep_lang(recs):
    n = len(recs)
    greedy = sum(int(r["greedy_ex"]) for r in recs) / n
    rows = {"greedy": greedy}
    for name, fn in SEL.items():
        vals = []
        for k in KS:
            kk = min(k, 8)
            acc = sum(exp_over_subsets(r["candidates"], kk, fn) for r in recs) / n
            vals.append(acc)
        rows[name] = vals
    return n, rows


def main():
    global SPLIT
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    SPLIT = ap.parse_args().split
    all_rows = {}
    print(f"{'lang':<5}{'n':<6}{'sel':<14}" + "".join(f"K={k:<6}" for k in KS) + "greedy")
    print("-" * 100)
    for lang in LANGS:
        n, rows = sweep_lang(load(lang))
        all_rows[lang] = rows
        g = rows["greedy"]
        for name in ["nonempty_vote", "majority_vote", "oracle"]:
            v = rows[name]
            print(f"{lang:<5}{n:<6}{name:<14}" + "".join(f"{x*100:<8.1f}" for x in v) +
                  f"{g*100:.1f}")
        print()

    # ---- mean across the 8 languages (per K), primary selector ----
    print("=" * 100)
    print("MEAN across 8 langs  (nonempty_vote, the C3 selector)")
    print(f"{'':<5}{'':<6}{'sel':<14}" + "".join(f"K={k:<6}" for k in KS) + "greedy")
    mean_nv = [sum(all_rows[l]["nonempty_vote"][i] for l in LANGS) / len(LANGS)
               for i in range(len(KS))]
    mean_g = sum(all_rows[l]["greedy"] for l in LANGS) / len(LANGS)
    mean_orc = [sum(all_rows[l]["oracle"][i] for l in LANGS) / len(LANGS)
                for i in range(len(KS))]
    print(f"{'MEAN':<5}{'':<6}{'nonempty_vote':<14}" +
          "".join(f"{x*100:<8.1f}" for x in mean_nv) + f"{mean_g*100:.1f}")
    print(f"{'MEAN':<5}{'':<6}{'oracle':<14}" +
          "".join(f"{x*100:<8.1f}" for x in mean_orc) + f"{mean_g*100:.1f}")

    print("\nmean nonempty_vote marginal gain (pts) vs previous K:")
    for i in range(1, len(KS)):
        print(f"  K={KS[i-1]}->{KS[i]}: {(mean_nv[i]-mean_nv[i-1])*100:+.2f}"
              f"   (vs greedy: {(mean_nv[i]-mean_g)*100:+.2f})")
    print(f"  K=1 vs greedy: {(mean_nv[0]-mean_g)*100:+.2f}")

    # ---- argmax K ----
    print("\nper-language argmax K (nonempty_vote) and value:")
    for lang in LANGS:
        v = all_rows[lang]["nonempty_vote"]
        bk = max(range(len(KS)), key=lambda i: v[i])
        # knee: smallest K within 0.5pt of the max
        knee = next(i for i in range(len(KS)) if v[i] >= v[bk] - 0.005)
        print(f"  {lang}: argmax K={KS[bk]} ({v[bk]*100:.1f})  "
              f"knee(within 0.5pt) K={KS[knee]} ({v[knee]*100:.1f})  "
              f"greedy {all_rows[lang]['greedy']*100:.1f}")
    bk = max(range(len(KS)), key=lambda i: mean_nv[i])
    knee = next(i for i in range(len(KS)) if mean_nv[i] >= mean_nv[bk] - 0.005)
    print(f"  GLOBAL(mean): argmax K={KS[bk]} ({mean_nv[bk]*100:.1f})  "
          f"knee K={KS[knee]} ({mean_nv[knee]*100:.1f})  greedy {mean_g*100:.1f}")

    # dump machine-readable
    out = {"langs": LANGS, "KS": KS,
           "per_lang": {l: {k: all_rows[l][k] if k == "greedy" else all_rows[l][k]
                            for k in ["greedy", "nonempty_vote", "majority_vote",
                                      "oracle", "random"]} for l in LANGS},
           "mean_nonempty_vote": mean_nv, "mean_oracle": mean_orc,
           "mean_greedy": mean_g}
    path = f"/path/to/XOver/data/diag_ml_{SPLIT}/ksweep_K1to8.json"
    json.dump(out, open(path, "w"), indent=2)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
