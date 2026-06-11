"""K-shot line figure: OQS vs number of shots (K=0..6), all 8 MOsmNL languages.

Generator: deepseek-v4-flash NON-THINKING, plain demos, temperature 0, dev split.
Demos selected by cross-lingual relevance-profile DPP (alpha=0.3, lambda=0.5, M=20);
K=1..5 are the prefix of the K=6 greedy selection (prefix-consistent). K=0 anchor =
the W1 baseline 0-shot run (deepseek-v4-flash default config). OQS only.

Visual style mirrors ``eval_backend/.../make_pareto_deepseek_v4_flash.py``.

Usage (from final/):  python scripts/plot_kshot.py
"""
from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/path/to/XOver")
FINAL = ROOT / "final"
SWEEP = FINAL / "outputs/predictions/ovq_kshot/eval_result"
W1 = ROOT / "outputs/predictions/W1_baseline/deepseek-v4-flash/eval_result"
OUT = FINAL / "outputs/predictions/ovq_kshot/kshot_oqs_deepseek_v4_flash.pdf"

KS = [0, 1, 2, 3, 4, 5, 6]
# short -> (full dataset token, display label, color, marker)
LANGS = [
    ("en",  "english",             "English",   "#1f77b4", "o"),
    ("zh",  "mandarin_simplified", "Mandarin",  "#d62728", "s"),
    ("yue", "cantonese",           "Cantonese", "#9467bd", "P"),
    ("fr",  "french",              "French",    "#2ca02c", "^"),
    ("de",  "german",              "German",    "#8c564b", "v"),
    ("ja",  "japanese",            "Japanese",  "#e377c2", "D"),
    ("ko",  "korean",              "Korean",    "#ff7f0e", "X"),
    ("ru",  "russian",             "Russian",   "#17becf", "<"),
]


def oqs_mean(path: Path) -> float:
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    return float(d["statistics"]["evaluation"]["OQS"]["mean"]) * 100.0


def oqs_for(short: str, full: str, k: int) -> float:
    if k == 0:
        p = W1 / f"{full}_dev_n1000_0shot_eval_results.json"
    else:
        p = SWEEP / f"deepseek-v4-flash_{short}_dev_k{k}_a0.3_lam0.5_eval_results.json"
    if not p.exists():
        print(f"  WARN missing {short} k={k}: {p}")
        return np.nan
    return oqs_mean(p)


plt.rcParams.update({
    "font.family": "serif",
    "font.size": 15,
    "axes.titlesize": 18,
    "axes.labelsize": 17,
    "legend.fontsize": 12,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "axes.linewidth": 1.2,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

fig, ax = plt.subplots(figsize=(8.2, 5.6))

for short, full, label, color, marker in LANGS:
    ys = [oqs_for(short, full, k) for k in KS]
    print(f"{label:<10} " + " ".join(f"{v:5.2f}" for v in ys))
    ax.plot(KS, ys, ls="-", lw=1.8, marker=marker, markersize=8,
            color=color, alpha=0.9, label=label)

ax.set_xlabel(r"Number of shots $K$")
ax.set_ylabel(r"OQS $\uparrow$")
ax.set_xticks(KS)
ax.grid(True, which="both", alpha=0.25, ls=":")
ax.set_title("MOsmNL dev — DeepSeek-V4-Flash, profile-DPP demos "
             r"($\alpha{=}0.3,\ \lambda{=}0.5$)", pad=10, fontsize=15)
ax.legend(loc="lower right", frameon=False, ncol=2)

fig.tight_layout()
fig.savefig(OUT, bbox_inches="tight", dpi=1200)
fig.savefig(OUT.with_suffix(".png"), dpi=200, bbox_inches="tight")
print(f"Saved: {OUT}")
print(f"Saved: {OUT.with_suffix('.png')}")
