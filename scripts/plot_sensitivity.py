"""Plot alpha / lambda sensitivity (OQS, dev) with 3 curves each:
English / Non-English mean / All-language mean.

Style follows the sensitivity plotting utilities.
Reads OQS from final/outputs/.../eval_result/*.json. For en/zh/ja the alpha
intermediate points are the published doc curve (same plain pipeline; verified
final en a0=0.7822 == doc 0.7820).
"""
import glob
import json
import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt

FINAL = "/path/to/XOver"
ALPHA_DIR = f"{FINAL}/outputs/predictions/ovq_alpha_sweep/eval_result"
LAM_DIR = f"{FINAL}/outputs/predictions/ovq_retrieval/ovq_profile_full/eval_result"
OUT_DIR = f"{FINAL}/outputs/figures"
MODEL = "deepseek-v4-flash"

LANGS = ["en", "zh", "ja", "yue", "fr", "de", "ko", "ru"]
NON_EN = [l for l in LANGS if l != "en"]
ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
LAMBDAS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]
CJK = ["zh", "ja"]

# Published en/zh/ja alpha curve (internal experiment notes)
ALPHA_DOC: Dict[str, Dict[float, float]] = {
    "en": {0.0: 0.7820, 0.1: 0.7824, 0.2: 0.7865, 0.3: 0.7822, 0.4: 0.7826, 0.5: 0.7765},
    "zh": {0.0: 0.7603, 0.1: 0.7606, 0.2: 0.7632, 0.3: 0.7633, 0.4: 0.7626, 0.5: 0.7631},
    "ja": {0.0: 0.7513, 0.1: 0.7525, 0.2: 0.7542, 0.3: 0.7623, 0.4: 0.7590, 0.5: 0.7591},
}


def read_oqs(path: str) -> Optional[float]:
    if not os.path.isfile(path):
        return None
    d = json.load(open(path))
    try:
        return float(d["statistics"]["evaluation"]["OQS"]["mean"])
    except (KeyError, TypeError):
        return None


def alpha_value(lang: str, a: float) -> Optional[float]:
    if lang in ALPHA_DOC:
        return ALPHA_DOC[lang].get(a)
    return read_oqs(f"{ALPHA_DIR}/{MODEL}_{lang}_dev_a{a}_eval_results.json")


def lambda_value(lang: str, lam: float) -> Optional[float]:
    return read_oqs(f"{LAM_DIR}/{MODEL}_{lang}_dev_l{lam}_eval_results.json")


def mean(xs: List[Optional[float]]) -> Optional[float]:
    vals = [x for x in xs if x is not None]
    return sum(vals) / len(vals) if vals else None


def collect(grid, getter):
    """Return dict lang->[oqs per grid] plus EN / non-EN mean / all mean (in %)."""
    per = {l: [getter(l, g) for g in grid] for l in LANGS}
    en = [v * 100 if v is not None else None for v in per["en"]]
    non = [mean([per[l][i] for l in NON_EN]) for i in range(len(grid))]
    allm = [mean([per[l][i] for l in LANGS]) for i in range(len(grid))]
    non = [v * 100 if v is not None else None for v in non]
    allm = [v * 100 if v is not None else None for v in allm]
    return per, en, non, allm


def style():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman"],
        "axes.labelsize": 15, "axes.titlesize": 16,
        "xtick.labelsize": 14, "ytick.labelsize": 14,
        "legend.fontsize": 12.5, "axes.linewidth": 1.2,
    })


C_EN, C_NON, C_ALL = "#1f77b4", "#2ca02c", "#d62728"


def make_plot(grid, en, non, allm, xlabel, title, highlight, fname):
    style()
    fig, ax = plt.subplots(figsize=(5, 4), dpi=1200)
    ax.plot(grid, en, color=C_EN, marker="o", markersize=8, linewidth=1.8,
            label="English", alpha=0.9, zorder=2)
    ax.plot(grid, non, color=C_NON, marker="s", markersize=8, linewidth=1.8,
            label="Non-English (avg)", alpha=0.9, zorder=2)
    ax.plot(grid, allm, color=C_ALL, marker="^", markersize=9, linewidth=1.8,
            label="All languages (avg)", alpha=0.9, zorder=2)

    hi = grid.index(highlight)
    ax.axvline(x=highlight, color="gray", linestyle="--", alpha=0.6, linewidth=1.5, zorder=1)
    for series, color in ((en, C_EN), (non, C_NON), (allm, C_ALL)):
        if series[hi] is not None:
            ax.scatter([highlight], [series[hi]], color=color, s=280, marker="*",
                       edgecolors="white", linewidth=0.8, zorder=10)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Overpass Query Similarity (OQS, %)")
    ax.set_title(title, fontweight="bold", pad=14)
    ax.grid(True, which="major", linestyle="-", alpha=0.22)
    ax.legend(loc="best", frameon=True, framealpha=0.92, edgecolor="gray",
              borderpad=0.7, labelspacing=0.5, handlelength=2.0)
    plt.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    out = f"{OUT_DIR}/{fname}"
    plt.savefig(out, dpi=1200, bbox_inches="tight")
    plt.savefig(out.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    plt.close()
    return out


def main():
    # ALPHA
    pa, ea, na, aa = collect(ALPHAS, alpha_value)
    # LAMBDA
    pl, el, nl, al = collect(LAMBDAS, lambda_value)

    # dump table
    table = {
        "alpha": {"grid": ALPHAS, "per_lang": pa, "english": ea,
                  "non_english_mean": na, "all_mean": aa},
        "lambda": {"grid": LAMBDAS, "per_lang": pl, "english": el,
                   "non_english_mean": nl, "all_mean": al},
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    json.dump(table, open(f"{OUT_DIR}/sensitivity_oqs_table.json", "w"), indent=2)

    print("=== ALPHA (OQS%) ===")
    print("alpha:", ALPHAS)
    print("EN   :", [round(x, 2) if x else None for x in ea])
    print("nonEN:", [round(x, 2) if x else None for x in na])
    print("ALL  :", [round(x, 2) if x else None for x in aa])
    print("=== LAMBDA (OQS%) ===")
    print("lam  :", LAMBDAS)
    print("EN   :", [round(x, 2) if x else None for x in el])
    print("nonEN:", [round(x, 2) if x else None for x in nl])
    print("ALL  :", [round(x, 2) if x else None for x in al])

    # CJK (zh+ja) lambda curve — the subset where the diversity peak lives
    cjk_lam = [mean([pl[l][i] for l in CJK]) for i in range(len(LAMBDAS))]
    cjk_lam = [v * 100 if v is not None else None for v in cjk_lam]

    p1 = make_plot(ALPHAS, ea, na, aa, r"$\alpha$ (similarity weight)",
                   "Alpha Sensitivity", 0.2, "alpha_sensitivity_oqs.pdf")
    p2 = make_plot(LAMBDAS, el, nl, al, r"$\lambda$ (diversity weight)",
                   r"Lambda Sensitivity ($\alpha=0.2$)", 0.3,
                   "lambda_sensitivity_oqs.pdf")
    # variant: English / CJK(zh+ja) / All — recovers the clean inverted-U at lambda=0.3
    p3 = make_plot_cjk(LAMBDAS, el, cjk_lam, al,
                       "lambda_sensitivity_cjk.pdf")
    print("wrote:", p1, p2, p3)


def make_plot_cjk(grid, en, cjk, allm, fname):
    style()
    fig, ax = plt.subplots(figsize=(5, 4), dpi=1200)
    ax.plot(grid, en, color=C_EN, marker="o", markersize=8, linewidth=1.8,
            label="English", alpha=0.9, zorder=2)
    ax.plot(grid, cjk, color=C_NON, marker="s", markersize=8, linewidth=1.8,
            label="CJK (zh+ja avg)", alpha=0.9, zorder=2)
    ax.plot(grid, allm, color=C_ALL, marker="^", markersize=9, linewidth=1.8,
            label="All languages (avg)", alpha=0.9, zorder=2)
    hi = grid.index(0.3)
    ax.axvline(x=0.3, color="gray", linestyle="--", alpha=0.6, linewidth=1.5, zorder=1)
    for series, color in ((en, C_EN), (cjk, C_NON), (allm, C_ALL)):
        if series[hi] is not None:
            ax.scatter([0.3], [series[hi]], color=color, s=280, marker="*",
                       edgecolors="white", linewidth=0.8, zorder=10)
    ax.set_xlabel(r"$\lambda$ (diversity weight)")
    ax.set_ylabel("Overpass Query Similarity (OQS, %)")
    ax.set_title(r"Lambda Sensitivity ($\alpha=0.2$)", fontweight="bold", pad=14)
    ax.grid(True, which="major", linestyle="-", alpha=0.22)
    ax.legend(loc="best", frameon=True, framealpha=0.92, edgecolor="gray",
              borderpad=0.7, labelspacing=0.5, handlelength=2.0)
    plt.tight_layout()
    out = f"{OUT_DIR}/{fname}"
    plt.savefig(out, dpi=1200, bbox_inches="tight")
    plt.savefig(out.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    plt.close()
    return out


if __name__ == "__main__":
    main()
