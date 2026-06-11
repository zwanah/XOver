"""6x6 alpha-lambda grid search -> three OQS heatmaps (English / Non-English / All).

Resolves each (alpha, lambda) cell from the cheapest available source:
  lambda==0          -> alpha sweep eval (en/zh/ja: published doc curve)
  alpha==0.2, l in {.1,.2,.3,.5} -> lambda sweep eval
  else               -> ovq_grid eval (this run)
Run: python final/scripts/plot_grid_heatmaps.py
"""
import json
import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

FINAL = "/path/to/XOver"
A_DIR = f"{FINAL}/outputs/predictions/ovq_alpha_sweep/eval_result"
L_DIR = f"{FINAL}/outputs/predictions/ovq_retrieval/ovq_profile_full/eval_result"
G_DIR = f"{FINAL}/outputs/predictions/ovq_grid/eval_result"
OUT = f"{FINAL}/outputs/figures"
MODEL = "deepseek-v4-flash"

LANGS = ["en", "zh", "ja", "yue", "fr", "de", "ko", "ru"]
NON_EN = [l for l in LANGS if l != "en"]
ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
LAMBDAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

ALPHA_DOC: Dict[str, Dict[float, float]] = {
    "en": {0.0: 0.7820, 0.1: 0.7824, 0.2: 0.7865, 0.3: 0.7822, 0.4: 0.7826, 0.5: 0.7765},
    "zh": {0.0: 0.7603, 0.1: 0.7606, 0.2: 0.7632, 0.3: 0.7633, 0.4: 0.7626, 0.5: 0.7631},
    "ja": {0.0: 0.7513, 0.1: 0.7525, 0.2: 0.7542, 0.3: 0.7623, 0.4: 0.7590, 0.5: 0.7591},
}


def _read(path: str) -> Optional[float]:
    if not os.path.isfile(path):
        return None
    try:
        return float(json.load(open(path))["statistics"]["evaluation"]["OQS"]["mean"])
    except (KeyError, TypeError, ValueError):
        return None


def cell(lang: str, a: float, lam: float) -> Optional[float]:
    if lam == 0.0:
        if lang in ALPHA_DOC:
            return ALPHA_DOC[lang].get(a)
        return _read(f"{A_DIR}/{MODEL}_{lang}_dev_a{a}_eval_results.json")
    if a == 0.2 and lam in (0.1, 0.2, 0.3, 0.5):
        return _read(f"{L_DIR}/{MODEL}_{lang}_dev_l{lam}_eval_results.json")
    return _read(f"{G_DIR}/{MODEL}_{lang}_dev_a{a}_l{lam}_eval_results.json")


def matrix(langs: List[str]) -> np.ndarray:
    """rows = lambda, cols = alpha; value = mean OQS% over langs (nan if any missing)."""
    M = np.full((len(LAMBDAS), len(ALPHAS)), np.nan)
    for i, lam in enumerate(LAMBDAS):
        for j, a in enumerate(ALPHAS):
            vals = [cell(l, a, lam) for l in langs]
            vals = [v for v in vals if v is not None]
            if len(vals) == len(langs):
                M[i, j] = 100.0 * sum(vals) / len(vals)
    return M


def heatmap(M: np.ndarray, title: str, fname: str):
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"]})
    fig, ax = plt.subplots(figsize=(5.2, 4.6), dpi=1200)
    im = ax.imshow(M, origin="lower", cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(ALPHAS)), [f"{a:g}" for a in ALPHAS], fontsize=12)
    ax.set_yticks(range(len(LAMBDAS)), [f"{l:g}" for l in LAMBDAS], fontsize=12)
    ax.set_xlabel(r"$\alpha$ (similarity weight)", fontsize=14)
    ax.set_ylabel(r"$\lambda$ (diversity weight)", fontsize=14)
    ax.set_title(title, fontweight="bold", fontsize=15, pad=10)
    # annotate
    finite = M[np.isfinite(M)]
    mid = (finite.max() + finite.min()) / 2 if finite.size else 0
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center",
                        color="white" if M[i, j] < mid else "black", fontsize=9)
    # star the argmax
    if finite.size:
        bi, bj = np.unravel_index(np.nanargmax(M), M.shape)
        ax.scatter([bj], [bi], s=420, marker="*", facecolors="none",
                   edgecolors="red", linewidths=2.0, zorder=5)
        ax.set_title(f"{title}\noptimum: " + r"$\alpha$=" + f"{ALPHAS[bj]:g}, "
                     + r"$\lambda$=" + f"{LAMBDAS[bi]:g} ({M[bi,bj]:.2f})",
                     fontweight="bold", fontsize=13, pad=10)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("OQS (%)", fontsize=12)
    plt.tight_layout()
    os.makedirs(OUT, exist_ok=True)
    out = f"{OUT}/{fname}"
    plt.savefig(out, dpi=1200, bbox_inches="tight")
    plt.savefig(out.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    plt.close()
    return out


def combined(mats: List[np.ndarray], titles: List[str], fname: str):
    """3 panels side by side, each with its own colour scale (EN~78 vs non-EN~76)."""
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"]})
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4), dpi=1200)
    for ax, M, title in zip(axes, mats, titles):
        im = ax.imshow(M, origin="lower", cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(ALPHAS)), [f"{a:g}" for a in ALPHAS], fontsize=11)
        ax.set_yticks(range(len(LAMBDAS)), [f"{l:g}" for l in LAMBDAS], fontsize=11)
        ax.set_xlabel(r"$\alpha$ (similarity weight)", fontsize=13)
        finite = M[np.isfinite(M)]
        mid = (finite.max() + finite.min()) / 2 if finite.size else 0
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if np.isfinite(M[i, j]):
                    ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center",
                            color="white" if M[i, j] < mid else "black", fontsize=8)
        bi, bj = np.unravel_index(np.nanargmax(M), M.shape)
        ax.scatter([bj], [bi], s=360, marker="*", facecolors="none",
                   edgecolors="red", linewidths=2.0, zorder=5)
        ax.set_title(f"{title}\n" + r"opt $\alpha$=" + f"{ALPHAS[bj]:g}, " + r"$\lambda$="
                     + f"{LAMBDAS[bi]:g} ({M[bi,bj]:.2f})", fontweight="bold", fontsize=12, pad=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    axes[0].set_ylabel(r"$\lambda$ (diversity weight)", fontsize=13)
    plt.tight_layout()
    out = f"{OUT}/{fname}"
    plt.savefig(out, dpi=1200, bbox_inches="tight")
    plt.savefig(out.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    plt.close()
    return out


def main():
    panels = [("en", ["en"], "Alpha-Lambda Grid (English)", "grid_heatmap_en.pdf"),
              ("nonen", NON_EN, "Alpha-Lambda Grid (Non-English avg)", "grid_heatmap_nonen.pdf"),
              ("all", LANGS, "Alpha-Lambda Grid (All languages avg)", "grid_heatmap_all.pdf")]
    table = {}
    mats = []
    for key, langs, title, fname in panels:
        M = matrix(langs)
        mats.append(M)
        table[key] = {"alphas": ALPHAS, "lambdas": LAMBDAS, "matrix_rows_lambda": M.tolist()}
        n_missing = int(np.isnan(M).sum())
        print(f"[{key}] filled {M.size - n_missing}/{M.size} cells")
        if n_missing == 0:
            print(heatmap(M, title, fname))
        else:
            print(f"  ... {n_missing} cells still missing, skipping plot")
    json.dump(table, open(f"{OUT}/grid_oqs_table.json", "w"), indent=2)
    if all(not np.isnan(M).any() for M in mats):
        print(combined(mats, ["English", "Non-English (avg)", "All languages (avg)"],
                       "grid_heatmap_combined.pdf"))


if __name__ == "__main__":
    main()
