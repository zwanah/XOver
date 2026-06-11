"""K-shot OQS figures by language group: English / Non-English / ALL (3 figures).

Each figure plots ONE k-shot curve (OQS vs K=0..6, group-averaged) against the
fine-tuned **SFT/LoRA range** band + the best FT/LoRA line, in the visual style of
``eval_backend/TACO/TACO_paper/Manuscript/figures/make_pareto_deepseek_v4_flash.py``.

- k-shot: deepseek-v4-flash NON-THINKING, plain demos, temp 0, dev; profile-DPP demos
  (alpha=0.3, lambda=0.5, M=20). K=1..5 sliced from the K=6 greedy selection;
  K=0 = W1 baseline 0-shot. Group value = mean OQS over the group's languages.
- SFT/LoRA band: all 6 LoRA configs' DEV OQS (3 per-language-trained backbones +
  3 pooled-8). Band = [min, max] of the group-average OQS across configs; the best
  config is drawn as a dashed line and labelled. (Includes the Qwen2.5-7B per-language
  config, whose dev OQS is anomalously low — band floor reflects that, per request.)

Usage (from this dir):  python plot_kshot_groups.py
"""
from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/path/to/XOver")
FINAL = ROOT / "final"
SWEEP = FINAL / "outputs/predictions/ovq_kshot/eval_result"
W1 = ROOT / "outputs/predictions/W1_baseline/deepseek-v4-flash/eval_result"
LORA = ROOT / "outputs/predictions/lora"
OUTDIR = Path(__file__).parent

KS = [1, 2, 3, 4, 5, 6]  # K=0 (zero-shot) dropped per request
# short -> full dataset token
LANGS = {
    "en": "english", "zh": "mandarin_simplified", "yue": "cantonese", "fr": "french",
    "de": "german", "ja": "japanese", "ko": "korean", "ru": "russian",
}
FULL = list(LANGS.values())
NON_EN = [f for f in FULL if f != "english"]

# Language groups: (key, title, member full-tokens)
GROUPS = [
    ("english", "English", ["english"]),
    ("non_english", "Non-English (avg of 7)", NON_EN),
    ("all", "All languages (avg of 8)", FULL),
]

# LoRA backbones -> display name. Each has per-language ({lang}) and pooled8 dirs.
BACKBONES = {
    "qwen3_4b": "LoRA-Qwen3-4B",
    "llama31_8b": "LoRA-Llama-3.1-8B",
    "qwen25_7b": "LoRA-Qwen2.5-7B",
}


def oqs_mean(path: Path) -> float:
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    return float(d["statistics"]["evaluation"]["OQS"]["mean"]) * 100.0


def _glob_one(dirpath: Path, pattern: str):
    hits = sorted(dirpath.glob(pattern))
    return hits[0] if hits else None


# ---- k-shot curve: per-language OQS at each K, then group-averaged ----
def kshot_lang(short: str, full: str, k: int) -> float:
    if k == 0:
        p = W1 / f"{full}_dev_n1000_0shot_eval_results.json"
    else:
        p = SWEEP / f"deepseek-v4-flash_{short}_dev_k{k}_a0.3_lam0.5_eval_results.json"
    return oqs_mean(p)


def kshot_curve(members: list) -> list:
    short_of = {v: k for k, v in LANGS.items()}
    out = []
    for k in KS:
        vals = [kshot_lang(short_of[m], m, k) for m in members]
        out.append(float(np.mean(vals)))
    return out


# ---- FT/LoRA configs: per-language dev OQS for all 6 configs ----
def load_ft_configs() -> dict:
    """Return {display_config_name: {full_lang: oqs}} for all 6 LoRA configs."""
    cfgs = {}
    for bb, disp in BACKBONES.items():
        # per-language-trained: one dir per language
        per = {}
        for full in FULL:
            d = LORA / f"lora_{bb}_{full}_seed42" / "eval_result"
            f = _glob_one(d, f"{full}_dev*eval_results.json") if d.is_dir() else None
            if f:
                per[full] = oqs_mean(f)
        if len(per) == len(FULL):
            cfgs[f"{disp} (per-lang)"] = per
        # pooled-8: one dir, all languages
        pd = LORA / f"lora_{bb}_pooled8_seed42" / "eval_result"
        pool = {}
        for full in FULL:
            f = _glob_one(pd, f"{full}_dev*eval_results.json") if pd.is_dir() else None
            if f:
                pool[full] = oqs_mean(f)
        if len(pool) == len(FULL):
            cfgs[f"{disp} (pooled-8)"] = pool
    return cfgs


def group_avg(per_lang: dict, members: list) -> float:
    return float(np.mean([per_lang[m] for m in members]))


def ft_band(cfgs: dict, members: list):
    """Return (lo, hi, best_name, best_val) over configs' group-average OQS."""
    scored = [(group_avg(d, members), name) for name, d in cfgs.items()]
    lo = min(scored)
    hi = max(scored)
    return lo[0], hi[0], hi[1], hi[0]


plt.rcParams.update({
    "font.family": "serif",
    "font.size": 15,
    "axes.titlesize": 18,
    "axes.labelsize": 17,
    "legend.fontsize": 13,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "axes.linewidth": 1.2,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

KSHOT_STYLE = dict(color="#9467bd", marker="P")


def make_figure(title: str, ys: list, lo: float, hi: float, best_name: str,
                best_val: float, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 5.4))

    # SFT/LoRA reference band + best line (mirrors make_pareto).
    ax.axhspan(lo, hi, color="#FFD27F", alpha=0.30, zorder=0, label="SFT/LoRA range")
    ax.axhline(best_val, color="#E08B00", ls="-.", lw=1.4, alpha=0.95, zorder=0,
               label=f"Best SFT/LoRA ({best_name}, {best_val:.1f})")

    # k-shot trajectory.
    ax.plot(KS, ys, ls="-", lw=2.0, markersize=10, alpha=0.9, zorder=3,
            label=r"$k$-Shot (retrieval ICL)", **KSHOT_STYLE)

    ax.set_xlabel(r"Number of shots $K$")
    ax.set_ylabel(r"OQS $\uparrow$")
    ax.set_xticks(KS)
    ax.grid(True, which="both", alpha=0.25, ls=":")
    ax.set_title(f"MOsmNL dev — {title}", pad=10, fontsize=16)
    ax.legend(loc="lower right", frameon=False)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=1200)
    fig.savefig(out_path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}  (+ .png)")


def main() -> None:
    cfgs = load_ft_configs()
    print(f"loaded {len(cfgs)} FT/LoRA configs: {list(cfgs)}")
    for key, title, members in GROUPS:
        ys = kshot_curve(members)
        lo, hi, best_name, best_val = ft_band(cfgs, members)
        print(f"\n[{key}] k-shot OQS: " + " ".join(f"K{k}={v:.2f}" for k, v in zip(KS, ys)))
        print(f"[{key}] FT band [{lo:.2f}, {hi:.2f}]  best={best_name} ({best_val:.2f})")
        make_figure(title, ys, lo, hi, best_name, best_val,
                    OUTDIR / f"kshot_oqs_{key}.pdf")


if __name__ == "__main__":
    main()
