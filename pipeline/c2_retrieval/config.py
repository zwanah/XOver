"""Load conf/fewshot.yaml and build the typed configs the step-2 modules need.

Mirrors src/cot/config.py: a thin YAML loader plus frozen dataclasses so the
assembler is config-driven and the paths live in one place.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import yaml


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass(frozen=True)
class PoolConfig:
    """Where the 6180-demo pool comes from, and how to round-trip the two files."""

    score_source_root: str
    cot_source_root: str
    # split -> (score_source_filename, cot_source_filename)
    splits: Dict[str, Tuple[str, str]]
    size: int

    def score_source_path(self, split: str) -> str:
        return os.path.join(self.score_source_root, self.splits[split][0])

    def cot_source_path(self, split: str) -> str:
        return os.path.join(self.cot_source_root, self.splits[split][1])


@dataclass(frozen=True)
class ScoreConfig:
    """Resolves the precomputed similarity / influence pkl paths."""

    root: str
    sim_encoder: str
    influence_model: str
    # profile-DPP kernel: the cross-lingual relevance profile is stacked over ALL of
    # these languages regardless of which language is being evaluated, so it uses its
    # own BARE base influence model (not the eval-language-suffixed ``influence_model``)
    # plus per-language suffixes ("" == english, "ZH"/"JA" == translated splits).
    # Empty model -> ``influence_model``; empty suffixes -> caller's default.
    profile_influence_model: str = ""
    profile_influence_suffixes: Tuple[str, ...] = ()

    def sim_path(self, split: str) -> str:
        return os.path.join(
            self.root, "Similarity_score", f"{self.sim_encoder}_{split}_scores.pkl"
        )

    def influence_path(self, split: str) -> str:
        return os.path.join(
            self.root, "Influence_score", f"{self.influence_model}_{split}_scores.pkl"
        )

    def profile_influence_paths(self, split: str, suffixes: Tuple[str, ...]) -> List[str]:
        """Per-language influence pkl paths for the profile kernel, one per suffix.

        ``""`` -> ``{model}_{split}_scores.pkl`` (english); a non-empty suffix ``S`` ->
        ``{model}_{S}_{split}_scores.pkl`` (translated split). ``model`` defaults to
        ``influence_model`` when ``profile_influence_model`` is empty.
        """
        base = os.path.join(self.root, "Influence_score")
        model = self.profile_influence_model or self.influence_model
        out: List[str] = []
        for s in suffixes:
            stem = f"{model}_{s}_{split}" if s else f"{model}_{split}"
            out.append(os.path.join(base, f"{stem}_scores.pkl"))
        return out


@dataclass(frozen=True)
class AssembleConfig:
    """Selection + assembly knobs (plain mode: no render variant)."""

    k: int
    alpha: float
    split: str
    dedup_prefix_len: int


@dataclass(frozen=True)
class SelectConfig:
    """Example-selection method + reranker knobs (optional ``select:`` conf block).

    Two methods only (other selection variants were explored and not adopted):
      * ``relevance`` (default) — top-k by ``alpha*sim + (1-alpha)*influence``.
      * ``dpp`` — cross-lingual relevance-profile DPP diversity over the top-M pool
        (``kernel_mode`` is fixed to ``profile``; ``lambda == 0`` reduces to relevance).
    """

    method: str = "relevance"        # "relevance" | "dpp"
    M: int = 20                      # candidate-pool size for the DPP reranker
    lam: float = 0.0                 # diversity weight (0 == relevance top-k)
    phi_mode: str = "linear"         # concave relevance transform: linear | log | sqrt
    kernel_mode: str = "profile"     # only "profile" is supported in final/


def build_pool_config(cfg: dict) -> PoolConfig:
    p = cfg["pool"]
    # split -> (score_source_file[, cot_source_file]); cot file optional (plain mode).
    splits = {
        k: (v[0], v[1]) if len(v) > 1 else (v[0], "")
        for k, v in p["splits"].items()
    }
    return PoolConfig(
        score_source_root=p["score_source_root"],
        cot_source_root=p.get("cot_source_root", ""),  # optional: only used in CoT mode
        splits=splits,
        size=int(p["size"]),
    )


def build_score_config(cfg: dict) -> ScoreConfig:
    s = cfg["scores"]
    return ScoreConfig(
        root=s["root"],
        sim_encoder=s["sim_encoder"],
        influence_model=s["influence_model"],
        profile_influence_model=s.get("profile_influence_model", ""),
        profile_influence_suffixes=tuple(s.get("profile_influence_suffixes", []) or []),
    )


def build_select_config(cfg: dict, **overrides) -> SelectConfig:
    """Read the optional ``select:`` block; default to relevance top-k (no behaviour change).

    A non-None override always wins. The conf uses key ``lambda`` while a CLI flag passes
    ``lam`` — resolve both (override > conf ``lambda`` > conf ``lam`` > 0.0).
    """
    d = dict(cfg.get("select", {}))
    o = {k: v for k, v in overrides.items() if v is not None}
    return SelectConfig(
        method=str(o.get("method", d.get("method", "relevance"))),
        M=int(o.get("M", d.get("M", 20))),
        lam=float(o.get("lam", d.get("lambda", d.get("lam", 0.0)))),
        phi_mode=str(o.get("phi_mode", d.get("phi_mode", "linear"))),
        kernel_mode=str(o.get("kernel_mode", d.get("kernel_mode", "profile"))),
    )


def build_assemble_config(cfg: dict, **overrides) -> AssembleConfig:
    d = dict(cfg.get("defaults", {}))
    d.update({k: v for k, v in overrides.items() if v is not None})
    return AssembleConfig(
        k=int(d["k"]),
        alpha=float(d["alpha"]),
        split=str(d["split"]),
        dedup_prefix_len=int(d["dedup_prefix_len"]),
    )
