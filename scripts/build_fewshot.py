"""Build relevance-ranked plain few-shot prompts for one (split, k, alpha) cell.

Loads TACO's precomputed sim/influence scores, builds the demo pool from the
score-source files, selects top-k demos per eval query (relevance =
``alpha*sim + (1-alpha)*influence``, dedup-by-gold-prefix), and writes a
questions.json-equivalent with one plain ``(#question, #OverpassQL)`` prompt per
query.

Three decoupled stages (see ``pipeline/c2_retrieval``): score -> select (returns
pool indices) -> assemble. Swap the selector to change the example-selection
method without touching scoring or assembly.

Usage (from final/):
    python scripts/build_fewshot.py                  # dev, defaults from conf
    python scripts/build_fewshot.py --split dev --k 6 --alpha 0.2
    python scripts/build_fewshot.py --sweep          # all {3,6}x{.1,.2,.3} cells
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.c2_retrieval.assemble import assemble_prompt  # noqa: E402
from pipeline.c2_retrieval.config import (  # noqa: E402
    SelectConfig,
    build_assemble_config,
    build_pool_config,
    build_score_config,
    build_select_config,
    load_config,
)
from pipeline.c2_retrieval.pool_map import PoolMap, load_eval_items  # noqa: E402
from pipeline.c2_retrieval.profile_kernel import ProfileKernelProvider  # noqa: E402
from pipeline.c2_retrieval.scores import ScoreTable  # noqa: E402
from pipeline.c2_retrieval.select import ProfileDppSelector, TopKSelector  # noqa: E402

logger = logging.getLogger("build_fewshot")


def make_selector(
    select_cfg: SelectConfig,
    *,
    alpha: float,
    dedup_prefix_len: int,
    kernel_provider=None,
):
    """Build the selector named by ``select_cfg.method`` (relevance | dpp)."""
    if select_cfg.method == "relevance":
        return TopKSelector(alpha=alpha, dedup_prefix_len=dedup_prefix_len)
    if select_cfg.method == "dpp":
        if select_cfg.kernel_mode != "profile":
            raise ValueError(
                f"final/ supports only kernel_mode=profile (got {select_cfg.kernel_mode!r}); "
                "only the profile kernel is supported in this release"
            )
        if kernel_provider is None:
            raise ValueError("dpp method needs a profile kernel_provider")
        return ProfileDppSelector(
            alpha=alpha,
            dedup_prefix_len=dedup_prefix_len,
            M=select_cfg.M,
            lam=select_cfg.lam,
            phi_mode=select_cfg.phi_mode,
            kernel_provider=kernel_provider,
        )
    raise ValueError(f"unknown select.method {select_cfg.method!r} (expected relevance|dpp)")


def build_cell(
    score_table: ScoreTable,
    pool_map: PoolMap,
    eval_items: list,
    select_cfg: SelectConfig,
    *,
    split: str,
    k: int,
    alpha: float,
    dedup_prefix_len: int,
    kernel_provider=None,
) -> dict:
    """Assemble plain prompts for every eval query in one parameter cell."""
    selector = make_selector(
        select_cfg, alpha=alpha, dedup_prefix_len=dedup_prefix_len,
        kernel_provider=kernel_provider,
    )
    questions = []
    total_dedup, short = 0, 0
    for qi in score_table.query_indices:
        qs = score_table.by_query[qi]
        item = eval_items[qi]  # query_index indexes positionally into the eval file
        sel = selector.select(qs, pool_map, k)
        prompt = assemble_prompt(sel.selected, pool_map, item["nl_question"])
        questions.append({
            "query_index": qi,
            "nl_question": item["nl_question"],
            "OverpassQL": item["OverpassQL"],
            "selected": sel.selected,
            "dedup_skips": sel.dedup_skips,
            "prompt": prompt,
        })
        total_dedup += len(sel.dedup_skips)
        if len(sel.selected) < k:
            short += 1

    return {
        "args": {
            "split": split, "k": k, "alpha": alpha,
            "dedup_prefix_len": dedup_prefix_len,
            "method": select_cfg.method, "lambda": select_cfg.lam,
            "M": select_cfg.M, "kernel_mode": select_cfg.kernel_mode,
        },
        "stats": {
            "n_queries": len(questions),
            "total_dedup_skips": total_dedup,
            "underfilled_queries": short,
        },
        "questions": questions,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Build step-2 relevance few-shot prompts (plain)")
    p.add_argument("--conf", default="conf/fewshot.yaml")
    p.add_argument("--split", default=None)
    p.add_argument("--k", type=int, default=None)
    p.add_argument("--alpha", type=float, default=None)
    p.add_argument("--sweep", action="store_true",
                   help="build all {k in 3,6} x {alpha in .1,.2,.3} cells")
    p.add_argument("--method", default=None, choices=["relevance", "dpp"],
                   help="override select.method (default from conf)")
    p.add_argument("--lam", type=float, default=None,
                   help="override profile-DPP diversity weight lambda (0 == relevance top-k)")
    p.add_argument("--M", type=int, default=None,
                   help="override profile-DPP candidate-pool size")
    opts = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")
    cfg = load_config(opts.conf)
    pool_cfg = build_pool_config(cfg)
    score_cfg = build_score_config(cfg)
    base = build_assemble_config(cfg, split=opts.split, k=opts.k, alpha=opts.alpha)
    select_cfg = build_select_config(cfg, method=opts.method, lam=opts.lam, M=opts.M)

    pool_map = PoolMap.build(pool_cfg)  # plain mode: pool from score-source, no CoT corpus
    eval_items = load_eval_items(
        os.path.join(cfg["eval"]["score_source_root"], cfg["eval"]["files"][base.split])
    )
    score_table = ScoreTable.load(
        score_cfg.sim_path(base.split), score_cfg.influence_path(base.split),
        pool_cfg.size,
    )

    # Profile-DPP needs the cross-lingual relevance-profile kernel (query-independent,
    # built once over the multilingual influence bank). Relevance top-k needs nothing.
    kernel_provider = None
    if select_cfg.method == "dpp":
        suffixes = score_cfg.profile_influence_suffixes
        paths = score_cfg.profile_influence_paths(base.split, suffixes)
        kernel_provider = ProfileKernelProvider.from_influence_pkls(paths, pool_cfg.size)

    if opts.sweep:
        cells = [(k, a) for k in (3, 6) for a in (0.1, 0.2, 0.3)]
    else:
        cells = [(base.k, base.alpha)]

    out_dir = cfg["run"]["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    # Tag the filename by method so relevance vs profile-DPP cells never collide.
    if select_cfg.method == "dpp":
        tag = f"profile_lam{select_cfg.lam}_M{select_cfg.M}"
    else:
        tag = "plain"
    for k, alpha in cells:
        result = build_cell(
            score_table, pool_map, eval_items, select_cfg,
            split=base.split, k=k, alpha=alpha,
            dedup_prefix_len=base.dedup_prefix_len,
            kernel_provider=kernel_provider,
        )
        fname = f"questions_{base.split}_k{k}_a{alpha}_{tag}.json"
        out_path = os.path.join(out_dir, fname)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        logger.info("wrote %s  %s", out_path, result["stats"])


if __name__ == "__main__":
    main()
