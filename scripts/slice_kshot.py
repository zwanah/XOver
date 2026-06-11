"""Slice the K=6 profile-DPP selection into K=1..5 cells (no re-selection).

The profile-DPP selector uses fixed-k *greedy* MAP inference, which is
prefix-consistent: the j-th greedy pick is identical whether the target is k=j or
k=6. Therefore ``selected(k=6)[:K] == selected(k=K)`` exactly, and we can build the
K=1..5 few-shot prompts by re-assembling the first K demos of the existing K=6 cell
instead of re-running the kernel + greedy. Demo order (Example 1..K) is the greedy
selection order, so the sliced prompt is the genuine K-shot prompt.

Reuses the same ``assemble_prompt`` renderer so the sliced cells are byte-identical
to what ``build_fewshot.py --k K`` would have produced.

Usage (from final/):
    python scripts/slice_kshot.py --alpha 0.3 --lam 0.5 --M 20 --split dev
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.c2_retrieval.assemble import assemble_prompt  # noqa: E402
from pipeline.c2_retrieval.config import build_pool_config, load_config  # noqa: E402
from pipeline.c2_retrieval.pool_map import PoolMap  # noqa: E402

logger = logging.getLogger("slice_kshot")

# lang -> (conf file, data out_dir). en uses the unsuffixed config.
LANGS = {
    "en": ("conf/fewshot_profile.yaml", "data/fewshot_profile"),
    "zh": ("conf/fewshot_zh_profile.yaml", "data/fewshot_zh_profile"),
    "yue": ("conf/fewshot_yue_profile.yaml", "data/fewshot_yue_profile"),
    "fr": ("conf/fewshot_fr_profile.yaml", "data/fewshot_fr_profile"),
    "de": ("conf/fewshot_de_profile.yaml", "data/fewshot_de_profile"),
    "ja": ("conf/fewshot_ja_profile.yaml", "data/fewshot_ja_profile"),
    "ko": ("conf/fewshot_ko_profile.yaml", "data/fewshot_ko_profile"),
    "ru": ("conf/fewshot_ru_profile.yaml", "data/fewshot_ru_profile"),
}


def main() -> None:
    p = argparse.ArgumentParser(description="Slice K=6 profile-DPP cell into K=1..5")
    p.add_argument("--alpha", type=float, default=0.3)
    p.add_argument("--lam", type=float, default=0.5)
    p.add_argument("--M", type=int, default=20)
    p.add_argument("--split", default="dev")
    p.add_argument("--ks", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    opts = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")

    src_k = 6
    tag = f"a{opts.alpha}_profile_lam{opts.lam}_M{opts.M}"
    src_name = f"questions_{opts.split}_k{src_k}_{tag}.json"

    # Pool is the same english demo bank for every language; build it once.
    pool_map = None
    for lang, (conf, out_dir) in LANGS.items():
        src_path = os.path.join(out_dir, src_name)
        if not os.path.exists(src_path):
            raise FileNotFoundError(f"[{lang}] missing K=6 cell {src_path}")
        with open(src_path, encoding="utf-8") as f:
            cell = json.load(f)
        assert cell["args"]["k"] == src_k, cell["args"]
        assert abs(cell["args"]["alpha"] - opts.alpha) < 1e-9, cell["args"]
        assert abs(cell["args"]["lambda"] - opts.lam) < 1e-9, cell["args"]

        if pool_map is None:
            pool_map = PoolMap.build(build_pool_config(load_config(conf)))

        for K in opts.ks:
            questions = []
            short = 0
            for q in cell["questions"]:
                sel = q["selected"][:K]
                if len(sel) < K:
                    short += 1
                prompt = assemble_prompt(sel, pool_map, q["nl_question"])
                questions.append({
                    "query_index": q["query_index"],
                    "nl_question": q["nl_question"],
                    "OverpassQL": q["OverpassQL"],
                    "selected": sel,
                    "dedup_skips": q.get("dedup_skips", []),
                    "prompt": prompt,
                })
            out = {
                "args": {**cell["args"], "k": K, "sliced_from_k": src_k},
                "stats": {"n_queries": len(questions), "underfilled_queries": short},
                "questions": questions,
            }
            out_name = f"questions_{opts.split}_k{K}_{tag}.json"
            out_path = os.path.join(out_dir, out_name)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False, indent=2)
            logger.info("[%s] wrote %s (underfilled=%d)", lang, out_path, short)


if __name__ == "__main__":
    main()
