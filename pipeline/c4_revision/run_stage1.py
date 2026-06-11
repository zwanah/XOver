"""CLI: Stage-1 output hygiene + truncation regeneration over a prediction file.

Reads a baseline prediction JSON, cleans prose/markdown leaks deterministically,
and regenerates truncated items once at a higher token budget with the original
model/settings (read from the file's `statistics` block; CLI flags override for
older files that predate the extended statistics). Writes a same-schema JSON so
the output feeds straight into `pipeline.c1_eval.runner --compute_execution` or the
downstream `pipeline.c4_revision.run` reviser.

Example:
    python -m pipeline.c4_revision.run_stage1 \
        --input_file outputs/predictions/.../english_dev_n1000_6shot.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

from .hygiene import DirtyKind, classify
from .regen import DEFAULT_REGEN_MAX_TOKENS, RegenConfig, make_prompt_builder, run_stage1

logger = logging.getLogger(__name__)

REPO_ROOT = "/path/to/XOver"


def _load_predictions(path: str) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    with open(path, "r", encoding="utf-8") as f:
        file_data = json.load(f)
    if isinstance(file_data, dict) and "data" in file_data:
        return file_data["data"], file_data.get("statistics")
    if isinstance(file_data, list):
        return file_data, None
    raise SystemExit(f"unknown input format: {type(file_data)}")


def _build_generate_fn(cfg: RegenConfig, num_keys: int):
    """Wire a real OpenAI-compatible generate fn using the *original* model and
    SYSTEM_PROMPT from generation, so a regen is identical except for the budget."""
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)
    from openai import OpenAI
    from scripts.run_baseline_llm import (  # noqa: E402
        PROVIDER_BASE_URL, _load_provider_keys, call_llm,
    )
    keys = _load_provider_keys(cfg.provider)
    n = max(1, min(num_keys, len(keys))) if num_keys > 0 else 1
    client = OpenAI(base_url=PROVIDER_BASE_URL[cfg.provider], api_key=keys[0])

    def generate_fn(user_message: str, max_tokens: int) -> str:
        return call_llm(client, cfg.model, user_message, max_tokens, cfg.temperature)

    return generate_fn


def main() -> None:
    p = argparse.ArgumentParser(
        description="MOsmNL Stage-1 — output hygiene + truncation regeneration")
    p.add_argument("--input_file", required=True, help="Baseline prediction JSON.")
    p.add_argument("--output_file", default=None,
                   help="Defaults to <input_dir>/stage1/<input_name>.json")
    p.add_argument("--regen_max_tokens", type=int, default=DEFAULT_REGEN_MAX_TOKENS,
                   help="Token budget for regenerating truncated predictions.")
    p.add_argument("--num_keys", type=int, default=0,
                   help="Cap on API keys (0 = single key). Regen is low-volume.")
    # Overrides for params missing from older statistics blocks.
    p.add_argument("--model", default=None, help="Override original gen model.")
    p.add_argument("--provider", default=None, choices=["aigcbest", "deepseek"])
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--shots", type=int, default=None)
    p.add_argument("--icl_pool_lang", default=None)
    p.add_argument("--icl_index_dir", default=None)
    p.add_argument("--mpnet_model", default=None)
    p.add_argument("--query_instruction", default=None)
    p.add_argument("--encoder_device", default=None)
    opts = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="[%(asctime)s] %(levelname)s %(message)s")

    if opts.output_file is None:
        in_dir = os.path.dirname(os.path.abspath(opts.input_file))
        in_name = os.path.basename(opts.input_file)
        opts.output_file = os.path.join(in_dir, "stage1", in_name)

    data, in_stats = _load_predictions(opts.input_file)
    if not data:
        raise SystemExit("input file has 0 samples")

    overrides = {
        "model": opts.model, "provider": opts.provider,
        "temperature": opts.temperature, "shots": opts.shots,
        "icl_pool_lang": opts.icl_pool_lang, "icl_index_dir": opts.icl_index_dir,
        "mpnet_model": opts.mpnet_model, "query_instruction": opts.query_instruction,
        "encoder_device": opts.encoder_device, "regen_max_tokens": opts.regen_max_tokens,
    }
    cfg = RegenConfig.from_statistics(in_stats, overrides)
    logger.info("regen config: %s", cfg)

    # Reuse stored prompts where present; only load the encoder/index for legacy
    # files that have truncated items lacking a saved `user_message`.
    needs_builder = any(
        classify(it.get("generated_text", "") or "") is DirtyKind.TRUNCATED
        and not (it.get("user_message") or "")
        for it in data
    )
    if needs_builder:
        logger.info("legacy file: some truncated items lack a stored prompt; "
                    "loading encoder + ICL index to reconstruct (shots=%d)", cfg.shots)
        prompt_builder = make_prompt_builder(cfg)
    else:
        logger.info("reusing stored user_message prompts; no encoder needed")
        prompt_builder = None

    generate_fn = _build_generate_fn(cfg, opts.num_keys)

    out_items, stage1_stats = run_stage1(
        data, in_stats, generate_fn=generate_fn, prompt_builder=prompt_builder,
        regen_max_tokens=cfg.regen_max_tokens)

    out_payload: Dict[str, Any] = {"data": out_items}
    out_payload["statistics"] = dict(in_stats) if in_stats else {}
    out_payload["statistics"]["stage1"] = stage1_stats

    os.makedirs(os.path.dirname(os.path.abspath(opts.output_file)), exist_ok=True)
    with open(opts.output_file, "w", encoding="utf-8") as f:
        json.dump(out_payload, f, ensure_ascii=False, indent=2)

    logger.info("stage-1 stats: %s", stage1_stats)
    logger.info("wrote: %s", opts.output_file)
    lang = (in_stats or {}).get("lang", "english")
    split = (in_stats or {}).get("split", "dev")
    logger.info("Next: python -m pipeline.c1_eval.runner --input_file %s "
                "--lang %s --split %s --compute_execution",
                opts.output_file, lang, split)


if __name__ == "__main__":
    main()
