"""Diagnostic step D2 — generate the candidate pool for the selection diagnostic.

For a seeded sample of dev-english queries, emit:
  * 1 **greedy** decode (temp 0.0)  → defines Pass@1 (the shippable pick)
  * K **sampled** decodes (temp 0.8) → the pool oracle Pass@K / voting run over

Generator is fixed to deepseek-v4-flash with reasoning DISABLED so candidate
diversity comes purely from temperature, not from a model that silently reasons
(which would collapse samples). Prompts come from the shipped best step-2 cell
``questions_dev_k6_a0.1_full.json`` so candidates reflect the real pipeline.

Output: ``data/diag/candidates.json`` — one record per query with ``greedy`` and
``candidates`` (list of K query strings). Execution happens later in
``scripts/diag_execute.py``.

Usage (from repo root):
    python scripts/diag_gen_candidates.py --limit 5        # smoke test (5 queries)
    python scripts/diag_gen_candidates.py                  # full 200 x (1+8)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple

# Sibling scripts live in this same scripts/ dir.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_baseline_llm as rbl  # noqa: E402

# NOTE: the candidate generator still reuses the CoT system prompt + query
# extractor from infer_fewshot. extract_cot_query also recovers the query from a
# plain (query-only) response, so this works in plain mode; COT_SYSTEM_PROMPT
# remains the prompt used here. De-CoT-ing the generation prompt (plain system
# prompt by default) is a follow-up — see final/README.md "Follow-ups".
from infer_fewshot import COT_SYSTEM_PROMPT, _load_bbox_map, extract_cot_query  # noqa: E402

logger = logging.getLogger("diag_gen")

GREEDY_SLOT = "g"


def _sample_indices(n_pool: int, n_sample: int, seed: int) -> List[int]:
    """Deterministic sample of ``n_sample`` positions from ``range(n_pool)``."""
    if n_sample >= n_pool:
        return list(range(n_pool))
    rng = random.Random(seed)
    return sorted(rng.sample(range(n_pool), n_sample))


def main() -> None:
    p = argparse.ArgumentParser(description="Generate candidate pool for the selection diagnostic")
    p.add_argument("--questions", default="data/fewshot/questions_dev_k6_a0.1_full.json")
    p.add_argument("--model", default="deepseek-v4-flash")
    p.add_argument("--provider", default="deepseek", choices=sorted(rbl.PROVIDER_BASE_URL))
    p.add_argument("--n_queries", type=int, default=200, help="seeded query sample size")
    p.add_argument("--k", type=int, default=8, help="sampled candidates per query")
    p.add_argument("--temperature", type=float, default=0.8, help="sampling temp for the K pool")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_tokens", type=int, default=4096)
    p.add_argument("--workers", type=int, default=128)
    p.add_argument("--num_keys", type=int, default=4)
    p.add_argument("--limit", type=int, default=0, help="cap queries for a smoke test (0 = use n_queries)")
    p.add_argument("--eval_source", default="/path/to/OsmNL_source")
    p.add_argument("--output", default="data/diag/candidates.json")
    p.add_argument("--resume", action="store_true")
    opts = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")

    with open(opts.questions, encoding="utf-8") as f:
        cell = json.load(f)
    split = cell["args"]["split"]
    questions = cell["questions"]
    bbox_map = _load_bbox_map(os.path.join(opts.eval_source, f"{split}_final.json"))

    n_sample = opts.limit if opts.limit > 0 else opts.n_queries
    sel = _sample_indices(len(questions), n_sample, opts.seed)
    chosen = [questions[i] for i in sel]
    logger.info("cell=%s split=%s pool=%d -> sampled %d queries (seed=%d), k=%d temp=%.2f",
                os.path.basename(opts.questions), split, len(questions), len(chosen),
                opts.seed, opts.k, opts.temperature)

    os.makedirs(os.path.dirname(opts.output), exist_ok=True)
    ckpt = opts.output.replace(".json", "_partial.json")
    # gen[qi][slot] = generated query string. slot = "g" (greedy) or 0..k-1.
    gen: Dict[str, Dict[str, str]] = {}
    if opts.resume and os.path.exists(ckpt):
        gen = json.load(open(ckpt, encoding="utf-8"))
        logger.info("resume: %d queries have partial generations", len(gen))

    # Build the flat task list: (query_index, slot, prompt, temperature).
    tasks: List[Tuple[int, str, str, float]] = []
    for q in chosen:
        qi = q["query_index"]
        have = gen.get(str(qi), {})
        if GREEDY_SLOT not in have:
            tasks.append((qi, GREEDY_SLOT, q["prompt"], 0.0))
        for s in range(opts.k):
            if str(s) not in have:
                tasks.append((qi, str(s), q["prompt"], opts.temperature))
    logger.info("%d generation calls to make (of %d total)",
                len(tasks), len(chosen) * (opts.k + 1))

    keys = rbl._load_provider_keys(opts.provider)
    base_url = rbl.PROVIDER_BASE_URL[opts.provider]
    from openai import OpenAI  # noqa: E402
    n_keys = max(1, min(opts.num_keys, len(keys)))
    clients = [OpenAI(base_url=base_url, api_key=keys[j % len(keys)]) for j in range(n_keys)]
    extra_body = {"thinking": {"type": "disabled"}}  # genuine non-reasoning generator

    def work(j: int):
        qi, slot, prompt, temp = tasks[j]
        messages = [{"role": "system", "content": COT_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}]
        raw = ""
        for attempt in range(3):
            try:
                resp = clients[j % n_keys].chat.completions.create(
                    model=opts.model, messages=messages,
                    max_tokens=opts.max_tokens, temperature=temp, extra_body=extra_body)
                raw = resp.choices[0].message.content or ""
                break
            except Exception as e:  # noqa: BLE001
                logger.warning("call failed (%d/3) qi=%s slot=%s: %s", attempt + 1, qi, slot, e)
                time.sleep(2 ** attempt)
        return qi, slot, extract_cot_query(raw)

    t0, done = time.time(), 0
    with ThreadPoolExecutor(max_workers=opts.workers) as pool:
        futs = [pool.submit(work, j) for j in range(len(tasks))]
        for fut in as_completed(futs):
            qi, slot, query = fut.result()
            gen.setdefault(str(qi), {})[slot] = query
            done += 1
            if done % 100 == 0 or done == len(tasks):
                json.dump(gen, open(ckpt, "w"), ensure_ascii=False, indent=2)
                logger.info("%d/%d gen done (%.1fs)", done, len(tasks), time.time() - t0)

    # Assemble per-query records.
    records = []
    n_empty_greedy = 0
    for q in chosen:
        qi = q["query_index"]
        g = gen.get(str(qi), {})
        greedy = g.get(GREEDY_SLOT, "")
        if not greedy:
            n_empty_greedy += 1
        records.append({
            "query_index": qi,
            "nl_question": q["nl_question"],
            "OverpassQL": q["OverpassQL"],
            "bbox": bbox_map[qi] if qi < len(bbox_map) else "",
            "greedy": greedy,
            "candidates": [g.get(str(s), "") for s in range(opts.k)],
        })
    payload = {
        "args": {"model": opts.model, "provider": opts.provider, "split": split,
                 "n_queries": len(records), "k": opts.k, "temperature": opts.temperature,
                 "seed": opts.seed, "cell": os.path.basename(opts.questions)},
        "records": records,
    }
    with open(opts.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    logger.info("wrote %s | %d queries, k=%d, empty_greedy=%d",
                opts.output, len(records), opts.k, n_empty_greedy)


if __name__ == "__main__":
    main()
