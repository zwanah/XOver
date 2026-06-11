"""Run step-2 CoT few-shot prompts through an LLM (default Gemini-3-Flash).

Reads a ``questions_*.json`` produced by ``scripts/build_fewshot.py`` (each entry
carries a pre-assembled CoT demonstration block in ``prompt``) and generates one
OverpassQL prediction per eval query. Unlike the W1 baseline (which forbids
explanation), this is a **chain-of-thought** setting: the demos show structured
reasoning fields ending in ``#OverpassQL:``, so the model is asked to reason in
the same format and we extract the query after the final ``#OverpassQL:`` marker.

Output follows the project prediction contract consumed by ``src/eval/runner.py``
(``query_index``, ``query_target_ovq``, ``generated_text``; ``bbox`` for EX).

Usage (from repo root):
    python scripts/infer_fewshot.py \
        --questions data/fewshot/questions_dev_k6_a0.2_full.json \
        --model gemini-3-flash-preview --provider aigcbest \
        --num_keys 4 --workers 128 --limit 5        # smoke test
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List

# Reuse the baseline LLM helpers (key loading, provider routing, retrying call,
# fence stripping) — single source of truth, no secret duplication.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_baseline_llm as rbl  # noqa: E402

logger = logging.getLogger("infer_fewshot")

# CoT generation wrapper. The demo blocks teach the format; this only states the
# task and the required tail so the query is machine-extractable.
COT_SYSTEM_PROMPT = (
    "You are an expert in OpenStreetMap and the Overpass Query Language (OverpassQL). "
    "You are given several worked examples. Each explains a question through structured "
    "reasoning fields (#reason, #scope, #find, #spatial, #combine, #ovql-plan) and ends "
    "with the final query on a line beginning '#OverpassQL:'. Answer the final question "
    "in the SAME format: produce the reasoning fields, then a line '#OverpassQL:' followed "
    "by ONLY the OverpassQL query. The query MUST start with '[out:' and be directly "
    "executable on an Overpass API endpoint. Use only tags, values, and areas grounded in "
    "the question; do not invent constraints."
)

# Plain (CoT-isolation) wrapper. Same demos, no reasoning shown or requested. The
# '#OverpassQL:' marker is kept so extraction is identical across arms; the only
# difference vs the CoT arm is the absence of chain-of-thought.
PLAIN_SYSTEM_PROMPT = (
    "You are an expert in OpenStreetMap and the Overpass Query Language (OverpassQL). "
    "You are given several worked examples, each pairing a question with its correct query "
    "on a line beginning '#OverpassQL:'. Answer the final question by outputting ONLY a line "
    "'#OverpassQL:' followed by the query — no reasoning, explanation, or commentary. The "
    "query MUST start with '[out:' and be directly executable on an Overpass API endpoint. "
    "Use only tags, values, and areas grounded in the question; do not invent constraints."
)

SYSTEM_PROMPTS = {"cot": COT_SYSTEM_PROMPT, "plain": PLAIN_SYSTEM_PROMPT}
# Back-compat alias (the recovery diagnostic imports SYSTEM_PROMPT).
SYSTEM_PROMPT = COT_SYSTEM_PROMPT


def extract_cot_query(raw: str) -> str:
    """Take the text after the last '#OverpassQL' marker, then strip to the query."""
    if not raw:
        return ""
    if "#OverpassQL" in raw:
        raw = raw.rsplit("#OverpassQL", 1)[1]
        # drop a leading ':' / whitespace left by the marker
        raw = raw.lstrip(": \n\t")
    return rbl.extract_query(raw)


def _load_bbox_map(eval_path: str) -> List[str]:
    """query_index -> bbox (positional), for EX-readiness; empty if file missing."""
    if not os.path.exists(eval_path):
        logger.warning("eval source not found for bbox: %s (bbox left empty)", eval_path)
        return []
    with open(eval_path, encoding="utf-8") as f:
        items = json.load(f)
    return [str((it.get("location", {}) or {}).get("bbox", "") or "") for it in items]


def main() -> None:
    p = argparse.ArgumentParser(description="Infer OverpassQL from step-2 CoT few-shot prompts")
    p.add_argument("--questions", required=True, help="path to a questions_*.json cell")
    p.add_argument("--model", default="gemini-3-flash-preview")
    p.add_argument("--mode", default="plain", choices=sorted(SYSTEM_PROMPTS),
                   help="plain = ask for query only (the main setting, default); "
                        "cot = ask for reasoning + query (legacy ablation arm)")
    p.add_argument("--provider", default="aigcbest", choices=sorted(rbl.PROVIDER_BASE_URL))
    p.add_argument("--thinking", default="auto", choices=["auto", "disabled", "enabled"],
                   help="DeepSeek reasoning toggle via extra_body. 'auto' = no override "
                        "(model default). 'disabled' = genuine non-reasoning generator "
                        "(reasoning_tokens=None) — the clean CoT-isolation discriminator.")
    p.add_argument("--limit", type=int, default=0, help="cap eval queries (0 = all)")
    p.add_argument("--max_tokens", type=int, default=24576,
                   help="OUTPUT ceiling, not a target — billed on tokens actually "
                        "generated, so a generous ceiling is free on short rows but "
                        "stops reasoning models (gemini: reasoning+visible-CoT share "
                        "the budget) truncating, which removes the --recover_empty pass. "
                        "Not K-dependent: input grows with shots, output budget does not "
                        "(k6 prompts ~2k tok vs a ~128k window).")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--workers", type=int, default=128)
    p.add_argument("--num_keys", type=int, default=4)
    p.add_argument("--key_index", type=int, default=0)
    p.add_argument("--eval_source",
                   default="/path/to/OsmNL_source",
                   help="root holding {split}_final.json for bbox lookup")
    p.add_argument("--output", default=None)
    p.add_argument("--resume", action="store_true",
                   help="reuse a *_partial.json checkpoint if present")
    p.add_argument("--recover_empty", action="store_true",
                   help="seed from the existing OUTPUT file and re-run ONLY entries "
                        "whose generated_text is empty (e.g. CoT truncation). Pair with "
                        "a higher --max_tokens.")
    opts = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")

    with open(opts.questions, encoding="utf-8") as f:
        cell = json.load(f)
    args, questions = cell["args"], cell["questions"]
    if opts.limit and opts.limit > 0:
        questions = questions[: opts.limit]
    split = args["split"]
    bbox_map = _load_bbox_map(os.path.join(opts.eval_source, f"{split}_final.json"))

    base = os.path.splitext(os.path.basename(opts.questions))[0]  # questions_dev_k6_a0.2_full
    tag = base.replace("questions_", "")
    out_dir = "outputs/predictions/ovq_fewshot"
    os.makedirs(out_dir, exist_ok=True)
    if opts.output is None:
        opts.output = os.path.join(out_dir, f"{opts.model.replace('/', '_')}_{tag}.json")
    ckpt = opts.output.replace(".json", "_partial.json")

    gen_map: Dict[str, str] = {}
    # finish_reason / completion_tokens per row, populated only for rows generated
    # this run (in-memory; not checkpointed). Lets us count truncation directly
    # instead of inferring it from empty extractions.
    meta_map: Dict[str, dict] = {}
    if opts.recover_empty and os.path.exists(opts.output):
        # Seed from the final output: keep good preds, re-run only the empty ones.
        prev = json.load(open(opts.output, encoding="utf-8"))["data"]
        gen_map = {str(r["query_index"]): r.get("generated_text", "") for r in prev}
        n_empty_prev = sum(1 for v in gen_map.values() if not v)
        logger.info("recover_empty: %d preds loaded, %d empty to re-run at max_tokens=%d",
                    len(gen_map), n_empty_prev, opts.max_tokens)
    elif opts.resume and os.path.exists(ckpt):
        gen_map = json.load(open(ckpt, encoding="utf-8"))
        logger.info("resume: %d predictions already done", len(gen_map))

    keys = rbl._load_provider_keys(opts.provider)
    base_url = rbl.PROVIDER_BASE_URL[opts.provider]
    from openai import OpenAI  # noqa: E402
    n_keys = max(1, min(opts.num_keys, len(keys)))
    clients = [
        OpenAI(base_url=base_url, api_key=keys[(opts.key_index + j) % len(keys)])
        for j in range(n_keys)
    ]
    def _needs_run(q) -> bool:
        key = str(q["query_index"])
        if key not in gen_map:
            return True
        return opts.recover_empty and not gen_map[key]  # empty -> re-run

    todo = [q for q in questions if _needs_run(q)]
    logger.info("provider=%s model=%s cell=%s -> %s | %d to generate (of %d), "
                "workers=%d num_keys=%d max_tokens=%d",
                opts.provider, opts.model, tag, opts.output, len(todo), len(questions),
                opts.workers, n_keys, opts.max_tokens)

    extra_body = ({"thinking": {"type": opts.thinking}}
                  if opts.thinking != "auto" else {})

    def work(j: int):
        q = todo[j]
        prompt = q["prompt"]
        messages = [{"role": "system", "content": SYSTEM_PROMPTS[opts.mode]},
                    {"role": "user", "content": prompt}]
        # rbl.call_llm hardcodes its own system prompt; call the client directly here.
        raw, finish, ctok = "", None, None
        for attempt in range(3):
            try:
                resp = clients[j % n_keys].chat.completions.create(
                    model=opts.model, messages=messages,
                    max_tokens=opts.max_tokens, temperature=opts.temperature,
                    extra_body=extra_body)
                raw = resp.choices[0].message.content or ""
                finish = resp.choices[0].finish_reason
                ctok = getattr(resp.usage, "completion_tokens", None) if resp.usage else None
                break
            except Exception as e:  # noqa: BLE001
                logger.warning("call failed (%d/3) q=%s: %s", attempt + 1,
                               q["query_index"], e)
                time.sleep(2 ** attempt)
        return (q["query_index"], extract_cot_query(raw),
                {"finish_reason": finish, "completion_tokens": ctok})

    t0, done = time.time(), 0
    with ThreadPoolExecutor(max_workers=opts.workers) as pool:
        futs = [pool.submit(work, j) for j in range(len(todo))]
        for fut in as_completed(futs):
            qi, gen, meta = fut.result()
            gen_map[str(qi)] = gen
            meta_map[str(qi)] = meta
            done += 1
            if done % 50 == 0 or done == len(todo):
                json.dump(gen_map, open(ckpt, "w"), ensure_ascii=False, indent=2)
                logger.info("%d/%d done (%.1fs)", done, len(todo), time.time() - t0)

    # Assemble the contract prediction file.
    records = []
    n_bad = 0
    for q in questions:
        gen = gen_map.get(str(q["query_index"]), "")
        if gen and not gen.startswith("[out:"):
            n_bad += 1
        qi = q["query_index"]
        rec = {
            "query_index": qi,
            "query_nl": q["nl_question"],
            "query_target_ovq": q["OverpassQL"],
            "generated_text": gen,
            "bbox": bbox_map[qi] if qi < len(bbox_map) else "",
        }
        meta = meta_map.get(str(qi))
        if meta is not None:
            rec["finish_reason"] = meta.get("finish_reason")
            rec["completion_tokens"] = meta.get("completion_tokens")
        records.append(rec)
    # finish_reason=='length' is the ground-truth truncation signal (vs inferring
    # from empty extractions). Only populated for rows generated this run.
    n_truncated = sum(1 for r in records if r.get("finish_reason") == "length")
    payload = {
        "data": records,
        "statistics": {
            "model": opts.model, "provider": opts.provider, "cell": args,
            "n": len(records), "n_non_canonical": n_bad,
            "n_empty": sum(1 for r in records if not r["generated_text"]),
            "n_truncated": n_truncated, "max_tokens": opts.max_tokens,
        },
    }
    with open(opts.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    logger.info("wrote %s | n=%d non_canonical=%d empty=%d truncated=%d (max_tokens=%d)",
                opts.output, len(records), n_bad, payload["statistics"]["n_empty"],
                n_truncated, opts.max_tokens)


if __name__ == "__main__":
    main()
