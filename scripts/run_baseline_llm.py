"""
Minimal baseline inference: NL -> OverpassQL via an AIGCBest-hosted LLM.

Produces a JSON file that `src/eval/runner.py` can consume directly:
    {"data": [{"query_index", "query_nl", "query_target_ovq",
               "generated_text", "bbox"}, ...],
     "statistics": {...}}

This is intentionally lean — zero-shot, single key, sequential by default.
Used to smoke-test the end-to-end MOsmNL pipeline before scaling out.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import numpy as np
from openai import OpenAI

logger = logging.getLogger(__name__)

# Provider routing: import keys from the single source of truth in eval_backend
# (avoids duplicating secrets across repos per security.md).
AIGCBEST_BASE_URL = "https://api2.aigcbest.top/v1"
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"

PROVIDER_BASE_URL = {
    "aigcbest": AIGCBEST_BASE_URL,
    "deepseek": DEEPSEEK_BASE_URL,
}


def _load_provider_keys(provider: str) -> List[str]:
    sys.path.insert(0, os.environ.get("EVAL_BACKEND_PATH", "/path/to/eval_backend"))
    try:
        from train_influence.config import API_KEYS  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "Could not import API_KEYS from eval_backend/train_influence/config.py."
        ) from e
    keys = [k for k in API_KEYS.get(provider, []) if k]
    if not keys:
        raise RuntimeError(f"No keys available in API_KEYS['{provider}'].")
    return keys

SYSTEM_PROMPT = (
    "You are an expert in OpenStreetMap and the Overpass Query Language (OverpassQL). "
    "Given a natural-language question about OSM data, output ONLY the OverpassQL "
    "query that answers it. Do not include any explanation, markdown, or commentary. "
    "The query MUST start with '[out:' and be directly executable on an Overpass API "
    "endpoint."
)

USER_TEMPLATE = "Question: {nl}\n\nOverpassQL:"

# Few-shot template (k>0). Each demo: "Example {i}\nQuestion: ...\nOverpassQL: ...\n"
# Demonstrations are retrieved from the same-language train+syn pool (MultiTEND §C).
FEWSHOT_DEMO_TEMPLATE = "Example {i}\nQuestion: {nl}\nOverpassQL: {ovq}\n"
FEWSHOT_TAIL_TEMPLATE = "\nNow answer:\nQuestion: {nl}\n\nOverpassQL:"


def load_icl_index(icl_index_dir: str, lang: str) -> Tuple[np.ndarray, List[Dict]]:
    """Load mpnet embeddings + key records for a language's train+syn pool."""
    emb_path = os.path.join(icl_index_dir, f"{lang}.npy")
    keys_path = os.path.join(icl_index_dir, f"{lang}_keys.json")
    if not (os.path.exists(emb_path) and os.path.exists(keys_path)):
        raise FileNotFoundError(
            f"ICL index missing for lang={lang}: expected {emb_path} and {keys_path}. "
            f"Run scripts/build_icl_index.py first."
        )
    emb = np.load(emb_path)  # shape (N, D), L2-normalized
    with open(keys_path, "r", encoding="utf-8") as f:
        keys = json.load(f)
    if emb.shape[0] != len(keys):
        raise ValueError(
            f"ICL index size mismatch: emb has {emb.shape[0]} rows but keys has {len(keys)}"
        )
    return emb, keys


def retrieve_topk(query_vec: np.ndarray, pool_emb: np.ndarray,
                  pool_keys: List[Dict], k: int) -> List[Dict]:
    """Return top-k pool entries by cosine similarity, deduped by gold OverpassQL prefix."""
    # pool_emb and query_vec assumed L2-normalized -> dot product == cosine sim.
    sims = pool_emb @ query_vec
    order = np.argsort(-sims)
    seen_prefix = set()
    out: List[Dict] = []
    for idx in order:
        rec = pool_keys[int(idx)]
        ovq = rec.get("OverpassQL", "")
        prefix = ovq[:80]
        if prefix in seen_prefix:
            continue
        seen_prefix.add(prefix)
        out.append(rec)
        if len(out) >= k:
            break
    return out


def build_fewshot_user_message(demos: List[Dict], test_nl: str) -> str:
    parts = []
    for i, d in enumerate(demos, start=1):
        parts.append(FEWSHOT_DEMO_TEMPLATE.format(
            i=i, nl=d.get("nl_question", ""), ovq=d.get("OverpassQL", "")))
    parts.append(FEWSHOT_TAIL_TEMPLATE.format(nl=test_nl))
    return "\n".join(parts)


_FENCE_RE = re.compile(r"```(?:overpass|overpassql|ql)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_query(raw: str) -> str:
    """Strip code fences / leading prose; return string starting with '[out:'.

    Models sometimes wrap output in ```...``` even when told not to.
    """
    if not raw:
        return ""
    m = _FENCE_RE.search(raw)
    if m:
        raw = m.group(1)
    raw = raw.strip()
    # If model prefixed something like "OverpassQL:\n", drop it.
    idx = raw.find("[out:")
    if idx > 0:
        raw = raw[idx:]
    return raw.strip()


def call_llm(client: OpenAI, model: str, user_message: str, max_tokens: int,
             temperature: float, max_retries: int = 3) -> str:
    last_err: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:  # noqa: BLE001 (LLM SDK throws many subclasses)
            last_err = e
            wait = 2 ** attempt
            logger.warning(f"LLM call failed (attempt {attempt+1}/{max_retries}): "
                           f"{type(e).__name__}: {e}; retry in {wait}s")
            time.sleep(wait)
    logger.error(f"LLM call gave up after {max_retries} attempts: {last_err}")
    return ""


def build_record(idx: int, item: Dict, generated: str, user_message: str = "") -> Dict:
    return {
        "query_index": item.get("sample_index", idx),
        "query_nl": item.get("nl_question", ""),
        "query_target_ovq": item.get("OverpassQL", ""),
        "generated_text": generated,
        "bbox": (item.get("location", {}) or {}).get("bbox", "") or "",
        # Exact user message sent to the model (system prompt is constant). Saved
        # so a revision/regeneration step can reuse the prompt verbatim instead of
        # re-running ICL retrieval (no encoder/index needed downstream).
        "user_message": user_message,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lang", default="english",
                   help="Language directory under dataset/OsmNL_<lang>/.")
    p.add_argument("--split", default="dev", choices=["dev", "test", "train", "syn"])
    p.add_argument("--limit", type=int, default=10,
                   help="Number of samples (from the start). 0 = all.")
    p.add_argument("--model", default="gemini-3-flash-preview",
                   help="Model name as accepted by AIGCBest /v1.")
    p.add_argument("--max_tokens", type=int, default=4096)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--workers", type=int, default=4,
                   help="Thread pool size for concurrent LLM calls.")
    p.add_argument("--output", default=None,
                   help="Output JSON path. Auto-named under outputs/predictions/.")
    p.add_argument("--key_index", type=int, default=0,
                   help="Index of the first provider key to use.")
    p.add_argument("--num_keys", type=int, default=1,
                   help="Number of provider keys to round-robin across worker "
                        "threads (starting at --key_index). >1 lets a single "
                        "process saturate the key pool, e.g. workers=128 + "
                        "num_keys=8 -> ~16 concurrent calls per key.")
    p.add_argument("--provider", default="aigcbest",
                   choices=sorted(PROVIDER_BASE_URL.keys()),
                   help="LLM provider routing (aigcbest|deepseek).")
    p.add_argument("--dataset_root",
                   default="/path/to/XOver/dataset",
                   help="Root that holds OsmNL_<lang>/ folders.")
    p.add_argument("--shots", type=int, default=0,
                   help="Number of few-shot demonstrations to prepend (0 = zero-shot).")
    p.add_argument("--icl_index_dir",
                   default="/path/to/XOver/data/icl_index",
                   help="Directory holding {lang}.npy + {lang}_keys.json built by "
                        "scripts/build_icl_index.py.")
    p.add_argument("--icl_pool_lang", default=None,
                   help="Override the ICL pool language. Default = --lang "
                        "(same-language pool, MultiTEND §C). Set to 'english' to "
                        "test the deployment-realistic English-only pool baseline.")
    p.add_argument("--mpnet_model",
                   default="paraphrase-multilingual-mpnet-base-v2",
                   help="sBERT model name for encoding test NLs (must match index build).")
    p.add_argument("--query_instruction", default=None,
                   help="Task description for instruction-tuned retrievers (e.g. "
                        "Qwen3-Embedding). When set, test NLs are encoded with the "
                        "Qwen3-style prompt 'Instruct: {instruction}\\nQuery:'. The "
                        "pool/document side is never instructed. Leave unset for "
                        "symmetric encoders like mpnet.")
    p.add_argument("--encoder_device", default=None,
                   help="Device for the sBERT encoder. Default: SentenceTransformer "
                        "auto-pick (cuda if available). Pass 'cpu' to avoid OOM when "
                        "many cells run in parallel.")
    opts = p.parse_args()
    if opts.icl_pool_lang is None:
        opts.icl_pool_lang = opts.lang

    logging.basicConfig(level=logging.INFO,
                        format="[%(asctime)s] %(levelname)s %(message)s")

    in_path = os.path.join(opts.dataset_root, f"OsmNL_{opts.lang}",
                           f"{opts.split}_final_{opts.lang}.json")
    if not os.path.exists(in_path):
        sys.exit(f"input not found: {in_path}")
    with open(in_path, "r", encoding="utf-8") as f:
        items: List[Dict] = json.load(f)
    if opts.limit and opts.limit > 0:
        items = items[: opts.limit]
    logger.info(f"loaded {len(items)} samples from {in_path}")

    if opts.output is None:
        out_dir = "/path/to/XOver/outputs/predictions"
        os.makedirs(out_dir, exist_ok=True)
        shot_tag = f"{opts.shots}shot"
        tag = (f"{opts.lang}_{opts.split}_{opts.model.replace('/', '_')}"
               f"_n{len(items)}_{shot_tag}")
        opts.output = os.path.join(out_dir, f"{tag}.json")

    # Load ICL index + encoder only if few-shot requested.
    pool_emb: Optional[np.ndarray] = None
    pool_keys: Optional[List[Dict]] = None
    encoder = None
    if opts.shots > 0:
        from sentence_transformers import SentenceTransformer  # heavy import
        pool_emb, pool_keys = load_icl_index(opts.icl_index_dir, opts.icl_pool_lang)
        logger.info(f"loaded ICL index for pool_lang={opts.icl_pool_lang} "
                    f"(test lang={opts.lang}): pool size={len(pool_keys)}, "
                    f"dim={pool_emb.shape[1]}")
        encoder = SentenceTransformer(opts.mpnet_model, device=opts.encoder_device)
        # Encode all test NLs once, batched, then retrieve per-row.
        nls = [it.get("nl_question", "") for it in items]
        encode_kwargs = dict(batch_size=64, normalize_embeddings=True,
                             show_progress_bar=False, convert_to_numpy=True)
        if opts.query_instruction:
            # Qwen3-Embedding asymmetric usage: query side gets an instruction,
            # the document/pool side does not (index was built prompt-free).
            encode_kwargs["prompt"] = f"Instruct: {opts.query_instruction}\nQuery:"
            logger.info(f"encoding {len(nls)} test NLs with {opts.mpnet_model} "
                        f"(query instruction enabled)")
        else:
            logger.info(f"encoding {len(nls)} test NLs with {opts.mpnet_model}")
        query_embs = encoder.encode(nls, **encode_kwargs)
    else:
        query_embs = None  # type: ignore[assignment]

    keys = _load_provider_keys(opts.provider)
    base_url = PROVIDER_BASE_URL[opts.provider]
    # Spread the worker threads across num_keys keys (round-robin by sample index)
    # so one process can saturate the full key pool. num_keys=1 -> single-key
    # behaviour (key chosen by key_index), matching the old per-cell model.
    n_keys = max(1, min(opts.num_keys, len(keys)))
    clients = [
        OpenAI(base_url=base_url,
               api_key=keys[(opts.key_index + j) % len(keys)])
        for j in range(n_keys)
    ]
    logger.info(f"provider={opts.provider} base_url={base_url} model={opts.model} "
                f"shots={opts.shots} pool_lang={opts.icl_pool_lang} "
                f"workers={opts.workers} num_keys={n_keys} temp={opts.temperature} "
                f"max_tokens={opts.max_tokens} -> {opts.output}")

    results: List[Optional[Dict]] = [None] * len(items)

    def work(i: int) -> int:
        item = items[i]
        nl = item.get("nl_question", "")
        if opts.shots > 0 and pool_emb is not None and pool_keys is not None:
            demos = retrieve_topk(query_embs[i], pool_emb, pool_keys, opts.shots)
            user_msg = build_fewshot_user_message(demos, nl)
        else:
            user_msg = USER_TEMPLATE.format(nl=nl)
        client = clients[i % n_keys]
        raw = call_llm(client, opts.model, user_msg, opts.max_tokens, opts.temperature)
        gen = extract_query(raw)
        if gen and not gen.startswith("[out:"):
            logger.warning(f"sample {i}: generated_text does not start with '[out:': "
                           f"{gen[:80]!r}")
        results[i] = build_record(i, item, gen, user_message=user_msg)
        return i

    t0 = time.time()
    n_done = 0
    with ThreadPoolExecutor(max_workers=max(1, opts.workers)) as pool:
        futs = [pool.submit(work, i) for i in range(len(items))]
        for fut in as_completed(futs):
            i = fut.result()
            n_done += 1
            if n_done % 10 == 0 or n_done == len(items):
                logger.info(f"progress {n_done}/{len(items)}")
    dt = time.time() - t0

    n_empty = sum(1 for r in results if not r or not r["generated_text"])
    n_canonical = sum(1 for r in results if r and r["generated_text"].startswith("[out:"))
    logger.info(f"done in {dt:.1f}s; empty={n_empty}, starts-with-[out:={n_canonical}")

    out_payload = {
        "data": results,
        "statistics": {
            "total_queries": len(results),
            "lang": opts.lang,
            "split": opts.split,
            "model": opts.model,
            "provider": opts.provider,
            "shots": opts.shots,
            "icl_pool_lang": opts.icl_pool_lang,
            "icl_index_dir": opts.icl_index_dir,
            "mpnet_model": opts.mpnet_model,
            "query_instruction": opts.query_instruction,
            "encoder_device": opts.encoder_device,
            "temperature": opts.temperature,
            "max_tokens": opts.max_tokens,
            "workers": opts.workers,
            "elapsed_sec": round(dt, 2),
            "empty_predictions": n_empty,
            "canonical_predictions": n_canonical,
        },
    }
    with open(opts.output, "w", encoding="utf-8") as f:
        json.dump(out_payload, f, ensure_ascii=False, indent=2)
    logger.info(f"wrote {opts.output}")


if __name__ == "__main__":
    main()
