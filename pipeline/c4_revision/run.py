"""CLI: revise a baseline prediction file, then eval the result.

Reads a baseline prediction JSON, runs execute-first repair on every
`generated_text`, and writes a new prediction JSON. Feed the output to
`pipeline.c1_eval.runner --compute_execution` to measure the EX delta.

Stage-2 of the C4 pipeline. Run Stage-1 (hygiene + truncation regen,
`pipeline.c4_revision.run_stage1`) FIRST, then feed its output here.

Example (parallel, remote + local, DeepSeek non-thinking, canonical chain):
    python -m pipeline.c4_revision.run \
        --input_file <stage1_out>.json \
        --lang english --split test \
        --model deepseek-v4-flash --provider deepseek --thinking disabled \
        --checkers syntax,geo \
        --overpass_urls http://localhost:12346/api/interpreter,http://REMOTE_OVERPASS_HOST:12347/api/interpreter \
        --max_workers 16
"""
from __future__ import annotations

import argparse
import logging
import os

from .checkers import get_checker
from .executor import DEFAULT_OVERPASS_URL, RealOverpassExecutor
from .llm import OpenAICompatLLM
from .reviser import Reviser, revise_prediction_file

logger = logging.getLogger(__name__)


def _set_no_proxy(urls: list[str]) -> None:
    """Add non-local Overpass hosts (e.g. remote REMOTE_OVERPASS_HOST) to NO_PROXY so
    their requests bypass any HTTP(S) proxy, matching the eval runner's recipe."""
    from urllib.parse import urlparse
    hosts = {urlparse(u).hostname for u in urls if urlparse(u).hostname
             and urlparse(u).hostname not in ("localhost", "127.0.0.1")}
    if not hosts:
        return
    for var in ("NO_PROXY", "no_proxy"):
        existing = {h.strip() for h in os.environ.get(var, "").split(",") if h.strip()}
        os.environ[var] = ",".join(sorted(existing | hosts))
    logger.info("NO_PROXY set for: %s", ",".join(sorted(hosts)))


def main() -> None:
    p = argparse.ArgumentParser(description="MOsmNL Revision — repair baseline predictions")
    p.add_argument("--input_file", required=True, help="Baseline prediction JSON.")
    p.add_argument("--output_file", default=None,
                   help="Defaults to <input_dir>/revised/<input_name>.json")
    p.add_argument("--lang", default="english")
    p.add_argument("--split", default="dev", choices=["dev", "test"])
    # LLM (repair) config
    p.add_argument("--model", required=True, help="Repair model name.")
    p.add_argument("--provider", default="aigcbest", choices=["aigcbest", "deepseek"])
    p.add_argument("--temperature", type=float, default=0.7,
                   help="Repair temperature (>0 so multiple attempts differ).")
    p.add_argument("--max_tokens", type=int, default=4096)
    p.add_argument("--thinking", default="auto",
                   choices=["auto", "disabled", "enabled"],
                   help="DeepSeek reasoning toggle via extra_body. 'auto' = no "
                        "override (model default). 'disabled' = non-thinking "
                        "generator (matches the baseline/CoT-isolation runs).")
    p.add_argument("--num_keys", type=int, default=0,
                   help="Cap on API keys used for repair fan-out (0 = all).")
    p.add_argument("--max_repair_attempts", type=int, default=3)
    p.add_argument("--checkers", default="syntax,geo",
                   help="Comma-separated checker names, applied in order. Default "
                        "is the P1-canonical execute-first chain syntax,geo. "
                        "Output hygiene + truncation regen run FIRST as Stage-1 "
                        "(pipeline.c4_revision.run_stage1); TagChecker dormant.")
    # Tag Base (only used when the `tags` checker is enabled)
    p.add_argument("--tag_kb_dir", default=None,
                   help="Tag Base dir (default: data/tag_kb).")
    p.add_argument("--tag_index_dir", default=None,
                   help="Tag embedding index dir (default: data/tag_index).")
    p.add_argument("--tag_encoder", default=None,
                   help="Tag retriever encoder name (default: mpnet).")
    p.add_argument("--tag_topk", type=int, default=12,
                   help="KB candidates retrieved per repair (default: 12).")
    p.add_argument("--tag_device", default=None,
                   help="Encoder device for the tag retriever (e.g. cuda:0).")
    # Execution backend
    p.add_argument("--overpass_urls", default=DEFAULT_OVERPASS_URL,
                   help="Comma-separated Overpass endpoints, e.g. "
                        "'http://localhost:12346/api/interpreter,"
                        "http://REMOTE_OVERPASS_HOST:12347/api/interpreter'. Slots are "
                        "round-robined over these up to --max_workers.")
    p.add_argument("--max_workers", type=int, default=1,
                   help="Parallel revision slots (one Overpass executor each, "
                        "round-robined over --overpass_urls). 1 = sequential.")
    p.add_argument("--cache_dir", default=None)
    p.add_argument("--timeout", type=int, default=300)
    opts = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="[%(asctime)s] %(levelname)s %(message)s")

    if opts.output_file is None:
        in_dir = os.path.dirname(os.path.abspath(opts.input_file))
        in_name = os.path.basename(opts.input_file)
        opts.output_file = os.path.join(in_dir, "revised", in_name)

    # Endpoints: round-robin the distinct URLs up to --max_workers slots, one
    # RealOverpassExecutor per slot. The Overpass sqlite cache is NOT thread
    # safe, so each slot gets its own executor and exactly one item is in-flight
    # per slot (mirrors pipeline.c1_eval.runner). Set NO_PROXY for any non-local
    # host (e.g. remote) or its requests route through a proxy and fail.
    urls = [u.strip() for u in opts.overpass_urls.split(",") if u.strip()]
    if not urls:
        urls = [DEFAULT_OVERPASS_URL]
    _set_no_proxy(urls)
    n_slots = max(1, opts.max_workers)
    slot_urls = [urls[i % len(urls)] for i in range(n_slots)]
    logger.info("revision endpoints (%d slot(s)): %s", n_slots, slot_urls)
    executors = [
        RealOverpassExecutor(url=u, split=opts.split,
                             cache_dir=opts.cache_dir, timeout=opts.timeout)
        for u in slot_urls
    ]

    # One shared LLM (thread-safe usage accounting; ask_n fans out per key).
    llm = OpenAICompatLLM(model_name=opts.model, provider=opts.provider,
                          temperature=opts.temperature, max_tokens=opts.max_tokens,
                          num_keys=opts.num_keys, thinking=opts.thinking)
    checker_names = [c.strip() for c in opts.checkers.split(",") if c.strip()]
    # Build the shared tag retriever once (lazy: only when `tags` is enabled).
    tag_retriever = None
    if "tags" in checker_names:
        from .tag_kb import (DEFAULT_ENCODER, DEFAULT_INDEX_DIR, DEFAULT_KB_DIR,
                             TagRetriever)
        tag_retriever = TagRetriever(
            kb_dir=opts.tag_kb_dir or DEFAULT_KB_DIR,
            index_dir=opts.tag_index_dir or DEFAULT_INDEX_DIR,
            encoder_name=opts.tag_encoder or DEFAULT_ENCODER,
            device=opts.tag_device)

    # One Reviser per executor: each owns its checkers bound to its endpoint,
    # sharing the single LLM and (read-only) tag retriever.
    def _build_reviser(executor: RealOverpassExecutor) -> Reviser:
        checkers = []
        for name in checker_names:
            kwargs = dict(llm=llm, executor=executor,
                          max_repair_attempts=opts.max_repair_attempts)
            if name == "tags":
                kwargs.update(retriever=tag_retriever, top_k=opts.tag_topk)
            checkers.append(get_checker(name)(**kwargs))
        return Reviser(checkers=checkers, executor=executor)

    revisers = [_build_reviser(ex) for ex in executors]

    stats = revise_prediction_file(opts.input_file, opts.output_file,
                                   lang=opts.lang, split=opts.split,
                                   revisers=revisers)
    for ex in executors:
        ex.save_cache()

    logger.info("revision stats: %s", stats)
    logger.info("Next: python -m pipeline.c1_eval.runner --input_file %s "
                "--lang %s --split %s --compute_execution",
                opts.output_file, opts.lang, opts.split)


if __name__ == "__main__":
    main()
