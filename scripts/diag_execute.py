"""Diagnostic step D3 — execute the candidate pool and capture denotations + EX.

For each query (gold, greedy, K candidates) this:
  * gets the **correctness label** from the project's canonical path
    (``evaluate_execution_single`` + ``apply_empty_match_policy``) — NOT a
    re-implementation, so EX matches ``src/eval/runner.py`` exactly;
  * gets the **denotation signature** from the raw ``op.query(prepare_query(...))``
    result, which is a cache hit after the canonical call ran the same prepared
    query — used only for grouping (voting / strata).

Output: ``data/diag/executed.json`` in the shape ``src/diag/analyze.py`` consumes
(``gold_num``, ``gold_has_error``, ``greedy_ex``, ``candidates=[{sig, ex}]``),
plus debugging fields. Run ``scripts/diag_analyze.py`` next.

Needs the frozen Overpass docker + the unified eval cache (see README.md).

Usage (from repo root):
    python scripts/diag_execute.py --limit 5     # pilot: first 5 queries
    python scripts/diag_execute.py               # full pool
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import tqdm

# Put final/ on the path so the consolidated `pipeline` package resolves.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pipeline.c3_selection.denotation import ERROR, signature, to_jsonable  # noqa: E402

# Canonical eval path — single source of truth for EX + invariants. Under the
# unified `pipeline` package there is no longer a two-`src` collision, so we
# import the runner cleanly (it inserts eval_backend onto sys.path on import).
from pipeline.c1_eval import runner as _runner  # noqa: E402
Overpass = _runner.Overpass
apply_empty_match_policy = _runner.apply_empty_match_policy
DEFAULT_CACHE_DIR = _runner.DEFAULT_CACHE_DIR
DEFAULT_OVERPASS_URLS = _runner.DEFAULT_OVERPASS_URLS
DEFAULT_NOMINATIM_URL = _runner.DEFAULT_NOMINATIM_URL
DEFAULT_CONVERT_URL = _runner.DEFAULT_CONVERT_URL
TIMEOUT_SEC = _runner.TIMEOUT_SEC

sys.path.insert(0, os.environ.get("EVAL_BACKEND_PATH", "/path/to/eval_backend"))
from utils.evaluation import Nominatim, load_error_cache  # noqa: E402
from utils.eval_utils import evaluate_execution_single, prepare_query  # noqa: E402

logger = logging.getLogger("diag_execute")


def _denote(op_instances, nominatim, query_text, bbox):
    """Return (jsonable_signature, num, has_error) for a single query string.

    Hits the cache when the canonical path already executed the same prepared
    query. A missing/non-canonical query is treated as an (unusable) error.
    """
    if not query_text or not query_text.lstrip().startswith("["):
        return to_jsonable(ERROR), -1, True
    prepared, err = prepare_query(query_text, bbox=bbox, timeout=TIMEOUT_SEC, nominatim=nominatim)
    if err:
        return to_jsonable(ERROR), -1, True
    # Two passes over both endpoints: a single transient (timeout/connection) on
    # a heavy query must not be mistaken for a genuine error. After the canonical
    # evaluate() has run the same prepared query, this is a cache hit anyway.
    last = None
    for _ in range(2):
        for op in op_instances:
            try:
                results, num = op.query(prepared)
                num = int(num)
                if num < 0:
                    # `op.query` caches timeouts as ('server_error_timeout', -1);
                    # a spurious cached timeout (e.g. from an overloaded docker)
                    # must not be read back as a real error. Refresh once.
                    results, num = op.query(prepared, use_cache=False)
                    num = int(num)
                return to_jsonable(signature(results, num)), num, (num < 0)
            except requests.ConnectionError as e:
                last = f"ConnectionError:{e}"
                continue
            except Exception as e:  # noqa: BLE001
                last = f"{type(e).__name__}:{e}"
                continue
    logger.warning("_denote fell through: head=%r last=%s", prepared[:80], str(last)[:160])
    return to_jsonable(ERROR), -1, True


def main() -> None:
    p = argparse.ArgumentParser(description="Execute candidate pool; capture denotation + EX")
    p.add_argument("--input", default="data/diag/candidates.json")
    p.add_argument("--output", default="data/diag/executed.json")
    p.add_argument("--split", default="dev")
    p.add_argument("--workers", type=int, default=8, help="concurrent queries (each ~K+2 execs)")
    p.add_argument("--cache_dir", default=DEFAULT_CACHE_DIR)
    p.add_argument("--overpass_urls", default=DEFAULT_OVERPASS_URLS)
    p.add_argument("--nominatim_url", default=DEFAULT_NOMINATIM_URL)
    p.add_argument("--limit", type=int, default=0, help="cap queries (pilot)")
    p.add_argument("--retry_errors", action="store_true")
    opts = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")

    with open(opts.input, encoding="utf-8") as f:
        payload = json.load(f)
    records = payload["records"]
    if opts.limit > 0:
        records = records[: opts.limit]
    logger.info("loaded %d query records from %s", len(records), opts.input)

    urls = [u.strip() for u in opts.overpass_urls.split(",") if u.strip()]
    op_instances = [
        Overpass(u, cache_dir=opts.cache_dir,
                 cache_filename=f"overpass_{opts.split}_cache", save_frequency=100)
        for u in urls
    ]
    nominatim = Nominatim(opts.nominatim_url, cache_dir=opts.cache_dir, save_frequency=100)
    error_cache = load_error_cache(opts.cache_dir)

    def _ex(gold, pred, bbox, idx):
        """Canonical EX label (with empty-match policy) for one (gold,pred)."""
        if not pred or not pred.lstrip().startswith("["):
            return 0, True, "missing_or_noncanonical_pred"
        try:
            res = evaluate_execution_single(
                gold, pred, bbox, False, error_cache, idx,
                TIMEOUT_SEC, opts.retry_errors, op_instances, nominatim,
                include_error_tracking=True)
            res = apply_empty_match_policy(res)
            return int(res["EX"]), bool(res.get("has_error", False)), res.get("info", "")
        except Exception as e:  # noqa: BLE001
            return 0, True, f"exc:{e}"

    def process(idx_rec):
        idx, rec = idx_rec
        gold = rec["OverpassQL"]
        bbox = rec.get("bbox", "")
        # greedy (Pass@1) — also robustly executes gold (multi-attempt) and caches it.
        greedy_ex, _, greedy_info = _ex(gold, rec.get("greedy", ""), bbox, idx)
        # K candidates: ex via canonical path + denotation signature
        cand_out = []
        for c in rec.get("candidates", []):
            ex, has_err, info = _ex(gold, c, bbox, idx)
            sig, num, sig_err = _denote(op_instances, nominatim, c, bbox)
            cand_out.append({"sig": sig, "ex": ex, "num": num,
                             "has_error": has_err or sig_err, "info": info})
        # gold denotation LAST: the evaluate() calls above hit gold once per
        # candidate, so by now a transient gold timeout has been retried away and
        # the success is cached. Computing it first (cold cache) mis-marked heavy
        # golds as S0 errors while their candidates scored ex==1.
        gold_sig, gold_num, gold_err = _denote(op_instances, nominatim, gold, bbox)
        return {
            "query_index": rec["query_index"],
            "gold_num": gold_num,
            "gold_has_error": gold_err,
            "gold_sig": gold_sig,
            "greedy_ex": greedy_ex,
            "greedy_info": greedy_info,
            "candidates": cand_out,
        }

    out = []
    with ThreadPoolExecutor(max_workers=opts.workers) as pool:
        futs = [pool.submit(process, (i, r)) for i, r in enumerate(records)]
        for fut in tqdm.tqdm(as_completed(futs), total=len(futs), desc="execute"):
            out.append(fut.result())

    for op in op_instances:
        op.save_cache()
        op.close()
    nominatim.save_cache()

    out.sort(key=lambda r: r["query_index"])
    result = {"args": payload.get("args", {}), "split": opts.split, "records": out}
    os.makedirs(os.path.dirname(opts.output), exist_ok=True)
    with open(opts.output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    logger.info("wrote %s | %d queries executed", opts.output, len(out))


if __name__ == "__main__":
    main()
