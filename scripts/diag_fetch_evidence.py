"""Pre-pass: fetch matched-tag evidence for contested top-k group reps.

For the confidence judge's richer-evidence ablation. Walks ``executed.json``,
finds CONTESTED queries (top-1 denotation consistency < threshold), takes the
top-k group representatives, and for each runs the *tag-variant* of its prepared
query against Overpass to collect what it actually matched. Results are cached
to a JSON keyed ``"<query_index>:<candidate_index>"`` so the (LLM) judge pass
reads evidence with no Overpass.

Overpass is the bottleneck (can't take high local concurrency), so this runs at
low concurrency over both endpoints with a client wait cap; the judge pass then
fans out over the LLM key pool separately.

Usage (from final/):
    python scripts/diag_fetch_evidence.py \
        --executed data/diag_p1_en/executed.json \
        --candidates data/diag_p1_en/candidates.json \
        --lang english --split dev --k 2 --threshold 0.8 \
        --out data/diag_p1_en/evidence_en.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.environ.get("EVAL_BACKEND_PATH", "/path/to/eval_backend"))

from pipeline.c3_selection.confidence import _rank_groups  # noqa: E402
from pipeline.c3_selection.denotation import freeze  # noqa: E402
from pipeline.c3_selection.evidence import aggregate_tags, tag_variant  # noqa: E402

# remote (REMOTE_OVERPASS_HOST:12347) is a second parallel endpoint but is often
# down; default to localhost only and let --endpoints add it back when up.
DEFAULT_ENDPOINTS = "http://localhost:12346/api/interpreter"


def post_overpass(url: str, data: str, timeout: int) -> dict:
    r = requests.post(url, data={"data": data}, timeout=timeout,
                      proxies={"http": None, "https": None})
    r.raise_for_status()
    return r.json()


def main() -> None:
    p = argparse.ArgumentParser(description="Fetch matched-tag evidence")
    p.add_argument("--executed", required=True)
    p.add_argument("--candidates", required=True)
    p.add_argument("--lang", default="english")
    p.add_argument("--split", default="dev")
    p.add_argument("--k", type=int, default=2)
    p.add_argument("--threshold", type=float, default=0.8)
    p.add_argument("--sample", type=int, default=8, help="out tags N")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--timeout", type=int, default=60, help="client wait cap (s)")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--endpoints", default=DEFAULT_ENDPOINTS,
                   help="comma-separated Overpass URLs (round-robin)")
    p.add_argument("--retries", type=int, default=2)
    p.add_argument("--out", required=True)
    opts = p.parse_args()
    endpoints = [u.strip() for u in opts.endpoints.split(",") if u.strip()]

    from pipeline.c1_eval.runner import Nominatim, DEFAULT_CACHE_DIR  # noqa: E402
    from utils.eval_utils import prepare_query  # noqa: E402
    nom = Nominatim("https://nominatim.openstreetmap.org/search.php",
                    cache_dir=DEFAULT_CACHE_DIR)

    ex_recs = json.load(open(opts.executed))["records"]
    craw = json.load(open(opts.candidates))
    crecs = craw["records"] if isinstance(craw, dict) and "records" in craw else craw
    txt_by_qi = {r["query_index"]: r for r in crecs}
    if opts.limit:
        ex_recs = ex_recs[:opts.limit]

    # Build the work list: (qi, cand_idx, query_text, bbox) for contested top-k.
    jobs: List[tuple] = []
    n_contested = 0
    for rec in ex_recs:
        qi = rec["query_index"]
        cands = [{"sig": freeze(c["sig"]), "ex": int(c["ex"])}
                 for c in rec["candidates"]]
        reps = _rank_groups(cands, prefer_nonempty=True)
        if len(reps) < 2 or reps[0].consistency >= opts.threshold:
            continue
        n_contested += 1
        ctext = txt_by_qi[qi]
        for rep in reps[:opts.k]:
            jobs.append((qi, rep.index, ctext["candidates"][rep.index],
                         ctext.get("bbox", "") or ""))

    # Resume: keep already-cached entries.
    cache: Dict[str, dict] = {}
    if os.path.exists(opts.out):
        try:
            cache = json.load(open(opts.out))
        except Exception:
            cache = {}
    todo = [j for j in jobs if f"{j[0]}:{j[1]}" not in cache]
    print(f"contested queries: {n_contested} | reps to fetch: {len(jobs)} "
          f"| cached: {len(jobs)-len(todo)} | todo: {len(todo)}")

    lock = threading.Lock()
    counter = {"done": 0, "ok": 0, "err": 0}

    def work(job, wid):
        qi, idx, qtext, bbox = job
        key = f"{qi}:{idx}"
        url = endpoints[wid % len(endpoints)]
        rec = {"qi": qi, "idx": idx}
        try:
            prepared, err = prepare_query(qtext, bbox=bbox, timeout=300,
                                          nominatim=nom)
            if err is not None or not prepared:
                rec["error"] = f"prepare: {str(err)[:80]}"
            else:
                tv = tag_variant(prepared, opts.sample)
                data = None
                last = None
                for attempt in range(opts.retries + 1):
                    try:
                        data = post_overpass(url, tv, opts.timeout)
                        break
                    except Exception as e:  # noqa: BLE001
                        last = e
                if data is None:
                    raise last
                els = data.get("elements", [])
                freq = aggregate_tags(els)
                rec["n_sampled"] = len(els)
                rec["tag_freq"] = dict(freq.most_common(20))
                rec["predicate_query"] = qtext
        except Exception as e:  # noqa: BLE001
            rec["error"] = str(e)[:120]
        with lock:
            cache[key] = rec
            counter["done"] += 1
            counter["ok" if "error" not in rec else "err"] += 1
            if counter["done"] % 50 == 0:
                json.dump(cache, open(opts.out, "w"), ensure_ascii=False)
                print(f"  ... {counter['done']}/{len(todo)} "
                      f"(ok={counter['ok']} err={counter['err']})", flush=True)
        return key

    with ThreadPoolExecutor(max_workers=opts.workers) as pool:
        futs = [pool.submit(work, j, i) for i, j in enumerate(todo)]
        for _ in as_completed(futs):
            pass
    json.dump(cache, open(opts.out, "w"), ensure_ascii=False, indent=0)
    try:
        nom.save_cache()
    except Exception:
        pass
    print(f"wrote {opts.out} | total cached={len(cache)} "
          f"ok={counter['ok']} err={counter['err']}")


if __name__ == "__main__":
    main()
