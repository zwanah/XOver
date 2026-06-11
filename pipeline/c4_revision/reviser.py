"""Core revision loop + prediction-file driver.

`Reviser` runs a candidate query through a checker chain (execute-first repair).
`revise_prediction_file` ingests a baseline prediction JSON, repairs every
`generated_text`, and writes a new prediction JSON in the *same schema* so
`pipeline.c1_eval.runner --compute_execution` runs on it unchanged. Each revised item
also carries `original_generated_text` and a `revision` metadata block for A/B
analysis (the runner ignores unknown fields).
"""
from __future__ import annotations

import json
import logging
import os
import queue
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .checkers import BaseChecker
from .executor import ExecResult, OQLExecutor

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RevisionResult:
    """Outcome of revising one query."""

    query: str
    result: ExecResult
    repaired: bool
    attempts: int
    error_before: Optional[str]


class Reviser:
    """Runs a query through the checker chain. The chain is applied in order;
    each checker may repair the query before the next sees it."""

    def __init__(self, checkers: List[BaseChecker], executor: OQLExecutor) -> None:
        self.checkers = checkers
        self.executor = executor

    def revise(self, query: str, nl_question: str, bbox: str) -> RevisionResult:
        before = self.executor.run(query, bbox)
        cur = query
        last = before
        total_attempts = 0
        for chk in self.checkers:
            cur, last, attempts = chk.check_and_revise(cur, nl_question, bbox)
            total_attempts += attempts
        return RevisionResult(
            query=cur,
            result=last,
            repaired=(cur != query),
            attempts=total_attempts,
            error_before=before.error,
        )


def _load_predictions(path: str) -> tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    with open(path, "r", encoding="utf-8") as f:
        file_data = json.load(f)
    if isinstance(file_data, dict) and "data" in file_data:
        return file_data["data"], file_data.get("statistics")
    if isinstance(file_data, list):
        return file_data, None
    raise SystemExit(f"unknown input format: {type(file_data)}")


def _qidx(item: Dict[str, Any]) -> Optional[int]:
    """Normalize `query_index` to int.

    Guards the silent str/int key mismatch: query_index is often a string in
    prediction files while the dataset keys bbox_map by int sample_index.
    """
    raw = item.get("query_index")
    try:
        return int(str(raw))
    except (TypeError, ValueError):
        return None


def _revise_one(reviser: Reviser, item: Dict[str, Any],
                bbox_map: Dict[int, str]) -> Tuple[Dict[str, Any], bool, int, bool, bool]:
    """Revise a single prediction item, returning the new item plus the four
    per-item counters (repaired, attempts, exec_ok_before, exec_ok_after)."""
    qidx = _qidx(item)
    bbox = bbox_map.get(qidx, item.get("bbox", "") or "")
    original = item.get("generated_text", "") or ""
    nl = item.get("query_nl", "") or item.get("nl_question", "")
    rev = reviser.revise(original, nl, bbox)

    new = dict(item)
    new["original_generated_text"] = original
    new["generated_text"] = rev.query
    new["revision"] = {
        "repaired": rev.repaired,
        "attempts": rev.attempts,
        "exec_ok_before": rev.error_before is None,
        "exec_ok_after": rev.result.ok,
        "error_before": rev.error_before,
    }
    return (new, rev.repaired, rev.attempts,
            rev.error_before is None, rev.result.ok)


def _process_pool(data: List[Dict[str, Any]], bbox_map: Dict[int, str],
                  pool: List[Reviser]) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Revise every item across a pool of revisers. Each reviser owns its own
    (non-thread-safe) Overpass executor, so exactly one item is in-flight per
    reviser: we spawn one worker thread per reviser, each draining a shared
    index queue. Distinct output indices are written without a lock; only the
    aggregate counters and the progress tick are guarded.
    """
    n = len(data)
    out: List[Optional[Dict[str, Any]]] = [None] * n
    work: "queue.Queue[int]" = queue.Queue()
    for i in range(n):
        work.put(i)

    lock = threading.Lock()
    agg = {"repaired": 0, "attempts": 0, "ok_before": 0, "ok_after": 0, "done": 0}

    def worker(rev: Reviser) -> None:
        while True:
            try:
                i = work.get_nowait()
            except queue.Empty:
                return
            new, repaired, attempts, ok_before, ok_after = _revise_one(
                rev, data[i], bbox_map)
            out[i] = new
            with lock:
                agg["repaired"] += int(repaired)
                agg["attempts"] += attempts
                agg["ok_before"] += int(ok_before)
                agg["ok_after"] += int(ok_after)
                agg["done"] += 1
                if agg["done"] % 200 == 0 or agg["done"] == n:
                    logger.info(f"revised {agg['done']}/{n} "
                                f"(repaired {agg['repaired']})")

    threads = [threading.Thread(target=worker, args=(rev,), daemon=True)
               for rev in pool]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return [it for it in out if it is not None], agg


def revise_prediction_file(input_file: str, output_file: str, lang: str,
                           split: str, reviser: Optional[Reviser] = None,
                           revisers: Optional[List[Reviser]] = None) -> Dict[str, Any]:
    """Repair every prediction in `input_file`, write to `output_file`.

    Pass a single `reviser` (sequential) or a `revisers` pool (one per Overpass
    endpoint slot) for parallel revision. bbox is resolved from the MOsmNL
    dataset by `query_index` (normalized to int), reusing the eval runner's
    loader — the single source of truth.
    """
    pool: List[Reviser] = list(revisers) if revisers else ([reviser] if reviser else [])
    if not pool:
        raise SystemExit("revise_prediction_file needs `reviser` or `revisers`.")

    # Reuse the runner's bbox loader so we key off the same dataset.
    from pipeline.c1_eval.runner import load_bbox_from_mosmnl
    bbox_map = load_bbox_from_mosmnl(split, lang)

    data, in_stats = _load_predictions(input_file)
    if not data:
        raise SystemExit("input file has 0 samples")

    hits = sum(1 for it in data if _qidx(it) in bbox_map)
    if hits == 0:
        raise SystemExit(
            f"No prediction query_index matched the {lang}/{split} bbox map "
            f"({len(bbox_map)} entries). Check --lang/--split or the index type."
        )
    logger.info(f"bbox map: {hits}/{len(data)} predictions matched ({lang}/{split})")
    logger.info(f"revising {len(data)} items across {len(pool)} endpoint slot(s)")

    out, agg = _process_pool(data, bbox_map, pool)
    n_repaired, n_attempts = agg["repaired"], agg["attempts"]
    n_ok_before, n_ok_after = agg["ok_before"], agg["ok_after"]

    total = len(out)
    revision_stats = {
        "total": total,
        "exec_ok_before": n_ok_before,
        "exec_ok_after": n_ok_after,
        "repaired": n_repaired,
        "total_repair_attempts": n_attempts,
        "lang": lang,
        "split": split,
        "checkers": [c.name for c in pool[0].checkers],
        "llm_usage": pool[0].checkers[0].llm.get_usage() if pool[0].checkers else {},
    }

    out_data: Dict[str, Any] = {"data": out}
    out_data["statistics"] = dict(in_stats) if in_stats else {}
    out_data["statistics"]["revision"] = revision_stats

    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False, indent=2)

    logger.info(f"revised {total}: ok {n_ok_before} -> {n_ok_after}, "
                f"repaired {n_repaired}, attempts {n_attempts}")
    logger.info(f"wrote: {output_file}")
    return revision_stats
