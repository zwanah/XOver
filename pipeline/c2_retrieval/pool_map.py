"""Reconstruct the 6180-demo pool order and map pool index -> CoT record.

The score vectors are indexed 0..6179 over the concatenated pool, in the order
the TACO embeddings were built: ``[train (6047) + syn (133)]``, file order, from
the **score-source** files (``OsmNL/Final/{train,syn}_final.json``). The CoT
corpus was built from a **different** physical file set
(``OsmNL_english/{train,syn}_final_english.json``) and stores ``(split,
sample_index)``.

This module bridges the two and enforces the spec §3 guard (fail loud, never a
warning): the same ``(split, sample_index)`` must denote the same question in
both file sets, so a score computed on the score-source NL truly corresponds to
the CoT reasoning built on the cot-source NL. The join itself is by
``(split, sample_index)`` (robust to file-order drift); the guard additionally
round-trips the two files positionally to catch any divergence early.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .config import PoolConfig

logger = logging.getLogger(__name__)

# Spec §3: round-trip at >= this many positions, including a syn position.
_MIN_ROUNDTRIP = 20


@dataclass(frozen=True)
class PoolEntry:
    """One demo in the scored pool (authoritative key + gold from score-source)."""

    pool_index: int
    split: str
    sample_index: int
    nl_question: str
    overpassql: str


def _load_json(path: str) -> List[Dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_eval_items(path: str) -> List[Dict]:
    """Eval split (score-source); ``query_index`` indexes positionally into this."""
    return _load_json(path)


def _roundtrip_positions(pool_size: int, split_spans: List[Tuple[str, int, int]]) -> List[int]:
    """Deterministic >=20 positions to check, guaranteed to include a syn position."""
    positions = set()
    # Evenly spaced sweep across the whole pool.
    step = max(1, pool_size // _MIN_ROUNDTRIP)
    positions.update(range(0, pool_size, step))
    # Force boundaries of every span (esp. the first syn position >= 6047).
    for _split, start, end in split_spans:
        positions.add(start)
        positions.add(end - 1)
    return sorted(p for p in positions if 0 <= p < pool_size)


class PoolMap:
    """Pool index -> PoolEntry, and pool index -> CoT record (or None if quarantined)."""

    def __init__(
        self,
        pool: List[PoolEntry],
        cot_by_key: Dict[Tuple[str, int], Dict],
    ):
        self.pool = pool
        self.cot_by_key = cot_by_key

    def entry(self, pool_index: int) -> PoolEntry:
        return self.pool[pool_index]

    def cot(self, pool_index: int) -> Optional[Dict]:
        e = self.pool[pool_index]
        return self.cot_by_key.get((e.split, e.sample_index))

    def has_cot(self, pool_index: int) -> bool:
        e = self.pool[pool_index]
        return (e.split, e.sample_index) in self.cot_by_key

    @property
    def coverage(self) -> Tuple[int, int]:
        covered = sum(1 for i in range(len(self.pool)) if self.has_cot(i))
        return covered, len(self.pool)

    @classmethod
    def build(
        cls,
        pool_cfg: PoolConfig,
        cot_records: Optional[List[Dict]] = None,
    ) -> "PoolMap":
        """Construct the pool from the score-source files.

        Plain mode (``cot_records is None``, the methodology §4 main setting):
        the pool is built directly from the score-source ``(nl_question,
        OverpassQL)`` rows — no CoT corpus, no score↔CoT round-trip guard (it is
        moot when there is no second file to misalign with).

        CoT mode (``cot_records`` given, kept for the CoT-vs-plain ablation):
        additionally loads the cot-source files and runs the §3 guard so a score
        computed on the score-source NL truly corresponds to its CoT reasoning.
        """
        pool: List[PoolEntry] = []
        split_spans: List[Tuple[str, int, int]] = []
        cot_src: Dict[str, List[Dict]] = {}  # cot-source rows per split (CoT mode only)
        plain = cot_records is None

        for split in pool_cfg.splits:
            score_rows = _load_json(pool_cfg.score_source_path(split))
            if not plain:
                cot_rows = _load_json(pool_cfg.cot_source_path(split))
                if len(score_rows) != len(cot_rows):
                    raise ValueError(
                        f"split {split}: score-source has {len(score_rows)} rows but "
                        f"cot-source has {len(cot_rows)} — file sets diverged"
                    )
                cot_src[split] = cot_rows
            start = len(pool)
            for row in score_rows:
                pool.append(
                    PoolEntry(
                        pool_index=len(pool),
                        split=split,
                        sample_index=int(row["sample_index"]),
                        nl_question=row.get("nl_question", ""),
                        overpassql=row.get("OverpassQL", ""),
                    )
                )
            split_spans.append((split, start, len(pool)))

        # (1)+(2): pool size must equal the score-vector length.
        if len(pool) != pool_cfg.size:
            raise ValueError(
                f"pool size {len(pool)} != configured/score-vector size {pool_cfg.size}"
            )

        if plain:
            logger.info("pool map OK (plain mode): %d demos, no CoT corpus", len(pool))
            return cls(pool, {})

        # (3): round-trip score-source vs cot-source at >=20 positions, incl. a
        # syn position >= the first syn index. Same (split, sample_index) must
        # denote the same question in both file sets.
        span_by_split = {s: (a, b) for s, a, b in split_spans}
        checked = 0
        for pos in _roundtrip_positions(len(pool), split_spans):
            e = pool[pos]
            local = pos - span_by_split[e.split][0]
            cot_row = cot_src[e.split][local]
            if int(cot_row["sample_index"]) != e.sample_index:
                raise ValueError(
                    f"§3 guard FAIL at pool pos {pos} ({e.split}): score-source "
                    f"sample_index={e.sample_index} != cot-source "
                    f"sample_index={cot_row['sample_index']} — pool ordering drift"
                )
            if cot_row.get("nl_question", "") != e.nl_question:
                raise ValueError(
                    f"§3 guard FAIL at pool pos {pos} ({e.split}, sample_index="
                    f"{e.sample_index}): nl_question differs between score-source "
                    f"and cot-source — the scored item != the CoT item"
                )
            checked += 1
        syn_checked = any(
            pool[p].split == "syn"
            for p in _roundtrip_positions(len(pool), split_spans)
        )
        if checked < _MIN_ROUNDTRIP or not syn_checked:
            raise ValueError(
                f"§3 guard incomplete: checked {checked} positions, "
                f"syn_checked={syn_checked} (need >={_MIN_ROUNDTRIP} incl. syn)"
            )

        cot_by_key = {(r["split"], int(r["sample_index"])): r for r in cot_records}
        covered = sum(1 for e in pool if (e.split, e.sample_index) in cot_by_key)
        logger.info(
            "pool map OK: %d demos, §3 guard passed at %d positions (syn incl.); "
            "CoT coverage %d/%d (%.1f%%)",
            len(pool), checked, covered, len(pool), 100.0 * covered / len(pool),
        )
        return cls(pool, cot_by_key)
