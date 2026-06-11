"""Tests for the §3 pool-map guard, position->key->CoT mapping, and coverage."""
import json

import pytest

from pipeline.c2_retrieval.config import PoolConfig
from pipeline.c2_retrieval.pool_map import PoolMap

N_TRAIN, N_SYN = 25, 8  # >= 20 so the round-trip guard has enough positions


def _row(split, si):
    return {
        "split": split, "sample_index": si,
        "nl_question": f"{split}-q-{si}",
        "OverpassQL": f"[out:json];{split}{si};out;",
    }


def _write(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f)


def _setup_sources(tmp_path, *, drift=None):
    """Write score-source + cot-source files; `drift` mutates one cot-source row."""
    score_root = tmp_path / "score"
    cot_root = tmp_path / "cot"
    score_root.mkdir()
    cot_root.mkdir()
    for split, n in (("train", N_TRAIN), ("syn", N_SYN)):
        score_rows = [_row(split, i) for i in range(n)]
        cot_rows = [_row(split, i) for i in range(n)]
        if drift and drift[0] == split:
            cot_rows[drift[1]][drift[2]] = drift[3]
        _write(score_root / f"{split}_final.json", score_rows)
        _write(cot_root / f"{split}_final_english.json", cot_rows)
    cfg = PoolConfig(
        score_source_root=str(score_root),
        cot_source_root=str(cot_root),
        splits={"train": ("train_final.json", "train_final_english.json"),
                "syn": ("syn_final.json", "syn_final_english.json")},
        size=N_TRAIN + N_SYN,
    )
    return cfg


def _corpus(skip=()):
    """CoT corpus records for the whole pool minus `skip` (split, sample_index)."""
    recs = []
    for split, n in (("train", N_TRAIN), ("syn", N_SYN)):
        for i in range(n):
            if (split, i) in skip:
                continue
            recs.append({**_row(split, i), "raw_response": f"#reason: {split}{i}",
                         "reason": "r", "scope": "s", "find": "f"})
    return recs


def test_guard_passes_and_maps(tmp_path):
    cfg = _setup_sources(tmp_path)
    pm = PoolMap.build(cfg, _corpus())
    assert len(pm.pool) == N_TRAIN + N_SYN
    # pool order = train then syn, file order
    assert pm.entry(0).split == "train" and pm.entry(0).sample_index == 0
    assert pm.entry(N_TRAIN).split == "syn" and pm.entry(N_TRAIN).sample_index == 0
    # position -> CoT record
    assert pm.cot(N_TRAIN)["raw_response"] == "#reason: syn0"
    assert pm.coverage == (N_TRAIN + N_SYN, N_TRAIN + N_SYN)


def test_guard_fails_on_sample_index_drift(tmp_path):
    cfg = _setup_sources(tmp_path, drift=("syn", 2, "sample_index", 999))
    with pytest.raises(ValueError, match="ordering drift"):
        PoolMap.build(cfg, _corpus())


def test_guard_fails_on_nl_drift(tmp_path):
    cfg = _setup_sources(tmp_path, drift=("train", 5, "nl_question", "MUTATED"))
    with pytest.raises(ValueError, match="nl_question differs"):
        PoolMap.build(cfg, _corpus())


def test_guard_fails_on_size_mismatch(tmp_path):
    cfg = _setup_sources(tmp_path)
    bad = PoolConfig(
        score_source_root=cfg.score_source_root,
        cot_source_root=cfg.cot_source_root,
        splits=cfg.splits, size=999,
    )
    with pytest.raises(ValueError, match="pool size"):
        PoolMap.build(bad, _corpus())


def test_coverage_and_has_cot_with_quarantine(tmp_path):
    cfg = _setup_sources(tmp_path)
    pm = PoolMap.build(cfg, _corpus(skip={("train", 3), ("syn", 1)}))
    covered, total = pm.coverage
    assert total == N_TRAIN + N_SYN
    assert covered == total - 2
    assert not pm.has_cot(3)              # train sample_index 3 quarantined
    assert pm.cot(3) is None
    assert not pm.has_cot(N_TRAIN + 1)    # syn sample_index 1 quarantined
    assert pm.has_cot(0)
