"""Unit tests for confidence-judge evidence helpers (``evidence.py``).

Pure logic + an injected fake poster (no network). Run: ``pytest tests/test_evidence.py``.
"""
from __future__ import annotations

from collections import Counter

from pipeline.c3_selection.evidence import (
    aggregate_tags,
    contrastive_tags,
    extract_predicates,
    fetch_tag_evidence,
    tag_variant,
)

PREP = ('[out:json][timeout:300];area(id:3600051477)->.searchArea;'
        '(nwr["historic"="castle"](area.searchArea););out;>;out skel qt;')


def test_tag_variant_preserves_settings_and_emits_tags():
    tv = tag_variant(PREP, n=8)
    assert tv.startswith("[out:json][timeout:300];")  # header kept (not eaten)
    assert tv.endswith("out tags 8;")
    assert ">;" not in tv and "out skel" not in tv      # recursion/skel dropped
    assert 'nwr["historic"="castle"]' in tv             # selection untouched


def test_tag_variant_handles_out_body_and_count_headers():
    q = '[out:json][timeout:25];node["amenity"="cafe"](area.a);out body;'
    tv = tag_variant(q, n=5)
    assert tv == '[out:json][timeout:25];node["amenity"="cafe"](area.a);out tags 5;'


def test_extract_predicates():
    p = extract_predicates(
        '[out:json];{{geocodeArea:"Berlin"}}->.a;(nwr["amenity"="cafe"]'
        '["wheelchair"](area.a);node["shop"~"super"](around:50.0););out;')
    assert p["area"] == "Berlin"
    assert "nwr" in p["element_types"] and "node" in p["element_types"]
    assert any("amenity=cafe" in f for f in p["tag_filters"])
    assert any(f == "wheelchair" for f in p["tag_filters"])  # key-presence
    assert any("shop~super" in f for f in p["tag_filters"])
    assert "around:50.0m" in p["spatial"]


def test_aggregate_and_contrastive():
    a = [{"tags": {"shop": "convenience", "name": "X"}},
         {"tags": {"shop": "convenience"}}]
    b = [{"tags": {"shop": "supermarket"}}]
    fa, fb = aggregate_tags(a), aggregate_tags(b)
    assert fa["shop=convenience"] == 2
    a_only, b_only, shared = contrastive_tags(
        type("E", (), {"tag_freq": fa})(), type("E", (), {"tag_freq": fb})())
    assert "shop=convenience" in a_only
    assert "shop=supermarket" in b_only
    assert shared == []  # no overlap here


def test_fetch_tag_evidence_with_fake_poster():
    def poster(tv):
        assert "out tags" in tv
        return {"elements": [{"tags": {"historic": "castle"}},
                             {"tags": {"historic": "castle", "ruins": "yes"}}]}
    ev = fetch_tag_evidence(PREP, poster, n=8, n_matched=120, query_text=PREP)
    assert ev.error is None
    assert ev.n_sampled == 2
    assert ev.tag_freq["historic=castle"] == 2
    assert "historic=castle" in ev.matched_tags_str()
    assert "castle" in ev.predicate_str()


def test_fetch_tag_evidence_poster_failure_degrades():
    def poster(tv):
        raise RuntimeError("timeout")
    ev = fetch_tag_evidence(PREP, poster, query_text=PREP)
    assert ev.error is not None
    assert "could not fetch" in ev.matched_tags_str()


if __name__ == "__main__":
    import sys
    import pytest
    sys.exit(pytest.main([__file__, "-q"]))
