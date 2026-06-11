"""Richer evidence for the confidence judge (the discrimination lever).

The bare-cardinality judge can't tell which contested denotation is correct
(~chance forced-choice accuracy). Our dominant error class is wrong OSM tags
and wrong area, so we give the judge the one thing the query text alone can't
reveal: **what each candidate actually matched in the database** — a
tag-frequency summary of the matched elements, plus a contrastive A-only /
B-only view.

Two pure helpers (no network), one network fetch (injectable):

* ``tag_variant`` rewrites a *prepared* query so it returns the matched set's
  tags only (``out tags N``), dropping geometry recursion — otherwise a sample
  drowns in untagged ``>;`` child nodes.
* ``extract_predicates`` normalises the query text (element types, key=value /
  regex / key-presence filters, area name, spatial clause) so the judge sees a
  clean schema view.
* ``fetch_tag_evidence`` issues the tag-variant query and aggregates tags; the
  HTTP call is passed in so the module stays unit-testable.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

# A poster issues a tag-variant query and returns the raw Overpass JSON dict
# (``{"elements": [...]}``) or raises. Injected so tests need no network.
Poster = Callable[[str], dict]

_SETTINGS_RE = re.compile(r"^(\[.*?\];)")
_OUT_STMT_RE = re.compile(r"\bout\b[^;]*;")
_RECURSE_RE = re.compile(r"(?<![A-Za-z0-9])[<>]\s*;")


def tag_variant(prepared: str, n: int = 8) -> str:
    """Rewrite a prepared query to emit only the matched set's tags.

    Preserves the ``[out:json][timeout:N];`` settings header, strips every
    output / recursion statement, and appends ``out tags <n>;``. Selection
    (area + filters) is untouched, so the matched set is identical to the
    original query's first ``out`` — we just ask for tags instead of geometry.
    """
    m = _SETTINGS_RE.match(prepared)
    header = m.group(1) if m else "[out:json][timeout:60];"
    rest = prepared[m.end():] if m else prepared
    rest = _OUT_STMT_RE.sub("", rest)
    rest = _RECURSE_RE.sub("", rest)
    rest = rest.strip()
    if rest and not rest.endswith(";"):
        rest += ";"
    return f"{header}{rest}out tags {n};"


# --- query-text normalisation (no network) ----------------------------------
_AREA_RE = re.compile(r"\{\{geocodeArea:\s*\"?([^\"}]+)\"?\s*\}\}")
_AREA_ID_RE = re.compile(r"area\(id:(\d+)\)")
_ELEM_RE = re.compile(r"\b(nwr|node|way|rel|relation)\s*[\[\(]")
_TAG_RE = re.compile(r"\[\s*([\"']?)([^\"'\]~=!]+)\1\s*(=|!=|~|!~)?\s*([\"']?)([^\"'\]]*)\4\s*\]")
_AROUND_RE = re.compile(r"around:\s*([\d.]+)")
_BBOX_RE = re.compile(r"\(\s*-?\d+\.\d+\s*,\s*-?\d+\.\d+\s*,\s*-?\d+\.\d+\s*,\s*-?\d+\.\d+\s*\)")


def extract_predicates(query: str) -> Dict[str, object]:
    """Pull a clean schema view from raw OverpassQL text (best-effort)."""
    elems = sorted({m.group(1) for m in _ELEM_RE.finditer(query)})
    filters: List[str] = []
    for m in _TAG_RE.finditer(query):
        key = m.group(2).strip()
        op = m.group(3) or ""
        val = m.group(5).strip()
        if not key or key.startswith("out") or ":" in key[:1]:
            continue
        filters.append(f"{key}{op}{val}" if op else key)
    area = None
    am = _AREA_RE.search(query)
    if am:
        area = am.group(1).strip()
    spatial = []
    ar = _AROUND_RE.search(query)
    if ar:
        spatial.append(f"around:{ar.group(1)}m")
    if _BBOX_RE.search(query):
        spatial.append("bbox")
    return {"element_types": elems, "tag_filters": filters,
            "area": area, "spatial": spatial}


@dataclass
class CandidateEvidence:
    """What one candidate query is and what it actually matched."""

    text: str
    predicates: Dict[str, object]
    n_matched: int = 0                       # cardinality (from the pool)
    n_sampled: int = 0                       # elements returned by tag-variant
    tag_freq: Counter = field(default_factory=Counter)
    error: Optional[str] = None              # tag-fetch failure (judge sees count only)

    def predicate_str(self) -> str:
        p = self.predicates
        parts = []
        if p.get("element_types"):
            parts.append("types: " + ",".join(p["element_types"]))
        if p.get("tag_filters"):
            parts.append("filters: " + " ; ".join(p["tag_filters"]))
        if p.get("area"):
            parts.append(f"area: {p['area']}")
        if p.get("spatial"):
            parts.append("spatial: " + ",".join(p["spatial"]))
        return " | ".join(parts) if parts else "(none parsed)"

    def matched_tags_str(self, top: int = 10) -> str:
        if self.error:
            return f"(could not fetch matched tags: {self.error})"
        if not self.tag_freq:
            return "(no tagged elements matched)"
        return ", ".join(f"{t}×{c}" for t, c in self.tag_freq.most_common(top))


def aggregate_tags(elements: List[dict]) -> Counter:
    freq: Counter = Counter()
    for e in elements:
        for k, v in (e.get("tags") or {}).items():
            freq[f"{k}={v}"] += 1
    return freq


def fetch_tag_evidence(prepared: str, poster: Poster, *, n: int = 8,
                       n_matched: int = 0,
                       query_text: str = "") -> CandidateEvidence:
    """Run the tag-variant query and build a CandidateEvidence."""
    preds = extract_predicates(query_text or prepared)
    ev = CandidateEvidence(text=query_text or prepared, predicates=preds,
                           n_matched=n_matched)
    try:
        data = poster(tag_variant(prepared, n))
        els = data.get("elements", []) if isinstance(data, dict) else []
        ev.tag_freq = aggregate_tags(els)
        ev.n_sampled = len(els)
    except Exception as e:  # noqa: BLE001
        ev.error = str(e)[:120]
    return ev


def contrastive_tags(ev_a: CandidateEvidence, ev_b: CandidateEvidence,
                     top: int = 8) -> Tuple[List[str], List[str], List[str]]:
    """Return (A-only, B-only, shared) tag descriptors, most frequent first."""
    a_keys = set(ev_a.tag_freq)
    b_keys = set(ev_b.tag_freq)
    a_only = sorted(a_keys - b_keys, key=lambda t: -ev_a.tag_freq[t])[:top]
    b_only = sorted(b_keys - a_keys, key=lambda t: -ev_b.tag_freq[t])[:top]
    shared = sorted(a_keys & b_keys,
                    key=lambda t: -(ev_a.tag_freq[t] + ev_b.tag_freq[t]))[:top]
    return a_only, b_only, shared
