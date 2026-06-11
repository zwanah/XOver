"""Layer-1 output hygiene: deterministic cleaning of dirty LLM OverpassQL.

The biggest free lever in the n=44 error mining is *output hygiene*: ~32% of
execution failures are decoding/format artefacts, not OverpassQL-competence
failures. Two dominant shapes:

* **Prose / CoT leak** — the model emits a complete query, then a backtick and
  chain-of-thought commentary ("Wait, ...", "Actually, ..."). The backtick is
  not valid OverpassQL, so cutting at the first backtick recovers the query
  deterministically (no network, no LLM).
* **Truncation** — `max_tokens` cut the output mid-query (unbalanced delimiters
  / no terminal `out` statement). Not recoverable by stripping; must regenerate
  upstream with a higher token budget (handled by a separate Layer-1 path).

These functions are pure text transforms with no I/O so they are unit-testable
without an Overpass server or an LLM. The error string the executor surfaces is
only a coarse bucket (`parse_error`, `preprocessing_error`), so routing is done
by inspecting the query text, not the error message.
"""
from __future__ import annotations

import re
from enum import Enum

# Code-fence + leading-prose strip, mirroring scripts.run_baseline_llm.extract_query
# so a revised file matches what generation already does, plus trailing cleanup.
_FENCE_RE = re.compile(r"```(?:overpass|overpassql|ql)?\s*(.*?)```",
                       re.DOTALL | re.IGNORECASE)

# Opening/closing delimiter pairs whose balance signals a complete query.
_OPEN = {"(": ")", "[": "]", "{": "}"}
_CLOSE = {v: k for k, v in _OPEN.items()}


class DirtyKind(str, Enum):
    """Why a generated_text failed, decided from the text alone."""

    CLEAN = "clean"            # extraction yields a syntactically complete query
    PROSE_LEAK = "prose_leak"  # complete query followed by backtick/CoT prose
    TRUNCATED = "truncated"    # cut off mid-query (unbalanced / no terminal out)
    OTHER = "other"            # non-empty but neither clearly leak nor truncation


def _strip_fence_and_lead(raw: str) -> str:
    """Drop code fences and any prose before the first `[out:` (leading hygiene)."""
    if not raw:
        return ""
    m = _FENCE_RE.search(raw)
    if m:
        raw = m.group(1)
    raw = raw.strip()
    idx = raw.find("[out:")
    if idx > 0:
        raw = raw[idx:]
    return raw.strip()


def _cut_trailing_prose(query: str) -> str:
    """Cut at the first backtick — it never appears in valid OverpassQL, so it
    reliably marks where leaked CoT/markdown prose begins."""
    bt = query.find("`")
    if bt != -1:
        query = query[:bt]
    return query.strip()


def clean_output(raw: str) -> str:
    """Deterministically extract the single query from a dirty model output.

    Strips code fences, leading prose before `[out:`, and trailing prose after
    the first backtick. Pure text transform — safe to run on every prediction.
    """
    return _cut_trailing_prose(_strip_fence_and_lead(raw))


def _balanced_delims(query: str) -> bool:
    """True iff (), [], {} nest correctly and double-quotes are paired.

    Quotes are tracked first: delimiters inside a string literal don't count.
    A backslash escapes the next char inside a quote.
    """
    stack: list[str] = []
    in_str = False
    escaped = False
    for ch in query:
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in _OPEN:
            stack.append(ch)
        elif ch in _CLOSE:
            if not stack or stack.pop() != _CLOSE[ch]:
                return False
    return not stack and not in_str


def _has_out_action(query: str) -> bool:
    """True iff the query has an `out ...;` action statement (not just the
    `[out:json]` setting header). A complete query always ends in one."""
    # Remove leading setting headers like [out:json][timeout:25] so the `out`
    # inside them is not mistaken for an action.
    body = re.sub(r"^\s*(\[[^\]]*\]\s*)+", "", query)
    return re.search(r"\bout\b[^;]*;", body) is not None


def is_truncated(query: str) -> bool:
    """Heuristic truncation test on an already-extracted query.

    Truncated iff delimiters are unbalanced OR there is no terminal `out ...;`
    action statement. Both are signatures of a `max_tokens` cutoff and are not
    recoverable by stripping — they need upstream regeneration.
    """
    if not query:
        return True
    return (not _balanced_delims(query)) or (not _has_out_action(query))


def classify(raw: str) -> DirtyKind:
    """Decide the hygiene failure mode of a raw generated_text.

    Order matters: a prose leak is only a leak if cutting the backtick leaves a
    *complete* query; otherwise the backtick sat on a truncated fragment and the
    real problem is truncation.
    """
    extracted = _strip_fence_and_lead(raw)
    if not extracted:
        return DirtyKind.TRUNCATED  # nothing query-like survived extraction
    had_backtick = "`" in extracted
    cleaned = _cut_trailing_prose(extracted)
    if is_truncated(cleaned):
        return DirtyKind.TRUNCATED
    if had_backtick or cleaned != extracted.strip():
        return DirtyKind.PROSE_LEAK
    # Extraction already produced a complete query with no trailing prose: the
    # parse failure (if any) is not an output-hygiene problem.
    return DirtyKind.CLEAN
