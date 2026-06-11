"""Execute-first repair checkers (DeepEye-SQL BaseChecker analogue).

A checker runs a query; only when execution fails does it ask the LLM to
repair it. This module is decoupled from any agent context — `check_and_revise`
takes the raw `query`, `nl_question` and `bbox` directly, so the Revision
module can be reused outside the original agent pipeline.

`BaseChecker` is the generic execute-first skeleton; `SyntaxChecker` supplies
the OverpassQL repair prompt (the highest-ROI checker per the C4 ablations).
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Type

from .executor import ExecResult, OQLExecutor
from .hygiene import DirtyKind, classify, clean_output
from .llm import BaseLLM

_FENCE_RE = re.compile(r"```(?:overpass|overpassql|ql)?\s*(.*?)```",
                       re.DOTALL | re.IGNORECASE)

# One OverpassQL bracket filter: ["key" op "value"]. op captured so callers can
# keep only equality (`=`) filters; regex/negation operators are out of scope.
_TAG_FILTER_RE = re.compile(
    r'\[\s*"?(?P<key>[\w:]+)"?\s*(?P<op>!=|!~|=|~)\s*"?(?P<val>[^"\]]+?)"?\s*\]')


def equality_tags(query: str) -> List[Tuple[str, str]]:
    """Extract literal equality `(key, value)` tags from a query.

    Only the `=` operator is returned; `!=`, `~`, `!~` (negation / regex) are
    skipped because their semantics are out of the TagChecker's scope.
    """
    out: List[Tuple[str, str]] = []
    for m in _TAG_FILTER_RE.finditer(query):
        if m.group("op") == "=":
            out.append((m.group("key"), m.group("val").strip()))
    return out


def extract_query(raw: str) -> str:
    """Strip code fences / leading prose; return string starting with '[out:'."""
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


_CHECKER_REGISTRY: Dict[str, Type["BaseChecker"]] = {}


def register_checker(name: str):
    """Class decorator registering a BaseChecker subclass under `name`."""
    def deco(cls: Type["BaseChecker"]) -> Type["BaseChecker"]:
        if name in _CHECKER_REGISTRY:
            raise ValueError(f"Checker '{name}' already registered "
                             f"to {_CHECKER_REGISTRY[name].__name__}")
        _CHECKER_REGISTRY[name] = cls
        return cls
    return deco


def get_checker(name: str) -> Type["BaseChecker"]:
    if name not in _CHECKER_REGISTRY:
        raise KeyError(f"Unknown checker '{name}'. "
                       f"Registered: {sorted(_CHECKER_REGISTRY)}")
    return _CHECKER_REGISTRY[name]


def available_checkers() -> List[str]:
    return sorted(_CHECKER_REGISTRY)


class BaseChecker(ABC):
    """Execute-first repair checker. Runs the query; only calls the LLM when
    execution fails. Returns the (possibly repaired) query, its ExecResult, and
    the number of repair attempts consumed (0 when the query was already ok)."""

    name: str = "checker"

    def __init__(self, llm: BaseLLM, executor: OQLExecutor,
                 max_repair_attempts: int = 3) -> None:
        self.llm = llm
        self.executor = executor
        self.max_repair_attempts = max_repair_attempts

    def check_and_revise(self, query: str, nl_question: str,
                         bbox: str) -> Tuple[str, ExecResult, int]:
        res = self.executor.run(query, bbox)
        if res.ok:
            return query, res, 0
        prompt = self._build_repair_prompt(query, res.error or "", nl_question)
        attempts = 0
        for raw in self.llm.ask_n(prompt, self.max_repair_attempts):
            attempts += 1
            cand = extract_query(raw)
            if not cand or not cand.startswith("["):  # R1: skip non-canonical
                continue
            cand_res = self.executor.run(cand, bbox)
            if cand_res.ok:
                return cand, cand_res, attempts
        return query, res, attempts

    @abstractmethod
    def _build_repair_prompt(self, query: str, error: str,
                             nl_question: str) -> List[Dict[str, str]]:
        """Build the chat-message list sent to the LLM for repair."""
        ...


@register_checker("syntax")
class SyntaxChecker(BaseChecker):
    """Repairs OverpassQL that fails to prepare/execute (syntax / parse errors)."""

    name = "syntax"

    def _build_repair_prompt(self, query: str, error: str,
                             nl_question: str) -> List[Dict[str, str]]:
        return [
            {"role": "system",
             "content": "You are an OpenStreetMap OverpassQL expert. Fix the "
                        "query so it is valid and executable. Return ONLY the "
                        "corrected OverpassQL, no prose."},
            {"role": "user",
             "content": (
                 f"Question: {nl_question}\n\n"
                 f"Broken OverpassQL:\n{query}\n\n"
                 f"Execution error:\n{error}\n\n"
                 "These Overpass-Turbo macros are supported and resolved before "
                 "execution — prefer them over invented syntax:\n"
                 "- Relative date `{{date:<n><unit>}}` where unit is "
                 "second/minute/hour/day/week/month/year (plural ok), e.g. "
                 "`{{date:3day}}`; the dataset snapshot date is 2022-07-04, so "
                 "relative dates count back from then. An ISO `YYYY-MM-DDThh:mm:ssZ` "
                 "literal is also accepted.\n"
                 "- User variable: define once with `{{key=value}}` then reference "
                 "as `{{key}}` later (e.g. `{{a=area(3600...)}}` ... `area.a`).\n"
                 "Return the corrected OverpassQL.")},
        ]


def _is_no_bbox(bbox: str) -> bool:
    """True when the record carries no bounding box (MOsmNL `bbox=none`)."""
    return (bbox or "").strip().lower() in ("", "none")


@register_checker("geo")
class GeoBindingChecker(BaseChecker):
    """Repairs spatial-scope binding when the record has no bbox (`bbox=none`).

    Error class (MOsmNL error type 2): the question names a place/coordinates,
    but with no bbox available the model emits an unresolvable ``{{bbox}}`` /
    ``{{center}}`` placeholder (→ empty parens → parse error), a malformed
    geocode binding (parse error), or an unbounded query (runtime timeout).
    All three fail execution, so this checker is execute-first and gated on
    ``bbox=none``; on any other bbox it defers to later checkers.

    Scope is *spatial binding only* — the repair prompt rebinds scope from a
    concrete anchor in the question (``{{geocodeArea}}`` / ``{{geocodeCoords}}``
    / a literal bbox) or, when the question names no concrete location, drops
    the broken placeholder so the query is at least valid. It must not touch
    tags, filters, or output statements, so EX gains are capped by any
    co-occurring tag/type errors — an accepted, documented limitation.
    """

    name = "geo"

    def check_and_revise(self, query: str, nl_question: str,
                         bbox: str) -> Tuple[str, ExecResult, int]:
        res = self.executor.run(query, bbox)
        if res.ok:
            return query, res, 0
        if not _is_no_bbox(bbox):  # only the no-bbox scope class is in scope
            return query, res, 0
        prompt = self._build_repair_prompt(query, res.error or "", nl_question)
        attempts = 0
        for raw in self.llm.ask_n(prompt, self.max_repair_attempts):
            attempts += 1
            cand = extract_query(raw)
            if not cand or not cand.startswith("["):  # R1: skip non-canonical
                continue
            cand_res = self.executor.run(cand, bbox)
            if cand_res.ok:
                return cand, cand_res, attempts
        return query, res, attempts

    def _build_repair_prompt(self, query: str, error: str,
                             nl_question: str) -> List[Dict[str, str]]:
        return [
            {"role": "system",
             "content": "You are an OpenStreetMap OverpassQL expert. Fix ONLY "
                        "the spatial scope binding of the query. Do not change "
                        "tags, filters, object types, or output statements. "
                        "Return ONLY the corrected OverpassQL, no prose."},
            {"role": "user",
             "content": (
                 f"Question: {nl_question}\n\n"
                 f"Broken OverpassQL:\n{query}\n\n"
                 f"Execution error:\n{error}\n\n"
                 "No bounding box or map view is available for this question, so "
                 "any `{{bbox}}` / `{{center}}` placeholder cannot resolve and an "
                 "unbounded query may time out. Bind the spatial scope using a "
                 "concrete anchor taken FROM THE QUESTION:\n"
                 "- A named place/region (search inside its polygon) -> "
                 "`{{geocodeArea:\"Name\"}}->.searchArea;` then add "
                 "`(area.searchArea)` to each selector "
                 "(`{{nominatimArea:\"Name\"}}` is an equivalent alternative).\n"
                 "- A named place as a rectangular box (not a polygon) -> "
                 "`[bbox:{{geocodeBbox:\"Name\"}}]` header.\n"
                 "- A single named OSM feature (one node/way/relation) -> "
                 "`{{geocodeId:\"Name\"}}`.\n"
                 "- A point/landmark with a distance -> "
                 "`(around:<meters>,{{geocodeCoords:\"Name\"}})`.\n"
                 "- Explicit coordinates in the question -> a literal "
                 "`(south,west,north,east)` filter or `[bbox:south,west,north,east]` "
                 "header.\n"
                 "- If the question names NO concrete place or coordinates (only "
                 "vague phrasing like \"the specified area\" / \"the current "
                 "view\"), REMOVE the `{{bbox}}` / `{{center}}` placeholder and any "
                 "broken scope binding so the query is at least valid; do NOT "
                 "invent a location.\n\n"
                 "Keep tags, filters, and output unchanged. Return the corrected "
                 "OverpassQL.")},
        ]


@register_checker("hygiene")
class HygieneChecker(BaseChecker):
    """Layer-1 output-hygiene repair (execution-gated, deterministic-first).

    Only the *output-hygiene* error class is in scope: a clean query buried in
    leaked CoT/markdown prose. The fix is a pure text strip (cut at the first
    backtick) — no LLM call — so it consumes 0 repair attempts. Truncation and
    other failures are passed through untouched for later layers / upstream
    regeneration; this checker never tries to repair them.
    """

    name = "hygiene"

    def check_and_revise(self, query: str, nl_question: str,
                         bbox: str) -> Tuple[str, ExecResult, int]:
        res = self.executor.run(query, bbox)
        if res.ok:
            return query, res, 0
        if classify(query) is not DirtyKind.PROSE_LEAK:
            return query, res, 0  # not a hygiene failure — leave for later layers
        cleaned = clean_output(query)
        if not cleaned.startswith("[") or cleaned == query:  # R1 + no-op guard
            return query, res, 0
        cleaned_res = self.executor.run(cleaned, bbox)
        if cleaned_res.ok:
            return cleaned, cleaned_res, 0
        return query, res, 0

    def _build_repair_prompt(self, query: str, error: str,
                             nl_question: str) -> List[Dict[str, str]]:
        # Deterministic checker — never calls the LLM, but the base class is
        # abstract on this method.
        return []


@register_checker("tags")
class TagChecker(BaseChecker):
    """Repairs Type-3 OSM tag/value substitution errors.

    Error class (MOsmNL error type 3, substitution subset): the query is
    syntactically valid and executes, but uses a wrong OSM key/value (e.g.
    ``motorcar`` for ``motor_car``, ``highway=motorway`` for ``motorroad=yes``)
    and so returns wrong/empty results. Because such queries *execute fine*, the
    execute-first gate never fires — this checker overrides it with a **static
    tag-validity gate** against the Tag Base: it only acts when a literal `=`
    tag is not a known OSM `key=value`.

    Scope is *tag keys/values only*. Filter operators (regex-vs-exact,
    negation), object types, spatial scope, and output statements are left
    untouched — those are out of scope and handled (or not) elsewhere. The
    repair is grounded by KB-retrieved candidate tags for the question, and a
    candidate is accepted only if it executes and strictly reduces the number of
    suspect tags, so a query whose tags are all KB-valid is never touched.
    """

    name = "tags"

    def __init__(self, llm: BaseLLM, executor: OQLExecutor,
                 retriever=None, max_repair_attempts: int = 3,
                 top_k: int = 12) -> None:
        super().__init__(llm, executor, max_repair_attempts)
        if retriever is None:
            from .tag_kb import TagRetriever
            retriever = TagRetriever()
        self.retriever = retriever
        self.top_k = top_k

    def _suspect_tags(self, query: str) -> List[Tuple[str, str]]:
        """Equality tags whose `key=value` is not in the Tag Base vocabulary."""
        valid_kv = self.retriever.valid_kv
        return [(k, v) for k, v in equality_tags(query)
                if f"{k}={v}" not in valid_kv]

    def check_and_revise(self, query: str, nl_question: str,
                         bbox: str) -> Tuple[str, ExecResult, int]:
        res = self.executor.run(query, bbox)
        suspects = self._suspect_tags(query)
        if not suspects:  # all tags KB-valid -> defer, never touch
            return query, res, 0

        candidates = self.retriever.retrieve(nl_question, self.top_k)
        prompt = self._build_repair_prompt(query, nl_question, suspects,
                                            candidates)
        attempts = 0
        for raw in self.llm.ask_n(prompt, self.max_repair_attempts):
            attempts += 1
            cand = extract_query(raw)
            if not cand or not cand.startswith("["):  # R1: skip non-canonical
                continue
            # Only accept a candidate that removes at least one suspect tag.
            if len(self._suspect_tags(cand)) >= len(suspects):
                continue
            cand_res = self.executor.run(cand, bbox)
            if cand_res.ok:
                return cand, cand_res, attempts
        return query, res, attempts

    def _build_repair_prompt(self, query: str, nl_question: str,
                             suspects: List[Tuple[str, str]],
                             candidates) -> List[Dict[str, str]]:
        suspect_lines = "\n".join(f"- {k}={v}" for k, v in suspects)
        cand_lines = "\n".join(f"- {e.tag_id}" for e, _ in candidates)
        return [
            {"role": "system",
             "content": "You are an OpenStreetMap OverpassQL expert. The query "
                        "uses OSM tags that are not valid OSM keys/values. Fix "
                        "ONLY the tag keys and values, choosing valid OSM tags "
                        "from the candidate list when appropriate. Do NOT change "
                        "the spatial scope, filter operators (=, !=, ~, !~), "
                        "object types, or output statements. Return ONLY the "
                        "corrected OverpassQL, no prose."},
            {"role": "user",
             "content": (
                 f"Question: {nl_question}\n\n"
                 f"Query:\n{query}\n\n"
                 f"Suspect tags (not valid OSM key=value):\n{suspect_lines}\n\n"
                 f"Candidate valid OSM tags for this question:\n{cand_lines}\n\n"
                 "Replace each suspect tag with the correct valid OSM tag, "
                 "keeping everything else identical. Return the corrected "
                 "OverpassQL.")},
        ]
