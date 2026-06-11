"""Execution backend for the Revision module.

`OQLExecutor` is the execute-first abstraction the checker chain runs against.
`RealOverpassExecutor` reuses the eval runner's `Overpass` (canonical-key
enforced) + the cross-repo `prepare_query`, sharing the same split-aware
`data/eval_cache/`. `FakeOQLExecutor` is a no-network stand-in for tests.

Empty result sets are a *valid* `ok=True` outcome (empty match), never a
failure — the checker chain must not try to repair them.
"""
from __future__ import annotations

import os
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import FrozenSet, Optional, Tuple

EVAL_BACKEND_PATH = os.environ.get("EVAL_BACKEND_PATH", "/path/to/eval_backend")
DEFAULT_OVERPASS_URL = "http://localhost:12346/api/interpreter"
DEFAULT_NOMINATIM_URL = "https://nominatim.openstreetmap.org/search.php"


@dataclass(frozen=True)
class ExecResult:
    """Outcome of executing one query against the backend.

    `element_ids` is the set of returned OSM element ids; an empty set is a
    valid empty match, not a failure.
    """

    ok: bool
    error: Optional[str]
    element_ids: FrozenSet[int]

    def __post_init__(self) -> None:
        if not self.ok and self.error is None:
            raise ValueError("ExecResult with ok=False must carry an error message")


class OQLExecutor(ABC):
    """Executes OverpassQL. Subclasses must honor the R1 canonical-key
    invariant: only canonical (prepared) queries reach the wire."""

    @abstractmethod
    def prepare(self, query: str, bbox: str) -> Tuple[Optional[str], Optional[str]]:
        """Return (prepared_query, None) on success or (None, error_msg) on failure."""

    @abstractmethod
    def execute(self, prepared: str) -> ExecResult:
        """Execute an already-prepared (canonical) query."""

    def run(self, query: str, bbox: str) -> ExecResult:
        """Prepare then execute. On prepare failure, return an error result
        without calling execute (so only prepared queries reach the wire)."""
        prepared, err = self.prepare(query, bbox)
        if err is not None:
            return ExecResult(ok=False, error=err, element_ids=frozenset())
        return self.execute(prepared)


class RealOverpassExecutor(OQLExecutor):
    """Wraps the eval runner's `Overpass` + cross-repo `prepare_query`, sharing
    the unified split-aware `data/eval_cache/`. Cache-first by construction.

    Heavy cross-repo imports happen lazily in __init__ so importing this module
    never triggers them. Single endpoint for v1; a parallel pool can wrap this
    later without changing the public API.
    """

    def __init__(self, url: str = DEFAULT_OVERPASS_URL, split: str = "dev",
                 cache_dir: Optional[str] = None, timeout: int = 300,
                 nominatim_url: str = DEFAULT_NOMINATIM_URL) -> None:
        if EVAL_BACKEND_PATH not in sys.path:
            sys.path.insert(0, EVAL_BACKEND_PATH)
        from pipeline.c1_eval.runner import Nominatim, Overpass, DEFAULT_CACHE_DIR
        cdir = cache_dir or DEFAULT_CACHE_DIR
        # Match runner.py's split-specific cache filename so we share the warmed
        # cache rather than a fresh, empty db.
        self._op = Overpass(url, cache_dir=cdir,
                            cache_filename=f"overpass_{split}_cache")
        # Wire a real Nominatim so {{geocodeArea}}/{{geocodeCoords}} macros
        # expand during prepare(); nominatim=None makes them false parse errors
        # and would corrupt any recovery measurement (shares runner's cache).
        self._nominatim = Nominatim(nominatim_url, cache_dir=cdir)
        self._timeout = timeout

    def prepare(self, query: str, bbox: str) -> Tuple[Optional[str], Optional[str]]:
        from utils.eval_utils import prepare_query  # noqa: E402 (sys.path set above)
        prepared, err = prepare_query(query, bbox=bbox, timeout=self._timeout,
                                      nominatim=self._nominatim)
        if err is not None:
            return None, str(err)
        return prepared, None

    def execute(self, prepared: str) -> ExecResult:
        try:
            raw = self._op.query(prepared)
            return self._to_result(raw)
        except Exception as e:  # noqa: BLE001
            return ExecResult(ok=False, error=str(e), element_ids=frozenset())

    @staticmethod
    def _to_result(raw) -> ExecResult:
        """Map the upstream `Overpass.query` tuple contract to an ExecResult.

        `Overpass.query` *always* returns a `(payload, count)` tuple:
        - payload is a list of "type_id" strings -> success (empty list = a
          valid empty match);
        - payload is a "count_N" string -> success (count query);
        - payload is an error-name string ("parse_error",
          "runtime_error_timeout", "static_error_*", "memory_error",
          "server_error_timeout", "unknown_error") with count -1/-2 -> failure.

        Note: element ids collapse to the trailing int, so `node_5` and `way_5`
        both map to `5`. Only emptiness/`len` of `element_ids` is read today;
        do not use this field for cross-type element-set comparison.
        """
        payload = raw[0] if isinstance(raw, tuple) and raw else raw
        if isinstance(payload, (list, tuple)):
            ids = frozenset(
                int(str(s).rsplit("_", 1)[-1])
                for s in payload
                if str(s).rsplit("_", 1)[-1].isdigit()
            )
            return ExecResult(ok=True, error=None, element_ids=ids)
        if isinstance(payload, str) and payload.startswith("count_"):
            return ExecResult(ok=True, error=None, element_ids=frozenset())
        error_type = str(payload) if payload is not None else "overpass_error"
        return ExecResult(ok=False, error=error_type, element_ids=frozenset())

    def save_cache(self) -> None:
        """Flush the underlying Overpass + Nominatim caches to disk."""
        self._op.save_cache()
        try:
            self._nominatim.save_cache()
        except Exception:  # noqa: BLE001  (best-effort flush)
            pass


class FakeOQLExecutor(OQLExecutor):
    """No-network executor for tests. `prepare` flags any query containing
    `bad_marker` as a syntax error and otherwise returns the stripped string.
    `execute` returns a scripted ExecResult per prepared key, defaulting to an
    empty-but-ok result."""

    def __init__(self, results: Optional[dict] = None, bad_marker: str = "BAD") -> None:
        self._results = results or {}
        self._bad = bad_marker

    def prepare(self, query: str, bbox: str) -> Tuple[Optional[str], Optional[str]]:
        if self._bad in query:
            return None, f"syntax error near {self._bad}"
        return query.strip(), None

    def execute(self, prepared: str) -> ExecResult:
        return self._results.get(
            prepared, ExecResult(ok=True, error=None, element_ids=frozenset()))
