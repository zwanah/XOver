"""Denotation signatures for OverpassQL execution results.

A *denotation* is what an Overpass query returns. ``op.query(prepared)`` yields
``(results, num)`` where (matching ``eval_backend/utils/eval_utils.py``):

* ``num < 0``  → execution/preprocessing error (``results`` is an error string)
* ``num == 0`` → empty result set
* ``results`` is a ``count_<N>`` string for ``out count`` queries (``num`` = N)
* ``results`` is a ``list`` of element identifiers otherwise

We turn that into a small *hashable* signature so candidates can be grouped by
"do they return the same thing". The canonical EX semantics compare
``ref_results == out_results`` (order-sensitive list equality) — our signature
preserves exactly that equality: ``signature(a) == signature(b)`` iff the two
results are EX-equal, with empties collapsing together (the MOsmNL empty-match
policy treats any two empty sets as matching).

These functions are pure; execution lives in ``scripts/diag_execute.py``.
"""
from __future__ import annotations

import hashlib
from typing import Any, Tuple

# Signature is a hashable tuple. First element is a discriminator tag.
Signature = Tuple[Any, ...]

ERROR = ("error",)
EMPTY = ("empty",)


def is_error(num: int) -> bool:
    """A negative element count marks an execution/preprocessing failure."""
    return int(num) < 0


def signature(results: Any, num: int) -> Signature:
    """Map a raw ``(results, num)`` pair to a hashable denotation signature.

    Error and empty are canonical singletons; ``count_*`` scalars key on their
    integer count; list results key on their exact (order-preserving) contents
    so the signature mirrors the ``results == results`` equality EX uses.
    """
    num = int(num)
    if num < 0:
        # Distinguish error kinds only loosely — errors never count as a
        # denotation for voting/grouping, so the exact text is informational.
        return ("error", str(results)[:120])
    if num == 0:
        return EMPTY
    if isinstance(results, list):
        # Hash the (order-preserving) element list rather than storing it — a
        # single gold can return 100k+ ids; the raw list would bloat the
        # persisted file to GBs. The hash preserves the exact ``results ==
        # results`` equality EX uses (same contents+order ⇒ same digest), so
        # candidates still group identically.
        h = hashlib.sha1()
        for e in results:
            h.update(repr(e).encode("utf-8"))
            h.update(b"\x00")
        return ("set", h.hexdigest())
    # count_<N> string or any other scalar shape.
    return ("scalar", str(results))


def to_jsonable(sig: Signature) -> list:
    """Signatures are tuples (hashable); JSON has only lists. Persist as a list."""
    return list(sig)


def freeze(obj: Any) -> Any:
    """Recursively turn JSON lists back into tuples so a loaded signature is
    hashable again (``("set", ["n/1", "w/2"])`` → ``("set", ("n/1", "w/2"))``)."""
    if isinstance(obj, list):
        return tuple(freeze(x) for x in obj)
    return obj


def is_empty_sig(sig: Signature) -> bool:
    return sig == EMPTY


def is_error_sig(sig: Signature) -> bool:
    return bool(sig) and sig[0] == "error"


def is_votable_sig(sig: Signature) -> bool:
    """Errors are not real denotations and are excluded from voting/grouping."""
    return not is_error_sig(sig)
