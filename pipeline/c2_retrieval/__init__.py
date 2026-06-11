"""C2: cross-lingual relevance-based few-shot retrieval.

Three decoupled stages so the example-selection method can be swapped freely:

  1. **score**  (:mod:`scores`)   — per-query relevance over the demo pool,
     ``R = alpha*sim + (1-alpha)*influence``; the pluggable :class:`ScoreFunction`
     seam is where a trained R1-R3 retriever later plugs in.
  2. **select** (:mod:`select`)   — turn the relevance ranking into top-k demo-pool
     **indices** (default :class:`TopKSelector`, dedup-by-gold-prefix).
  3. **assemble** (:mod:`assemble`) — render the selected indices into a *plain*
     ``(#question, #OverpassQL)`` few-shot prompt (methodology §4 main setting).
"""
from .assemble import assemble_prompt, render_block
from .dpp import greedy_dpp_select
from .pool_map import PoolMap, PoolEntry, load_eval_items
from .profile_kernel import ProfileKernelProvider
from .scores import QueryScores, ScoreFunction, ScoreTable
from .select import (
    ProfileDppSelector,
    Selector,
    SelectionResult,
    TopKSelector,
)

__all__ = [
    "ScoreFunction",
    "ScoreTable",
    "QueryScores",
    "PoolMap",
    "PoolEntry",
    "load_eval_items",
    "Selector",
    "SelectionResult",
    "TopKSelector",
    "ProfileDppSelector",
    "ProfileKernelProvider",
    "greedy_dpp_select",
    "render_block",
    "assemble_prompt",
]
