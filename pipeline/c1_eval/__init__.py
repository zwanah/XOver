"""MOsmNL eval pipeline.

Wraps eval_backend's TACO/OSMT5 evaluation utilities to:
  - point at the unified cache (data/eval_cache/)
  - enforce canonical-key precondition (R1 from plan §6)
  - apply the empty-match policy (§3a from plan):
    empty-vs-empty + no_error -> EX=1, EX_soft=1.0
"""

__all__ = ['runner']
