"""Candidate-selection diagnostic for Text-to-OverpassQL.

Pure, testable core for the Pass@K + denotation-voting experiment that gates
whether a learned verifier has a wedge over execution-only selection. Execution
(against the frozen Overpass docker) lives in ``scripts/diag_execute.py``.
"""
from . import analyze, denotation, selectors, strata

__all__ = ["analyze", "denotation", "selectors", "strata"]
