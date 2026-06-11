"""Revision — a standalone verification-repair module.

Execute-first repair of generated OverpassQL: run a candidate, and only when
execution fails ask an LLM to repair it. Reuses the eval runner's Overpass
backend + cross-repo `prepare_query`, so a revised prediction file feeds
straight into `pipeline.c1_eval.runner` for EX comparison.

Extracted from the C4 Layer-1 design to be reusable across future designs;
decoupled from the original agent pipeline (no QueryContext / Stage / registry).
"""
from .checkers import (
    BaseChecker,
    GeoBindingChecker,
    HygieneChecker,
    SyntaxChecker,
    TagChecker,
    available_checkers,
    equality_tags,
    extract_query,
    get_checker,
    register_checker,
)
from .tag_kb import TagKBEntry, TagRetriever, build_vocab, load_tag_kb
from .hygiene import DirtyKind, classify, clean_output, is_truncated
from .executor import (
    ExecResult,
    FakeOQLExecutor,
    OQLExecutor,
    RealOverpassExecutor,
)
from .llm import BaseLLM, FakeLLM, OpenAICompatLLM
from .regen import RegenConfig, make_prompt_builder, regenerate_item, run_stage1
from .reviser import Reviser, RevisionResult, revise_prediction_file

__all__ = [
    "ExecResult",
    "OQLExecutor",
    "RealOverpassExecutor",
    "FakeOQLExecutor",
    "BaseLLM",
    "OpenAICompatLLM",
    "FakeLLM",
    "BaseChecker",
    "SyntaxChecker",
    "GeoBindingChecker",
    "HygieneChecker",
    "TagChecker",
    "register_checker",
    "get_checker",
    "available_checkers",
    "extract_query",
    "equality_tags",
    "TagKBEntry",
    "TagRetriever",
    "load_tag_kb",
    "build_vocab",
    "clean_output",
    "classify",
    "is_truncated",
    "DirtyKind",
    "Reviser",
    "RevisionResult",
    "revise_prediction_file",
    "RegenConfig",
    "make_prompt_builder",
    "regenerate_item",
    "run_stage1",
]
