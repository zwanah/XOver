"""Stage-1 output hygiene + truncation regeneration.

Stage-1 is a self-contained pre-pass over a baseline prediction file. It routes
each `generated_text` by inspecting the query *text* (never the eval error
string, which is the coarse `parse_error` bucket — see hygiene.py):

* **CLEAN** — left untouched.
* **PROSE_LEAK / OTHER** — `clean_output` strips markdown/CoT deterministically
  (no network, no LLM).
* **TRUNCATED** — the `max_tokens` cutoff is not recoverable by stripping, so we
  regenerate **once** with the original model/settings at a higher token budget
  (`regen_max_tokens`, default 16384). If the regenerated output is still
  truncated, we mark it unrecoverable and fall back to `clean_output` on the
  original partial text.

The hygiene-clean file this produces feeds straight into `pipeline.c1_eval.runner` or
the downstream execute-first `Reviser` (syntax/scope/tag/logic). Regeneration is
dependency-injected (`generate_fn`, `prompt_builder`) so `run_stage1` is
unit-testable without a network or an encoder; the real wiring lives in
`run_stage1.py`.

Determinism note: at temperature 0 a larger budget *should* reproduce the same
prefix and complete the tail, but hosted endpoints don't guarantee strict greedy
decoding, so the regenerated tail may diverge slightly from the original.
"""
from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from .hygiene import DirtyKind, classify, clean_output, is_truncated

logger = logging.getLogger(__name__)

REPO_ROOT = "/path/to/XOver"
DEFAULT_REGEN_MAX_TOKENS = 16384

# A prompt builder maps a test NL question to the user message; a generate fn
# maps (user_message, max_tokens) to the raw model output.
PromptBuilder = Callable[[str], str]
GenerateFn = Callable[[str, int], str]


@dataclass(frozen=True)
class RegenConfig:
    """The original generation settings to reproduce, read from a prediction
    file's `statistics` block (CLI overrides fill any gap in older files)."""

    model: str
    provider: str = "aigcbest"
    temperature: float = 0.0
    shots: int = 0
    icl_pool_lang: str = "english"
    icl_index_dir: str = "/path/to/XOver/data/icl_index"
    mpnet_model: str = "paraphrase-multilingual-mpnet-base-v2"
    query_instruction: Optional[str] = None
    encoder_device: Optional[str] = None
    regen_max_tokens: int = DEFAULT_REGEN_MAX_TOKENS

    @classmethod
    def from_statistics(cls, stats: Optional[Dict[str, Any]],
                        overrides: Optional[Dict[str, Any]] = None) -> "RegenConfig":
        """Build from a prediction file's statistics block. `overrides` (CLI
        flags) win over recorded values, which win over dataclass defaults."""
        stats = stats or {}
        overrides = {k: v for k, v in (overrides or {}).items() if v is not None}
        fields = ("model", "provider", "temperature", "shots", "icl_pool_lang",
                  "icl_index_dir", "mpnet_model", "query_instruction",
                  "encoder_device", "regen_max_tokens")
        merged: Dict[str, Any] = {}
        for f in fields:
            if f in overrides:
                merged[f] = overrides[f]
            elif f in stats and stats[f] is not None:
                merged[f] = stats[f]
        if "model" not in merged:
            raise ValueError(
                "RegenConfig needs a model name: not in statistics and no "
                "--model override given.")
        return cls(**merged)


def make_prompt_builder(cfg: RegenConfig) -> PromptBuilder:
    """Build the closure that reconstructs the original prompt for a given NL.

    Zero-shot returns `USER_TEMPLATE`. Few-shot loads the sBERT encoder + ICL
    index once and retrieves demonstrations per call, reusing
    `scripts.run_baseline_llm` so the prompt matches generation exactly.
    """
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)
    from scripts.run_baseline_llm import (  # noqa: E402
        USER_TEMPLATE, build_fewshot_user_message, load_icl_index, retrieve_topk,
    )

    if cfg.shots <= 0:
        return lambda nl: USER_TEMPLATE.format(nl=nl)

    from sentence_transformers import SentenceTransformer  # heavy import
    pool_emb, pool_keys = load_icl_index(cfg.icl_index_dir, cfg.icl_pool_lang)
    encoder = SentenceTransformer(cfg.mpnet_model, device=cfg.encoder_device)
    encode_kwargs: Dict[str, Any] = dict(
        batch_size=1, normalize_embeddings=True,
        show_progress_bar=False, convert_to_numpy=True)
    if cfg.query_instruction:
        encode_kwargs["prompt"] = f"Instruct: {cfg.query_instruction}\nQuery:"

    def builder(nl: str) -> str:
        vec = encoder.encode([nl], **encode_kwargs)[0]
        demos = retrieve_topk(vec, pool_emb, pool_keys, cfg.shots)
        return build_fewshot_user_message(demos, nl)

    return builder


def regenerate_item(raw_partial: str, user_message: str, generate_fn: GenerateFn,
                    regen_max_tokens: int = DEFAULT_REGEN_MAX_TOKENS,
                    ) -> Tuple[str, str]:
    """Regenerate one truncated prediction at a higher token budget.

    `user_message` is the exact prompt that produced the truncated output (reused
    verbatim — the system prompt is constant), so the regen is identical except
    for the budget. Returns (final_query, action): `regen_ok` if the new output is
    a complete canonical query; otherwise `regen_failed_fallback` with
    `clean_output` of the original partial text.
    """
    regen_raw = generate_fn(user_message, regen_max_tokens)
    cleaned = clean_output(regen_raw)
    if cleaned.startswith("[") and not is_truncated(cleaned):
        return cleaned, "regen_ok"
    return clean_output(raw_partial), "regen_failed_fallback"


def run_stage1(items: List[Dict[str, Any]], statistics: Optional[Dict[str, Any]] = None,
               generate_fn: Optional[GenerateFn] = None,
               prompt_builder: Optional[PromptBuilder] = None,
               regen_max_tokens: int = DEFAULT_REGEN_MAX_TOKENS,
               ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Run Stage-1 over prediction items.

    Regeneration reuses each item's stored `user_message` (the exact prompt that
    produced the truncated output) verbatim — no encoder/index needed. For legacy
    files that predate prompt persistence, `prompt_builder` reconstructs it from
    `query_nl` (the ICL fallback). When a TRUNCATED item has neither a stored
    prompt nor a `prompt_builder`, and/or no `generate_fn`, it falls back to
    `clean_output` (action `regen_skipped`) so the pass never crashes.

    Each output item keeps the original schema plus `original_generated_text`, the
    updated `generated_text`, and a `stage1` metadata block.
    """
    out: List[Dict[str, Any]] = []
    actions: Dict[str, int] = {}
    regen_calls = 0

    for item in items:
        raw = item.get("generated_text", "") or ""
        nl = item.get("query_nl", "") or item.get("nl_question", "")
        kind = classify(raw)

        if kind is DirtyKind.CLEAN:
            final, action = raw, "none"
        elif kind in (DirtyKind.PROSE_LEAK, DirtyKind.OTHER):
            final, action = clean_output(raw), "hygiene_clean"
        else:  # TRUNCATED
            user_message = item.get("user_message") or ""
            if not user_message and prompt_builder is not None:
                user_message = prompt_builder(nl)
            if generate_fn is not None and user_message:
                final, action = regenerate_item(
                    raw, user_message, generate_fn, regen_max_tokens)
                regen_calls += 1
            else:
                final, action = clean_output(raw), "regen_skipped"

        new = dict(item)
        new["original_generated_text"] = raw
        new["generated_text"] = final
        new["stage1"] = {
            "kind": kind.value,
            "action": action,
            "regen_max_tokens": regen_max_tokens if action.startswith("regen") else None,
        }
        out.append(new)
        actions[action] = actions.get(action, 0) + 1

    stage1_stats: Dict[str, Any] = {
        "total": len(out),
        "actions": actions,
        "regen_calls": regen_calls,
        "regen_max_tokens": regen_max_tokens,
    }
    logger.info("stage-1: %d items, actions=%s, regen_calls=%d",
                len(out), actions, regen_calls)
    return out, stage1_stats
