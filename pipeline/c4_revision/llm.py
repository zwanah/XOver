"""LLM clients for the Revision module.

`BaseLLM` is the abstract chat interface with thread-safe token accounting.
`OpenAICompatLLM` is the real client (AIGCBest / DeepSeek), reusing the API
keys from eval_backend. `FakeLLM` is a no-network stand-in for tests.

`ask_n` samples n independent completions — the repair loop uses it to try
several fixes. With a real client at temperature 0 the n samples would be
identical, so the reviser defaults repair temperature > 0 (see run.py).
"""
from __future__ import annotations

import os
import logging
import sys
import threading
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

EVAL_BACKEND_PATH = os.environ.get("EVAL_BACKEND_PATH", "/path/to/eval_backend")
PROVIDER_BASE_URL = {
    "aigcbest": "https://api2.aigcbest.top/v1",
    "deepseek": "https://api.deepseek.com/v1",
}


class BaseLLM(ABC):
    """Abstract OpenAI-compatible chat LLM with thread-safe token accounting."""

    def __init__(self, model_name: str, temperature: float = 0.0,
                 max_tokens: int = 4096) -> None:
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self._usage_lock = threading.Lock()

    @abstractmethod
    def ask(self, messages: List[Dict[str, str]]) -> str:
        """Send multi-turn messages, return response text (or '' on failure)."""

    def ask_n(self, messages: List[Dict[str, str]], n: int) -> List[str]:
        """Sample n independent completions. Default loops `ask`; subclasses may
        batch. n <= 0 returns []."""
        return [self.ask(messages) for _ in range(max(0, n))]

    def get_usage(self) -> Dict[str, int]:
        with self._usage_lock:
            return {
                "prompt_tokens": self.total_prompt_tokens,
                "completion_tokens": self.total_completion_tokens,
                "total_tokens": self.total_prompt_tokens + self.total_completion_tokens,
            }

    def reset_usage(self) -> None:
        with self._usage_lock:
            self.total_prompt_tokens = 0
            self.total_completion_tokens = 0

    def _add_usage(self, prompt_tokens: int = 0, completion_tokens: int = 0) -> None:
        with self._usage_lock:
            self.total_prompt_tokens += int(prompt_tokens or 0)
            self.total_completion_tokens += int(completion_tokens or 0)


def load_provider_keys(provider: str) -> List[str]:
    """Import API keys from the single source of truth in eval_backend."""
    if EVAL_BACKEND_PATH not in sys.path:
        sys.path.insert(0, EVAL_BACKEND_PATH)
    try:
        from train_influence.config import API_KEYS  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise RuntimeError("Could not import API_KEYS from "
                           "eval_backend/train_influence/config.py.") from e
    keys = [k for k in API_KEYS.get(provider, []) if k]
    if not keys:
        raise RuntimeError(f"No keys available in API_KEYS['{provider}'].")
    return keys


class OpenAICompatLLM(BaseLLM):
    """OpenAI-compatible chat client (AIGCBest / DeepSeek). Builds a client pool
    (one per key) so `ask_n` fans out one in-flight call per key."""

    def __init__(self, model_name: str, provider: str = "aigcbest",
                 temperature: float = 0.0, max_tokens: int = 4096,
                 key_index: int = 0, api_key: Optional[str] = None,
                 api_keys: Optional[List[str]] = None, num_keys: int = 0,
                 base_url: Optional[str] = None, max_retries: int = 3,
                 thinking: str = "auto") -> None:
        super().__init__(model_name, temperature, max_tokens)
        from openai import OpenAI
        # Reasoning toggle, sent via OpenAI-compatible `extra_body`.
        # "auto" = no override (model default); "disabled" = genuine
        # non-reasoning generator; "enabled" = force reasoning. Mirrors
        # scripts/infer_fewshot.py so DeepSeek non-thinking matches the
        # baseline/CoT-isolation runs.
        self.thinking = thinking
        if base_url is None:
            base_url = PROVIDER_BASE_URL[provider]
        if api_keys is not None:
            keys = list(api_keys)
        elif api_key is not None:
            keys = [api_key]
        else:
            keys = load_provider_keys(provider)
            keys = keys[key_index % len(keys):] + keys[:key_index % len(keys)]
        if num_keys > 0:
            keys = keys[:num_keys]
        self.max_retries = max_retries
        self.clients = [OpenAI(base_url=base_url, api_key=k) for k in keys]
        self.client = self.clients[0]  # default for single-shot `ask`

    def ask(self, messages: List[Dict[str, str]], client=None) -> str:
        client = client or self.client
        extra_body = ({"thinking": {"type": self.thinking}}
                      if self.thinking != "auto" else {})
        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                resp = client.chat.completions.create(
                    model=self.model_name, messages=messages,
                    max_tokens=self.max_tokens, temperature=self.temperature,
                    extra_body=extra_body)
                usage = getattr(resp, "usage", None)
                if usage is not None:
                    self._add_usage(getattr(usage, "prompt_tokens", 0),
                                    getattr(usage, "completion_tokens", 0))
                return resp.choices[0].message.content or ""
            except Exception as e:  # noqa: BLE001
                last_err = e
                logger.warning(f"LLM call failed (attempt {attempt+1}/"
                               f"{self.max_retries}): {type(e).__name__}: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
        logger.error(f"LLM call gave up after {self.max_retries}: {last_err}")
        return ""

    def ask_n(self, messages: List[Dict[str, str]], n: int) -> List[str]:
        """Sample n completions concurrently, round-robin over the key pool
        (one in-flight request per key). n <= 0 returns []."""
        if n <= 0:
            return []
        workers = min(n, len(self.clients))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(self.ask, messages,
                                   self.clients[i % len(self.clients)])
                       for i in range(n)]
            return [f.result() for f in futures]


class FakeLLM(BaseLLM):
    """No-network LLM for tests. Returns a fixed `response`, or cycles through
    `responses` if provided (lets repair tests script distinct outputs)."""

    def __init__(self, response: str = "[out:json][timeout:25];out;",
                 responses: Optional[List[str]] = None,
                 prompt_tokens: int = 0, completion_tokens: int = 0) -> None:
        super().__init__(model_name="fake", temperature=0.0, max_tokens=0)
        self._response = response
        self._responses = responses
        self._i = 0
        self._pt = prompt_tokens
        self._ct = completion_tokens

    def ask(self, messages: List[Dict[str, str]]) -> str:
        self._add_usage(self._pt, self._ct)
        if self._responses:
            r = self._responses[self._i % len(self._responses)]
            self._i += 1
            return r
        return self._response
