"""OpenRouter API client for LLM generation and reranking.

Reads ``OPENROUTER_API_KEY`` from environment; fails loudly if missing.
Uses the ``openai`` package (already a project dependency) since OpenRouter
provides an OpenAI-compatible API.

Usage::

    from feg_rag.generation.openrouter_client import OpenRouterClient

    client = OpenRouterClient(model="qwen/qwen-2.5-7b-instruct")
    response = client.chat(messages=[{"role": "user", "content": "Hello"}])
    print(response.content)
    print(response.usage)
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from openai import OpenAI


# ═════════════════════════════════════════════════════════════════════════════
# Data structures
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class TokenUsage:
    """Token usage and cost estimate for one LLM call."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0
    provider: str = "openrouter"
    model: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "estimated_cost_usd": self.estimated_cost_usd,
            "provider": self.provider,
            "model": self.model,
        }


@dataclass
class ChatResponse:
    """Structured response from the OpenRouter chat API."""
    content: str = ""
    raw_response: Dict[str, Any] = field(default_factory=dict)
    usage: TokenUsage = field(default_factory=TokenUsage)
    finish_reason: str = ""
    model_used: str = ""
    latency_seconds: float = 0.0


# ═════════════════════════════════════════════════════════════════════════════
# Cost estimation (approximate, per 1M tokens)
# ═════════════════════════════════════════════════════════════════════════════

_MODEL_PRICING: Dict[str, Dict[str, float]] = {
    "qwen/qwen-2.5-7b-instruct": {"prompt": 0.07, "completion": 0.07},
    "qwen/qwen-2.5-14b-instruct": {"prompt": 0.15, "completion": 0.15},
    "qwen/qwen-2.5-32b-instruct": {"prompt": 0.30, "completion": 0.30},
    "qwen/qwen-2.5-72b-instruct": {"prompt": 0.90, "completion": 0.90},
    "default": {"prompt": 0.10, "completion": 0.10},
}

_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


# ═════════════════════════════════════════════════════════════════════════════
# Client
# ═════════════════════════════════════════════════════════════════════════════

class OpenRouterClient:
    """Reusable OpenRouter chat-completion client via the OpenAI SDK.

    Args:
        model: OpenRouter model slug (e.g. ``"qwen/qwen-2.5-7b-instruct"``).
        temperature: Sampling temperature (default 0.0).
        max_tokens: Maximum completion tokens.
        timeout: HTTP request timeout in seconds.
        max_retries: Maximum retry attempts (handled by OpenAI SDK).
        base_url: Override the OpenRouter base URL.
    """

    def __init__(
        self,
        model: str = "qwen/qwen-2.5-7b-instruct",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        timeout: int = 120,
        max_retries: int = 3,
        base_url: Optional[str] = None,
    ):
        # --- API key ---
        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        if not api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY environment variable is not set.\n"
                "  Bash:   export OPENROUTER_API_KEY=\"...\"\n"
                "  PS:     $env:OPENROUTER_API_KEY=\"...\""
            )

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries

        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url or _OPENROUTER_BASE_URL,
            timeout=timeout,
            max_retries=max_retries,
        )

        # Stats
        self.total_calls: int = 0
        self.cumulative_cost_usd: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> ChatResponse:
        """Send a chat completion request and return structured response.

        Args:
            messages: List of ``{"role": ..., "content": ...}`` dicts.
            temperature: Override instance default.
            max_tokens: Override instance default.
            response_format: Optional ``{"type": "json_object"}`` for JSON mode.

        Returns:
            ``ChatResponse`` with content, usage, and metadata.
        """
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
        }
        if response_format:
            kwargs["response_format"] = response_format

        t_start = time.time()

        completion = self._client.chat.completions.create(**kwargs)

        latency = time.time() - t_start

        # --- Extract content ---
        choice = completion.choices[0]
        content = choice.message.content or ""
        finish_reason = choice.finish_reason or ""

        # --- Extract usage ---
        usage = TokenUsage(
            prompt_tokens=completion.usage.prompt_tokens if completion.usage else 0,
            completion_tokens=completion.usage.completion_tokens if completion.usage else 0,
            total_tokens=completion.usage.total_tokens if completion.usage else 0,
            estimated_cost_usd=self._estimate_cost(
                completion.usage.prompt_tokens if completion.usage else 0,
                completion.usage.completion_tokens if completion.usage else 0,
            ),
            provider="openrouter",
            model=self.model,
        )

        self.total_calls += 1
        self.cumulative_cost_usd += usage.estimated_cost_usd

        return ChatResponse(
            content=content,
            raw_response=completion.model_dump() if hasattr(completion, "model_dump") else {},
            usage=usage,
            finish_reason=finish_reason,
            model_used=completion.model or self.model,
            latency_seconds=latency,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Estimate USD cost from token counts using approximate pricing."""
        pricing = _MODEL_PRICING.get(
            self.model,
            _MODEL_PRICING.get("default", {"prompt": 0.10, "completion": 0.10}),
        )
        prompt_cost = (prompt_tokens / 1_000_000) * pricing["prompt"]
        completion_cost = (completion_tokens / 1_000_000) * pricing["completion"]
        return round(prompt_cost + completion_cost, 8)

    def close(self) -> None:
        """Close the underlying client."""
        self._client.close()

    def __repr__(self) -> str:
        return (
            f"OpenRouterClient(model={self.model!r}, "
            f"calls={self.total_calls}, cost=${self.cumulative_cost_usd:.4f})"
        )
