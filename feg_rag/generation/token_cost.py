"""Token cost tracking and reporting for LLM experiments.

Accumulates usage across multiple LLM calls and produces per-method and
overall cost summaries.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from feg_rag.generation.openrouter_client import TokenUsage


@dataclass
class CostSummary:
    """Aggregated cost summary for a method or experiment."""

    method: str
    num_calls: int = 0
    num_failures: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0

    @property
    def avg_tokens_per_call(self) -> float:
        if self.num_calls == 0:
            return 0.0
        return self.total_tokens / self.num_calls

    @property
    def failure_rate(self) -> float:
        total = self.num_calls + self.num_failures
        if total == 0:
            return 0.0
        return self.num_failures / total

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "num_calls": self.num_calls,
            "num_failures": self.num_failures,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "estimated_cost_usd": round(self.estimated_cost_usd, 6),
            "avg_tokens_per_call": round(self.avg_tokens_per_call, 1),
            "failure_rate": round(self.failure_rate, 4),
        }


class TokenCostTracker:
    """Accumulate token usage across multiple methods and LLM calls."""

    def __init__(self):
        self._methods: Dict[str, CostSummary] = {}
        self._all_usages: List[Dict[str, Any]] = []

    def record(
        self,
        method: str,
        usage: TokenUsage,
        success: bool = True,
        query_id: str = "",
    ) -> None:
        """Record one LLM call's token usage.

        Args:
            method: Method name (e.g. "llm_rerank_top50").
            usage: TokenUsage from the OpenRouterClient response.
            success: Whether the call produced a valid result.
            query_id: Optional query identifier for per-call tracking.
        """
        if method not in self._methods:
            self._methods[method] = CostSummary(method=method)

        summary = self._methods[method]
        if success:
            summary.num_calls += 1
            summary.prompt_tokens += usage.prompt_tokens
            summary.completion_tokens += usage.completion_tokens
            summary.total_tokens += usage.total_tokens
            summary.estimated_cost_usd += usage.estimated_cost_usd
        else:
            summary.num_failures += 1
            # Still count tokens for failed calls (they were consumed)
            summary.prompt_tokens += usage.prompt_tokens
            summary.completion_tokens += usage.completion_tokens
            summary.total_tokens += usage.total_tokens
            summary.estimated_cost_usd += usage.estimated_cost_usd

        self._all_usages.append({
            "query_id": query_id,
            "method": method,
            "success": success,
            **usage.to_dict(),
        })

    def get_summary(self, method: str) -> Optional[CostSummary]:
        """Get cost summary for a single method."""
        return self._methods.get(method)

    def get_all_summaries(self) -> List[CostSummary]:
        """Get sorted list of all method summaries (most expensive first)."""
        return sorted(
            self._methods.values(),
            key=lambda x: x.estimated_cost_usd,
            reverse=True,
        )

    def get_total_cost(self) -> float:
        """Total estimated cost across all methods."""
        return sum(s.estimated_cost_usd for s in self._methods.values())

    def save_csv(self, path: Path) -> None:
        """Write a cost summary CSV file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        summaries = self.get_all_summaries()
        with open(path, "w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=[
                "method", "num_calls", "num_failures", "prompt_tokens",
                "completion_tokens", "total_tokens", "estimated_cost_usd",
                "avg_tokens_per_call", "failure_rate",
            ])
            writer.writeheader()
            for s in summaries:
                writer.writerow(s.to_dict())
            # Total row
            writer.writerow({
                "method": "TOTAL",
                "num_calls": sum(s.num_calls for s in summaries),
                "num_failures": sum(s.num_failures for s in summaries),
                "prompt_tokens": sum(s.prompt_tokens for s in summaries),
                "completion_tokens": sum(s.completion_tokens for s in summaries),
                "total_tokens": sum(s.total_tokens for s in summaries),
                "estimated_cost_usd": round(sum(s.estimated_cost_usd for s in summaries), 6),
                "avg_tokens_per_call": "",
                "failure_rate": "",
            })

    def save_json(self, path: Path) -> None:
        """Write full per-call usage records as JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({
                "summaries": [s.to_dict() for s in self.get_all_summaries()],
                "total_cost_usd": round(self.get_total_cost(), 6),
                "calls": self._all_usages,
            }, fh, indent=2, ensure_ascii=False)

    def print_summary(self) -> None:
        """Print a human-readable cost summary to stdout."""
        summaries = self.get_all_summaries()
        if not summaries:
            print("  (no LLM calls recorded)")
            return

        print(f"\n{'=' * 70}")
        print(f"  TOKEN COST SUMMARY")
        print(f"{'=' * 70}")
        header = (
            f"{'Method':<30} {'Calls':>6} {'Fail':>6} "
            f"{'Prompt':>10} {'Compl':>10} {'Total':>10} {'Cost USD':>10}"
        )
        print(header)
        print("-" * 70)
        for s in summaries:
            row = (
                f"{s.method:<30} {s.num_calls:>6} {s.num_failures:>6} "
                f"{s.prompt_tokens:>10,} {s.completion_tokens:>10,} "
                f"{s.total_tokens:>10,} ${s.estimated_cost_usd:>9.4f}"
            )
            print(row)
        print("-" * 70)
        total = self.get_total_cost()
        total_tokens = sum(s.total_tokens for s in summaries)
        total_calls = sum(s.num_calls for s in summaries)
        print(
            f"{'TOTAL':<30} {total_calls:>6} {'':>6} "
            f"{'':>10} {'':>10} {total_tokens:>10,} ${total:>9.4f}"
        )
        print(f"{'=' * 70}\n")
