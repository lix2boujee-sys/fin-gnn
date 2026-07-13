"""Disk-backed cache for LLM responses to support resume.

Cache key is a hash of (experiment_name, method_name, query_id, model_name,
candidate_ids/evidence_ids, prompt_version). This ensures:
  - Same query + same candidates + same model → cache hit.
  - Different experiment or method → cache miss.
  - Prompt changes (version bump) → cache miss.

Usage::

    from feg_rag.generation.llm_cache import LLMCache

    cache = LLMCache("cache/llm_calls")

    # Before calling the LLM:
    cached = cache.get("table3", "ppr", "query_123", "qwen-7b",
                       candidate_ids=["c1","c2"], prompt_version="v1")
    if cached:
        return cached

    # After a successful call:
    cache.put("table3", "ppr", "query_123", "qwen-7b",
              candidate_ids=["c1","c2"], prompt_version="v1",
              response={"ranked_candidate_ids": ["c2","c1"], "rationale": "..."})
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


class LLMCache:
    """Disk-backed cache for LLM call results, keyed by experiment/query/candidates."""

    def __init__(self, cache_dir: str | Path = "cache/llm_calls"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._hits: int = 0
        self._misses: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(
        self,
        experiment_name: str,
        method_name: str,
        query_id: str,
        model_name: str,
        candidate_ids: List[str],
        prompt_version: str = "v1",
    ) -> Optional[Dict[str, Any]]:
        """Retrieve a cached LLM response, or None if not found.

        Args:
            experiment_name: e.g. "table3".
            method_name: e.g. "ppr".
            query_id: FinDER sample ID.
            model_name: OpenRouter model slug.
            candidate_ids: Ordered list of chunk IDs used as input.
            prompt_version: Version tag for prompt changes.

        Returns:
            Cached response dict, or None.
        """
        cache_key = self._make_key(
            experiment_name, method_name, query_id, model_name,
            candidate_ids, prompt_version,
        )
        cache_path = self._cache_path(cache_key)

        if cache_path.exists():
            try:
                with open(cache_path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                self._hits += 1
                return data
            except (json.JSONDecodeError, OSError):
                # Corrupt cache file — delete and treat as miss
                cache_path.unlink(missing_ok=True)

        self._misses += 1
        return None

    def put(
        self,
        experiment_name: str,
        method_name: str,
        query_id: str,
        model_name: str,
        candidate_ids: List[str],
        prompt_version: str,
        response: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Store an LLM response in the cache.

        Args:
            experiment_name: e.g. "table3".
            method_name: e.g. "ppr".
            query_id: FinDER sample ID.
            model_name: OpenRouter model slug.
            candidate_ids: Ordered list of chunk IDs.
            prompt_version: Version tag.
            response: The parsed LLM response dict.
            metadata: Optional extra metadata to store alongside.
        """
        cache_key = self._make_key(
            experiment_name, method_name, query_id, model_name,
            candidate_ids, prompt_version,
        )
        cache_path = self._cache_path(cache_key)

        data = {
            "cache_key": cache_key,
            "experiment": experiment_name,
            "method": method_name,
            "query_id": query_id,
            "model": model_name,
            "prompt_version": prompt_version,
            "response": response,
            "metadata": metadata or {},
        }
        with open(cache_path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False)

    def clear(self) -> int:
        """Delete all cached entries. Returns number of files removed."""
        count = 0
        for f in self.cache_dir.glob("*.json"):
            f.unlink()
            count += 1
        self._hits = 0
        self._misses = 0
        return count

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def hits(self) -> int:
        return self._hits

    @property
    def misses(self) -> int:
        return self._misses

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        if total == 0:
            return 0.0
        return self._hits / total

    @property
    def size(self) -> int:
        """Number of cached entries."""
        return len(list(self.cache_dir.glob("*.json")))

    def stats(self) -> Dict[str, Any]:
        return {
            "cache_dir": str(self.cache_dir),
            "entries": self.size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self.hit_rate, 4),
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _make_key(
        experiment_name: str,
        method_name: str,
        query_id: str,
        model_name: str,
        candidate_ids: List[str],
        prompt_version: str,
    ) -> str:
        """Produce a deterministic hash key."""
        canonical = json.dumps({
            "experiment": experiment_name,
            "method": method_name,
            "query_id": query_id,
            "model": model_name,
            # Preserve order: LLM prompts are order-sensitive, and generation
            # top-k evidence order is part of the experimental condition.
            "candidate_ids": list(candidate_ids),
            "prompt_version": prompt_version,
        }, sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(canonical.encode()).hexdigest()[:32]

    def _cache_path(self, cache_key: str) -> Path:
        return self.cache_dir / f"{cache_key}.json"
