"""Robust JSON response parser for LLM outputs.

Handles the common case where LLMs wrap JSON in markdown code fences,
add trailing commas, or include extra text before/after the JSON object.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ParseResult:
    """Result of parsing an LLM JSON response."""

    success: bool
    parsed: Dict[str, Any] = field(default_factory=dict)
    raw_text: str = ""
    error_message: str = ""
    recovery_attempted: bool = False
    recovery_succeeded: bool = False


# Regex to find JSON objects (greedy match for outermost braces)
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
# Markdown code fence: ```json ... ```
_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?```", re.DOTALL)


def parse_json_response(raw_text: str) -> ParseResult:
    """Parse JSON from an LLM response with conservative recovery.

    Strategy (in order):
      1. Direct ``json.loads`` on the raw text.
      2. Strip markdown code fences and try again.
      3. Extract the first JSON object with regex and try again.
      4. Fix common JSON issues (trailing commas, single quotes) and retry.

    Args:
        raw_text: The raw string response from the LLM.

    Returns:
        ``ParseResult`` with ``success=True`` and ``parsed`` dict if parsing
        succeeded, or ``success=False`` with ``error_message`` otherwise.
    """
    text = raw_text.strip()
    if not text:
        return ParseResult(
            success=False,
            raw_text=raw_text,
            error_message="Empty response text",
        )

    # Strategy 1: Direct parse
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return ParseResult(success=True, parsed=parsed, raw_text=raw_text)
    except json.JSONDecodeError:
        pass

    # Strategy 2: Remove code fences
    fence_match = _CODE_FENCE_RE.search(text)
    if fence_match:
        try:
            parsed = json.loads(fence_match.group(1).strip())
            if isinstance(parsed, dict):
                return ParseResult(
                    success=True, parsed=parsed, raw_text=raw_text,
                    recovery_attempted=True, recovery_succeeded=True,
                )
        except json.JSONDecodeError:
            pass

    # Strategy 3: Extract first JSON object with regex
    obj_match = _JSON_OBJECT_RE.search(text)
    if obj_match:
        try:
            parsed = json.loads(obj_match.group(0))
            if isinstance(parsed, dict):
                return ParseResult(
                    success=True, parsed=parsed, raw_text=raw_text,
                    recovery_attempted=True, recovery_succeeded=True,
                )
        except json.JSONDecodeError:
            pass

    # Strategy 4: Fix common JSON issues
    if obj_match:
        candidate = obj_match.group(0)
        fixed = _repair_json(candidate)
        if fixed:
            try:
                parsed = json.loads(fixed)
                if isinstance(parsed, dict):
                    return ParseResult(
                        success=True, parsed=parsed, raw_text=raw_text,
                        recovery_attempted=True, recovery_succeeded=True,
                    )
            except json.JSONDecodeError:
                pass

    return ParseResult(
        success=False,
        raw_text=raw_text,
        error_message="Failed to parse JSON from LLM response after all recovery attempts",
        recovery_attempted=True,
        recovery_succeeded=False,
    )


def parse_reranker_response(raw_text: str) -> ParseResult:
    """Parse a reranker response, expecting:
    ``{"ranked_candidate_ids": [...], "rationale": "..."}``
    """
    result = parse_json_response(raw_text)
    if result.success:
        # Validate expected keys
        if "ranked_candidate_ids" not in result.parsed:
            # Try to recover: if only a list is returned, wrap it
            if isinstance(result.parsed, list):
                result.parsed = {
                    "ranked_candidate_ids": result.parsed,
                    "rationale": "",
                }
            else:
                result.success = False
                result.error_message = (
                    "Parsed JSON missing required key 'ranked_candidate_ids'"
                )
    return result


def parse_generator_response(raw_text: str) -> ParseResult:
    """Parse a generator response, expecting:
    ``{"answer": "...", "evidence_ids_used": [...], "confidence": "..."}``
    """
    result = parse_json_response(raw_text)
    if result.success:
        # Validate expected keys
        missing = [
            k for k in ("answer", "evidence_ids_used", "confidence")
            if k not in result.parsed
        ]
        if missing:
            # If only "answer" is present, add defaults
            if "answer" in result.parsed:
                result.parsed.setdefault("evidence_ids_used", [])
                result.parsed.setdefault("confidence", "medium")
            else:
                result.success = False
                result.error_message = (
                    f"Parsed JSON missing required keys: {missing}"
                )
    return result


# ═════════════════════════════════════════════════════════════════════════════
# JSON repair helpers
# ═════════════════════════════════════════════════════════════════════════════

def _repair_json(text: str) -> Optional[str]:
    """Attempt to fix common JSON formatting errors.

    Fixes applied:
      - Remove trailing commas before ``}`` or ``]``.
      - Replace single quotes with double quotes (conservative).
      - Escape unescaped newlines inside string values.
    """
    fixed = text

    # Remove trailing commas before closing brackets/braces
    fixed = re.sub(r",\s*([}\]])", r"\1", fixed)

    # Replace single quotes used for keys/values (conservative)
    # Only if the text has more single-quoted keys than double-quoted
    if fixed.count("'") > fixed.count('"') * 2:
        # Replace single quotes around keys: 'key':
        fixed = re.sub(r"'([^']+)'\s*:", r'"\1":', fixed)
        # Replace single quotes around simple string values
        fixed = re.sub(r":\s*'([^']*)'", r': "\1"', fixed)

    # Fix unescaped control characters in strings
    # (Naive: remove raw newlines/tabs inside string values)
    fixed = re.sub(r'(?<="[^"]*)\n(?=[^"]*")', r"\\n", fixed)

    return fixed if fixed != text else None
