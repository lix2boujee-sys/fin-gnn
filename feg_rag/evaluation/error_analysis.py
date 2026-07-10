"""Error analysis for financial RAG failures.

Paper plan §13: classify errors into structured categories.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

from feg_rag.data.chunker import Chunk


# ═════════════════════════════════════════════════════════════════════════════
# Error types
# ═════════════════════════════════════════════════════════════════════════════

class ErrorType(str, Enum):
    """Structured error categories (paper plan §13)."""

    WRONG_YEAR = "wrong_year"
    WRONG_METRIC = "wrong_metric"
    WRONG_TABLE = "wrong_table"
    WRONG_COMPANY = "wrong_company"
    MISSING_TABLE = "missing_table_evidence"
    ARITHMETIC_ERROR = "arithmetic_error"
    UNIT_ERROR = "unit_error"
    UNSUPPORTED = "unsupported_generation"
    OTHER = "other"


ERROR_DESCRIPTIONS = {
    ErrorType.WRONG_YEAR: "Retrieved evidence for wrong fiscal year",
    ErrorType.WRONG_METRIC: "Retrieved wrong financial metric (e.g., revenue vs operating income)",
    ErrorType.WRONG_TABLE: "Retrieved wrong table (e.g., balance sheet vs income statement)",
    ErrorType.WRONG_COMPANY: "Retrieved evidence from wrong company/filing",
    ErrorType.MISSING_TABLE: "Only text evidence retrieved; table evidence missing",
    ErrorType.ARITHMETIC_ERROR: "Numbers correct but calculation wrong",
    ErrorType.UNIT_ERROR: "Unit confusion (million/billion, absolute/percentage)",
    ErrorType.UNSUPPORTED: "Answer contains claims not found in any evidence",
    ErrorType.OTHER: "Error does not fit above categories",
}


@dataclass
class ErrorCase:
    """A single error case with its classification."""

    question_id: str
    question: str
    gold_answer: str
    generated_answer: str
    retrieved_chunk_ids: List[str]
    gold_evidence_ids: List[str]
    error_type: ErrorType
    description: str = ""


# ═════════════════════════════════════════════════════════════════════════════
# Analyzer
# ═════════════════════════════════════════════════════════════════════════════

class ErrorAnalyzer:
    """Classify and analyse RAG errors."""

    _YEAR_RE = re.compile(r"\b((?:19|20)\d{2})\b")

    _METRIC_PATTERNS = [
        r"\b(revenue|net\s+income|operating\s+income|eps|ebitda)\b",
        r"\b(gross\s+profit|total\s+assets|cash\s+flow)\b",
    ]
    _METRIC_RE = re.compile("|".join(_METRIC_PATTERNS), re.IGNORECASE)

    _UNIT_RE = re.compile(
        r"\b(million|billion|thousand|trillion|percent|%)\b", re.IGNORECASE
    )

    def analyze(
        self,
        results: List[Dict],
        chunk_map: Optional[Dict[str, Chunk]] = None,
    ) -> Tuple[List[ErrorCase], Dict[ErrorType, int]]:
        """Classify all error cases in results.

        Args:
            results: List of per-sample result dicts. Each must have at minimum
                     question_id, question, gold_answer, generated_answer,
                     retrieved_chunk_ids, gold_evidence_ids, and answer_is_correct.
            chunk_map: Optional mapping from chunk_id → Chunk for content inspection.

        Returns:
            (error_cases, error_type_counts)
        """
        errors: List[ErrorCase] = []
        counts: Counter[ErrorType] = Counter()

        for r in results:
            if r.get("answer_is_correct", False):
                continue

            error_type = self._classify(r, chunk_map)
            errors.append(
                ErrorCase(
                    question_id=r.get("question_id", ""),
                    question=r.get("question", ""),
                    gold_answer=r.get("gold_answer", ""),
                    generated_answer=r.get("generated_answer", ""),
                    retrieved_chunk_ids=r.get("retrieved_chunk_ids", []),
                    gold_evidence_ids=r.get("gold_evidence_ids", []),
                    error_type=error_type,
                    description=ERROR_DESCRIPTIONS.get(error_type, ""),
                )
            )
            counts[error_type] += 1

        return errors, dict(counts)

    def _classify(
        self, r: Dict, chunk_map: Optional[Dict[str, Chunk]] = None
    ) -> ErrorType:
        """Classify a single error case."""
        question = r.get("question", "")
        generated = r.get("generated_answer", "")
        gold = r.get("gold_answer", "")
        retrieved_ids = r.get("retrieved_chunk_ids", [])
        gold_ids = r.get("gold_evidence_ids", [])

        # Check: unsupported generation
        if not retrieved_ids:
            return ErrorType.UNSUPPORTED

        # Check: year mismatch
        q_years = set(self._YEAR_RE.findall(question))
        g_years = set(self._YEAR_RE.findall(generated))
        if q_years and g_years and not (q_years & g_years):
            return ErrorType.WRONG_YEAR

        # Check: metric mismatch (simplified)
        q_metrics = set(
            m.group(0).lower() for m in self._METRIC_RE.finditer(question)
        )
        g_metrics = set(
            m.group(0).lower() for m in self._METRIC_RE.finditer(generated)
        )
        if q_metrics and g_metrics and not (q_metrics & g_metrics):
            return ErrorType.WRONG_METRIC

        # Check: unit mismatch
        q_units = set(
            m.group(0).lower() for m in self._UNIT_RE.finditer(question)
        )
        g_units = set(
            m.group(0).lower() for m in self._UNIT_RE.finditer(generated)
        )
        if q_units and g_units and q_units != g_units:
            return ErrorType.UNIT_ERROR

        # Check: arithmetic error (heuristic: numbers appear but answer wrong)
        num_re = re.compile(r"\d+")
        if num_re.search(question) and num_re.search(generated):
            # If numbers are involved and answer is wrong, likely arithmetic
            if gold.replace(",", "") != generated.replace(",", ""):
                return ErrorType.ARITHMETIC_ERROR

        # Check: no evidence overlap
        if gold_ids and not (set(gold_ids) & set(retrieved_ids)):
            return ErrorType.MISSING_TABLE

        return ErrorType.OTHER

    def report(
        self, errors: List[ErrorCase], counts: Dict[ErrorType, int]
    ) -> str:
        """Generate a human-readable error analysis report."""
        total = sum(counts.values())
        if total == 0:
            return "No errors to report."

        lines = ["=" * 60, "ERROR ANALYSIS REPORT", "=" * 60, ""]
        lines.append(f"Total errors: {total}")
        lines.append("")

        for etype in ErrorType:
            count = counts.get(etype, 0)
            pct = count / total * 100 if total > 0 else 0
            desc = ERROR_DESCRIPTIONS.get(etype, "")
            lines.append(f"  {etype.value:25s}  {count:4d}  ({pct:5.1f}%)  {desc}")

        lines.append("")
        lines.append("—" * 60)
        lines.append("Example errors (up to 3 per type):")

        for etype in ErrorType:
            examples = [e for e in errors if e.error_type == etype][:3]
            if examples:
                lines.append(f"\n[{etype.value}]")
                for ex in examples:
                    lines.append(f"  Q: {ex.question[:120]}")
                    lines.append(f"  Gold: {ex.gold_answer[:80]}")
                    lines.append(f"  Pred: {ex.generated_answer[:80]}")
                    lines.append("")

        return "\n".join(lines)
