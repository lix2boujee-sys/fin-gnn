"""Deterministic answer quality evaluator for Experiment 3.

Computes answer-level metrics without using an LLM judge:
  - Answer Accuracy (normalized string match)
  - Numerical Consistency (number extraction + comparison)
  - Faithfulness (answer claims supported by evidence)
  - Evidence Hit@5 (gold evidence appears in top-5)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from feg_rag.data.chunker import Chunk


# ═════════════════════════════════════════════════════════════════════════════
# Regex patterns
# ═════════════════════════════════════════════════════════════════════════════

_NUMBER_RE = re.compile(
    r"\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b"
    r"|\b\d+\.?\d*\s*(?:million|billion|thousand|trillion)\b"
    r"|\b\d+\.?\d*\s*%"
    r"|\$\s*\d{1,3}(?:,\d{3})*(?:\.\d+)?\b",
    re.IGNORECASE,
)

_UNIT_RE = re.compile(
    r"\b(million|billion|thousand|trillion|percent|%|dollars?\$?)\b",
    re.IGNORECASE,
)

_YEAR_RE = re.compile(r"\b((?:19|20)\d{2})\b")


# ═════════════════════════════════════════════════════════════════════════════
# Data structures
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class AnswerEvalResult:
    """Per-answer evaluation result."""

    question_id: str
    generated_answer: str
    gold_answer: str

    # Accuracy
    exact_match: bool = False
    relaxed_match: bool = False
    answer_accuracy: float = 0.0  # 1.0 = exact, 0.5 = relaxed, 0.0 = wrong

    # Numerical consistency
    numbers_in_generated: List[str] = field(default_factory=list)
    numbers_in_gold: List[str] = field(default_factory=list)
    numbers_matched_count: int = 0
    numerical_consistency: float = 0.0  # fraction of gold numbers matched

    # Evidence support
    evidence_hit_at_5: bool = False
    evidence_ids_used: List[str] = field(default_factory=list)
    gold_evidence_ids: List[str] = field(default_factory=list)

    # Faithfulness (simple: does answer contain numbers NOT in evidence)
    numbers_in_evidence: List[str] = field(default_factory=list)
    numbers_in_answer_not_in_evidence: int = 0
    faithfulness_score: float = 1.0  # 1.0 = all answer numbers found in evidence

    # Metadata
    is_insufficient_evidence: bool = False
    parse_failed: bool = False
    issues: List[str] = field(default_factory=list)


@dataclass
class AggregateEvalResult:
    """Aggregated evaluation across all queries for one method."""

    method_name: str
    num_samples: int = 0
    num_parse_failures: int = 0

    answer_accuracy: float = 0.0
    exact_match_rate: float = 0.0
    relaxed_match_rate: float = 0.0
    numerical_consistency: float = 0.0
    faithfulness_score: float = 0.0
    evidence_hit_at_5: float = 0.0
    insufficient_evidence_rate: float = 0.0

    @classmethod
    def from_results(
        cls,
        method_name: str,
        per_answer: List[AnswerEvalResult],
    ) -> "AggregateEvalResult":
        n = len(per_answer)
        if n == 0:
            return cls(method_name=method_name)

        return cls(
            method_name=method_name,
            num_samples=n,
            num_parse_failures=sum(1 for r in per_answer if r.parse_failed),
            answer_accuracy=float(np.mean([r.answer_accuracy for r in per_answer])),
            exact_match_rate=float(np.mean([float(r.exact_match) for r in per_answer])),
            relaxed_match_rate=float(np.mean([float(r.relaxed_match) for r in per_answer])),
            numerical_consistency=float(np.mean([r.numerical_consistency for r in per_answer])),
            faithfulness_score=float(np.mean([r.faithfulness_score for r in per_answer])),
            evidence_hit_at_5=float(np.mean([float(r.evidence_hit_at_5) for r in per_answer])),
            insufficient_evidence_rate=float(np.mean([float(r.is_insufficient_evidence) for r in per_answer])),
        )

    def to_dict(self) -> Dict:
        return {
            "method": self.method_name,
            "num_samples": self.num_samples,
            "num_parse_failures": self.num_parse_failures,
            "answer_accuracy": round(self.answer_accuracy, 4),
            "exact_match_rate": round(self.exact_match_rate, 4),
            "relaxed_match_rate": round(self.relaxed_match_rate, 4),
            "numerical_consistency": round(self.numerical_consistency, 4),
            "faithfulness_score": round(self.faithfulness_score, 4),
            "evidence_hit_at_5": round(self.evidence_hit_at_5, 4),
            "insufficient_evidence_rate": round(self.insufficient_evidence_rate, 4),
        }


# ═════════════════════════════════════════════════════════════════════════════
# Evaluator
# ═════════════════════════════════════════════════════════════════════════════

class AnswerEvaluator:
    """Deterministic answer quality evaluator — no LLM judge."""

    def evaluate(
        self,
        question_id: str,
        generated_answer: str,
        gold_answer: str,
        evidence_chunks: List[Chunk],
        gold_evidence_ids: List[str],
        evidence_ids_used: Optional[List[str]] = None,
    ) -> AnswerEvalResult:
        """Evaluate one generated answer against gold answer and evidence.

        Args:
            question_id: Sample ID.
            generated_answer: The LLM-generated answer text.
            gold_answer: The ground-truth answer.
            evidence_chunks: The top-k evidence chunks used for generation.
            gold_evidence_ids: The ground-truth evidence chunk IDs.
            evidence_ids_used: Evidence IDs the LLM claimed to use.

        Returns:
            ``AnswerEvalResult`` with all metrics.
        """
        result = AnswerEvalResult(
            question_id=question_id,
            generated_answer=generated_answer,
            gold_answer=gold_answer,
            evidence_ids_used=evidence_ids_used or [],
            gold_evidence_ids=gold_evidence_ids,
        )

        gen = generated_answer.strip()
        if not gen:
            result.parse_failed = True
            result.issues.append("Empty generated answer")
            return result

        # Check insufficient evidence
        if "INSUFFICIENT_EVIDENCE" in gen.upper():
            result.is_insufficient_evidence = True

        # ---- Accuracy ----
        gen_norm = _normalize(gen)
        gold_norm = _normalize(gold_answer)

        result.exact_match = gen_norm == gold_norm
        result.relaxed_match = _relaxed_match(gen_norm, gold_norm)

        if result.exact_match:
            result.answer_accuracy = 1.0
        elif result.relaxed_match:
            result.answer_accuracy = 0.5
        else:
            result.answer_accuracy = 0.0

        # ---- Numerical consistency ----
        result.numbers_in_generated = _extract_numbers(gen)
        result.numbers_in_gold = _extract_numbers(gold_answer)

        if result.numbers_in_gold:
            matched = 0
            gold_nums_normalized = {_normalize_number(n) for n in result.numbers_in_gold}
            for n in result.numbers_in_generated:
                if _normalize_number(n) in gold_nums_normalized:
                    matched += 1
            result.numbers_matched_count = matched
            result.numerical_consistency = matched / len(result.numbers_in_gold)
        else:
            # No numbers in gold → perfect consistency
            result.numerical_consistency = 1.0

        # ---- Evidence Hit@5 ----
        retrieved_ids = [c.chunk_id for c in evidence_chunks[:5]]
        if gold_evidence_ids:
            result.evidence_hit_at_5 = bool(set(retrieved_ids) & set(gold_evidence_ids))

        # ---- Faithfulness ----
        evidence_text = " ".join(c.text for c in evidence_chunks)
        ev_numbers = set(_normalize_number(n) for n in _extract_numbers(evidence_text))

        result.numbers_in_evidence = list(ev_numbers)
        ans_numbers_normalized = {_normalize_number(n) for n in result.numbers_in_generated}
        result.numbers_in_answer_not_in_evidence = len(ans_numbers_normalized - ev_numbers)

        if ans_numbers_normalized:
            result.faithfulness_score = 1.0 - (
                result.numbers_in_answer_not_in_evidence / len(ans_numbers_normalized)
            )
        else:
            result.faithfulness_score = 1.0

        # Collect issues
        if not result.exact_match and not result.relaxed_match:
            result.issues.append("Answer does not match gold (exact or relaxed)")
        if result.numerical_consistency < 1.0:
            result.issues.append(
                f"Numerical inconsistency: {result.numbers_matched_count}/"
                f"{len(result.numbers_in_gold)} numbers matched"
            )
        if result.numbers_in_answer_not_in_evidence > 0:
            result.issues.append(
                f"Faithfulness: {result.numbers_in_answer_not_in_evidence} numbers "
                f"in answer not found in evidence"
            )
        if not result.evidence_hit_at_5:
            result.issues.append("No gold evidence in top-5 retrieved")

        return result


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════

def _normalize(text: str) -> str:
    """Normalize answer text: lowercase, collapse whitespace, strip trailing period."""
    text = text.strip()
    # Remove markdown code fences if present
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```", "", text)
    return re.sub(r"\s+", " ", text.lower().strip().rstrip("."))


def _relaxed_match(gen: str, gold: str) -> bool:
    """Check if generated answer contains the gold answer as a substring,
    or if they share substantial content."""
    if not gen or not gold:
        return False
    # Gold contained in generated
    if gold in gen:
        return True
    # Generated contained in gold
    if gen in gold:
        return True
    # Shared token ratio
    gen_tokens = set(gen.split())
    gold_tokens = set(gold.split())
    if not gold_tokens:
        return False
    intersection = gen_tokens & gold_tokens
    return len(intersection) / len(gold_tokens) >= 0.8


def _extract_numbers(text: str) -> List[str]:
    """Extract number strings from text."""
    return [m.group(0).strip() for m in _NUMBER_RE.finditer(text)]


def _normalize_number(num_str: str) -> str:
    """Normalize a number string for comparison.

    Handles: commas, dollar signs, percent signs, unit words.
    """
    s = num_str.lower().strip()
    # Remove $ sign
    s = s.replace("$", "")
    # Remove commas in numbers
    s = re.sub(r"(\d),(\d)", r"\1\2", s)
    # Normalize unit words to abbreviations
    s = s.replace("million", "m").replace("billion", "b")
    s = s.replace("thousand", "k").replace("trillion", "t")
    # Collapse whitespace
    s = re.sub(r"\s+", "", s)
    return s
