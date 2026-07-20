"""Deterministic answer quality evaluator (v2).

Computes answer-level metrics without using an LLM judge:
  - Answer Accuracy (normalized string match, token F1, numeric metrics)
  - Numerical Consistency (number extraction + comparison)
  - Faithfulness (answer claims supported by evidence)
  - Evidence retrieval metrics (Hit@5, Recall@5, All-Gold@5, MRR)
  - Abstention handling (abstentions do NOT get faithfulness/numcon credit)

Key changes in v2 (P0-2):
  - Abstentions: faithfulness=None, numerical_consistency=None
  - New metrics: answer_rate, accuracy_all, accuracy_answered, etc.
  - P0-11: requires_arithmetic_verification flag
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


# =============================================================================
# Regex patterns
# =============================================================================

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

_INSUFFICIENT_RE = re.compile(r"insufficient\s*evidence", re.IGNORECASE)


# =============================================================================
# Data structures
# =============================================================================

@dataclass
class AnswerEvalResult:
    """Per-answer evaluation result (v2)."""

    question_id: str
    generated_answer: str
    gold_answer: str

    # Accuracy
    exact_match: bool = False
    relaxed_match: bool = False
    answer_accuracy: float = 0.0  # legacy: 1.0 exact, 0.5 relaxed, 0.0 wrong
    normalized_exact_match: float = 0.0  # P0-10
    token_f1: float = 0.0  # P0-10
    numeric_exact_match: float = 0.0
    numeric_tolerance_match: float = 0.0
    unit_match: float = 0.0

    # Numerical consistency (None for abstentions, P0-2)
    numbers_in_generated: List[str] = field(default_factory=list)
    numbers_in_gold: List[str] = field(default_factory=list)
    numbers_matched_count: int = 0
    numerical_consistency: Optional[float] = None

    # Evidence support
    evidence_hit_at_5: bool = False
    evidence_recall_at_5: float = 0.0  # P0-9
    all_gold_covered_at_5: bool = False  # P0-9
    evidence_mrr: float = 0.0  # P0-9
    evidence_ids_used: List[str] = field(default_factory=list)
    gold_evidence_ids: List[str] = field(default_factory=list)

    # Faithfulness (None for abstentions, P0-2)
    numbers_in_evidence: List[str] = field(default_factory=list)
    numbers_in_answer_not_in_evidence: int = 0
    faithfulness_score: Optional[float] = None
    requires_arithmetic_verification: bool = False  # P0-11

    # Abstention (P0-2)
    is_abstention: bool = False
    answered: int = 1

    # Metadata
    is_insufficient_evidence: bool = False
    parse_failed: bool = False
    issues: List[str] = field(default_factory=list)


@dataclass
class AggregateEvalResult:
    """Aggregated evaluation across all queries for one method (v2)."""

    method_name: str
    num_samples: int = 0
    num_parse_failures: int = 0

    # Answer rate (P0-2)
    answer_rate: float = 0.0
    insufficient_evidence_rate: float = 0.0

    # Accuracy
    answer_accuracy: float = 0.0  # legacy
    exact_match_rate: float = 0.0
    relaxed_match_rate: float = 0.0
    normalized_exact_match: float = 0.0  # P0-10
    token_f1: float = 0.0
    numeric_exact_match: float = 0.0
    numeric_tolerance_match: float = 0.0
    unit_match: float = 0.0

    # Accuracy (answered only, P0-2)
    answer_accuracy_answered: float = 0.0
    token_f1_answered: float = 0.0

    # Numerical consistency (answered, applicable only, P0-2)
    numerical_consistency: float = 0.0
    numerical_applicable_count: int = 0

    # Faithfulness (answered only, P0-2)
    faithfulness_score: float = 0.0
    unsupported_rate_answered: float = 0.0

    # Evidence retrieval
    evidence_hit_at_5: float = 0.0
    evidence_recall_at_5: float = 0.0  # P0-9
    all_gold_covered_at_5: float = 0.0  # P0-9
    evidence_mrr: float = 0.0  # P0-9

    # P0-11
    arithmetic_verification_required_rate: float = 0.0

    @classmethod
    def from_results(
        cls,
        method_name: str,
        per_answer: List[AnswerEvalResult],
    ) -> "AggregateEvalResult":
        n = len(per_answer)
        if n == 0:
            return cls(method_name=method_name)

        # Answered subset
        answered = [r for r in per_answer if r.answered == 1]
        n_answered = len(answered)

        # Numerical applicable (answered only)
        num_applicable = [r for r in answered
                          if r.numerical_consistency is not None]

        # Faithfulness subset (answered, non-None)
        faith_subset = [r for r in answered if r.faithfulness_score is not None]

        return cls(
            method_name=method_name,
            num_samples=n,
            num_parse_failures=sum(1 for r in per_answer if r.parse_failed),
            answer_rate=n_answered / max(n, 1),
            insufficient_evidence_rate=float(
                np.mean([float(r.is_insufficient_evidence) for r in per_answer])
            ),
            # Accuracy (all)
            answer_accuracy=float(np.mean([r.answer_accuracy for r in per_answer])),
            exact_match_rate=float(np.mean([float(r.exact_match) for r in per_answer])),
            relaxed_match_rate=float(np.mean([float(r.relaxed_match) for r in per_answer])),
            normalized_exact_match=float(np.mean([r.normalized_exact_match for r in per_answer])),
            token_f1=float(np.mean([r.token_f1 for r in per_answer])),
            numeric_exact_match=float(np.mean([r.numeric_exact_match for r in per_answer])),
            numeric_tolerance_match=float(np.mean([r.numeric_tolerance_match for r in per_answer])),
            unit_match=float(np.mean([r.unit_match for r in per_answer])),
            # Accuracy (answered only)
            answer_accuracy_answered=float(
                np.mean([r.answer_accuracy for r in answered])
            ) if n_answered > 0 else 0.0,
            token_f1_answered=float(
                np.mean([r.token_f1 for r in answered])
            ) if n_answered > 0 else 0.0,
            # Numerical consistency
            numerical_consistency=float(
                np.mean([r.numerical_consistency for r in num_applicable
                         if r.numerical_consistency is not None])
            ) if num_applicable else 0.0,
            numerical_applicable_count=len(num_applicable),
            # Faithfulness
            faithfulness_score=float(
                np.mean([r.faithfulness_score for r in faith_subset
                         if r.faithfulness_score is not None])
            ) if faith_subset else 0.0,
            unsupported_rate_answered=float(
                np.mean([float(r.numbers_in_answer_not_in_evidence > 0) for r in answered])
            ) if n_answered > 0 else 0.0,
            # Evidence retrieval
            evidence_hit_at_5=float(
                np.mean([float(r.evidence_hit_at_5) for r in per_answer])
            ),
            evidence_recall_at_5=float(
                np.mean([r.evidence_recall_at_5 for r in per_answer])
            ),
            all_gold_covered_at_5=float(
                np.mean([float(r.all_gold_covered_at_5) for r in per_answer])
            ),
            evidence_mrr=float(
                np.mean([r.evidence_mrr for r in per_answer])
            ),
            # P0-11
            arithmetic_verification_required_rate=float(
                np.mean([float(r.requires_arithmetic_verification) for r in per_answer])
            ),
        )

    def to_dict(self) -> Dict:
        return {
            "method": self.method_name,
            "num_samples": self.num_samples,
            "num_parse_failures": self.num_parse_failures,
            "answer_rate": round(self.answer_rate, 4),
            "insufficient_evidence_rate": round(self.insufficient_evidence_rate, 4),
            "answer_accuracy": round(self.answer_accuracy, 4),
            "answer_accuracy_answered": round(self.answer_accuracy_answered, 4),
            "exact_match_rate": round(self.exact_match_rate, 4),
            "relaxed_match_rate": round(self.relaxed_match_rate, 4),
            "normalized_exact_match": round(self.normalized_exact_match, 4),
            "token_f1": round(self.token_f1, 4),
            "token_f1_answered": round(self.token_f1_answered, 4),
            "numeric_exact_match": round(self.numeric_exact_match, 4),
            "numeric_tolerance_match": round(self.numeric_tolerance_match, 4),
            "unit_match": round(self.unit_match, 4),
            "numerical_consistency": round(self.numerical_consistency, 4),
            "numerical_applicable_count": self.numerical_applicable_count,
            "faithfulness_score": round(self.faithfulness_score, 4),
            "unsupported_rate_answered": round(self.unsupported_rate_answered, 4),
            "evidence_hit_at_5": round(self.evidence_hit_at_5, 4),
            "evidence_recall_at_5": round(self.evidence_recall_at_5, 4),
            "all_gold_covered_at_5": round(self.all_gold_covered_at_5, 4),
            "evidence_mrr": round(self.evidence_mrr, 4),
            "arithmetic_verification_required_rate": round(
                self.arithmetic_verification_required_rate, 4
            ),
        }


# =============================================================================
# Evaluator
# =============================================================================

class AnswerEvaluator:
    """Deterministic answer quality evaluator — no LLM judge (v2)."""

    def evaluate(
        self,
        question_id: str,
        generated_answer: str,
        gold_answer: str,
        evidence_chunks: List,
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

        # --- Abstention detection (P0-2) ---
        result.is_insufficient_evidence = bool(_INSUFFICIENT_RE.search(gen))
        result.is_abstention = _is_abstention(gen)

        if result.is_abstention:
            result.answered = 0
            result.answer_accuracy = 0.0
            result.faithfulness_score = None
            result.numerical_consistency = None

        if not gen:
            result.parse_failed = True
            result.issues.append("Empty generated answer")
            result.answered = 0
            return result

        # ---- Accuracy (P0-10) ----
        gen_norm = _normalize(gen)
        gold_norm = _normalize(gold_answer)

        result.exact_match = gen_norm == gold_norm
        result.relaxed_match = _relaxed_match(gen_norm, gold_norm)

        if result.exact_match:
            result.answer_accuracy = 1.0
        elif result.relaxed_match:
            result.answer_accuracy = 0.5

        # Normalized exact match
        gen_simple = re.sub(r"\s+", " ", gen.lower().strip().rstrip("."))
        gold_simple = re.sub(r"\s+", " ", gold_answer.lower().strip().rstrip("."))
        result.normalized_exact_match = 1.0 if gen_simple == gold_simple else 0.0

        # Token F1
        result.token_f1 = _compute_token_f1(gen_norm, gold_norm)

        # Numeric metrics
        gen_numbers = _extract_numbers(gen)
        gold_numbers = _extract_numbers(gold_answer)
        result.numeric_exact_match = _compute_number_set_match(gen_numbers, gold_numbers)
        result.numeric_tolerance_match = _compute_numeric_tolerance_match(gen_numbers, gold_numbers)
        result.unit_match = _compute_unit_match(gen, gold_answer)

        # ---- Numerical consistency (P0-2: None for abstentions) ----
        result.numbers_in_generated = gen_numbers
        result.numbers_in_gold = gold_numbers

        if not result.is_abstention:
            if gold_numbers:
                matched = 0
                gold_nums_normalized = {_normalize_number(n) for n in gold_numbers}
                for n in gen_numbers:
                    if _normalize_number(n) in gold_nums_normalized:
                        matched += 1
                result.numbers_matched_count = matched
                result.numerical_consistency = matched / len(gold_numbers)
            else:
                result.numerical_consistency = 1.0

        # ---- Evidence retrieval metrics (P0-9) ----
        retrieved_ids = [c.chunk_id if hasattr(c, 'chunk_id') else str(c)
                         for c in evidence_chunks[:5]]
        if gold_evidence_ids:
            top5_set = set(retrieved_ids)
            gold_set = set(gold_evidence_ids)

            # Hit@5
            result.evidence_hit_at_5 = bool(top5_set & gold_set)

            # Recall@5
            if gold_set:
                result.evidence_recall_at_5 = len(top5_set & gold_set) / len(gold_set)

            # All-Gold-Covered@5
            result.all_gold_covered_at_5 = gold_set.issubset(top5_set)

            # MRR
            for rank, eid in enumerate(retrieved_ids, start=1):
                if eid in gold_set:
                    result.evidence_mrr = 1.0 / rank
                    break

        # ---- Faithfulness (P0-2: None for abstentions) ----
        if not result.is_abstention:
            evidence_text = " ".join(
                c.text if hasattr(c, 'text') else str(c)
                for c in evidence_chunks
            )
            ev_numbers = set(_normalize_number(n) for n in _extract_numbers(evidence_text))
            result.numbers_in_evidence = list(ev_numbers)
            ans_numbers_normalized = {_normalize_number(n) for n in gen_numbers}
            result.numbers_in_answer_not_in_evidence = len(ans_numbers_normalized - ev_numbers)

            if ans_numbers_normalized:
                result.faithfulness_score = 1.0 - (
                    result.numbers_in_answer_not_in_evidence / len(ans_numbers_normalized)
                )
            else:
                result.faithfulness_score = 1.0

            # P0-11: Check arithmetic verification
            result.requires_arithmetic_verification = _check_arithmetic(
                gen, evidence_text
            )

        # Collect issues
        if result.is_abstention:
            result.issues.append("Abstention (insufficient evidence)")
        elif not result.exact_match and not result.relaxed_match:
            result.issues.append("Answer does not match gold (exact or relaxed)")
        if result.numerical_consistency is not None and result.numerical_consistency < 1.0:
            result.issues.append(
                f"Numerical inconsistency: {result.numbers_matched_count}/"
                f"{len(result.numbers_in_gold)} numbers matched"
            )
        if result.numbers_in_answer_not_in_evidence > 0:
            result.issues.append(
                f"Faithfulness: {result.numbers_in_answer_not_in_evidence} numbers "
                f"in answer not found in evidence"
            )
        if result.requires_arithmetic_verification:
            result.issues.append("Requires arithmetic verification")
        if not result.evidence_hit_at_5 and gold_evidence_ids:
            result.issues.append("No gold evidence in top-5 retrieved")

        return result


# =============================================================================
# Helpers
# =============================================================================

def _is_abstention(answer: str) -> bool:
    """Check if answer is an abstention (P0-2)."""
    text = answer.strip().lower().rstrip(".!")
    return text == "insufficient evidence"


def _normalize(text: str) -> str:
    text = text.strip()
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text.lower().strip().rstrip("."))


def _relaxed_match(gen: str, gold: str) -> bool:
    if not gen or not gold:
        return False
    if gold in gen or gen in gold:
        return True
    gen_tokens = set(gen.split())
    gold_tokens = set(gold.split())
    if not gold_tokens:
        return False
    intersection = gen_tokens & gold_tokens
    return len(intersection) / len(gold_tokens) >= 0.8


def _extract_numbers(text: str) -> List[str]:
    return [m.group(0).strip() for m in _NUMBER_RE.finditer(text)]


def _normalize_number(num_str: str) -> str:
    s = num_str.lower().strip()
    s = s.replace("$", "")
    s = re.sub(r"(\d),(\d)", r"\1\2", s)
    s = s.replace("million", "m").replace("billion", "b")
    s = s.replace("thousand", "k").replace("trillion", "t")
    s = re.sub(r"\s+", "", s)
    return s


def _compute_token_f1(pred: str, gold: str) -> float:
    pred_tokens = set(pred.split())
    gold_tokens = set(gold.split())
    if not gold_tokens and not pred_tokens:
        return 1.0
    if not gold_tokens or not pred_tokens:
        return 0.0
    tp = len(pred_tokens & gold_tokens)
    prec = tp / len(pred_tokens)
    rec = tp / len(gold_tokens)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def _compute_number_set_match(gen_nums: List[str], ref_nums: List[str]) -> float:
    if not gen_nums:
        return 1.0
    ref_set = {_normalize_number(n) for n in ref_nums}
    matched = sum(1 for n in gen_nums if _normalize_number(n) in ref_set)
    return matched / len(gen_nums)


def _parse_number_value(num_str: str) -> Optional[float]:
    s = num_str.lower().strip().replace("$", "").replace(",", "").replace("%", "")
    multipliers = {"million": 1e6, "billion": 1e9, "thousand": 1e3, "trillion": 1e12}
    for word, mult in multipliers.items():
        if word in s:
            s = s.replace(word, "").strip()
            try:
                return float(s) * mult
            except ValueError:
                return None
    try:
        return float(s)
    except ValueError:
        return None


def _compute_numeric_tolerance_match(gen_nums: List[str], ref_nums: List[str]) -> float:
    if not gen_nums:
        return 1.0
    ref_vals = [v for n in ref_nums if (v := _parse_number_value(n)) is not None]
    if not ref_vals:
        return 0.0
    matched = 0
    for n in gen_nums:
        v = _parse_number_value(n)
        if v is None:
            continue
        for rv in ref_vals:
            if abs(v - rv) / max(abs(rv), 1) < 0.01:
                matched += 1
                break
    return matched / len(gen_nums)


def _compute_unit_match(gen: str, ref: str) -> float:
    gen_units = set(m.group(1).lower() for m in _UNIT_RE.finditer(gen))
    ref_units = set(m.group(1).lower() for m in _UNIT_RE.finditer(ref))
    if not gen_units:
        return 1.0
    return len(gen_units & ref_units) / len(gen_units)


def _check_arithmetic(answer: str, evidence_text: str) -> bool:
    """Check if answer contains numbers not directly in evidence (P0-11)."""
    gen_numbers = _extract_numbers(answer)
    if not gen_numbers:
        return False
    ev_numbers = {_normalize_number(n) for n in _extract_numbers(evidence_text)}
    for n in gen_numbers:
        n_norm = _normalize_number(n)
        if n_norm not in ev_numbers:
            try:
                n_val = _parse_number_value(n)
                found = False
                for ev_n in _extract_numbers(evidence_text):
                    ev_val = _parse_number_value(ev_n)
                    if ev_val is not None and n_val is not None:
                        if abs(n_val - ev_val) / max(abs(ev_val), 1) < 0.01:
                            found = True
                            break
                if not found:
                    return True
            except (ValueError, ZeroDivisionError):
                return True
    return False
