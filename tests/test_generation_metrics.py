"""Tests for generation metrics (P0-2, P0-9, P0-10, P0-11).

Validates:
  1. Abstention does not get Faithfulness/NumCon credit (P0-2).
  2. Evidence Recall@5, All-Gold-Covered@5, Evidence MRR (P0-9).
  3. Token F1 and numeric metrics (P0-10).
  4. Arithmetic verification flag (P0-11).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.generation_with_selected_evidence import (
    GenerationEvaluator,
    GenEvalResult,
    _compute_token_f1,
    _normalize,
    _extract_numbers,
    _normalize_number,
    _parse_number_value,
    AggregateGenMetrics,
)


class TestAbstentionHandling:
    """P0-2: Abstention must NOT get Faithfulness/NumCon credit."""

    def test_abstention_faithfulness_none(self):
        """Abstention should have faithfulness=None."""
        evaluator = GenerationEvaluator()
        result = evaluator.evaluate(
            query_id="q1",
            method="Test",
            query="What is revenue?",
            reference_answer="100M",
            generated_answer="insufficient evidence",
            evidence_ids=["c1", "c2"],
            evidence_texts=["Revenue data not available.", "No financial info."],
            gold_evidence_ids=["c1"],
            gold_aligned=True,
        )
        assert result.is_abstention
        assert result.answered == 0
        assert result.faithfulness is None
        assert result.numerical_consistency is None
        assert result.answer_accuracy == 0.0

    def test_abstention_not_counted_as_answered(self):
        """Abstention should have answered=0."""
        evaluator = GenerationEvaluator()
        result = evaluator.evaluate(
            query_id="q1",
            method="Test",
            query="What is revenue?",
            reference_answer="100M",
            generated_answer="insufficient evidence",
            evidence_ids=["c1"],
            evidence_texts=["No data."],
            gold_evidence_ids=[],
            gold_aligned=False,
        )
        assert result.answered == 0

    def test_aggregate_separates_answered_metrics(self):
        """Answered-only metrics exclude abstentions."""
        evaluator = GenerationEvaluator()

        answered = evaluator.evaluate(
            query_id="q1", method="Test", query="R?", reference_answer="100 million",
            generated_answer="100 million",
            evidence_ids=["c1"], evidence_texts=["Revenue 100M"],
            gold_evidence_ids=[], gold_aligned=False,
        )
        abstain = evaluator.evaluate(
            query_id="q2", method="Test", query="P?", reference_answer="50M",
            generated_answer="insufficient evidence",
            evidence_ids=["c2"], evidence_texts=["No data"],
            gold_evidence_ids=[], gold_aligned=False,
        )

        agg = GenerationEvaluator.aggregate(
            "Test", [answered, abstain], gold_aligned_count=0,
        )
        # answer_rate = 1/2
        assert agg.answer_rate == 0.5
        assert agg.insufficient_evidence_rate == 0.5
        # accuracy_all: abstention counts as 0
        assert agg.accuracy_all == 0.5  # (1.0 + 0.0) / 2
        # accuracy_answered: only the answered query
        assert agg.accuracy_answered == 1.0

    def test_normal_answer_has_metrics(self):
        """Normal (non-abstention) answer should get normal metrics."""
        evaluator = GenerationEvaluator()
        result = evaluator.evaluate(
            query_id="q1",
            method="Test",
            query="What is revenue?",
            reference_answer="100 million",
            generated_answer="Revenue is 100 million.",
            evidence_ids=["c1", "c2"],
            evidence_texts=["Revenue is 100 million.", "Additional data."],
            gold_evidence_ids=["c1"],
            gold_aligned=True,
        )
        assert not result.is_abstention
        assert result.answered == 1
        assert result.faithfulness is not None
        assert result.numerical_consistency is not None


class TestEvidenceRetrievalMetrics:
    """P0-9: Evidence Recall@5, All-Gold-Covered@5, Evidence MRR."""

    def test_evidence_hit_at_5(self):
        evaluator = GenerationEvaluator()
        result = evaluator.evaluate(
            query_id="q1", method="Test", query="Q?", reference_answer="A",
            generated_answer="Answer.",
            evidence_ids=["c1", "c2", "c3", "c4", "c5"],
            evidence_texts=["t"] * 5,
            gold_evidence_ids=["c1", "c6"],  # c1 hit, c6 miss
            gold_aligned=True,
        )
        assert result.evidence_hit_at_5 == 1

    def test_evidence_recall_at_5(self):
        evaluator = GenerationEvaluator()
        result = evaluator.evaluate(
            query_id="q1", method="Test", query="Q?", reference_answer="A",
            generated_answer="Answer.",
            evidence_ids=["c1", "c2", "c3", "c4", "c5"],
            evidence_texts=["t"] * 5,
            gold_evidence_ids=["c1", "c3", "c7"],  # 2 of 3 in top-5
            gold_aligned=True,
        )
        assert result.evidence_recall_at_5 == pytest.approx(2.0 / 3.0)

    def test_all_gold_covered_at_5(self):
        evaluator = GenerationEvaluator()
        # Full coverage
        result = evaluator.evaluate(
            query_id="q1", method="Test", query="Q?", reference_answer="A",
            generated_answer="Answer.",
            evidence_ids=["c1", "c2", "c3"],
            evidence_texts=["t"] * 3,
            gold_evidence_ids=["c1", "c2"],
            gold_aligned=True,
        )
        assert result.all_gold_covered_at_5 == 1

        # Partial coverage
        result2 = evaluator.evaluate(
            query_id="q2", method="Test", query="Q2?", reference_answer="A2",
            generated_answer="Answer2.",
            evidence_ids=["c1", "c3"],
            evidence_texts=["t"] * 2,
            gold_evidence_ids=["c1", "c2", "c3"],
            gold_aligned=True,
        )
        assert result2.all_gold_covered_at_5 == 0  # c2 not in top

    def test_evidence_mrr(self):
        evaluator = GenerationEvaluator()
        # First gold at rank 2 -> MRR = 1/2
        result = evaluator.evaluate(
            query_id="q1", method="Test", query="Q?", reference_answer="A",
            generated_answer="Answer.",
            evidence_ids=["c9", "c1", "c3", "c4", "c5"],
            evidence_texts=["t"] * 5,
            gold_evidence_ids=["c1"],  # first occurrence at rank 2
            gold_aligned=True,
        )
        assert result.evidence_mrr == 0.5

        # No gold in top-5
        result2 = evaluator.evaluate(
            query_id="q2", method="Test", query="Q2?", reference_answer="A2",
            generated_answer="Answer2.",
            evidence_ids=["c9", "c10"],
            evidence_texts=["t"] * 2,
            gold_evidence_ids=["c1"],
            gold_aligned=True,
        )
        assert result2.evidence_mrr == 0.0

    def test_unaligned_query_no_evidence_metrics(self):
        """Unaligned queries should not contribute evidence metrics."""
        evaluator = GenerationEvaluator()
        result = evaluator.evaluate(
            query_id="q1", method="Test", query="Q?", reference_answer="A",
            generated_answer="Answer.",
            evidence_ids=["c1"],
            evidence_texts=["t"],
            gold_evidence_ids=["c1"],
            gold_aligned=False,  # NOT aligned
        )
        assert result.evidence_hit_at_5 == 0
        assert result.evidence_recall_at_5 == 0.0
        assert result.evidence_mrr == 0.0


class TestTokenF1:
    """P0-10: Token F1 and numeric metrics."""

    def test_token_f1_perfect(self):
        assert _compute_token_f1("revenue is 100 million", "revenue is 100 million") == 1.0

    def test_token_f1_partial(self):
        f1 = _compute_token_f1("revenue was about 100 million", "revenue is 100 million")
        assert 0.0 < f1 < 1.0

    def test_token_f1_no_overlap(self):
        assert _compute_token_f1("profit increased", "revenue declined") == 0.0

    def test_token_f1_empty(self):
        assert _compute_token_f1("", "") == 1.0
        assert _compute_token_f1("something", "") == 0.0


class TestArithmeticVerification:
    """P0-11: Arithmetic verification flag."""

    def test_arithmetic_flag_when_calculated(self):
        """Numbers derived from calculation should be flagged, not unsupported."""
        evaluator = GenerationEvaluator()
        result = evaluator.evaluate(
            query_id="q1", method="Test", query="What is the growth rate?",
            reference_answer="20%",
            generated_answer="The growth rate is 20%.",
            evidence_ids=["c1"],
            # Evidence has 120M and 100M but not 20%
            evidence_texts=["Revenue was 120M in 2023 and 100M in 2022."],
            gold_evidence_ids=[],
            gold_aligned=False,
        )
        assert result.requires_arithmetic_verification
        # Should still be marked as potentially unsupported (conservative)
        # But the flag is the key output

    def test_direct_number_not_flagged(self):
        """Direct numbers in evidence should NOT be flagged."""
        evaluator = GenerationEvaluator()
        result = evaluator.evaluate(
            query_id="q1", method="Test", query="What is revenue?",
            reference_answer="100 million",
            generated_answer="Revenue is 100 million.",
            evidence_ids=["c1"],
            evidence_texts=["Revenue was 100 million in 2023."],
            gold_evidence_ids=[],
            gold_aligned=False,
        )
        assert not result.requires_arithmetic_verification


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
