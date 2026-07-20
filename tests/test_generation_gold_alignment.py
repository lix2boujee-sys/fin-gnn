"""Tests for gold alignment coverage (P0-1).

Validates:
  1. Unaligned gold queries do NOT enter Hit/Recall denominator.
  2. Gold alignment coverage is computed correctly.
  3. Paper mode raises on low coverage.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Insert source
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.generation_with_selected_evidence import (
    compute_gold_alignment,
    check_gold_alignment_threshold,
    GenEvalResult,
    GenerationEvaluator,
)


class TestGoldAlignment:
    """P0-1: Gold alignment coverage tracking."""

    def test_compute_gold_alignment_full(self):
        """All queries aligned should give 100% coverage."""
        query_ids = ["q1", "q2", "q3"]
        gold_map = {"q1": ["c1"], "q2": ["c2"], "q3": ["c3"]}
        result = compute_gold_alignment(query_ids, gold_map)
        assert result["gold_alignment_coverage"] == 1.0
        assert result["aligned_count"] == 3
        assert result["unaligned_count"] == 0

    def test_compute_gold_alignment_partial(self):
        """Partial alignment should give correct ratio."""
        query_ids = ["q1", "q2", "q3", "q4"]
        gold_map = {"q1": ["c1"], "q3": ["c3"]}
        result = compute_gold_alignment(query_ids, gold_map)
        assert result["gold_alignment_coverage"] == 0.5
        assert result["aligned_count"] == 2
        assert result["unaligned_count"] == 2
        assert "q2" in result["gold_unaligned_queries"]

    def test_compute_gold_alignment_empty(self):
        """No gold map should give 0% coverage."""
        query_ids = ["q1", "q2"]
        gold_map = {}
        result = compute_gold_alignment(query_ids, gold_map)
        assert result["gold_alignment_coverage"] == 0.0
        assert result["aligned_count"] == 0

    def test_check_threshold_warn(self, capsys):
        """Below threshold without paper_mode should warn, not error."""
        alignment = {
            "gold_alignment_coverage": 0.5,
            "aligned_count": 5,
            "total_queries": 10,
            "unaligned_count": 5,
        }
        check_gold_alignment_threshold(alignment, 0.95, paper_mode=False)
        captured = capsys.readouterr()
        assert "WARN" in captured.out or "WARN" in captured.err

    def test_check_threshold_error_paper_mode(self):
        """Below threshold with paper_mode should raise RuntimeError."""
        alignment = {
            "gold_alignment_coverage": 0.5,
            "aligned_count": 5,
            "total_queries": 10,
            "unaligned_count": 5,
        }
        with pytest.raises(RuntimeError, match="FATAL"):
            check_gold_alignment_threshold(alignment, 0.95, paper_mode=True)

    def test_check_threshold_ok(self, capsys):
        """Above threshold should pass silently."""
        alignment = {
            "gold_alignment_coverage": 0.98,
            "aligned_count": 98,
            "total_queries": 100,
            "unaligned_count": 2,
        }
        check_gold_alignment_threshold(alignment, 0.95, paper_mode=True)
        # No error raised

    def test_evidence_hit_excludes_unaligned(self):
        """Evidence Hit/Recall should only count gold-aligned queries (P0-1)."""
        evaluator = GenerationEvaluator()

        # Aligned query with gold evidence
        aligned_result = evaluator.evaluate(
            query_id="q1",
            method="Test",
            query="What is revenue?",
            reference_answer="100M",
            generated_answer="Revenue was 100 million.",
            evidence_ids=["c1", "c2", "c3", "c4", "c5"],
            evidence_texts=["Revenue was 100 million."] * 5,
            gold_evidence_ids=["c1", "c3"],
            gold_aligned=True,
        )
        assert aligned_result.evidence_hit_at_5 == 1
        assert aligned_result.evidence_recall_at_5 == 1.0  # both c1 and c3 in top-5
        assert aligned_result.all_gold_covered_at_5 == 1

        # Unaligned query: should NOT contribute to evidence metrics
        unaligned_result = evaluator.evaluate(
            query_id="q2",
            method="Test",
            query="What is profit?",
            reference_answer="50M",
            generated_answer="Profit was 50 million.",
            evidence_ids=["c10", "c11", "c12", "c13", "c14"],
            evidence_texts=["Profit was 50 million."] * 5,
            gold_evidence_ids=["c1", "c3"],
            gold_aligned=False,  # key: not aligned
        )
        assert unaligned_result.evidence_hit_at_5 == 0
        assert unaligned_result.evidence_recall_at_5 == 0.0

        # Aggregate: only aligned queries count
        agg = GenerationEvaluator.aggregate(
            method="Test",
            per_answer=[aligned_result, unaligned_result],
            gold_aligned_count=1,
        )
        assert agg.gold_alignment_coverage == 0.5  # 1/2 aligned
        assert agg.evidence_hit_at_5 == 1.0  # only q1 counts
        assert agg.evidence_hit_at_5_n == 1  # only 1 aligned query


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
