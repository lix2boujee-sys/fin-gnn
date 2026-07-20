"""Tests for token budget and used evidence IDs (P0-12).

Validates:
  1. used_evidence_ids reflects post-budget truncation.
  2. dropped_evidence_ids are recorded.
  3. Evidence Hit/Recall/MRR are based on used IDs, not requested.
  4. Duplicate chunks are removed before budget.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.generation_with_selected_evidence import (
    get_top_k_evidence_with_budget,
    GenerationEvaluator,
    GenEvalResult,
)
from feg_rag.generation.evidence_schema import (
    truncate_by_token_budget,
    deduplicate_evidence,
    RankedEvidence,
)


class TestTokenBudget:
    """P0-12: Token budget truncation."""

    def test_truncate_by_budget_drops_low_rank(self):
        """Low-rank evidence should be dropped when budget is tight."""
        evidence = [
            RankedEvidence("q1", "c1", 1, 0.9, "short text", None, "f.jsonl"),
            RankedEvidence("q1", "c2", 2, 0.8, "a" * 8000, None, "f.jsonl"),  # ~2000 tokens
            RankedEvidence("q1", "c3", 3, 0.7, "b" * 8000, None, "f.jsonl"),  # ~2000 tokens
        ]
        used, dropped, tokens = truncate_by_token_budget(
            evidence, max_input_tokens=500, max_tokens_per_evidence=None,
        )
        # c1 (~3 tokens) fits in 500 budget
        # c2 (~2000 tokens) exceeds remaining ~497 budget -> dropped
        # c3 (~2000 tokens) also exceeds remaining ~497 budget -> dropped
        assert len(used) == 1
        assert used[0].chunk_id == "c1"
        assert len(dropped) == 2
        assert dropped[0].chunk_id == "c2"
        assert dropped[1].chunk_id == "c3"

    def test_no_budget_uses_all(self):
        """Without budget, all evidence should be used."""
        evidence = [
            RankedEvidence("q1", "c1", 1, 0.9, "t1", None, "f.jsonl"),
            RankedEvidence("q1", "c2", 2, 0.8, "t2", None, "f.jsonl"),
        ]
        used, dropped, tokens = truncate_by_token_budget(
            evidence, max_input_tokens=None, max_tokens_per_evidence=None,
        )
        assert len(used) == 2
        assert len(dropped) == 0

    def test_truncate_with_per_evidence_limit(self):
        """Per-chunk token limit should apply."""
        evidence = [
            RankedEvidence("q1", "c1", 1, 0.9, "short", None, "f.jsonl"),
            RankedEvidence("q1", "c2", 2, 0.8, "a" * 4000, None, "f.jsonl"),
        ]
        used, dropped, tokens = truncate_by_token_budget(
            evidence, max_input_tokens=1000, max_tokens_per_evidence=100,
        )
        # c1 (~2 tokens) fits, c2 capped at 100 tokens, both fit in 1000 budget
        assert len(used) == 2


class TestDedupBeforeBudget:
    """Dedup should happen before token budget truncation."""

    def test_dedup_preserves_best_rank(self):
        """Dedup keeps the first (lowest rank) occurrence of each chunk_id."""
        evidence = [
            RankedEvidence("q1", "c1", 1, 0.95, "best text", None, "f.jsonl"),
            RankedEvidence("q1", "c2", 2, 0.80, "text2", None, "f.jsonl"),
            RankedEvidence("q1", "c1", 3, 0.60, "worse text", None, "f.jsonl"),
        ]
        deduped = deduplicate_evidence(evidence)
        assert len(deduped) == 2
        assert deduped[0].chunk_id == "c1"
        assert deduped[0].text == "best text"  # kept rank-1 text
        assert deduped[1].chunk_id == "c2"


class TestUsedEvidenceIDs:
    """Evidence metrics must use used_evidence_ids, not requested (P0-12)."""

    def test_eval_uses_used_ids(self):
        """Evaluation should use the actual evidence IDs seen by the model."""
        evaluator = GenerationEvaluator()
        # The model only saw c3 (rank 1), but c1 was gold
        result = evaluator.evaluate(
            query_id="q1", method="Test", query="Q?", reference_answer="A",
            generated_answer="Answer.",
            evidence_ids=["c3", "c4"],  # used_evidence_ids (after budget truncation)
            evidence_texts=["text3", "text4"],
            gold_evidence_ids=["c1", "c3"],  # gold includes c3
            gold_aligned=True,
        )
        # Only c3 is in the used set and is gold
        assert result.evidence_hit_at_5 == 1  # c3 hit
        assert result.evidence_recall_at_5 == 0.5  # 1 of 2 gold found


class TestGetTopKEvidenceWithBudget:
    """Integration test for get_top_k_evidence_with_budget."""

    def test_extracts_and_dedupes(self):
        """Should extract, dedupe, and apply budget."""
        chunk_meta = {
            "c1": {"text": "Revenue was 100M in 2023.", "doc_id": "d1"},
            "c2": {"text": "Profit was 50M.", "doc_id": "d2"},
            "c3": {"text": "Cash flow positive.", "doc_id": "d3"},
        }
        cross_ref = {}

        rec = {
            "question_id": "q1",
            "retrieved_chunk_ids": ["c1", "c2", "c2", "c3"],  # c2 duplicated
        }

        used_ids, used_texts, dropped, tokens, truncated = get_top_k_evidence_with_budget(
            rec, "simple", chunk_meta, cross_ref, top_k=3,
            max_input_tokens=None, max_tokens_per_evidence=None,
        )

        # c2 deduped, so we get c1, c2, c3
        assert len(used_ids) == 3
        assert used_ids == ["c1", "c2", "c3"]
        assert used_texts[0] == "Revenue was 100M in 2023."
        assert not truncated

    def test_missing_text_placeholder(self):
        """Chunks without text should get placeholder."""
        chunk_meta = {}
        cross_ref = {}
        rec = {
            "question_id": "q1",
            "retrieved_chunk_ids": ["c_missing"],
        }
        used_ids, used_texts, dropped, tokens, truncated = get_top_k_evidence_with_budget(
            rec, "simple", chunk_meta, cross_ref, top_k=1,
        )
        assert used_texts[0].startswith("[Evidence text not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
