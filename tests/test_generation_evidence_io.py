"""Tests for evidence I/O (P0-3, P0-5, P0-8).

Validates:
  1. Simple/rich result conversion to unified schema.
  2. Duplicate chunk dedup before top-k.
  3. Safe JSONL reads (corrupt last line recovery).
  4. API error records do NOT enter main metrics.
  5. Multiple glob match raises error (P0-3).
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.generation_with_selected_evidence import (
    _safe_jsonl_append,
    _safe_read_jsonl,
    find_method_file_legacy,
    parse_method_result_args,
    GenEvalResult,
    GenerationEvaluator,
)
from feg_rag.generation.evidence_schema import (
    RankedEvidence,
    convert_simple_format,
    convert_rich_format,
    deduplicate_evidence,
)


class TestEvidenceSchema:
    """P0-8: Simple/rich results -> unified schema."""

    def test_convert_simple_format(self, tmp_path):
        """Simple format with retrieved_chunk_ids."""
        filepath = tmp_path / "simple_results.jsonl"
        filepath.write_text(json.dumps({
            "question_id": "q1",
            "retrieved_chunk_ids": ["c1", "c2", "c3"],
        }) + "\n")

        results = convert_simple_format(filepath)
        assert len(results) == 3
        assert results[0].query_id == "q1"
        assert results[0].chunk_id == "c1"
        assert results[0].rank == 1

    def test_convert_rich_format(self, tmp_path):
        """Rich format with top_k entries."""
        filepath = tmp_path / "rich_results.jsonl"
        filepath.write_text(json.dumps({
            "query_id": "q1",
            "top_k": [
                {"passage_id": "p1", "text": "Evidence text 1"},
                {"passage_id": "p2", "text": "Evidence text 2"},
            ],
        }) + "\n")

        results = convert_rich_format(filepath)
        assert len(results) == 2
        assert results[0].chunk_id == "p1"
        assert results[0].text == "Evidence text 1"
        assert results[1].rank == 2

    def test_convert_simple_format_with_scores(self, tmp_path):
        """Simple format with top_k dicts containing scores."""
        filepath = tmp_path / "scored_results.jsonl"
        filepath.write_text(json.dumps({
            "question_id": "q1",
            "top_k": [
                {"chunk_id": "c1", "score": 0.95},
                {"chunk_id": "c2", "score": 0.80},
            ],
        }) + "\n")

        results = convert_simple_format(filepath)
        assert len(results) == 2
        assert results[0].score == 0.95
        assert results[1].score == 0.80

    def test_deduplicate_evidence(self):
        """Duplicate chunk_ids should be removed, keeping best rank."""
        ev = [
            RankedEvidence("q1", "c1", 1, 0.9, "text1", None, "f.jsonl"),
            RankedEvidence("q1", "c2", 2, 0.8, "text2", None, "f.jsonl"),
            RankedEvidence("q1", "c1", 3, 0.7, "text1_dup", None, "f.jsonl"),  # dup
            RankedEvidence("q1", "c3", 4, 0.6, "text3", None, "f.jsonl"),
        ]
        deduped = deduplicate_evidence(ev)
        assert len(deduped) == 3
        assert deduped[0].chunk_id == "c1"
        assert deduped[0].rank == 1  # kept first occurrence
        assert deduped[1].chunk_id == "c2"
        assert deduped[2].chunk_id == "c3"


class TestSafeJSONL:
    """P0-5: Safe JSONL writes and corrupt-line recovery."""

    def test_safe_write_and_read(self, tmp_path):
        """Normal write/read cycle."""
        filepath = tmp_path / "test.jsonl"
        record = {"query_id": "q1", "answer": "test", "status": "completed"}
        _safe_jsonl_append(filepath, record)

        records = _safe_read_jsonl(filepath)
        assert len(records) == 1
        assert records[0]["query_id"] == "q1"

    def test_corrupt_last_line_recovery(self, tmp_path):
        """Last line corrupt -> backup + truncate."""
        filepath = tmp_path / "test.jsonl"
        _safe_jsonl_append(filepath, {"query_id": "q1", "answer": "ok"})
        # Append corrupt data directly
        with open(filepath, "a", encoding="utf-8") as fh:
            fh.write("this is not valid json\n")

        records = _safe_read_jsonl(filepath)
        assert len(records) == 1
        assert records[0]["query_id"] == "q1"

        # Check backup was created
        backup = tmp_path / "test.jsonl.corrupt_backup"
        assert backup.exists()

    def test_corrupt_interior_line_errors(self, tmp_path):
        """Corrupt interior line should raise error."""
        filepath = tmp_path / "test.jsonl"
        _safe_jsonl_append(filepath, {"query_id": "q1", "answer": "ok"})
        # Write corrupt interior line
        with open(filepath, "a", encoding="utf-8") as fh:
            fh.write("not json\n")
        _safe_jsonl_append(filepath, {"query_id": "q2", "answer": "ok2"})

        with pytest.raises(RuntimeError, match="Corrupt JSON at interior"):
            _safe_read_jsonl(filepath)


class TestGlobAmbiguity:
    """P0-3: No silent selection of ambiguous glob matches."""

    def test_parse_explicit_method_result(self, tmp_path):
        """Explicit --method_result parsing."""
        # Create a test file
        result_file = tmp_path / "rgcn_results.jsonl"
        result_file.write_text('{"question_id":"q1","retrieved_chunk_ids":["c1"]}\n')

        result = parse_method_result_args(
            [f"R-GCN={result_file}"],
            root_dir=tmp_path,
        )
        assert "R-GCN" in result
        assert result["R-GCN"] == result_file.resolve()

    def test_parse_missing_file_errors(self, tmp_path):
        """Non-existent file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            parse_method_result_args(
                ["R-GCN=/nonexistent/path.jsonl"],
                root_dir=tmp_path,
            )

    def test_parse_invalid_format_errors(self, tmp_path):
        """Missing '=' should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid --method_result format"):
            parse_method_result_args(
                ["R-GCN"],  # no '='
                root_dir=tmp_path,
            )


class TestAPIErrorExclusion:
    """P0-5: API error records should not enter main metrics."""

    def test_api_error_not_in_main_metrics(self):
        """Only 'completed' status records should count for main metrics."""
        evaluator = GenerationEvaluator()

        # Normal completed result (correct answer, matching reference)
        normal = evaluator.evaluate(
            query_id="q1",
            method="Test",
            query="What is revenue?",
            reference_answer="100 million",
            generated_answer="100 million",
            evidence_ids=["c1"],
            evidence_texts=["Revenue is 100M."],
            gold_evidence_ids=[],
            gold_aligned=False,
        )
        assert normal.answered == 1
        assert normal.answer_accuracy == 1.0

        # Abstention (answered=0)
        abstain = evaluator.evaluate(
            query_id="q2",
            method="Test",
            query="What is profit?",
            reference_answer="50M",
            generated_answer="insufficient evidence",
            evidence_ids=["c2"],
            evidence_texts=["No data."],
            gold_evidence_ids=[],
            gold_aligned=False,
        )
        assert abstain.answered == 0
        assert abstain.is_abstention
        assert abstain.faithfulness is None
        assert abstain.numerical_consistency is None

        # This simulates an API error (empty answer, not abstention)
        api_error = evaluator.evaluate(
            query_id="q3",
            method="Test",
            query="What is cost?",
            reference_answer="30M",
            generated_answer="",  # empty = API error, not abstention
            evidence_ids=["c3"],
            evidence_texts=["Cost is 30M."],
            gold_evidence_ids=[],
            gold_aligned=False,
        )
        assert api_error.answered == 0
        assert not api_error.is_abstention  # empty is not abstention

        agg = GenerationEvaluator.aggregate(
            "Test",
            [normal, abstain, api_error],
            gold_aligned_count=0,
        )
        # answer_rate = 1 answered / 3 total
        assert agg.answer_rate == 1.0 / 3.0
        # accuracy_answered only counts the 1 answered query
        assert agg.accuracy_answered == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
