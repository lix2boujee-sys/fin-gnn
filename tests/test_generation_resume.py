"""Tests for resume safety and manifest validation (P0-4, P0-5).

Validates:
  1. Resume rejects changed config.
  2. Resume accepts matching config.
  3. Run manifest critical field comparison.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.generation_with_selected_evidence import (
    RunManifest,
    _safe_jsonl_append,
    _safe_read_jsonl,
)


class TestRunManifest:
    """P0-4: Run manifest validation for resume."""

    def test_identical_manifests_compatible(self):
        """Identical manifests should be resume-compatible."""
        m1 = RunManifest(
            corpus_cache_sha256="abc123",
            generator_model="qwen/qwen-2.5-7b-instruct",
            generator_provider="openrouter",
            prompt_sha256="def456",
            method_files={"R-GCN": "/path/to/rgcn.jsonl"},
        )
        m2 = RunManifest(
            corpus_cache_sha256="abc123",
            generator_model="qwen/qwen-2.5-7b-instruct",
            generator_provider="openrouter",
            prompt_sha256="def456",
            method_files={"R-GCN": "/path/to/rgcn.jsonl"},
        )
        mismatches = m1.check_resume_compatible(m2)
        assert len(mismatches) == 0

    def test_different_corpus_cache_incompatible(self):
        """Different corpus cache hash should be incompatible."""
        m1 = RunManifest(corpus_cache_sha256="abc123")
        m2 = RunManifest(corpus_cache_sha256="xyz789")
        mismatches = m1.check_resume_compatible(m2)
        assert any("corpus_cache_sha256" in m for m in mismatches)

    def test_different_model_incompatible(self):
        """Different generator model should be incompatible."""
        m1 = RunManifest(generator_model="qwen/qwen-2.5-7b-instruct")
        m2 = RunManifest(generator_model="meta-llama/Llama-3.1-8B-Instruct")
        mismatches = m1.check_resume_compatible(m2)
        assert any("generator_model" in m for m in mismatches)

    def test_different_method_files_incompatible(self):
        """Different method file paths should be incompatible."""
        m1 = RunManifest(method_files={"R-GCN": "/path/a.jsonl"})
        m2 = RunManifest(method_files={"R-GCN": "/path/b.jsonl"})
        mismatches = m1.check_resume_compatible(m2)
        assert any("method_files" in m for m in mismatches)

    def test_different_top_k_incompatible(self):
        """Different top_k should be incompatible."""
        m1 = RunManifest(top_k_evidence=5)
        m2 = RunManifest(top_k_evidence=10)
        mismatches = m1.check_resume_compatible(m2)
        assert any("top_k_evidence" in m for m in mismatches)

    def test_manifest_to_from_dict_roundtrip(self):
        """Manifest should survive to_dict/from_dict round-trip."""
        m1 = RunManifest(
            corpus_cache_sha256="abc123",
            generator_model="test-model",
            method_files={"Test": "/path/test.jsonl"},
            method_sources={"Test": {"filepath": "/path/test.jsonl", "format": "simple"}},
            generator_parameters={"temperature": 0.0, "max_new_tokens": 128},
            created_at="2026-01-01T00:00:00",
        )
        d = m1.to_dict()
        m2 = RunManifest.from_dict(d)
        assert m2.corpus_cache_sha256 == m1.corpus_cache_sha256
        assert m2.generator_model == m1.generator_model
        assert m2.method_files == m1.method_files
        assert m2.generator_parameters == m1.generator_parameters

    def test_critical_fields_match(self):
        """critical_fields should exclude non-critical metadata."""
        m = RunManifest(
            corpus_cache_sha256="abc",
            created_at="2026-01-01",
            paper_mode=True,
        )
        critical = m.critical_fields()
        assert "corpus_cache_sha256" in critical
        assert "created_at" not in critical
        assert "paper_mode" not in critical


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
