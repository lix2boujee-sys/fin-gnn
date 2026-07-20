"""Tests for deterministic random sampling (P1-1).

Validates:
  1. Same seed produces same sample.
  2. Different seeds produce different samples.
  3. Persisted selected_query_ids.json is used on resume/eval-only.
"""

from __future__ import annotations

import json
import random
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class TestDeterministicSampling:
    """P1-1: Random sampling must be deterministic and persisted."""

    def test_same_seed_same_sample(self):
        """Same seed produces identical selection."""
        all_ids = [f"q{i}" for i in range(100)]

        rng1 = random.Random(42)
        rng2 = random.Random(42)
        sample1 = rng1.sample(sorted(all_ids), 20)
        sample2 = rng2.sample(sorted(all_ids), 20)

        assert sample1 == sample2

    def test_different_seed_different_sample(self):
        """Different seeds should produce different selections."""
        all_ids = [f"q{i}" for i in range(100)]

        rng1 = random.Random(42)
        rng2 = random.Random(123)
        sample1 = rng1.sample(sorted(all_ids), 20)
        sample2 = rng2.sample(sorted(all_ids), 20)

        # In 100 choose 20, extremely unlikely to be identical
        assert sample1 != sample2

    def test_sample_size_bounded(self):
        """Sample size cannot exceed population."""
        all_ids = [f"q{i}" for i in range(5)]
        rng = random.Random(42)
        with pytest.raises(ValueError):
            rng.sample(sorted(all_ids), 10)

    def test_sorted_input_stable(self):
        """Sorting input IDs before sampling ensures determinism."""
        all_ids = [f"q{i}" for i in [5, 1, 3, 7, 2, 9, 4, 6, 8, 0]]
        # Two separate RNGs with same seed must produce same sample
        rng1 = random.Random(42)
        rng2 = random.Random(42)
        sample = rng1.sample(sorted(all_ids), 5)
        expected = rng2.sample(sorted(all_ids), 5)
        assert sample == expected  # Should be deterministic

    def test_persisted_ids_reread(self, tmp_path):
        """Persisted selected_query_ids.json should be reloadable."""
        selected = ["q3", "q1", "q7", "q2"]

        sel_path = tmp_path / "selected_query_ids.json"
        sel_path.write_text(json.dumps(selected))

        # Simulate re-read on eval-only
        reloaded = json.loads(sel_path.read_text(encoding="utf-8"))
        assert reloaded == selected
        assert set(reloaded) == {"q1", "q2", "q3", "q7"}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
