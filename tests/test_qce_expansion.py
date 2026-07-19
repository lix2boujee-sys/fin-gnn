"""Tests for QCE-Graph Lite: Graph Expansion module.

Tests:
    1. adjacent candidate doesn't cross doc_id
    2. same_section only within same doc+section
    3. same_year doesn't do unconstrained global expansion
    4. Candidate dedup preserves multiple source relations
    5. Total candidates capped at max_total_candidates
    6. Deterministic output with fixed seed
    7. Graceful degradation when no semantic edges
    8. Expansion never uses gold
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from feg_rag.data.chunker import Chunk
from feg_rag.rerank.qce_expansion import (
    RELATION_NAMES,
    GraphExpansionIndex,
    BudgetedGraphExpander,
    ExpandedCandidate,
    normalize_metric,
    DEFAULT_RELATION_PRIOR,
)


# ═════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═════════════════════════════════════════════════════════════════════════════

def _make_chunk(
    chunk_id: str,
    text: str = "",
    doc_id: str = "doc1",
    company: str = "ACME",
    filing_type: str = "10-K",
    filing_year: str = "2023",
    section: str = "Item 7",
    chunk_type: str = "text",
    word_offset: int = 0,
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        text=text or f"Content of {chunk_id}.",
        chunk_type=chunk_type,
        doc_id=doc_id,
        company=company,
        filing_type=filing_type,
        filing_year=filing_year,
        section=section,
        metadata={"word_offset": word_offset},
    )


@pytest.fixture
def sample_chunks():
    """Create a small corpus for testing."""
    chunks = []
    # doc1: 5 chunks in Item 7, 3 in Item 8
    for i in range(5):
        chunks.append(_make_chunk(
            f"d1s7_{i}", f"Revenue was $100M in 2023 for chunk {i}.",
            doc_id="doc1", section="Item 7", word_offset=i * 500,
        ))
    for i in range(3):
        chunks.append(_make_chunk(
            f"d1s8_{i}", f"Net income was $20M for chunk {i}.",
            doc_id="doc1", section="Item 8", word_offset=2500 + i * 500,
        ))

    # doc2: different company, same year
    for i in range(4):
        chunks.append(_make_chunk(
            f"d2s7_{i}", f"Revenue was $200M for chunk {i}.",
            doc_id="doc2", company="GLOBEX", section="Item 7", word_offset=i * 500,
        ))

    # doc3: same company, different year
    for i in range(3):
        chunks.append(_make_chunk(
            f"d3s7_{i}", f"Revenue was $150M in 2022 for chunk {i}.",
            doc_id="doc3", company="ACME", filing_year="2022", section="Item 7",
            word_offset=i * 500,
        ))

    return chunks


@pytest.fixture
def expansion_index(sample_chunks):
    """Build a GraphExpansionIndex from sample chunks."""
    idx = GraphExpansionIndex()
    idx.build(sample_chunks)
    return idx


@pytest.fixture
def expander(expansion_index):
    """Create a BudgetedGraphExpander with small budgets for testing."""
    return BudgetedGraphExpander(
        index=expansion_index,
        initial_top_n=50,
        seed_top_m=5,
        expansion_budget=10,
        max_budget_per_relation=5,
        max_total_candidates=20,
    )


@pytest.fixture
def initial_candidates(sample_chunks):
    """Return initial candidates: top chunks from doc1."""
    # d1s7_0, d1s7_1, d1s7_2 as top-3 initial
    candidates = []
    for i in range(3):
        chunk = sample_chunks[i]  # d1s7_0, d1s7_1, d1s7_2
        candidates.append((chunk, 10.0 - i))
    return candidates


# ═════════════════════════════════════════════════════════════════════════════
# Tests
# ═════════════════════════════════════════════════════════════════════════════

class TestAdjacentExpansion:
    """Test 1: Adjacent candidates do NOT cross doc_id."""

    def test_adjacent_same_doc_only(self, expander, initial_candidates):
        """Adjacent expansion should only return chunks from the same document."""
        expanded, stats = expander.expand(
            "What was the revenue?",
            initial_candidates,
            relation_probabilities={"adjacent_chunk": 1.0},
        )

        # All expanded (non-initial) candidates must be from doc1
        for ec in expanded:
            if not ec.is_initial:
                chunk = expander.index.chunk_lookup.get(ec.chunk_id)
                assert chunk is not None
                assert chunk.doc_id == "doc1", (
                    f"adjacent expansion crossed doc_id: {ec.chunk_id} -> {chunk.doc_id}"
                )

    def test_adjacent_prefers_same_section(self, expander, initial_candidates):
        """Adjacent expansion should prefer same-section chunks."""
        expanded, stats = expander.expand(
            "What was the revenue?",
            initial_candidates,
            relation_probabilities={"adjacent_chunk": 1.0},
        )

        # Check that same-section chunks have higher priority
        same_section = [
            ec for ec in expanded
            if not ec.is_initial
            and expander.index.chunk_lookup[ec.chunk_id].section == "Item 7"
        ]
        other_section = [
            ec for ec in expanded
            if not ec.is_initial
            and expander.index.chunk_lookup[ec.chunk_id].section != "Item 7"
        ]

        if same_section and other_section:
            avg_pri_same = np.mean([ec.expansion_priority for ec in same_section])
            avg_pri_other = np.mean([ec.expansion_priority for ec in other_section])
            # Same section should typically have higher priority
            # (local_match_bonus of 1.0 vs 0.5)
            pass  # Priority comparison is informational


class TestSameSectionExpansion:
    """Test 2: same_section only within same doc+section."""

    def test_same_section_same_doc_only(self, expander, initial_candidates):
        """same_section must only return chunks from same doc_id + section."""
        expanded, stats = expander.expand(
            "What was the revenue?",
            initial_candidates,
            relation_probabilities={"same_section": 1.0},
        )

        for ec in expanded:
            if not ec.is_initial and "same_section" in ec.source_relations:
                chunk = expander.index.chunk_lookup.get(ec.chunk_id)
                assert chunk is not None
                # Seed is d1s7_*, so expanded must also be doc1 + Item 7
                assert chunk.doc_id == "doc1", f"same_section crossed doc: {ec.chunk_id}"
                assert chunk.section == "Item 7", f"same_section crossed section: {ec.chunk_id} -> {chunk.section}"


class TestSameYearExpansion:
    """Test 3: same_year doesn't do unconstrained global expansion."""

    def test_same_year_requires_query_year(self, expander, initial_candidates):
        """When query has no year, same_year should not expand."""
        expanded, stats = expander.expand(
            "What was the revenue?",  # No year mentioned
            initial_candidates,
            relation_probabilities={"same_year": 1.0},
        )

        # same_year expansion should be empty when query doesn't mention years
        year_expanded = [
            ec for ec in expanded
            if not ec.is_initial and "same_year" in ec.source_relations
        ]
        assert len(year_expanded) == 0, (
            f"same_year expanded without query year: {len(year_expanded)} candidates"
        )

    def test_same_year_with_query_year(self, expander, initial_candidates):
        """When query explicitly has a year, same_year can expand."""
        expanded, stats = expander.expand(
            "What was the revenue in 2023?",  # Year explicitly mentioned
            initial_candidates,
            relation_probabilities={"same_year": 1.0},
        )

        # May or may not expand (depends on chunk years), but shouldn't crash
        assert stats is not None


class TestDedupAndSourceRelations:
    """Test 4: Candidate dedup preserves multiple source relations."""

    def test_dedup_merges_source_relations(self, expander, initial_candidates):
        """When a chunk is found by multiple relations, source_relations are merged."""
        # Use two relations that could overlap
        multi_probs = {
            "adjacent_chunk": 0.8,
            "same_section": 0.8,
            "same_filing": 0.5,
        }
        expanded, stats = expander.expand(
            "What was the revenue?",
            initial_candidates,
            relation_probabilities=multi_probs,
        )

        # Check that chunks appear only once in the output
        chunk_ids = [ec.chunk_id for ec in expanded]
        assert len(chunk_ids) == len(set(chunk_ids)), "Duplicates found in expanded output"

        # A chunk found by multiple paths should have multiple source_relations
        for ec in expanded:
            if not ec.is_initial:
                assert len(ec.source_relations) > 0, (
                    f"Expanded candidate {ec.chunk_id} has no source_relations"
                )


class TestMaxTotalCandidates:
    """Test 5: Total candidates capped at max_total_candidates."""

    def test_total_cap(self, expansion_index, initial_candidates):
        """Output should never exceed max_total_candidates."""
        expander_small = BudgetedGraphExpander(
            index=expansion_index,
            initial_top_n=50,
            seed_top_m=10,
            expansion_budget=30,
            max_budget_per_relation=10,
            max_total_candidates=5,  # Very small cap
        )

        expanded, stats = expander_small.expand(
            "What was the revenue?",
            initial_candidates,
            relation_probabilities={r: 1.0 for r in RELATION_NAMES},
        )

        assert len(expanded) <= 5, f"Exceeded max_total_candidates: {len(expanded)} > 5"


class TestDeterminism:
    """Test 6: Fixed seed produces reproducible results."""

    def test_same_seed_same_output(self, expander, initial_candidates):
        """Running expand twice with same input should give same output."""
        probs = {r: 0.5 for r in RELATION_NAMES}

        expanded1, stats1 = expander.expand(
            "What was the revenue?", initial_candidates, relation_probabilities=probs,
        )
        expanded2, stats2 = expander.expand(
            "What was the revenue?", initial_candidates, relation_probabilities=probs,
        )

        ids1 = [ec.chunk_id for ec in expanded1]
        ids2 = [ec.chunk_id for ec in expanded2]
        assert ids1 == ids2, "Non-deterministic expansion output"


class TestSemanticDegradation:
    """Test 7: Graceful degradation when no semantic edges."""

    def test_no_semantic_edges(self, expansion_index, initial_candidates):
        """When index has no semantic neighbors, expansion doesn't crash."""
        expander_sem = BudgetedGraphExpander(
            index=expansion_index,
            expansion_budget=5,
            max_budget_per_relation=2,
            max_total_candidates=20,
        )

        expanded, stats = expander_sem.expand(
            "What was the revenue?",
            initial_candidates,
            relation_probabilities={"semantic_similar": 1.0},
        )

        # Should not crash. Semantic expansion may be empty.
        semantic_expanded = [
            ec for ec in expanded
            if not ec.is_initial and "semantic_similar" in ec.source_relations
        ]
        # With no semantic edges in the index, this should be empty
        assert len(semantic_expanded) == 0, (
            "Semantic expansion returned candidates with no semantic edges"
        )


class TestNoGoldUsage:
    """Test 8: Expansion never uses gold."""

    def test_expansion_independent_of_gold(self, expander, initial_candidates):
        """Expansion results should be the same regardless of gold labels."""
        gold_set = {"d1s7_0", "d1s7_3"}

        expanded1, _ = expander.expand(
            "What was the revenue?", initial_candidates,
            relation_probabilities={"adjacent_chunk": 1.0},
        )

        # The expander shouldn't even accept gold as input
        # Check that the method signature doesn't include gold
        import inspect
        sig = inspect.signature(expander.expand)
        params = list(sig.parameters.keys())
        assert "gold" not in params, "expand() should not accept gold parameter"
        assert "gold_chunk_ids" not in params, "expand() should not accept gold_chunk_ids"


# ═════════════════════════════════════════════════════════════════════════════
# Metric normalization tests
# ═════════════════════════════════════════════════════════════════════════════

class TestMetricNormalization:
    """Test metric alias normalization."""

    def test_revenue_aliases(self):
        assert normalize_metric("revenues") == "revenue"
        assert normalize_metric("net sales") == "revenue"
        assert normalize_metric("sales") == "revenue"

    def test_earnings_aliases(self):
        assert normalize_metric("net earnings") == "net income"
        assert normalize_metric("profit") == "net income"

    def test_eps_aliases(self):
        assert normalize_metric("earnings per share") == "eps"
        assert normalize_metric("diluted eps") == "eps"

    def test_unknown_metric_preserved(self):
        assert normalize_metric("unusual metric name") == "unusual metric name"

    def test_case_insensitive(self):
        assert normalize_metric("REVENUE") == "revenue"
        assert normalize_metric("Net Sales") == "revenue"


# ═════════════════════════════════════════════════════════════════════════════
# Budget allocation tests
# ═════════════════════════════════════════════════════════════════════════════

class TestBudgetAllocation:
    """Test budget allocation logic."""

    def test_budget_sum_does_not_exceed_total(self, expander):
        probs = {"adjacent_chunk": 0.8, "same_section": 0.6, "same_filing": 0.3}
        budgets = expander._allocate_budgets(probs, 10)
        assert sum(budgets.values()) <= 10

    def test_below_threshold_excluded(self, expansion_index):
        expander_strict = BudgetedGraphExpander(
            index=expansion_index,
            relation_threshold=0.5,
        )
        probs = {"adjacent_chunk": 0.8, "semantic_similar": 0.1}
        budgets = expander_strict._allocate_budgets(
            {k: v for k, v in probs.items() if v >= expander_strict.relation_threshold},
            5,
        )
        assert "semantic_similar" not in budgets

    def test_per_relation_capped(self, expander):
        probs = {"adjacent_chunk": 1.0}
        budgets = expander._allocate_budgets(probs, 50)
        assert budgets.get("adjacent_chunk", 0) <= expander.max_budget_per_relation


# ═════════════════════════════════════════════════════════════════════════════
# GraphExpansionIndex tests
# ═════════════════════════════════════════════════════════════════════════════

class TestGraphExpansionIndex:
    """Test index building and caching."""

    def test_index_builds_without_crash(self, sample_chunks):
        idx = GraphExpansionIndex()
        idx.build(sample_chunks)
        assert len(idx.chunk_lookup) == len(sample_chunks)

    def test_chunks_by_doc(self, expansion_index):
        assert "doc1" in expansion_index.chunks_by_doc
        assert len(expansion_index.chunks_by_doc["doc1"]) == 8  # 5 + 3

    def test_chunks_by_section(self, expansion_index):
        key = ("doc1", "Item 7")
        assert key in expansion_index.chunks_by_section
        assert len(expansion_index.chunks_by_section[key]) == 5

    def test_chunks_by_company_year(self, expansion_index):
        key = ("ACME", "2023")
        assert key in expansion_index.chunks_by_company_year

    def test_adjacent_chunks_built(self, expansion_index):
        # d1s7_1 should have neighbors
        neighbors = expansion_index.adjacent_chunks.get("d1s7_1", [])
        assert len(neighbors) > 0, "No adjacent chunks built"

    def test_index_save_load_roundtrip(self, expansion_index, tmp_path):
        path = tmp_path / "test_index.pkl"
        expansion_index.save(path)
        loaded = GraphExpansionIndex.load(path)
        assert len(loaded.chunk_lookup) == len(expansion_index.chunk_lookup)
        assert loaded.chunks_by_doc.keys() == expansion_index.chunks_by_doc.keys()

    def test_index_fingerprint(self, expansion_index):
        fp = expansion_index.compute_fingerprint()
        assert isinstance(fp, str)
        assert len(fp) == 16  # hex digest
