"""Tests for PPR reranker (feg_rag.rerank.ppr).

Covers: bidirectional graph, metric seed propagation, retrieval fusion,
empty inputs, and no-seed stability.
"""

import networkx as nx

from feg_rag.data.chunker import Chunk
from feg_rag.graph.builder import FinancialEvidenceGraph
from feg_rag.rerank.ppr import (
    ppr_rerank,
    _to_bidirectional_ppr_graph,
    _build_subgraph,
    _normalise_dict,
)


# ---------------------------------------------------------------------------
# Helper: mini FinancialEvidenceGraph with chunk → metric one-way edges
# ---------------------------------------------------------------------------

def _make_mini_graph() -> FinancialEvidenceGraph:
    """Create a small FinancialEvidenceGraph with two chunks and one metric.

    Edges:
        c_revenue  →  metric::revenue   (one-way, as in the real graph)
        c_noise    has no metric edges
    """
    g = FinancialEvidenceGraph()
    g._add_node("c_revenue", "chunk", text="Revenue was $100M in 2023.")
    g._add_node("c_noise", "chunk", text="The office is located in Delaware.")
    g._add_node("metric::revenue", "metric", name="revenue")
    g._add_edge("c_revenue", "metric::revenue", "chunk-mentions-metric", weight=0.8)
    return g


def _make_chunks() -> list:
    return [
        Chunk(chunk_id="c_revenue", text="Revenue was $100M in 2023.",
              chunk_type="text"),
        Chunk(chunk_id="c_noise", text="The office is located in Delaware.",
              chunk_type="text"),
    ]


# ---------------------------------------------------------------------------
# Test 1: PPR metric seed propagates back to chunk via bidirectional graph
# ---------------------------------------------------------------------------

def test_ppr_metric_seed_propagates_back():
    """With one-way edge c_revenue → metric::revenue, the bidirectional PPR
    graph should allow score to flow from metric seed back to c_revenue."""
    graph = _make_mini_graph()
    chunks = _make_chunks()

    result = ppr_rerank(
        graph, chunks,
        candidate_chunk_ids=["c_revenue", "c_noise"],
        seed_chunk_ids=[],
        seed_metric_names=["revenue"],
        seed_year_values=[],
    )

    assert len(result) == 2
    scores = {cid: score for cid, score in result}
    assert scores["c_revenue"] > scores["c_noise"], (
        f"Expected c_revenue > c_noise but got {scores}"
    )


# ---------------------------------------------------------------------------
# Test 2: retrieval_scores fusion affects ranking
# ---------------------------------------------------------------------------

def test_retrieval_scores_fusion():
    """retrieval_weight controls the blend of retrieval vs PPR scores."""
    graph = _make_mini_graph()
    chunks = _make_chunks()

    # retrieval_scores favour c_noise, but graph favours c_revenue
    ret_scores = {"c_revenue": 0.2, "c_noise": 0.9}

    # retrieval_weight=1.0 → pure retrieval → c_noise ranks first
    r1 = ppr_rerank(
        graph, chunks,
        candidate_chunk_ids=["c_revenue", "c_noise"],
        seed_chunk_ids=[],
        seed_metric_names=["revenue"],
        seed_year_values=[],
        retrieval_scores=ret_scores,
        retrieval_weight=1.0,
    )
    assert r1[0][0] == "c_noise", f"retrieval_weight=1.0 should rank c_noise first, got {r1}"

    # retrieval_weight=0.0 → pure PPR → c_revenue ranks first
    r0 = ppr_rerank(
        graph, chunks,
        candidate_chunk_ids=["c_revenue", "c_noise"],
        seed_chunk_ids=[],
        seed_metric_names=["revenue"],
        seed_year_values=[],
        retrieval_scores=ret_scores,
        retrieval_weight=0.0,
    )
    assert r0[0][0] == "c_revenue", f"retrieval_weight=0.0 should rank c_revenue first, got {r0}"

    # retrieval_weight=0.5 → mixed, both candidates present, no crash
    r5 = ppr_rerank(
        graph, chunks,
        candidate_chunk_ids=["c_revenue", "c_noise"],
        seed_chunk_ids=[],
        seed_metric_names=["revenue"],
        seed_year_values=[],
        retrieval_scores=ret_scores,
        retrieval_weight=0.5,
    )
    assert len(r5) == 2
    cids = {cid for cid, _ in r5}
    assert cids == {"c_revenue", "c_noise"}


# ---------------------------------------------------------------------------
# Test 3: No seeds → return all candidates with uniform scores
# ---------------------------------------------------------------------------

def test_no_seeds_returns_all_candidates():
    """When no seeds are provided, all candidates receive equal scores."""
    graph = _make_mini_graph()
    chunks = _make_chunks()

    result = ppr_rerank(
        graph, chunks,
        candidate_chunk_ids=["c_revenue", "c_noise"],
        seed_chunk_ids=[],
        seed_metric_names=[],
        seed_year_values=[],
    )

    assert len(result) == 2
    scores = [s for _, s in result]
    # All scores should be equal (uniform distribution)
    assert abs(scores[0] - scores[1]) < 1e-8, f"Scores should be equal, got {scores}"


# ---------------------------------------------------------------------------
# Test 4: Empty candidates → return []
# ---------------------------------------------------------------------------

def test_empty_candidates_returns_empty():
    graph = _make_mini_graph()
    chunks = _make_chunks()

    result = ppr_rerank(
        graph, chunks,
        candidate_chunk_ids=[],
        seed_chunk_ids=[],
        seed_metric_names=[],
        seed_year_values=[],
    )

    assert result == []


# ---------------------------------------------------------------------------
# Test: _to_bidirectional_ppr_graph
# ---------------------------------------------------------------------------

def test_to_bidirectional_ppr_graph():
    """Verify bidirectional conversion preserves nodes and adds reverse edges."""
    nxg = nx.MultiDiGraph()
    nxg.add_node("a")
    nxg.add_node("b")
    nxg.add_node("c")  # isolated node
    nxg.add_edge("a", "b", weight=0.8)
    nxg.add_edge("a", "b", weight=0.5)  # multi-edge → max weight wins

    ppr_g = _to_bidirectional_ppr_graph(nxg)

    # All nodes present
    assert set(ppr_g.nodes()) == {"a", "b", "c"}

    # Both directions exist
    assert ppr_g.has_edge("a", "b")
    assert ppr_g.has_edge("b", "a")

    # Max weight used (0.8, not 0.5)
    assert ppr_g.edges["a", "b"]["weight"] == 0.8
    assert ppr_g.edges["b", "a"]["weight"] == 0.8

    # Isolated node preserved
    assert "c" in ppr_g.nodes()


# ---------------------------------------------------------------------------
# Test: _normalise_dict
# ---------------------------------------------------------------------------

def test_normalise_dict():
    assert _normalise_dict({}) == {}
    assert _normalise_dict({"a": 5.0}) == {"a": 0.5}  # single entry
    assert _normalise_dict({"a": 1.0, "b": 1.0}) == {"a": 0.5, "b": 0.5}  # all equal
    d = _normalise_dict({"a": 0.0, "b": 10.0})
    assert d["a"] == 0.0
    assert d["b"] == 1.0


# ---------------------------------------------------------------------------
# Test: _build_subgraph accepts nx.DiGraph
# ---------------------------------------------------------------------------

def test_build_subgraph_accepts_digraph():
    g = nx.DiGraph()
    g.add_node("c1")
    g.add_node("c2")
    g.add_node("metric::revenue")
    g.add_edge("c1", "metric::revenue")
    g.add_edge("metric::revenue", "c1")  # bidirectional already

    sub = _build_subgraph(
        g,
        candidate_chunk_ids=["c1", "c2"],
        seed_chunk_ids=[],
        seed_metric_names=["revenue"],
        seed_year_values=[],
    )

    assert "c1" in sub.nodes()
    assert "c2" in sub.nodes()
    assert "metric::revenue" in sub.nodes()
