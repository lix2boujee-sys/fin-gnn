"""Tests for R-GCN reranker (feg_rag.rerank.rgcn).

Covers: query-specific features, dataset integration, fusion normalization,
unknown relation handling, API compatibility.
"""

import numpy as np
import torch

from feg_rag.data.chunker import Chunk
from feg_rag.graph.builder import FinancialEvidenceGraph
from feg_rag.rerank.rgcn import (
    RGCNReranker,
    RGCNFusionReranker,
    RGCNRerankDataset,
    RGCNLayer,
)
from feg_rag.rerank.query_features import (
    QUERY_FEATURE_DIM,
    build_query_augmented_features,
)
from feg_rag.rerank.scoring import normalise_score_map


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chunk(chunk_id: str, text: str, company: str = "", filing_year: str = "") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        text=text,
        chunk_type="text",
        company=company,
        filing_year=filing_year,
    )


def _make_mini_graph() -> FinancialEvidenceGraph:
    """Create a tiny graph with 2 chunks, metrics, years, and typed edges."""
    g = FinancialEvidenceGraph()
    g._add_node("c_A", "chunk", text="Revenue was $100M in 2023.",
                company="Acme Inc", filing_year="2023")
    g._add_node("c_B", "chunk", text="Net income was $20M in 2021.",
                company="Acme Inc", filing_year="2021")
    g._add_node("metric::revenue", "metric", name="revenue")
    g._add_node("metric::net_income", "metric", name="net_income")
    g._add_node("year::2023", "year", name="2023")
    g._add_node("year::2021", "year", name="2021")
    g._add_edge("c_A", "metric::revenue", "chunk-mentions-metric", weight=0.8)
    g._add_edge("c_A", "year::2023", "chunk-mentions-year", weight=0.8)
    g._add_edge("c_B", "metric::net_income", "chunk-mentions-metric", weight=0.8)
    g._add_edge("c_B", "year::2021", "chunk-mentions-year", weight=0.8)
    return g


def _make_features(graph: FinancialEvidenceGraph, dim: int = 32) -> dict:
    rng = np.random.RandomState(42)
    features = {}
    for node_id in graph.graph.nodes():
        features[node_id] = rng.randn(dim).astype(np.float32)
    return features


def _make_chunk_lookup() -> dict:
    return {
        "c_A": _make_chunk("c_A", "Revenue was $100M in 2023.",
                            company="Acme Inc", filing_year="2023"),
        "c_B": _make_chunk("c_B", "Net income was $20M in 2021.",
                            company="Acme Inc", filing_year="2021"),
    }


# ---------------------------------------------------------------------------
# Test 1: query-specific features change with query
# ---------------------------------------------------------------------------

def test_query_features_change_with_query_rgcn():
    """Same chunk, different queries → different query feature values."""
    chunk_lookup = _make_chunk_lookup()
    base_features = {"c_A": np.zeros(32, dtype=np.float32),
                     "c_B": np.zeros(32, dtype=np.float32)}

    query_a = "What was the revenue in 2023?"
    query_b = "What was the net income in 2021?"

    ret_scores = {"c_A": 0.8, "c_B": 0.5}

    aug_a = build_query_augmented_features(
        base_features, ["c_A", "c_B"], query_a,
        chunk_lookup=chunk_lookup, retrieval_scores=ret_scores,
    )
    aug_b = build_query_augmented_features(
        base_features, ["c_A", "c_B"], query_b,
        chunk_lookup=chunk_lookup, retrieval_scores=ret_scores,
    )

    # Query-specific features (year_match col=3, metric_match col=4) should differ
    assert aug_a[0, 4] != aug_b[0, 4] or aug_a[0, 3] != aug_b[0, 3], (
        f"Query features should differ between queries. "
        f"A: year={aug_a[0,3]} metric={aug_a[0,4]}, "
        f"B: year={aug_b[0,3]} metric={aug_b[0,4]}"
    )


# ---------------------------------------------------------------------------
# Test 2: RGCNRerankDataset uses question
# ---------------------------------------------------------------------------

def test_rgcn_dataset_uses_question():
    """RGCNRerankDataset.__getitem__ should produce feature dim > base dim,
    proving query features were appended."""
    graph = _make_mini_graph()
    features = _make_features(graph, dim=32)
    chunk_lookup = _make_chunk_lookup()

    samples = [{
        "positive": "c_A",
        "negative": "c_B",
        "question": "What was the revenue in 2023?",
        "retrieval_scores": {"c_A": 0.9, "c_B": 0.3},
    }]

    dataset = RGCNRerankDataset(samples, graph, features, chunk_lookup=chunk_lookup)
    x, adj_list, pos_idx, neg_idx = dataset[0]

    base_dim = 32
    expected_dim = base_dim + QUERY_FEATURE_DIM
    assert x.shape[1] == expected_dim, (
        f"Expected feature dim {expected_dim} (base={base_dim} + aug={QUERY_FEATURE_DIM}), "
        f"got {x.shape[1]}"
    )

    # adj_list should be a tensor stack of per-relation adjacencies
    assert adj_list.ndim == 3, f"Expected 3D adj_list tensor, got shape {adj_list.shape}"
    assert adj_list.shape[0] >= 1  # at least 1 relation
    # pos/neg indices should be valid
    N = x.shape[0]
    assert 0 <= pos_idx < N
    assert 0 <= neg_idx < N


# ---------------------------------------------------------------------------
# Test 3: fusion normalization prevents raw logit dominance
# ---------------------------------------------------------------------------

def test_rgcn_fusion_normalization():
    """Normalised scores prevent one component from raw-scale dominance."""
    ret_map = {"c_A": 0.9, "c_B": 0.5}
    graph_map = {"c_A": 0.01, "c_B": 0.02}
    gnn_map = {"c_A": 0.1, "c_B": 10.0}  # B is 100x A's raw value

    ret_norm = normalise_score_map(ret_map)
    graph_norm = normalise_score_map(graph_map)
    gnn_norm = normalise_score_map(gnn_map)

    # After normalisation, all values in [0, 1]
    for d in [ret_norm, graph_norm, gnn_norm]:
        for v in d.values():
            assert 0.0 <= v <= 1.0

    # alpha=1.0 → retrieval dominates → c_A wins
    score_a = 1.0 * ret_norm["c_A"] + 0.0 * graph_norm.get("c_A", 0) + 0.0 * gnn_norm["c_A"]
    score_b = 1.0 * ret_norm["c_B"] + 0.0 * graph_norm.get("c_B", 0) + 0.0 * gnn_norm["c_B"]
    assert score_a > score_b, "alpha=1.0 should rank by retrieval"

    # gamma=1.0 → GNN dominates → c_B wins (higher GNN logit)
    score_a2 = 0.0 * ret_norm["c_A"] + 0.0 * graph_norm.get("c_A", 0) + 1.0 * gnn_norm["c_A"]
    score_b2 = 0.0 * ret_norm["c_B"] + 0.0 * graph_norm.get("c_B", 0) + 1.0 * gnn_norm["c_B"]
    assert score_b2 > score_a2, "gamma=1.0 should rank by GNN"

    # Mixed: no explosion from raw logit scale
    mixed_a = 0.3 * ret_norm["c_A"] + 0.3 * graph_norm.get("c_A", 0) + 0.4 * gnn_norm["c_A"]
    mixed_b = 0.3 * ret_norm["c_B"] + 0.3 * graph_norm.get("c_B", 0) + 0.4 * gnn_norm["c_B"]
    assert -10.0 < mixed_a < 10.0
    assert -10.0 < mixed_b < 10.0


# ---------------------------------------------------------------------------
# Test 4: unknown relation not mapped to 0
# ---------------------------------------------------------------------------

def test_unknown_relation_skipped():
    """Edges with unknown relation types should be skipped, not pollute relation 0."""
    graph = FinancialEvidenceGraph()
    graph._add_node("c_X", "chunk", text="Test chunk X.")
    graph._add_node("c_Y", "chunk", text="Test chunk Y.")
    # Add edge with type that won't be in relation_map
    graph._add_edge("c_X", "c_Y", "some-unknown-type", weight=0.5)

    features = {
        "c_X": np.zeros(32, dtype=np.float32),
        "c_Y": np.zeros(32, dtype=np.float32),
    }

    # Explicit relation_map with only known types
    relation_map = {"chunk-mentions-metric": 0}

    samples = [{
        "positive": "c_X",
        "negative": "c_Y",
        "question": "test?",
    }]

    # Should not crash on unknown relation
    dataset = RGCNRerankDataset(samples, graph, features, relation_map=relation_map)
    x, adj_list, pos_idx, neg_idx = dataset[0]

    # adj_list[0] should be for "chunk-mentions-metric" (relation 0)
    # The unknown edge should NOT appear in relation 0
    assert adj_list.shape[0] == dataset.num_relations
    # Verify relation 0 adjacency is all zeros (no known edges of type 0)
    adj_0 = adj_list[0].numpy()
    assert np.all(adj_0 == 0.0), (
        f"Relation 0 should be all zeros (unknown edges skipped), got non-zero entries"
    )


# ---------------------------------------------------------------------------
# Test 5: API compatibility
# ---------------------------------------------------------------------------

def test_rgcn_rerank_api():
    """RGCNFusionReranker.rerank() returns correct number of (Chunk, float)."""
    graph = _make_mini_graph()
    features = _make_features(graph)

    base_dim = next(iter(features.values())).shape[0]
    relation_map = {"chunk-mentions-metric": 0, "chunk-mentions-year": 1}
    model = RGCNReranker(
        in_dim=base_dim + QUERY_FEATURE_DIM,
        hidden_dim=16, out_dim=8, num_relations=2,
    )
    reranker = RGCNFusionReranker(
        model, relation_map=relation_map,
        alpha=0.3, beta=0.3, gamma=0.4, device="cpu",
    )

    chunks = [
        _make_chunk("c_A", "Revenue was $100M in 2023.",
                     company="Acme Inc", filing_year="2023"),
        _make_chunk("c_B", "Net income was $20M in 2021.",
                     company="Acme Inc", filing_year="2021"),
    ]
    candidate_chunks = [(chunks[0], 0.9), (chunks[1], 0.5)]
    ppr_scores = {"c_A": 0.7, "c_B": 0.3}

    result = reranker.rerank(
        "What was the revenue in 2023?",
        candidate_chunks,
        graph,
        features,
        ppr_scores=ppr_scores,
    )

    assert len(result) == 2
    for chunk, score in result:
        assert isinstance(chunk, Chunk)
        assert isinstance(score, float)
    cids = [c.chunk_id for c, _ in result]
    assert set(cids) == {"c_A", "c_B"}


# ---------------------------------------------------------------------------
# Test: RGCNLayer forward works with identity norm_list
# ---------------------------------------------------------------------------

def test_rgcn_layer_forward():
    """RGCNLayer.forward should work with pre-normalised adj and identity norm_list."""
    layer = RGCNLayer(in_dim=10, out_dim=6, num_relations=2)

    x = torch.randn(3, 10)  # 3 nodes, 10-dim features
    # Pre-normalised adjacency (identity-like for test)
    adj = torch.eye(3)
    adj_list = [adj.clone(), adj.clone()]
    norm_list = [torch.ones(3), torch.ones(3)]

    out = layer(x, adj_list, norm_list)
    assert out.shape == (3, 6)
    assert not torch.isnan(out).any()

