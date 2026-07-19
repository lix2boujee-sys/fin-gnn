"""Tests for GraphSAGE / GATv2 / GNN reranker (feg_rag.rerank.gnn).

Covers: self-loop, query-specific features, fusion normalization, API compat,
GATv2 forward shape, dense attention masking, multi-head aggregation, param count.
"""

import numpy as np
import torch

from feg_rag.data.chunker import Chunk
from feg_rag.graph.builder import FinancialEvidenceGraph
from feg_rag.rerank.gnn import (
    GraphSAGEReranker,
    GNNFusionReranker,
    RerankDataset,
    DenseGATv2Layer,
    GATv2Reranker,
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


def _make_mini_graph(chunks=None) -> FinancialEvidenceGraph:
    """Create a tiny graph with 2 chunk nodes and one edge."""
    g = FinancialEvidenceGraph()
    g._add_node("c_A", "chunk", text="Revenue was $100M in 2023.",
                company="Acme Inc", filing_year="2023")
    g._add_node("c_B", "chunk", text="Net income was $20M in 2021.",
                company="Acme Inc", filing_year="2021")
    g._add_node("metric::revenue", "metric", name="revenue")
    g._add_node("year::2023", "year", name="2023")
    g._add_node("company::Acme Inc", "company", name="Acme Inc")
    g._add_edge("c_A", "metric::revenue", "chunk-mentions-metric", weight=0.8)
    g._add_edge("c_A", "year::2023", "chunk-mentions-year", weight=0.8)
    g._add_edge("c_B", "metric::net_income", "chunk-mentions-metric", weight=0.8)
    g._add_edge("c_B", "year::2021", "chunk-mentions-year", weight=0.8)
    return g


def _make_features(graph: FinancialEvidenceGraph, dim: int = 32) -> dict:
    """Create random features for all nodes."""
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
# Test 1: self-loop on adjacency
# ---------------------------------------------------------------------------

def test_self_loop_in_dataset():
    """RerankDataset should produce adjacency with non-zero diagonal after
    normalisation (self-loop preserves own-node features)."""
    graph = _make_mini_graph()
    features = _make_features(graph)
    chunk_lookup = _make_chunk_lookup()

    samples = [{
        "positive": "c_A",
        "negative": "c_B",
        "question": "What was the revenue in 2023?",
        "retrieval_scores": {"c_A": 0.9, "c_B": 0.3},
    }]

    dataset = RerankDataset(samples, graph, features, chunk_lookup=chunk_lookup)
    x, adj, pos_idx, neg_idx = dataset[0]

    # Adjacency diagonal should be > 0 (self-loop preserved through norm)
    diag = adj.diagonal().numpy()
    assert np.all(diag > 0), f"Expected self-loop in adj diag, got {diag}"

    # pos/neg indices should be valid
    N = adj.shape[0]
    assert 0 <= pos_idx < N
    assert 0 <= neg_idx < N


# ---------------------------------------------------------------------------
# Test 2: query-specific features change with query
# ---------------------------------------------------------------------------

def test_query_features_change_with_query():
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

    # Query A: revenue + 2023 → metric_match for c_A should differ
    # c_A mentions revenue, c_B mentions net_income
    # year_match col=3, metric_match col=4
    assert aug_a[0, 4] != aug_b[0, 4] or aug_a[0, 3] != aug_b[0, 3], (
        f"Query features should differ between queries. "
        f"A: year={aug_a[0,3]} metric={aug_a[0,4]}, "
        f"B: year={aug_b[0,3]} metric={aug_b[0,4]}"
    )


# ---------------------------------------------------------------------------
# Test 3: fusion normalization prevents raw logit dominance
# ---------------------------------------------------------------------------

def test_fusion_normalization():
    """Raw GNN logits should be normalised before fusion, preventing
    one component from dominating regardless of weight settings."""
    # score maps with very different scales
    ret_map = {"c_A": 0.9, "c_B": 0.5}
    graph_map = {"c_A": 0.01, "c_B": 0.02}

    # GNN logits: B is 100x bigger raw value
    gnn_map = {"c_A": 0.1, "c_B": 10.0}

    ret_norm = normalise_score_map(ret_map)
    graph_norm = normalise_score_map(graph_map)
    gnn_norm = normalise_score_map(gnn_map)

    # After normalisation, gnn values should be in [0, 1]
    assert 0.0 <= gnn_norm["c_A"] <= 1.0
    assert 0.0 <= gnn_norm["c_B"] <= 1.0

    # alpha=1.0 → retrieval only → c_A (0.9) beats c_B (0.5)
    alpha_score_a = 1.0 * ret_norm["c_A"] + 0.0 * graph_norm["c_A"] + 0.0 * gnn_norm["c_A"]
    alpha_score_b = 1.0 * ret_norm["c_B"] + 0.0 * graph_norm["c_B"] + 0.0 * gnn_norm["c_B"]
    assert alpha_score_a > alpha_score_b, "alpha=1.0 should rank by retrieval"

    # gamma=1.0 → GNN only → c_B should rank first (higher raw logit)
    gamma_score_a = 0.0 * ret_norm["c_A"] + 0.0 * graph_norm["c_A"] + 1.0 * gnn_norm["c_A"]
    gamma_score_b = 0.0 * ret_norm["c_B"] + 0.0 * graph_norm["c_B"] + 1.0 * gnn_norm["c_B"]
    assert gamma_score_b > gamma_score_a, "gamma=1.0 should rank by GNN"

    # Mixed weights: both scores in [0,1] range, no raw-logit dominance
    mixed_a = 0.3 * ret_norm["c_A"] + 0.3 * graph_norm["c_A"] + 0.4 * gnn_norm["c_A"]
    mixed_b = 0.3 * ret_norm["c_B"] + 0.3 * graph_norm["c_B"] + 0.4 * gnn_norm["c_B"]
    assert -10.0 < mixed_a < 10.0  # sanity: no explosion
    assert -10.0 < mixed_b < 10.0


# ---------------------------------------------------------------------------
# Test 4: rerank API compatibility
# ---------------------------------------------------------------------------

def test_rerank_api():
    """GNNFusionReranker.rerank() returns correct number of (Chunk, float) tuples."""
    graph = _make_mini_graph()
    features = _make_features(graph)

    # Model with augmented feature dim
    base_dim = next(iter(features.values())).shape[0]
    model = GraphSAGEReranker(in_dim=base_dim + QUERY_FEATURE_DIM, hidden_dim=16, out_dim=8)
    reranker = GNNFusionReranker(model, alpha=0.3, beta=0.3, gamma=0.4, device="cpu")

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
# GATv2 Tests
# ---------------------------------------------------------------------------

def test_gatv2_forward_shape():
    """GATv2Reranker.forward produces expected output shape (N, 1)."""
    in_dim = 32
    hidden_dim = 16
    out_dim = 8
    heads = 2
    N = 10

    model = GATv2Reranker(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        out_dim=out_dim,
        heads=heads,
        dropout=0.0,
    )
    model.eval()

    x = torch.randn(N, in_dim)
    adj = torch.eye(N)  # self-loop only, simplified for shape test

    with torch.no_grad():
        scores = model(x, adj)

    assert scores.shape == (N, 1), f"Expected (N, 1), got {scores.shape}"


def test_gatv2_dense_layer_attention_shape():
    """DenseGATv2Layer produces expected output shapes for concat modes."""
    in_dim, out_dim, heads, N = 16, 8, 4, 20

    # Test concat=True
    layer_concat = DenseGATv2Layer(in_dim, out_dim, heads=heads, concat=True)
    x = torch.randn(N, in_dim)
    adj = torch.eye(N)  # self-loops
    adj[0, 1] = adj[1, 0] = 1.0  # add one edge
    out = layer_concat(x, adj)
    assert out.shape == (N, heads * out_dim), \
        f"concat=True: expected ({N}, {heads * out_dim}), got {out.shape}"

    # Test concat=False
    layer_avg = DenseGATv2Layer(in_dim, out_dim, heads=heads, concat=False)
    out_avg = layer_avg(x, adj)
    assert out_avg.shape == (N, out_dim), \
        f"concat=False: expected ({N}, {out_dim}), got {out_avg.shape}"


def test_gatv2_rerank_api():
    """GNNFusionReranker + GATv2Reranker works with rerank API."""
    graph = _make_mini_graph()
    features = _make_features(graph)

    base_dim = next(iter(features.values())).shape[0]
    model = GATv2Reranker(
        in_dim=base_dim + QUERY_FEATURE_DIM,
        hidden_dim=16,
        out_dim=8,
        heads=2,
        dropout=0.0,
    )
    reranker = GNNFusionReranker(model, alpha=0.3, beta=0.3, gamma=0.4, device="cpu")

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


def test_gatv2_parameter_count():
    """GATv2 has more parameters than GraphSAGE (multi-head attention)."""
    in_dim = 32 + QUERY_FEATURE_DIM
    hidden_dim = 16
    out_dim = 8

    sage = GraphSAGEReranker(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim)
    gatv2 = GATv2Reranker(in_dim=in_dim, hidden_dim=hidden_dim,
                           out_dim=out_dim, heads=2)

    sage_params = sum(p.numel() for p in sage.parameters())
    gatv2_params = sum(p.numel() for p in gatv2.parameters())

    # GATv2 should have more params due to multi-head attention projections
    assert gatv2_params > sage_params, \
        f"GATv2 params ({gatv2_params}) should exceed GraphSAGE params ({sage_params})"


# ---------------------------------------------------------------------------
# Test: normalise_score_map edge cases
# ---------------------------------------------------------------------------

def test_normalise_score_map():
    assert normalise_score_map({}) == {}
    assert normalise_score_map({"x": 5.0}) == {"x": 0.5}
    result = normalise_score_map({"a": 1.0, "b": 1.0})
    assert result == {"a": 0.5, "b": 0.5}
    result = normalise_score_map({"a": 0.0, "b": 10.0})
    assert result["a"] == 0.0
    assert result["b"] == 1.0
