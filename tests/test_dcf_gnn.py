import numpy as np
import torch

from feg_rag.data.chunker import Chunk
from feg_rag.graph.builder import FinancialEvidenceGraph
from feg_rag.rerank.dcf_gnn import (
    DCFGNNFusionReranker,
    DCFRerankDataset,
    DCFGNNReranker,
    financial_match_features,
    infer_query_type_features,
    split_relation_channels,
)
from feg_rag.rerank.query_features import QUERY_FEATURE_DIM


def _chunk(chunk_id: str, text: str, company: str = "", year: str = "") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        text=text,
        chunk_type="text",
        company=company,
        filing_year=year,
    )


def _graph() -> FinancialEvidenceGraph:
    g = FinancialEvidenceGraph()
    g._add_node("c_a", "chunk", text="Acme revenue was 100 in 2023.", company="Acme", filing_year="2023")
    g._add_node("c_b", "chunk", text="Beta net income was 20 in 2021.", company="Beta", filing_year="2021")
    g._add_node("metric::revenue", "metric", name="revenue")
    g._add_node("year::2023", "year", name="2023")
    g._add_edge("c_a", "metric::revenue", "chunk-mentions-metric", weight=0.8)
    g._add_edge("c_a", "year::2023", "chunk-mentions-year", weight=0.8)
    g._add_edge("c_a", "c_b", "semantic-similar", weight=1.0)
    g._add_edge("c_a", "c_b", "same-company", weight=0.5)
    return g


def _features(g: FinancialEvidenceGraph, dim: int = 16):
    rng = np.random.RandomState(7)
    return {n: rng.randn(dim).astype(np.float32) for n in g.graph.nodes()}


def test_relation_split_keeps_structural_and_semantic_separate():
    relation_map = {
        "chunk-mentions-year": 0,
        "semantic-similar": 1,
        "same-metric": 2,
    }
    structural, semantic = split_relation_channels(relation_map)
    assert "chunk-mentions-year" in structural
    assert "semantic-similar" in semantic
    assert "same-metric" in semantic


def test_query_type_features_are_nonempty():
    q = "Compare revenue growth between 2022 and 2023"
    feats = infer_query_type_features(q)
    assert feats.shape == (4,)
    assert feats[0] == 1.0
    assert feats[1] == 1.0


def test_financial_match_features_detect_conflicts():
    c = _chunk("c_b", "Beta net income was 20 in 2021.", company="Beta", year="2021")
    feats = financial_match_features("What was Acme revenue in 2023?", c)
    assert feats[5] == 1.0 or feats[6] == 1.0 or feats[7] == 1.0


def test_dcf_dataset_shapes():
    g = _graph()
    features = _features(g)
    samples = [{
        "positive": "c_a",
        "negative": "c_b",
        "question": "What was Acme revenue in 2023?",
        "retrieval_scores": {"c_a": 1.0, "c_b": 0.5},
    }]
    ds = DCFRerankDataset(samples, g, features, chunk_lookup={
        "c_a": _chunk("c_a", "Acme revenue was 100 in 2023.", "Acme", "2023"),
        "c_b": _chunk("c_b", "Beta net income was 20 in 2021.", "Beta", "2021"),
    })
    x, s_adj, m_adj, qtype, match, pos, neg, diag = ds[0]
    assert x.shape[1] == 16 + QUERY_FEATURE_DIM
    assert s_adj.ndim == 3
    assert m_adj.ndim == 3
    assert qtype.shape[1] == 4
    assert match.shape[1] == 8
    assert 0 <= pos < x.shape[0]
    assert 0 <= neg < x.shape[0]


def test_dcf_rerank_api_returns_sorted_pairs():
    g = _graph()
    features = _features(g)
    chunks = [
        _chunk("c_a", "Acme revenue was 100 in 2023.", "Acme", "2023"),
        _chunk("c_b", "Beta net income was 20 in 2021.", "Beta", "2021"),
    ]
    relation_map = {
        "chunk-mentions-metric": 0,
        "chunk-mentions-year": 1,
        "semantic-similar": 2,
        "same-company": 3,
    }
    structural, semantic = split_relation_channels(relation_map)
    model = DCFGNNReranker(
        in_dim=16 + QUERY_FEATURE_DIM,
        hidden_dim=12,
        out_dim=8,
        num_structural_relations=len(structural),
        num_semantic_relations=len(semantic),
        dropout=0.0,
    )
    rr = DCFGNNFusionReranker(
        model,
        structural_relation_map=structural,
        semantic_relation_map=semantic,
        device="cpu",
    )
    out = rr.rerank(
        "What was Acme revenue in 2023?",
        [(chunks[0], 1.0), (chunks[1], 0.5)],
        g,
        features,
    )
    assert len(out) == 2
    assert out[0][1] >= out[1][1]
    assert rr.last_diagnostics
