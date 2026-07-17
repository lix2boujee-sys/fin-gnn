import numpy as np
import torch

from feg_rag.graph.builder import FinancialEvidenceGraph
from feg_rag.rerank.finpath_rgcn import FinPathRGCNReranker
from feg_rag.rerank.path_encoder import (
    FinancialPathExtractor,
    LearnablePathEncoder,
    PathAggregator,
    build_path_vocab,
    tensorize_paths,
)


def _mini_finpath_graph() -> FinancialEvidenceGraph:
    g = FinancialEvidenceGraph()
    g._add_node("company::Acme Inc", "company", name="Acme Inc")
    g._add_node("company::Other Inc", "company", name="Other Inc")
    g._add_node("filing::Acme Inc_10-K_2023", "filing", company="Acme Inc", filing_type="10-K", filing_year="2023")
    g._add_node("filing::Other Inc_10-K_2023", "filing", company="Other Inc", filing_type="10-K", filing_year="2023")
    g._add_node("section::MD&A", "section", name="MD&A")
    g._add_node("c_good", "chunk", text="Revenue was higher in 2023.", company="Acme Inc", filing_year="2023")
    g._add_node("c_bad_year", "chunk", text="Revenue was higher in 2022.", company="Acme Inc", filing_year="2022")
    g._add_node("metric::revenue", "metric", name="revenue")
    g._add_node("metric::profit", "metric", name="profit")
    g._add_node("year::2023", "year", name="2023")
    g._add_node("year::2022", "year", name="2022")
    g._add_edge("company::Acme Inc", "filing::Acme Inc_10-K_2023", "company-has-filing")
    g._add_edge("company::Other Inc", "filing::Other Inc_10-K_2023", "company-has-filing")
    g._add_edge("filing::Acme Inc_10-K_2023", "section::MD&A", "filing-has-section")
    g._add_edge("section::MD&A", "c_good", "section-has-chunk")
    g._add_edge("section::MD&A", "c_bad_year", "section-has-chunk")
    g._add_edge("c_good", "metric::revenue", "chunk-mentions-metric")
    g._add_edge("c_good", "year::2023", "chunk-mentions-year")
    g._add_edge("c_bad_year", "metric::revenue", "chunk-mentions-metric")
    g._add_edge("c_bad_year", "year::2022", "chunk-mentions-year")
    g._add_edge("c_bad_year", "metric::profit", "chunk-mentions-metric")
    g._add_edge("c_bad_year", "c_good", "semantic-similar")
    return g


def _entities():
    return {"company": "Acme Inc", "years": ["2023"], "metrics": ["revenue"], "filing_type": "10-K"}


def test_path_extractor_returns_metric_path():
    g = _mini_finpath_graph()
    paths = FinancialPathExtractor().extract_paths(g, ["c_good"], _entities())
    assert any(p.path_type == "chunk_metric" for p in paths["c_good"])
    assert any(p.match_flags["metric_match"] == 1 for p in paths["c_good"])


def test_path_extractor_returns_year_conflict_path():
    g = _mini_finpath_graph()
    paths = FinancialPathExtractor().extract_paths(g, ["c_bad_year"], _entities())
    assert any(p.path_type == "year_conflict" for p in paths["c_bad_year"])
    assert any(p.conflict_flags["year_conflict"] == 1 for p in paths["c_bad_year"])


def test_learnable_path_encoder_handles_variable_paths():
    g = _mini_finpath_graph()
    paths_map = FinancialPathExtractor().extract_paths(g, ["c_good", "c_bad_year"], _entities())
    vocab = build_path_vocab(paths_map)
    tensors = tensorize_paths(paths_map["c_good"], vocab, max_paths=8, max_path_len=4)
    enc = LearnablePathEncoder(
        num_relations=max(vocab["relation"].values()) + 1,
        num_node_types=max(vocab["node_type"].values()) + 1,
        num_path_types=max(vocab["path_type"].values()) + 1,
        hidden_dim=16,
    )
    out = enc(
        tensors["relation_ids"],
        tensors["src_node_type_ids"],
        tensors["dst_node_type_ids"],
        tensors["path_type_ids"],
        tensors["flag_features"],
        tensors["step_mask"],
    )
    assert out.shape == (8, 16)
    assert not torch.isnan(out).any()


def test_path_aggregator_no_path_embedding():
    agg = PathAggregator(hidden_dim=16)
    path_embeddings = torch.zeros(4, 16)
    path_mask = torch.zeros(4, dtype=torch.bool)
    query_embedding = torch.zeros(16)
    out, debug = agg(path_embeddings, query_embedding, path_mask=path_mask)
    assert out.shape == (16,)
    assert debug["max_attention_index"] is None


def test_residual_fusion_preserves_rgcn_when_tau_zero():
    g = _mini_finpath_graph()
    paths_map = FinancialPathExtractor().extract_paths(g, ["c_good", "c_bad_year"], _entities())
    vocab = build_path_vocab(paths_map)
    model = FinPathRGCNReranker(vocab=vocab, hidden_dim=16, tau=0.0, device="cpu")
    ranked = model.rerank(
        query="What was revenue in 2023?",
        query_id="q1",
        candidate_chunk_ids=["c_good", "c_bad_year"],
        retrieval_scores=[0.4, 0.9],
        graph=g,
        query_entities=_entities(),
        rgcn_scores=[0.2, 0.8],
    )
    assert ranked[0][0] == "c_bad_year"
    assert np.isclose(ranked[0][1], 0.8)


def test_finpath_reranker_output_sorted_and_complete():
    g = _mini_finpath_graph()
    paths_map = FinancialPathExtractor().extract_paths(g, ["c_good", "c_bad_year"], _entities())
    vocab = build_path_vocab(paths_map)
    model = FinPathRGCNReranker(vocab=vocab, hidden_dim=16, tau=0.2, device="cpu")
    ranked = model.rerank(
        query="What was revenue in 2023?",
        query_id="q1",
        candidate_chunk_ids=["c_good", "c_bad_year"],
        retrieval_scores=[0.9, 0.4],
        graph=g,
        query_entities=_entities(),
        rgcn_scores=[0.7, 0.1],
    )
    assert len(ranked) == 2
    assert {cid for cid, _ in ranked} == {"c_good", "c_bad_year"}
    assert ranked[0][1] >= ranked[1][1]

