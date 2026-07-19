from feg_rag.graph.builder import FinancialEvidenceGraph
from feg_rag.rerank.finmuse import FinMUSESetReranker, evidence_set_metrics


def _graph():
    g = FinancialEvidenceGraph()
    g._add_node("company::Acme Inc", "company", name="Acme Inc")
    g._add_node("filing::Acme Inc_10-K_2023", "filing", company="Acme Inc", filing_type="10-K", filing_year="2023")
    g._add_node("section::MD&A", "section", name="MD&A")
    g._add_node("c_seed", "chunk", text="Revenue in 2023.", company="Acme Inc", filing_year="2023")
    g._add_node("c_companion", "chunk", text="Revenue explanation.", company="Acme Inc", filing_year="2023")
    g._add_node("c_wrong_year", "chunk", text="Revenue in 2021.", company="Acme Inc", filing_year="2021")
    g._add_node("metric::revenue", "metric", name="revenue")
    g._add_node("year::2023", "year", name="2023")
    g._add_node("year::2021", "year", name="2021")
    g._add_edge("company::Acme Inc", "filing::Acme Inc_10-K_2023", "company-has-filing")
    g._add_edge("filing::Acme Inc_10-K_2023", "section::MD&A", "filing-has-section")
    g._add_edge("section::MD&A", "c_seed", "section-has-chunk")
    g._add_edge("section::MD&A", "c_companion", "section-has-chunk")
    g._add_edge("c_seed", "filing::Acme Inc_10-K_2023", "chunk-belongs-to-filing")
    g._add_edge("c_companion", "filing::Acme Inc_10-K_2023", "chunk-belongs-to-filing")
    g._add_edge("c_wrong_year", "filing::Acme Inc_10-K_2023", "chunk-belongs-to-filing")
    g._add_edge("c_seed", "metric::revenue", "chunk-mentions-metric")
    g._add_edge("c_seed", "year::2023", "chunk-mentions-year")
    g._add_edge("c_companion", "metric::revenue", "chunk-mentions-metric")
    g._add_edge("c_companion", "year::2023", "chunk-mentions-year")
    g._add_edge("c_wrong_year", "metric::revenue", "chunk-mentions-metric")
    g._add_edge("c_wrong_year", "year::2021", "chunk-mentions-year")
    g._add_edge("c_seed", "c_companion", "semantic-similar")
    return g


def test_finmuse_builds_set_and_ranks_selected_first():
    g = _graph()
    reranker = FinMUSESetReranker(max_set_size=3, seed_top_k=2)
    ranked, best, _ = reranker.rerank(
        "What was Acme revenue in 2023?",
        ["c_seed", "c_companion", "c_wrong_year"],
        [1.0, 0.8, 0.7],
        g,
        {"company": "Acme Inc", "years": ["2023"], "metrics": ["revenue"]},
    )
    assert best.passage_ids[0] == "c_seed"
    assert "c_companion" in best.passage_ids
    assert len(ranked) == 3
    assert set(ranked) == {"c_seed", "c_companion", "c_wrong_year"}
    assert ranked.index("c_wrong_year") > ranked.index("c_seed")


def test_finmuse_conflict_penalizes_wrong_year():
    g = _graph()
    reranker = FinMUSESetReranker(max_set_size=3)
    prof_good = reranker.chunk_profile(g, "c_seed")
    prof_bad = reranker.chunk_profile(g, "c_wrong_year")
    q = {"company": {"acme inc"}, "year": {"2023"}, "metric": {"revenue"}, "filing": set(), "section": set()}
    assert reranker._conflict_score([prof_bad], q) > reranker._conflict_score([prof_good], q)


def test_evidence_set_metrics_shape():
    g = _graph()
    reranker = FinMUSESetReranker(max_set_size=3)
    _ranked, best, _ = reranker.rerank(
        "What was Acme revenue in 2023?",
        ["c_seed", "c_companion", "c_wrong_year"],
        [1.0, 0.8, 0.7],
        g,
        {"company": "Acme Inc", "years": ["2023"], "metrics": ["revenue"]},
    )
    m = evidence_set_metrics(
        [{"gold_evidence_ids": ["c_seed", "c_companion"]}],
        [best],
    )
    assert set(m) == {
        "evidence_set_gold_coverage",
        "query_entity_coverage",
        "conflict_rate",
        "redundancy_rate",
    }
