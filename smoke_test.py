"""Minimal smoke test — no LLM API, no HF download needed.

Verifies: data loading → chunking → BM25 retrieval → graph → PPR rerank.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from feg_rag.config import Config
from feg_rag.data.loader import load_dataset
from feg_rag.data.chunker import chunk_text
from feg_rag.retrieval.bm25 import BM25Retriever
from feg_rag.graph.builder import build_financial_evidence_graph
from feg_rag.graph.entities import extract_entities
from feg_rag.rerank.ppr import ppr_rerank
from feg_rag.evaluation.metrics import compute_all_metrics


def main():
    cfg = Config.from_yaml("configs/default.yaml")

    # 1. Data
    print("=" * 50)
    print("[1] Loading FinDER data...")
    samples = load_dataset("finder", cfg.data_dir)[:20]
    print(f"    {len(samples)} samples loaded")
    print(f"    Sample Q: {samples[0]['question'][:100]}")
    print(f"    Sample A: {samples[0]['answer'][:100]}")
    print(f"    Evidence texts: {len(samples[0]['evidence_texts'])} pieces")

    # 2. Chunking
    print("\n[2] Chunking evidence texts...")
    corpus, gold_map = [], {}
    for s in samples:
        gold_ids = []
        for text in s["evidence_texts"]:
            for c in chunk_text(text, cfg.chunk_size, cfg.chunk_overlap, doc_id=s["id"]):
                corpus.append(c)
                gold_ids.append(c.chunk_id)
        gold_map[s["id"]] = gold_ids
    print(f"    {len(corpus)} total chunks")

    # 3. BM25
    print("\n[3] BM25 retrieval...")
    bm25 = BM25Retriever(k1=1.5, b=0.75)
    bm25.index(corpus)

    all_results = []
    for s in samples:
        retrieved = bm25.search(s["question"], top_k=20)
        retrieved_ids = [c.chunk_id for c, _ in retrieved]
        all_results.append({
            "question_id": s["id"],
            "question": s["question"],
            "gold_answer": s["answer"],
            "gold_evidence_ids": gold_map[s["id"]],
            "retrieved_chunk_ids": retrieved_ids,
        })

    # 4. Evaluate retrieval
    print("\n[4] BM25 retrieval metrics:")
    er = compute_all_metrics("bm25", all_results, k_values=[1, 3, 5, 10])
    print(f"    Evidence R@1:  {er.evidence_recall.get(1, 0):.4f}")
    print(f"    Evidence R@3:  {er.evidence_recall.get(3, 0):.4f}")
    print(f"    Evidence R@5:  {er.evidence_recall.get(5, 0):.4f}")
    print(f"    Evidence R@10: {er.evidence_recall.get(10, 0):.4f}")
    print(f"    MRR:           {er.mrr:.4f}")

    # 5. Graph construction
    print("\n[5] Building financial evidence graph...")
    entity_map = extract_entities(corpus)
    graph = build_financial_evidence_graph(corpus, entity_map=entity_map,
                                           add_semantic_edges=False)
    print(f"    Nodes: {graph.num_nodes}, Edges: {graph.num_edges}")
    print(f"    Edge type counts: {graph.edge_type_counts()}")

    # 6. PPR reranking
    print("\n[6] PPR reranking...")
    from feg_rag.graph.entities import EntityExtractor
    extractor = EntityExtractor()

    ppr_results = []
    for s in samples:
        retrieved = bm25.search(s["question"], top_k=20)
        candidate_ids = [c.chunk_id for c, _ in retrieved]
        q_metrics = extractor.extract_metrics(s["question"])
        q_years = extractor.extract_years(s["question"])

        ppr_scores = ppr_rerank(
            graph, corpus, candidate_ids,
            seed_chunk_ids=candidate_ids[:5],
            seed_metric_names=list(q_metrics),
            seed_year_values=list(q_years),
        )
        ppr_ids = [cid for cid, _ in sorted(ppr_scores, key=lambda x: x[1], reverse=True)]
        ppr_results.append({
            "question_id": s["id"],
            "question": s["question"],
            "gold_answer": s["answer"],
            "gold_evidence_ids": gold_map[s["id"]],
            "retrieved_chunk_ids": ppr_ids,
        })

    # 7. Evaluate PPR
    print("\n[7] PPR reranking metrics:")
    er_ppr = compute_all_metrics("ppr", ppr_results, k_values=[1, 3, 5, 10])
    print(f"    Evidence R@1:  {er_ppr.evidence_recall.get(1, 0):.4f}")
    print(f"    Evidence R@3:  {er_ppr.evidence_recall.get(3, 0):.4f}")
    print(f"    Evidence R@5:  {er_ppr.evidence_recall.get(5, 0):.4f}")
    print(f"    Evidence R@10: {er_ppr.evidence_recall.get(10, 0):.4f}")
    print(f"    MRR:           {er_ppr.mrr:.4f}")

    # 8. Compare
    print("\n" + "=" * 50)
    print("SUMMARY: BM25 vs PPR")
    print("=" * 50)
    print(f"{'Metric':<20} {'BM25':>8} {'PPR':>8} {'Delta':>8}")
    print("-" * 44)
    for k in [1, 3, 5, 10]:
        b = er.evidence_recall.get(k, 0)
        p = er_ppr.evidence_recall.get(k, 0)
        d = p - b
        print(f"{f'Evidence R@{k}':<20} {b:>8.4f} {p:>8.4f} {d:>+8.4f}")
    b_mrr = er.mrr
    p_mrr = er_ppr.mrr
    print(f"{'MRR':<20} {b_mrr:>8.4f} {p_mrr:>8.4f} {p_mrr - b_mrr:>+8.4f}")

    print("\n[OK] All checks passed! Framework is ready for experiments.")


if __name__ == "__main__":
    main()
