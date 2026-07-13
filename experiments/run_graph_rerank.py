"""Graph reranking experiment: Hybrid → PPR/GNN rerank → LLM answer.

Paper plan §10.1: compares Hybrid-RAG, Graph-PPR-RAG, GNN-RAG.

Usage:
    python experiments/run_graph_rerank.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from feg_rag.config import Config
from feg_rag.data.chunker import Chunk
from feg_rag.data.loader import load_dataset
from feg_rag.evaluation.metrics import compute_all_metrics
from feg_rag.generation.llm import LLMGenerator
from feg_rag.graph.builder import build_financial_evidence_graph
from feg_rag.graph.entities import extract_entities
from feg_rag.graph.features import build_node_features
from feg_rag.rerank.ppr import ppr_rerank
from feg_rag.rerank.train import build_corpus, print_loss_summary, train_gnn_reranker, warmup_retrieval_scores
from feg_rag.retrieval.bm25 import BM25Retriever
from feg_rag.retrieval.dense import DenseRetriever
from feg_rag.retrieval.hybrid import HybridRetriever


def main() -> None:
    parser = argparse.ArgumentParser(description="Graph reranking experiments")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--num_samples", type=int, default=50)
    parser.add_argument("--train_gnn", action="store_true", help="Train GNN reranker")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    cfg.ensure_dirs()

    # ---- 1. Data ----
    print("[1/6] Loading data...")
    samples = load_dataset(cfg.datasets[0], cfg.data_dir)
    if args.num_samples > 0:
        samples = samples[: args.num_samples]
    print(f"  {len(samples)} QA samples")

    # ---- 2. Corpus ----
    print("[2/6] Building chunk corpus...")
    corpus_chunks, gold_map = build_corpus(samples, cfg)
    print(f"  {len(corpus_chunks)} total chunks")

    # ---- 3. Retrieval ----
    print("[3/6] Building retrieval indices...")
    bm25 = BM25Retriever(k1=cfg.retrieval["bm25_k1"], b=cfg.retrieval["bm25_b"])
    bm25.index(corpus_chunks)
    dense = DenseRetriever(
        model_name=cfg.retrieval["dense_model"],
        query_instruction=cfg.retrieval.get("dense_query_instruction"),
        e5_max_seq_length=cfg.retrieval.get("e5_max_seq_length", 512),
        e5_batch_size=cfg.retrieval.get("e5_batch_size"),
        debug=cfg.retrieval.get("debug_dense", False),
    )
    dense.index(corpus_chunks)
    hybrid = HybridRetriever(bm25, dense, alpha=cfg.retrieval["hybrid_alpha"])

    # ---- 4. Graph ----
    print("[4/6] Building financial evidence graph...")
    entity_map = extract_entities(corpus_chunks)
    graph = build_financial_evidence_graph(
        corpus_chunks, entity_map=entity_map, add_semantic_edges=False
    )
    print(f"  Graph: {graph.num_nodes} nodes, {graph.num_edges} edges")
    print(f"  Edge types: {graph.edge_type_counts()}")

    # Build features (placeholder embeddings for now)
    retrieval_scores = warmup_retrieval_scores(samples, hybrid, top_k=50)
    features = build_node_features(
        graph, corpus_chunks, entity_map, retrieval_scores
    )

    # ---- 5. Rerank + Answer ----
    print("[5/6] Running reranking experiments...")
    top_k_retrieval = cfg.retrieval["top_k"]
    top_k_gen = cfg.generation["top_k_evidence"]
    generator = LLMGenerator(
        model=cfg.generation["model"],
        temperature=cfg.generation["temperature"],
        max_tokens=cfg.generation["max_tokens"],
    )

    # Optionally train GNN
    gnn_reranker = None
    if args.train_gnn:
        print("  Training GNN reranker...")
        gnn_reranker, history, _ = train_gnn_reranker(
            samples, hybrid, graph, features, gold_map, cfg
        )
        if gnn_reranker and history:
            print_loss_summary(history)

    all_results: Dict[str, List[Dict]] = {
        "hybrid_rag": [],
        "graph_ppr_rag": [],
    }
    if gnn_reranker:
        all_results["gnn_rag"] = []

    for i, s in enumerate(samples):
        if i % 10 == 0:
            print(f"  {i}/{len(samples)}")

        question = s["question"]
        q_id = s["id"]

        # Hybrid retrieval
        hybrid_results = hybrid.search(question, top_k=top_k_retrieval)
        candidate_ids = [c.chunk_id for c, _ in hybrid_results]

        # --- Hybrid-RAG ---
        top_chunks = [c for c, _ in hybrid_results[:top_k_gen]]
        gen = generator.generate(question, top_chunks)
        all_results["hybrid_rag"].append(
            _make_result(s, gen, candidate_ids, gold_map.get(q_id, []))
        )

        # --- PPR Rerank ---
        from feg_rag.graph.entities import EntityExtractor
        extractor = EntityExtractor()
        q_metrics = extractor.extract_metrics(question)
        q_years = extractor.extract_years(question)

        ppr_scores = ppr_rerank(
            graph,
            corpus_chunks,
            candidate_ids,
            seed_chunk_ids=candidate_ids[:10],
            seed_metric_names=list(q_metrics),
            seed_year_values=list(q_years),
            alpha=cfg.rerank["ppr_alpha"],
        )
        ppr_sorted = sorted(ppr_scores, key=lambda x: x[1], reverse=True)
        ppr_chunk_ids = [cid for cid, _ in ppr_sorted[:top_k_gen]]
        ppr_chunks = [_find_chunk(cid, corpus_chunks) for cid in ppr_chunk_ids]
        ppr_chunks = [c for c in ppr_chunks if c is not None]
        gen_ppr = generator.generate(question, ppr_chunks)
        all_results["graph_ppr_rag"].append(
            _make_result(
                s, gen_ppr, [cid for cid, _ in ppr_sorted], gold_map.get(q_id, [])
            )
        )

        # --- GNN Rerank ---
        if gnn_reranker:
            ppr_dict = dict(ppr_scores)
            gnn_reranked = gnn_reranker.rerank(
                question, hybrid_results, graph, features, ppr_scores=ppr_dict
            )
            gnn_chunks = [c for c, _ in gnn_reranked[:top_k_gen]]
            gen_gnn = generator.generate(question, gnn_chunks)
            all_results["gnn_rag"].append(
                _make_result(
                    s,
                    gen_gnn,
                    [c.chunk_id for c, _ in gnn_reranked],
                    gold_map.get(q_id, []),
                )
            )

    # ---- 6. Evaluate ----
    print("[6/6] Computing metrics...")
    k_vals = cfg.evaluation["recall_k_values"]
    summary = {}
    for method_name, results in all_results.items():
        er = compute_all_metrics(method_name, results, k_values=k_vals)
        summary[method_name] = {
            "evidence_recall": er.evidence_recall,
            "evidence_precision": er.evidence_precision,
            "mrr": er.mrr,
            "ndcg": er.ndcg,
            "answer_accuracy": er.answer_accuracy,
            "f1": er.f1,
            "num_samples": er.num_samples,
        }

    # Print
    print("\n" + "=" * 60)
    print("GRAPH RERANKING RESULTS")
    print("=" * 60)
    for method, metrics in summary.items():
        print(f"\n{method}:")
        print(f"  Answer Accuracy: {metrics['answer_accuracy']:.4f}")
        print(f"  MRR:             {metrics['mrr']:.4f}")
        for k in k_vals:
            if k in metrics["evidence_recall"]:
                print(f"  Evidence R@{k}:    {metrics['evidence_recall'][k]:.4f}")

    output_path = args.output or str(cfg.output_dir / "graph_rerank_results.json")
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════

def _find_chunk(cid: str, chunks: List[Chunk]) -> Chunk | None:
    for c in chunks:
        if c.chunk_id == cid:
            return c
    return None


def _make_result(sample, gen, retrieved_ids, gold_ids) -> Dict:
    import re
    pred = re.sub(r"\s+", " ", gen.answer.lower().strip().rstrip("."))
    gold = re.sub(r"\s+", " ", sample["answer"].lower().strip().rstrip("."))
    return {
        "question_id": sample["id"],
        "question": sample["question"],
        "gold_answer": sample["answer"],
        "generated_answer": gen.answer,
        "gold_evidence_ids": gold_ids,
        "retrieved_chunk_ids": retrieved_ids,
        "answer_is_correct": pred == gold,
        "is_consistent": len(gen.cited_chunk_ids) > 0,
        "is_hallucination": "INSUFFICIENT_EVIDENCE" not in gen.answer.upper()
        and len(gen.cited_chunk_ids) == 0,
    }


if __name__ == "__main__":
    main()
