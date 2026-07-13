"""Batch experiment runner for FEG-RAG.

Runs all baselines specified in a config file, caches results, and outputs
structured JSON for table generation.

Usage:
    python experiments/runner.py --config configs/table1_retrieval.yaml
    python experiments/runner.py --config configs/table1_retrieval.yaml --dry-run
    python experiments/runner.py --config configs/table1_retrieval.yaml --method all
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from feg_rag.config import Config
from feg_rag.data.chunker import Chunk, chunk_text
from feg_rag.data.hard_negatives import generate_hard_negatives
from feg_rag.data.loader import load_dataset
from feg_rag.evaluation.metrics import compute_all_metrics
from feg_rag.graph.builder import build_financial_evidence_graph
from feg_rag.graph.entities import EntityExtractor, extract_entities
from feg_rag.retrieval.bm25 import BM25Retriever
from feg_rag.retrieval.dense import DenseRetriever
from feg_rag.retrieval.hybrid import HybridRetriever
from feg_rag.retrieval.cross_encoder import CrossEncoderReranker
from feg_rag.rerank.ppr import ppr_rerank
from feg_rag.rerank.fusion import ConstraintScorer, FusionScorer


# ═════════════════════════════════════════════════════════════════════════════
# Method registry
# ═════════════════════════════════════════════════════════════════════════════

RETRIEVAL_METHODS = ["bm25", "dense", "hybrid"]
RERANK_METHODS = ["cross_encoder", "ppr", "ppr_constraint"]  # GNN added if trained


def run_retrieval_baselines(
    cfg: Config,
    samples: List[Dict],
    corpus_chunks: List[Chunk],
    gold_map: Dict[str, List[str]],
    top_k: int = 50,
) -> Dict[str, List[Dict]]:
    """Run BM25, Dense, and Hybrid retrieval for all samples.

    Returns:
        Dict[method_name → list of per-sample result dicts]
    """
    # Build indices
    bm25 = BM25Retriever(k1=cfg.retrieval.get("bm25_k1", 1.5),
                          b=cfg.retrieval.get("bm25_b", 0.75))
    bm25.index(corpus_chunks)

    dense = DenseRetriever(
        model_name=cfg.retrieval.get("dense_model", "all-MiniLM-L6-v2"),
        query_instruction=cfg.retrieval.get("dense_query_instruction"),
        e5_max_seq_length=cfg.retrieval.get("e5_max_seq_length", 512),
        e5_batch_size=cfg.retrieval.get("e5_batch_size"),
        debug=cfg.retrieval.get("debug_dense", False),
    )
    dense.index(corpus_chunks)

    hybrid = HybridRetriever(bm25, dense, alpha=cfg.retrieval.get("hybrid_alpha", 0.5))

    retrievers = {"bm25": bm25, "dense": dense, "hybrid": hybrid}

    all_results: Dict[str, List[Dict]] = {}

    for method_name, retriever in retrievers.items():
        print(f"  Running {method_name}...")
        results: List[Dict] = []
        for s in samples:
            if method_name == "hybrid":
                retrieved = retriever.search(s["question"], top_k=top_k)
            else:
                retrieved = retriever.search(s["question"], top_k=top_k)

            retrieved_ids = [c.chunk_id for c, _ in retrieved]
            results.append({
                "question_id": s["id"],
                "question": s["question"],
                "gold_answer": s["answer"],
                "gold_evidence_ids": gold_map.get(s["id"], []),
                "retrieved_chunk_ids": retrieved_ids,
            })
        all_results[method_name] = results

    return all_results


def run_rerank_baselines(
    cfg: Config,
    samples: List[Dict],
    corpus_chunks: List[Chunk],
    gold_map: Dict[str, List[str]],
    retrieval_results: Dict[str, List[Dict]],
    graph,
    top_k: int = 50,
) -> Dict[str, List[Dict]]:
    """Run cross-encoder, PPR, and PPR+constraint reranking.

    Returns:
        Dict[method_name → list of per-sample result dicts]
    """
    chunk_by_id = {c.chunk_id: c for c in corpus_chunks}
    extractor = EntityExtractor()

    # Use hybrid retrieval results as the base for reranking
    hybrid_results = retrieval_results.get("hybrid", [])
    if not hybrid_results:
        print("  [WARN] No hybrid results to rerank; skipping.")
        return {}

    # Build retrieval score lookup per query
    bm25 = BM25Retriever(k1=cfg.retrieval.get("bm25_k1", 1.5),
                          b=cfg.retrieval.get("bm25_b", 0.75))
    bm25.index(corpus_chunks)

    rerank_results: Dict[str, List[Dict]] = {}

    # --- Cross-Encoder ---
    print("  Running cross-encoder reranker...")
    ce = CrossEncoderReranker(
        model_name=cfg.cross_encoder.get("model_name", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
        batch_size=cfg.cross_encoder.get("batch_size", 32),
    )
    ce_results: List[Dict] = []
    for i, hr in enumerate(hybrid_results):
        if i % 20 == 0:
            print(f"    {i}/{len(hybrid_results)}")
        # Get top-N candidates from hybrid
        candidates = []
        for cid in hr["retrieved_chunk_ids"][:cfg.cross_encoder.get("top_k_rerank", 100)]:
            c = chunk_by_id.get(cid)
            if c:
                candidates.append((c, 1.0))  # uniform prior

        reranked = ce.rerank(hr["question"], candidates, top_k=top_k)
        ce_results.append({
            "question_id": hr["question_id"],
            "question": hr["question"],
            "gold_answer": hr["gold_answer"],
            "gold_evidence_ids": hr["gold_evidence_ids"],
            "retrieved_chunk_ids": [c.chunk_id for c, _ in reranked],
        })
    rerank_results["hybrid+cross_encoder"] = ce_results

    # --- PPR ---
    print("  Running PPR reranker...")
    ppr_results: List[Dict] = []
    for i, hr in enumerate(hybrid_results):
        if i % 20 == 0:
            print(f"    {i}/{len(hybrid_results)}")
        candidate_ids = hr["retrieved_chunk_ids"][:top_k]
        q_metrics = extractor.extract_metrics(hr["question"])
        q_years = extractor.extract_years(hr["question"])

        ppr_scores = ppr_rerank(
            graph, corpus_chunks, candidate_ids,
            seed_chunk_ids=candidate_ids[:10],
            seed_metric_names=list(q_metrics),
            seed_year_values=list(q_years),
            alpha=cfg.rerank.get("ppr_alpha", 0.85),
        )
        ppr_results.append({
            "question_id": hr["question_id"],
            "question": hr["question"],
            "gold_answer": hr["gold_answer"],
            "gold_evidence_ids": hr["gold_evidence_ids"],
            "retrieved_chunk_ids": [cid for cid, _ in ppr_scores],
        })
    rerank_results["hybrid+ppr"] = ppr_results

    # --- PPR + Constraint ---
    print("  Running PPR + Constraint fusion...")
    constraint_scorer = ConstraintScorer(
        company_weight=cfg.constraint.get("company_match_weight", 1.0),
        year_weight=cfg.constraint.get("year_match_weight", 1.0),
        metric_weight=cfg.constraint.get("metric_match_weight", 0.8),
        filing_type_weight=cfg.constraint.get("filing_type_match_weight", 0.5),
    )
    fusion = FusionScorer(
        alpha=cfg.rerank.get("fusion_alpha", 0.3),
        beta=cfg.rerank.get("fusion_beta", 0.3),
        gamma=0.0,  # no GNN
        delta=cfg.rerank.get("fusion_delta", 0.1),
        constraint_scorer=constraint_scorer,
    )
    pprc_results: List[Dict] = []
    for i, (hr, pr) in enumerate(zip(hybrid_results, ppr_results)):
        candidate_ids = hr["retrieved_chunk_ids"][:top_k]
        chunks = [chunk_by_id[cid] for cid in candidate_ids if cid in chunk_by_id]

        # Build score dicts
        ret_scores = {}  # We use PPR scores
        # Fetch BM25 scores
        for j, cid in enumerate(candidate_ids):
            ret_scores[cid] = float(top_k - j) / top_k  # pseudo retrieval score

        ppr_scores = {}
        ppr_id_to_score: Dict[str, float] = {}
        for cid, s in ppr_scores if isinstance(ppr_scores, dict) else {}:
            if not isinstance(ppr_scores, dict):
                # Build from pr (PPR result)
                ppr_sorted = pr["retrieved_chunk_ids"]
                for rank, cid_p in enumerate(ppr_sorted):
                    ppr_id_to_score[cid_p] = 1.0 - rank / len(ppr_sorted) if ppr_sorted else 0.0
                ppr_scores = ppr_id_to_score
                break
        if not ppr_scores:
            ppr_sorted = pr["retrieved_chunk_ids"]
            for rank, cid_p in enumerate(ppr_sorted):
                ppr_scores[cid_p] = 1.0 - rank / max(len(ppr_sorted), 1)

        fused = fusion.fuse(hr["question"], chunks, ret_scores, graph_scores=ppr_scores)
        pprc_results.append({
            "question_id": hr["question_id"],
            "question": hr["question"],
            "gold_answer": hr["gold_answer"],
            "gold_evidence_ids": hr["gold_evidence_ids"],
            "retrieved_chunk_ids": [c.chunk_id for c, _ in fused],
        })
    rerank_results["hybrid+ppr+constraint"] = pprc_results

    return rerank_results


# ═════════════════════════════════════════════════════════════════════════════
# Main runner
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="FEG-RAG batch experiment runner")
    parser.add_argument("--config", required=True, help="Path to experiment YAML config")
    parser.add_argument("--methods", nargs="+",
                        default=["bm25", "dense", "hybrid", "cross_encoder", "ppr", "ppr_constraint"],
                        help="Methods to evaluate")
    parser.add_argument("--num_samples", type=int, default=0,
                        help="Limit samples (0=all)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only print what would be run")
    parser.add_argument("--no-cache", action="store_true",
                        help="Skip cache, recompute everything")
    parser.add_argument("--output", type=str, default="",
                        help="Override output filename")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)

    if args.dry_run:
        print(f"Config: {args.config}")
        print(f"Datasets: {cfg.datasets}")
        print(f"Methods: {args.methods}")
        print(f"Num samples: {args.num_samples or 'ALL'}")
        print("Dry run -- no experiments executed.")
        return

    # ---- Load data ----
    print("=" * 60)
    print("[1/5] Loading data...")
    all_samples: List[Dict] = []
    for ds_name in cfg.datasets:
        try:
            samples = load_dataset(ds_name, cfg.data_dir)
            all_samples.extend(samples)
            print(f"  {ds_name}: {len(samples)} samples")
        except FileNotFoundError as e:
            print(f"  [SKIP] {ds_name}: {e}")

    if args.num_samples > 0:
        all_samples = all_samples[:args.num_samples]
    print(f"  Total: {len(all_samples)} QA samples")

    # ---- Build chunks & gold map ----
    print("[2/5] Chunking...")
    corpus_chunks: List[Chunk] = []
    gold_map: Dict[str, List[str]] = {}

    for s in all_samples:
        gold_ids = []
        for text in s["evidence_texts"]:
            for c in chunk_text(text, cfg.chunk_size, cfg.chunk_overlap, doc_id=s["id"]):
                corpus_chunks.append(c)
                gold_ids.append(c.chunk_id)
        gold_map[s["id"]] = gold_ids

    print(f"  {len(corpus_chunks)} total chunks")

    # ---- Retrieval baselines ----
    print("[3/5] Running retrieval baselines...")
    ret_results = run_retrieval_baselines(cfg, all_samples, corpus_chunks, gold_map,
                                           top_k=cfg.retrieval.get("top_k", 50))

    # ---- Build graph ----
    print("[4/5] Building financial evidence graph...")
    entity_map = extract_entities(corpus_chunks)
    use_weights = cfg.graph.get("use_edge_weights", False)
    edge_weights = cfg.graph.get("edge_weights", None) if use_weights else None
    graph = build_financial_evidence_graph(
        corpus_chunks,
        entity_map=entity_map,
        add_semantic_edges=False,
        use_edge_weights=use_weights,
        edge_weight_map=edge_weights,
    )
    print(f"  Graph: {graph.num_nodes} nodes, {graph.num_edges} edges")

    # ---- Reranking baselines ----
    print("[5/5] Running reranking baselines...")
    rerank_results = run_rerank_baselines(
        cfg, all_samples, corpus_chunks, gold_map, ret_results, graph,
        top_k=cfg.retrieval.get("top_k", 50),
    )

    # ---- Evaluate all ----
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    k_vals = cfg.evaluation.get("recall_k_values", [1, 3, 5, 10, 20])
    all_eval: Dict[str, Dict] = {}

    # Merge retrieval + rerank results
    all_method_results = {**ret_results, **rerank_results}

    for method_name, results in all_method_results.items():
        er = compute_all_metrics(method_name, results, k_values=k_vals)
        summary = {
            "method": method_name,
            "num_samples": er.num_samples,
            "evidence_recall": er.evidence_recall,
            "evidence_precision": er.evidence_precision,
            "mrr": round(er.mrr, 4),
            "ndcg": {str(k): round(v, 4) for k, v in er.ndcg.items()},
            "answer_accuracy": round(er.answer_accuracy, 4),
        }
        all_eval[method_name] = summary
        print(f"\n  [{method_name}]")
        print(f"    MRR: {summary['mrr']}")
        for k in k_vals:
            print(f"    R@{k}: {summary['evidence_recall'].get(k, 0):.4f}")

    # ---- Save ----
    output_dir = cfg.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_name = args.output or f"experiment_results_{ts}.json"
    output_path = output_dir / output_name

    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump({
            "config": args.config,
            "timestamp": ts,
            "num_samples": len(all_samples),
            "datasets": cfg.datasets,
            "results": all_eval,
        }, fh, indent=2, default=str)

    print(f"\nResults saved to {output_path}")
    print("Done.")


if __name__ == "__main__":
    main()
