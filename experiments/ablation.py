"""Systematic ablation experiment runner for FEG-RAG.

Implements the five ablation studies from the experiment design document:
  1. Graph structure ablation (§10.1)
  2. Edge type ablation (§10.2)
  3. Reranker ablation (§10.3)
  4. Verifier ablation (§10.4)
  5. Hard negative ablation (§10.5)

Usage:
    python experiments/ablation.py --ablation graph_structure
    python experiments/ablation.py --ablation edge_types
    python experiments/ablation.py --ablation reranker
    python experiments/ablation.py --ablation verifier
    python experiments/ablation.py --ablation hard_negatives
    python experiments/ablation.py --ablation all
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

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
from feg_rag.rerank.ppr import ppr_rerank
from feg_rag.rerank.fusion import ConstraintScorer, FusionScorer


# ═════════════════════════════════════════════════════════════════════════════
# 1. Graph Structure Ablation (§10.1)
# ═════════════════════════════════════════════════════════════════════════════

def ablation_graph_structure(
    cfg: Config,
    samples: List[Dict],
    corpus_chunks: List[Chunk],
    gold_map: Dict[str, List[str]],
    hybrid: HybridRetriever,
    entity_map: Dict,
    chunk_embeddings: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, Dict]:
    """Compare different graph structures for reranking.

    Variants:
        no_graph         — Hybrid retrieval only, no graph
        semantic_only    — Only semantic-similarity edges
        financial_only   — Financial structure edges only (no semantic)
        financial+semantic — Both financial and semantic edges
        full_weighted    — Full graph with financial constraint-aware edge weights
    """
    print("\n" + "=" * 60)
    print("ABLATION: Graph Structure")
    print("=" * 60)

    results: Dict[str, Dict] = {}
    extractor = EntityExtractor()
    top_k = cfg.retrieval.get("top_k", 50)
    k_vals = cfg.evaluation.get("recall_k_values", [1, 3, 5, 10, 20])

    # --- no_graph: Hybrid only ---
    print("\n[1/5] No Graph (Hybrid only)...")
    hybrid_results = []
    for s in samples:
        retrieved = hybrid.search(s["question"], top_k=top_k)
        hybrid_results.append({
            "question_id": s["id"],
            "question": s["question"],
            "gold_answer": s["answer"],
            "gold_evidence_ids": gold_map.get(s["id"], []),
            "retrieved_chunk_ids": [c.chunk_id for c, _ in retrieved],
        })
    er = compute_all_metrics("no_graph", hybrid_results, k_values=k_vals)
    results["no_graph"] = _metrics_to_dict(er)

    # --- semantic_only: graph with ONLY semantic edges ---
    print("[2/5] Semantic Graph Only...")
    g_sem = build_financial_evidence_graph(
        corpus_chunks,
        entity_map=entity_map,
        chunk_embeddings=chunk_embeddings,
        add_semantic_edges=True,
        add_company_nodes=False,
        add_filing_nodes=False,
        add_section_nodes=False,
    )
    sem_results = _run_ppr_eval(corpus_chunks, samples, gold_map, hybrid, g_sem,
                                 extractor, top_k, k_vals)
    results["semantic_only"] = _metrics_to_dict(sem_results)

    # --- financial_only: financial structure edges, no semantic ---
    print("[3/5] Financial Structure Graph Only...")
    g_fin = build_financial_evidence_graph(
        corpus_chunks,
        entity_map=entity_map,
        add_semantic_edges=False,
        add_company_nodes=True,
        add_filing_nodes=True,
        add_section_nodes=True,
        use_edge_weights=False,
    )
    fin_res = _run_ppr_eval(corpus_chunks, samples, gold_map, hybrid, g_fin,
                             extractor, top_k, k_vals)
    results["financial_only"] = _metrics_to_dict(fin_res)

    # --- financial+semantic: both types of edges ---
    print("[4/5] Financial + Semantic Graph...")
    g_full = build_financial_evidence_graph(
        corpus_chunks,
        entity_map=entity_map,
        chunk_embeddings=chunk_embeddings,
        add_semantic_edges=(chunk_embeddings is not None),
        add_company_nodes=True,
        add_filing_nodes=True,
        add_section_nodes=True,
        use_edge_weights=False,
    )
    full_res = _run_ppr_eval(corpus_chunks, samples, gold_map, hybrid, g_full,
                              extractor, top_k, k_vals)
    results["financial+semantic"] = _metrics_to_dict(full_res)

    # --- full_weighted: full graph with weights ---
    print("[5/5] Full Weighted Graph...")
    g_wt = build_financial_evidence_graph(
        corpus_chunks,
        entity_map=entity_map,
        chunk_embeddings=chunk_embeddings,
        add_semantic_edges=(chunk_embeddings is not None),
        add_company_nodes=True,
        add_filing_nodes=True,
        add_section_nodes=True,
        use_edge_weights=True,
        edge_weight_map=cfg.graph.get("edge_weights", None),
    )
    wt_res = _run_ppr_eval(corpus_chunks, samples, gold_map, hybrid, g_wt,
                            extractor, top_k, k_vals)
    results["full_weighted"] = _metrics_to_dict(wt_res)

    return results


# ═════════════════════════════════════════════════════════════════════════════
# 2. Edge Type Ablation (§10.2)
# ═════════════════════════════════════════════════════════════════════════════

EDGE_GROUPS = {
    "company": ["same-company", "company-has-filing"],
    "filing": ["chunk-belongs-to-filing", "filing-has-section", "same-filing-year"],
    "section": ["section-has-chunk", "filing-has-section"],
    "metric": ["chunk-mentions-metric", "same-metric"],
    "year": ["chunk-mentions-year", "same-year"],
    "semantic": ["semantic-similar"],
}


def ablation_edge_types(
    cfg: Config,
    samples: List[Dict],
    corpus_chunks: List[Chunk],
    gold_map: Dict[str, List[str]],
    hybrid: HybridRetriever,
    entity_map: Dict,
    chunk_embeddings: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, Dict]:
    """Test removing specific edge types from the full graph.

    Variants:
        full_graph   — Full weighted graph (all edges)
        wo_company   — Remove company edges
        wo_filing    — Remove filing edges
        wo_section   — Remove section edges
        wo_metric    — Remove metric edges
        wo_year      — Remove year edges
        wo_semantic  — Remove semantic edges
    """
    print("\n" + "=" * 60)
    print("ABLATION: Edge Types")
    print("=" * 60)

    results: Dict[str, Dict] = {}
    extractor = EntityExtractor()
    top_k = cfg.retrieval.get("top_k", 50)
    k_vals = cfg.evaluation.get("recall_k_values", [1, 3, 5, 10, 20])

    # base graph settings
    base_kwargs = dict(
        entity_map=entity_map,
        chunk_embeddings=chunk_embeddings,
        add_semantic_edges=(chunk_embeddings is not None),
        add_company_nodes=True,
        add_filing_nodes=True,
        add_section_nodes=True,
        use_edge_weights=True,
        edge_weight_map=cfg.graph.get("edge_weights", None),
    )

    # --- full_graph ---
    print("\n[1/7] Full Graph...")
    g_full = build_financial_evidence_graph(corpus_chunks, **base_kwargs)
    full_res = _run_ppr_eval(corpus_chunks, samples, gold_map, hybrid, g_full,
                              extractor, top_k, k_vals)
    results["full_graph"] = _metrics_to_dict(full_res)

    # --- wo_* variants ---
    variants = {
        "wo_company": "company",
        "wo_filing": "filing",
        "wo_section": "section",
        "wo_metric": "metric",
        "wo_year": "year",
        "wo_semantic": "semantic",
    }

    for i, (variant_key, group_name) in enumerate(variants.items(), start=2):
        print(f"[{i}/7] {variant_key} (remove {group_name} edges)...")
        # Build graph without these edges
        removed = set(EDGE_GROUPS.get(group_name, []))
        filtered_kwargs = dict(base_kwargs)
        if group_name == "semantic":
            filtered_kwargs["add_semantic_edges"] = False
            g = build_financial_evidence_graph(corpus_chunks, **filtered_kwargs)
        else:
            g = build_financial_evidence_graph(corpus_chunks, **filtered_kwargs)
            # Remove edges of specified types
            edges_to_remove = [
                (u, v, k) for u, v, k, etype in g.graph.edges(keys=True, data="edge_type")
                if etype in removed
            ]
            g.graph.remove_edges_from(edges_to_remove)

        res = _run_ppr_eval(corpus_chunks, samples, gold_map, hybrid, g,
                             extractor, top_k, k_vals)
        results[variant_key] = _metrics_to_dict(res)

    return results


# ═════════════════════════════════════════════════════════════════════════════
# 3. Reranker Ablation (§10.3)
# ═════════════════════════════════════════════════════════════════════════════

def ablation_reranker(
    cfg: Config,
    samples: List[Dict],
    corpus_chunks: List[Chunk],
    gold_map: Dict[str, List[str]],
    hybrid: HybridRetriever,
    entity_map: Dict,
    chunk_embeddings: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, Dict]:
    """Compare different reranking methods.

    Variants:
        hybrid          — No reranking
        hybrid+ppr      — PPR graph reranking
        hybrid+ppr+cons — PPR + Constraint scoring
    """
    print("\n" + "=" * 60)
    print("ABLATION: Reranker")
    print("=" * 60)

    results: Dict[str, Dict] = {}
    extractor = EntityExtractor()
    top_k = cfg.retrieval.get("top_k", 50)
    k_vals = cfg.evaluation.get("recall_k_values", [1, 3, 5, 10, 20])

    # Build full weighted graph
    g = build_financial_evidence_graph(
        corpus_chunks, entity_map=entity_map,
        chunk_embeddings=chunk_embeddings,
        add_semantic_edges=(chunk_embeddings is not None),
        add_company_nodes=True, add_filing_nodes=True, add_section_nodes=True,
        use_edge_weights=True,
        edge_weight_map=cfg.graph.get("edge_weights", None),
    )

    # --- hybrid (no reranking) ---
    print("\n[1/3] Hybrid (no reranking)...")
    hybrid_results = []
    for s in samples:
        retrieved = hybrid.search(s["question"], top_k=top_k)
        hybrid_results.append({
            "question_id": s["id"],
            "question": s["question"],
            "gold_answer": s["answer"],
            "gold_evidence_ids": gold_map.get(s["id"], []),
            "retrieved_chunk_ids": [c.chunk_id for c, _ in retrieved],
        })
    er = compute_all_metrics("hybrid", hybrid_results, k_values=k_vals)
    results["hybrid"] = _metrics_to_dict(er)

    # --- hybrid + PPR ---
    print("[2/3] Hybrid + PPR...")
    ppr_res = _run_ppr_eval(corpus_chunks, samples, gold_map, hybrid, g,
                             extractor, top_k, k_vals)
    results["hybrid+ppr"] = _metrics_to_dict(ppr_res)

    # --- hybrid + PPR + Constraint ---
    print("[3/3] Hybrid + PPR + Constraint...")
    constraint_scorer = ConstraintScorer(
        company_weight=cfg.constraint.get("company_match_weight", 1.0),
        year_weight=cfg.constraint.get("year_match_weight", 1.0),
        metric_weight=cfg.constraint.get("metric_match_weight", 0.8),
        filing_type_weight=cfg.constraint.get("filing_type_match_weight", 0.5),
    )
    fusion = FusionScorer(
        alpha=cfg.rerank.get("fusion_alpha", 0.3),
        beta=cfg.rerank.get("fusion_beta", 0.3),
        gamma=0.0,
        delta=cfg.rerank.get("fusion_delta", 0.1),
        constraint_scorer=constraint_scorer,
    )
    chunk_by_id = {c.chunk_id: c for c in corpus_chunks}

    pprc_results = []
    for s in samples:
        retrieved = hybrid.search(s["question"], top_k=top_k)
        candidate_ids = [c.chunk_id for c, _ in retrieved]
        q_m = extractor.extract_metrics(s["question"])
        q_y = extractor.extract_years(s["question"])

        ppr_scores = ppr_rerank(
            g, corpus_chunks, candidate_ids,
            seed_chunk_ids=candidate_ids[:10],
            seed_metric_names=list(q_m),
            seed_year_values=list(q_y),
            alpha=cfg.rerank.get("ppr_alpha", 0.85),
        )
        ppr_map = {cid: s for cid, s in ppr_scores}

        chunks = [chunk_by_id[cid] for cid in candidate_ids if cid in chunk_by_id]
        ret_map = {}
        for j, cid in enumerate(candidate_ids):
            ret_map[cid] = 1.0 - j / max(len(candidate_ids), 1)

        fused = fusion.fuse(s["question"], chunks, ret_map, graph_scores=ppr_map)
        pprc_results.append({
            "question_id": s["id"],
            "question": s["question"],
            "gold_answer": s["answer"],
            "gold_evidence_ids": gold_map.get(s["id"], []),
            "retrieved_chunk_ids": [c.chunk_id for c, _ in fused],
        })
    er_pprc = compute_all_metrics("hybrid+ppr+constraint", pprc_results, k_values=k_vals)
    results["hybrid+ppr+constraint"] = _metrics_to_dict(er_pprc)

    return results


# ═════════════════════════════════════════════════════════════════════════════
# 4. Verifier Ablation (§10.4)
# ═════════════════════════════════════════════════════════════════════════════

def ablation_verifier(
    cfg: Config,
    samples: List[Dict],
    corpus_chunks: List[Chunk],
    gold_map: Dict[str, List[str]],
    hybrid: HybridRetriever,
    entity_map: Dict,
    chunk_embeddings: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, Dict]:
    """Compare FEG-RAG with and without the numerical verifier.

    Note: This requires LLM generation to test properly. Falls back to recording
    verifier results on existing generation data if available.
    """
    print("\n" + "=" * 60)
    print("ABLATION: Verifier (with / without)")
    print("=" * 60)
    print("  [NOTE] This ablation requires LLM generation results.")
    print("  Run with generation enabled to populate verifier metrics.")
    # Stub: record placeholder — verifier metrics come from generation step
    return {
        "feg_rag": {"note": "Verifier ablation requires generation (run full pipeline)"},
        "feg_rag+verifier": {"note": "Verifier ablation requires generation (run full pipeline)"},
    }


# ═════════════════════════════════════════════════════════════════════════════
# 5. Hard Negative Ablation (§10.5)
# ═════════════════════════════════════════════════════════════════════════════

def ablation_hard_negatives(
    cfg: Config,
    samples: List[Dict],
    corpus_chunks: List[Chunk],
    gold_map: Dict[str, List[str]],
    hybrid: HybridRetriever,
    entity_map: Dict,
    chunk_embeddings: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, Dict]:
    """Compare GNN training with different negative sampling strategies.

    Variants:
        random     — Random negatives from corpus
        top_retrieved — Top retrieval results that are not gold
        finance_hard — Finance-specific hard negatives (same-company-wrong-year, etc.)

    Note: This requires GNN training. Records the configuration and neg counts.
    """
    print("\n" + "=" * 60)
    print("ABLATION: Hard Negatives")
    print("=" * 60)
    print("  [NOTE] Hard negative ablation requires GNN training.")
    print("  Generating negative samples for each strategy...")

    chunk_by_id = {c.chunk_id: c for c in corpus_chunks}

    for s in samples[:5]:  # Demo with 5 samples
        gold_ids = gold_map.get(s["id"], [])
        gold_chunks = [chunk_by_id[gid] for gid in gold_ids if gid in chunk_by_id]

        # Random negatives
        all_ids = [c.chunk_id for c in corpus_chunks if c.chunk_id not in set(gold_ids)]
        rng = np.random.default_rng(42)
        random_neg = rng.choice(all_ids, min(10, len(all_ids)), replace=False)

        # Finance-specific hard negatives
        hard_neg = generate_hard_negatives(gold_chunks, corpus_chunks, num_negatives=10)

        print(f"  Sample {s['id']}: {len(gold_ids)} gold, "
              f"{len(random_neg)} random neg, {len(hard_neg)} hard neg")

    return {
        "random_negatives": {"num_per_sample": 10, "strategy": "random"},
        "top_retrieved_negatives": {"num_per_sample": 10, "strategy": "top_retrieved"},
        "finance_hard_negatives": {"num_per_sample": 10, "strategy": "finance_specific"},
    }


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════

def _run_ppr_eval(
    corpus_chunks: List[Chunk],
    samples: List[Dict],
    gold_map: Dict[str, List[str]],
    hybrid: HybridRetriever,
    graph,
    extractor: EntityExtractor,
    top_k: int,
    k_vals: List[int],
):
    """Run PPR reranking and evaluate."""
    results = []
    for s in samples:
        retrieved = hybrid.search(s["question"], top_k=top_k)
        candidate_ids = [c.chunk_id for c, _ in retrieved]
        q_m = extractor.extract_metrics(s["question"])
        q_y = extractor.extract_years(s["question"])

        ppr_scores = ppr_rerank(
            graph, corpus_chunks, candidate_ids,
            seed_chunk_ids=candidate_ids[:10],
            seed_metric_names=list(q_m),
            seed_year_values=list(q_y),
        )
        results.append({
            "question_id": s["id"],
            "question": s["question"],
            "gold_answer": s["answer"],
            "gold_evidence_ids": gold_map.get(s["id"], []),
            "retrieved_chunk_ids": [cid for cid, _ in ppr_scores],
        })
    return compute_all_metrics("ppr", results, k_values=k_vals)


def _metrics_to_dict(er) -> Dict:
    """Convert EvalResult to a plain dict."""
    return {
        "num_samples": er.num_samples,
        "evidence_recall": er.evidence_recall,
        "evidence_precision": er.evidence_precision,
        "mrr": round(er.mrr, 4),
        "ndcg": {str(k): round(v, 4) for k, v in er.ndcg.items()},
        "answer_accuracy": round(er.answer_accuracy, 4),
    }


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════

ABLATIONS = {
    "graph_structure": ablation_graph_structure,
    "edge_types": ablation_edge_types,
    "reranker": ablation_reranker,
    "verifier": ablation_verifier,
    "hard_negatives": ablation_hard_negatives,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="FEG-RAG ablation experiments")
    parser.add_argument("--ablation", required=True,
                        choices=list(ABLATIONS) + ["all"],
                        help="Which ablation study to run")
    parser.add_argument("--config", default="configs/default.yaml",
                        help="Path to base config")
    parser.add_argument("--num_samples", type=int, default=30,
                        help="Limit samples (0=all)")
    parser.add_argument("--output-dir", default="outputs",
                        help="Output directory")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    cfg.ensure_dirs()

    # ---- Load data ----
    print("Loading data...")
    all_samples: List[Dict] = []
    for ds_name in cfg.datasets:
        try:
            samples = load_dataset(ds_name, cfg.data_dir)
            all_samples.extend(samples)
        except FileNotFoundError as e:
            print(f"  [SKIP] {ds_name}: {e}")

    if args.num_samples > 0:
        all_samples = all_samples[:args.num_samples]
    print(f"  {len(all_samples)} samples")

    # ---- Build chunks ----
    corpus_chunks: List[Chunk] = []
    gold_map: Dict[str, List[str]] = {}
    for s in all_samples:
        gold_ids = []
        for text in s["evidence_texts"]:
            for c in chunk_text(text, cfg.chunk_size, cfg.chunk_overlap, doc_id=s["id"]):
                corpus_chunks.append(c)
                gold_ids.append(c.chunk_id)
        gold_map[s["id"]] = gold_ids

    print(f"  {len(corpus_chunks)} chunks")

    # ---- Build retrieval index ----
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

    # ---- Build entity map ----
    entity_map = extract_entities(corpus_chunks)

    # ---- Run ablation(s) ----
    ablations_to_run = (
        list(ABLATIONS) if args.ablation == "all" else [args.ablation]
    )

    all_results: Dict[str, Dict] = {}
    for ab_name in ablations_to_run:
        fn = ABLATIONS[ab_name]
        ab_results = fn(cfg, all_samples, corpus_chunks, gold_map, hybrid, entity_map)
        all_results[ab_name] = ab_results

    # ---- Save ----
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"ablation_{args.ablation}_{ts}.json"

    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump({
            "ablation": args.ablation,
            "timestamp": ts,
            "num_samples": len(all_samples),
            "results": all_results,
        }, fh, indent=2, default=str)

    print(f"\nAblation results saved to {output_path}")
    print("Done.")


if __name__ == "__main__":
    main()
