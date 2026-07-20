"""Legacy Table 1: Evidence Retrieval & Reranking Performance.

Evaluates all retrieval and reranking methods WITHOUT LLM generation.
Methods: BM25, Dense, Hybrid, Hybrid+CrossEncoder, Hybrid+PPR.

Legacy warning: this script builds a candidate pool from annotated FinDER gold
evidence snippets plus distractors. It is useful for diagnostics, but paper
benchmark results should use ``table1_non_llm_reranking_comparison.py``, which
builds the corpus from source filings and aligns gold evidence back to those
source-document chunks.

Usage:
    python experiments/table1_retrieval_rerank.py --num_samples 500
    python experiments/table1_retrieval_rerank.py --num_samples 0  # all samples
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from feg_rag.config import Config
from feg_rag.data.chunker import Chunk, chunk_text, chunk_report
from feg_rag.data.loader import load_dataset
from feg_rag.retrieval.bm25 import BM25Retriever
from feg_rag.retrieval.dense import DenseRetriever
from feg_rag.retrieval.hybrid import HybridRetriever
from feg_rag.retrieval.cross_encoder import CrossEncoderReranker
from feg_rag.graph.builder import build_financial_evidence_graph
from feg_rag.graph.entities import extract_entities, EntityExtractor
from feg_rag.rerank.ppr import ppr_rerank
from feg_rag.rerank.fusion import FusionScorer


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def build_corpus(
    samples: List[Dict],
    cfg: Config,
    max_distractor_files: int = 50,
) -> Tuple[List[Chunk], Dict[str, List[str]]]:
    """Build legacy gold-snippet candidate pool plus 10-K distractors."""
    corpus_chunks: List[Chunk] = []
    gold_map: Dict[str, List[str]] = {}

    # 1. Gold evidence chunks
    print("  Building chunks from FinDER evidence...")
    for s in samples:
        gold_ids = []
        for text in s["evidence_texts"]:
            for c in chunk_text(
                text,
                chunk_size=cfg.chunk_size,
                chunk_overlap=cfg.chunk_overlap,
                doc_id=s["id"],
            ):
                corpus_chunks.append(c)
                gold_ids.append(c.chunk_id)
        gold_map[s["id"]] = gold_ids

    # 2. 10-K distractor chunks
    edgar_dir = cfg.edgar_dir
    txt_files = []
    if edgar_dir.exists():
        # Look for .txt or .html files
        txt_files = list(edgar_dir.rglob("*.txt"))
        if not txt_files:
            txt_files = list(edgar_dir.rglob("*.html"))
        if not txt_files:
            # Try nested directory
            for subdir in edgar_dir.iterdir():
                if subdir.is_dir() and subdir.name != "__MACOSX":
                    txt_files.extend(subdir.rglob("*.txt"))
                    txt_files.extend(subdir.rglob("*.html"))

    print(f"  Found {len(txt_files)} distractor files, using up to {max_distractor_files}")
    for tf in txt_files[:max_distractor_files]:
        try:
            corpus_chunks.extend(
                chunk_report(tf, cfg.chunk_size, cfg.chunk_overlap)
            )
        except Exception as e:
            pass  # skip unreadable files

    print(f"  Corpus: {len(corpus_chunks)} total chunks "
          f"({len(gold_map)} QA pairs with gold evidence)")
    return corpus_chunks, gold_map


def compute_retrieval_metrics(
    method_name: str,
    samples: List[Dict],
    retrieved_ids_list: List[List[str]],
    gold_map: Dict[str, List[str]],
    k_values: List[int],
) -> Dict:
    """Compute Recall@K, MRR, nDCG from retrieval results."""
    n = len(samples)
    if n == 0:
        return {}

    # Evidence Recall@K
    recall = {}
    precision = {}
    for k in k_values:
        r_vals, p_vals = [], []
        for s, ret_ids in zip(samples, retrieved_ids_list):
            gold = set(gold_map.get(s["id"], []))
            top_k = ret_ids[:k]
            if gold:
                r_vals.append(len(gold & set(top_k)) / len(gold))
            else:
                r_vals.append(0.0)
            if top_k:
                p_vals.append(len(gold & set(top_k)) / len(top_k))
            else:
                p_vals.append(0.0)
        recall[k] = float(np.mean(r_vals))
        precision[k] = float(np.mean(p_vals))

    # MRR
    mrr_vals = []
    for s, ret_ids in zip(samples, retrieved_ids_list):
        gold = set(gold_map.get(s["id"], []))
        for rank, cid in enumerate(ret_ids, start=1):
            if cid in gold:
                mrr_vals.append(1.0 / rank)
                break
        else:
            mrr_vals.append(0.0)
    mrr = float(np.mean(mrr_vals))

    # nDCG@K
    ndcg = {}
    for k in k_values:
        ndcg_vals = []
        for s, ret_ids in zip(samples, retrieved_ids_list):
            gold = set(gold_map.get(s["id"], []))
            top_k = ret_ids[:k]
            dcg = 0.0
            for i, cid in enumerate(top_k):
                rel = 1.0 if cid in gold else 0.0
                dcg += rel / np.log2(i + 2)
            ideal_rels = min(len(gold), k)
            idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_rels))
            ndcg_vals.append(dcg / idcg if idcg > 0 else 0.0)
        ndcg[k] = float(np.mean(ndcg_vals))

    return {
        "method": method_name,
        "num_samples": n,
        "evidence_recall": recall,
        "evidence_precision": precision,
        "mrr": mrr,
        "ndcg": ndcg,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Table 1: Retrieval & Reranking")
    parser.add_argument("--config", default="configs/table1_retrieval.yaml")
    parser.add_argument("--num_samples", type=int, default=200,
                        help="Number of QA samples (0=all)")
    parser.add_argument("--max_distractor_files", type=int, default=50,
                        help="Max 10-K files to chunk as distractors")
    parser.add_argument("--skip_dense", action="store_true",
                        help="Skip dense retrieval (faster)")
    parser.add_argument("--skip_cross_encoder", action="store_true",
                        help="Skip cross-encoder (faster)")
    parser.add_argument("--skip_ppr", action="store_true",
                        help="Skip PPR reranking")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    cfg.ensure_dirs()

    k_values = cfg.evaluation["recall_k_values"]
    top_k = cfg.retrieval["top_k"]
    print(f"Config: top_k={top_k}, k_values={k_values}")
    print(f"Samples: {args.num_samples} (0=all)")

    # ═══════════════════════════════════════════════════════════════
    # Step 1: Load data
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("[1/5] Loading FinDER data...")
    print("=" * 60)
    samples = load_dataset("finder", cfg.data_dir)
    if args.num_samples > 0:
        samples = samples[: args.num_samples]
    print(f"  Loaded {len(samples)} QA samples")

    # ═══════════════════════════════════════════════════════════════
    # Step 2: Build corpus
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("[2/5] Building chunk corpus...")
    print("=" * 60)
    corpus_chunks, gold_map = build_corpus(
        samples, cfg, max_distractor_files=args.max_distractor_files
    )
    n_gold_chunks = sum(len(v) for v in gold_map.values())

    # ═══════════════════════════════════════════════════════════════
    # Step 3: Build retrieval indices
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("[3/5] Building retrieval indices...")
    print("=" * 60)

    t0 = time.time()
    print("  BM25 index...")
    bm25 = BM25Retriever(k1=cfg.retrieval["bm25_k1"], b=cfg.retrieval["bm25_b"])
    bm25.index(corpus_chunks)
    print(f"    {time.time() - t0:.1f}s")

    t0 = time.time()
    if not args.skip_dense:
        print("  Dense index (downloading model if needed)...")
        dense = DenseRetriever(
            model_name=cfg.retrieval["dense_model"],
            query_instruction=cfg.retrieval.get("dense_query_instruction"),
            e5_max_seq_length=cfg.retrieval.get("e5_max_seq_length", 512),
            e5_batch_size=cfg.retrieval.get("e5_batch_size"),
            debug=cfg.retrieval.get("debug_dense", False),
        )
        dense.index(corpus_chunks)
        print(f"    {time.time() - t0:.1f}s")
    else:
        dense = None

    hybrid = HybridRetriever(bm25, dense, alpha=cfg.retrieval["hybrid_alpha"]) if dense else None

    # Cross-encoder
    cross_encoder = None
    if not args.skip_cross_encoder:
        try:
            print("  Loading cross-encoder...")
            t0 = time.time()
            cross_encoder = CrossEncoderReranker(
                model_name=cfg.cross_encoder["model_name"],
                batch_size=cfg.cross_encoder["batch_size"],
            )
            print(f"    {time.time() - t0:.1f}s")
        except Exception as e:
            print(f"  [SKIP] Cross-encoder: {e}")

    # ═══════════════════════════════════════════════════════════════
    # Step 4: Build graph
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("[4/5] Building Financial Evidence Graph...")
    print("=" * 60)

    graph = None
    entity_map = None
    if not args.skip_ppr:
        t0 = time.time()
        print("  Extracting entities...")
        entity_map = extract_entities(corpus_chunks)
        metric_count = len({m for e in entity_map.values() for m in e.metrics})
        year_count = len({y for e in entity_map.values() for y in e.years})
        print(f"  Unique metrics: {metric_count}, Unique years: {year_count}")

        print("  Building graph...")
        graph = build_financial_evidence_graph(
            corpus_chunks,
            entity_map=entity_map,
            add_semantic_edges=False,
            add_same_entity_edges=True,
            max_same_entity_edges=10,
            use_edge_weights=True,
        )
        print(f"  Graph: {graph.num_nodes} nodes, {graph.num_edges} edges")
        edge_counts = graph.edge_type_counts()
        for etype, count in sorted(edge_counts.items()):
            print(f"    {etype}: {count}")
        print(f"    {time.time() - t0:.1f}s")

    # ═══════════════════════════════════════════════════════════════
    # Step 5: Run retrieval + reranking on all queries
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print(f"[5/5] Running retrieval on {len(samples)} queries...")
    print("=" * 60)

    results: Dict[str, Dict] = {}
    extractor = EntityExtractor()

    # --- BM25 ---
    print("\n  --- BM25 ---")
    t0 = time.time()
    bm25_ids = []
    for i, s in enumerate(samples):
        if i % 50 == 0:
            print(f"    {i}/{len(samples)} ({time.time() - t0:.1f}s)")
        ret = bm25.search(s["question"], top_k=top_k)
        bm25_ids.append([c.chunk_id for c, _ in ret])
    results["BM25"] = compute_retrieval_metrics("BM25", samples, bm25_ids, gold_map, k_values)
    print(f"    Done in {time.time() - t0:.1f}s")

    # --- Dense ---
    if dense is not None:
        print("\n  --- Dense ---")
        t0 = time.time()
        dense_ids = []
        for i, s in enumerate(samples):
            if i % 50 == 0:
                print(f"    {i}/{len(samples)} ({time.time() - t0:.1f}s)")
            ret = dense.search(s["question"], top_k=top_k)
            dense_ids.append([c.chunk_id for c, _ in ret])
        results["Dense"] = compute_retrieval_metrics("Dense", samples, dense_ids, gold_map, k_values)
        print(f"    Done in {time.time() - t0:.1f}s")

    # --- Hybrid ---
    if hybrid is not None:
        print("\n  --- Hybrid ---")
        t0 = time.time()
        hybrid_ids = []
        hybrid_scores = []  # store (chunk_id, score) for Cross-Encoder and PPR
        for i, s in enumerate(samples):
            if i % 50 == 0:
                print(f"    {i}/{len(samples)} ({time.time() - t0:.1f}s)")
            ret = hybrid.search(s["question"], top_k=top_k)
            hybrid_ids.append([c.chunk_id for c, _ in ret])
            hybrid_scores.append(ret)  # keep Chunk objects for reranking
        results["Hybrid"] = compute_retrieval_metrics("Hybrid", samples, hybrid_ids, gold_map, k_values)
        print(f"    Done in {time.time() - t0:.1f}s")

    # --- Hybrid + Cross-Encoder ---
    if cross_encoder is not None and hybrid is not None:
        print("\n  --- Hybrid + Cross-Encoder ---")
        t0 = time.time()
        ce_ids = []
        for i, s in enumerate(samples):
            if i % 20 == 0:
                print(f"    {i}/{len(samples)} ({time.time() - t0:.1f}s)")
            reranked = cross_encoder.rerank(
                s["question"],
                hybrid_scores[i],
                top_k=min(top_k, cfg.cross_encoder["top_k_rerank"]),
            )
            ce_ids.append([c.chunk_id for c, _ in reranked])
        results["Hybrid+CE"] = compute_retrieval_metrics(
            "Hybrid+CrossEncoder", samples, ce_ids, gold_map, k_values
        )
        print(f"    Done in {time.time() - t0:.1f}s")

    # --- Hybrid + PPR ---
    if graph is not None and hybrid is not None and not args.skip_ppr:
        print("\n  --- Hybrid + PPR ---")
        t0 = time.time()
        ppr_ids = []
        for i, s in enumerate(samples):
            if i % 20 == 0:
                print(f"    {i}/{len(samples)} ({time.time() - t0:.1f}s)")
            candidate_ids = hybrid_ids[i]
            q_metrics = extractor.extract_metrics(s["question"])
            q_years = extractor.extract_years(s["question"])

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
            ppr_ids.append([cid for cid, _ in ppr_sorted])
        results["Hybrid+PPR"] = compute_retrieval_metrics(
            "Hybrid+PPR", samples, ppr_ids, gold_map, k_values
        )
        print(f"    Done in {time.time() - t0:.1f}s")

    # ═══════════════════════════════════════════════════════════════
    # Print Results Table
    # ═══════════════════════════════════════════════════════════════
    print("\n\n" + "=" * 80)
    print("TABLE 1: EVIDENCE RETRIEVAL & RERANKING PERFORMANCE")
    print("=" * 80)

    # Header
    header = f"{'Method':<25}"
    for k in k_values:
        header += f" {'R@'+str(k):>8}"
    header += f" {'MRR':>8} {'nDCG@10':>8}"
    print(header)
    print("-" * len(header))

    # Rows
    for method_name in ["BM25", "Dense", "Hybrid", "Hybrid+CE", "Hybrid+PPR"]:
        r = results.get(method_name)
        if r is None:
            continue
        row = f"{method_name:<25}"
        for k in k_values:
            row += f" {r['evidence_recall'].get(k, 0):>8.4f}"
        row += f" {r['mrr']:>8.4f}"
        ndcg10 = r['ndcg'].get(10, 0)
        row += f" {ndcg10:>8.4f}"
        print(row)

    # Delta rows
    print("-" * len(header))
    baseline = results.get("Hybrid")
    for method_name in ["Hybrid+CE", "Hybrid+PPR"]:
        r = results.get(method_name)
        if r is None or baseline is None:
            continue
        row = f"{'  Δ '+method_name:<25}"
        for k in k_values:
            delta = r['evidence_recall'].get(k, 0) - baseline['evidence_recall'].get(k, 0)
            row += f" {delta:>+8.4f}"
        delta_mrr = r['mrr'] - baseline['mrr']
        row += f" {delta_mrr:>+8.4f}"
        delta_ndcg = r['ndcg'].get(10, 0) - baseline['ndcg'].get(10, 0)
        row += f" {delta_ndcg:>+8.4f}"
        print(row)

    # ═══════════════════════════════════════════════════════════════
    # Save results
    # ═══════════════════════════════════════════════════════════════
    output_path = args.output or str(
        cfg.output_dir / f"table1_results_{_timestamp()}.json"
    )
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nResults saved to: {output_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
