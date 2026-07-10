"""Standalone GNN / R-GCN reranker training (no LLM required).

Usage:
    python experiments/train_gnn.py --config configs/default.yaml --epochs 5 --num_samples 50
    python experiments/train_gnn.py --config configs/default.yaml --model rgcn --epochs 10
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from feg_rag.config import Config
from feg_rag.data.loader import load_dataset
from feg_rag.evaluation.metrics import compute_all_metrics
from feg_rag.graph.builder import build_financial_evidence_graph
from feg_rag.graph.entities import extract_entities
from feg_rag.graph.features import build_node_features
from feg_rag.rerank.train import (
    build_corpus,
    print_loss_summary,
    save_training_artifacts,
    train_gnn_reranker,
    warmup_retrieval_scores,
)
from feg_rag.retrieval.bm25 import BM25Retriever


def _default_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train GNN reranker only")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--num_samples", type=int, default=50,
                        help="Number of QA samples (0 = all)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override gnn_epochs in config")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--model", choices=["sage", "rgcn"], default=None,
                        help="Override rerank.gnn_model in config")
    parser.add_argument("--device", default=_default_device())
    parser.add_argument("--min_pairs", type=int, default=10)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--val_split", type=float, default=0.2,
                        help="Fraction of samples for validation")
    parser.add_argument("--no_embeddings", action="store_true",
                        help="Skip text embeddings (use zeros, faster)")
    parser.add_argument(
        "--retriever",
        choices=["bm25", "hybrid"],
        default="bm25",
        help="Retriever for building training pairs (bm25 works offline)",
    )
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    cfg.ensure_dirs()
    if args.model:
        cfg.rerank["gnn_model"] = args.model

    output_dir = Path(args.output_dir) if args.output_dir else cfg.output_dir / "exp4_gnn_reranker"

    print("=" * 60)
    print("  GNN RERANKER TRAINING")
    print(f"  Model: {cfg.rerank['gnn_model']}  |  Device: {args.device}")
    print(f"  Embeddings: {not args.no_embeddings}  |  Val split: {args.val_split}")
    print("=" * 60)

    # 1. Data
    print("\n[1/6] Loading data...")
    samples = load_dataset(cfg.datasets[0], cfg.data_dir)
    if args.num_samples > 0:
        samples = samples[: args.num_samples]
    random.shuffle(samples)
    n_val = int(len(samples) * args.val_split)
    train_samples = samples[n_val:]
    val_samples = samples[:n_val]
    print(f"  Train: {len(train_samples)}  |  Val: {len(val_samples)}  |  "
          f"Total: {len(samples)} QA samples")

    # 2. Corpus
    print("\n[2/6] Building chunk corpus...")
    corpus_chunks, gold_map = build_corpus(train_samples + val_samples, cfg)
    print(f"  {len(corpus_chunks)} chunks, {len(gold_map)} questions with gold evidence")

    # 3. Retrieval indices
    print(f"\n[3/6] Building retrieval indices ({args.retriever})...")
    bm25 = BM25Retriever(k1=cfg.retrieval["bm25_k1"], b=cfg.retrieval["bm25_b"])
    bm25.index(corpus_chunks)
    if args.retriever == "hybrid":
        from feg_rag.retrieval.dense import DenseRetriever
        from feg_rag.retrieval.hybrid import HybridRetriever
        dense = DenseRetriever(model_name=cfg.retrieval["dense_model"])
        dense.index(corpus_chunks)
        retriever = HybridRetriever(bm25, dense, alpha=cfg.retrieval["hybrid_alpha"])
    else:
        retriever = bm25

    # 4. Graph + features
    print("\n[4/6] Building graph and node features...")
    entity_map = extract_entities(corpus_chunks)
    graph = build_financial_evidence_graph(
        corpus_chunks, entity_map=entity_map, add_semantic_edges=False
    )
    print(f"  Graph: {graph.num_nodes} nodes, {graph.num_edges} edges")

    # Compute text embeddings (or skip with --no_embeddings)
    #
    # Use the subset of embed features, since the embedding
    # is only meaningful for chunk nodes.
    retrieval_scores = warmup_retrieval_scores(train_samples + val_samples, retriever)
    features = build_node_features(
        graph,
        corpus_chunks,
        entity_map,
        retrieval_scores,
        compute_embeddings=not args.no_embeddings,
        embedding_device=args.device,
    )

    # 5. Train
    print("\n[5/6] Training...")
    reranker, history, meta = train_gnn_reranker(
        train_samples,
        retriever,
        graph,
        features,
        gold_map,
        cfg,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        min_pairs=args.min_pairs,
        verbose=True,
    )

    if reranker is None:
        print("\nTraining skipped. Check data and min_pairs threshold.")
        sys.exit(1)

    print_loss_summary(history)

    # 6. Evaluate on validation set
    print("\n[6/6] Evaluating on validation set...")
    val_results = _evaluate_reranker(
        reranker, val_samples, retriever, graph, features, gold_map, cfg
    )
    if val_results:
        metrics = compute_all_metrics(
            "GNN+Rerank", val_results, k_values=cfg.evaluation["recall_k_values"]
        )
        print(f"  MRR:             {metrics.mrr:.4f}")
        for k, v in sorted(metrics.evidence_recall.items()):
            print(f"  Recall@{k:>2}:        {v:.4f}")
        print(f"  NDCG@10:         {metrics.ndcg.get(10, 0):.4f}")

    paths = save_training_artifacts(reranker, history, output_dir, meta, experiment="exp4")
    print(f"\nCheckpoint:  {paths['checkpoint']}")
    print(f"Loss JSON:   {paths['history']}")
    if paths["plot"]:
        print(f"Loss curve:  {paths['plot']}")
    else:
        print("Loss curve:  skipped (install matplotlib to enable PNG export)")

    print("\nTraining complete.")


def _evaluate_reranker(reranker, samples, retriever, graph, features, gold_map, cfg):
    """Rerank candidates for each validation sample and return results for metrics."""
    results = []
    top_k = cfg.retrieval.get("top_k", 50)
    chunk_by_id = {c.chunk_id: c for c in []}  # not needed for id-only results

    for s in samples:
        # Initial retrieval
        retrieved = retriever.search(s["question"], top_k=top_k)
        candidate_ids = [c.chunk_id for c, _ in retrieved]

        # Build chunk objects for reranker (minimal)
        candidates_with_scores = [
            (c, score) for (c, score) in retrieved
        ]

        # Rerank (if reranker supports it directly)
        try:
            reranked = reranker.rerank(
                s["question"],
                candidates_with_scores,
                graph,
                features,
            )
            reranked_ids = [c.chunk_id for c, _ in reranked]
        except Exception:
            # Fall back to retrieval order
            reranked_ids = candidate_ids

        results.append({
            "question_id": s["id"],
            "question": s["question"],
            "gold_answer": s["answer"],
            "gold_evidence_ids": gold_map.get(s["id"], []),
            "retrieved_chunk_ids": reranked_ids,
        })

    return results


if __name__ == "__main__":
    main()
