"""Shared GNN / R-GCN training utilities."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from feg_rag.config import Config
from feg_rag.data.chunker import Chunk
from feg_rag.data.corpus import build_benchmark_corpus
from feg_rag.graph.builder import FinancialEvidenceGraph
from feg_rag.graph.features import build_node_features
from feg_rag.rerank.gnn import GNNFusionReranker, GraphSAGEReranker, RerankDataset
from feg_rag.rerank.qfe_rgcn import (
    QFERGCNFusionReranker,
    QFERGCNRerankDataset,
    QFERGCNReranker,
    EntityGatedScoringHead,
    QUERY_EMBED_DIM,
    derive_query_vector,
    build_query_embedding_cache,
)
from feg_rag.rerank.rgcn import RGCNFusionReranker, RGCNRerankDataset, RGCNReranker
from feg_rag.rerank.query_features import QUERY_FEATURE_DIM
from feg_rag.retrieval.bm25 import BM25Retriever
from feg_rag.retrieval.hybrid import HybridRetriever

Retriever = Union[BM25Retriever, HybridRetriever]


def build_corpus(
    samples: List[Dict],
    cfg: Config,
    allow_gold_only_corpus: bool | None = None,
) -> Tuple[List[Chunk], Dict[str, List[str]]]:
    """Build document corpus and align gold evidence to corpus chunk IDs."""
    corpus, gold_map, alignments = build_benchmark_corpus(
        samples,
        cfg,
        allow_gold_only_corpus=allow_gold_only_corpus,
    )
    failed = [a for a in alignments if not a.matched_chunk_ids]
    if failed:
        print(f"  [WARN] Gold alignment failed for {len(failed)} evidence snippets")
    return corpus, gold_map


def build_train_pairs(
    samples: List[Dict],
    retriever: Retriever,
    gold_map: Dict[str, List[str]],
    top_k: int = 50,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Build (positive, negative) chunk pairs from retrieval results.

    Returns:
        train_samples: List of dicts with keys:
            positive, negative, question, retrieval_scores
        meta: Dict with num_pairs, num_queries_with_gold_in_topk,
            num_queries_without_gold_in_topk.
    """
    train_samples: List[Dict[str, Any]] = []
    num_with_gold = 0
    num_without_gold = 0

    for s in samples:
        gold = set(gold_map.get(s["id"], []))
        if not gold:
            continue
        retrieved = retriever.search(s["question"], top_k=top_k)
        pos = None
        negs: List[str] = []
        retrieval_scores: Dict[str, float] = {}
        for c, score in retrieved:
            retrieval_scores[c.chunk_id] = float(score)
            if c.chunk_id in gold and pos is None:
                pos = c.chunk_id
            elif c.chunk_id not in gold:
                negs.append(c.chunk_id)
        if pos and negs:
            train_samples.append({
                "positive": pos,
                "negative": negs[0],
                "question": s["question"],
                "retrieval_scores": retrieval_scores,
            })
            num_with_gold += 1
        elif pos:
            num_with_gold += 1
        else:
            num_without_gold += 1

    meta = {
        "num_pairs": len(train_samples),
        "num_queries_with_gold_in_topk": num_with_gold,
        "num_queries_without_gold_in_topk": num_without_gold,
    }
    return train_samples, meta


def warmup_retrieval_scores(
    samples: List[Dict],
    retriever: Retriever,
    limit: int = 5,
    top_k: int = 50,
) -> Dict[str, float]:
    """Collect retrieval scores for node feature construction."""
    retrieval_scores: Dict[str, float] = {}
    for s in samples[:limit]:
        for c, score in retriever.search(s["question"], top_k=top_k):
            if c.chunk_id not in retrieval_scores:
                retrieval_scores[c.chunk_id] = score
    return retrieval_scores


def train_gnn_reranker(
    samples: List[Dict],
    retriever: Retriever,
    graph: FinancialEvidenceGraph,
    features: Dict[str, Any],
    gold_map: Dict[str, List[str]],
    cfg: Config,
    *,
    epochs: Optional[int] = None,
    batch_size: int = 32,
    device: str = "cpu",
    min_pairs: int = 10,
    verbose: bool = True,
    corpus_chunks: Optional[List[Chunk]] = None,
    query_embeddings: Optional[Dict[str, np.ndarray]] = None,
) -> Tuple[Optional[Union[GNNFusionReranker, RGCNFusionReranker, QFERGCNFusionReranker]], List[float], Dict[str, Any]]:
    """Train GraphSAGE, R-GCN, or QFE-RGCN reranker and return model + loss history."""
    train_samples, pair_meta = build_train_pairs(
        samples,
        retriever,
        gold_map,
        top_k=cfg.retrieval["top_k"],
    )
    meta: Dict[str, Any] = {
        "num_pairs": len(train_samples),
        "model": cfg.rerank["gnn_model"],
        "epochs": epochs or cfg.rerank["gnn_epochs"],
        "lr": cfg.rerank["gnn_lr"],
        "batch_size": batch_size,
    }
    meta.update(pair_meta)

    if len(train_samples) < min_pairs:
        if verbose:
            print(
                f"  Not enough training pairs ({len(train_samples)} < {min_pairs}); skipping."
            )
        return None, [], meta

    if verbose:
        print(f"  Training pairs: {len(train_samples)}")
        print(f"  Model: {cfg.rerank['gnn_model']}  epochs: {meta['epochs']}")

    # Build chunk lookup for query-augmented features
    chunk_lookup: Dict[str, Chunk] = {}
    if corpus_chunks:
        chunk_lookup = {c.chunk_id: c for c in corpus_chunks}

    # Feature dim includes base + query-augmented features
    base_dim = next(iter(features.values())).shape[0]
    feat_dim = base_dim + QUERY_FEATURE_DIM
    model_type = cfg.rerank["gnn_model"].lower()

    if model_type == "qfe_rgcn":
        # QFE-RGCN uses query-aware relation gates that are per-query.
        # The collate function uses q_embeds[0] which would cause all samples
        # in a batch to share the first query embedding.  Forcing batch_size=1
        # ensures each forward pass has exactly one query.
        if batch_size != 1:
            if verbose:
                print(f"  [qfe_rgcn] Forcing batch_size=1 (was {batch_size}) "
                      f"because QFE-RGCN relation gates are per-query.")
            batch_size = 1
            meta["batch_size"] = 1

        # Pre-compute query embeddings for QFE-RGCN
        all_questions = [s.get("question", "") for s in train_samples]
        if query_embeddings is None:
            query_embeddings = build_query_embedding_cache(all_questions)
        else:
            # Ensure all training questions have embeddings
            for q in all_questions:
                if q not in query_embeddings:
                    query_embeddings[q] = derive_query_vector(q)

        dataset = QFERGCNRerankDataset(
            train_samples, graph, features,
            query_embeddings=query_embeddings,
            chunk_lookup=chunk_lookup if chunk_lookup else None,
            query_embed_dim=QUERY_EMBED_DIM,
        )
        gnn_model = QFERGCNReranker(
            in_dim=feat_dim,
            hidden_dim=cfg.rerank["gnn_hidden"],
            out_dim=cfg.rerank.get("gnn_out_dim", 64),
            num_relations=dataset.num_relations,
            query_embed_dim=QUERY_EMBED_DIM,
            dropout=cfg.rerank["gnn_dropout"],
        )
        scoring_head = EntityGatedScoringHead(
            query_embed_dim=QUERY_EMBED_DIM,
            chunk_proj_dim=64,
            gnn_out_dim=cfg.rerank.get("gnn_out_dim", 64),
            base_feat_dim=base_dim,
            hidden_dim=cfg.rerank["gnn_hidden"],
            dropout=cfg.rerank["gnn_dropout"],
        )
        reranker: Union[GNNFusionReranker, RGCNFusionReranker, QFERGCNFusionReranker] = QFERGCNFusionReranker(
            gnn_model,
            scoring_head,
            relation_map=dataset.relation_map,
            query_embeddings=query_embeddings,
            query_embed_dim=QUERY_EMBED_DIM,
            device=device,
        )

    elif model_type == "rgcn":
        dataset = RGCNRerankDataset(train_samples, graph, features,
                                    chunk_lookup=chunk_lookup if chunk_lookup else None)
        model = RGCNReranker(
            in_dim=feat_dim,
            hidden_dim=cfg.rerank["gnn_hidden"],
            num_relations=dataset.num_relations,
            dropout=cfg.rerank["gnn_dropout"],
        )
        reranker: Union[GNNFusionReranker, RGCNFusionReranker] = RGCNFusionReranker(
            model,
            relation_map=dataset.relation_map,
            alpha=cfg.rerank["fusion_alpha"],
            beta=cfg.rerank["fusion_beta"],
            gamma=cfg.rerank["fusion_gamma"],
            device=device,
        )
    else:
        dataset = RerankDataset(train_samples, graph, features,
                                chunk_lookup=chunk_lookup if chunk_lookup else None)
        model = GraphSAGEReranker(
            in_dim=feat_dim,
            hidden_dim=cfg.rerank["gnn_hidden"],
            dropout=cfg.rerank["gnn_dropout"],
        )
        reranker = GNNFusionReranker(
            model,
            alpha=cfg.rerank["fusion_alpha"],
            beta=cfg.rerank["fusion_beta"],
            gamma=cfg.rerank["fusion_gamma"],
            device=device,
        )

    history = reranker.fit(
        dataset,
        epochs=meta["epochs"],
        lr=meta["lr"],
        batch_size=batch_size,
        verbose=verbose,
    )
    meta["final_loss"] = history[-1] if history else None
    meta["initial_loss"] = history[0] if history else None
    return reranker, history, meta


def save_training_artifacts(
    reranker: GNNFusionReranker | RGCNFusionReranker,
    history: List[float],
    output_dir: Path,
    meta: Optional[Dict[str, Any]] = None,
    *,
    experiment: str = "exp4",
) -> Dict[str, Path]:
    """Save checkpoint, loss history JSON, and optional loss curve PNG.

    Args:
        reranker: Trained reranker model.
        history: List of per-epoch loss values.
        output_dir: Directory to save artifacts (e.g. outputs/exp4_gnn_reranker/).
        meta: Optional metadata dict to embed in loss history JSON.
        experiment: Experiment identifier for file prefix (e.g. "exp4").
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / "model_checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    ckpt_path = ckpt_dir / f"{experiment}_gnn_reranker_{stamp}.pt"
    history_path = output_dir / f"{experiment}_loss_history_{stamp}.json"
    plot_path = output_dir / f"{experiment}_loss_curve_{stamp}.png"

    reranker.save(ckpt_path)

    payload = {
        "train_loss": [{"epoch": i + 1, "loss": loss} for i, loss in enumerate(history)],
        "meta": meta or {},
        "timestamp": stamp,
    }
    with open(history_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    try:
        import matplotlib.pyplot as plt

        epochs = list(range(1, len(history) + 1))
        plt.figure(figsize=(8, 4))
        plt.plot(epochs, history, marker="o", linewidth=2)
        plt.xlabel("Epoch")
        plt.ylabel("Train Loss")
        plt.title("GNN Reranker Training Loss")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=120)
        plt.close()
    except ImportError:
        plot_path = None

    return {
        "checkpoint": ckpt_path,
        "history": history_path,
        "plot": plot_path,
    }


def print_loss_summary(history: List[float]) -> None:
    """Print a compact loss summary table with mini text chart."""
    if not history:
        print("No training history to display.")
        return

    print("\n" + "=" * 60)
    print("  TRAINING LOSS SUMMARY")
    print("=" * 60)

    # Build mini text-based loss curve (inline, no matplotlib needed)
    losses = history
    loss_min = min(losses)
    loss_max = max(losses)
    span = max(loss_max - loss_min, 1e-8)

    for i, loss in enumerate(losses, 1):
        # Normalised bar: each # ≈ 1/30 of the range (ASCII-safe)
        bar_len = max(1, int(round((loss - loss_min) / span * 30)))
        bar = "#" * bar_len + "." * (30 - bar_len)
        delta_str = ""
        if i > 1:
            d = loss - losses[i - 2]
            arrow = "v" if d < 0 else "^"
            delta_str = f" {arrow}{abs(d):.4f}"
        flag = ""
        if loss == loss_min:
            flag = " <-- best"
        print(f"  Epoch {i:>3}/{len(history)}  | {bar} | loss={loss:.4f}{delta_str}{flag}")

    print("-" * 60)
    print(f"  Initial:  {history[0]:.4f}")
    print(f"  Final:    {history[-1]:.4f}")
    if len(history) > 1:
        delta = history[-1] - history[0]
        print(f"  d total:  {delta:+.4f}  ({delta / history[0] * 100:+.1f}%)")
    print("=" * 60)
