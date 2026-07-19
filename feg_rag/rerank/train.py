"""Shared GNN / R-GCN training utilities."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch.nn as nn

from feg_rag.config import Config
from feg_rag.data.chunker import Chunk
from feg_rag.data.corpus import build_benchmark_corpus
from feg_rag.graph.builder import FinancialEvidenceGraph
from feg_rag.graph.features import build_node_features
from feg_rag.rerank.gnn import GNNFusionReranker, GraphSAGEReranker, GATv2Reranker, RerankDataset
from feg_rag.rerank.qfe_rgcn import (
    QFERGCNFusionReranker,
    QFERGCNFusionRerankerV2,
    QFERGCNRerankDataset,
    QFERGCNReranker,
    EntityGatedScoringHead,
    RetrievalPreservedFusionHead,
    QUERY_EMBED_DIM,
    BGE_QUERY_EMBED_DIM,
    derive_query_vector,
    build_query_embedding_cache,
    build_bge_query_embedding_cache,
)
from feg_rag.rerank.rgcn import RGCNFusionReranker, RGCNRerankDataset, RGCNReranker, LiteRGCNReranker
from feg_rag.rerank.dcf_gnn import DCFGNNFusionReranker, DCFRerankDataset, DCFGNNReranker
from feg_rag.rerank.c2_dcf_gnn import C2DCFFusionReranker, C2DCFDataset, C2DCFGNNReranker
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


def build_train_pairs_multi_neg(
    samples: List[Dict],
    retriever: Retriever,
    gold_map: Dict[str, List[str]],
    top_k: int = 50,
    negatives_per_positive: int = 5,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Build (positive, negative) pairs with multiple hard negatives per query.

    For each query that has gold evidence in the top-*k* retrieval results,
    up to *negatives_per_positive* non-gold chunks are selected as hard
    negatives (in retrieval rank order).  One pair is created for each
    (positive, negative) combination, giving the model more training signal.

    Returns:
        train_samples: List of dicts with keys:
            positive, negative, question, retrieval_scores
        meta: Dict with num_pairs, num_queries_with_gold_in_topk, etc.
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
            # Take up to K hard negatives (already in retrieval rank order)
            for neg in negs[:negatives_per_positive]:
                train_samples.append({
                    "positive": pos,
                    "negative": neg,
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
        "negatives_per_positive": negatives_per_positive,
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
    negatives_per_positive: int = 5,
    min_retrieval_weight: float = 0.35,
    delta_reg: float = 0.05,
    query_encoder: str = "heuristic",
    query_embedding_cache: Optional[str] = None,
    checkpoint_dir: Optional[str] = None,
    checkpoint_prefix: str = "gnn_reranker",
    checkpoint_every: int = 1,
) -> Tuple[Optional[Union[GNNFusionReranker, RGCNFusionReranker, DCFGNNFusionReranker, C2DCFFusionReranker, QFERGCNFusionReranker, QFERGCNFusionRerankerV2]], List[float], Dict[str, Any]]:
    """Train GraphSAGE, GATv2, R-GCN, DCF-GNN, C2-DCF-GNN, QFE-RGCN, or QFE-RGCN v2 reranker."""
    model_type = cfg.rerank["gnn_model"].lower()

    # v2 uses multi-negative pairs; v1 / others use single-negative pairs
    if model_type == "qfe_rgcn_v2":
        train_samples, pair_meta = build_train_pairs_multi_neg(
            samples, retriever, gold_map,
            top_k=cfg.retrieval["top_k"],
            negatives_per_positive=negatives_per_positive,
        )
    else:
        train_samples, pair_meta = build_train_pairs(
            samples, retriever, gold_map,
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

    elif model_type == "qfe_rgcn_v2":
        # ---- QFE-RGCN v2: retrieval-preserved fusion ----
        if batch_size != 1:
            if verbose:
                print(f"  [qfe_rgcn_v2] Forcing batch_size=1 (was {batch_size}) "
                      f"because QFE-RGCN relation gates are per-query.")
            batch_size = 1
            meta["batch_size"] = 1
        meta["negatives_per_positive"] = negatives_per_positive
        meta["min_retrieval_weight"] = min_retrieval_weight
        meta["delta_reg"] = delta_reg
        meta["query_encoder"] = query_encoder

        # Pre-compute query embeddings (BGE or heuristic)
        all_questions = [s.get("question", "") for s in train_samples]
        if query_encoder == "bge":
            if verbose:
                print("  [qfe_rgcn_v2] Building BGE query embeddings...")
            query_embeddings = build_bge_query_embedding_cache(
                all_questions,
                device=device,
                cache_path=query_embedding_cache,
                fail_on_missing_model=True,
            )
        else:
            if query_embeddings is None:
                query_embeddings = build_query_embedding_cache(all_questions)
            else:
                for q in all_questions:
                    if q not in query_embeddings:
                        query_embeddings[q] = derive_query_vector(q)

        # Build learnable projection for BGE embeddings
        query_projection: Optional[nn.Module] = None
        actual_query_embed_dim = QUERY_EMBED_DIM
        if query_encoder == "bge" and query_embeddings:
            sample_emb = next(iter(query_embeddings.values()))
            bge_dim = sample_emb.shape[0]
            if bge_dim != QUERY_EMBED_DIM:
                query_projection = nn.Linear(bge_dim, QUERY_EMBED_DIM).to(device)
                if verbose:
                    print(f"  [qfe_rgcn_v2] Learnable projection: "
                          f"{bge_dim} → {QUERY_EMBED_DIM}")

        dataset = QFERGCNRerankDataset(
            train_samples, graph, features,
            query_embeddings=query_embeddings,
            chunk_lookup=chunk_lookup if chunk_lookup else None,
            query_embed_dim=QUERY_EMBED_DIM,
        )
        gnn_out_dim = cfg.rerank.get("gnn_out_dim", 64)
        gnn_model = QFERGCNReranker(
            in_dim=feat_dim,
            hidden_dim=cfg.rerank["gnn_hidden"],
            out_dim=gnn_out_dim,
            num_relations=dataset.num_relations,
            query_embed_dim=QUERY_EMBED_DIM,
            dropout=cfg.rerank["gnn_dropout"],
        )
        scoring_head = EntityGatedScoringHead(
            query_embed_dim=QUERY_EMBED_DIM,
            chunk_proj_dim=64,
            gnn_out_dim=gnn_out_dim,
            base_feat_dim=base_dim,
            hidden_dim=cfg.rerank["gnn_hidden"],
            dropout=cfg.rerank["gnn_dropout"],
        )
        fusion_head = RetrievalPreservedFusionHead(
            query_embed_dim=QUERY_EMBED_DIM,
            hidden_dim=64,
            min_ret_weight=min_retrieval_weight,
            dropout=0.1,
        )
        gnn_proj = nn.Linear(gnn_out_dim, 1)
        # Determine raw embedding dimension
        _raw_embed_dim: Optional[int] = None
        if query_encoder == "bge" and query_embeddings:
            _sample_emb = next(iter(query_embeddings.values()))
            _raw_embed_dim = _sample_emb.shape[0]
        reranker = QFERGCNFusionRerankerV2(
            gnn_model,
            scoring_head,
            fusion_head,
            gnn_proj,
            relation_map=dataset.relation_map,
            query_embeddings=query_embeddings,
            query_embed_dim=QUERY_EMBED_DIM,
            min_retrieval_weight=min_retrieval_weight,
            delta_reg=delta_reg,
            device=device,
            query_projection=query_projection,
            query_encoder=query_encoder,
            query_embedding_dim_raw=_raw_embed_dim,
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
    elif model_type == "rgcn_lite":
        dataset = RGCNRerankDataset(train_samples, graph, features,
                                    chunk_lookup=chunk_lookup if chunk_lookup else None)
        lite_hidden = cfg.rerank.get("rgcn_lite_hidden", 96)
        lite_out_dim = cfg.rerank.get("rgcn_lite_out_dim", 48)
        model = LiteRGCNReranker(
            in_dim=feat_dim,
            hidden_dim=lite_hidden,
            out_dim=lite_out_dim,
            num_relations=dataset.num_relations,
            dropout=cfg.rerank["gnn_dropout"],
        )
        reranker = RGCNFusionReranker(
            model,
            relation_map=dataset.relation_map,
            # Lite is an efficiency baseline: keep retrieval as the anchor and
            # use the small graph model as a conservative correction.
            alpha=cfg.rerank.get("rgcn_lite_fusion_alpha", 0.85),
            beta=cfg.rerank.get("rgcn_lite_fusion_beta", 0.0),
            gamma=cfg.rerank.get("rgcn_lite_fusion_gamma", 0.15),
            device=device,
        )
        meta["rgcn_lite_hidden"] = lite_hidden
        meta["rgcn_lite_out_dim"] = lite_out_dim
        meta["rgcn_lite_fusion_alpha"] = reranker.alpha
        meta["rgcn_lite_fusion_beta"] = reranker.beta
        meta["rgcn_lite_fusion_gamma"] = reranker.gamma
    elif model_type == "gatv2":
        # ---- GATv2: multi-head dense attention for evidence reranking ----
        # Dense GATv2 attention is heavy — force batch_size=1 per query subgraph
        if batch_size != 1:
            if verbose:
                print(f"  [gatv2] Forcing batch_size=1 (was {batch_size}) "
                      f"because dense GATv2 attention is per-subgraph.")
            batch_size = 1
            meta["batch_size"] = 1
        gatv2_hidden = cfg.rerank.get("gatv2_hidden", 96)
        gatv2_out_dim = cfg.rerank.get("gatv2_out_dim", 48)
        gatv2_heads = cfg.rerank.get("gatv2_heads", 4)
        meta["gatv2_heads"] = gatv2_heads
        meta["gatv2_hidden_dim"] = gatv2_hidden
        meta["gatv2_out_dim"] = gatv2_out_dim
        # Strong baseline fusion: keep retrieval as anchor but give the
        # residual GATv2 correction enough weight to actually move candidates.
        meta["gatv2_fusion_alpha"] = cfg.rerank.get("gatv2_fusion_alpha", 0.70)
        meta["gatv2_fusion_beta"] = cfg.rerank.get("gatv2_fusion_beta", 0.0)
        meta["gatv2_fusion_gamma"] = cfg.rerank.get("gatv2_fusion_gamma", 0.30)
        dataset = RerankDataset(train_samples, graph, features,
                                chunk_lookup=chunk_lookup if chunk_lookup else None)
        model = GATv2Reranker(
            in_dim=feat_dim,
            hidden_dim=gatv2_hidden,
            out_dim=gatv2_out_dim,
            heads=gatv2_heads,
            dropout=cfg.rerank["gnn_dropout"],
        )
        reranker = GNNFusionReranker(
            model,
            alpha=meta["gatv2_fusion_alpha"],
            beta=meta["gatv2_fusion_beta"],
            gamma=meta["gatv2_fusion_gamma"],
            device=device,
        )
    elif model_type == "dcf_gnn":
        dataset = DCFRerankDataset(
            train_samples, graph, features,
            chunk_lookup=chunk_lookup if chunk_lookup else None,
        )
        model = DCFGNNReranker(
            in_dim=feat_dim,
            hidden_dim=cfg.rerank["gnn_hidden"],
            out_dim=cfg.rerank.get("gnn_out_dim", 64),
            num_structural_relations=dataset.num_structural_relations,
            num_semantic_relations=dataset.num_semantic_relations,
            dropout=cfg.rerank["gnn_dropout"],
        )
        reranker = DCFGNNFusionReranker(
            model,
            structural_relation_map=dataset.structural_relation_map,
            semantic_relation_map=dataset.semantic_relation_map,
            alpha=cfg.rerank.get("dcf_fusion_alpha", cfg.rerank["fusion_alpha"]),
            beta=cfg.rerank.get("dcf_fusion_beta", cfg.rerank["fusion_beta"]),
            gamma=cfg.rerank.get("dcf_fusion_gamma", cfg.rerank["fusion_gamma"]),
            device=device,
            chunk_lookup=dataset.chunk_lookup,
            incident_edges=dataset.incident_edges,
        )
        meta["num_structural_relations"] = dataset.num_structural_relations
        meta["num_semantic_relations"] = dataset.num_semantic_relations
    elif model_type == "c2_dcf_gnn":
        dataset = C2DCFDataset(
            train_samples, graph, features,
            chunk_lookup=chunk_lookup if chunk_lookup else None,
        )
        model = C2DCFGNNReranker(
            in_dim=feat_dim,
            hidden_dim=cfg.rerank["gnn_hidden"],
            out_dim=cfg.rerank.get("gnn_out_dim", 64),
            num_structural_relations=dataset.num_structural_relations,
            num_semantic_relations=dataset.num_semantic_relations,
            top_k=cfg.rerank.get("c2_top_k", 2),
            tau=cfg.rerank.get("c2_tau", 0.15),
            dropout=cfg.rerank["gnn_dropout"],
        )
        reranker = C2DCFFusionReranker(
            model,
            structural_relation_map=dataset.structural_relation_map,
            semantic_relation_map=dataset.semantic_relation_map,
            # C2 raw expert-mixture scores need conservative calibration.
            # Do not inherit the generic GNN fusion defaults, which give the
            # lightly trained C2 branch too much control during reranking.
            alpha=cfg.rerank.get("c2_fusion_alpha", 0.85),
            beta=cfg.rerank.get("c2_fusion_beta", 0.0),
            gamma=cfg.rerank.get("c2_fusion_gamma", 0.15),
            device=device,
            chunk_lookup=dataset.chunk_lookup,
            incident_edges=dataset.incident_edges,
            route_contrastive_lambda=cfg.rerank.get("c2_route_contrastive_lambda", 0.05),
            confidence_lambda=cfg.rerank.get("c2_confidence_lambda", 0.05),
            margin=cfg.rerank.get("c2_margin", 0.1),
        )
        meta["num_structural_relations"] = dataset.num_structural_relations
        meta["num_semantic_relations"] = dataset.num_semantic_relations
        meta["c2_top_k"] = cfg.rerank.get("c2_top_k", 2)
        meta["c2_tau"] = cfg.rerank.get("c2_tau", 0.15)
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

    # Per-epoch checkpoint support (C2-DCF-GNN and future models)
    fit_kwargs: Dict[str, Any] = dict(
        epochs=meta["epochs"],
        lr=meta["lr"],
        batch_size=batch_size,
        verbose=verbose,
    )
    if model_type == "c2_dcf_gnn" and checkpoint_dir:
        fit_kwargs["checkpoint_dir"] = checkpoint_dir
        fit_kwargs["checkpoint_prefix"] = checkpoint_prefix
        fit_kwargs["checkpoint_every"] = checkpoint_every

    history = reranker.fit(dataset, **fit_kwargs)
    meta["final_loss"] = history[-1] if history else None
    meta["initial_loss"] = history[0] if history else None
    return reranker, history, meta


def save_training_artifacts(
    reranker: GNNFusionReranker | RGCNFusionReranker | DCFGNNFusionReranker | C2DCFFusionReranker | QFERGCNFusionReranker | QFERGCNFusionRerankerV2,
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
