"""Table 1: Non-LLM Reranking Comparison.

Compares evidence ranking performance across retrieval and reranking methods
WITHOUT any LLM involvement. Proves that graph-based reranking improves
evidence ranking over non-graph baselines.

Methods compared:
    Best Retriever (Hybrid E5-Mistral)
    + Cross-Encoder
    + PPR
    + GraphSAGE
    + R-GCN
    + R-GCN + Constraint Score
    Final Graph (Ours) — QFE-RGCN
    Fast Final Graph (Ours) — Lightweight Query-Adaptive Fusion

Metrics: Recall@5, Recall@10, MRR, nDCG@10

Usage:
    # Smoke test
    python experiments/table1_non_llm_reranking_comparison.py \\
        --config configs/table1_non_llm_reranking_e5_mistral.yaml \\
        --num_samples 5 --epochs 1 --sanity

    # Full run
    python experiments/table1_non_llm_reranking_comparison.py \\
        --config configs/table1_non_llm_reranking_e5_mistral.yaml \\
        --device cuda --epochs 10
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch.nn as nn

from feg_rag.config import Config
from feg_rag.data.chunker import Chunk
from feg_rag.data.corpus import build_benchmark_corpus
from feg_rag.data.loader import load_dataset
from feg_rag.evaluation.metrics import compute_all_metrics
from feg_rag.graph.builder import build_financial_evidence_graph
from feg_rag.graph.entities import EntityExtractor, extract_entities
from feg_rag.graph.features import build_node_features
from feg_rag.rerank.fusion import ConstraintScorer, FusionScorer
from feg_rag.rerank.gnn import GNNFusionReranker, GraphSAGEReranker
from feg_rag.rerank.ppr import ppr_rerank
from feg_rag.rerank.qfe_rgcn import (
    QFERGCNFusionReranker,
    QFERGCNFusionRerankerV2,
    QFERGCNReranker,
    QFERGCNRerankDataset,
    EntityGatedScoringHead,
    RetrievalPreservedFusionHead,
    QUERY_EMBED_DIM,
    BGE_QUERY_EMBED_DIM,
    derive_query_vector,
    build_query_embedding_cache,
    build_bge_query_embedding_cache,
)
from feg_rag.rerank.query_features import QUERY_FEATURE_DIM
from feg_rag.rerank.rgcn import RGCNFusionReranker, RGCNReranker
from feg_rag.rerank.train import (
    build_train_pairs,
    save_training_artifacts,
    train_gnn_reranker,
    warmup_retrieval_scores,
)
from feg_rag.rerank.mono_t5 import run_mono_t5_reranking
from feg_rag.rerank.list_t5 import run_list_t5_reranking
from feg_rag.rerank.fast_final_graph import (
    FastFinalGraphDataset,
    QueryAdaptiveFusionReranker,
    load_jsonl_results,
    save_checkpoint,
    load_checkpoint,
    train_fast_final_graph,
    evaluate_fast_final_graph,
    compute_config_fingerprint,
    save_dataset_cache,
    load_dataset_cache,
    PAIR_FEATURE_DIM,
    QUERY_ONLY_FEATURE_DIM,
)
from feg_rag.retrieval.bm25 import BM25Retriever
from feg_rag.retrieval.cross_encoder import CrossEncoderReranker
from feg_rag.retrieval.dense import DenseRetriever
from feg_rag.retrieval.hybrid import HybridRetriever


class CandidatePoolRetriever:
    """Replay a fixed candidate pool produced by a prior retriever run."""

    def __init__(
        self,
        results_jsonl: str | Path,
        chunk_by_id: Dict[str, Chunk],
        *,
        name: str = "CandidatePool",
    ):
        self.results_jsonl = Path(results_jsonl)
        self.chunk_by_id = chunk_by_id
        self.name = name
        self._by_question: Dict[str, List[str]] = {}
        self._load()

    def _load(self) -> None:
        if not self.results_jsonl.exists():
            raise FileNotFoundError(f"Candidate results not found: {self.results_jsonl}")
        with self.results_jsonl.open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                question = row.get("question")
                ids = list(row.get("retrieved_chunk_ids") or [])
                if question:
                    self._by_question[question] = ids
        if not self._by_question:
            raise ValueError(f"No candidate rows loaded from {self.results_jsonl}")

    def search(self, query: str, top_k: int = 50):
        ids = self._by_question.get(query, [])[:top_k]
        results = []
        for rank, cid in enumerate(ids):
            chunk = self.chunk_by_id.get(cid)
            if chunk is None:
                continue
            score = float(top_k - rank)
            results.append((chunk, score))
        return results

    @property
    def num_queries(self) -> int:
        return len(self._by_question)


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════

def _default_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _build_corpus(
    samples: List[Dict],
    cfg: Config,
    *,
    allow_gold_only_corpus: bool = False,
) -> Tuple[List[Chunk], Dict[str, List[str]]]:
    """Build the benchmark corpus from source filings, then align gold evidence."""
    corpus, gold_map, alignments = build_benchmark_corpus(
        samples,
        cfg,
        allow_gold_only_corpus=allow_gold_only_corpus,
    )
    failed = sum(1 for a in alignments if a.warning)
    if failed:
        print(f"  [WARN] Gold alignment failed for {failed} evidence snippets")
    return corpus, gold_map


def _resolve_cache_path(path: Optional[str], cfg: Config) -> Optional[Path]:
    if not path:
        return None
    p = Path(path)
    if not p.is_absolute():
        p = cfg.root_dir / p
    return p


def _atomic_pickle_dump(data: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as fh:
        pickle.dump(data, fh, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)


def _load_or_build_corpus(
    samples: List[Dict],
    cfg: Config,
    cache_path: Optional[Path],
    *,
    rebuild: bool,
    allow_gold_only_corpus: bool,
) -> Tuple[List[Chunk], Dict[str, List[str]]]:
    if cache_path and cache_path.exists() and not rebuild:
        with cache_path.open("rb") as fh:
            data = pickle.load(fh)
        corpus_chunks = data["corpus_chunks"]
        gold_map = data["gold_map"]
        print(f"  [corpus-cache] Loaded: {cache_path}")
        return corpus_chunks, gold_map

    corpus_chunks, gold_map = _build_corpus(
        samples,
        cfg,
        allow_gold_only_corpus=allow_gold_only_corpus,
    )
    if cache_path:
        _atomic_pickle_dump(
            {"corpus_chunks": corpus_chunks, "gold_map": gold_map},
            cache_path,
        )
        print(f"  [corpus-cache] Saved: {cache_path}")
    return corpus_chunks, gold_map


def _fingerprint_ids(ids: List[str]) -> str:
    import hashlib

    h = hashlib.sha256()
    for item in ids:
        h.update(item.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()[:16]


def _graph_cache_metadata(
    corpus_chunks: List[Chunk],
    samples: List[Dict],
    cfg: Config,
    args: argparse.Namespace,
    embedding_dim: int,
) -> Dict:
    return {
        "version": 1,
        "num_chunks": len(corpus_chunks),
        "chunk_ids_hash": _fingerprint_ids([c.chunk_id for c in corpus_chunks]),
        "sample_ids_hash": _fingerprint_ids([s["id"] for s in samples]),
        "chunk_size": cfg.chunk_size,
        "chunk_overlap": cfg.chunk_overlap,
        "top_n": args.top_n,
        "dense_model": cfg.retrieval.get("dense_model", "all-MiniLM-L6-v2"),
        "hybrid_alpha": cfg.retrieval.get("hybrid_alpha", 0.5),
        "embedding_dim": embedding_dim,
        "candidate_results_jsonl": args.candidate_results_jsonl or "",
        "candidate_pool_name": args.candidate_pool_name,
        "graph_feature_retriever_cache": args.graph_feature_retriever_cache or "",
        "graph_options": {
            "add_semantic_edges": False,
            "add_company_nodes": True,
            "add_filing_nodes": True,
            "add_section_nodes": True,
            "add_same_entity_edges": True,
            "max_same_entity_edges": 30,
            "use_edge_weights": True,
        },
    }


def _metadata_matches(cached: Dict, expected: Dict) -> bool:
    keys = [
        "version",
        "num_chunks",
        "chunk_ids_hash",
        "sample_ids_hash",
        "chunk_size",
        "chunk_overlap",
        "top_n",
        "dense_model",
        "hybrid_alpha",
        "embedding_dim",
        "candidate_results_jsonl",
        "candidate_pool_name",
        "graph_feature_retriever_cache",
        "graph_options",
    ]
    return all(cached.get(k) == expected.get(k) for k in keys)


def _load_or_build_graph_features(
    corpus_chunks: List[Chunk],
    samples: List[Dict],
    cfg: Config,
    args: argparse.Namespace,
    hybrid: HybridRetriever,
    chunk_embeddings: Dict[str, np.ndarray],
    embedding_dim: int,
    cache_path: Optional[Path],
):
    expected_meta = _graph_cache_metadata(
        corpus_chunks, samples, cfg, args, embedding_dim
    )

    if cache_path and cache_path.exists() and not args.rebuild_graph_cache:
        try:
            with cache_path.open("rb") as fh:
                data = pickle.load(fh)
            if _metadata_matches(data.get("metadata", {}), expected_meta):
                print(f"  [graph-cache] Loaded: {cache_path}")
                return data["entity_map"], data["graph"], data["features"]
            print("  [graph-cache] Metadata mismatch; rebuilding graph/features")
        except Exception as exc:
            print(f"  [graph-cache] Load failed; rebuilding graph/features: {exc}")

    entity_map = extract_entities(corpus_chunks)
    graph = build_financial_evidence_graph(
        corpus_chunks,
        entity_map=entity_map,
        add_semantic_edges=False,
        add_company_nodes=True,
        add_filing_nodes=True,
        add_section_nodes=True,
        add_same_entity_edges=True,
        max_same_entity_edges=30,
        use_edge_weights=True,
    )
    print(f"  Graph: {graph.num_nodes} nodes, {graph.num_edges} edges")

    retrieval_scores = warmup_retrieval_scores(
        samples, hybrid, top_k=args.top_n,
    )
    features = build_node_features(
        graph, corpus_chunks, entity_map, retrieval_scores,
        chunk_embeddings=chunk_embeddings,
        compute_embeddings=False,
        embedding_device=args.device,
    )

    if cache_path:
        _atomic_pickle_dump(
            {
                "metadata": expected_meta,
                "entity_map": entity_map,
                "graph": graph,
                "features": features,
            },
            cache_path,
        )
        print(f"  [graph-cache] Saved: {cache_path}")

    return entity_map, graph, features


def _load_chunk_embeddings_from_retriever_cache(
    cache_dir: str | Path,
    corpus_chunks: List[Chunk],
) -> Tuple[Dict[str, np.ndarray], int]:
    """Load chunk embeddings from a saved retriever cache."""
    cache_dir = Path(cache_dir)
    meta_path = cache_dir / "meta.pkl"
    emb_path = cache_dir / "embeddings.npy"
    if not meta_path.exists() or not emb_path.exists():
        raise FileNotFoundError(
            f"Retriever cache must contain meta.pkl and embeddings.npy: {cache_dir}"
        )

    with meta_path.open("rb") as fh:
        meta = pickle.load(fh)
    cached_chunks = meta.get("chunks")
    if not cached_chunks:
        raise ValueError(f"Retriever cache has no chunks in meta.pkl: {cache_dir}")

    embeddings = np.load(str(emb_path), mmap_mode="r")
    if embeddings.shape[0] != len(cached_chunks):
        raise ValueError(
            f"Embedding/chunk count mismatch in {cache_dir}: "
            f"{embeddings.shape[0]} embeddings vs {len(cached_chunks)} chunks"
        )

    corpus_ids = {c.chunk_id for c in corpus_chunks}
    chunk_embeddings: Dict[str, np.ndarray] = {}
    missing = 0
    for idx, chunk in enumerate(cached_chunks):
        if chunk.chunk_id not in corpus_ids:
            missing += 1
            continue
        chunk_embeddings[chunk.chunk_id] = np.asarray(embeddings[idx], dtype=np.float32)

    if len(chunk_embeddings) != len(corpus_chunks):
        raise ValueError(
            f"Retriever cache does not cover the active corpus: "
            f"{len(chunk_embeddings)}/{len(corpus_chunks)} chunks matched "
            f"({missing} cached chunks outside corpus)."
        )

    embedding_dim = int(embeddings.shape[1]) if embeddings.ndim == 2 else 0
    print(
        f"  [feature-cache] Loaded chunk embeddings: {cache_dir} "
        f"({len(chunk_embeddings)} chunks, dim={embedding_dim})"
    )
    return chunk_embeddings, embedding_dim


def _make_result(
    sample: Dict,
    retrieved_ids: List[str],
    gold_ids: List[str],
    method: str,
) -> Dict:
    return {
        "question_id": sample["id"],
        "question": sample["question"],
        "gold_answer": sample.get("answer", ""),
        "gold_evidence_ids": gold_ids,
        "retrieved_chunk_ids": retrieved_ids,
        "method": method,
    }


def _has_results(out_dir: Path) -> bool:
    sentinels = ["metrics_full.json", "table1_non_llm_reranking_comparison.csv"]
    return any((out_dir / s).exists() for s in sentinels)


METHOD_FILES = {
    "best_retriever": "best_retriever_results.jsonl",
    "cross_encoder": "cross_encoder_results.jsonl",
    "ppr": "ppr_results.jsonl",
    "graphsage": "graphsage_results.jsonl",
    "rgcn": "rgcn_results.jsonl",
    "rgcn_constraint": "rgcn_constraint_results.jsonl",
    "qfe_rgcn": "qfe_rgcn_results.jsonl",
    "qfe_rgcn_v2": "qfe_rgcn_v2_results.jsonl",
    "qfe_rgcn_ppr": "qfe_rgcn_ppr_results.jsonl",
    "mono_t5": "mono_t5_results.jsonl",
    "list_t5": "list_t5_results.jsonl",
    "fast_final_graph": "fast_final_graph_results.jsonl",
}

METHOD_ORDER = [
    "best_retriever", "cross_encoder", "ppr",
    "graphsage", "rgcn", "rgcn_constraint", "qfe_rgcn", "qfe_rgcn_v2", "qfe_rgcn_ppr",
    "mono_t5", "list_t5", "fast_final_graph",
]

METHOD_LABELS = {
    "best_retriever": "Best Retriever",
    "cross_encoder": "+ Cross-Encoder",
    "ppr": "+ PPR",
    "graphsage": "+ GraphSAGE",
    "rgcn": "+ R-GCN",
    "rgcn_constraint": "+ R-GCN + Constraint Score",
    "qfe_rgcn": "Final Graph v1 (Ours)",
    "qfe_rgcn_v2": "Final Graph v2 (Ours)",
    "qfe_rgcn_ppr": "Final Graph v1 (Ours) + PPR",
    "mono_t5": "MonoT5",
    "list_t5": "ListT5",
    "fast_final_graph": "Fast Final Graph (Ours)",
}


def _write_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


# ═════════════════════════════════════════════════════════════════════════════
# Method runners
# ═════════════════════════════════════════════════════════════════════════════

def run_best_retriever(
    samples: List[Dict],
    hybrid: HybridRetriever,
    gold_map: Dict[str, List[str]],
    top_n: int = 50,
    output_k: int = 10,
) -> List[Dict]:
    """Hybrid retrieval baseline ("Best Retriever")."""
    results = []
    for s in samples:
        hr = hybrid.search(s["question"], top_k=top_n)
        ids = [c.chunk_id for c, _ in hr[:output_k]]
        results.append(_make_result(s, ids, gold_map.get(s["id"], []), "best_retriever"))
    return results


def run_cross_encoder(
    samples: List[Dict],
    hybrid: HybridRetriever,
    gold_map: Dict[str, List[str]],
    cross_encoder: CrossEncoderReranker,
    top_n: int = 100,
    output_k: int = 10,
) -> List[Dict]:
    """Cross-Encoder reranking on top of hybrid retrieval."""
    results = []
    for s in samples:
        hr = hybrid.search(s["question"], top_k=top_n)
        reranked = cross_encoder.rerank(s["question"], hr, top_k=output_k)
        ids = [c.chunk_id for c, _ in reranked[:output_k]]
        results.append(_make_result(s, ids, gold_map.get(s["id"], []), "cross_encoder"))
    return results


def run_ppr(
    samples: List[Dict],
    hybrid: HybridRetriever,
    graph,
    corpus_chunks: List[Chunk],
    gold_map: Dict[str, List[str]],
    extractor: EntityExtractor,
    ppr_alpha: float = 0.85,
    top_n: int = 50,
    output_k: int = 10,
) -> List[Dict]:
    """PPR reranking on top of hybrid retrieval."""
    results = []
    for s in samples:
        hr = hybrid.search(s["question"], top_k=top_n)
        candidate_ids = [c.chunk_id for c, _ in hr]
        retrieval_scores = {c.chunk_id: float(score) for c, score in hr}

        q_metrics = extractor.extract_metrics(s["question"])
        q_years = extractor.extract_years(s["question"])

        ppr_scores = ppr_rerank(
            graph, corpus_chunks, candidate_ids,
            seed_chunk_ids=candidate_ids[:10],
            seed_metric_names=list(q_metrics),
            seed_year_values=list(q_years),
            alpha=ppr_alpha,
            retrieval_scores=retrieval_scores,
        )
        ids = [cid for cid, _ in ppr_scores[:output_k]]
        results.append(_make_result(s, ids, gold_map.get(s["id"], []), "ppr"))
    return results


def run_gnn_reranker(
    samples: List[Dict],
    hybrid: HybridRetriever,
    graph,
    features: Dict[str, np.ndarray],
    gold_map: Dict[str, List[str]],
    chunk_by_id: Dict[str, Chunk],
    extractor: EntityExtractor,
    reranker,
    method_name: str,
    top_n: int = 50,
    output_k: int = 10,
    ppr_alpha: float = 0.85,
    use_ppr: bool = False,
    progress_every: int = 100,
    partial_output_dir: Optional[Path] = None,
) -> List[Dict]:
    """Run a trained GNN reranker on test samples."""
    results = []
    t_start = time.time()
    partial_path = None
    if partial_output_dir is not None:
        partial_path = partial_output_dir / f"{METHOD_FILES.get(method_name, method_name + '_results.jsonl')}.partial"
    if use_ppr:
        print(f"    [{method_name}] PPR auxiliary scores enabled for GNN evaluation")
    else:
        print(f"    [{method_name}] PPR auxiliary scores disabled for GNN evaluation")

    for idx, s in enumerate(samples, 1):
        hr = hybrid.search(s["question"], top_k=top_n)
        candidate_ids = [c.chunk_id for c, _ in hr]
        retrieval_scores = {c.chunk_id: float(score) for c, score in hr}

        ppr_scores: Dict[str, float] = {}
        if use_ppr:
            q_metrics = extractor.extract_metrics(s["question"])
            q_years = extractor.extract_years(s["question"])
            ppr_scores = dict(ppr_rerank(
                graph, [], candidate_ids,
                seed_chunk_ids=candidate_ids[:10],
                seed_metric_names=list(q_metrics),
                seed_year_values=list(q_years),
                alpha=ppr_alpha,
                retrieval_scores=retrieval_scores,
            ))

        try:
            reranked = reranker.rerank(
                s["question"], hr, graph, features,
                ppr_scores=ppr_scores,
            )
            ids = [c.chunk_id for c, _ in reranked[:output_k]]
        except Exception:
            ids = candidate_ids[:output_k]

        results.append(_make_result(s, ids, gold_map.get(s["id"], []), method_name))
        if progress_every > 0 and (idx % progress_every == 0 or idx == len(samples)):
            elapsed = time.time() - t_start
            rate = idx / max(elapsed, 1e-6)
            eta = (len(samples) - idx) / max(rate, 1e-6)
            print(
                f"    [{method_name}] eval {idx}/{len(samples)} "
                f"({idx / max(len(samples), 1):.1%}) elapsed={elapsed:.1f}s "
                f"eta={eta:.1f}s",
                flush=True,
            )
            if partial_path is not None:
                _write_jsonl(partial_path, results)
    return results


def run_rgcn_constraint(
    samples: List[Dict],
    hybrid: HybridRetriever,
    graph,
    features: Dict[str, np.ndarray],
    gold_map: Dict[str, List[str]],
    chunk_by_id: Dict[str, Chunk],
    extractor: EntityExtractor,
    rgcn_reranker,
    constraint_scorer: ConstraintScorer,
    cfg: Config,
    method_name: str = "rgcn_constraint",
    top_n: int = 50,
    output_k: int = 10,
    ppr_alpha: float = 0.85,
) -> List[Dict]:
    """R-GCN + Constraint Score reranking."""
    fusion = FusionScorer(
        alpha=cfg.rerank.get("fusion_alpha", 0.3),
        beta=cfg.rerank.get("fusion_beta", 0.3),
        gamma=cfg.rerank.get("fusion_gamma", 0.3),
        delta=cfg.rerank.get("fusion_delta", 0.1),
        constraint_scorer=constraint_scorer,
    )

    results = []
    for s in samples:
        hr = hybrid.search(s["question"], top_k=top_n)
        chunks = [c for c, _ in hr]
        candidate_ids = [c.chunk_id for c in chunks]
        retrieval_scores = {c.chunk_id: float(score) for c, score in hr}

        q_metrics = extractor.extract_metrics(s["question"])
        q_years = extractor.extract_years(s["question"])
        ppr_scores = dict(ppr_rerank(
            graph, [], candidate_ids,
            seed_chunk_ids=candidate_ids[:10],
            seed_metric_names=list(q_metrics),
            seed_year_values=list(q_years),
            alpha=ppr_alpha,
            retrieval_scores=retrieval_scores,
        ))

        # Get R-GCN scores
        try:
            gnn_reranked = rgcn_reranker.rerank(
                s["question"], hr, graph, features,
                ppr_scores=ppr_scores,
            )
            gnn_scores = {c.chunk_id: score for c, score in gnn_reranked}
        except Exception:
            gnn_scores = {}

        # Fuse with constraint
        fused = fusion.fuse(
            s["question"], chunks,
            retrieval_scores=retrieval_scores,
            graph_scores=ppr_scores,
            gnn_scores=gnn_scores,
        )
        ids = [c.chunk_id for c, _ in fused[:output_k]]
        results.append(_make_result(s, ids, gold_map.get(s["id"], []), method_name))
    return results


def run_qfe_rgcn(
    samples: List[Dict],
    hybrid: HybridRetriever,
    graph,
    features: Dict[str, np.ndarray],
    gold_map: Dict[str, List[str]],
    chunk_by_id: Dict[str, Chunk],
    extractor: EntityExtractor,
    qfe_reranker: QFERGCNFusionReranker,
    query_embeddings: Dict[str, np.ndarray],
    method_name: str = "qfe_rgcn",
    top_n: int = 50,
    output_k: int = 10,
    ppr_alpha: float = 0.85,
    *,
    use_ppr: bool = False,
    allow_fallback: bool = False,
    progress_every: int = 100,
    partial_output_dir: Optional[Path] = None,
    resume_rerank: bool = False,
    score_cache_path: Optional[Path] = None,
    checkpoint_path: Optional[str] = None,
) -> List[Dict]:
    """QFE-RGCN reranking with entity-gated evidence scoring.

    Uses the trainable EntityGatedScoringHead instead of hand-crafted linear
    fusion.  PPR is NOT computed by default — it is an independent baseline,
    not a required input for Final Graph (Ours).  Set ``use_ppr=True`` for
    ablation studies only.

    Args:
        use_ppr: If True, compute PPR scores and pass them as auxiliary
            graph features.  Default False — PPR is a separate baseline.
        allow_fallback: If True, fall back to candidate_ids[:output_k] when
            rerank raises an exception (with explicit logging).  Default
            False — fail fast so bugs are not silently hidden.
        progress_every: Print progress and write partial results every N
            newly-processed queries (0 to disable).
        partial_output_dir: If set, write partial JSONL and flush score cache
            at each progress checkpoint.
        resume_rerank: Skip queries already present in the partial JSONL.
        score_cache_path: Directory for persistent rerank-score cache.
        checkpoint_path: Path to the loaded checkpoint (used as part of the
            score-cache key to detect model changes).
    """
    results: List[Dict] = []
    completed_ids: Set[str] = set()

    # -- resolve partial path -------------------------------------------------
    partial_path: Optional[Path] = None
    if partial_output_dir is not None:
        partial_output_dir = Path(partial_output_dir)
        fname = METHOD_FILES.get(method_name, f"{method_name}_results.jsonl")
        partial_path = partial_output_dir / f"{fname}.partial"

    # -- resume from partial --------------------------------------------------
    if resume_rerank and partial_path is not None and partial_path.exists():
        with open(partial_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    results.append(r)
                    completed_ids.add(r["question_id"])
                except json.JSONDecodeError:
                    continue
        if results:
            print(
                f"    [{method_name}] Resumed {len(results)} completed queries "
                f"from {partial_path}"
            )

    # -- score cache ----------------------------------------------------------
    score_cache: Dict[str, Dict] = {}
    checkpoint_hash = ""
    qfe_config_hash = ""
    use_ppr_flag = use_ppr or method_name == "qfe_rgcn_ppr"

    if score_cache_path is not None:
        score_cache_dir = Path(score_cache_path)
        score_cache = _load_qfe_score_cache(score_cache_dir, method_name)
        if checkpoint_path is not None:
            checkpoint_hash = _compute_file_hash(Path(checkpoint_path))
        # Infer config hash from the loaded reranker
        gnn_hidden = int(qfe_reranker.model.conv1.self_loop.out_features)
        gnn_out_dim = int(qfe_reranker.model.out_dim)
        base_feat_dim = int(qfe_reranker.scoring_head.chunk_proj[0].in_features)
        qfe_config_hash = _compute_qfe_config_hash(
            query_embed_dim=qfe_reranker.query_embed_dim,
            relation_map=qfe_reranker.relation_map,
            gnn_hidden=gnn_hidden,
            gnn_out_dim=gnn_out_dim,
            base_feat_dim=base_feat_dim,
        )
        if score_cache:
            print(
                f"    [{method_name}] Loaded {len(score_cache)} cached scores "
                f"from {score_cache_dir}"
            )

    # -- main loop ------------------------------------------------------------
    t_start = time.time()
    new_count = 0
    cache_hits = 0
    cache_dirty = False

    for idx, s in enumerate(samples, 1):
        if s["id"] in completed_ids:
            continue

        hr = hybrid.search(s["question"], top_k=top_n)
        chunks = [c for c, _ in hr]
        candidate_ids = [c.chunk_id for c in chunks]

        # --- score cache lookup ----------------------------------------------
        cache_key: Optional[str] = None
        if score_cache_path is not None and checkpoint_hash:
            cache_key = _compute_qfe_score_cache_key(
                question_id=s["id"],
                candidate_ids=candidate_ids,
                checkpoint_hash=checkpoint_hash,
                method_name=method_name,
                use_ppr=use_ppr_flag,
                top_n=top_n,
                output_k=output_k,
                config_hash=qfe_config_hash,
            )
            if cache_key in score_cache:
                cached_entry = score_cache[cache_key]
                cached_ids = list(cached_entry.get("reranked_ids", []))[:output_k]
                results.append(
                    _make_result(s, cached_ids, gold_map.get(s["id"], []), method_name)
                )
                cache_hits += 1
                continue

        # --- PPR (only if explicitly requested) ------------------------------
        ppr_scores: Dict[str, float] = {}
        if use_ppr_flag:
            q_metrics = extractor.extract_metrics(s["question"])
            q_years = extractor.extract_years(s["question"])
            ppr_scores = dict(ppr_rerank(
                graph, [], candidate_ids,
                seed_chunk_ids=candidate_ids[:10],
                seed_metric_names=list(q_metrics),
                seed_year_values=list(q_years),
                alpha=ppr_alpha,
                retrieval_scores={c.chunk_id: float(score) for c, score in hr},
            ))

        # --- query embedding -------------------------------------------------
        if s["question"] not in qfe_reranker.query_embeddings:
            qfe_reranker.query_embeddings[s["question"]] = derive_query_vector(
                s["question"], dim=qfe_reranker.query_embed_dim,
            )

        # --- rerank ----------------------------------------------------------
        if allow_fallback:
            try:
                reranked = qfe_reranker.rerank(
                    s["question"], hr, graph, features,
                    ppr_scores=ppr_scores,
                )
                ids = [c.chunk_id for c, _ in reranked[:output_k]]
                scores = [float(score) for _, score in reranked[:output_k]]
            except Exception as exc:
                print(
                    f"    [{method_name} FALLBACK] query_id={s['id']} "
                    f"exception={type(exc).__name__}: {exc}"
                )
                ids = candidate_ids[:output_k]
                scores = [0.0] * len(ids)
                r = _make_result(s, ids, gold_map.get(s["id"], []), method_name)
                r["_fallback"] = True
                r["_fallback_exception"] = f"{type(exc).__name__}: {exc}"
                results.append(r)
                new_count += 1
                continue
        else:
            reranked = qfe_reranker.rerank(
                s["question"], hr, graph, features,
                ppr_scores=ppr_scores,
            )
            ids = [c.chunk_id for c, _ in reranked[:output_k]]
            scores = [float(score) for _, score in reranked[:output_k]]

        results.append(
            _make_result(s, ids, gold_map.get(s["id"], []), method_name)
        )
        new_count += 1

        # --- persist to score cache ------------------------------------------
        if cache_key is not None and score_cache_path is not None:
            score_cache[cache_key] = {
                "cache_key": cache_key,
                "question_id": s["id"],
                "reranked_ids": ids,
                "reranked_scores": scores,
            }
            cache_dirty = True

        # --- progress logging ------------------------------------------------
        total_done = len(results)
        if progress_every > 0 and (
            new_count % progress_every == 0
            or total_done == len(samples)
        ):
            elapsed = time.time() - t_start
            rate = new_count / max(elapsed, 1e-6)
            remaining = len(samples) - total_done
            eta = remaining / max(rate, 1e-6)
            pct = total_done / max(len(samples), 1) * 100
            print(
                f"    [{method_name}] eval {total_done}/{len(samples)} "
                f"({pct:.1f}%) elapsed={elapsed:.1f}s eta={eta:.1f}s",
                flush=True,
            )

        # --- partial checkpoint ----------------------------------------------
        if (
            partial_path is not None
            and progress_every > 0
            and new_count % progress_every == 0
        ):
            _write_jsonl(partial_path, results)
            if cache_dirty and score_cache_path is not None:
                _save_qfe_score_cache(
                    Path(score_cache_path), method_name, score_cache
                )
                cache_dirty = False

    # -- final flush ----------------------------------------------------------
    total_elapsed = time.time() - t_start
    if partial_path is not None:
        _write_jsonl(partial_path, results)
    if cache_dirty and score_cache_path is not None:
        _save_qfe_score_cache(Path(score_cache_path), method_name, score_cache)

    print(
        f"    [{method_name}] Evaluation done in {total_elapsed:.1f}s "
        f"(cache hits: {cache_hits}, computed: {new_count})"
    )

    return results


def run_mono_t5(
    samples: List[Dict],
    chunk_by_id: Dict[str, Chunk],
    candidate_pool: Dict[str, List[str]],
    gold_map: Dict[str, List[str]],
    *,
    model_name_or_path: str = "castorini/monot5-base-msmarco",
    batch_size: int = 8,
    max_length: int = 512,
    device: str = "cuda",
    use_fp16: bool = False,
    top_n: int = 50,
    output_k: int = 10,
    score_cache_dir: Optional[Path] = None,
    rebuild_score_cache: bool = False,
    resume_rerank: bool = False,
    candidate_pool_name: str = "BGE-M3-Dense",
    candidate_results_jsonl: str = "",
    corpus_cache: str = "",
    allow_fallback: bool = False,
    checkpoint_every: int = 100,
    output_dir: Optional[Path] = None,
) -> List[Dict]:
    """Run MonoT5 reranking on the fixed BGE-M3 candidate pool.

    Thin wrapper around :func:`feg_rag.rerank.mono_t5.run_mono_t5_reranking`
    that adds partial-JSONL writing at the experiment level.
    """
    partial_path = None
    if output_dir is not None:
        partial_path = output_dir / "mono_t5_results.jsonl.partial"

    results = run_mono_t5_reranking(
        samples=samples,
        chunk_by_id=chunk_by_id,
        candidate_pool=candidate_pool,
        gold_map=gold_map,
        model_name_or_path=model_name_or_path,
        batch_size=batch_size,
        max_length=max_length,
        device=device,
        use_fp16=use_fp16,
        top_n=top_n,
        output_k=output_k,
        score_cache_dir=score_cache_dir,
        rebuild_score_cache=rebuild_score_cache,
        resume_rerank=resume_rerank,
        candidate_pool_name=candidate_pool_name,
        candidate_results_jsonl=candidate_results_jsonl,
        corpus_cache=corpus_cache,
        allow_fallback=allow_fallback,
        checkpoint_every=checkpoint_every,
        partial_output_path=partial_path,
    )

    # Write final JSONL (no .partial suffix)
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        final_path = output_dir / "mono_t5_results.jsonl"
        with open(final_path, "w", encoding="utf-8") as fh:
            for r in results:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        # Remove partial if final is written
        if partial_path and partial_path.exists():
            partial_path.unlink(missing_ok=True)

    return results


def run_list_t5(
    samples: List[Dict],
    chunk_by_id: Dict[str, Chunk],
    candidate_pool: Dict[str, List[str]],
    gold_map: Dict[str, List[str]],
    *,
    model_name_or_path: str = "Soyoung97/ListT5-base",
    batch_size: int = 8,
    max_length: int = 128,
    listwise_k: int = 5,
    out_k: int = 2,
    device: str = "cuda",
    use_fp16: bool = False,
    top_n: int = 50,
    output_k: int = 10,
    decision_cache_dir: Optional[Path] = None,
    rebuild_decision_cache: bool = False,
    resume_rerank: bool = False,
    candidate_pool_name: str = "BGE-M3-Dense",
    candidate_results_jsonl: str = "",
    corpus_cache: str = "",
    allow_fallback: bool = False,
    checkpoint_every: int = 100,
    output_dir: Optional[Path] = None,
) -> List[Dict]:
    """Run ListT5 reranking on the fixed BGE-M3 candidate pool."""
    partial_path = None
    if output_dir is not None:
        partial_path = output_dir / "list_t5_results.jsonl.partial"

    results = run_list_t5_reranking(
        samples=samples,
        chunk_by_id=chunk_by_id,
        candidate_pool=candidate_pool,
        gold_map=gold_map,
        model_name_or_path=model_name_or_path,
        batch_size=batch_size,
        max_length=max_length,
        listwise_k=listwise_k,
        out_k=out_k,
        device=device,
        use_fp16=use_fp16,
        top_n=top_n,
        output_k=output_k,
        decision_cache_dir=decision_cache_dir,
        rebuild_decision_cache=rebuild_decision_cache,
        resume_rerank=resume_rerank,
        candidate_pool_name=candidate_pool_name,
        candidate_results_jsonl=candidate_results_jsonl,
        corpus_cache=corpus_cache,
        allow_fallback=allow_fallback,
        checkpoint_every=checkpoint_every,
        partial_output_path=partial_path,
    )

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        final_path = output_dir / "list_t5_results.jsonl"
        with open(final_path, "w", encoding="utf-8") as fh:
            for r in results:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        if partial_path and partial_path.exists():
            partial_path.unlink(missing_ok=True)

    return results


def run_fast_final_graph_method(
    train_samples: List[Dict],
    eval_samples: List[Dict],
    gold_map: Dict[str, List[str]],
    chunk_by_id: Dict[str, Chunk],
    bge_results_jsonl: str | Path,
    rgcn_results_jsonl: str | Path,
    monot5_results_jsonl: str | Path,
    output_dir: Path,
    *,
    ppr_results_jsonl: Optional[str | Path] = None,
    model_cache_path: Optional[str | Path] = None,
    epochs: int = 20,
    batch_size: int = 512,
    lr: float = 1e-3,
    min_rgcn_weight: float = 0.35,
    min_bge_weight: float = 0.15,
    max_entity_weight: float = 0.10,
    delta_scale: float = 0.05,
    hard_negatives: int = 10,
    top_n: int = 50,
    output_k: int = 10,
    device: str = "cpu",
    val_split: float = 0.2,
    split_seed: int = 42,
    load_checkpoint_path: Optional[str] = None,
    progress_every: int = 100,
    eval_only: bool = False,
) -> Tuple[List[Dict], Optional[QueryAdaptiveFusionReranker], Dict]:
    """Train (or load) and evaluate the Fast Final Graph fusion reranker.

    Returns
    -------
    results : list[dict]
        Standard per-query result dicts.
    model : QueryAdaptiveFusionReranker or None
    meta : dict
        Training metadata including learned weights.
    """
    import random

    t_total = time.time()
    meta: Dict = {
        "method": "fast_final_graph",
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "min_rgcn_weight": min_rgcn_weight,
        "min_bge_weight": min_bge_weight,
        "max_entity_weight": max_entity_weight,
        "delta_scale": delta_scale,
        "hard_negatives": hard_negatives,
    }

    # ---- 1. Load external results ----
    print(f"\n  [fast_final_graph] Loading external results...")
    bge_results = load_jsonl_results(bge_results_jsonl)
    rgcn_results = load_jsonl_results(rgcn_results_jsonl)
    monot5_results = load_jsonl_results(monot5_results_jsonl)
    print(f"    BGE queries: {len(bge_results)}")
    print(f"    R-GCN queries: {len(rgcn_results)}")
    print(f"    MonoT5 queries: {len(monot5_results)}")

    # ---- 1b. Optionally load PPR ----
    ppr_results: Optional[Dict[str, Dict]] = None
    if ppr_results_jsonl:
        ppr_results = load_jsonl_results(ppr_results_jsonl)
        print(f"    [fast_final_graph] PPR/graph source: loaded {len(ppr_results)} queries")
    else:
        print(f"    [fast_final_graph] PPR/graph source: disabled")

    # ---- 2. Build model ----
    model = QueryAdaptiveFusionReranker(
        min_rgcn_weight=min_rgcn_weight,
        min_bge_weight=min_bge_weight,
        max_entity_weight=max_entity_weight,
        delta_scale=delta_scale,
        use_delta_mlp=True,
        dropout=0.1,
    )

    if load_checkpoint_path:
        # eval-only path
        print(f"\n  [fast_final_graph] Loading checkpoint: {load_checkpoint_path}")
        model, ckpt_meta = load_checkpoint(load_checkpoint_path, device=device)
        meta["checkpoint_loaded"] = str(load_checkpoint_path)
        meta.update({f"ckpt_{k}": v for k, v in ckpt_meta.items() if isinstance(v, (int, float, str, bool))})
        print(f"    Loaded in {time.time() - t_total:.1f}s")
    else:
        if eval_only:
            raise ValueError(
                "--eval_only_checkpoint requested but "
                "--load_fast_graph_checkpoint was not provided. "
                "Pass --load_fast_graph_checkpoint PATH to load a saved checkpoint."
            )

        # ---- 3. Build or load cached train / val datasets ----
        print(f"\n  [fast_final_graph] Building training dataset...")
        t0 = time.time()

        # Split train_samples into train / val for early stopping
        rng = random.Random(split_seed)
        shuffled = list(train_samples)
        rng.shuffle(shuffled)
        n_val = max(1, int(len(shuffled) * val_split))
        train_subset = shuffled[n_val:]
        val_subset = shuffled[:n_val]

        # Compute config fingerprint (shared for build & cache).
        # Include the concrete split IDs so smoke-test caches cannot be reused
        # accidentally for full runs with the same input result files.
        fingerprint = compute_config_fingerprint(
            bge_results_jsonl=bge_results_jsonl,
            rgcn_results_jsonl=rgcn_results_jsonl,
            monot5_results_jsonl=monot5_results_jsonl,
            ppr_results_jsonl=ppr_results_jsonl,
            train_sample_ids=[s["id"] for s in train_subset],
            val_sample_ids=[s["id"] for s in val_subset],
            top_n=top_n,
            hard_negatives=hard_negatives,
            min_rgcn_weight=min_rgcn_weight,
            min_bge_weight=min_bge_weight,
            split_seed=split_seed,
            val_split=val_split,
        )
        print(f"    Config fingerprint: {fingerprint}")

        # Try loading from cache
        train_dataset: FastFinalGraphDataset
        val_dataset: Optional[FastFinalGraphDataset] = None
        cache_hit = False

        if model_cache_path:
            cache_path = Path(model_cache_path)
            if not cache_path.is_absolute():
                cache_path = output_dir / cache_path
            cached_train, cached_val, cache_status = load_dataset_cache(
                cache_path, fingerprint,
            )
            if cache_status == "ok" and cached_train is not None:
                train_dataset = cached_train
                val_dataset = cached_val
                cache_hit = True
                print(f"    [fast_final_graph] Dataset cache HIT: {cache_path}")
                print(f"    Train pairs: {train_dataset.num_pairs} "
                      f"(from {len(train_subset)} queries)")
                if val_dataset is not None:
                    print(f"    Val pairs:   {val_dataset.num_pairs} "
                          f"(from {len(val_subset)} queries)")
                print(f"    Dataset load time: {time.time() - t0:.1f}s")
            else:
                print(f"    [fast_final_graph] Dataset cache MISS: {cache_status}")
                if cache_status.startswith("fingerprint mismatch"):
                    print(f"    [fast_final_graph] Cached={cache_status.split('cached=')[1].split(' ')[0] if 'cached=' in cache_status else '?'}")
                    print(f"    [fast_final_graph] Expected={fingerprint}")

        if not cache_hit:
            train_dataset = FastFinalGraphDataset(
                train_subset, gold_map, chunk_by_id,
                bge_results, rgcn_results, monot5_results,
                ppr_results=ppr_results,
                hard_negatives=hard_negatives, top_n=top_n,
            )
            print(f"    Train pairs: {train_dataset.num_pairs} "
                  f"(from {len(train_subset)} queries)")

            if len(val_subset) > 0:
                try:
                    val_dataset = FastFinalGraphDataset(
                        val_subset, gold_map, chunk_by_id,
                        bge_results, rgcn_results, monot5_results,
                        ppr_results=ppr_results,
                        hard_negatives=hard_negatives, top_n=top_n,
                    )
                    print(f"    Val pairs:   {val_dataset.num_pairs} "
                          f"(from {len(val_subset)} queries)")
                except ValueError:
                    print(f"    [fast_final_graph] No val pairs (all queries lack "
                          f"gold in candidate pool); skipping validation monitoring.")

            print(f"    Dataset build time: {time.time() - t0:.1f}s")

            # Save to cache
            if model_cache_path:
                cache_path = Path(model_cache_path)
                if not cache_path.is_absolute():
                    cache_path = output_dir / cache_path
                try:
                    save_dataset_cache(
                        cache_path, train_dataset, val_dataset,
                        fingerprint, meta={"train_subset_size": len(train_subset)},
                    )
                    print(f"    [fast_final_graph] Dataset cache saved: {cache_path}")
                except Exception as exc:
                    print(f"    [fast_final_graph] WARNING: Failed to save dataset cache: {exc}")

        # ---- 4. Train ----
        print(f"\n  [fast_final_graph] Training...")
        print(f"    Epochs: {epochs}, batch_size: {batch_size}, lr: {lr}")
        print(f"    min_rgcn_weight: {min_rgcn_weight}, "
              f"min_bge_weight: {min_bge_weight}")
        print(f"    max_entity_weight: {max_entity_weight}, "
              f"delta_scale: {delta_scale}")
        t0 = time.time()

        train_losses, val_losses = train_fast_final_graph(
            model, train_dataset, val_dataset,
            epochs=epochs, batch_size=batch_size, lr=lr,
            device=device, verbose=True,
        )
        train_time = time.time() - t0
        print(f"    Training done in {train_time:.1f}s "
              f"({train_time / 60:.1f}m)")

        meta["train_time_s"] = round(train_time, 1)
        meta["final_train_loss"] = round(train_losses[-1], 6) if train_losses else None
        if val_losses:
            meta["final_val_loss"] = round(val_losses[-1], 6)

        # ---- 5. Compute average learned weights ----
        print(f"\n  [fast_final_graph] Computing average learned weights...")
        from torch.utils.data import DataLoader as _FFGDataLoader
        train_loader = _FFGDataLoader(
            train_dataset, batch_size=batch_size, shuffle=False,
        )
        avg_weights = model.compute_average_weights(train_loader, device=device)
        for k, v in avg_weights.items():
            print(f"    {k}: {v:.4f}")
        meta.update(avg_weights)

    # ---- 6. Evaluate ----
    print(f"\n  [fast_final_graph] Evaluating on {len(eval_samples)} samples...")
    t0 = time.time()

    partial_path = output_dir / "fast_final_graph_results.jsonl.partial"

    results = evaluate_fast_final_graph(
        model, eval_samples, gold_map, chunk_by_id,
        bge_results, rgcn_results, monot5_results,
        ppr_results=ppr_results,
        top_n=top_n, output_k=output_k, device=device,
        batch_size=batch_size, progress_every=progress_every,
        partial_output_path=partial_path,
    )
    eval_time = time.time() - t0
    print(f"    Evaluation done in {eval_time:.1f}s ({eval_time / 60:.1f}m)")
    meta["eval_time_s"] = round(eval_time, 1)

    total_time = time.time() - t_total
    print(f"    Total time: {total_time:.1f}s ({total_time / 60:.1f}m)")
    meta["total_time_s"] = round(total_time, 1)

    return results, model, meta


# ═════════════════════════════════════════════════════════════════════════════
# Output
# ═════════════════════════════════════════════════════════════════════════════

def _write_outputs(
    output_dir: Path,
    all_results: Dict[str, List[Dict]],
    summaries: Dict[str, Dict],
    k_values: List[int],
    command: str = "",
) -> None:
    """Write all Table 1 output files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Per-method JSONL
    for method, results in all_results.items():
        fname = METHOD_FILES.get(method, f"{method}_results.jsonl")
        with open(output_dir / fname, "w", encoding="utf-8") as fh:
            for r in results:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Full metrics JSON
    with open(output_dir / "metrics_full.json", "w", encoding="utf-8") as fh:
        json.dump(summaries, fh, indent=2, ensure_ascii=False)

    # Comparison CSV
    csv_path = output_dir / "table1_non_llm_reranking_comparison.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as fh:
        fieldnames = ["Method"] + [f"Recall@{k}" for k in k_values] + ["MRR"] + [f"nDCG@{k}" for k in k_values] + ["num_samples"]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for method in METHOD_ORDER:
            m = summaries.get(method)
            if m is None:
                continue
            row = {"Method": METHOD_LABELS.get(method, method)}
            for k in k_values:
                row[f"Recall@{k}"] = round(m.get(f"recall@{k}", 0), 4)
            row["MRR"] = round(m.get("mrr", 0), 4)
            for k in k_values:
                row[f"nDCG@{k}"] = round(m.get(f"ndcg@{k}", 0), 4)
            row["num_samples"] = m.get("num_samples", 0)
            writer.writerow(row)

    # Comparison Markdown
    md_path = output_dir / "table1_non_llm_reranking_comparison.md"
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write("# Table 1: Non-LLM Reranking Comparison\n\n")
        fh.write(f"Generated: {datetime.now().isoformat()}\n\n")
        fh.write("## Run command\n\n```bash\n")
        fh.write(f"{command}\n```\n\n")
        fh.write("## Results\n\n")
        header = "| Method |"
        sep = "|---|"
        for k in k_values:
            header += f" Recall@{k} |"
            sep += "---|"
        header += " MRR |"
        sep += "---|"
        for k in k_values:
            header += f" nDCG@{k} |"
            sep += "---|"
        fh.write(header + "\n")
        fh.write(sep + "\n")
        for method in METHOD_ORDER:
            m = summaries.get(method)
            if m is None:
                continue
            row = f"| {METHOD_LABELS.get(method, method)} |"
            for k in k_values:
                row += f" {m.get(f'recall@{k}', 0):.4f} |"
            row += f" {m.get('mrr', 0):.4f} |"
            for k in k_values:
                row += f" {m.get(f'ndcg@{k}', 0):.4f} |"
            fh.write(row + "\n")
        fh.write("\n## Key claim\n\n")
        fh.write("> Graph-based reranking with query-aware relation gates (QFE-RGCN) improves evidence ranking without relying on an LLM reranker.\n\n")

    # README
    readme_path = output_dir / "README.md"
    with open(readme_path, "w", encoding="utf-8") as fh:
        fh.write("# Experiment: Table 1 — Non-LLM Reranking Comparison\n\n")
        fh.write("Compares evidence ranking performance of retrieval + reranking methods.\n\n")
        fh.write("## Methods\n\n")
        for method in METHOD_ORDER:
            label = METHOD_LABELS.get(method, method)
            fh.write(f"- **{label}**\n")
        fh.write("\n## Output files\n\n")
        for fname in [
            "table1_non_llm_reranking_comparison.csv",
            "table1_non_llm_reranking_comparison.md",
        ]:
            fh.write(f"- `{fname}`\n")
        for fname in METHOD_FILES.values():
            fh.write(f"- `{fname}`\n")
        fh.write("- `metrics_full.json`\n")
        fh.write("- `README.md`\n")
        fh.write(f"\nGenerated: {datetime.now().isoformat()}\n")


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def _write_method_results(output_dir: Path, method: str, results: List[Dict]) -> Path:
    """Persist one method immediately so an interrupted long run keeps results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fname = METHOD_FILES.get(method, f"{method}_results.jsonl")
    path = output_dir / fname
    with open(path, "w", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path


def _summarize_results(
    all_results: Dict[str, List[Dict]],
    k_values: List[int],
) -> Dict[str, Dict]:
    summaries: Dict[str, Dict] = {}
    for method, results in all_results.items():
        er = compute_all_metrics(method, results, k_values=k_values)
        summaries[method] = {
            "method": method,
            "num_samples": er.num_samples,
            "mrr": round(er.mrr, 4),
        }
        for k in k_values:
            summaries[method][f"recall@{k}"] = round(er.evidence_recall.get(k, 0), 4)
            summaries[method][f"ndcg@{k}"] = round(er.ndcg.get(k, 0), 4)
    return summaries


def _write_progress_outputs(
    output_dir: Path,
    all_results: Dict[str, List[Dict]],
    k_values: List[int],
) -> None:
    """Write partial metrics/CSV after every completed method."""
    summaries = _summarize_results(all_results, k_values)
    with open(output_dir / "metrics_partial.json", "w", encoding="utf-8") as fh:
        json.dump(summaries, fh, indent=2, ensure_ascii=False)

    csv_path = output_dir / "table1_non_llm_reranking_comparison.partial.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as fh:
        fieldnames = (
            ["Method"]
            + [f"Recall@{k}" for k in k_values]
            + ["MRR"]
            + [f"nDCG@{k}" for k in k_values]
            + ["num_samples"]
        )
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for method in METHOD_ORDER:
            m = summaries.get(method)
            if m is None:
                continue
            row = {"Method": METHOD_LABELS.get(method, method)}
            for k in k_values:
                row[f"Recall@{k}"] = round(m.get(f"recall@{k}", 0), 4)
            row["MRR"] = round(m.get("mrr", 0), 4)
            for k in k_values:
                row[f"nDCG@{k}"] = round(m.get(f"ndcg@{k}", 0), 4)
            row["num_samples"] = m.get("num_samples", 0)
            writer.writerow(row)


def _checkpoint_method(
    output_dir: Path,
    method: str,
    all_results: Dict[str, List[Dict]],
    k_values: List[int],
) -> None:
    _write_method_results(output_dir, method, all_results[method])
    _write_progress_outputs(output_dir, all_results, k_values)
    print(f"    [checkpoint] Wrote partial outputs through '{method}'")


def _load_graphsage_checkpoint(
    checkpoint_path: str | Path,
    *,
    base_feature_dim: int,
    cfg: Config,
    device: str,
) -> GNNFusionReranker:
    model = GraphSAGEReranker(
        in_dim=base_feature_dim + QUERY_FEATURE_DIM,
        hidden_dim=cfg.rerank["gnn_hidden"],
        dropout=cfg.rerank["gnn_dropout"],
    )
    return GNNFusionReranker.load(checkpoint_path, model=model, device=device)


def _load_rgcn_checkpoint(
    checkpoint_path: str | Path,
    *,
    base_feature_dim: int,
    cfg: Config,
    device: str,
) -> RGCNFusionReranker:
    import torch

    ckpt = torch.load(checkpoint_path, map_location=device)
    relation_map = ckpt.get("relation_map", {})
    num_relations = max(relation_map.values()) + 1 if relation_map else 1
    model = RGCNReranker(
        in_dim=base_feature_dim + QUERY_FEATURE_DIM,
        hidden_dim=cfg.rerank["gnn_hidden"],
        num_relations=num_relations,
        dropout=cfg.rerank["gnn_dropout"],
    )
    return RGCNFusionReranker.load(checkpoint_path, model=model, device=device)


# ═════════════════════════════════════════════════════════════════════════════
# QFE-RGCN checkpoint loading & score cache helpers
# ═════════════════════════════════════════════════════════════════════════════

def _compute_file_hash(file_path: Path) -> str:
    """Compute a stable hash of a file for cache identification.

    Hashes the first 64 KiB of content plus file mtime and size so that
    the same checkpoint file always maps to the same hash.
    """
    stat = file_path.stat()
    h = hashlib.sha256()
    with open(file_path, "rb") as fh:
        h.update(fh.read(65536))
    h.update(str(stat.st_mtime).encode())
    h.update(str(stat.st_size).encode())
    return h.hexdigest()[:16]


def _compute_qfe_config_hash(
    query_embed_dim: int,
    relation_map: Dict[str, int],
    gnn_hidden: int,
    gnn_out_dim: int,
    base_feat_dim: int,
) -> str:
    """Hash the QFE model architecture configuration for cache key derivation."""
    parts = [
        str(query_embed_dim),
        json.dumps(dict(sorted(relation_map.items())), sort_keys=True),
        str(gnn_hidden),
        str(gnn_out_dim),
        str(base_feat_dim),
    ]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def _compute_qfe_score_cache_key(
    question_id: str,
    candidate_ids: List[str],
    checkpoint_hash: str,
    method_name: str,
    use_ppr: bool,
    top_n: int,
    output_k: int,
    config_hash: str,
) -> str:
    """Derive a deterministic cache key for a single query's QFE rerank result."""
    parts = [
        question_id,
        _fingerprint_ids(candidate_ids),
        checkpoint_hash,
        method_name,
        str(int(use_ppr)),
        str(top_n),
        str(output_k),
        config_hash,
    ]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:24]


def _load_qfe_score_cache(cache_dir: Path, method_name: str) -> Dict[str, Dict]:
    """Load QFE score cache from a JSON file.  Returns dict keyed by cache_key."""
    cache_file = cache_dir / f"{method_name}.json"
    if not cache_file.exists():
        return {}
    try:
        with open(cache_file, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, IOError):
        return {}


def _save_qfe_score_cache(cache_dir: Path, method_name: str, cache: Dict[str, Dict]) -> None:
    """Atomically write QFE score cache to disk."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{method_name}.json"
    tmp_file = cache_dir / f"{method_name}.json.tmp"
    with open(tmp_file, "w", encoding="utf-8") as fh:
        json.dump(cache, fh, ensure_ascii=False)
    tmp_file.replace(cache_file)


def _load_qfe_reranker_from_checkpoint(
    checkpoint_path: str | Path,
    feature_dim: int,
    cfg: Config,
    device: str,
) -> Tuple[QFERGCNFusionReranker, Dict]:
    """Load a QFE-RGCN reranker from a saved checkpoint.

    Infers model/scoring-head dimensions from the stored state dicts so the
    caller does not need to track architecture hyperparameters separately.
    """
    import torch

    ckpt = torch.load(checkpoint_path, map_location=device)
    model_state = ckpt["model_state"]
    scoring_head_state = ckpt["scoring_head_state"]

    # Infer dimensions from saved parameter shapes
    in_dim: int = model_state["conv1.self_loop.weight"].shape[1]
    hidden_dim: int = model_state["conv1.self_loop.weight"].shape[0]
    out_dim: int = model_state["conv2.self_loop.weight"].shape[0]
    base_feat_dim: int = scoring_head_state["chunk_proj.0.weight"].shape[1]

    # Validate that checkpoint in_dim matches current feature_dim
    expected_in_dim = feature_dim + QUERY_FEATURE_DIM
    if in_dim != expected_in_dim:
        raise ValueError(
            f"QFE checkpoint in_dim mismatch: "
            f"checkpoint in_dim={in_dim}, "
            f"current feature_dim={feature_dim}, "
            f"QUERY_FEATURE_DIM={QUERY_FEATURE_DIM}, "
            f"expected_in_dim={expected_in_dim}. "
            f"The graph feature cache / embedding dim does not match the "
            f"checkpoint. Rebuild with matching --graph_feature_retriever_cache "
            f"or re-train."
        )

    relation_map: Dict[str, int] = ckpt.get("relation_map", {})
    num_relations = max(relation_map.values()) + 1 if relation_map else 1
    query_embed_dim: int = ckpt.get("query_embed_dim", QUERY_EMBED_DIM)

    # Build model architecture matching the checkpoint
    gnn_model = QFERGCNReranker(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        out_dim=out_dim,
        num_relations=num_relations,
        query_embed_dim=query_embed_dim,
        dropout=cfg.rerank.get("gnn_dropout", 0.3),
    )
    scoring_head = EntityGatedScoringHead(
        query_embed_dim=query_embed_dim,
        chunk_proj_dim=64,
        gnn_out_dim=out_dim,
        base_feat_dim=base_feat_dim,
        hidden_dim=hidden_dim,
        dropout=cfg.rerank.get("gnn_dropout", 0.3),
    )

    # Restore parameters
    gnn_model.load_state_dict(model_state)
    scoring_head.load_state_dict(scoring_head_state)

    query_embeddings: Dict[str, np.ndarray] = {}

    reranker = QFERGCNFusionReranker(
        model=gnn_model,
        scoring_head=scoring_head,
        relation_map=relation_map,
        query_embeddings=query_embeddings,
        query_embed_dim=query_embed_dim,
        device=device,
    )

    meta = {
        "checkpoint_path": str(checkpoint_path),
        "in_dim": in_dim,
        "hidden_dim": hidden_dim,
        "out_dim": out_dim,
        "base_feat_dim": base_feat_dim,
        "num_relations": num_relations,
        "query_embed_dim": query_embed_dim,
    }
    return reranker, meta


def _load_qfe_v2_reranker_from_checkpoint(
    checkpoint_path: str | Path,
    feature_dim: int,
    cfg: Config,
    device: str,
) -> Tuple[QFERGCNFusionRerankerV2, Dict]:
    """Load a QFE-RGCN v2 reranker from a saved checkpoint.

    Infers dimensions from stored state dicts and reconstructs the full
    v2 architecture (QFERGCNReranker + EntityGatedScoringHead +
    RetrievalPreservedFusionHead + gnn_proj).
    """
    import torch

    ckpt = torch.load(checkpoint_path, map_location=device)
    version = ckpt.get("version", 1)
    if version != 2:
        raise ValueError(
            f"Expected v2 checkpoint (version=2), got version={version}. "
            f"Use --load_qfe_checkpoint with a v1 checkpoint for qfe_rgcn."
        )

    model_state = ckpt["model_state"]
    scoring_head_state = ckpt["scoring_head_state"]
    fusion_head_state = ckpt["fusion_head_state"]
    gnn_proj_state = ckpt["gnn_proj_state"]

    # Infer dimensions
    in_dim: int = model_state["conv1.self_loop.weight"].shape[1]
    hidden_dim: int = model_state["conv1.self_loop.weight"].shape[0]
    out_dim: int = model_state["conv2.self_loop.weight"].shape[0]
    base_feat_dim: int = scoring_head_state["chunk_proj.0.weight"].shape[1]

    # Validate dimension match
    expected_in_dim = feature_dim + QUERY_FEATURE_DIM
    if in_dim != expected_in_dim:
        raise ValueError(
            f"QFE v2 checkpoint in_dim mismatch: "
            f"checkpoint in_dim={in_dim}, "
            f"current feature_dim={feature_dim}, "
            f"QUERY_FEATURE_DIM={QUERY_FEATURE_DIM}, "
            f"expected_in_dim={expected_in_dim}."
        )

    relation_map: Dict[str, int] = ckpt.get("relation_map", {})
    num_relations = max(relation_map.values()) + 1 if relation_map else 1
    query_embed_dim: int = ckpt.get("query_embed_dim", QUERY_EMBED_DIM)
    min_ret_weight: float = ckpt.get("min_retrieval_weight", 0.35)

    # Build architecture
    gnn_model = QFERGCNReranker(
        in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim,
        num_relations=num_relations, query_embed_dim=query_embed_dim,
        dropout=cfg.rerank.get("gnn_dropout", 0.3),
    )
    scoring_head = EntityGatedScoringHead(
        query_embed_dim=query_embed_dim, chunk_proj_dim=64,
        gnn_out_dim=out_dim, base_feat_dim=base_feat_dim,
        hidden_dim=hidden_dim, dropout=cfg.rerank.get("gnn_dropout", 0.3),
    )
    fusion_head = RetrievalPreservedFusionHead(
        query_embed_dim=query_embed_dim, hidden_dim=64,
        min_ret_weight=min_ret_weight, dropout=0.1,
    )
    gnn_proj = nn.Linear(out_dim, 1)

    # Restore parameters
    gnn_model.load_state_dict(model_state)
    scoring_head.load_state_dict(scoring_head_state)
    fusion_head.load_state_dict(fusion_head_state)
    gnn_proj.load_state_dict(gnn_proj_state)

    # Optional BGE projection
    query_projection = None
    if "query_projection_state" in ckpt:
        proj_state = ckpt["query_projection_state"]
        bge_dim = proj_state["weight"].shape[1]
        query_projection = nn.Linear(bge_dim, query_embed_dim).to(device)
        query_projection.load_state_dict(proj_state)

    query_embeddings: Dict[str, np.ndarray] = {}

    reranker = QFERGCNFusionRerankerV2(
        model=gnn_model, scoring_head=scoring_head,
        fusion_head=fusion_head, gnn_proj=gnn_proj,
        relation_map=relation_map, query_embeddings=query_embeddings,
        query_embed_dim=query_embed_dim,
        min_retrieval_weight=min_ret_weight, delta_reg=0.05,
        device=device, query_projection=query_projection,
        query_encoder=ckpt.get("query_encoder", "heuristic"),
        query_embedding_dim_raw=ckpt.get("query_embedding_dim_raw"),
    )

    meta = {
        "checkpoint_path": str(checkpoint_path), "version": 2,
        "in_dim": in_dim, "hidden_dim": hidden_dim, "out_dim": out_dim,
        "base_feat_dim": base_feat_dim, "num_relations": num_relations,
        "query_embed_dim": query_embed_dim,
        "min_retrieval_weight": min_ret_weight,
    }
    return reranker, meta


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Table 1: Non-LLM Reranking Comparison"
    )
    parser.add_argument("--config", default="configs/table1_non_llm_reranking_e5_mistral.yaml")
    parser.add_argument("--output_dir", default=None,
                        help="Output directory (default from config)")
    parser.add_argument("--num_samples", type=int, default=0,
                        help="Limit samples (0=all FinDER)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override GNN training epochs")
    parser.add_argument("--device", default=_default_device(),
                        help="Device for GNN training (cuda/cpu)")
    parser.add_argument("--dense_device", default="cpu",
                        help="Device for dense encoding")
    parser.add_argument("--dense_batch_size", type=int, default=None,
                        help="Batch size for dense encoding")
    parser.add_argument("--top_n", type=int, default=50,
                        help="Candidates from initial retrieval")
    parser.add_argument("--output_k", type=int, default=10,
                        help="Top-k evidence to evaluate")
    parser.add_argument("--val_split", type=float, default=0.2,
                        help="Train/val split ratio")
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--skip_gnn", action="store_true",
                        help="Skip GNN training (GraphSAGE, R-GCN, R-GCN+Constraint)")
    parser.add_argument("--corpus_cache", default=None,
                        help="Pickle cache for full benchmark corpus/gold_map")
    parser.add_argument("--rebuild_corpus_cache", action="store_true",
                        help="Rebuild corpus cache even if it exists")
    parser.add_argument("--allow_gold_only_corpus", action="store_true",
                        help="Debug only: allow corpus built from gold evidence snippets")
    parser.add_argument("--graph_cache", default=None,
                        help="Pickle cache for entity_map, graph, and node features")
    parser.add_argument("--rebuild_graph_cache", action="store_true",
                        help="Rebuild graph/features cache even if it exists")
    parser.add_argument("--graph_feature_retriever_cache", default=None,
                        help="Load chunk embeddings from a retriever cache")
    parser.add_argument("--candidate_results_jsonl", default=None,
                        help="Use fixed candidate pool from a prior retriever JSONL")
    parser.add_argument("--candidate_pool_name", default="CandidatePool",
                        help="Display name for --candidate_results_jsonl")
    parser.add_argument("--methods", default="best_retriever,cross_encoder,ppr,graphsage,rgcn,rgcn_constraint",
                        help="Comma-separated methods to run (available: best_retriever, cross_encoder, ppr, graphsage, rgcn, rgcn_constraint, qfe_rgcn, qfe_rgcn_v2, qfe_rgcn_ppr, mono_t5, list_t5, fast_final_graph)")
    parser.add_argument("--load_graphsage_checkpoint", default=None,
                        help="Load a saved GraphSAGE checkpoint and evaluate without retraining")
    parser.add_argument("--load_rgcn_checkpoint", default=None,
                        help="Load a saved R-GCN checkpoint and evaluate without retraining")
    parser.add_argument("--eval_only_checkpoint", action="store_true",
                        help="For GNN methods, require checkpoint loading and skip training")
    parser.add_argument("--sanity", action="store_true",
                        help="Sanity mode: minimal samples, 1 epoch")
    parser.add_argument("--overwrite_output_dir", action="store_true",
                        help="Overwrite existing results")
    parser.add_argument("--qfe_use_ppr", action="store_true",
                        help="QFE-RGCN: also compute PPR scores as auxiliary graph features (ablation only)")
    parser.add_argument("--gnn_use_ppr", action="store_true",
                        help="GraphSAGE/R-GCN: compute PPR auxiliary scores during evaluation (slow; default off)")
    parser.add_argument("--rerank_checkpoint_every", type=int, default=100,
                        help="Write reranker partial JSONL and print progress every N queries")
    parser.add_argument("--allow_rerank_fallback", action="store_true",
                        help="Allow fallback to candidate top-k when rerank raises an exception (debug only)")

    # ---- MonoT5 ----
    parser.add_argument("--mono_t5_model", default="castorini/monot5-base-msmarco",
                        help="MonoT5 model name or local path")
    parser.add_argument("--mono_t5_batch_size", type=int, default=8,
                        help="Batch size for MonoT5 inference")
    parser.add_argument("--mono_t5_max_length", type=int, default=512,
                        help="Max token length for MonoT5")
    parser.add_argument("--mono_t5_fp16", action="store_true",
                        help="Load MonoT5 in half precision")
    parser.add_argument("--mono_t5_score_cache", default="cache/rerank_scores/mono_t5_bge_top50",
                        help="Score cache directory for MonoT5")

    # ---- ListT5 ----
    parser.add_argument("--list_t5_model", default="Soyoung97/ListT5-base",
                        help="ListT5 model name or local path")
    parser.add_argument("--list_t5_batch_size", type=int, default=8,
                        help="Reserved batch size knob for ListT5 inference")
    parser.add_argument("--list_t5_max_length", type=int, default=128,
                        help="Max token length per ListT5 passage")
    parser.add_argument("--list_t5_listwise_k", type=int, default=5,
                        help="Number of passages per ListT5 listwise comparison")
    parser.add_argument("--list_t5_out_k", type=int, default=2,
                        help="Number of winners kept from each ListT5 comparison")
    parser.add_argument("--list_t5_fp16", action="store_true",
                        help="Load ListT5 in half precision")
    parser.add_argument("--list_t5_decision_cache", default="cache/rerank_scores/list_t5_bge_top50",
                        help="Decision cache directory for ListT5")

    # ---- Shared rerank cache controls ----
    parser.add_argument("--rebuild_rerank_score_cache", action="store_true",
                        help="Rebuild MonoT5/ListT5 cache even if it exists")
    parser.add_argument("--resume_rerank", action="store_true",
                        help="Resume reranking from partial cache (MonoT5/ListT5/QFE-RGCN)")
    parser.add_argument("--load_qfe_checkpoint", default=None,
                        help="Load a saved QFE-RGCN checkpoint and evaluate without retraining")
    parser.add_argument("--qfe_score_cache", default=None,
                        help="Cache directory for QFE rerank scores to avoid recomputation")
    parser.add_argument("--qfe_eval_subgraph_cache", default=None,
                        help="Cache directory for QFE eval subgraphs (optional, experimental)")

    # ---- QFE-RGCN v2 specific ----
    parser.add_argument("--qfe_delta_reg", type=float, default=0.05,
                        help="Delta regularisation weight for v2 (default 0.05)")
    parser.add_argument("--qfe_min_retrieval_weight", type=float, default=0.35,
                        help="Floor on retrieval weight in v2 fusion (default 0.35)")
    parser.add_argument("--gnn_negatives_per_positive", type=int, default=5,
                        help="Hard negatives per positive for QFE training (default 5)")
    parser.add_argument("--qfe_query_encoder", default="heuristic",
                        choices=["heuristic", "bge"],
                        help="Query encoder for QFE-RGCN v2: heuristic or bge (BGE-M3)")
    parser.add_argument("--qfe_query_embedding_cache", default="cache/query_embeddings/bge_m3_queries.pkl",
                        help="Pickle cache for BGE query embeddings")

    # ---- Fast Final Graph ----
    parser.add_argument("--fast_graph_rgcn_results_jsonl", default=None,
                        help="R-GCN results JSONL for Fast Final Graph feature extraction")
    parser.add_argument("--fast_graph_monot5_results_jsonl", default=None,
                        help="MonoT5 results JSONL for Fast Final Graph feature extraction")
    parser.add_argument("--fast_graph_ppr_results_jsonl", default=None,
                        help="PPR results JSONL for Fast Final Graph (optional; graph source disabled if omitted)")
    parser.add_argument("--fast_graph_model_cache", default=None,
                        help="Pickle cache path for Fast Final Graph training dataset")
    parser.add_argument("--fast_graph_epochs", type=int, default=20,
                        help="Training epochs for Fast Final Graph (default 20)")
    parser.add_argument("--fast_graph_batch_size", type=int, default=512,
                        help="Batch size for Fast Final Graph training (default 512)")
    parser.add_argument("--fast_graph_lr", type=float, default=1e-3,
                        help="Learning rate for Fast Final Graph (default 1e-3)")
    parser.add_argument("--fast_graph_min_rgcn_weight", type=float, default=0.35,
                        help="Minimum weight floor for R-GCN source (default 0.35)")
    parser.add_argument("--fast_graph_min_bge_weight", type=float, default=0.15,
                        help="Minimum weight floor for BGE source (default 0.15)")
    parser.add_argument("--fast_graph_max_entity_weight", type=float, default=0.10,
                        help="Maximum weight cap for entity heuristic source (default 0.10)")
    parser.add_argument("--fast_graph_delta_scale", type=float, default=0.05,
                        help="Scale for bounded tanh residual delta (default 0.05)")
    parser.add_argument("--fast_graph_hard_negatives", type=int, default=10,
                        help="Hard negatives per positive for Fast Final Graph (default 10)")
    parser.add_argument("--fast_graph_eval_split", default="all", choices=["all", "heldout"],
                        help="Evaluation split for Fast Final Graph: all or heldout (default all)")
    parser.add_argument("--load_fast_graph_checkpoint", default=None,
                        help="Load a saved Fast Final Graph checkpoint and evaluate without retraining")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    cfg.ensure_dirs()

    # Sanity mode overrides
    if args.sanity:
        args.num_samples = args.num_samples or 10
        args.epochs = args.epochs or 1
        args.top_n = 20
        if args.output_dir is None:
            args.output_dir = "outputs/table1_non_llm_reranking_sanity"

    if args.epochs:
        cfg.rerank["gnn_epochs"] = args.epochs
    selected_methods = {
        m.strip() for m in args.methods.split(",") if m.strip()
    }

    output_dir = Path(args.output_dir or cfg.output_dir)
    if not output_dir.is_absolute():
        output_dir = cfg.root_dir / output_dir

    # Overwrite guard
    if _has_results(output_dir) and not args.overwrite_output_dir:
        print(f"\nERROR: Output directory '{output_dir}' already contains results.")
        print(f"  Use --overwrite_output_dir to replace, or --output_dir for new.")
        sys.exit(1)

    k_values = cfg.evaluation.get("recall_k_values", [5, 10])
    device = args.device

    print("=" * 60)
    print("  TABLE 1: Non-LLM Reranking Comparison")
    print("=" * 60)
    print(f"  Output:       {output_dir}")
    print(f"  Config:       {args.config}")
    print(f"  Device:       {device} (GNN), {args.dense_device} (dense)")
    print(f"  Epochs:       {cfg.rerank.get('gnn_epochs', 'default')}")
    print(f"  Top-N:        {args.top_n}")
    print(f"  Output-K:     {args.output_k}")
    print(f"  Skip GNN:     {args.skip_gnn}")
    print(f"  Sanity:       {args.sanity}")
    print(f"  Corpus cache: {args.corpus_cache or 'disabled'}")
    print(f"  Graph cache:  {args.graph_cache or 'disabled'}")
    print(f"  Feature cache: {args.graph_feature_retriever_cache or 'disabled'}")
    print(f"  Candidate pool: {args.candidate_pool_name if args.candidate_results_jsonl else 'Hybrid'}")
    print(f"  Methods:      {','.join(sorted(selected_methods))}")

    # ---- 1. Load data ----
    print("\n[1/5] Loading FinDER data...")
    samples = load_dataset("finder", cfg.data_dir)
    if args.num_samples > 0:
        samples = samples[:args.num_samples]
    rng = random.Random(args.split_seed)
    rng.shuffle(samples)
    n_val = max(1, int(len(samples) * args.val_split))
    train_samples = samples[n_val:]
    test_samples = samples[:n_val]  # use val split as test for GNN training
    eval_samples = samples  # evaluate on all for metrics
    print(f"  Train: {len(train_samples)}  |  Val/Test: {len(test_samples)}  |  "
          f"Eval (all): {len(eval_samples)}")

    # ---- 2. Build corpus ----
    print("[2/5] Building corpus...")
    corpus_cache = _resolve_cache_path(args.corpus_cache, cfg)
    graph_cache = _resolve_cache_path(args.graph_cache, cfg)
    corpus_chunks, gold_map = _load_or_build_corpus(
        samples,
        cfg,
        corpus_cache,
        rebuild=args.rebuild_corpus_cache,
        allow_gold_only_corpus=args.allow_gold_only_corpus,
    )
    chunk_by_id = {c.chunk_id: c for c in corpus_chunks}
    gold_chunk_ids: Set[str] = set()
    for gids in gold_map.values():
        gold_chunk_ids.update(gids)
    print(f"  {len(corpus_chunks)} chunks ({len(gold_chunk_ids)} gold, "
          f"{len(corpus_chunks) - len(gold_chunk_ids)} distractors)")

    # ---- 3. Retrieval indices ----
    print("[3/5] Preparing candidate pool and node embeddings...")
    candidate_retriever = None
    candidate_label = "Hybrid E5-Mistral"
    if args.candidate_results_jsonl:
        candidate_path = _resolve_cache_path(args.candidate_results_jsonl, cfg)
        candidate_retriever = CandidatePoolRetriever(
            candidate_path,
            chunk_by_id,
            name=args.candidate_pool_name,
        )
        candidate_label = args.candidate_pool_name
        print(
            f"  Candidate pool: {candidate_label} "
            f"({candidate_retriever.num_queries} queries) from {candidate_path}"
        )

    if args.graph_feature_retriever_cache:
        feature_cache = _resolve_cache_path(args.graph_feature_retriever_cache, cfg)
        chunk_embeddings, embedding_dim = _load_chunk_embeddings_from_retriever_cache(
            feature_cache,
            corpus_chunks,
        )
        if candidate_retriever is None:
            raise ValueError(
                "--graph_feature_retriever_cache without --candidate_results_jsonl "
                "would skip the retriever needed for candidate search."
            )
    else:
        print("  Building BM25/Dense/Hybrid for candidates and node embeddings...")
        bm25 = BM25Retriever(
            k1=cfg.retrieval.get("bm25_k1", 1.5),
            b=cfg.retrieval.get("bm25_b", 0.75),
        )
        bm25.index(corpus_chunks)
        dense = DenseRetriever(
            model_name=cfg.retrieval.get("dense_model", "all-MiniLM-L6-v2"),
            device=args.dense_device,
            query_instruction=cfg.retrieval.get("dense_query_instruction"),
            e5_max_seq_length=cfg.retrieval.get("e5_max_seq_length", 512),
            e5_batch_size=cfg.retrieval.get("e5_batch_size"),
            debug=cfg.retrieval.get("debug_dense", False),
        )
        dense.index(corpus_chunks, batch_size=args.dense_batch_size)
        chunk_embeddings = dense.chunk_embeddings()
        embedding_dim = next(iter(chunk_embeddings.values())).shape[0] if chunk_embeddings else 0
        print(f"  Dense backend: {dense.backend}, embedding dim: {embedding_dim}")

        if embedding_dim == 0:
            print("  [ERROR] Dense embeddings empty!")
            sys.exit(1)

        if candidate_retriever is None:
            candidate_retriever = HybridRetriever(
                bm25, dense,
                alpha=cfg.retrieval.get("hybrid_alpha", 0.5),
            )

    if candidate_retriever is None:
        raise RuntimeError("No candidate retriever configured.")

    # Cross-encoder
    cross_encoder = CrossEncoderReranker(
        model_name=cfg.cross_encoder.get("model_name", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
        batch_size=cfg.cross_encoder.get("batch_size", 32),
    )

    # ---- 4. Graph + features ----
    print("[4/5] Building graph and features...")
    entity_map, graph, features = _load_or_build_graph_features(
        corpus_chunks,
        train_samples + test_samples,
        cfg,
        args,
        candidate_retriever,
        chunk_embeddings,
        embedding_dim,
        graph_cache,
    )
    feature_dim = next(iter(features.values())).shape[0]
    print(f"  Feature dim: {feature_dim}")

    # ---- 5. Run methods ----
    print("\n[5/5] Running evidence ranking comparison...")
    extractor = EntityExtractor()
    all_results: Dict[str, List[Dict]] = {}
    t_total_start = time.time()

    # 5a. Best Retriever / fixed candidate pool
    if "best_retriever" in selected_methods:
        print(f"\n  [best_retriever] {candidate_label}...")
        t0 = time.time()
        all_results["best_retriever"] = run_best_retriever(
            eval_samples, candidate_retriever, gold_map,
            top_n=args.top_n, output_k=args.output_k,
        )
        print(f"    {len(all_results['best_retriever'])} queries in {time.time() - t0:.1f}s")
        _checkpoint_method(output_dir, "best_retriever", all_results, k_values)

    # 5b. Cross-Encoder
    if "cross_encoder" in selected_methods:
        print(f"\n  [cross_encoder] {candidate_label} + Cross-Encoder...")
        t0 = time.time()
        all_results["cross_encoder"] = run_cross_encoder(
            eval_samples, candidate_retriever, gold_map, cross_encoder,
            top_n=min(args.top_n * 2, 100), output_k=args.output_k,
        )
        print(f"    {len(all_results['cross_encoder'])} queries in {time.time() - t0:.1f}s")
        _checkpoint_method(output_dir, "cross_encoder", all_results, k_values)

    # 5c. PPR
    if "ppr" in selected_methods:
        print(f"\n  [ppr] {candidate_label} + PPR...")
        t0 = time.time()
        all_results["ppr"] = run_ppr(
            eval_samples, candidate_retriever, graph, corpus_chunks, gold_map, extractor,
            ppr_alpha=cfg.rerank.get("ppr_alpha", 0.85),
            top_n=args.top_n, output_k=args.output_k,
        )
        print(f"    {len(all_results['ppr'])} queries in {time.time() - t0:.1f}s")
        _checkpoint_method(output_dir, "ppr", all_results, k_values)

    # 5d/e/f. GNN methods (train on train_samples, eval on eval_samples)
    if not args.skip_gnn:
        # Train GraphSAGE
        if "graphsage" in selected_methods:
            t0 = time.time()
            if args.load_graphsage_checkpoint:
                print("\n  [graphsage] Loading GraphSAGE checkpoint...")
                sage_reranker = _load_graphsage_checkpoint(
                    args.load_graphsage_checkpoint,
                    base_feature_dim=feature_dim,
                    cfg=cfg,
                    device=device,
                )
                print(f"    Loaded: {args.load_graphsage_checkpoint}")
            else:
                if args.eval_only_checkpoint:
                    raise ValueError(
                        "--eval_only_checkpoint requested but "
                        "--load_graphsage_checkpoint was not provided"
                    )
                print("\n  [graphsage] Training GraphSAGE reranker...")
                cfg.rerank["gnn_model"] = "sage"
                sage_reranker, sage_history, sage_meta = train_gnn_reranker(
                    train_samples, candidate_retriever, graph, features, gold_map, cfg,
                    epochs=cfg.rerank.get("gnn_epochs", 10),
                    device=device, min_pairs=5, verbose=True,
                )
            if sage_reranker is not None:
                if not args.load_graphsage_checkpoint:
                    save_training_artifacts(
                        sage_reranker, sage_history, output_dir, sage_meta,
                        experiment="table1_graphsage",
                    )
                    print(f"    Training done in {time.time() - t0:.1f}s")
                print(f"    Evaluating GraphSAGE on {len(eval_samples)} samples...")
                t_eval = time.time()
                all_results["graphsage"] = run_gnn_reranker(
                    eval_samples, candidate_retriever, graph, features, gold_map,
                    chunk_by_id, extractor, sage_reranker, "graphsage",
                    top_n=args.top_n, output_k=args.output_k,
                    ppr_alpha=cfg.rerank.get("ppr_alpha", 0.85),
                    use_ppr=args.gnn_use_ppr,
                    progress_every=args.rerank_checkpoint_every,
                    partial_output_dir=output_dir,
                )
                print(f"    Evaluation done in {time.time() - t_eval:.1f}s")
                _checkpoint_method(output_dir, "graphsage", all_results, k_values)
            else:
                print("    [SKIP] GraphSAGE training failed (not enough pairs)")

        # Train R-GCN
        if "rgcn" in selected_methods or "rgcn_constraint" in selected_methods:
            t0 = time.time()
            if args.load_rgcn_checkpoint:
                print("\n  [rgcn] Loading R-GCN checkpoint...")
                rgcn_reranker = _load_rgcn_checkpoint(
                    args.load_rgcn_checkpoint,
                    base_feature_dim=feature_dim,
                    cfg=cfg,
                    device=device,
                )
                print(f"    Loaded: {args.load_rgcn_checkpoint}")
            else:
                if args.eval_only_checkpoint:
                    raise ValueError(
                        "--eval_only_checkpoint requested but "
                        "--load_rgcn_checkpoint was not provided"
                    )
                print("\n  [rgcn] Training R-GCN reranker...")
                cfg.rerank["gnn_model"] = "rgcn"
                rgcn_reranker, rgcn_history, rgcn_meta = train_gnn_reranker(
                    train_samples, candidate_retriever, graph, features, gold_map, cfg,
                    epochs=cfg.rerank.get("gnn_epochs", 10),
                    device=device, min_pairs=5, verbose=True,
                )
            if rgcn_reranker is not None:
                if not args.load_rgcn_checkpoint:
                    save_training_artifacts(
                        rgcn_reranker, rgcn_history, output_dir, rgcn_meta,
                        experiment="table1_rgcn",
                    )
                    print(f"    Training done in {time.time() - t0:.1f}s")
                if "rgcn" in selected_methods:
                    print(f"    Evaluating R-GCN on {len(eval_samples)} samples...")
                    t_eval = time.time()
                    all_results["rgcn"] = run_gnn_reranker(
                        eval_samples, candidate_retriever, graph, features, gold_map,
                        chunk_by_id, extractor, rgcn_reranker, "rgcn",
                        top_n=args.top_n, output_k=args.output_k,
                        ppr_alpha=cfg.rerank.get("ppr_alpha", 0.85),
                        use_ppr=args.gnn_use_ppr,
                        progress_every=args.rerank_checkpoint_every,
                        partial_output_dir=output_dir,
                    )
                    print(f"    Evaluation done in {time.time() - t_eval:.1f}s")
                    _checkpoint_method(output_dir, "rgcn", all_results, k_values)

                if "rgcn_constraint" in selected_methods:
                    print("\n  [rgcn_constraint] R-GCN + Constraint Score...")
                    t_eval = time.time()
                    constraint_scorer = ConstraintScorer(
                        company_weight=cfg.constraint.get("company_match_weight", 1.0),
                        year_weight=cfg.constraint.get("year_match_weight", 1.0),
                        metric_weight=cfg.constraint.get("metric_match_weight", 0.8),
                        filing_type_weight=cfg.constraint.get("filing_type_match_weight", 0.5),
                    )
                    cfg.rerank["fusion_delta"] = 0.1
                    all_results["rgcn_constraint"] = run_rgcn_constraint(
                        eval_samples, candidate_retriever, graph, features, gold_map,
                        chunk_by_id, extractor, rgcn_reranker, constraint_scorer, cfg,
                        method_name="rgcn_constraint",
                        top_n=args.top_n, output_k=args.output_k,
                        ppr_alpha=cfg.rerank.get("ppr_alpha", 0.85),
                    )
                    print(f"    {len(all_results['rgcn_constraint'])} queries in {time.time() - t_eval:.1f}s")
                    _checkpoint_method(output_dir, "rgcn_constraint", all_results, k_values)
            else:
                print("    [SKIP] R-GCN training failed (not enough pairs)")

        # Train / Load QFE-RGCN (covers both "qfe_rgcn" and "qfe_rgcn_ppr")
        if "qfe_rgcn" in selected_methods or "qfe_rgcn_ppr" in selected_methods:
            qfe_score_cache_path: Optional[Path] = None
            if args.qfe_score_cache:
                qfe_score_cache_path = _resolve_cache_path(args.qfe_score_cache, cfg)

            # Track which checkpoint to use for score cache
            _qfe_checkpoint_for_cache: Optional[str] = args.load_qfe_checkpoint

            if args.load_qfe_checkpoint:
                # ---- eval-only: load checkpoint, skip training ----
                print("\n  [qfe_rgcn] Loading QFE-RGCN checkpoint (eval-only)...")
                print(f"    [qfe_rgcn] Checkpoint: {args.load_qfe_checkpoint}")
                print(f"    [qfe_rgcn] PPR: {'enabled (--qfe_use_ppr)' if args.qfe_use_ppr else 'disabled (default)'}")
                print(f"    [qfe_rgcn] Fallback: {'enabled (--allow_rerank_fallback)' if args.allow_rerank_fallback else 'disabled (default, fail-fast)'}")
                t0 = time.time()
                qfe_reranker, qfe_meta = _load_qfe_reranker_from_checkpoint(
                    args.load_qfe_checkpoint,
                    feature_dim=feature_dim,
                    cfg=cfg,
                    device=device,
                )
                print(f"    Loaded in {time.time() - t0:.1f}s "
                      f"(relations={qfe_meta['num_relations']}, "
                      f"query_embed_dim={qfe_meta['query_embed_dim']})")

                # Build query embeddings for eval samples
                qfe_query_embeddings = build_query_embedding_cache(
                    [s["question"] for s in eval_samples]
                )
                for q, emb in qfe_query_embeddings.items():
                    qfe_reranker.query_embeddings[q] = emb

            else:
                if args.eval_only_checkpoint:
                    raise ValueError(
                        "--eval_only_checkpoint requested but "
                        "--load_qfe_checkpoint was not provided for qfe_rgcn/qfe_rgcn_ppr. "
                        "Pass --load_qfe_checkpoint PATH to load a saved QFE-RGCN checkpoint."
                    )

                # ---- train from scratch ----
                print("\n  [qfe_rgcn] Training QFE-RGCN reranker...")
                print(f"    [qfe_rgcn] PPR: {'enabled (--qfe_use_ppr)' if args.qfe_use_ppr else 'disabled (default)'}")
                print(f"    [qfe_rgcn] Fallback: {'enabled (--allow_rerank_fallback)' if args.allow_rerank_fallback else 'disabled (default, fail-fast)'}")
                print(f"    [qfe_rgcn] batch_size: 1 (forced — query-aware relation gates are per-query)")
                t0 = time.time()
                cfg.rerank["gnn_model"] = "qfe_rgcn"

                # Pre-compute query embeddings for all samples
                all_questions = [s["question"] for s in train_samples]
                qfe_query_embeddings = build_query_embedding_cache(all_questions)
                print(f"    Query embeddings: {len(qfe_query_embeddings)} unique queries "
                      f"(dim={QUERY_EMBED_DIM})")

                qfe_reranker, qfe_history, qfe_meta = train_gnn_reranker(
                    train_samples, candidate_retriever, graph, features, gold_map, cfg,
                    epochs=cfg.rerank.get("gnn_epochs", 10),
                    device=device, min_pairs=5, verbose=True,
                    query_embeddings=qfe_query_embeddings,
                )

            if qfe_reranker is not None:
                if not args.load_qfe_checkpoint:
                    qfe_artifacts = save_training_artifacts(
                        qfe_reranker, qfe_history, output_dir, qfe_meta,
                        experiment="table1_qfe_rgcn",
                    )
                    print(f"    Training done in {time.time() - t0:.1f}s")
                    # Use the newly saved checkpoint for score cache
                    _qfe_checkpoint_for_cache = str(qfe_artifacts["checkpoint"])
                    if qfe_score_cache_path is not None:
                        print(f"    [qfe_rgcn] Using checkpoint for score cache: "
                              f"{_qfe_checkpoint_for_cache}")

                # Add eval queries to query_embeddings
                for s in eval_samples:
                    q = s["question"]
                    if q not in qfe_reranker.query_embeddings:
                        qfe_reranker.query_embeddings[q] = derive_query_vector(
                            q, dim=qfe_reranker.query_embed_dim,
                        )

                # Resolve score cache path for eval
                _qfe_score_cache: Optional[Path] = None
                if args.qfe_score_cache:
                    _qfe_score_cache = _resolve_cache_path(args.qfe_score_cache, cfg)

                # Common kwargs for run_qfe_rgcn
                if _qfe_score_cache is not None and _qfe_checkpoint_for_cache:
                    print(f"    [qfe_rgcn] Using checkpoint for score cache: "
                          f"{_qfe_checkpoint_for_cache}")
                qfe_eval_kwargs = dict(
                    progress_every=args.rerank_checkpoint_every,
                    partial_output_dir=output_dir,
                    resume_rerank=args.resume_rerank,
                    score_cache_path=_qfe_score_cache,
                    checkpoint_path=_qfe_checkpoint_for_cache,
                )

                if "qfe_rgcn" in selected_methods:
                    print(f"    Evaluating QFE-RGCN on {len(eval_samples)} samples...")
                    t_eval = time.time()
                    all_results["qfe_rgcn"] = run_qfe_rgcn(
                        eval_samples, candidate_retriever, graph, features, gold_map,
                        chunk_by_id, extractor, qfe_reranker, qfe_query_embeddings,
                        method_name="qfe_rgcn",
                        top_n=args.top_n, output_k=args.output_k,
                        ppr_alpha=cfg.rerank.get("ppr_alpha", 0.85),
                        use_ppr=args.qfe_use_ppr,
                        allow_fallback=args.allow_rerank_fallback,
                        **qfe_eval_kwargs,
                    )
                    print(f"    Evaluation done in {time.time() - t_eval:.1f}s")
                    _checkpoint_method(output_dir, "qfe_rgcn", all_results, k_values)

                # QFE-RGCN + PPR (ablation: same model, but with PPR graph features)
                if "qfe_rgcn_ppr" in selected_methods:
                    print(f"\n  [qfe_rgcn_ppr] Evaluating QFE-RGCN + PPR (ablation)...")
                    t_eval2 = time.time()
                    all_results["qfe_rgcn_ppr"] = run_qfe_rgcn(
                        eval_samples, candidate_retriever, graph, features, gold_map,
                        chunk_by_id, extractor, qfe_reranker, qfe_query_embeddings,
                        method_name="qfe_rgcn_ppr",
                        top_n=args.top_n, output_k=args.output_k,
                        ppr_alpha=cfg.rerank.get("ppr_alpha", 0.85),
                        use_ppr=True,
                        allow_fallback=args.allow_rerank_fallback,
                        **qfe_eval_kwargs,
                    )
                    print(f"    Evaluation done in {time.time() - t_eval2:.1f}s")
                    _checkpoint_method(output_dir, "qfe_rgcn_ppr", all_results, k_values)
            else:
                print("    [SKIP] QFE-RGCN training failed (not enough pairs)")

        # ---- Train / Load QFE-RGCN v2 ----
        if "qfe_rgcn_v2" in selected_methods:
            _v2_score_cache: Optional[Path] = None
            if args.qfe_score_cache:
                _v2_score_cache = _resolve_cache_path(args.qfe_score_cache, cfg)

            # Determine which checkpoint to use for score cache
            _v2_checkpoint_for_cache: Optional[str] = args.load_qfe_checkpoint

            if args.load_qfe_checkpoint:
                # ---- eval-only: load v2 checkpoint ----
                print("\n  [qfe_rgcn_v2] Loading QFE-RGCN v2 checkpoint (eval-only)...")
                print(f"    [qfe_rgcn_v2] Checkpoint: {args.load_qfe_checkpoint}")
                t0 = time.time()
                qfe_v2_reranker, qfe_v2_meta = _load_qfe_v2_reranker_from_checkpoint(
                    args.load_qfe_checkpoint,
                    feature_dim=feature_dim,
                    cfg=cfg,
                    device=device,
                )
                print(f"    Loaded in {time.time() - t0:.1f}s "
                      f"(v{qfe_v2_meta['version']}, "
                      f"relations={qfe_v2_meta['num_relations']}, "
                      f"query_embed_dim={qfe_v2_meta['query_embed_dim']})")

                # Build query embeddings for eval — must match checkpoint encoder
                if (qfe_v2_reranker.query_projection is not None
                        or qfe_v2_reranker.query_encoder == "bge"):
                    print(f"    [qfe_rgcn_v2] Building BGE query embeddings for eval...")
                    qfe_v2_query_embeddings = build_bge_query_embedding_cache(
                        [s["question"] for s in eval_samples],
                        device=device,
                        cache_path=args.qfe_query_embedding_cache,
                        fail_on_missing_model=True,
                    )
                else:
                    qfe_v2_query_embeddings = build_query_embedding_cache(
                        [s["question"] for s in eval_samples]
                    )
                for q, emb in qfe_v2_query_embeddings.items():
                    qfe_v2_reranker.query_embeddings[q] = emb

            else:
                if args.eval_only_checkpoint:
                    raise ValueError(
                        "--eval_only_checkpoint requested but "
                        "--load_qfe_checkpoint was not provided for qfe_rgcn_v2."
                    )

                # ---- train v2 from scratch ----
                print("\n  [qfe_rgcn_v2] Training QFE-RGCN v2 reranker...")
                print(f"    [qfe_rgcn_v2] query_encoder: {args.qfe_query_encoder}")
                print(f"    [qfe_rgcn_v2] min_retrieval_weight: {args.qfe_min_retrieval_weight}")
                print(f"    [qfe_rgcn_v2] delta_reg: {args.qfe_delta_reg}")
                print(f"    [qfe_rgcn_v2] negatives_per_positive: {args.gnn_negatives_per_positive}")
                print(f"    [qfe_rgcn_v2] batch_size: 1 (forced)")
                t0 = time.time()
                cfg.rerank["gnn_model"] = "qfe_rgcn_v2"

                # Build BGE embeddings for train + eval (so eval queries are ready)
                qfe_v2_query_embeddings = None
                if args.qfe_query_encoder == "bge":
                    all_questions = [s["question"] for s in train_samples + eval_samples]
                    qfe_v2_query_embeddings = build_bge_query_embedding_cache(
                        all_questions,
                        device=device,
                        cache_path=args.qfe_query_embedding_cache,
                        fail_on_missing_model=True,
                    )
                    print(f"    BGE query embeddings: {len(qfe_v2_query_embeddings)} unique "
                          f"(train+eval)")

                qfe_v2_reranker, qfe_v2_history, qfe_v2_meta = train_gnn_reranker(
                    train_samples, candidate_retriever, graph, features, gold_map, cfg,
                    epochs=cfg.rerank.get("gnn_epochs", 10),
                    device=device, min_pairs=5, verbose=True,
                    query_embeddings=qfe_v2_query_embeddings,
                    negatives_per_positive=args.gnn_negatives_per_positive,
                    min_retrieval_weight=args.qfe_min_retrieval_weight,
                    delta_reg=args.qfe_delta_reg,
                    query_encoder=args.qfe_query_encoder,
                    query_embedding_cache=args.qfe_query_embedding_cache,
                )

            if qfe_v2_reranker is not None:
                if not args.load_qfe_checkpoint:
                    v2_artifacts = save_training_artifacts(
                        qfe_v2_reranker, qfe_v2_history, output_dir, qfe_v2_meta,
                        experiment="table1_qfe_rgcn_v2",
                    )
                    print(f"    Training done in {time.time() - t0:.1f}s")
                    # Use the newly saved checkpoint for score cache
                    _v2_checkpoint_for_cache = str(v2_artifacts["checkpoint"])
                    if _v2_score_cache is not None:
                        print(f"    [qfe_rgcn_v2] Using checkpoint for score cache: "
                              f"{_v2_checkpoint_for_cache}")

                # Add eval queries to query_embeddings — must match encoder
                if qfe_v2_reranker.query_projection is not None:
                    # BGE: all queries should already be in cache from build step above
                    for s in eval_samples:
                        q = s["question"]
                        if q not in qfe_v2_reranker.query_embeddings:
                            raise ValueError(
                                f"Eval query '{q[:80]}...' missing from BGE query "
                                f"embedding cache.  Rebuild with "
                                f"--qfe_query_embedding_cache."
                            )
                else:
                    # Heuristic: fill missing with derive_query_vector
                    for s in eval_samples:
                        q = s["question"]
                        if q not in qfe_v2_reranker.query_embeddings:
                            qfe_v2_reranker.query_embeddings[q] = derive_query_vector(
                                q, dim=qfe_v2_reranker.query_embed_dim,
                            )

                if _v2_score_cache is not None and _v2_checkpoint_for_cache:
                    print(f"    [qfe_rgcn_v2] Using checkpoint for score cache: "
                          f"{_v2_checkpoint_for_cache}")

                v2_eval_kwargs = dict(
                    progress_every=args.rerank_checkpoint_every,
                    partial_output_dir=output_dir,
                    resume_rerank=args.resume_rerank,
                    score_cache_path=_v2_score_cache,
                    checkpoint_path=_v2_checkpoint_for_cache,
                )

                print(f"    Evaluating QFE-RGCN v2 on {len(eval_samples)} samples...")
                t_eval = time.time()
                all_results["qfe_rgcn_v2"] = run_qfe_rgcn(
                    eval_samples, candidate_retriever, graph, features, gold_map,
                    chunk_by_id, extractor, qfe_v2_reranker,
                    qfe_v2_reranker.query_embeddings,
                    method_name="qfe_rgcn_v2",
                    top_n=args.top_n, output_k=args.output_k,
                    ppr_alpha=cfg.rerank.get("ppr_alpha", 0.85),
                    use_ppr=False,
                    allow_fallback=args.allow_rerank_fallback,
                    **v2_eval_kwargs,
                )
                print(f"    Evaluation done in {time.time() - t_eval:.1f}s")
                _checkpoint_method(output_dir, "qfe_rgcn_v2", all_results, k_values)
            else:
                print("    [SKIP] QFE-RGCN v2 training failed (not enough pairs)")
    else:
        print("\n  [skip_gnn] Skipping GraphSAGE, R-GCN, and R-GCN+Constraint.")

    # ---- Fast Final Graph (lightweight fusion, no GNN needed) ----
    if "fast_final_graph" in selected_methods:
        if not args.candidate_results_jsonl:
            raise ValueError(
                "Fast Final Graph requires a fixed BGE candidate pool. "
                "Pass --candidate_results_jsonl pointing to bge_m3_dense_results.jsonl."
            )
        if not args.fast_graph_rgcn_results_jsonl:
            raise ValueError(
                "Fast Final Graph requires R-GCN results. "
                "Pass --fast_graph_rgcn_results_jsonl."
            )
        if not args.fast_graph_monot5_results_jsonl:
            raise ValueError(
                "Fast Final Graph requires MonoT5 results. "
                "Pass --fast_graph_monot5_results_jsonl."
            )

        print(f"\n  [fast_final_graph] Fast Final Graph (Ours)")
        print(f"    [fast_final_graph] epochs: {args.fast_graph_epochs}, "
              f"batch_size: {args.fast_graph_batch_size}, lr: {args.fast_graph_lr}")
        print(f"    [fast_final_graph] min_rgcn_weight: {args.fast_graph_min_rgcn_weight}, "
              f"min_bge_weight: {args.fast_graph_min_bge_weight}")
        print(f"    [fast_final_graph] max_entity_weight: {args.fast_graph_max_entity_weight}, "
              f"delta_scale: {args.fast_graph_delta_scale}")
        print(f"    [fast_final_graph] hard_negatives: {args.fast_graph_hard_negatives}")
        if args.fast_graph_ppr_results_jsonl:
            print(f"    [fast_final_graph] PPR source: {args.fast_graph_ppr_results_jsonl}")
        if args.fast_graph_model_cache:
            print(f"    [fast_final_graph] Model cache: {args.fast_graph_model_cache}")
        if args.load_fast_graph_checkpoint:
            print(f"    [fast_final_graph] Eval-only: loading {args.load_fast_graph_checkpoint}")
        fast_graph_eval_samples = eval_samples
        if args.fast_graph_eval_split == "heldout":
            fast_graph_eval_samples = test_samples
        print(f"    [fast_final_graph] eval_split: {args.fast_graph_eval_split} "
              f"({len(fast_graph_eval_samples)} samples)")

        t0 = time.time()

        # Resolve paths
        bge_jsonl = str(_resolve_cache_path(args.candidate_results_jsonl, cfg))
        rgcn_jsonl = str(_resolve_cache_path(args.fast_graph_rgcn_results_jsonl, cfg))
        monot5_jsonl = str(_resolve_cache_path(args.fast_graph_monot5_results_jsonl, cfg))

        ppr_jsonl: Optional[str] = None
        if args.fast_graph_ppr_results_jsonl:
            ppr_jsonl = str(_resolve_cache_path(args.fast_graph_ppr_results_jsonl, cfg))

        model_cache: Optional[str] = None
        if args.fast_graph_model_cache:
            model_cache = str(_resolve_cache_path(args.fast_graph_model_cache, cfg))

        ffg_results, ffg_model, ffg_meta = run_fast_final_graph_method(
            train_samples=train_samples,
            eval_samples=fast_graph_eval_samples,
            gold_map=gold_map,
            chunk_by_id=chunk_by_id,
            bge_results_jsonl=bge_jsonl,
            rgcn_results_jsonl=rgcn_jsonl,
            monot5_results_jsonl=monot5_jsonl,
            output_dir=output_dir,
            ppr_results_jsonl=ppr_jsonl,
            model_cache_path=model_cache,
            epochs=args.fast_graph_epochs,
            batch_size=args.fast_graph_batch_size,
            lr=args.fast_graph_lr,
            min_rgcn_weight=args.fast_graph_min_rgcn_weight,
            min_bge_weight=args.fast_graph_min_bge_weight,
            max_entity_weight=args.fast_graph_max_entity_weight,
            delta_scale=args.fast_graph_delta_scale,
            hard_negatives=args.fast_graph_hard_negatives,
            top_n=args.top_n,
            output_k=args.output_k,
            device=device,
            val_split=args.val_split,
            split_seed=args.split_seed,
            load_checkpoint_path=args.load_fast_graph_checkpoint,
            progress_every=args.rerank_checkpoint_every,
            eval_only=args.eval_only_checkpoint,
        )

        all_results["fast_final_graph"] = ffg_results

        # Save checkpoint and artifacts
        if not args.load_fast_graph_checkpoint and ffg_model is not None:
            ckpt_dir = output_dir / "model_checkpoints"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            ckpt_path = ckpt_dir / f"fast_final_graph_{stamp}.pt"
            save_checkpoint(ffg_model, ckpt_path, meta=ffg_meta)
            print(f"    [fast_final_graph] Checkpoint saved: {ckpt_path}")

            # Save loss history
            history_path = output_dir / f"fast_final_graph_loss_history_{stamp}.json"
            with open(history_path, "w", encoding="utf-8") as fh:
                json.dump(ffg_meta, fh, indent=2)
            print(f"    [fast_final_graph] Loss history saved: {history_path}")

        _checkpoint_method(output_dir, "fast_final_graph", all_results, k_values)
        print(f"    [fast_final_graph] Total time: {time.time() - t0:.1f}s")

    # ---- MonoT5 (BGE-M3 top-50 reranking, Table II) ----
    if "mono_t5" in selected_methods:
        if not args.candidate_results_jsonl:
            raise ValueError(
                "MonoT5 must rerank a fixed BGE-M3 candidate pool. "
                "Pass --candidate_results_jsonl pointing to bge_m3_dense_results.jsonl."
            )
        print(f"\n  [mono_t5] MonoT5 reranking on BGE-M3 top-{args.top_n}...")
        print(f"    [mono_t5] Model: {args.mono_t5_model}")
        print(f"    [mono_t5] Batch size: {args.mono_t5_batch_size}, max_length: {args.mono_t5_max_length}")
        print(f"    [mono_t5] FP16: {args.mono_t5_fp16}")
        print(f"    [mono_t5] Score cache: {args.mono_t5_score_cache}")
        print(f"    [mono_t5] Fallback: {'enabled (--allow_rerank_fallback)' if args.allow_rerank_fallback else 'disabled (default, fail-fast)'}")
        t0 = time.time()

        # Build candidate_pool dict: question -> list of chunk_ids from BGE-M3
        candidate_pool: Dict[str, List[str]] = {}
        for s in eval_samples:
            hr = candidate_retriever.search(s["question"], top_k=args.top_n)
            candidate_pool[s["question"]] = [c.chunk_id for c, _ in hr]

        # Resolve score cache path
        mono_score_cache = _resolve_cache_path(args.mono_t5_score_cache, cfg)

        # Resolve corpus_cache and candidate_results_jsonl as absolute strings
        corpus_cache_str = str(_resolve_cache_path(args.corpus_cache, cfg)) if args.corpus_cache else ""
        candidate_jsonl_str = str(_resolve_cache_path(args.candidate_results_jsonl, cfg)) if args.candidate_results_jsonl else ""

        all_results["mono_t5"] = run_mono_t5(
            eval_samples, chunk_by_id, candidate_pool, gold_map,
            model_name_or_path=args.mono_t5_model,
            batch_size=args.mono_t5_batch_size,
            max_length=args.mono_t5_max_length,
            device=device,
            use_fp16=args.mono_t5_fp16,
            top_n=args.top_n,
            output_k=args.output_k,
            score_cache_dir=mono_score_cache,
            rebuild_score_cache=args.rebuild_rerank_score_cache,
            resume_rerank=args.resume_rerank,
            candidate_pool_name=args.candidate_pool_name,
            candidate_results_jsonl=candidate_jsonl_str,
            corpus_cache=corpus_cache_str,
            allow_fallback=args.allow_rerank_fallback,
            checkpoint_every=args.rerank_checkpoint_every,
            output_dir=output_dir,
        )
        print(f"    {len(all_results['mono_t5'])} queries in {time.time() - t0:.1f}s")
        _checkpoint_method(output_dir, "mono_t5", all_results, k_values)

    # ---- ListT5 (BGE-M3 top-50 reranking, Table II) ----
    if "list_t5" in selected_methods:
        if not args.candidate_results_jsonl:
            raise ValueError(
                "ListT5 must rerank a fixed BGE-M3 candidate pool. "
                "Pass --candidate_results_jsonl pointing to bge_m3_dense_results.jsonl."
            )

        print(f"\n  [list_t5] ListT5 reranking on BGE-M3 top-{args.top_n}...")
        print(f"    [list_t5] Model: {args.list_t5_model}")
        print(f"    [list_t5] listwise_k={args.list_t5_listwise_k}, out_k={args.list_t5_out_k}, max_length={args.list_t5_max_length}")
        print(f"    [list_t5] FP16: {args.list_t5_fp16}")
        print(f"    [list_t5] Decision cache: {args.list_t5_decision_cache}")
        print(f"    [list_t5] Fallback: {'enabled (--allow_rerank_fallback)' if args.allow_rerank_fallback else 'disabled (default, fail-fast)'}")
        t0 = time.time()

        # Reuse candidate_pool from MonoT5 if already built, otherwise build.
        if "candidate_pool" not in locals():
            candidate_pool = {}
            for s in eval_samples:
                hr = candidate_retriever.search(s["question"], top_k=args.top_n)
                candidate_pool[s["question"]] = [c.chunk_id for c, _ in hr]

        list_decision_cache = _resolve_cache_path(args.list_t5_decision_cache, cfg)

        corpus_cache_str = str(_resolve_cache_path(args.corpus_cache, cfg)) if args.corpus_cache else ""
        candidate_jsonl_str = str(_resolve_cache_path(args.candidate_results_jsonl, cfg)) if args.candidate_results_jsonl else ""

        all_results["list_t5"] = run_list_t5(
            eval_samples, chunk_by_id, candidate_pool, gold_map,
            model_name_or_path=args.list_t5_model,
            batch_size=args.list_t5_batch_size,
            max_length=args.list_t5_max_length,
            listwise_k=args.list_t5_listwise_k,
            out_k=args.list_t5_out_k,
            device=device,
            use_fp16=args.list_t5_fp16,
            top_n=args.top_n,
            output_k=args.output_k,
            decision_cache_dir=list_decision_cache,
            rebuild_decision_cache=args.rebuild_rerank_score_cache,
            resume_rerank=args.resume_rerank,
            candidate_pool_name=args.candidate_pool_name,
            candidate_results_jsonl=candidate_jsonl_str,
            corpus_cache=corpus_cache_str,
            allow_fallback=args.allow_rerank_fallback,
            checkpoint_every=args.rerank_checkpoint_every,
            output_dir=output_dir,
        )
        print(f"    {len(all_results['list_t5'])} queries in {time.time() - t0:.1f}s")
        _checkpoint_method(output_dir, "list_t5", all_results, k_values)

    total_time = time.time() - t_total_start

    # ---- Compute summaries ----
    print("\n" + "=" * 70)
    print("  TABLE 1 RESULTS")
    print("=" * 70)

    summaries: Dict[str, Dict] = {}
    for method, results in all_results.items():
        er = compute_all_metrics(method, results, k_values=k_values)
        summaries[method] = {
            "method": method,
            "num_samples": er.num_samples,
            "mrr": round(er.mrr, 4),
        }
        for k in k_values:
            summaries[method][f"recall@{k}"] = round(er.evidence_recall.get(k, 0), 4)
            summaries[method][f"ndcg@{k}"] = round(er.ndcg.get(k, 0), 4)

    # Print summary table
    print_labels = {
        "best_retriever": "Best Retriever      ",
        "cross_encoder": "+ Cross-Encoder      ",
        "ppr": "+ PPR                ",
        "graphsage": "+ GraphSAGE          ",
        "rgcn": "+ R-GCN              ",
        "rgcn_constraint": "+ R-GCN + Constraint ",
        "qfe_rgcn": "Final Graph v1 (Ours)",
        "qfe_rgcn_v2": "Final Graph v2 (Ours)",
        "qfe_rgcn_ppr": "+ QFE-RGCN v1 + PPR  ",
        "mono_t5": "MonoT5               ",
        "list_t5": "ListT5               ",
        "fast_final_graph": "Fast Final Graph (Ours)",
    }
    header = f"{'Method':<30} {'MRR':>7}"
    for k in k_values:
        header += f" {'R@'+str(k):>8} {'nDCG@'+str(k):>8}"
    print(header)
    print("-" * (30 + 7 + 18 * len(k_values)))
    for method in METHOD_ORDER:
        m = summaries.get(method)
        if m is None:
            continue
        label = print_labels.get(method, method)
        row = f"{label:<30} {m['mrr']:>7.4f}"
        for k in k_values:
            row += f" {m.get(f'recall@{k}', 0):>8.4f} {m.get(f'ndcg@{k}', 0):>8.4f}"
        print(row)

    print(f"\n  Total time: {total_time:.1f}s")
    print(f"  Methods evaluated: {list(all_results.keys())}")

    # ---- Write outputs ----
    _write_outputs(
        output_dir, all_results, summaries, k_values,
        command=" ".join(sys.argv),
    )

    print(f"\nOutput: {output_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
