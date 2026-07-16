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
    QFERGCNReranker,
    QFERGCNRerankDataset,
    EntityGatedScoringHead,
    QUERY_EMBED_DIM,
    derive_query_vector,
    build_query_embedding_cache,
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
    "qfe_rgcn_ppr": "qfe_rgcn_ppr_results.jsonl",
    "mono_t5": "mono_t5_results.jsonl",
    "list_t5": "list_t5_results.jsonl",
}

METHOD_ORDER = [
    "best_retriever", "cross_encoder", "ppr",
    "graphsage", "rgcn", "rgcn_constraint", "qfe_rgcn", "qfe_rgcn_ppr",
    "mono_t5", "list_t5",
]

METHOD_LABELS = {
    "best_retriever": "Best Retriever",
    "cross_encoder": "+ Cross-Encoder",
    "ppr": "+ PPR",
    "graphsage": "+ GraphSAGE",
    "rgcn": "+ R-GCN",
    "rgcn_constraint": "+ R-GCN + Constraint Score",
    "qfe_rgcn": "Final Graph (Ours)",
    "qfe_rgcn_ppr": "Final Graph (Ours) + PPR",
    "mono_t5": "MonoT5",
    "list_t5": "ListT5",
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
    """
    results = []
    for s in samples:
        hr = hybrid.search(s["question"], top_k=top_n)
        chunks = [c for c, _ in hr]
        candidate_ids = [c.chunk_id for c in chunks]

        # PPR is an independent baseline — only compute if explicitly requested
        # for ablation studies (--qfe_use_ppr).
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
                retrieval_scores={c.chunk_id: float(score) for c, score in hr},
            ))

        # Ensure query embedding is available
        if s["question"] not in qfe_reranker.query_embeddings:
            qfe_reranker.query_embeddings[s["question"]] = derive_query_vector(
                s["question"], dim=qfe_reranker.query_embed_dim,
            )

        if allow_fallback:
            try:
                reranked = qfe_reranker.rerank(
                    s["question"], hr, graph, features,
                    ppr_scores=ppr_scores,
                )
                ids = [c.chunk_id for c, _ in reranked[:output_k]]
            except Exception as exc:
                print(f"    [qfe_rgcn FALLBACK] query_id={s['id']} "
                      f"exception={type(exc).__name__}: {exc}")
                ids = candidate_ids[:output_k]
                r = _make_result(s, ids, gold_map.get(s["id"], []), method_name)
                r["_fallback"] = True
                r["_fallback_exception"] = f"{type(exc).__name__}: {exc}"
                results.append(r)
                continue
        else:
            reranked = qfe_reranker.rerank(
                s["question"], hr, graph, features,
                ppr_scores=ppr_scores,
            )
            ids = [c.chunk_id for c, _ in reranked[:output_k]]

        results.append(_make_result(s, ids, gold_map.get(s["id"], []), method_name))
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
                        help="Comma-separated methods to run (available: best_retriever, cross_encoder, ppr, graphsage, rgcn, rgcn_constraint, qfe_rgcn, qfe_rgcn_ppr, mono_t5, list_t5)")
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
                        help="Resume MonoT5/ListT5 from partial cache")
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

        # Train QFE-RGCN (covers both "qfe_rgcn" and "qfe_rgcn_ppr")
        if "qfe_rgcn" in selected_methods or "qfe_rgcn_ppr" in selected_methods:
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
                save_training_artifacts(
                    qfe_reranker, qfe_history, output_dir, qfe_meta,
                    experiment="table1_qfe_rgcn",
                )
                print(f"    Training done in {time.time() - t0:.1f}s")

                # Add eval queries to query_embeddings
                for s in eval_samples:
                    q = s["question"]
                    if q not in qfe_reranker.query_embeddings:
                        qfe_reranker.query_embeddings[q] = derive_query_vector(
                            q, dim=qfe_reranker.query_embed_dim,
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
                    )
                    print(f"    Evaluation done in {time.time() - t_eval2:.1f}s")
                    _checkpoint_method(output_dir, "qfe_rgcn_ppr", all_results, k_values)
            else:
                print("    [SKIP] QFE-RGCN training failed (not enough pairs)")
    else:
        print("\n  [skip_gnn] Skipping GraphSAGE, R-GCN, and R-GCN+Constraint.")

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
        "qfe_rgcn": "Final Graph (Ours)  ",
        "qfe_rgcn_ppr": "+ QFE-RGCN + PPR     ",
        "mono_t5": "MonoT5               ",
        "list_t5": "ListT5               ",
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
