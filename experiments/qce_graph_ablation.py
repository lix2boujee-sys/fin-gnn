"""QCE-Graph Lite Ablation Study.

Primary methods (strict top-50 reranker, no candidate expansion):
    initial_retriever  — baseline retrieval
    rgcn               — read existing R-GCN results (no retraining)
    qce_rerank         — residual QCE reranker with R-GCN
    qce_rerank_no_rgcn — residual QCE reranker without R-GCN

Expansion ablation (kept for diagnostics, NOT default):
    qce_expansion_fixed
    qce_expansion_router
    qce_expansion_full

Usage:
    # Smoke test
    python experiments/qce_graph_ablation.py --sanity --device cpu --epochs 2

    # Full run
    python experiments/qce_graph_ablation.py \\
        --methods initial_retriever,rgcn,qce_rerank,qce_rerank_no_rgcn \\
        --initial_results_jsonl outputs/.../bge_m3_dense_results.jsonl \\
        --rgcn_results_jsonl outputs/.../rgcn_results.jsonl \\
        --corpus_cache cache/table1_full_corpus_seq4096.pkl \\
        --graph_cache cache/table2_graph_features_bge_pool_seq4096.pkl \\
        --top_n 50 --epochs 10 --eval_scope all --device cuda
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import random
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from feg_rag.config import Config
from feg_rag.data.chunker import Chunk
from feg_rag.data.corpus import build_benchmark_corpus
from feg_rag.data.loader import load_dataset
from feg_rag.evaluation.metrics import compute_all_metrics
from feg_rag.graph.builder import FinancialEvidenceGraph, build_financial_evidence_graph
from feg_rag.graph.entities import EntityExtractor, extract_entities
from feg_rag.graph.features import build_node_features
from feg_rag.rerank.qce_expansion import (
    RELATION_NAMES, NUM_RELATIONS, DEFAULT_RELATION_PRIOR,
    GraphExpansionIndex, BudgetedGraphExpander, ExpandedCandidate,
    compute_expansion_diagnostics,
)
from feg_rag.rerank.qce_features import (
    QUERY_FEATURE_DIM_QCE, SUPPORT_FEATURE_DIM, CONFLICT_FEATURE_DIM,
    build_qce_query_features, extract_support_features, extract_conflict_features,
)
from feg_rag.rerank.qce_graph import (
    QueryRelationRouter, CounterfactualEvidenceScorer,
    QCEGraphLiteReranker, QCEInferencePipeline, QCEFixedCandidatePipeline,
    compute_qce_loss, save_qce_checkpoint, load_qce_checkpoint,
)
from feg_rag.rerank.qce_dataset import (
    build_qce_training_candidates, build_qce_rerank_candidates,
    QCERerankDataset, collate_qce_pairs,
)
from feg_rag.rerank.scoring import normalise_score_map
from feg_rag.retrieval.bm25 import BM25Retriever
from feg_rag.retrieval.dense import DenseRetriever
from feg_rag.retrieval.hybrid import HybridRetriever


# ═════════════════════════════════════════════════════════════════════════════
# Constants
# ═════════════════════════════════════════════════════════════════════════════

PRIMARY_METHODS = [
    "initial_retriever", "rgcn",
    "qce_rerank", "qce_rerank_no_rgcn",
]

EXPANSION_METHODS = [
    "qce_expansion_fixed", "qce_expansion_router", "qce_expansion_full",
]

METHOD_ORDER = PRIMARY_METHODS + EXPANSION_METHODS

METHOD_LABELS = {
    "initial_retriever": "Initial Retriever",
    "rgcn": "R-GCN (baseline)",
    "qce_rerank": "QCE Rerank",
    "qce_rerank_no_rgcn": "QCE Rerank (no R-GCN)",
    "qce_expansion_fixed": "QCE Expansion Fixed",
    "qce_expansion_router": "QCE Expansion Router",
    "qce_expansion_full": "QCE Expansion Full",
}

DEFAULT_METHODS = "initial_retriever,rgcn,qce_rerank,qce_rerank_no_rgcn"


# ═════════════════════════════════════════════════════════════════════════════
# Candidate Pool Retriever
# ═════════════════════════════════════════════════════════════════════════════

class CandidatePoolRetriever:
    """Replay a fixed candidate pool from a prior retriever JSONL."""

    def __init__(self, results_jsonl: str | Path, chunk_by_id: Dict[str, Chunk]):
        self.results_jsonl = Path(results_jsonl)
        self.chunk_by_id = chunk_by_id
        self._by_question: Dict[str, List[Tuple[Chunk, float]]] = {}
        self._load()

    def _load(self) -> None:
        if not self.results_jsonl.exists():
            raise FileNotFoundError(f"Candidate results not found: {self.results_jsonl}")
        with self.results_jsonl.open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                question = row.get("question", "")
                ids = row.get("retrieved_chunk_ids", [])
                if question and ids:
                    results = []
                    for rank, cid in enumerate(ids):
                        chunk = self.chunk_by_id.get(cid)
                        if chunk is not None:
                            results.append((chunk, float(len(ids) - rank)))
                    if results:
                        self._by_question[question] = results

    def search(self, query: str, top_k: int = 50):
        return self._by_question.get(query, [])[:top_k]

    @property
    def num_queries(self) -> int:
        return len(self._by_question)


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════

def _default_device() -> str:
    try:
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _resolve_path(path_str: Optional[str], root_dir: Path) -> Optional[Path]:
    if not path_str:
        return None
    p = Path(path_str)
    if not p.is_absolute():
        p = root_dir / p
    return p


def _write_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_jsonl(path: str | Path) -> List[Dict]:
    results = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def _load_rgcn_scores(jsonl_path: str | Path) -> Dict[str, Dict[str, float]]:
    """Load R-GCN results: question_id -> {chunk_id: rank_score}."""
    scores: Dict[str, Dict[str, float]] = {}
    path = Path(jsonl_path)
    if not path.exists():
        print(f"  [WARN] R-GCN results not found: {path}")
        return scores
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            qid = row.get("question_id", "")
            cids = row.get("retrieved_chunk_ids", [])
            if qid and cids:
                scores[qid] = {
                    cid: float(len(cids) - i) for i, cid in enumerate(cids)
                }
    return scores


# ═════════════════════════════════════════════════════════════════════════════
# Method runners
# ═════════════════════════════════════════════════════════════════════════════

def run_initial_retriever(
    samples: List[Dict], retriever, gold_map: Dict[str, List[str]],
    top_n: int = 50, output_k: int = 10,
) -> List[Dict]:
    results = []
    for s in samples:
        hr = retriever.search(s["question"], top_k=top_n)
        ids = [c.chunk_id for c, _ in hr[:output_k]]
        results.append({
            "question_id": s["id"], "question": s["question"],
            "gold_evidence_ids": gold_map.get(s["id"], []),
            "retrieved_chunk_ids": ids, "method": "initial_retriever",
        })
    return results


def run_rgcn_readonly(
    samples: List[Dict], rgcn_results_jsonl: str | Path,
    gold_map: Dict[str, List[str]], output_k: int = 10,
) -> List[Dict]:
    rgcn_data = _read_jsonl(rgcn_results_jsonl)
    by_question: Dict[str, List[str]] = {}
    for row in rgcn_data:
        q = row.get("question", row.get("question_id", ""))
        cids = row.get("retrieved_chunk_ids", [])
        if q:
            by_question[q] = cids

    results = []
    for s in samples:
        cids = by_question.get(s["question"], by_question.get(s["id"], []))[:output_k]
        results.append({
            "question_id": s["id"], "question": s["question"],
            "gold_evidence_ids": gold_map.get(s["id"], []),
            "retrieved_chunk_ids": cids, "method": "rgcn",
        })
    return results


def run_qce_fixed_pipeline(
    samples: List[Dict], retriever, pipeline: QCEFixedCandidatePipeline,
    gold_map: Dict[str, List[str]], method_name: str,
    top_n: int = 50, output_k: int = 10,
    progress_every: int = 100,
) -> Tuple[List[Dict], List[Dict]]:
    """Run QCE fixed-candidate reranker and collect debug examples."""
    results = []
    debug_examples = []
    t0 = time.time()

    for idx, s in enumerate(samples):
        hr = retriever.search(s["question"], top_k=top_n)
        gold = set(gold_map.get(s["id"], []))

        ranked, meta = pipeline.rerank(s["question"], hr, output_k=output_k)
        ranked_ids = [cid for cid, _ in ranked]

        # Build debug entry
        top10_before = [c.chunk_id for c, _ in hr[:output_k]]
        top10_after = ranked_ids
        moved_up_gold = []
        moved_down_conflicts = []

        # Check which gold moved up
        before_ranks = {c.chunk_id: i for i, (c, _) in enumerate(hr, start=1)}
        after_ranks = {cid: i for i, cid in enumerate(ranked_ids, start=1)}
        for gid in gold:
            br = before_ranks.get(gid, 999)
            ar = after_ranks.get(gid, 999)
            if ar < br:
                moved_up_gold.append({"chunk_id": gid, "before_rank": br, "after_rank": ar})

        debug_examples.append({
            "question_id": s["id"], "question": s["question"],
            "gold_evidence_ids": list(gold),
            "top10_before": top10_before, "top10_after": top10_after,
            "moved_up_gold": moved_up_gold,
            "moved_down_conflicts": moved_down_conflicts,
            "top_candidates": meta.get("debug_candidates", []),
            "relation_probabilities": meta.get("relation_probabilities", {}),
            "method": method_name,
        })

        results.append({
            "question_id": s["id"], "question": s["question"],
            "gold_evidence_ids": list(gold),
            "initial_chunk_ids": meta.get("initial_chunk_ids", []),
            "expanded_chunk_ids": meta.get("expanded_chunk_ids", []),
            "retrieved_chunk_ids": ranked_ids,
            "relation_probabilities": meta.get("relation_probabilities", {}),
            "method": method_name,
        })

        if progress_every > 0 and (idx + 1) % progress_every == 0:
            elapsed = time.time() - t0
            rate = (idx + 1) / max(elapsed, 1e-6)
            eta = (len(samples) - idx - 1) / max(rate, 1e-6)
            print(f"  [{method_name}] {idx + 1}/{len(samples)} "
                  f"elapsed={elapsed:.1f}s eta={eta:.1f}s")

    return results, debug_examples


# ═════════════════════════════════════════════════════════════════════════════
# Training
# ═════════════════════════════════════════════════════════════════════════════

def train_qce_model(
    model: QCEGraphLiteReranker,
    train_dataset: QCERerankDataset,
    val_dataset: Optional[QCERerankDataset] = None,
    epochs: int = 30, batch_size: int = 512, lr: float = 0.001,
    weight_decay: float = 0.00001,
    lambda_router: float = 0.0, lambda_scale: float = 0.001,
    use_router_loss: bool = False,
    early_stopping_patience: int = 5, eval_every: int = 2,
    device: str = "cpu", progress_every: int = 100, verbose: bool = True,
) -> Tuple[List[float], List[float], Dict[str, Any]]:
    from torch.utils.data import DataLoader

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_qce_pairs,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_qce_pairs,
        )

    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    best_epoch, best_state, patience_counter = 0, None, 0
    t_start = time.time()

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_rank_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            loss_dict = compute_qce_loss(
                model, batch, lambda_router=lambda_router,
                lambda_scale=lambda_scale, use_router_loss=use_router_loss,
            )
            loss = loss_dict["loss"]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_rank_loss += loss_dict["rank_loss"].item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        train_losses.append(avg_loss)

        val_loss = None
        if val_loader is not None and ((epoch + 1) % eval_every == 0):
            model.eval()
            val_total = 0.0
            val_n = 0
            with torch.no_grad():
                for batch in val_loader:
                    loss_dict = compute_qce_loss(
                        model, batch, lambda_router=lambda_router,
                        lambda_scale=lambda_scale, use_router_loss=use_router_loss,
                    )
                    val_total += loss_dict["loss"].item()
                    val_n += 1
            val_loss = val_total / max(val_n, 1)
            val_losses.append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    if verbose:
                        print(f"  Early stopping at epoch {epoch + 1}")
                    break

        if verbose:
            s_scale = model.support_scale.item()
            c_scale = model.conflict_scale.item()
            e_scale = model.context_scale.item()
            elapsed = time.time() - t_start
            print(
                f"  Epoch {epoch + 1:>3}/{epochs} | loss={avg_loss:.4f} "
                f"(rank={epoch_rank_loss/max(n_batches,1):.4f}) | "
                f"scales: s={s_scale:.4f} c={c_scale:.4f} e={e_scale:.4f} | "
                f"{elapsed:.0f}s"
                + (f" val={val_loss:.4f}" if val_loss is not None else "")
            )

    total_time = time.time() - t_start
    if best_state is not None:
        model.load_state_dict(best_state)

    meta = {
        "epochs_completed": epoch + 1, "best_epoch": best_epoch,
        "best_val_loss": best_val_loss if best_val_loss != float("inf") else None,
        "train_time_s": round(total_time, 1),
        "param_count": model.parameter_count,
        "final_support_scale": model.support_scale.item(),
        "final_conflict_scale": model.conflict_scale.item(),
        "final_context_scale": model.context_scale.item(),
    }
    return train_losses, val_losses, meta


# ═════════════════════════════════════════════════════════════════════════════
# Output writers
# ═════════════════════════════════════════════════════════════════════════════

def write_outputs(
    output_dir: Path,
    all_results: Dict[str, List[Dict]],
    all_metrics: Dict[str, Dict],
    all_debug_examples: Dict[str, List[Dict]],
    k_values: List[int],
    config: Dict,
    command: str = "",
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Config snapshot
    with open(output_dir / "config_snapshot.yaml", "w", encoding="utf-8") as fh:
        import yaml
        yaml.dump(config, fh, default_flow_style=False)

    # Per-method result JSONL
    for method, results in all_results.items():
        _write_jsonl(output_dir / f"{method}_results.jsonl", results)

    # Per-method debug examples
    for method, debug_entries in all_debug_examples.items():
        if debug_entries:
            with open(output_dir / f"{method}_debug_examples.json", "w", encoding="utf-8") as fh:
                json.dump(debug_entries, fh, indent=2, ensure_ascii=False)

    # Feature diagnostics
    diagnostics = {}
    for method in METHOD_ORDER:
        m = all_metrics.get(method)
        if m is None:
            continue
        diagnostics[method] = {
            "method": METHOD_LABELS.get(method, method),
            "num_samples": m.get("num_samples", 0),
            "mrr": m.get("mrr", 0),
        }
        for k in k_values:
            diagnostics[method][f"recall@{k}"] = m.get(f"recall@{k}", 0)
            diagnostics[method][f"ndcg@{k}"] = m.get(f"ndcg@{k}", 0)

    with open(output_dir / "qce_feature_diagnostics.json", "w", encoding="utf-8") as fh:
        json.dump(diagnostics, fh, indent=2, ensure_ascii=False)

    # Metrics CSV
    with open(output_dir / "metrics_summary.csv", "w", encoding="utf-8", newline="") as fh:
        fieldnames = (["Method"] + [f"Recall@{k}" for k in k_values]
                      + ["MRR"] + [f"nDCG@{k}" for k in k_values] + ["num_samples"])
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for method in METHOD_ORDER:
            m = all_metrics.get(method)
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

    # Run log
    with open(output_dir / "run.log", "w", encoding="utf-8") as fh:
        fh.write(f"Command: {command}\n")
        fh.write(f"Timestamp: {datetime.now().isoformat()}\n")
        fh.write(json.dumps(diagnostics, indent=2))


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="QCE-Graph Lite Ablation Study")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output_dir", default="outputs/qce_graph")
    parser.add_argument("--num_samples", type=int, default=0)
    parser.add_argument("--sanity", action="store_true")
    parser.add_argument("--device", default=_default_device())
    parser.add_argument("--seeds", default="42", help="Comma-separated seeds")
    parser.add_argument("--methods", default=DEFAULT_METHODS)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--progress_every", type=int, default=100)
    parser.add_argument("--eval_scope", default="all",
                        choices=["all", "heldout"],
                        help="all = full 5703, heldout = val split only")

    # Data paths
    parser.add_argument("--initial_results_jsonl", default=None)
    parser.add_argument("--rgcn_results_jsonl", default=None)
    parser.add_argument("--graph_cache", default=None)
    parser.add_argument("--corpus_cache", default=None)

    # Rerank params
    parser.add_argument("--top_n", type=int, default=50)
    parser.add_argument("--expansion_budget", type=int, default=30)
    parser.add_argument("--max_budget_per_relation", type=int, default=10)
    parser.add_argument("--max_total_candidates", type=int, default=80)

    # Training params
    parser.add_argument("--lambda_scale", type=float, default=0.001)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--split_seed", type=int, default=42)

    # Output control
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--cache_dir", default="outputs/qce_graph/cache")

    args = parser.parse_args()

    # Sanity overrides
    if args.sanity:
        args.num_samples = args.num_samples or 20
        args.epochs = min(args.epochs, 3)
        args.top_n = 20
        args.expansion_budget = 10
        args.max_total_candidates = 30
        if args.output_dir == "outputs/qce_graph":
            args.output_dir = "outputs/qce_graph/sanity"

    cfg = Config.from_yaml(args.config)
    root_dir = cfg.root_dir

    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    selected_methods = {m.strip() for m in args.methods.split(",") if m.strip()}
    k_values = [5, 10]
    device = args.device

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = root_dir / output_dir

    cache_dir = Path(args.cache_dir)
    if not cache_dir.is_absolute():
        cache_dir = root_dir / cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  QCE-Graph Lite Ablation Study")
    print("=" * 60)
    print(f"  Output:       {output_dir}")
    print(f"  Device:       {device}")
    print(f"  Seeds:        {seeds}")
    print(f"  Methods:      {sorted(selected_methods)}")
    print(f"  Eval scope:   {args.eval_scope}")
    print(f"  Epochs:       {args.epochs}")
    print(f"  Top-N:        {args.top_n}")
    print(f"  Sanity:       {args.sanity}")

    # ---- 1. Load data ----
    print("\n[1/6] Loading data...")
    samples = load_dataset("finder", cfg.data_dir)
    if args.num_samples > 0:
        samples = samples[: args.num_samples]

    corpus_cache_path = _resolve_path(args.corpus_cache, root_dir)
    if corpus_cache_path and corpus_cache_path.exists():
        with open(corpus_cache_path, "rb") as fh:
            cache_data = pickle.load(fh)
        corpus_chunks = cache_data["corpus_chunks"]
        gold_map = cache_data["gold_map"]
        print(f"  Loaded corpus from cache: {corpus_cache_path}")
    else:
        corpus_chunks, gold_map, _ = build_benchmark_corpus(samples, cfg)
        if corpus_cache_path:
            corpus_cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(corpus_cache_path, "wb") as fh:
                pickle.dump({"corpus_chunks": corpus_chunks, "gold_map": gold_map}, fh)

    chunk_by_id = {c.chunk_id: c for c in corpus_chunks}
    print(f"  {len(corpus_chunks)} chunks, {len(samples)} queries")

    # ---- 2. Build retriever ----
    print("\n[2/6] Building retriever...")
    if args.initial_results_jsonl:
        initial_path = _resolve_path(args.initial_results_jsonl, root_dir)
        retriever = CandidatePoolRetriever(initial_path, chunk_by_id)
        print(f"  Candidate pool: {initial_path} ({retriever.num_queries} queries)")
    else:
        bm25 = BM25Retriever()
        bm25.index(corpus_chunks)
        dense = DenseRetriever(
            model_name=cfg.retrieval.get("dense_model", "all-MiniLM-L6-v2"),
            device="cpu",
        )
        dense.index(corpus_chunks)
        retriever = HybridRetriever(bm25, dense, alpha=0.5)
        print(f"  Built hybrid retriever")

    # ---- 3. Build graph and expansion index ----
    print("\n[3/6] Building graph and expansion index...")
    graph_cache_path = _resolve_path(args.graph_cache, root_dir)
    if graph_cache_path and graph_cache_path.exists():
        with open(graph_cache_path, "rb") as fh:
            graph_data = pickle.load(fh)
        graph = graph_data.get("graph")
        entity_map = graph_data.get("entity_map", {})
        print(f"  Loaded graph from cache: {graph_cache_path}")
    else:
        entity_map = extract_entities(corpus_chunks)
        graph = build_financial_evidence_graph(
            corpus_chunks, entity_map=entity_map,
            add_semantic_edges=False, add_company_nodes=True,
            add_filing_nodes=True, add_section_nodes=True,
            add_same_entity_edges=True, max_same_entity_edges=30,
            use_edge_weights=True,
        )
        print(f"  Built graph: {graph.num_nodes} nodes, {graph.num_edges} edges")

    index_cache_path = cache_dir / "graph_expansion_index.pkl"
    if index_cache_path.exists():
        print(f"  Loading expansion index from cache: {index_cache_path}")
        index = GraphExpansionIndex.load(index_cache_path)
    else:
        index = GraphExpansionIndex().build(corpus_chunks, graph)
        index.save(index_cache_path)
        print(f"  Saved expansion index to cache: {index_cache_path}")

    # ---- 4. Build expander (for expansion ablation only) ----
    expander = BudgetedGraphExpander(
        index=index, initial_top_n=args.top_n,
        seed_top_m=min(15, args.top_n),
        expansion_budget=args.expansion_budget,
        max_budget_per_relation=args.max_budget_per_relation,
        max_total_candidates=args.max_total_candidates,
    )

    # ---- 5. Load R-GCN scores ----
    rgcn_scores: Dict[str, Dict[str, float]] = {}
    if args.rgcn_results_jsonl:
        rgcn_path = _resolve_path(args.rgcn_results_jsonl, root_dir)
        rgcn_scores = _load_rgcn_scores(rgcn_path)
        print(f"  Loaded R-GCN scores for {len(rgcn_scores)} queries")

    # ---- 6. Run across seeds ----
    print(f"\n[4/6] Running methods (eval_scope={args.eval_scope})...")
    t_total = time.time()

    all_seed_results: Dict[str, List[Dict]] = defaultdict(list)
    all_seed_metrics: Dict[str, List[Dict]] = defaultdict(list)
    all_debug_examples: Dict[str, List[Dict]] = defaultdict(list)

    for seed in seeds:
        print(f"\n{'─' * 50}")
        print(f"  Seed {seed}")
        print(f"{'─' * 50}")

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # Split
        rng = random.Random(args.split_seed)
        shuffled = list(samples)
        rng.shuffle(shuffled)
        n_val = max(1, int(len(shuffled) * args.val_split))
        train_samples = shuffled[n_val:]
        heldout_samples = shuffled[:n_val]

        if args.eval_scope == "heldout":
            eval_samples = heldout_samples
        else:
            eval_samples = samples  # all

        print(f"  Train: {len(train_samples)}, Eval: {len(eval_samples)}")

        use_rgcn = bool(args.rgcn_results_jsonl)

        # ---- initial_retriever ----
        if "initial_retriever" in selected_methods:
            print(f"\n  [initial_retriever] Running...")
            t0 = time.time()
            results = run_initial_retriever(
                eval_samples, retriever, gold_map, top_n=args.top_n,
            )
            print(f"    {len(results)} queries in {time.time() - t0:.1f}s")
            all_seed_results["initial_retriever"].extend(results)
            er = compute_all_metrics("initial_retriever", results, k_values=k_values)
            all_seed_metrics["initial_retriever"].append({
                "seed": seed, "mrr": er.mrr,
                **{f"recall@{k}": er.evidence_recall.get(k, 0) for k in k_values},
                **{f"ndcg@{k}": er.ndcg.get(k, 0) for k in k_values},
            })

        # ---- rgcn (read-only) ----
        if "rgcn" in selected_methods:
            if not args.rgcn_results_jsonl:
                print("  [rgcn] SKIP: --rgcn_results_jsonl required")
            else:
                print(f"\n  [rgcn] Reading existing results...")
                t0 = time.time()
                results = run_rgcn_readonly(
                    eval_samples, args.rgcn_results_jsonl, gold_map,
                )
                print(f"    {len(results)} queries in {time.time() - t0:.1f}s")
                all_seed_results["rgcn"].extend(results)
                er = compute_all_metrics("rgcn", results, k_values=k_values)
                all_seed_metrics["rgcn"].append({
                    "seed": seed, "mrr": er.mrr,
                    **{f"recall@{k}": er.evidence_recall.get(k, 0) for k in k_values},
                    **{f"ndcg@{k}": er.ndcg.get(k, 0) for k in k_values},
                })

        # ---- QCE Rerank (primary, fixed-candidate) ----
        qce_rerank_methods = selected_methods & {"qce_rerank", "qce_rerank_no_rgcn"}
        if qce_rerank_methods:
            # Build fixed-candidate training pairs
            print(f"\n  Building fixed-candidate training pairs...")
            t0 = time.time()

            # Add retrieval_scores for training
            for s in train_samples:
                if "retrieval_scores" not in s:
                    hr = retriever.search(s["question"], top_k=args.top_n)
                    s["retrieval_scores"] = {c.chunk_id: float(s) for c, s in hr}

            train_pairs, pair_stats = build_qce_rerank_candidates(
                train_samples, retriever, chunk_by_id, gold_map,
                top_n=args.top_n,
                rgcn_scores=rgcn_scores if use_rgcn else None,
                seed=seed,
            )
            print(f"    Built {len(train_pairs)} training pairs in {time.time() - t0:.1f}s")

            if len(train_pairs) == 0:
                print("    [WARN] No training pairs — skipping QCE rerank methods")
            else:
                # Train/val split
                rng_pairs = random.Random(seed)
                pairs_shuffled = list(train_pairs)
                rng_pairs.shuffle(pairs_shuffled)
                n_val_pairs = max(1, int(len(pairs_shuffled) * args.val_split))
                val_pairs = pairs_shuffled[:n_val_pairs]
                tr_pairs = pairs_shuffled[n_val_pairs:]

                # Shared dataset for both qce_rerank and qce_rerank_no_rgcn
                shared_train_ds = QCERerankDataset(tr_pairs, use_rgcn_score=use_rgcn)
                shared_val_ds = QCERerankDataset(val_pairs, use_rgcn_score=use_rgcn)
                print(f"    Train pairs: {len(tr_pairs)}, Val pairs: {len(val_pairs)}")

                for method_name in sorted(qce_rerank_methods):
                    use_rgcn_this = (method_name == "qce_rerank" and use_rgcn)
                    print(f"\n  [{method_name}] Training (use_rgcn={use_rgcn_this})...")
                    t0 = time.time()

                    model = QCEGraphLiteReranker(use_rgcn_score=use_rgcn_this)
                    train_losses, val_losses, train_meta = train_qce_model(
                        model, shared_train_ds, shared_val_ds,
                        epochs=args.epochs, batch_size=args.batch_size,
                        lr=args.lr, lambda_scale=args.lambda_scale,
                        use_router_loss=False,
                        device=device, progress_every=args.progress_every,
                    )

                    print(f"    Training done in {time.time() - t0:.1f}s")
                    print(f"    Parameters: {train_meta['param_count']}")
                    print(f"    Scales: s={train_meta['final_support_scale']:.4f} "
                          f"c={train_meta['final_conflict_scale']:.4f} "
                          f"e={train_meta['final_context_scale']:.4f}")

                    # Save checkpoint
                    ckpt_path = output_dir / f"{method_name}_seed{seed}.pt"
                    save_qce_checkpoint(model, ckpt_path, meta=train_meta)

                    # Inference
                    pipeline = QCEFixedCandidatePipeline(
                        model=model, index=index, chunk_lookup=chunk_by_id,
                        device=device, initial_top_n=args.top_n,
                        use_rgcn_score=use_rgcn_this,
                    )

                    print(f"    Evaluating...")
                    t_eval = time.time()
                    results, debug_examples = run_qce_fixed_pipeline(
                        eval_samples, retriever, pipeline, gold_map,
                        method_name=method_name, top_n=args.top_n,
                        progress_every=args.progress_every,
                    )
                    print(f"    Evaluation done in {time.time() - t_eval:.1f}s")

                    all_seed_results[method_name].extend(results)
                    all_debug_examples[method_name].extend(debug_examples)

                    er = compute_all_metrics(method_name, results, k_values=k_values)
                    all_seed_metrics[method_name].append({
                        "seed": seed, "mrr": er.mrr,
                        **{f"recall@{k}": er.evidence_recall.get(k, 0) for k in k_values},
                        **{f"ndcg@{k}": er.ndcg.get(k, 0) for k in k_values},
                    })

        # ---- Expansion methods (ablation only) ----
        expansion_methods = selected_methods & {
            "qce_expansion_fixed", "qce_expansion_router", "qce_expansion_full",
        }
        if expansion_methods:
            print(f"\n  [expansion-ablation] Building expansion training pairs...")
            t0 = time.time()

            for s in train_samples:
                if "retrieval_scores" not in s:
                    hr = retriever.search(s["question"], top_k=args.top_n)
                    s["retrieval_scores"] = {c.chunk_id: float(s) for c, s in hr}

            exp_train_pairs, exp_stats = build_qce_training_candidates(
                train_samples, expander, index, chunk_by_id, gold_map,
                initial_top_n=args.top_n, seed=seed,
            )
            print(f"    Built {len(exp_train_pairs)} expansion pairs in {time.time() - t0:.1f}s")

            if len(exp_train_pairs) == 0:
                print("    [WARN] No expansion pairs — skipping expansion methods")
            else:
                rng_exp = random.Random(seed)
                exp_shuffled = list(exp_train_pairs)
                rng_exp.shuffle(exp_shuffled)
                n_val_exp = max(1, int(len(exp_shuffled) * args.val_split))
                exp_val = exp_shuffled[:n_val_exp]
                exp_tr = exp_shuffled[n_val_exp:]

                exp_train_ds = QCERerankDataset(exp_tr, use_rgcn_score=use_rgcn)
                exp_val_ds = QCERerankDataset(exp_val, use_rgcn_score=use_rgcn)

                print(f"    Expansion train: {len(exp_tr)}, val: {len(exp_val)}")

                for method_name in sorted(expansion_methods):
                    print(f"\n  [{method_name}] Training...")
                    t0 = time.time()

                    use_router = (method_name != "qce_expansion_fixed")

                    model = QCEGraphLiteReranker(use_rgcn_score=use_rgcn)
                    train_losses, val_losses, train_meta = train_qce_model(
                        model, exp_train_ds, exp_val_ds,
                        epochs=args.epochs, batch_size=args.batch_size,
                        lr=args.lr, lambda_scale=args.lambda_scale,
                        use_router_loss=use_router,
                        device=device, progress_every=args.progress_every,
                    )

                    print(f"    Training done in {time.time() - t0:.1f}s")

                    ckpt_path = output_dir / f"{method_name}_seed{seed}.pt"
                    save_qce_checkpoint(model, ckpt_path, meta=train_meta)

                    pipeline = QCEInferencePipeline(
                        model=model, expander=expander, index=index,
                        chunk_lookup=chunk_by_id, device=device,
                        initial_top_n=args.top_n, use_rgcn_score=use_rgcn,
                    )

                    results, _ = run_qce_model_expansion(
                        eval_samples, retriever, pipeline, gold_map,
                        method_name=method_name, top_n=args.top_n,
                        progress_every=args.progress_every,
                    )
                    all_seed_results[method_name].extend(results)

                    er = compute_all_metrics(method_name, results, k_values=k_values)
                    all_seed_metrics[method_name].append({
                        "seed": seed, "mrr": er.mrr,
                        **{f"recall@{k}": er.evidence_recall.get(k, 0) for k in k_values},
                        **{f"ndcg@{k}": er.ndcg.get(k, 0) for k in k_values},
                    })

    # ---- Compute mean ± std across seeds ----
    print(f"\n{'=' * 60}")
    print("  RESULTS (mean ± std over seeds)")
    print(f"{'=' * 60}")

    final_metrics: Dict[str, Dict] = {}
    for method in METHOD_ORDER:
        metrics_list = all_seed_metrics.get(method, [])
        if not metrics_list:
            continue

        n_seeds = len(metrics_list)
        aggr: Dict[str, Any] = {"method": method, "num_seeds": n_seeds}

        for mn in ["mrr"] + [f"recall@{k}" for k in k_values] + [f"ndcg@{k}" for k in k_values]:
            vals = [m[mn] for m in metrics_list]
            aggr[mn] = round(float(np.mean(vals)), 4)
            aggr[f"{mn}_std"] = round(float(np.std(vals)), 4)

        if all_seed_results.get(method):
            aggr["num_samples"] = len(all_seed_results[method]) // n_seeds if n_seeds > 0 else 0

        final_metrics[method] = aggr

        label = METHOD_LABELS.get(method, method)
        parts = [f"{label:<30}"]
        parts.append(f"MRR={aggr.get('mrr', 0):.4f}±{aggr.get('mrr_std', 0):.4f}")
        for k in k_values:
            parts.append(f"R@{k}={aggr.get(f'recall@{k}', 0):.4f}±{aggr.get(f'recall@{k}_std', 0):.4f}")
            parts.append(f"nDCG@{k}={aggr.get(f'ndcg@{k}', 0):.4f}±{aggr.get(f'ndcg@{k}_std', 0):.4f}")
        print("  ".join(parts))

    total_time = time.time() - t_total
    print(f"\n  Total time: {total_time:.1f}s ({total_time / 60:.1f}m)")

    # ---- Write outputs ----
    print(f"\n[5/6] Writing outputs...")
    write_outputs(
        output_dir, all_seed_results, final_metrics, all_debug_examples,
        k_values,
        config={"args": vars(args), "seeds": seeds, "eval_scope": args.eval_scope},
        command=" ".join(sys.argv),
    )

    print(f"\n[6/6] Done!")
    print(f"  Output: {output_dir}")


def run_qce_model_expansion(
    samples, retriever, pipeline, gold_map,
    method_name, top_n=50, output_k=10, progress_every=100,
) -> Tuple[List[Dict], List]:
    """Run QCE expansion model for evaluation (ablation)."""
    results = []
    t0 = time.time()

    for idx, s in enumerate(samples):
        hr = retriever.search(s["question"], top_k=top_n)
        ranked, meta = pipeline.rerank(s["question"], hr, output_k=output_k)
        ranked_ids = [cid for cid, _ in ranked]

        results.append({
            "question_id": s["id"], "question": s["question"],
            "gold_evidence_ids": gold_map.get(s["id"], []),
            "initial_chunk_ids": meta.get("initial_chunk_ids", []),
            "expanded_chunk_ids": meta.get("expanded_chunk_ids", []),
            "retrieved_chunk_ids": ranked_ids,
            "relation_probabilities": meta.get("relation_probabilities", {}),
            "method": method_name,
        })

        if progress_every > 0 and (idx + 1) % progress_every == 0:
            elapsed = time.time() - t0
            rate = (idx + 1) / max(elapsed, 1e-6)
            eta = (len(samples) - idx - 1) / max(rate, 1e-6)
            print(f"  [{method_name}] {idx + 1}/{len(samples)} "
                  f"elapsed={elapsed:.1f}s eta={eta:.1f}s")

    return results, []


if __name__ == "__main__":
    main()
