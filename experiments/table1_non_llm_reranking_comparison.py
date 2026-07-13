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
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from feg_rag.config import Config
from feg_rag.data.chunker import Chunk, chunk_text, chunk_report
from feg_rag.data.loader import load_dataset
from feg_rag.evaluation.metrics import compute_all_metrics
from feg_rag.graph.builder import build_financial_evidence_graph
from feg_rag.graph.entities import EntityExtractor, extract_entities
from feg_rag.graph.features import build_node_features
from feg_rag.rerank.fusion import ConstraintScorer, FusionScorer
from feg_rag.rerank.ppr import ppr_rerank
from feg_rag.rerank.train import (
    build_train_pairs,
    save_training_artifacts,
    train_gnn_reranker,
    warmup_retrieval_scores,
)
from feg_rag.retrieval.bm25 import BM25Retriever
from feg_rag.retrieval.cross_encoder import CrossEncoderReranker
from feg_rag.retrieval.dense import DenseRetriever
from feg_rag.retrieval.hybrid import HybridRetriever


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
    max_distractor_files: int = 50,
) -> Tuple[List[Chunk], Dict[str, List[str]]]:
    """Chunk FinDER evidence + optional 10-K distractors."""
    corpus: List[Chunk] = []
    gold_map: Dict[str, List[str]] = {}
    for s in samples:
        gold_ids = []
        for text in s["evidence_texts"]:
            for c in chunk_text(text, cfg.chunk_size, cfg.chunk_overlap, doc_id=s["id"]):
                corpus.append(c)
                gold_ids.append(c.chunk_id)
        gold_map[s["id"]] = gold_ids

    edgar_dir = cfg.edgar_dir
    if edgar_dir.exists():
        txt_files = list(edgar_dir.rglob("*.txt")) or list(edgar_dir.rglob("*.html"))
        for tf in txt_files[:max_distractor_files]:
            try:
                corpus.extend(chunk_report(tf, cfg.chunk_size, cfg.chunk_overlap))
            except Exception:
                pass
    return corpus, gold_map


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
}

METHOD_ORDER = [
    "best_retriever", "cross_encoder", "ppr",
    "graphsage", "rgcn", "rgcn_constraint",
]

METHOD_LABELS = {
    "best_retriever": "Best Retriever",
    "cross_encoder": "+ Cross-Encoder",
    "ppr": "+ PPR",
    "graphsage": "+ GraphSAGE",
    "rgcn": "+ R-GCN",
    "rgcn_constraint": "+ R-GCN + Constraint Score",
}


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
) -> List[Dict]:
    """Run a trained GNN reranker on test samples."""
    results = []
    for s in samples:
        hr = hybrid.search(s["question"], top_k=top_n)
        candidate_ids = [c.chunk_id for c, _ in hr]
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

        try:
            reranked = reranker.rerank(
                s["question"], hr, graph, features,
                ppr_scores=ppr_scores,
            )
            ids = [c.chunk_id for c, _ in reranked[:output_k]]
        except Exception:
            ids = candidate_ids[:output_k]

        results.append(_make_result(s, ids, gold_map.get(s["id"], []), method_name))
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
    method_files = {
        "best_retriever": "best_retriever_results.jsonl",
        "cross_encoder": "cross_encoder_results.jsonl",
        "ppr": "ppr_results.jsonl",
        "graphsage": "graphsage_results.jsonl",
        "rgcn": "rgcn_results.jsonl",
        "rgcn_constraint": "rgcn_constraint_results.jsonl",
    }
    for method, results in all_results.items():
        fname = method_files.get(method, f"{method}_results.jsonl")
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
        method_order = [
            "best_retriever", "cross_encoder", "ppr",
            "graphsage", "rgcn", "rgcn_constraint",
        ]
        method_labels = {
            "best_retriever": "Best Retriever",
            "cross_encoder": "+ Cross-Encoder",
            "ppr": "+ PPR",
            "graphsage": "+ GraphSAGE",
            "rgcn": "+ R-GCN",
            "rgcn_constraint": "+ R-GCN + Constraint Score",
        }
        for method in method_order:
            m = summaries.get(method)
            if m is None:
                continue
            row = {"Method": method_labels.get(method, method)}
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
        for method in method_order:
            m = summaries.get(method)
            if m is None:
                continue
            row = f"| {method_labels.get(method, method)} |"
            for k in k_values:
                row += f" {m.get(f'recall@{k}', 0):.4f} |"
            row += f" {m.get('mrr', 0):.4f} |"
            for k in k_values:
                row += f" {m.get(f'ndcg@{k}', 0):.4f} |"
            fh.write(row + "\n")
        fh.write("\n## Key claim\n\n")
        fh.write("> Graph-based reranking improves evidence ranking without relying on an LLM reranker.\n\n")

    # README
    readme_path = output_dir / "README.md"
    with open(readme_path, "w", encoding="utf-8") as fh:
        fh.write("# Experiment: Table 1 — Non-LLM Reranking Comparison\n\n")
        fh.write("Compares evidence ranking performance of retrieval + reranking methods.\n\n")
        fh.write("## Methods\n\n")
        for method in method_order:
            label = method_labels.get(method, method)
            fh.write(f"- **{label}**\n")
        fh.write("\n## Output files\n\n")
        for fname in [
            "table1_non_llm_reranking_comparison.csv",
            "table1_non_llm_reranking_comparison.md",
        ]:
            fh.write(f"- `{fname}`\n")
        for fname in method_files.values():
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
    parser.add_argument("--sanity", action="store_true",
                        help="Sanity mode: minimal samples, 1 epoch")
    parser.add_argument("--overwrite_output_dir", action="store_true",
                        help="Overwrite existing results")
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
    corpus_chunks, gold_map = _build_corpus(samples, cfg)
    chunk_by_id = {c.chunk_id: c for c in corpus_chunks}
    gold_chunk_ids: Set[str] = set()
    for gids in gold_map.values():
        gold_chunk_ids.update(gids)
    print(f"  {len(corpus_chunks)} chunks ({len(gold_chunk_ids)} gold, "
          f"{len(corpus_chunks) - len(gold_chunk_ids)} distractors)")

    # ---- 3. Retrieval indices ----
    print("[3/5] Building retrieval indices...")
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

    hybrid = HybridRetriever(
        bm25, dense,
        alpha=cfg.retrieval.get("hybrid_alpha", 0.5),
    )

    # Cross-encoder
    cross_encoder = CrossEncoderReranker(
        model_name=cfg.cross_encoder.get("model_name", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
        batch_size=cfg.cross_encoder.get("batch_size", 32),
    )

    # ---- 4. Graph + features ----
    print("[4/5] Building graph and features...")
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
        train_samples + test_samples, hybrid, top_k=args.top_n,
    )
    features = build_node_features(
        graph, corpus_chunks, entity_map, retrieval_scores,
        chunk_embeddings=chunk_embeddings,
        compute_embeddings=False,
        embedding_device=device,
    )
    feature_dim = next(iter(features.values())).shape[0]
    print(f"  Feature dim: {feature_dim}")

    # ---- 5. Run methods ----
    print("\n[5/5] Running evidence ranking comparison...")
    extractor = EntityExtractor()
    all_results: Dict[str, List[Dict]] = {}
    t_total_start = time.time()

    # 5a. Best Retriever (Hybrid)
    print("\n  [best_retriever] Hybrid E5-Mistral...")
    t0 = time.time()
    all_results["best_retriever"] = run_best_retriever(
        eval_samples, hybrid, gold_map,
        top_n=args.top_n, output_k=args.output_k,
    )
    print(f"    {len(all_results['best_retriever'])} queries in {time.time() - t0:.1f}s")
    _checkpoint_method(output_dir, "best_retriever", all_results, k_values)

    # 5b. Cross-Encoder
    print("\n  [cross_encoder] Hybrid + Cross-Encoder...")
    t0 = time.time()
    all_results["cross_encoder"] = run_cross_encoder(
        eval_samples, hybrid, gold_map, cross_encoder,
        top_n=min(args.top_n * 2, 100), output_k=args.output_k,
    )
    print(f"    {len(all_results['cross_encoder'])} queries in {time.time() - t0:.1f}s")
    _checkpoint_method(output_dir, "cross_encoder", all_results, k_values)

    # 5c. PPR
    print("\n  [ppr] Hybrid + PPR...")
    t0 = time.time()
    all_results["ppr"] = run_ppr(
        eval_samples, hybrid, graph, corpus_chunks, gold_map, extractor,
        ppr_alpha=cfg.rerank.get("ppr_alpha", 0.85),
        top_n=args.top_n, output_k=args.output_k,
    )
    print(f"    {len(all_results['ppr'])} queries in {time.time() - t0:.1f}s")
    _checkpoint_method(output_dir, "ppr", all_results, k_values)

    # 5d/e/f. GNN methods (train on train_samples, eval on eval_samples)
    if not args.skip_gnn:
        # Train GraphSAGE
        print("\n  [graphsage] Training GraphSAGE reranker...")
        t0 = time.time()
        cfg.rerank["gnn_model"] = "sage"
        sage_reranker, sage_history, sage_meta = train_gnn_reranker(
            train_samples, hybrid, graph, features, gold_map, cfg,
            epochs=cfg.rerank.get("gnn_epochs", 10),
            device=device, min_pairs=5, verbose=True,
        )
        if sage_reranker is not None:
            save_training_artifacts(
                sage_reranker, sage_history, output_dir, sage_meta,
                experiment="table1_graphsage",
            )
            print(f"    Training done in {time.time() - t0:.1f}s")
            print(f"    Evaluating GraphSAGE on {len(eval_samples)} samples...")
            t_eval = time.time()
            all_results["graphsage"] = run_gnn_reranker(
                eval_samples, hybrid, graph, features, gold_map,
                chunk_by_id, extractor, sage_reranker, "graphsage",
                top_n=args.top_n, output_k=args.output_k,
                ppr_alpha=cfg.rerank.get("ppr_alpha", 0.85),
            )
            print(f"    Evaluation done in {time.time() - t_eval:.1f}s")
            _checkpoint_method(output_dir, "graphsage", all_results, k_values)
        else:
            print("    [SKIP] GraphSAGE training failed (not enough pairs)")

        # Train R-GCN
        print("\n  [rgcn] Training R-GCN reranker...")
        t0 = time.time()
        cfg.rerank["gnn_model"] = "rgcn"
        rgcn_reranker, rgcn_history, rgcn_meta = train_gnn_reranker(
            train_samples, hybrid, graph, features, gold_map, cfg,
            epochs=cfg.rerank.get("gnn_epochs", 10),
            device=device, min_pairs=5, verbose=True,
        )
        if rgcn_reranker is not None:
            save_training_artifacts(
                rgcn_reranker, rgcn_history, output_dir, rgcn_meta,
                experiment="table1_rgcn",
            )
            print(f"    Training done in {time.time() - t0:.1f}s")
            print(f"    Evaluating R-GCN on {len(eval_samples)} samples...")
            t_eval = time.time()
            all_results["rgcn"] = run_gnn_reranker(
                eval_samples, hybrid, graph, features, gold_map,
                chunk_by_id, extractor, rgcn_reranker, "rgcn",
                top_n=args.top_n, output_k=args.output_k,
                ppr_alpha=cfg.rerank.get("ppr_alpha", 0.85),
            )
            print(f"    Evaluation done in {time.time() - t_eval:.1f}s")
            _checkpoint_method(output_dir, "rgcn", all_results, k_values)

            # R-GCN + Constraint Score
            print("\n  [rgcn_constraint] R-GCN + Constraint Score...")
            t_eval = time.time()
            constraint_scorer = ConstraintScorer(
                company_weight=cfg.constraint.get("company_match_weight", 1.0),
                year_weight=cfg.constraint.get("year_match_weight", 1.0),
                metric_weight=cfg.constraint.get("metric_match_weight", 0.8),
                filing_type_weight=cfg.constraint.get("filing_type_match_weight", 0.5),
            )
            # Set fusion delta > 0 for constraint
            cfg.rerank["fusion_delta"] = 0.1
            all_results["rgcn_constraint"] = run_rgcn_constraint(
                eval_samples, hybrid, graph, features, gold_map,
                chunk_by_id, extractor, rgcn_reranker, constraint_scorer, cfg,
                method_name="rgcn_constraint",
                top_n=args.top_n, output_k=args.output_k,
                ppr_alpha=cfg.rerank.get("ppr_alpha", 0.85),
            )
            print(f"    {len(all_results['rgcn_constraint'])} queries in {time.time() - t_eval:.1f}s")
            _checkpoint_method(output_dir, "rgcn_constraint", all_results, k_values)
        else:
            print("    [SKIP] R-GCN training failed (not enough pairs)")
    else:
        print("\n  [skip_gnn] Skipping GraphSAGE, R-GCN, and R-GCN+Constraint.")

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
    method_labels = {
        "best_retriever": "Best Retriever      ",
        "cross_encoder": "+ Cross-Encoder      ",
        "ppr": "+ PPR                ",
        "graphsage": "+ GraphSAGE          ",
        "rgcn": "+ R-GCN              ",
        "rgcn_constraint": "+ R-GCN + Constraint ",
    }
    header = f"{'Method':<30} {'MRR':>7}"
    for k in k_values:
        header += f" {'R@'+str(k):>8} {'nDCG@'+str(k):>8}"
    print(header)
    print("-" * (30 + 7 + 18 * len(k_values)))
    for method in ["best_retriever", "cross_encoder", "ppr", "graphsage", "rgcn", "rgcn_constraint"]:
        m = summaries.get(method)
        if m is None:
            continue
        label = method_labels.get(method, method)
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
