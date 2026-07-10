"""Experiment 4: GNN Evidence Reranker (GraphSAGE / R-GCN).

Trains GraphSAGE and R-GCN rerankers on the Financial Evidence Graph using
gold evidence + finance-specific hard negatives, then compares against
Hybrid, Cross-Encoder, and PPR baselines.

Paper question:
    Can a trained graph reranker (GraphSAGE / R-GCN) outperform
    PPR and non-graph rerankers on evidence ranking?

Includes:
    - Model comparison: Hybrid vs PPR vs GraphSAGE vs R-GCN vs FEG-Rerank
    - Graph structure ablation: No Graph / Semantic / Financial / Full
    - Edge type ablation: w/o Company / w/o Year / w/o Metric / etc.
    - Hard negative ablation: Random / Top-Retrieved / Finance-Specific

Usage:
    # Sanity check (fast, verifies pipeline integrity)
    python experiments/exp4_gnn_reranker.py --config configs/default.yaml --device cuda --sanity

    # Full experiment
    python experiments/exp4_gnn_reranker.py --config configs/default.yaml --device cuda \\
        --epochs 10 --num_samples 500 --batch_size 16 --top_n 50

    # Skip ablations (faster)
    python experiments/exp4_gnn_reranker.py --no_ablation

    # Selective ablation
    python experiments/exp4_gnn_reranker.py --ablation edge,hard_negative
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
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from feg_rag.config import Config
from feg_rag.data.chunker import Chunk, chunk_text, chunk_report
from feg_rag.data.hard_negatives import (
    CorpusIndex,
    generate_hard_negatives_fast,
)
from feg_rag.data.loader import load_dataset
from feg_rag.evaluation.metrics import compute_all_metrics
from feg_rag.graph.builder import build_financial_evidence_graph
from feg_rag.graph.entities import EntityExtractor, extract_entities
from feg_rag.graph.features import build_node_features
from feg_rag.rerank.fusion import ConstraintScorer, FusionScorer
from feg_rag.rerank.ppr import ppr_rerank
from feg_rag.rerank.train import (
    build_corpus,
    build_train_pairs,
    print_loss_summary,
    save_training_artifacts,
    train_gnn_reranker,
    warmup_retrieval_scores,
)
from feg_rag.retrieval.bm25 import BM25Retriever
from feg_rag.retrieval.dense import DenseRetriever
from feg_rag.retrieval.hybrid import HybridRetriever


def _default_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


# =============================================================================
# Expected constants (for validation)
# =============================================================================

EXPECTED_DENSE_DIM = 384   # all-MiniLM-L6-v2
EXPECTED_FEATURE_DIM = 391  # 384 + 1 + 3 + 2 + 1


# =============================================================================
# Progress / status tracking
# =============================================================================

def write_run_status(output_dir: Path, **fields) -> None:
    """Write or update a lightweight run_status.json for progress monitoring."""
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "run_status.json"

    current: dict = {}
    if status_path.exists():
        try:
            current = json.loads(status_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    current.update(fields)
    current["last_updated"] = datetime.now().isoformat()

    status_path.write_text(
        json.dumps(current, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


# =============================================================================
# Helpers
# =============================================================================

def build_exp_corpus(
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
        for tf in (list(edgar_dir.rglob("*.txt")) or list(edgar_dir.rglob("*.html")))[:max_distractor_files]:
            try:
                corpus.extend(chunk_report(tf, cfg.chunk_size, cfg.chunk_overlap))
            except Exception:
                pass
    return corpus, gold_map


def build_result_dict(sample, retrieved_ids, gold_ids, method="unknown") -> Dict:
    return {
        "question_id": sample["id"],
        "question": sample["question"],
        "gold_answer": sample.get("answer", ""),
        "gold_evidence_ids": gold_ids,
        "retrieved_chunk_ids": retrieved_ids,
        "method": method,
    }


def compute_all_summaries(
    all_results: Dict[str, List[Dict]],
    k_values: List[int],
) -> Dict[str, Dict]:
    summaries = {}
    for method, results in all_results.items():
        er = compute_all_metrics(method, results, k_values=k_values)
        summaries[method] = {
            "method": method,
            "num_samples": er.num_samples,
            "evidence_recall": er.evidence_recall,
            "mrr": round(er.mrr, 4),
            "ndcg": {str(k): round(v, 4) for k, v in er.ndcg.items()},
        }
    return summaries


# =============================================================================
# Core: run evaluation of one reranker on test samples
# =============================================================================

def evaluate_reranker_on_samples(
    reranker,
    samples: List[Dict],
    hybrid: HybridRetriever,
    graph,
    features: Dict[str, np.ndarray],
    gold_map: Dict[str, List[str]],
    chunk_by_id: Dict[str, Chunk],
    extractor: EntityExtractor,
    top_n: int = 50,
    output_k: int = 10,
    ppr_alpha: float = 0.85,
) -> List[Dict]:
    """Run reranker on test samples and return per-query results."""
    results = []
    for s in samples:
        question = s["question"]
        gold_ids = gold_map.get(s["id"], [])

        # Hybrid retrieval
        hybrid_results = hybrid.search(question, top_k=top_n)
        candidate_ids = [c.chunk_id for c, _ in hybrid_results]

        # PPR scores for fusion
        q_metrics = extractor.extract_metrics(question)
        q_years = extractor.extract_years(question)
        ppr_scores = dict(ppr_rerank(
            graph, [], candidate_ids,
            seed_chunk_ids=candidate_ids[:10],
            seed_metric_names=list(q_metrics),
            seed_year_values=list(q_years),
            alpha=ppr_alpha,
        ))

        # Rerank
        try:
            reranked = reranker.rerank(
                question, hybrid_results, graph, features,
                ppr_scores=ppr_scores,
            )
            reranked_ids = [c.chunk_id for c, _ in reranked[:output_k]]
        except Exception:
            reranked_ids = candidate_ids[:output_k]

        results.append(build_result_dict(s, reranked_ids, gold_ids, "gnn"))
    return results


# =============================================================================
# Ablation: edge types
# =============================================================================

EDGE_ABLATION_CONFIGS = {
    "full_graph": {},
    "wo_company": {"add_company_nodes": False},
    "wo_filing": {"add_filing_nodes": False},
    "wo_section": {"add_section_nodes": False},
    "wo_metric": {},  # handled specially below
    "wo_year": {},    # handled specially below
    "wo_semantic": {"add_semantic_edges": False},
}


def run_edge_ablation(
    cfg: Config,
    train_samples: List[Dict],
    val_samples: List[Dict],
    corpus_chunks: List[Chunk],
    gold_map: Dict[str, List[str]],
    retriever,
    chunk_embeddings: Dict[str, np.ndarray],
    entity_map: Dict,
    output_dir: Path,
    device: str = "cpu",
) -> Dict[str, Dict]:
    """Train and evaluate with different edge type configurations."""
    print("\n  Running edge type ablation...")
    results: Dict[str, Dict] = {}

    for ab_name, ab_kwargs in EDGE_ABLATION_CONFIGS.items():
        t_start = time.time()
        print(f"    [{ab_name}]")
        # Build graph with ablated config
        g = build_financial_evidence_graph(
            corpus_chunks,
            entity_map=entity_map,
            add_semantic_edges=ab_kwargs.get("add_semantic_edges", True),
            add_company_nodes=ab_kwargs.get("add_company_nodes", True),
            add_filing_nodes=ab_kwargs.get("add_filing_nodes", True),
            add_section_nodes=ab_kwargs.get("add_section_nodes", True),
            add_same_entity_edges=True,
            max_same_entity_edges=30,
            use_edge_weights=True,
        )
        print(f"      Graph: {g.num_nodes} nodes, {g.num_edges} edges")

        # Features
        retrieval_scores = warmup_retrieval_scores(
            train_samples + val_samples, retriever
        )
        features = build_node_features(
            g, corpus_chunks, entity_map, retrieval_scores,
            chunk_embeddings=chunk_embeddings,
            compute_embeddings=False,
            embedding_device=device,
        )

        # Train
        t_train = time.time()
        reranker, history, meta = train_gnn_reranker(
            train_samples, retriever, g, features, gold_map, cfg,
            epochs=cfg.rerank.get("gnn_epochs", 10),
            device=device, min_pairs=1, verbose=False,
        )
        train_time = time.time() - t_train

        if reranker is None:
            print(f"      [SKIP] Not enough training pairs")
            continue

        # Evaluate
        t_eval = time.time()
        chunk_by_id = {c.chunk_id: c for c in corpus_chunks}
        extractor = EntityExtractor()
        val_results_list = evaluate_reranker_on_samples(
            reranker, val_samples, retriever, g, features,
            gold_map, chunk_by_id, extractor,
        )
        er = compute_all_metrics(
            ab_name, val_results_list,
            k_values=cfg.evaluation["recall_k_values"],
        )
        eval_time = time.time() - t_eval
        results[ab_name] = {
            "mrr": round(er.mrr, 4),
            "recall@5": round(er.evidence_recall.get(5, 0), 4),
            "recall@10": round(er.evidence_recall.get(10, 0), 4),
            "ndcg@10": round(er.ndcg.get(10, 0), 4),
        }
        elapsed = time.time() - t_start
        print(f"      MRR={results[ab_name]['mrr']}, R@10={results[ab_name]['recall@10']}")
        print(f"      train={train_time:.1f}s  eval={eval_time:.1f}s  total={elapsed:.1f}s")

    # Save
    csv_path = output_dir / "edge_type_ablation.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["setting", "mrr", "recall@5", "recall@10", "ndcg@10"])
        for ab_name in EDGE_ABLATION_CONFIGS:
            r = results.get(ab_name)
            if r:
                writer.writerow([ab_name, r["mrr"], r["recall@5"],
                                 r["recall@10"], r["ndcg@10"]])
    print(f"    Saved to {csv_path.name}")
    return results


# =============================================================================
# Ablation: hard negatives
# =============================================================================

def run_hard_negative_ablation(
    cfg: Config,
    train_samples: List[Dict],
    val_samples: List[Dict],
    corpus_chunks: List[Chunk],
    gold_map: Dict[str, List[str]],
    retriever,
    graph,
    chunk_embeddings: Dict[str, np.ndarray],
    entity_map: Dict,
    output_dir: Path,
    device: str = "cpu",
) -> Dict[str, Dict]:
    """Train with different negative sampling strategies."""
    print("\n  Running hard negative ablation...")

    results: Dict[str, Dict] = {}
    chunk_by_id = {c.chunk_id: c for c in corpus_chunks}

    # Pre-build corpus index for fast hard-negative generation
    print("    Building corpus index for fast hard-negative mining...")
    t_idx = time.time()
    corpus_index = CorpusIndex(corpus_chunks)
    print(f"    {corpus_index}  (built in {time.time() - t_idx:.1f}s)")

    # Strategy 1: Random negatives
    print("    [random_negatives]")
    t0 = time.time()
    random_pairs = _build_pairs_random(train_samples, retriever, gold_map,
                                       corpus_chunks, cfg)
    print(f"      Generated {len(random_pairs)} random-neg pairs")
    results.update(_train_and_eval_ablation(
        "random_negatives", random_pairs, train_samples, val_samples,
        retriever, graph, corpus_chunks, entity_map, gold_map, chunk_by_id,
        cfg, output_dir, device, chunk_embeddings,
    ))

    # Strategy 2: Top-retrieved negatives (already in build_train_pairs)
    print("    [top_retrieved_negatives]")
    t0 = time.time()
    top_pairs = build_train_pairs(
        train_samples, retriever, gold_map,
        top_k=cfg.retrieval.get("top_k", 50),
    )
    print(f"      Generated {len(top_pairs)} top-retrieved-neg pairs")
    results.update(_train_and_eval_ablation(
        "top_retrieved_negatives", top_pairs, train_samples, val_samples,
        retriever, graph, corpus_chunks, entity_map, gold_map, chunk_by_id,
        cfg, output_dir, device, chunk_embeddings,
    ))

    # Strategy 3: Finance-specific hard negatives (indexed, fast)
    print("    [finance_hard_negatives]")
    t0 = time.time()
    hard_pairs, hneg_stats = _build_finance_hard_pairs_fast(
        train_samples, gold_map, corpus_index,
    )
    for strat_name, count in hneg_stats.items():
        print(f"      {strat_name}: {count} pairs")
    print(f"      Total finance hard-neg pairs: {len(hard_pairs)}  "
          f"(built in {time.time() - t0:.1f}s)")

    if hard_pairs:
        results.update(_train_and_eval_ablation(
            "finance_hard_negatives", hard_pairs, train_samples, val_samples,
            retriever, graph, corpus_chunks, entity_map, gold_map, chunk_by_id,
            cfg, output_dir, device, chunk_embeddings,
        ))
    else:
        print("      [SKIP] No finance-specific hard negatives generated")

    # Save
    csv_path = output_dir / "hard_negative_ablation.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["setting", "mrr", "recall@5", "recall@10", "ndcg@10"])
        for name, r in results.items():
            writer.writerow([name, r["mrr"], r["recall@5"], r["recall@10"], r["ndcg@10"]])
    print(f"    Saved to {csv_path.name}")
    return results


def _build_pairs_random(
    samples, retriever, gold_map, corpus_chunks, cfg,
) -> List[Dict]:
    """Build pairs with random negatives."""
    pairs = []
    all_chunk_ids = [c.chunk_id for c in corpus_chunks]
    for s in samples:
        gold = set(gold_map.get(s["id"], []))
        if not gold:
            continue
        pos = next(iter(gold))
        # Random non-gold chunk
        negs = [cid for cid in all_chunk_ids if cid not in gold]
        if negs:
            pairs.append({"positive": pos, "negative": random.choice(negs)})
    return pairs


def _build_finance_hard_pairs_fast(
    samples: List[Dict],
    gold_map: Dict[str, List[str]],
    corpus_index: CorpusIndex,
) -> Tuple[List[Dict], Dict[str, int]]:
    """Build (positive, finance_hard_negative) training pairs using pre-built index.

    Uses :func:`generate_hard_negatives_fast` to avoid per-sample full-corpus scans.

    Returns:
        (pairs, stats) where stats is {strategy_name: pair_count}.
    """
    pairs: List[Dict] = []
    strategy_counts: Dict[str, int] = defaultdict(int)

    for s in samples:
        gold_ids = gold_map.get(s["id"], [])
        if not gold_ids:
            continue
        gold_chunks = [corpus_index.get_chunk(gid) for gid in gold_ids]
        gold_chunks = [c for c in gold_chunks if c is not None]
        if not gold_chunks:
            continue

        strategy_buckets = generate_hard_negatives_fast(
            gold_chunks, corpus_index, num_negatives=10,
        )

        # Pick one negative from the best available strategy bucket
        pos = gold_chunks[0].chunk_id
        for strat_name in ["same_metric_wrong_year", "same_year_wrong_metric",
                           "same_section", "random_fallback"]:
            bucket = strategy_buckets.get(strat_name, [])
            if bucket:
                pairs.append({"positive": pos, "negative": bucket[0].chunk_id})
                strategy_counts[strat_name] += 1
                break

    return pairs, dict(strategy_counts)


def _train_and_eval_ablation(
    name: str,
    train_pairs: List[Dict],
    train_samples: List[Dict],
    val_samples: List[Dict],
    retriever,
    graph,
    corpus_chunks: List[Chunk],
    entity_map: Dict,
    gold_map: Dict[str, List[str]],
    chunk_by_id: Dict[str, Chunk],
    cfg: Config,
    output_dir: Path,
    device: str,
    chunk_embeddings: Dict[str, np.ndarray],
) -> Dict[str, Dict]:
    """Train a model with given pairs and evaluate.

    Only calls ``_train_with_pairs`` once -- no duplicate training.
    """
    if len(train_pairs) < 5:
        print(f"      [SKIP] Only {len(train_pairs)} pairs for {name}")
        return {}

    print(f"      Training with {len(train_pairs)} pairs...")

    retrieval_scores = warmup_retrieval_scores(
        train_samples + val_samples, retriever
    )
    features = build_node_features(
        graph, corpus_chunks, entity_map, retrieval_scores,
        chunk_embeddings=chunk_embeddings,
        compute_embeddings=False,
        embedding_device=device,
    )

    # Train with custom pairs (no duplicate train_gnn_reranker call)
    t_train = time.time()
    reranker = _train_with_pairs(
        train_pairs, graph, features, cfg, device,
    )
    train_time = time.time() - t_train

    if reranker is None:
        print(f"      [SKIP] Training failed for {name}")
        return {}

    t_eval = time.time()
    extractor = EntityExtractor()
    val_results_list = evaluate_reranker_on_samples(
        reranker, val_samples, retriever, graph, features,
        gold_map, chunk_by_id, extractor,
    )
    er = compute_all_metrics(
        name, val_results_list,
        k_values=cfg.evaluation["recall_k_values"],
    )
    eval_time = time.time() - t_eval

    result = {
        "mrr": round(er.mrr, 4),
        "recall@5": round(er.evidence_recall.get(5, 0), 4),
        "recall@10": round(er.evidence_recall.get(10, 0), 4),
        "ndcg@10": round(er.ndcg.get(10, 0), 4),
    }
    print(f"      MRR={result['mrr']}, R@10={result['recall@10']}")
    print(f"      train={train_time:.1f}s  eval={eval_time:.1f}s")
    return {name: result}


def _train_with_pairs(
    train_pairs: List[Dict],
    graph,
    features: Dict[str, np.ndarray],
    cfg: Config,
    device: str = "cpu",
):
    """Train a GNN reranker with explicit positive/negative pairs."""
    from feg_rag.rerank.gnn import GNNFusionReranker, GraphSAGEReranker, RerankDataset
    from feg_rag.rerank.rgcn import RGCNFusionReranker, RGCNReranker, RGCNRerankDataset

    feat_dim = next(iter(features.values())).shape[0]
    model_type = cfg.rerank.get("gnn_model", "sage").lower()

    if model_type == "rgcn":
        dataset = RGCNRerankDataset(train_pairs, graph, features)
        model = RGCNReranker(
            in_dim=feat_dim,
            hidden_dim=cfg.rerank.get("gnn_hidden", 128),
            num_relations=dataset.num_relations,
            dropout=cfg.rerank.get("gnn_dropout", 0.3),
        )
        reranker = RGCNFusionReranker(
            model, relation_map=dataset.relation_map,
            alpha=cfg.rerank.get("fusion_alpha", 0.3),
            beta=cfg.rerank.get("fusion_beta", 0.3),
            gamma=cfg.rerank.get("fusion_gamma", 0.4),
            device=device,
        )
    else:
        dataset = RerankDataset(train_pairs, graph, features)
        model = GraphSAGEReranker(
            in_dim=feat_dim,
            hidden_dim=cfg.rerank.get("gnn_hidden", 128),
            dropout=cfg.rerank.get("gnn_dropout", 0.3),
        )
        reranker = GNNFusionReranker(
            model,
            alpha=cfg.rerank.get("fusion_alpha", 0.3),
            beta=cfg.rerank.get("fusion_beta", 0.3),
            gamma=cfg.rerank.get("fusion_gamma", 0.4),
            device=device,
        )

    history = reranker.fit(
        dataset,
        epochs=cfg.rerank.get("gnn_epochs", 10),
        lr=cfg.rerank.get("gnn_lr", 0.001),
        batch_size=32,
        verbose=False,
    )
    return reranker


# =============================================================================
# Output
# =============================================================================

def write_exp4_outputs(
    output_dir: Path,
    all_results: Dict[str, List[Dict]],
    summaries: Dict[str, Dict],
    k_values: List[int],
    ablation_results: Optional[Dict] = None,
) -> None:
    """Write all Exp4 outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Per-method JSONL
    for method, results in all_results.items():
        fname = f"{method.replace('+', '_')}_results.jsonl"
        with open(output_dir / fname, "w", encoding="utf-8") as fh:
            for r in results:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Metrics CSV
    csv_path = output_dir / "metrics_summary.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as fh:
        fieldnames = ["method", "mrr"] + \
                     [f"recall@{k}" for k in k_values] + \
                     [f"ndcg@{k}" for k in k_values]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for method, m in summaries.items():
            row = {"method": method, "mrr": m["mrr"]}
            for k in k_values:
                row[f"recall@{k}"] = round(m["evidence_recall"].get(k, 0), 4)
                row[f"ndcg@{k}"] = round(float(m["ndcg"].get(str(k), 0)), 4)
            writer.writerow(row)


def write_exp4_readme(
    output_dir: Path,
    summaries: Dict[str, Dict],
    k_values: List[int],
    model_type: str,
) -> None:
    """Generate Exp4 README."""
    hybrid = summaries.get("hybrid", {})
    gnn_best = max(
        [(m, s) for m, s in summaries.items()
         if m not in ("hybrid", "hybrid+ppr")],
        key=lambda x: x[1].get("mrr", 0),
        default=("N/A", {}),
    )

    lines = [
        "# Experiment 4: GNN Evidence Reranker",
        "",
        f"Trained {model_type.upper()} reranker on Financial Evidence Graph.",
        "",
        "## Run command",
        "",
        "```bash",
        "python experiments/exp4_gnn_reranker.py \\",
        f"  --config configs/default.yaml \\",
        f"  --output_dir outputs/exp4_gnn_reranker \\",
        f"  --epochs 10",
        "```",
        "",
        "## Model comparison",
        "",
        "| Method | MRR |" + "|".join(f" R@{k} " for k in k_values) + "|",
        "|---|---|" + "|".join("---" for _ in k_values) + "|",
    ]
    method_order = [
        "hybrid", "hybrid+cross_encoder", "hybrid+ppr",
        "hybrid+sage", "hybrid+rgcn", "feg_rerank",
    ]
    for method in method_order:
        m = summaries.get(method)
        if m is None:
            continue
        label = {
            "hybrid": "Hybrid Retrieval",
            "hybrid+cross_encoder": "Hybrid + Cross-Encoder",
            "hybrid+ppr": "Hybrid + PPR",
            "hybrid+sage": "Hybrid + GraphSAGE",
            "hybrid+rgcn": "Hybrid + R-GCN",
            "feg_rerank": "FEG-Rerank",
        }.get(method, method)
        row = f"| {label} | {m['mrr']}"
        for k in k_values:
            row += f" | {m['evidence_recall'].get(k, 0):.4f}"
        row += " |"
        lines.append(row)

    lines.extend([
        "",
        "## Output files",
        "",
        "| File | Description |",
        "|---|---|",
        "| `model_checkpoints/exp4_*.pt` | Trained model weights |",
        "| `exp4_loss_history_*.json` | Training loss history |",
        "| `metrics_summary.csv` | Aggregate metrics |",
        "| `*_results.jsonl` | Per-query results per method |",
        "| `edge_type_ablation.csv` | Edge type ablation results |",
        "| `hard_negative_ablation.csv` | Hard negative ablation results |",
        "| `train_config.yaml` | Training configuration |",
        "",
        "## Answers to Exp4 questions",
        "",
        f"1. **GNN > Hybrid?** " +
        ("Yes" if gnn_best[1].get("mrr", 0) > hybrid.get("mrr", 0) else "Check results"),
        f"2. **R-GCN > GraphSAGE?** Compare rgcn vs graphsage rows above.",
        "3. **GNN > PPR?** Compare gnn rows vs hybrid+ppr row --",
        "   GNN can learn edge importance, PPR uses fixed weights.",
        "4. **Hard negatives effective?** See hard_negative_ablation.csv.",
        "5. **Removing year/metric edges hurts?** See edge_type_ablation.csv.",
        "",
        f"Generated: {datetime.now().isoformat()}",
    ])

    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


# =============================================================================
# Output directory validation
# =============================================================================

# Sentinel files that indicate a completed Exp4 run
_EXP4_SENTINELS = [
    "metrics_summary.csv",
    "hybrid_results.jsonl",
    "train_config.yaml",
]


def _has_existing_results(out_dir: Path) -> bool:
    """Check whether *out_dir* already contains Exp4 outputs."""
    if not out_dir.exists():
        return False
    return any((out_dir / s).exists() for s in _EXP4_SENTINELS)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Exp4: GNN Evidence Reranker")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output_dir", default=None,
                        help="Output directory (default: auto, based on --sanity)")
    parser.add_argument("--num_samples", type=int, default=0,
                        help="0 = all FinDER")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override config gnn_epochs")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--model", choices=["sage", "rgcn"], default=None,
                        help="Override config gnn_model")
    parser.add_argument("--device", default=_default_device(),
                        help="Device for GNN training/eval (cuda recommended)")
    parser.add_argument("--dense_device", default="cpu",
                        help="Device for Dense encoding (cpu avoids 4GB GPU OOM)")
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--top_n", type=int, default=50)
    parser.add_argument("--no_ablation", action="store_true",
                        help="Skip all ablation studies")
    parser.add_argument("--ablation", type=str, default="edge,hard_negative",
                        help="Comma-separated ablation types to run "
                             "(edge,hard_negative), or 'none'")
    parser.add_argument("--no_embeddings", action="store_true",
                        help="Skip text embeddings (use zeros)")
    parser.add_argument("--sanity", action="store_true",
                        help="Sanity mode: 30 samples, 1 epoch, no ablation")
    parser.add_argument("--overwrite_output_dir", action="store_true",
                        help="Allow overwriting existing results")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    cfg.ensure_dirs()

    if args.model:
        cfg.rerank["gnn_model"] = args.model

    # ---- Sanity mode overrides ----
    if args.sanity:
        args.num_samples = args.num_samples or 30
        args.epochs = args.epochs or 1
        args.top_n = 20
        args.no_ablation = True
        if args.output_dir is None:
            args.output_dir = "outputs/exp4_sanity_neural"
        print("=" * 60)
        print("  SANITY MODE")
        print(f"  samples={args.num_samples}  epochs={args.epochs}  "
              f"top_n={args.top_n}  ablation=OFF")
        print("=" * 60)
    else:
        if args.output_dir is None:
            args.output_dir = "outputs/exp4_gnn_reranker"

    if args.epochs:
        cfg.rerank["gnn_epochs"] = args.epochs

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = cfg.root_dir / output_dir

    # ---- Overwrite guard ----
    if _has_existing_results(output_dir) and not args.overwrite_output_dir:
        print(f"\nERROR: Output directory '{output_dir}' already contains results.")
        print(f"  Use --overwrite_output_dir to replace existing outputs, or")
        print(f"  specify a different --output_dir.")
        sys.exit(1)

    # ---- Ablation parsing ----
    if args.no_ablation:
        run_edge_abl = False
        run_hneg_abl = False
    else:
        abl_parts = [x.strip() for x in args.ablation.split(",") if x.strip()]
        run_edge_abl = "edge" in abl_parts
        run_hneg_abl = "hard_negative" in abl_parts
        if "none" in abl_parts:
            run_edge_abl = False
            run_hneg_abl = False

    k_values = cfg.evaluation.get("recall_k_values", [1, 3, 5, 10, 20])
    model_type = cfg.rerank.get("gnn_model", "sage")
    device = args.device

    print("=" * 60)
    print(f"  EXP4: GNN Evidence Reranker ({model_type.upper()})")
    print("=" * 60)
    print(f"  Output:    {output_dir}")
    print(f"  Model:     {model_type}")
    print(f"  Device:    {device} (GNN)")
    print(f"  Dense:     {args.dense_device} (encoding, avoids GPU OOM)")
    print(f"  Epochs:    {cfg.rerank.get('gnn_epochs', 'default')}")
    print(f"  Val split: {args.val_split}")
    print(f"  Ablation:  edge={run_edge_abl}  hard_negative={run_hneg_abl}")

    write_run_status(output_dir, current_stage="starting", output_path=str(output_dir))

    # ---- 1. Data ----
    print("\n[1/6] Loading data...")
    samples = load_dataset("finder", cfg.data_dir)
    total_available = len(samples)
    if args.num_samples > 0:
        samples = samples[:args.num_samples]
    random.shuffle(samples)
    n_val = max(1, int(len(samples) * args.val_split))
    train_samples = samples[n_val:]
    val_samples = samples[:n_val]
    print(f"  Train: {len(train_samples)}  |  Val: {len(val_samples)}  |  "
          f"Total: {len(samples)}  |  Available: {total_available}")

    write_run_status(output_dir,
                     current_stage="data_loaded",
                     train_samples=len(train_samples),
                     val_samples=len(val_samples),
                     total_samples=len(samples),
                     total_available=total_available)

    # ---- 2. Corpus ----
    print("[2/6] Building corpus...")
    corpus_chunks, gold_map = build_exp_corpus(samples, cfg)
    chunk_by_id = {c.chunk_id: c for c in corpus_chunks}

    # Count gold-only chunks
    gold_chunk_ids: Set[str] = set()
    for gids in gold_map.values():
        gold_chunk_ids.update(gids)
    print(f"  {len(corpus_chunks)} chunks ({len(gold_chunk_ids)} gold evidence, "
          f"{len(corpus_chunks) - len(gold_chunk_ids)} distractors)")
    print(f"  {len(gold_map)} queries with gold evidence")

    # Sanity check: corpus must have distractors
    if args.sanity and len(corpus_chunks) <= len(gold_chunk_ids) + 5:
        print("  [WARN] SANITY WARNING: Corpus has very few distractors. "
              "10-K data may be missing!")
    if args.sanity:
        print(f"  [OK] Corpus includes 10-K distractors: "
              f"{len(corpus_chunks)} total > {len(gold_chunk_ids)} gold-only")

    write_run_status(output_dir,
                     current_stage="corpus_built",
                     corpus_chunks=len(corpus_chunks),
                     gold_evidence_chunks=len(gold_chunk_ids),
                     distractor_chunks=len(corpus_chunks) - len(gold_chunk_ids))

    # ---- 3. Retrieval indices ----
    print("[3/6] Building retrieval indices...")
    bm25 = BM25Retriever(
        k1=cfg.retrieval.get("bm25_k1", 1.5),
        b=cfg.retrieval.get("bm25_b", 0.75),
    )
    bm25.index(corpus_chunks)
    dense = DenseRetriever(
        model_name=cfg.retrieval.get("dense_model", "all-MiniLM-L6-v2"),
        device=args.dense_device,
    )
    dense.index(corpus_chunks)
    chunk_embeddings = dense.chunk_embeddings()
    embedding_dim = next(iter(chunk_embeddings.values())).shape[0] if chunk_embeddings else 0

    print(f"  Dense backend:       {dense.backend}")
    print(f"  Dense embedding dim: {embedding_dim}")

    # Sanity / validation check
    if embedding_dim == 0:
        print("  [ERROR] Dense embeddings are empty! Cannot proceed.")
        sys.exit(1)
    if embedding_dim != EXPECTED_DENSE_DIM:
        print(f"  [WARN] WARNING: Dense embedding dim is {embedding_dim}, "
              f"expected {EXPECTED_DENSE_DIM} for all-MiniLM-L6-v2.")

    hybrid = HybridRetriever(bm25, dense, alpha=cfg.retrieval.get("hybrid_alpha", 0.5))

    write_run_status(output_dir,
                     current_stage="retrieval_indices_built",
                     dense_backend=dense.backend,
                     dense_embedding_dim=embedding_dim)

    # ---- 4. Graph + features ----
    print("[4/6] Building graph and features...")
    entity_map = extract_entities(corpus_chunks)
    graph = build_financial_evidence_graph(
        corpus_chunks,
        entity_map=entity_map,
        add_semantic_edges=True,
        add_company_nodes=True,
        add_filing_nodes=True,
        add_section_nodes=True,
        add_same_entity_edges=True,
        max_same_entity_edges=30,
        use_edge_weights=True,
    )
    print(f"  Graph: {graph.num_nodes} nodes, {graph.num_edges} edges")

    retrieval_scores = warmup_retrieval_scores(
        train_samples + val_samples, hybrid, top_k=args.top_n,
    )
    features = build_node_features(
        graph, corpus_chunks, entity_map, retrieval_scores,
        chunk_embeddings=None if args.no_embeddings else chunk_embeddings,
        compute_embeddings=False,
        embedding_device=device,
    )
    feature_dim = next(iter(features.values())).shape[0]
    print(f"  Feature dim: {feature_dim}")

    # Feature dim validation
    if chunk_embeddings and not args.no_embeddings:
        if feature_dim != EXPECTED_FEATURE_DIM:
            print(f"  [WARN] WARNING: Feature dim is {feature_dim}, expected "
                  f"{EXPECTED_FEATURE_DIM} for neural embeddings "
                  f"(384 + 1 + 3 + 2 + 1). Check node type encoding!")
    if args.sanity:
        if feature_dim == EXPECTED_FEATURE_DIM:
            print(f"  [OK] Feature dim == {EXPECTED_FEATURE_DIM} (correct)")
        else:
            print(f"  [MISMATCH] Feature dim {feature_dim} != {EXPECTED_FEATURE_DIM} "
                  f"(MISMATCH -- pipeline issue!)")

    write_run_status(output_dir,
                     current_stage="graph_built",
                     graph_nodes=graph.num_nodes,
                     graph_edges=graph.num_edges,
                     feature_dim=feature_dim,
                     dense_backend=dense.backend,
                     dense_embedding_dim=embedding_dim)

    # ---- 5. Train GNN ----
    print("[5/6] Training GNN reranker...")
    t0 = time.time()
    reranker, history, meta = train_gnn_reranker(
        train_samples, hybrid, graph, features, gold_map, cfg,
        epochs=cfg.rerank.get("gnn_epochs", 10),
        batch_size=args.batch_size,
        device=device,
        min_pairs=5,
        verbose=True,
    )

    if reranker is None:
        print("\nERROR: Training failed. Not enough training pairs. "
              "Try more samples or lower min_pairs.")
        write_run_status(output_dir, current_stage="training_failed",
                         error="Not enough training pairs")
        sys.exit(1)

    train_time = time.time() - t0
    print_loss_summary(history)
    print(f"  Training time: {train_time:.1f}s")

    # Save checkpoint
    paths = save_training_artifacts(
        reranker, history, output_dir, meta, experiment="exp4",
    )
    print(f"Checkpoint: {paths['checkpoint']}")

    write_run_status(output_dir,
                     current_stage="training_done",
                     train_time_seconds=round(train_time, 1),
                     final_loss=history[-1] if history else None,
                     train_pairs=meta.get("num_pairs", 0))

    # ---- 6. Evaluate all methods ----
    print("\n[6/6] Evaluating all methods on validation set...")
    extractor = EntityExtractor()
    all_results: Dict[str, List[Dict]] = {}

    # Hybrid baseline
    print("  Hybrid...")
    all_results["hybrid"] = []
    for s in val_samples:
        results = hybrid.search(s["question"], top_k=args.top_n)
        ids = [c.chunk_id for c, _ in results[:10]]
        all_results["hybrid"].append(
            build_result_dict(s, ids, gold_map.get(s["id"], []), "hybrid")
        )

    # PPR baseline
    print("  PPR...")
    all_results["hybrid+ppr"] = []
    for s in val_samples:
        hybrid_results = hybrid.search(s["question"], top_k=args.top_n)
        candidate_ids = [c.chunk_id for c, _ in hybrid_results]
        q_metrics = extractor.extract_metrics(s["question"])
        q_years = extractor.extract_years(s["question"])
        ppr_scores = ppr_rerank(
            graph, corpus_chunks, candidate_ids,
            seed_chunk_ids=candidate_ids[:10],
            seed_metric_names=list(q_metrics),
            seed_year_values=list(q_years),
            alpha=cfg.rerank.get("ppr_alpha", 0.85),
        )
        ppr_ids = [cid for cid, _ in ppr_scores[:10]]
        all_results["hybrid+ppr"].append(
            build_result_dict(s, ppr_ids, gold_map.get(s["id"], []), "hybrid+ppr")
        )

    # GNN reranker
    print(f"  GNN ({model_type})...")
    method_name = f"hybrid+{model_type}"
    all_results[method_name] = evaluate_reranker_on_samples(
        reranker, val_samples, hybrid, graph, features,
        gold_map, chunk_by_id, extractor,
        top_n=args.top_n,
    )
    for r in all_results[method_name]:
        r["method"] = method_name

    # Compute summaries
    summaries = compute_all_summaries(all_results, k_values)

    # Print summary
    print("\n" + "=" * 70)
    print("  EXP4 RESULTS: GNN RERANKER")
    print("=" * 70)
    header = f"{'Method':<30} {'MRR':>7}"
    for k in k_values:
        header += f" {'R@'+str(k):>8}"
    print(header)
    print("-" * 70)
    for method, m in summaries.items():
        label = {
            "hybrid": "Hybrid",
            "hybrid+ppr": "Hybrid + PPR",
            "hybrid+sage": "Hybrid + GraphSAGE",
            "hybrid+rgcn": "Hybrid + R-GCN",
            "feg_rerank": "FEG-Rerank",
        }.get(method, method)
        row = f"{label:<30} {m['mrr']:>7.4f}"
        for k in k_values:
            row += f" {m['evidence_recall'].get(k, 0):>8.4f}"
        print(row)

    write_run_status(output_dir,
                     current_stage="evaluation_done",
                     methods_evaluated=list(all_results.keys()))

    # ---- Ablation studies ----
    ablation_results = None
    if run_edge_abl or run_hneg_abl:
        print("\n" + "=" * 60)
        print("  ABLATION STUDIES")
        print("=" * 60)

    if run_edge_abl:
        edge_results = run_edge_ablation(
            cfg, train_samples, val_samples, corpus_chunks,
            gold_map, hybrid, chunk_embeddings, entity_map, output_dir, device,
        )
        ablation_results = ablation_results or {}
        ablation_results["edge_ablation"] = edge_results
        write_run_status(output_dir,
                         current_stage="edge_ablation_done",
                         edge_ablations=list(edge_results.keys()))

    if run_hneg_abl:
        hn_results = run_hard_negative_ablation(
            cfg, train_samples, val_samples, corpus_chunks,
            gold_map, hybrid, graph, chunk_embeddings, entity_map, output_dir, device,
        )
        ablation_results = ablation_results or {}
        ablation_results["hard_negative_ablation"] = hn_results
        write_run_status(output_dir,
                         current_stage="hard_negative_ablation_done",
                         hneg_ablations=list(hn_results.keys()))

    # ---- Write outputs ----
    total_elapsed = time.time() - t0
    print(f"\nTotal time: {total_elapsed:.1f}s")
    write_exp4_outputs(output_dir, all_results, summaries, k_values)
    write_exp4_readme(output_dir, summaries, k_values, model_type)

    # Save train config
    config_path = output_dir / "train_config.yaml"
    import yaml
    with open(config_path, "w", encoding="utf-8") as fh:
        yaml.dump(dict(cfg.to_dict(), **{
            "exp4_meta": {
                "model": model_type,
                "train_samples": len(train_samples),
                "val_samples": len(val_samples),
                "corpus_chunks": len(corpus_chunks),
                "gold_evidence_chunks": len(gold_chunk_ids),
                "graph_nodes": graph.num_nodes,
                "graph_edges": graph.num_edges,
                "feature_dim": feature_dim,
                "dense_backend": dense.backend,
                "dense_embedding_dim": embedding_dim,
                "final_loss": history[-1] if history else None,
                "elapsed_seconds": total_elapsed,
            }
        }), fh, default_flow_style=False)

    write_run_status(output_dir,
                     current_stage="done",
                     total_elapsed_seconds=round(total_elapsed, 1))

    print(f"\nOutput: {output_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
