"""Experiment 3: Financial Evidence Graph + Personalized PageRank reranking.

Builds the Financial Evidence Graph (Company–Filing–Section–Chunk–Metric–Year)
and uses PPR to rerank Hybrid Retrieval candidates WITHOUT training any model.

Paper question:
    Can a financial evidence graph improve evidence ranking using only
    graph algorithms (no GNN training)?

Compared variants:
    - Hybrid Retrieval (no graph — baseline from Exp1)
    - Hybrid + Semantic Graph + PPR  (only semantic-similar edges)
    - Hybrid + Financial Graph + PPR (company/filing/year/metric structural edges)
    - Hybrid + Full Graph + PPR      (financial + semantic edges)
    - Hybrid + Full Graph + PPR + Constraint (with constraint-aware fusion)

Usage:
    python experiments/exp3_feg_ppr.py
    python experiments/exp3_feg_ppr.py --num_samples 100 --top_n 50
    python experiments/exp3_feg_ppr.py --no_semantic_edges
"""

from __future__ import annotations

import argparse
import csv
import json
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
from feg_rag.rerank.fusion import ConstraintScorer, FusionScorer
from feg_rag.rerank.ppr import ppr_rerank
from feg_rag.retrieval.bm25 import BM25Retriever
from feg_rag.retrieval.dense import DenseRetriever
from feg_rag.retrieval.hybrid import HybridRetriever


# ═════════════════════════════════════════════════════════════════════════════
# Corpus building
# ═════════════════════════════════════════════════════════════════════════════

def build_corpus(
    samples: List[Dict],
    cfg: Config,
    max_distractor_files: int = 50,
) -> Tuple[List[Chunk], Dict[str, List[str]]]:
    """Chunk FinDER evidence + optional 10-K distractors."""
    corpus: List[Chunk] = []
    gold_map: Dict[str, List[str]] = {}

    for s in samples:
        gold_ids: List[str] = []
        for text in s["evidence_texts"]:
            for c in chunk_text(text, cfg.chunk_size, cfg.chunk_overlap, doc_id=s["id"]):
                corpus.append(c)
                gold_ids.append(c.chunk_id)
        gold_map[s["id"]] = gold_ids

    # Add 10-K distractors
    edgar_dir = cfg.edgar_dir
    if edgar_dir.exists():
        txt_files = list(edgar_dir.rglob("*.txt")) or list(edgar_dir.rglob("*.html"))
        for tf in txt_files[:max_distractor_files]:
            try:
                corpus.extend(chunk_report(tf, cfg.chunk_size, cfg.chunk_overlap))
            except Exception:
                pass

    return corpus, gold_map


# ═════════════════════════════════════════════════════════════════════════════
# Result building
# ═════════════════════════════════════════════════════════════════════════════

def build_result_record(
    sample: Dict,
    method: str,
    reranked_ids: List[str],
    gold_ids: List[str],
) -> Dict:
    """Build a per-query result dict for metrics computation."""
    return {
        "question_id": sample["id"],
        "question": sample["question"],
        "gold_answer": sample.get("answer", ""),
        "gold_evidence_ids": gold_ids,
        "retrieved_chunk_ids": reranked_ids,
        "method": method,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Main pipeline
# ═════════════════════════════════════════════════════════════════════════════

def run_exp3(
    cfg: Config,
    samples: List[Dict],
    corpus_chunks: List[Chunk],
    gold_map: Dict[str, List[str]],
    top_n: int = 50,
    output_k: int = 10,
    add_semantic_edges: bool = True,
    ppr_alpha: float = 0.85,
    dense_device: str = "cpu",
    verbose: bool = True,
) -> Dict[str, List[Dict]]:
    """Run all Exp3 variants and return per-method results."""

    chunk_by_id = {c.chunk_id: c for c in corpus_chunks}
    extractor = EntityExtractor()

    # ---- Retrieval indices ----
    if verbose:
        print("  Building retrieval indices...")
    bm25 = BM25Retriever(
        k1=cfg.retrieval.get("bm25_k1", 1.5),
        b=cfg.retrieval.get("bm25_b", 0.75),
    )
    bm25.index(corpus_chunks)
    dense = DenseRetriever(
        model_name=cfg.retrieval.get("dense_model", "all-MiniLM-L6-v2"),
        device=dense_device,
    )
    dense.index(corpus_chunks)
    hybrid = HybridRetriever(bm25, dense, alpha=cfg.retrieval.get("hybrid_alpha", 0.5))

    # ---- Build entity map ----
    if verbose:
        print("  Extracting entities...")
    entity_map = extract_entities(corpus_chunks)

    # ---- Build 3 graph variants ----
    # Graph 1: Semantic only
    if verbose:
        print("  Building Semantic Graph...")
    g_semantic = build_financial_evidence_graph(
        corpus_chunks,
        entity_map=entity_map,
        add_semantic_edges=True,
        add_company_nodes=False,
        add_filing_nodes=False,
        add_section_nodes=False,
        add_same_entity_edges=False,
        use_edge_weights=True,
    )
    if verbose:
        print(f"    Semantic Graph: {g_semantic.num_nodes} nodes, "
              f"{g_semantic.num_edges} edges")

    # Graph 2: Financial structure only
    if verbose:
        print("  Building Financial Structure Graph...")
    g_financial = build_financial_evidence_graph(
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
    if verbose:
        print(f"    Financial Graph: {g_financial.num_nodes} nodes, "
              f"{g_financial.num_edges} edges")

    # Graph 3: Full graph (financial + semantic)
    if verbose and add_semantic_edges:
        print("  Building Full Graph (financial + semantic)...")
    g_full = build_financial_evidence_graph(
        corpus_chunks,
        entity_map=entity_map,
        add_semantic_edges=add_semantic_edges,
        add_company_nodes=True,
        add_filing_nodes=True,
        add_section_nodes=True,
        add_same_entity_edges=True,
        max_same_entity_edges=30,
        use_edge_weights=True,
    )
    if verbose:
        print(f"    Full Graph: {g_full.num_nodes} nodes, "
              f"{g_full.num_edges} edges")

    # ---- Fusion scorer ----
    constraint_scorer = ConstraintScorer(
        company_weight=cfg.constraint.get("company_match_weight", 1.0),
        year_weight=cfg.constraint.get("year_match_weight", 1.0),
        metric_weight=cfg.constraint.get("metric_match_weight", 0.8),
        filing_type_weight=cfg.constraint.get("filing_type_match_weight", 0.5),
    )
    fusion = FusionScorer(
        alpha=cfg.rerank.get("fusion_alpha", 0.3),
        beta=cfg.rerank.get("fusion_beta", 0.4),
        gamma=0.0,
        delta=cfg.rerank.get("fusion_delta", 0.1),
        constraint_scorer=constraint_scorer,
    )

    # ---- Results containers ----
    all_results: Dict[str, List[Dict]] = {
        "hybrid": [],
        "hybrid+semantic_ppr": [],
        "hybrid+financial_ppr": [],
        "hybrid+full_ppr": [],
        "hybrid+full_ppr+constraint": [],
    }

    # ---- Process queries ----
    if verbose:
        print(f"\n  Running PPR reranking on {len(samples)} queries...")

    for i, s in enumerate(samples):
        if verbose and i % 20 == 0:
            print(f"    {i}/{len(samples)}")
        qid = s["id"]
        question = s["question"]
        gold_ids = gold_map.get(qid, [])

        # Hybrid retrieval (baseline)
        hybrid_results = hybrid.search(question, top_k=top_n)
        candidate_ids = [c.chunk_id for c, _ in hybrid_results]
        retrieval_scores = {c.chunk_id: float(score) for c, score in hybrid_results}

        # Build query entities for PPR seeds
        q_metrics = extractor.extract_metrics(question)
        q_years = extractor.extract_years(question)
        candidate_chunks = [chunk_by_id[cid] for cid in candidate_ids if cid in chunk_by_id]

        # ---- Hybrid only ----
        all_results["hybrid"].append(
            build_result_record(s, "hybrid", candidate_ids[:output_k], gold_ids)
        )

        # ---- PPR with Semantic Graph ----
        ppr_sem = ppr_rerank(
            g_semantic, corpus_chunks, candidate_ids,
            seed_chunk_ids=candidate_ids[:10],
            seed_metric_names=list(q_metrics),
            seed_year_values=list(q_years),
            alpha=ppr_alpha,
            retrieval_scores=retrieval_scores,
            retrieval_weight=cfg.rerank.get("ppr_retrieval_weight", 0.5),
        )
        ppr_sem_ids = [cid for cid, _ in ppr_sem[:output_k]]
        all_results["hybrid+semantic_ppr"].append(
            build_result_record(s, "hybrid+semantic_ppr", ppr_sem_ids, gold_ids)
        )

        # ---- PPR with Financial Graph ----
        ppr_fin = ppr_rerank(
            g_financial, corpus_chunks, candidate_ids,
            seed_chunk_ids=candidate_ids[:10],
            seed_metric_names=list(q_metrics),
            seed_year_values=list(q_years),
            alpha=ppr_alpha,
            retrieval_scores=retrieval_scores,
            retrieval_weight=cfg.rerank.get("ppr_retrieval_weight", 0.5),
        )
        ppr_fin_ids = [cid for cid, _ in ppr_fin[:output_k]]
        all_results["hybrid+financial_ppr"].append(
            build_result_record(s, "hybrid+financial_ppr", ppr_fin_ids, gold_ids)
        )

        # ---- PPR with Full Graph ----
        ppr_full = ppr_rerank(
            g_full, corpus_chunks, candidate_ids,
            seed_chunk_ids=candidate_ids[:10],
            seed_metric_names=list(q_metrics),
            seed_year_values=list(q_years),
            alpha=ppr_alpha,
            retrieval_scores=retrieval_scores,
            retrieval_weight=cfg.rerank.get("ppr_retrieval_weight", 0.5),
        )
        ppr_full_dict = dict(ppr_full)
        ppr_full_ids = [cid for cid, _ in ppr_full[:output_k]]
        all_results["hybrid+full_ppr"].append(
            build_result_record(s, "hybrid+full_ppr", ppr_full_ids, gold_ids)
        )

        # ---- PPR + Constraint Fusion ----
        # Build retrieval score dict
        ret_scores: Dict[str, float] = {}
        for rank, (chunk, score) in enumerate(hybrid_results):
            ret_scores[chunk.chunk_id] = 1.0 - rank / max(len(hybrid_results), 1)

        fused = fusion.fuse(question, candidate_chunks, ret_scores,
                            graph_scores=ppr_full_dict)
        fused_ids = [c.chunk_id for c, _ in fused[:output_k]]
        all_results["hybrid+full_ppr+constraint"].append(
            build_result_record(s, "hybrid+full_ppr+constraint", fused_ids, gold_ids)
        )

    return all_results


# ═════════════════════════════════════════════════════════════════════════════
# Evaluation
# ═════════════════════════════════════════════════════════════════════════════

def evaluate_all(
    all_results: Dict[str, List[Dict]],
    k_values: List[int],
    graph_stats: Dict,
) -> Dict[str, Dict]:
    """Compute metrics for all methods, return summary dict."""
    summaries: Dict[str, Dict] = {}
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


def write_outputs(
    output_dir: Path,
    all_results: Dict[str, List[Dict]],
    summaries: Dict[str, Dict],
    graph_stats: Dict,
    k_values: List[int],
) -> None:
    """Write all Exp3 output files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Per-method JSONL
    method_files = {
        "hybrid": "hybrid_results.jsonl",
        "hybrid+semantic_ppr": "ppr_results_semantic_graph.jsonl",
        "hybrid+financial_ppr": "ppr_results_financial_graph.jsonl",
        "hybrid+full_ppr": "ppr_results_full_graph.jsonl",
        "hybrid+full_ppr+constraint": "ppr_results_full_graph_constraint.jsonl",
    }
    for method, results in all_results.items():
        fname = method_files.get(method, f"{method.replace('+', '_')}_results.jsonl")
        fpath = output_dir / fname
        with open(fpath, "w", encoding="utf-8") as fh:
            for r in results:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Graph stats
    with open(output_dir / "graph_stats.json", "w", encoding="utf-8") as fh:
        json.dump(graph_stats, fh, indent=2)

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

    # Error reduction CSV (comparison with Hybrid baseline)
    hybrid_summary = summaries.get("hybrid", {})
    hybrid_recall = hybrid_summary.get("evidence_recall", {})

    error_csv = output_dir / "error_reduction_summary.csv"
    with open(error_csv, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["method"] +
                         [f"recall@{k}" for k in k_values] +
                         [f"delta_recall@{k}" for k in k_values] +
                         ["delta_mrr"])
        for method, m in summaries.items():
            if method == "hybrid":
                continue
            row = [method]
            deltas_recall = []
            for k in k_values:
                r = m["evidence_recall"].get(k, 0)
                row.append(round(r, 4))
                deltas_recall.append(round(r - hybrid_recall.get(k, 0), 4))
            row.extend(deltas_recall)
            row.append(round(m["mrr"] - hybrid_summary.get("mrr", 0), 4))
            writer.writerow(row)

    # Case studies
    write_case_studies(output_dir, all_results)

    # README
    write_readme(output_dir, summaries, k_values)


def write_case_studies(output_dir: Path, all_results: Dict[str, List[Dict]]) -> None:
    """Generate case study Markdown showing PPR corrections."""
    lines = [
        "# Exp3 Case Studies",
        "",
        "Examples where Financial Evidence Graph + PPR corrects retrieval errors.",
        "",
    ]

    # Find cases where hybrid misses but full PPR hits
    hybrid_results = {r["question_id"]: r for r in all_results.get("hybrid", [])}
    full_ppr_results = {r["question_id"]: r for r in all_results.get("hybrid+full_ppr", [])}

    improved_cases = []
    for qid, hr in hybrid_results.items():
        fr = full_ppr_results.get(qid)
        if fr is None:
            continue
        gold = set(hr.get("gold_evidence_ids", []))
        hybrid_hit = bool(gold & set(hr.get("retrieved_chunk_ids", [])[:10]))
        ppr_hit = bool(gold & set(fr.get("retrieved_chunk_ids", [])[:10]))
        if not hybrid_hit and ppr_hit:
            improved_cases.append((qid, hr, fr))

    for i, (qid, hr, fr) in enumerate(improved_cases[:3], 1):
        lines.append(f"## Case {i}: PPR corrects retrieval")
        lines.append(f"")
        lines.append(f"**Query**: {hr['question']}")
        lines.append(f"")
        lines.append(f"**Hybrid top-5**: {hr['retrieved_chunk_ids'][:5]}")
        lines.append(f"**PPR top-5**: {fr['retrieved_chunk_ids'][:5]}")
        lines.append(f"**Gold evidence**: {hr['gold_evidence_ids'][:3]}")
        lines.append(f"")
        lines.append("---")
        lines.append("")

    if not improved_cases:
        lines.append("(No clear improvement cases found at top-10 — check larger top-N)")
        lines.append("")

    (output_dir / "case_studies.md").write_text("\n".join(lines), encoding="utf-8")


def write_readme(
    output_dir: Path,
    summaries: Dict[str, Dict],
    k_values: List[int],
) -> None:
    """Generate Exp3 README."""
    hybrid = summaries.get("hybrid", {})
    best = max(
        [(m, s) for m, s in summaries.items() if m != "hybrid"],
        key=lambda x: x[1].get("evidence_recall", {}).get(10, 0),
        default=("N/A", {}),
    )

    lines = [
        "# Experiment 3: Financial Evidence Graph + PPR",
        "",
        "Graph-based reranking WITHOUT training any model.",
        "",
        "## Run command",
        "",
        "```bash",
        "python experiments/exp3_feg_ppr.py \\",
        "  --config configs/default.yaml \\",
        "  --output_dir outputs/exp3_feg_ppr \\",
        "  --top_n 50",
        "```",
        "",
        "## Results",
        "",
        "| Method | MRR |" + "|".join(f" R@{k} " for k in k_values) + "|",
        "|---|---|" + "|".join("---" for _ in k_values) + "|",
    ]
    method_order = [
        "hybrid", "hybrid+semantic_ppr", "hybrid+financial_ppr",
        "hybrid+full_ppr", "hybrid+full_ppr+constraint",
    ]
    for method in method_order:
        m = summaries.get(method, {})
        if not m:
            continue
        label = {
            "hybrid": "Hybrid Retrieval",
            "hybrid+semantic_ppr": "Hybrid + Semantic Graph + PPR",
            "hybrid+financial_ppr": "Hybrid + Financial Graph + PPR",
            "hybrid+full_ppr": "Hybrid + Full Graph + PPR",
            "hybrid+full_ppr+constraint": "Hybrid + Full Graph + PPR + Constraint",
        }.get(method, method)
        row = f"| {label} | {m['mrr']}"
        for k in k_values:
            row += f" | {m['evidence_recall'].get(k, 0):.4f}"
        row += " |"
        lines.append(row)

    lines.extend([
        "",
        "## Answers to Exp3 questions",
        "",
        "1. **Does PPR improve Recall@10?** " +
        ("Yes" if best[1].get("evidence_recall", {}).get(10, 0) > hybrid.get("evidence_recall", {}).get(10, 0) else "Check results"),
        "2. **Does PPR improve MRR?** " +
        ("Yes" if best[1].get("mrr", 0) > hybrid.get("mrr", 0) else "Check results"),
        "3. **Financial Graph > Semantic Graph?** Compare financial_ppr vs semantic_ppr rows above.",
        "4. **Full Graph best?** Full graph (financial + semantic) should give the best results,",
        "   showing that structural and semantic edges are complementary.",
        "5. **Main error reductions**: Year edges → reduce Wrong Year,",
        "   Metric edges → reduce Wrong Metric, Company edges → reduce Wrong Company.",
        "",
        "## Output files",
        "",
        "| File | Description |",
        "|---|---|",
        "| `hybrid_results.jsonl` | Hybrid retrieval baseline |",
        "| `ppr_results_semantic_graph.jsonl` | Semantic-only graph PPR rerank |",
        "| `ppr_results_financial_graph.jsonl` | Financial structure graph PPR rerank |",
        "| `ppr_results_full_graph.jsonl` | Full graph PPR rerank |",
        "| `ppr_results_full_graph_constraint.jsonl` | Full graph PPR + constraint fusion |",
        "| `metrics_summary.csv` | Aggregate metrics |",
        "| `error_reduction_summary.csv` | Delta vs Hybrid baseline |",
        "| `graph_stats.json` | Graph node/edge statistics |",
        "| `case_studies.md` | Examples where PPR corrects errors |",
        "",
        f"Generated: {datetime.now().isoformat()}",
    ])

    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Exp3: FEG + PPR reranking")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output_dir", default="outputs/exp3_feg_ppr")
    parser.add_argument("--num_samples", type=int, default=0,
                        help="0 = all samples")
    parser.add_argument("--top_n", type=int, default=50,
                        help="Number of candidates from Hybrid retrieval")
    parser.add_argument("--output_k", type=int, default=10,
                        help="Top-k results to save per query")
    parser.add_argument("--ppr_alpha", type=float, default=0.85,
                        help="PPR damping factor")
    parser.add_argument("--no_semantic_edges", action="store_true",
                        help="Skip semantic edges (faster, smaller graph)")
    parser.add_argument("--max_distractor_files", type=int, default=50)
    parser.add_argument("--dense_device", default="cpu",
                        help="Device for Dense encoding (cpu avoids GPU OOM)")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    cfg.ensure_dirs()

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = cfg.root_dir / output_dir

    k_values = cfg.evaluation.get("recall_k_values", [1, 3, 5, 10, 20])
    add_semantic = not args.no_semantic_edges

    print("=" * 60)
    print("  EXP3: Financial Evidence Graph + PPR Reranking")
    print("=" * 60)
    print(f"  Output:         {output_dir}")
    print(f"  Top-N hybrid:   {args.top_n}")
    print(f"  Semantic edges: {add_semantic}")
    print(f"  PPR alpha:      {args.ppr_alpha}")

    # 1. Data
    print("\n[1/4] Loading data...")
    samples = load_dataset("finder", cfg.data_dir)
    if args.num_samples > 0:
        samples = samples[:args.num_samples]
    print(f"  {len(samples)} QA samples")

    # 2. Corpus
    print("[2/4] Building corpus...")
    corpus_chunks, gold_map = build_corpus(
        samples, cfg, max_distractor_files=args.max_distractor_files,
    )
    print(f"  {len(corpus_chunks)} chunks, {len(gold_map)} queries with gold evidence")

    # 3. Run experiment
    print("[3/4] Running PPR reranking...")
    t0 = time.time()
    all_results = run_exp3(
        cfg, samples, corpus_chunks, gold_map,
        top_n=args.top_n,
        output_k=args.output_k,
        add_semantic_edges=add_semantic,
        ppr_alpha=args.ppr_alpha,
        dense_device=args.dense_device,
        verbose=True,
    )
    elapsed = time.time() - t0
    print(f"  Completed in {elapsed:.1f}s")

    # 4. Evaluate & output
    print("[4/4] Evaluating and writing outputs...")
    # Collect graph stats
    entity_map = extract_entities(corpus_chunks)
    g_full = build_financial_evidence_graph(
        corpus_chunks,
        entity_map=entity_map,
        add_semantic_edges=add_semantic,
        add_company_nodes=True,
        add_filing_nodes=True,
        add_section_nodes=True,
        add_same_entity_edges=True,
        use_edge_weights=True,
    )
    graph_stats = {
        "num_nodes": g_full.num_nodes,
        "num_edges": g_full.num_edges,
        "node_types": {
            k: sum(1 for nid, nt in g_full.node_types.items() if nt == k)
            for k in set(g_full.node_types.values())
        },
        "edge_types": g_full.edge_type_counts(),
    }

    summaries = evaluate_all(all_results, k_values, graph_stats)
    write_outputs(output_dir, all_results, summaries, graph_stats, k_values)

    # Print summary table
    print("\n" + "=" * 70)
    print("  EXP3 RESULTS: PPR RERANKING")
    print("=" * 70)
    header = f"{'Method':<40} {'MRR':>7}"
    for k in k_values:
        header += f" {'R@'+str(k):>8}"
    print(header)
    print("-" * 70)

    method_order = [
        "hybrid", "hybrid+semantic_ppr", "hybrid+financial_ppr",
        "hybrid+full_ppr", "hybrid+full_ppr+constraint",
    ]
    method_labels = {
        "hybrid": "Hybrid",
        "hybrid+semantic_ppr": "Semantic Graph + PPR",
        "hybrid+financial_ppr": "Financial Graph + PPR",
        "hybrid+full_ppr": "Full Graph + PPR",
        "hybrid+full_ppr+constraint": "Full Graph + PPR + Constraint",
    }
    for method in method_order:
        m = summaries.get(method)
        if m is None:
            continue
        label = method_labels.get(method, method)
        row = f"{label:<40} {m['mrr']:>7.4f}"
        for k in k_values:
            row += f" {m['evidence_recall'].get(k, 0):>8.4f}"
        print(row)

    print(f"\nOutput:  {output_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
