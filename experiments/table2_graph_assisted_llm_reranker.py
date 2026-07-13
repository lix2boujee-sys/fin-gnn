"""Table 2: Graph-Assisted LLM Reranker (auxiliary experiment).

Tests whether graph-based reranking can serve as a pre-filter before LLM
reranking, reducing the number of candidates sent to the LLM while keeping
performance close to direct LLM rerank top-50.

Methods:
    LLM rerank top-50   (50 candidates)
    LLM rerank top-20   (20 candidates)
    Cross-Encoder → LLM (10 candidates)
    PPR → LLM           (10 candidates)
    R-GCN → LLM         (10 candidates)

Metrics: Recall@5, MRR, nDCG@5, Token Cost

Usage:
    # Smoke test
    python experiments/table2_graph_assisted_llm_reranker.py \\
        --config configs/table2_graph_assisted_llm_reranker_qwen25_7b.yaml \\
        --num_samples 3 --sanity

    # Controlled subset
    python experiments/table2_graph_assisted_llm_reranker.py \\
        --config configs/table2_graph_assisted_llm_reranker_qwen25_7b.yaml \\
        --num_samples 100 --device cuda
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
from feg_rag.generation.llm_cache import LLMCache
from feg_rag.generation.llm_prompts import build_reranker_messages
from feg_rag.generation.llm_response_parser import parse_reranker_response
from feg_rag.generation.openrouter_client import OpenRouterClient, TokenUsage
from feg_rag.generation.token_cost import TokenCostTracker
from feg_rag.graph.builder import build_financial_evidence_graph
from feg_rag.graph.entities import EntityExtractor, extract_entities
from feg_rag.graph.features import build_node_features
from feg_rag.rerank.ppr import ppr_rerank
from feg_rag.rerank.train import (
    train_gnn_reranker,
    warmup_retrieval_scores,
)
from feg_rag.retrieval.bm25 import BM25Retriever
from feg_rag.retrieval.cross_encoder import CrossEncoderReranker
from feg_rag.retrieval.dense import DenseRetriever
from feg_rag.retrieval.hybrid import HybridRetriever


# ═════════════════════════════════════════════════════════════════════════════
# Constants
# ═════════════════════════════════════════════════════════════════════════════

_PROMPT_VERSION = "v2"  # reranker prompt version


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════

def _default_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _build_corpus(samples, cfg, max_distractor_files=50):
    corpus, gold_map = [], {}
    for s in samples:
        gold_ids = []
        for text in s["evidence_texts"]:
            for c in chunk_text(text, cfg.chunk_size, cfg.chunk_overlap, doc_id=s["id"]):
                corpus.append(c)
                gold_ids.append(c.chunk_id)
        gold_map[s["id"]] = gold_ids
    edgar_dir = cfg.edgar_dir
    if edgar_dir.exists():
        files = list(edgar_dir.rglob("*.txt")) or list(edgar_dir.rglob("*.html"))
        for tf in files[:max_distractor_files]:
            try:
                corpus.extend(chunk_report(tf, cfg.chunk_size, cfg.chunk_overlap))
            except Exception:
                pass
    return corpus, gold_map


def _has_results(out_dir: Path) -> bool:
    sentinels = ["table2_graph_assisted_llm_reranker.csv", "token_cost_summary.csv"]
    return any((out_dir / s).exists() for s in sentinels)


def _get_prefiltered_candidates(
    method: str,
    sample: Dict,
    hybrid: HybridRetriever,
    graph,
    features: Dict[str, np.ndarray],
    chunk_by_id: Dict[str, Chunk],
    extractor: EntityExtractor,
    cross_encoder: Optional[CrossEncoderReranker],
    reranker: Optional[object],
    cfg: Config,
    top_n: int = 50,
    output_count: int = 10,
) -> Tuple[List[Chunk], str]:
    """Get pre-filtered candidates for the given method.

    Returns:
        (candidate_chunks, method_label_for_logging).
    """
    hr = hybrid.search(sample["question"], top_k=top_n)
    chunks = [c for c, _ in hr]
    candidate_ids = [c.chunk_id for c in chunks]
    retrieval_scores = {c.chunk_id: float(score) for c, score in hr}

    q_metrics = extractor.extract_metrics(sample["question"])
    q_years = extractor.extract_years(sample["question"])

    if method == "llm_rerank_top50":
        return chunks[:50], "LLM rerank top-50"

    elif method == "llm_rerank_top20":
        return chunks[:20], "LLM rerank top-20"

    elif method == "cross_encoder_to_llm":
        if cross_encoder is None:
            return chunks[:output_count], "Cross-Encoder → LLM"
        reranked = cross_encoder.rerank(sample["question"], hr, top_k=output_count)
        return [c for c, _ in reranked[:output_count]], "Cross-Encoder → LLM"

    elif method == "ppr_to_llm":
        ppr_scores = ppr_rerank(
            graph, [], candidate_ids,
            seed_chunk_ids=candidate_ids[:10],
            seed_metric_names=list(q_metrics),
            seed_year_values=list(q_years),
            alpha=cfg.rerank.get("ppr_alpha", 0.85),
            retrieval_scores=retrieval_scores,
        )
        ranked_ids = [cid for cid, _ in ppr_scores[:output_count]]
        result = [chunk_by_id[cid] for cid in ranked_ids if cid in chunk_by_id]
        return result, "PPR → LLM"

    elif method == "rgcn_to_llm":
        if reranker is None:
            return chunks[:output_count], "R-GCN → LLM"
        ppr_raw = dict(ppr_rerank(
            graph, [], candidate_ids,
            seed_chunk_ids=candidate_ids[:10],
            seed_metric_names=list(q_metrics),
            seed_year_values=list(q_years),
            alpha=cfg.rerank.get("ppr_alpha", 0.85),
            retrieval_scores=retrieval_scores,
        ))
        try:
            reranked = reranker.rerank(
                sample["question"], hr, graph, features,
                ppr_scores=ppr_raw,
            )
            return [c for c, _ in reranked[:output_count]], "R-GCN → LLM"
        except Exception:
            return chunks[:output_count], "R-GCN → LLM"

    return chunks[:output_count], method


# ═════════════════════════════════════════════════════════════════════════════
# LLM Reranker runner
# ═════════════════════════════════════════════════════════════════════════════

def run_llm_reranker(
    samples: List[Dict],
    method: str,
    client: OpenRouterClient,
    cache: LLMCache,
    cost_tracker: TokenCostTracker,
    get_candidates_fn,
    **fn_kwargs,
) -> Tuple[List[Dict], List[Dict]]:
    """Run LLM reranker on all samples.

    For each query:
      1. Get pre-filtered candidates.
      2. Send to LLM to rerank.
      3. Parse ranked_candidate_ids.
      4. Use result as final ranking for metric computation.

    Returns:
        (rerank_results, failures).
    """
    results: List[Dict] = []
    failures: List[Dict] = []

    for i, s in enumerate(samples):
        qid = s["id"]
        if (i + 1) % 50 == 0:
            print(f"    [{method}] {i+1}/{len(samples)} "
                  f"(cache hits={cache.hits})")

        gold_ids = fn_kwargs.get("gold_map", {}).get(qid, [])

        # Get pre-filtered candidates
        candidates, _label = get_candidates_fn(s, **fn_kwargs)
        candidate_ids = [c.chunk_id for c in candidates]

        # Check cache
        cached = cache.get(
            "table2", method, qid, client.model,
            candidate_ids=candidate_ids,
            prompt_version=_PROMPT_VERSION,
        )
        if cached:
            ranked_ids = cached.get("response", {}).get("ranked_candidate_ids", candidate_ids)
            usage_dict = cached.get("metadata", {}).get("usage", {})
            usage = TokenUsage(
                prompt_tokens=usage_dict.get("prompt_tokens", 0),
                completion_tokens=usage_dict.get("completion_tokens", 0),
                total_tokens=usage_dict.get("total_tokens", 0),
                estimated_cost_usd=usage_dict.get("estimated_cost_usd", 0),
                provider="openrouter", model=client.model,
            )
            cost_tracker.record(method, usage, success=True, query_id=qid)
            results.append({
                "question_id": qid,
                "question": s["question"],
                "gold_answer": s.get("answer", ""),
                "gold_evidence_ids": gold_ids,
                "retrieved_chunk_ids": ranked_ids,
                "method": method,
                "from_cache": True,
            })
            continue

        # Build messages and call LLM
        messages = build_reranker_messages(s["question"], candidates)
        try:
            response = client.chat(
                messages,
                temperature=fn_kwargs.get("temperature", 0.0),
                max_tokens=fn_kwargs.get("max_tokens", 1024),
                response_format={"type": "json_object"},
            )
        except Exception as e:
            failures.append({
                "question_id": qid, "method": method,
                "error": str(e), "timestamp": datetime.now().isoformat(),
            })
            cost_tracker.record(method, TokenUsage(model=client.model),
                              success=False, query_id=qid)
            # Fall back to original candidate order
            results.append({
                "question_id": qid, "question": s["question"],
                "gold_answer": s.get("answer", ""),
                "gold_evidence_ids": gold_ids,
                "retrieved_chunk_ids": candidate_ids,
                "method": method, "from_cache": False, "call_failed": True,
            })
            continue

        # Parse
        parse_result = parse_reranker_response(response.content)
        cost_tracker.record(method, response.usage, success=parse_result.success, query_id=qid)

        if parse_result.success:
            ranked_ids = parse_result.parsed.get("ranked_candidate_ids", [])
            # Ensure all candidate IDs are present (some may be missing from LLM output)
            ranked_set = set(ranked_ids)
            for cid in candidate_ids:
                if cid not in ranked_set:
                    ranked_ids.append(cid)
        else:
            ranked_ids = candidate_ids
            failures.append({
                "question_id": qid, "method": method,
                "error": parse_result.error_message,
                "raw_response": response.content[:500],
                "timestamp": datetime.now().isoformat(),
            })

        # Cache
        cache.put(
            "table2", method, qid, client.model,
            candidate_ids=candidate_ids,
            prompt_version=_PROMPT_VERSION,
            response={"ranked_candidate_ids": ranked_ids},
            metadata={
                "usage": response.usage.to_dict(),
                "parse_success": parse_result.success,
            },
        )

        results.append({
            "question_id": qid,
            "question": s["question"],
            "gold_answer": s.get("answer", ""),
            "gold_evidence_ids": gold_ids,
            "retrieved_chunk_ids": ranked_ids,
            "method": method,
            "from_cache": False,
            "call_failed": False,
        })

    return results, failures


# ═════════════════════════════════════════════════════════════════════════════
# Output
# ═════════════════════════════════════════════════════════════════════════════

_METHOD_ORDER = [
    "llm_rerank_top50", "llm_rerank_top20",
    "cross_encoder_to_llm", "ppr_to_llm", "rgcn_to_llm",
]

_METHOD_LABELS = {
    "llm_rerank_top50": "LLM rerank top-50",
    "llm_rerank_top20": "LLM rerank top-20",
    "cross_encoder_to_llm": "Cross-Encoder → LLM",
    "ppr_to_llm": "PPR → LLM",
    "rgcn_to_llm": "R-GCN → LLM",
}

_CANDIDATE_COUNTS = {
    "llm_rerank_top50": 50,
    "llm_rerank_top20": 20,
    "cross_encoder_to_llm": 10,
    "ppr_to_llm": 10,
    "rgcn_to_llm": 10,
}


def _write_outputs(
    output_dir: Path,
    all_results: Dict[str, List[Dict]],
    all_failures: List[Dict],
    summaries: Dict[str, Dict],
    cost_tracker: TokenCostTracker,
    cache_stats: Dict,
    k_values: List[int],
    command: str = "",
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    method_files = {
        "llm_rerank_top50": "llm_rerank_top50_results.jsonl",
        "llm_rerank_top20": "llm_rerank_top20_results.jsonl",
        "cross_encoder_to_llm": "cross_encoder_to_llm_results.jsonl",
        "ppr_to_llm": "ppr_to_llm_results.jsonl",
        "rgcn_to_llm": "rgcn_to_llm_results.jsonl",
    }
    for method, results in all_results.items():
        fname = method_files.get(method, f"{method}_results.jsonl")
        with open(output_dir / fname, "w", encoding="utf-8") as fh:
            for r in results:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    if all_failures:
        with open(output_dir / "llm_call_failures.jsonl", "w", encoding="utf-8") as fh:
            for f in all_failures:
                fh.write(json.dumps(f, ensure_ascii=False) + "\n")

    with open(output_dir / "metrics_full.json", "w", encoding="utf-8") as fh:
        json.dump(summaries, fh, indent=2, ensure_ascii=False)

    cost_tracker.save_csv(output_dir / "token_cost_summary.csv")

    # CSV
    csv_path = output_dir / "table2_graph_assisted_llm_reranker.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as fh:
        fieldnames = (
            ["Method", "Candidates"] +
            [f"Recall@{k}" for k in k_values] + ["MRR"] +
            [f"nDCG@{k}" for k in k_values] +
            ["Prompt Tokens", "Total Tokens", "Cost USD", "Num Samples"]
        )
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for method in _METHOD_ORDER:
            m = summaries.get(method)
            if m is None:
                continue
            cost = cost_tracker.get_summary(method)
            row = {
                "Method": _METHOD_LABELS.get(method, method),
                "Candidates": _CANDIDATE_COUNTS.get(method, "?"),
                "MRR": m.get("mrr", 0),
                "Num Samples": m.get("num_samples", 0),
            }
            for k in k_values:
                row[f"Recall@{k}"] = round(m.get(f"recall@{k}", 0), 4)
                row[f"nDCG@{k}"] = round(m.get(f"ndcg@{k}", 0), 4)
            if cost:
                row["Prompt Tokens"] = cost.prompt_tokens
                row["Total Tokens"] = cost.total_tokens
                row["Cost USD"] = round(cost.estimated_cost_usd, 6)
            writer.writerow(row)

    # Markdown
    md_path = output_dir / "table2_graph_assisted_llm_reranker.md"
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write("# Table 2: Graph-Assisted LLM Reranker\n\n")
        fh.write(f"Generated: {datetime.now().isoformat()}\n\n")
        fh.write("| Method | Cand |")
        for k in k_values:
            fh.write(f" Recall@{k} |")
        fh.write(" MRR |")
        for k in k_values:
            fh.write(f" nDCG@{k} |")
        fh.write(" Tokens | Cost |\n")
        fh.write("|---|")
        for _ in range(3 + 2 * len(k_values)):
            fh.write("---|")
        fh.write("\n")
        for method in _METHOD_ORDER:
            m = summaries.get(method)
            if m is None:
                continue
            cost = cost_tracker.get_summary(method)
            row = (
                f"| {_METHOD_LABELS.get(method, method)} |"
                f" {_CANDIDATE_COUNTS.get(method, '?')} |"
            )
            for k in k_values:
                row += f" {m.get(f'recall@{k}', 0):.4f} |"
            row += f" {m.get('mrr', 0):.4f} |"
            for k in k_values:
                row += f" {m.get(f'ndcg@{k}', 0):.4f} |"
            if cost:
                row += f" {cost.total_tokens:,} | ${cost.estimated_cost_usd:.4f} |"
            else:
                row += " - | - |"
            fh.write(row + "\n")
        fh.write("\n## Key claim\n\n")
        fh.write("> Graph reranking can act as a pre-filter for the LLM reranker, "
                 "reducing LLM input candidate count and token cost while preserving "
                 "or approaching the effectiveness of direct LLM rerank top-50.\n\n")

    # README
    readme = output_dir / "README.md"
    with open(readme, "w", encoding="utf-8") as fh:
        fh.write("# Experiment: Table 2 — Graph-Assisted LLM Reranker\n\n")
        fh.write("Auxiliary experiment: tests graph pre-filtering for LLM reranking.\n\n")
        fh.write("## Output files\n\n")
        for fname in [
            "table2_graph_assisted_llm_reranker.csv",
            "table2_graph_assisted_llm_reranker.md",
            "token_cost_summary.csv",
            "llm_call_failures.jsonl",
            "metrics_full.json",
        ]:
            fh.write(f"- `{fname}`\n")
        for fname in method_files.values():
            fh.write(f"- `{fname}`\n")
        fh.write(f"\nGenerated: {datetime.now().isoformat()}\n")


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Table 2: Graph-Assisted LLM Reranker"
    )
    parser.add_argument("--config", default="configs/table2_graph_assisted_llm_reranker_qwen25_7b.yaml")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--num_samples", type=int, default=0)
    parser.add_argument("--llm_model", default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--max_candidates", type=int, default=50,
                        help="Max candidates for LLM rerank top-N methods")
    parser.add_argument("--prefilter_method", type=str, default=None,
                        help="Run only this prefilter method (e.g. ppr_to_llm)")
    parser.add_argument("--device", default=_default_device())
    parser.add_argument("--dense_device", default="cpu")
    parser.add_argument("--dense_batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--top_n", type=int, default=50)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite_output_dir", action="store_true")
    parser.add_argument("--sanity", action="store_true")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    cfg.ensure_dirs()

    if args.sanity:
        args.num_samples = args.num_samples or 3
        args.epochs = args.epochs or 1
        if args.output_dir is None:
            args.output_dir = "outputs/table2_graph_assisted_llm_reranker_sanity"

    if args.epochs:
        cfg.rerank["gnn_epochs"] = args.epochs

    llm_model = args.llm_model or cfg.generation.get("llm_model", cfg._raw.get("llm", {}).get("model", "qwen/qwen-2.5-7b-instruct"))
    output_dir = Path(args.output_dir or cfg.output_dir)
    if not output_dir.is_absolute():
        output_dir = cfg.root_dir / output_dir

    if _has_results(output_dir) and not args.overwrite_output_dir and not args.resume:
        print(f"\nERROR: '{output_dir}' already has results. Use --overwrite_output_dir.")
        sys.exit(1)

    k_values = cfg.evaluation.get("recall_k_values", [5])

    print("=" * 60)
    print("  TABLE 2: Graph-Assisted LLM Reranker")
    print("=" * 60)
    print(f"  Output:       {output_dir}")
    print(f"  LLM Model:    {llm_model}")
    print(f"  Temperature:  {args.temperature}")
    print(f"  Samples:      {args.num_samples or 'all'}")
    print(f"  Resume:       {args.resume}")

    # Init OpenRouter
    print("\n[init] OpenRouter client...")
    try:
        client = OpenRouterClient(
            model=llm_model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
    except RuntimeError as e:
        print(f"  ERROR: {e}")
        sys.exit(1)
    print(f"  {client}")

    cache = LLMCache(cfg.cache_dir / "llm_calls")
    cost_tracker = TokenCostTracker()

    # ---- 1. Data ----
    print("\n[1/4] Loading data...")
    samples = load_dataset("finder", cfg.data_dir)
    if args.num_samples > 0:
        samples = samples[:args.num_samples]
    rng = random.Random(args.split_seed)
    rng.shuffle(samples)
    n_val = max(1, int(len(samples) * args.val_split))
    train_samples = samples[n_val:]
    eval_samples = samples[:n_val]
    print(f"  Train: {len(train_samples)}  |  Eval: {len(eval_samples)}")

    # ---- 2. Corpus + retrieval + graph + features ----
    print("[2/4] Building corpus, retrieval, graph...")
    corpus_chunks, gold_map = _build_corpus(samples, cfg)
    chunk_by_id = {c.chunk_id: c for c in corpus_chunks}

    bm25 = BM25Retriever(k1=cfg.retrieval.get("bm25_k1", 1.5), b=cfg.retrieval.get("bm25_b", 0.75))
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
    hybrid = HybridRetriever(bm25, dense, alpha=cfg.retrieval.get("hybrid_alpha", 0.5))

    cross_encoder = CrossEncoderReranker(
        model_name=cfg.cross_encoder.get("model_name", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
        batch_size=cfg.cross_encoder.get("batch_size", 32),
    )

    entity_map = extract_entities(corpus_chunks)
    graph = build_financial_evidence_graph(
        corpus_chunks, entity_map=entity_map,
        add_semantic_edges=False,
        add_company_nodes=True, add_filing_nodes=True, add_section_nodes=True,
        add_same_entity_edges=True, max_same_entity_edges=30,
        use_edge_weights=True,
    )

    retrieval_scores = warmup_retrieval_scores(
        train_samples + eval_samples, hybrid, top_k=args.top_n,
    )
    features = build_node_features(
        graph, corpus_chunks, entity_map, retrieval_scores,
        chunk_embeddings=chunk_embeddings,
        compute_embeddings=False, embedding_device=args.device,
    )
    print(f"  Graph: {graph.num_nodes}n, {graph.num_edges}e  |  "
          f"Feature dim: {next(iter(features.values())).shape[0]}")

    # ---- 3. Train R-GCN (needed for rgcn_to_llm) ----
    extractor = EntityExtractor()
    rgcn_reranker = None

    needs_rgcn = not args.prefilter_method or args.prefilter_method == "rgcn_to_llm"
    if needs_rgcn:
        print("\n[3/4] Training R-GCN for pre-filter...")
        cfg.rerank["gnn_model"] = "rgcn"
        rgcn_reranker, rgcn_hist, rgcn_meta = train_gnn_reranker(
            train_samples, hybrid, graph, features, gold_map, cfg,
            epochs=cfg.rerank.get("gnn_epochs", 10),
            device=args.device, min_pairs=5, verbose=False,
        )
        if rgcn_reranker:
            print(f"    R-GCN trained: final_loss={rgcn_hist[-1]:.4f}" if rgcn_hist else "")
        else:
            print("    [SKIP] R-GCN training failed — rgcn_to_llm will be unavailable")

    # ---- 4. Run LLM reranker for each method ----
    print("\n[4/4] Running LLM reranking...")
    all_results: Dict[str, List[Dict]] = {}
    all_failures: List[Dict] = []

    methods_to_run = _METHOD_ORDER
    if args.prefilter_method:
        if args.prefilter_method in _METHOD_ORDER:
            methods_to_run = [args.prefilter_method]
        else:
            print(f"  Unknown prefilter_method '{args.prefilter_method}'")
            sys.exit(1)

    # Candidate counts for each method
    method_candidate_counts = {
        "llm_rerank_top50": min(args.max_candidates, 50),
        "llm_rerank_top20": min(args.max_candidates, 20),
        "cross_encoder_to_llm": 10,
        "ppr_to_llm": 10,
        "rgcn_to_llm": 10,
    }

    for method in methods_to_run:
        if method == "cross_encoder_to_llm" and cross_encoder is None:
            print(f"\n  [{method}] SKIPPED (no cross-encoder)")
            continue
        if method == "rgcn_to_llm" and rgcn_reranker is None:
            print(f"\n  [{method}] SKIPPED (R-GCN not available)")
            continue

        print(f"\n  [{method}] {_METHOD_LABELS.get(method, method)}...")
        t0 = time.time()

        output_count = method_candidate_counts.get(method, 10)
        results, failures = run_llm_reranker(
            eval_samples, method, client, cache, cost_tracker,
            _get_prefiltered_candidates,
            method=method,
            hybrid=hybrid,
            graph=graph,
            features=features,
            chunk_by_id=chunk_by_id,
            extractor=extractor,
            cross_encoder=cross_encoder if method == "cross_encoder_to_llm" else None,
            reranker=rgcn_reranker if method == "rgcn_to_llm" else None,
            cfg=cfg,
            top_n=args.top_n,
            output_count=output_count,
            gold_map=gold_map,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        all_results[method] = results
        all_failures.extend(failures)
        print(f"    {len(results)} results, {len(failures)} failures "
              f"in {time.time() - t0:.1f}s")

    # ---- Compute summaries ----
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

    # Print summary
    print("\n" + "=" * 70)
    print("  TABLE 2 RESULTS")
    print("=" * 70)
    header = f"{'Method':<30} {'Cand':>5} {'MRR':>7}"
    for k in k_values:
        header += f" {'R@'+str(k):>8} {'nDCG@'+str(k):>8}"
    header += f" {'Tokens':>10} {'Cost USD':>10}"
    print(header)
    print("-" * (60 + 18 * len(k_values)))
    for method in _METHOD_ORDER:
        m = summaries.get(method)
        if m is None:
            continue
        cost = cost_tracker.get_summary(method)
        row = (
            f"{_METHOD_LABELS.get(method, method):<30} "
            f"{_CANDIDATE_COUNTS.get(method, '?'):>5} "
            f"{m['mrr']:>7.4f}"
        )
        for k in k_values:
            row += f" {m.get(f'recall@{k}', 0):>8.4f} {m.get(f'ndcg@{k}', 0):>8.4f}"
        if cost:
            row += f" {cost.total_tokens:>10,} ${cost.estimated_cost_usd:>9.4f}"
        else:
            row += f" {'-':>10} {'-':>10}"
        print(row)

    cost_tracker.print_summary()

    # ---- Write outputs ----
    cache_stats = cache.stats()
    _write_outputs(
        output_dir, all_results, all_failures, summaries,
        cost_tracker, cache_stats, k_values,
        command=" ".join(sys.argv),
    )

    print(f"\nOutput: {output_dir}")
    print(f"Cache: {cache_stats}")
    print("Done.")


if __name__ == "__main__":
    main()
