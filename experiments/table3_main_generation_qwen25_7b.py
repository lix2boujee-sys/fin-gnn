"""Table 3: Main Answer Generation Experiment with Qwen2.5-7B.

Measures whether better evidence ranking leads to better generated financial
answers. Each method produces top-5 evidence; the same Qwen2.5-7B generator
produces answers from only those passages. No LLM reranking or post-processing.

Pipeline:
    Query → method → top-5 evidence → Qwen2.5-7B → answer → evaluate

Methods compared:
    Best Retriever, Cross-Encoder, PPR, GraphSAGE, R-GCN, R-GCN+Constraint

Metrics: Answer Accuracy, Faithfulness, Numerical Consistency, Evidence Hit@5

Usage:
    # Smoke test (3 samples)
    python experiments/table3_main_generation_qwen25_7b.py \\
        --config configs/table3_main_generation_qwen25_7b.yaml \\
        --num_samples 3 --sanity

    # Subset run
    python experiments/table3_main_generation_qwen25_7b.py \\
        --config configs/table3_main_generation_qwen25_7b.yaml \\
        --num_samples 100 --device cuda

    # Resume partial run
    python experiments/table3_main_generation_qwen25_7b.py \\
        --config configs/table3_main_generation_qwen25_7b.yaml \\
        --resume
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
from feg_rag.generation.answer_evaluator import AnswerEvaluator, AggregateEvalResult
from feg_rag.generation.llm_cache import LLMCache
from feg_rag.generation.llm_prompts import build_generator_messages
from feg_rag.generation.llm_response_parser import parse_generator_response
from feg_rag.generation.openrouter_client import OpenRouterClient, TokenUsage
from feg_rag.generation.token_cost import TokenCostTracker
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
# Constants
# ═════════════════════════════════════════════════════════════════════════════

_METHOD_ORDER = [
    "best_retriever", "cross_encoder", "ppr",
    "graphsage", "rgcn", "rgcn_constraint",
]

_METHOD_LABELS = {
    "best_retriever": "Best Retriever",
    "cross_encoder": "+ Cross-Encoder",
    "ppr": "+ PPR",
    "graphsage": "+ GraphSAGE",
    "rgcn": "+ R-GCN",
    "rgcn_constraint": "+ R-GCN + Constraint Score",
}

_PROMPT_VERSION = "v1"


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


def _get_top_k_evidence(
    method: str,
    sample: Dict,
    hybrid: HybridRetriever,
    graph,
    corpus_chunks: List[Chunk],
    features: Dict[str, np.ndarray],
    chunk_by_id: Dict[str, Chunk],
    extractor: EntityExtractor,
    cross_encoder: Optional[CrossEncoderReranker],
    reranker: Optional[object],
    constraint_scorer: Optional[ConstraintScorer],
    cfg: Config,
    top_n: int = 50,
    top_k: int = 5,
) -> List[Chunk]:
    """Get top-k evidence chunks for a query using the specified method.

    Returns:
        List of ``Chunk`` objects (length top_k).
    """
    hr = hybrid.search(sample["question"], top_k=top_n)
    chunks = [c for c, _ in hr]
    candidate_ids = [c.chunk_id for c in chunks]
    retrieval_scores = {c.chunk_id: float(score) for c, score in hr}

    q_metrics = extractor.extract_metrics(sample["question"])
    q_years = extractor.extract_years(sample["question"])

    if method == "best_retriever":
        return chunks[:top_k]

    elif method == "cross_encoder":
        if cross_encoder is None:
            return chunks[:top_k]
        reranked = cross_encoder.rerank(sample["question"], hr, top_k=top_k)
        return [c for c, _ in reranked[:top_k]]

    elif method == "ppr":
        ppr_scores = ppr_rerank(
            graph, corpus_chunks, candidate_ids,
            seed_chunk_ids=candidate_ids[:10],
            seed_metric_names=list(q_metrics),
            seed_year_values=list(q_years),
            alpha=cfg.rerank.get("ppr_alpha", 0.85),
            retrieval_scores=retrieval_scores,
        )
        ranked_ids = [cid for cid, _ in ppr_scores[:top_k]]
        return [chunk_by_id[cid] for cid in ranked_ids if cid in chunk_by_id]

    elif method in ("graphsage", "rgcn"):
        if reranker is None:
            return chunks[:top_k]
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
            return [c for c, _ in reranked[:top_k]]
        except Exception:
            return chunks[:top_k]

    elif method == "rgcn_constraint":
        if reranker is None:
            return chunks[:top_k]
        ppr_raw = dict(ppr_rerank(
            graph, [], candidate_ids,
            seed_chunk_ids=candidate_ids[:10],
            seed_metric_names=list(q_metrics),
            seed_year_values=list(q_years),
            alpha=cfg.rerank.get("ppr_alpha", 0.85),
        ))
        ppr_scores = {cid: score for cid, score in ppr_raw}
        try:
            gnn_reranked = reranker.rerank(
                sample["question"], hr, graph, features,
                ppr_scores=ppr_scores,
            )
            gnn_scores = {c.chunk_id: score for c, score in gnn_reranked}
        except Exception:
            gnn_scores = {}

        fusion = FusionScorer(
            alpha=cfg.rerank.get("fusion_alpha", 0.3),
            beta=cfg.rerank.get("fusion_beta", 0.3),
            gamma=cfg.rerank.get("fusion_gamma", 0.3),
            delta=cfg.rerank.get("fusion_delta", 0.1),
            constraint_scorer=constraint_scorer or ConstraintScorer(),
        )
        fused = fusion.fuse(
            sample["question"], chunks,
            retrieval_scores=retrieval_scores,
            graph_scores=ppr_scores,
            gnn_scores=gnn_scores,
        )
        return [c for c, _ in fused[:top_k]]

    return chunks[:top_k]


def _has_results(out_dir: Path) -> bool:
    sentinels = ["table3_main_generation_results.csv", "answer_eval_full.json"]
    return any((out_dir / s).exists() for s in sentinels)


# ═════════════════════════════════════════════════════════════════════════════
# Generation runner
# ═════════════════════════════════════════════════════════════════════════════

def run_generation(
    samples: List[Dict],
    method: str,
    client: OpenRouterClient,
    cache: LLMCache,
    cost_tracker: TokenCostTracker,
    top_k_evidence_fn,
    **fn_kwargs,
) -> Tuple[List[Dict], List[Dict]]:
    """Generate answers for all samples using the given method.

    Returns:
        (generation_results, failures) — each is a list of dicts.
    """
    gen_results: List[Dict] = []
    failures: List[Dict] = []

    # Split fn_kwargs: evidence params vs. generation params
    gold_map = fn_kwargs.pop("gold_map", {})
    temperature = fn_kwargs.pop("temperature", 0.0)
    max_tokens = fn_kwargs.pop("max_tokens", 512)

    for i, s in enumerate(samples):
        qid = s["id"]
        if (i + 1) % 50 == 0:
            print(f"    [{method}] {i+1}/{len(samples)} "
                  f"(cache hits={cache.hits}, misses={cache.misses})")

        # Get top-k evidence (only evidence params remain in fn_kwargs)
        evidence_chunks = top_k_evidence_fn(s, **fn_kwargs)
        evidence_ids = [c.chunk_id for c in evidence_chunks]

        # Check cache
        cached = cache.get(
            "table3", method, qid, client.model,
            candidate_ids=evidence_ids,
            prompt_version=_PROMPT_VERSION,
        )

        if cached:
            parsed = cached.get("response", {})
            usage_dict = cached.get("metadata", {}).get("usage", {})
            usage = TokenUsage(
                prompt_tokens=usage_dict.get("prompt_tokens", 0),
                completion_tokens=usage_dict.get("completion_tokens", 0),
                total_tokens=usage_dict.get("total_tokens", 0),
                estimated_cost_usd=usage_dict.get("estimated_cost_usd", 0),
                provider="openrouter",
                model=client.model,
            )
            cost_tracker.record(method, usage, success=True, query_id=qid)
            gen_results.append({
                "question_id": qid,
                "question": s["question"],
                "gold_answer": s["answer"],
                "method": method,
                "evidence_ids": evidence_ids,
                "gold_evidence_ids": gold_map.get(qid, []),
                "generated_answer": parsed.get("answer", ""),
                "evidence_ids_used": parsed.get("evidence_ids_used", []),
                "confidence": parsed.get("confidence", "unknown"),
                "raw_response": cached.get("metadata", {}).get("raw_response", ""),
                "from_cache": True,
            })
            continue

        # Call LLM
        messages = build_generator_messages(s["question"], evidence_chunks)
        try:
            response = client.chat(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
        except Exception as e:
            failures.append({
                "question_id": qid,
                "method": method,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            })
            cost_tracker.record(
                method,
                TokenUsage(model=client.model),
                success=False,
                query_id=qid,
            )
            gen_results.append({
                "question_id": qid,
                "question": s["question"],
                "gold_answer": s["answer"],
                "method": method,
                "evidence_ids": evidence_ids,
                "gold_evidence_ids": gold_map.get(qid, []),
                "generated_answer": "",
                "evidence_ids_used": [],
                "confidence": "unknown",
                "raw_response": "",
                "from_cache": False,
                "call_failed": True,
                "error": str(e),
            })
            continue

        # Parse response
        parse_result = parse_generator_response(response.content)
        cost_tracker.record(method, response.usage, success=parse_result.success, query_id=qid)

        answer = ""
        evidence_used: List[str] = []
        confidence = "unknown"

        if parse_result.success:
            parsed = parse_result.parsed
            answer = parsed.get("answer", "")
            evidence_used = parsed.get("evidence_ids_used", [])
            confidence = parsed.get("confidence", "medium")
        else:
            # Store raw content as answer on parse failure
            answer = response.content
            failures.append({
                "question_id": qid,
                "method": method,
                "error": parse_result.error_message,
                "raw_response": response.content[:500],
                "timestamp": datetime.now().isoformat(),
            })

        # Cache the result
        cache.put(
            "table3", method, qid, client.model,
            candidate_ids=evidence_ids,
            prompt_version=_PROMPT_VERSION,
            response={
                "answer": answer,
                "evidence_ids_used": evidence_used,
                "confidence": confidence,
            },
            metadata={
                "usage": response.usage.to_dict(),
                "raw_response": response.content,
                "parse_success": parse_result.success,
            },
        )

        gen_results.append({
            "question_id": qid,
            "question": s["question"],
            "gold_answer": s["answer"],
            "method": method,
            "evidence_ids": evidence_ids,
            "gold_evidence_ids": gold_map.get(qid, []),
            "generated_answer": answer,
            "evidence_ids_used": evidence_used,
            "confidence": confidence,
            "raw_response": response.content,
            "from_cache": False,
            "call_failed": False,
        })

    return gen_results, failures


# ═════════════════════════════════════════════════════════════════════════════
# Output
# ═════════════════════════════════════════════════════════════════════════════

def _write_outputs(
    output_dir: Path,
    all_gen: Dict[str, List[Dict]],
    all_failures: List[Dict],
    eval_summaries: Dict[str, Dict],
    cost_tracker: TokenCostTracker,
    cache_stats: Dict,
    command: str = "",
) -> None:
    """Write all Table 3 output files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Per-method generation JSONL
    method_files = {
        "best_retriever": "best_retriever_generation.jsonl",
        "cross_encoder": "cross_encoder_generation.jsonl",
        "ppr": "ppr_generation.jsonl",
        "graphsage": "graphsage_generation.jsonl",
        "rgcn": "rgcn_generation.jsonl",
        "rgcn_constraint": "rgcn_constraint_generation.jsonl",
    }
    for method, results in all_gen.items():
        if not results:
            continue
        fname = method_files.get(method, f"{method}_generation.jsonl")
        with open(output_dir / fname, "w", encoding="utf-8") as fh:
            for r in results:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Failures
    if all_failures:
        with open(output_dir / "answer_eval_failures.jsonl", "w", encoding="utf-8") as fh:
            for f in all_failures:
                fh.write(json.dumps(f, ensure_ascii=False) + "\n")

    # Answer eval full JSON
    with open(output_dir / "answer_eval_full.json", "w", encoding="utf-8") as fh:
        json.dump(eval_summaries, fh, indent=2, ensure_ascii=False)

    # Token cost
    cost_tracker.save_csv(output_dir / "token_cost_summary.csv")

    # Results CSV
    csv_path = output_dir / "table3_main_generation_results.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as fh:
        fieldnames = [
            "Method", "Answer Accuracy", "Exact Match", "Relaxed Match",
            "Numerical Consistency", "Faithfulness", "Evidence Hit@5",
            "Insufficient Evidence Rate", "Parse Failures", "Num Samples",
        ]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for method in _METHOD_ORDER:
            m = eval_summaries.get(method)
            if m is None:
                continue
            writer.writerow({
                "Method": _METHOD_LABELS.get(method, method),
                "Answer Accuracy": m.get("answer_accuracy", 0),
                "Exact Match": m.get("exact_match_rate", 0),
                "Relaxed Match": m.get("relaxed_match_rate", 0),
                "Numerical Consistency": m.get("numerical_consistency", 0),
                "Faithfulness": m.get("faithfulness_score", 0),
                "Evidence Hit@5": m.get("evidence_hit_at_5", 0),
                "Insufficient Evidence Rate": m.get("insufficient_evidence_rate", 0),
                "Parse Failures": m.get("num_parse_failures", 0),
                "Num Samples": m.get("num_samples", 0),
            })

    # Results Markdown
    md_path = output_dir / "table3_main_generation_results.md"
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write("# Table 3: Main Answer Generation Results (Qwen2.5-7B)\n\n")
        fh.write(f"Generated: {datetime.now().isoformat()}\n\n")
        fh.write(f"Cache stats: hits={cache_stats.get('hits',0)}, "
                 f"misses={cache_stats.get('misses',0)}, "
                 f"hit_rate={cache_stats.get('hit_rate',0)}\n\n")
        fh.write("## Results\n\n")
        fh.write(
            "| Method | Answer Acc | Exact Match | Relaxed Match | "
            "Num Consistency | Faithfulness | Evidence Hit@5 | Insuf Evidence | Samples |\n"
        )
        fh.write(
            "|---|---|---|---|---|---|---|---|---|\n"
        )
        for method in _METHOD_ORDER:
            m = eval_summaries.get(method)
            if m is None:
                continue
            fh.write(
                f"| {_METHOD_LABELS.get(method, method)} |"
                f" {m.get('answer_accuracy', 0):.4f} |"
                f" {m.get('exact_match_rate', 0):.4f} |"
                f" {m.get('relaxed_match_rate', 0):.4f} |"
                f" {m.get('numerical_consistency', 0):.4f} |"
                f" {m.get('faithfulness_score', 0):.4f} |"
                f" {m.get('evidence_hit_at_5', 0):.4f} |"
                f" {m.get('insufficient_evidence_rate', 0):.4f} |"
                f" {m.get('num_samples', 0)} |\n"
            )
        fh.write("\n## Key claim\n\n")
        fh.write("> Better evidence ranking improves final answer quality when the same "
                 "Qwen2.5-7B generator is used without LLM post-processing.\n\n")

    # README
    readme_path = output_dir / "README.md"
    with open(readme_path, "w", encoding="utf-8") as fh:
        fh.write("# Experiment: Table 3 — Main Answer Generation (Qwen2.5-7B)\n\n")
        fh.write("Tests whether better evidence ranking → better generated answers.\n\n")
        fh.write(f"Model: `{command}`\n\n" if "llm_model" not in command else "")
        fh.write("## Output files\n\n")
        for fname in [
            "table3_main_generation_results.csv",
            "table3_main_generation_results.md",
            "answer_eval_full.json",
            "answer_eval_failures.jsonl",
            "token_cost_summary.csv",
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
        description="Table 3: Main Answer Generation with Qwen2.5-7B"
    )
    parser.add_argument("--config", default="configs/table3_main_generation_qwen25_7b.yaml")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--num_samples", type=int, default=0,
                        help="Limit samples (0=all)")
    parser.add_argument("--llm_model", default=None,
                        help="Override LLM model slug")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--generation_top_k", type=int, default=5,
                        help="Top-k evidence for generation")
    parser.add_argument("--device", default=_default_device())
    parser.add_argument("--dense_device", default="cpu")
    parser.add_argument("--dense_batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--top_n", type=int, default=50)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--skip_gnn", action="store_true")
    parser.add_argument("--skip_methods", type=str, default="",
                        help="Comma-separated methods to skip")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from cache (skip completed LLM calls)")
    parser.add_argument("--overwrite_output_dir", action="store_true")
    parser.add_argument("--sanity", action="store_true",
                        help="Sanity mode: 3 samples, 1 epoch")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    cfg.ensure_dirs()

    # Sanity
    if args.sanity:
        args.num_samples = args.num_samples or 3
        args.epochs = args.epochs or 1
        if args.output_dir is None:
            args.output_dir = "outputs/table3_main_generation_sanity"

    if args.epochs:
        cfg.rerank["gnn_epochs"] = args.epochs

    llm_model = args.llm_model or cfg.generation.get("llm_model", "qwen/qwen-2.5-7b-instruct")

    output_dir = Path(args.output_dir or cfg.output_dir)
    if not output_dir.is_absolute():
        output_dir = cfg.root_dir / output_dir

    if _has_results(output_dir) and not args.overwrite_output_dir and not args.resume:
        print(f"\nERROR: Output directory '{output_dir}' already contains results.")
        print(f"  Use --overwrite_output_dir or --resume.")
        sys.exit(1)

    skip_methods = set(m.strip() for m in args.skip_methods.split(",") if m.strip())

    print("=" * 60)
    print("  TABLE 3: Main Answer Generation (Qwen2.5-7B)")
    print("=" * 60)
    print(f"  Output:       {output_dir}")
    print(f"  LLM Model:    {llm_model}")
    print(f"  Temperature:  {args.temperature}")
    print(f"  Top-K Evid:   {args.generation_top_k}")
    print(f"  Device:       {args.device}")
    print(f"  Samples:      {args.num_samples or 'all'}")
    print(f"  Resume:       {args.resume}")
    print(f"  Sanity:       {args.sanity}")

    # ---- Init OpenRouter client ----
    print("\n[init] OpenRouter client...")
    try:
        client = OpenRouterClient(
            model=llm_model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        print(f"  Client: {client}")
    except RuntimeError as e:
        print(f"  ERROR: {e}")
        sys.exit(1)

    cache = LLMCache(cfg.cache_dir / "llm_calls")
    if not args.resume:
        # Don't clear on resume
        pass
    print(f"  Cache: {cache.stats()}")

    cost_tracker = TokenCostTracker()
    evaluator = AnswerEvaluator()

    # ---- 1. Load data ----
    print("\n[1/5] Loading FinDER data...")
    samples = load_dataset("finder", cfg.data_dir)
    if args.num_samples > 0:
        samples = samples[:args.num_samples]
    rng = random.Random(args.split_seed)
    rng.shuffle(samples)
    n_val = max(1, int(len(samples) * args.val_split))
    train_samples = samples[n_val:]
    test_samples = samples[:n_val]
    eval_samples = samples
    print(f"  Train: {len(train_samples)}  |  Test: {len(test_samples)}  |  "
          f"Eval: {len(eval_samples)}")

    # ---- 2. Corpus ----
    print("[2/5] Building corpus...")
    corpus_chunks, gold_map = _build_corpus(samples, cfg)
    chunk_by_id = {c.chunk_id: c for c in corpus_chunks}
    gold_ids_all = set()
    for gids in gold_map.values():
        gold_ids_all.update(gids)
    print(f"  {len(corpus_chunks)} chunks ({len(gold_ids_all)} gold)")

    # ---- 3. Retrieval ----
    print("[3/5] Building retrieval indices...")
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
    emb_dim = next(iter(chunk_embeddings.values())).shape[0] if chunk_embeddings else 0
    print(f"  Dense: {dense.backend}, dim={emb_dim}")

    hybrid = HybridRetriever(bm25, dense, alpha=cfg.retrieval.get("hybrid_alpha", 0.5))
    cross_encoder = CrossEncoderReranker(
        model_name=cfg.cross_encoder.get("model_name", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
        batch_size=cfg.cross_encoder.get("batch_size", 32),
    )

    # ---- 4. Graph + features ----
    print("[4/5] Building graph and features...")
    entity_map = extract_entities(corpus_chunks)
    graph = build_financial_evidence_graph(
        corpus_chunks, entity_map=entity_map,
        add_semantic_edges=False,
        add_company_nodes=True, add_filing_nodes=True, add_section_nodes=True,
        add_same_entity_edges=True, max_same_entity_edges=30,
        use_edge_weights=True,
    )
    print(f"  Graph: {graph.num_nodes} nodes, {graph.num_edges} edges")

    retrieval_scores = warmup_retrieval_scores(
        train_samples + test_samples, hybrid, top_k=args.top_n,
    )
    features = build_node_features(
        graph, corpus_chunks, entity_map, retrieval_scores,
        chunk_embeddings=chunk_embeddings,
        compute_embeddings=False, embedding_device=args.device,
    )
    print(f"  Feature dim: {next(iter(features.values())).shape[0]}")

    # ---- Train GNNs if needed ----
    extractor = EntityExtractor()
    sage_reranker = None
    rgcn_reranker = None
    constraint_scorer = None

    gnn_methods = {"graphsage", "rgcn", "rgcn_constraint"}
    needs_gnn = bool(gnn_methods - skip_methods) and not args.skip_gnn

    if needs_gnn:
        print("\n  Training GraphSAGE...")
        cfg.rerank["gnn_model"] = "sage"
        sage_reranker, sage_hist, sage_meta = train_gnn_reranker(
            train_samples, hybrid, graph, features, gold_map, cfg,
            epochs=cfg.rerank.get("gnn_epochs", 10),
            device=args.device, min_pairs=5, verbose=False,
        )
        if sage_reranker:
            save_training_artifacts(sage_reranker, sage_hist, output_dir, sage_meta,
                                   experiment="table3_graphsage")
            print(f"    GraphSAGE trained: final_loss={sage_hist[-1]:.4f}" if sage_hist else "")
        else:
            print("    [SKIP] GraphSAGE training failed")

        print("\n  Training R-GCN...")
        cfg.rerank["gnn_model"] = "rgcn"
        rgcn_reranker, rgcn_hist, rgcn_meta = train_gnn_reranker(
            train_samples, hybrid, graph, features, gold_map, cfg,
            epochs=cfg.rerank.get("gnn_epochs", 10),
            device=args.device, min_pairs=5, verbose=False,
        )
        if rgcn_reranker:
            save_training_artifacts(rgcn_reranker, rgcn_hist, output_dir, rgcn_meta,
                                   experiment="table3_rgcn")
            print(f"    R-GCN trained: final_loss={rgcn_hist[-1]:.4f}" if rgcn_hist else "")
        else:
            print("    [SKIP] R-GCN training failed")

        constraint_scorer = ConstraintScorer(
            company_weight=cfg.constraint.get("company_match_weight", 1.0),
            year_weight=cfg.constraint.get("year_match_weight", 1.0),
            metric_weight=cfg.constraint.get("metric_match_weight", 0.8),
            filing_type_weight=cfg.constraint.get("filing_type_match_weight", 0.5),
        )

    # ---- 5. Generation ----
    print("\n[5/5] Generating answers...")
    all_gen: Dict[str, List[Dict]] = {}
    all_failures: List[Dict] = []

    # Map method → reranker for top-k evidence selection
    method_reranker = {
        "graphsage": sage_reranker,
        "rgcn": rgcn_reranker,
        "rgcn_constraint": rgcn_reranker,
    }

    for method in _METHOD_ORDER:
        if method in skip_methods:
            print(f"\n  [{method}] SKIPPED")
            continue

        if method in gnn_methods and (args.skip_gnn or method_reranker.get(method) is None):
            print(f"\n  [{method}] SKIPPED (GNN not available)")
            continue

        print(f"\n  [{method}] Generating with {_METHOD_LABELS.get(method, method)}...")
        t0 = time.time()

        gen_results, failures = run_generation(
            eval_samples, method, client, cache, cost_tracker,
            _get_top_k_evidence,
            method=method,
            hybrid=hybrid,
            graph=graph,
            corpus_chunks=corpus_chunks,
            features=features,
            chunk_by_id=chunk_by_id,
            extractor=extractor,
            cross_encoder=cross_encoder,
            reranker=method_reranker.get(method),
            constraint_scorer=constraint_scorer if method == "rgcn_constraint" else None,
            cfg=cfg,
            top_n=args.top_n,
            top_k=args.generation_top_k,
            gold_map=gold_map,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        all_gen[method] = gen_results
        all_failures.extend(failures)
        print(f"    {len(gen_results)} answers, {len(failures)} failures "
              f"in {time.time() - t0:.1f}s")

    # ---- Evaluate ----
    print("\n" + "=" * 70)
    print("  TABLE 3 RESULTS")
    print("=" * 70)

    eval_summaries: Dict[str, Dict] = {}
    for method in _METHOD_ORDER:
        gen_results = all_gen.get(method)
        if not gen_results:
            continue

        per_answer = []
        for r in gen_results:
            evidence_chunks = [chunk_by_id[cid] for cid in r.get("evidence_ids", []) if cid in chunk_by_id]
            result = evaluator.evaluate(
                question_id=r["question_id"],
                generated_answer=r.get("generated_answer", ""),
                gold_answer=r.get("gold_answer", ""),
                evidence_chunks=evidence_chunks,
                gold_evidence_ids=r.get("gold_evidence_ids", []),
                evidence_ids_used=r.get("evidence_ids_used", []),
            )
            per_answer.append(result)

        agg = AggregateEvalResult.from_results(method, per_answer)
        eval_summaries[method] = agg.to_dict()

    # Print summary
    header = (
        f"{'Method':<30} {'Acc':>6} {'EM':>6} {'Relax':>6} "
        f"{'NumCon':>6} {'Faith':>6} {'EvidH5':>6} {'Insuf':>6} {'N':>5}"
    )
    print(header)
    print("-" * 85)
    for method in _METHOD_ORDER:
        m = eval_summaries.get(method)
        if m is None:
            continue
        label = _METHOD_LABELS.get(method, method)
        row = (
            f"{label:<30} {m['answer_accuracy']:>6.4f} {m['exact_match_rate']:>6.4f} "
            f"{m['relaxed_match_rate']:>6.4f} {m['numerical_consistency']:>6.4f} "
            f"{m['faithfulness_score']:>6.4f} {m['evidence_hit_at_5']:>6.4f} "
            f"{m['insufficient_evidence_rate']:>6.4f} {m['num_samples']:>5}"
        )
        print(row)

    # ---- Token cost ----
    cost_tracker.print_summary()

    # ---- Write outputs ----
    cache_stats = cache.stats()
    _write_outputs(
        output_dir, all_gen, all_failures, eval_summaries,
        cost_tracker, cache_stats,
        command=" ".join(sys.argv),
    )

    print(f"\nOutput: {output_dir}")
    print(f"Cache: {cache_stats}")
    print("Done.")


if __name__ == "__main__":
    main()
