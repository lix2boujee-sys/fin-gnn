"""Baseline experiment: BM25, Dense, Hybrid retrieval → LLM answer.

Paper plan §10.1: compares No-RAG LLM, BM25-RAG, Dense-RAG, Hybrid-RAG.

Usage:
    python experiments/run_baselines.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from feg_rag.config import Config
from feg_rag.data.chunker import Chunk, chunk_text
from feg_rag.data.loader import load_dataset
from feg_rag.evaluation.metrics import compute_all_metrics
from feg_rag.generation.llm import LLMGenerator
from feg_rag.retrieval.bm25 import BM25Retriever
from feg_rag.retrieval.dense import DenseRetriever
from feg_rag.retrieval.hybrid import HybridRetriever


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline retrieval experiments")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--num_samples", type=int, default=50,
                        help="Number of QA samples to evaluate (0=all)")
    parser.add_argument("--output", default=None, help="Path to save results JSON")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    cfg.ensure_dirs()
    print(f"Config loaded. Output dir: {cfg.output_dir}")

    # ---- 1. Load data ----
    print("\n[1/5] Loading dataset...")
    samples = load_dataset(cfg.datasets[0], cfg.data_dir)
    if args.num_samples > 0:
        samples = samples[: args.num_samples]
    print(f"  Loaded {len(samples)} QA samples")

    # ---- 2. Build chunk corpus ----
    print("\n[2/5] Building chunk corpus from evidence texts...")
    corpus_chunks: List[Chunk] = []
    gold_map: Dict[str, List[str]] = {}  # question_id → gold chunk_ids
    for s in samples:
        gold_ids = []
        for text in s["evidence_texts"]:
            chunks = chunk_text(
                text,
                chunk_size=cfg.chunk_size,
                chunk_overlap=cfg.chunk_overlap,
                doc_id=s["id"],
            )
            for c in chunks:
                corpus_chunks.append(c)
                gold_ids.append(c.chunk_id)
        gold_map[s["id"]] = gold_ids

    # Also chunk 10-K documents as distractors (if available)
    edgar_dir = cfg.edgar_dir
    txt_files = list(edgar_dir.rglob("*.txt")) if edgar_dir.exists() else []
    for tf in txt_files[:5]:  # limit to 5 reports for speed
        try:
            from feg_rag.data.chunker import chunk_report
            corpus_chunks.extend(chunk_report(tf, cfg.chunk_size, cfg.chunk_overlap))
        except Exception:
            pass
    print(f"  Corpus: {len(corpus_chunks)} chunks total")

    # ---- 3. Build retrievers ----
    print("\n[3/5] Building retrieval indices...")

    print("  - BM25 index...")
    bm25 = BM25Retriever(k1=cfg.retrieval["bm25_k1"], b=cfg.retrieval["bm25_b"])
    bm25.index(corpus_chunks)

    print("  - Dense index...")
    dense = DenseRetriever(
        model_name=cfg.retrieval["dense_model"],
        query_instruction=cfg.retrieval.get("dense_query_instruction"),
        e5_max_seq_length=cfg.retrieval.get("e5_max_seq_length", 512),
        e5_batch_size=cfg.retrieval.get("e5_batch_size"),
        debug=cfg.retrieval.get("debug_dense", False),
    )
    dense.index(corpus_chunks)

    hybrid = HybridRetriever(bm25, dense, alpha=cfg.retrieval["hybrid_alpha"])

    # ---- 4. Run retrieval + answer ----
    print("\n[4/5] Running retrieval + answer generation...")

    generator = LLMGenerator(
        model=cfg.generation["model"],
        temperature=cfg.generation["temperature"],
        max_tokens=cfg.generation["max_tokens"],
    )
    top_k_gen = cfg.generation["top_k_evidence"]
    top_k_retrieval = cfg.retrieval["top_k"]

    all_results: Dict[str, List[Dict]] = {
        "bm25_rag": [],
        "dense_rag": [],
        "hybrid_rag": [],
    }

    for method, retriever in [
        ("bm25_rag", lambda q: bm25.search(q, top_k=top_k_retrieval)),
        ("dense_rag", lambda q: dense.search(q, top_k=top_k_retrieval)),
        ("hybrid_rag", lambda q: hybrid.search(q, top_k=top_k_retrieval)),
    ]:
        print(f"\n  --- {method} ---")
        for i, s in enumerate(samples):
            if i % 10 == 0:
                print(f"    {i}/{len(samples)}")
            question = s["question"]
            retrieved = retriever(question)
            top_chunks = [c for c, _ in retrieved[:top_k_gen]]
            retrieved_ids = [c.chunk_id for c, _ in retrieved]

            # Generate answer
            gen = generator.generate(question, top_chunks)

            all_results[method].append(
                {
                    "question_id": s["id"],
                    "question": question,
                    "gold_answer": s["answer"],
                    "generated_answer": gen.answer,
                    "gold_evidence_ids": gold_map.get(s["id"], []),
                    "retrieved_chunk_ids": retrieved_ids,
                    "answer_is_correct": _is_correct(gen.answer, s["answer"]),
                }
            )

    # ---- 5. Evaluate ----
    print("\n[5/5] Computing metrics...")
    k_vals = cfg.evaluation["recall_k_values"]
    summary: Dict[str, Dict] = {}

    for method_name, results in all_results.items():
        er = compute_all_metrics(method_name, results, k_values=k_vals)
        summary[method_name] = {
            "evidence_recall": er.evidence_recall,
            "evidence_precision": er.evidence_precision,
            "mrr": er.mrr,
            "ndcg": er.ndcg,
            "answer_accuracy": er.answer_accuracy,
            "f1": er.f1,
            "num_samples": er.num_samples,
        }

    # ---- Print ----
    print("\n" + "=" * 60)
    print("BASELINE RESULTS")
    print("=" * 60)
    for method, metrics in summary.items():
        print(f"\n{method}:")
        print(f"  Answer Accuracy: {metrics['answer_accuracy']:.4f}")
        print(f"  MRR:             {metrics['mrr']:.4f}")
        print(f"  Evidence R@5:    {metrics['evidence_recall'].get(5, 0):.4f}")
        print(f"  Evidence R@10:   {metrics['evidence_recall'].get(10, 0):.4f}")
        print(f"  F1:              {metrics['f1']:.4f}")

    # ---- Save ----
    output_path = args.output or str(cfg.output_dir / "baseline_results.json")
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


def _is_correct(pred: str, gold: str) -> bool:
    """Simple exact-match after normalisation."""
    import re
    p = re.sub(r"\s+", " ", pred.lower().strip().rstrip("."))
    g = re.sub(r"\s+", " ", gold.lower().strip().rstrip("."))
    return p == g


if __name__ == "__main__":
    main()
