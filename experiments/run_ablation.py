"""Ablation experiments.

Paper plan §11:
  1. Graph structure ablation (semantic vs financial edges)
  2. Edge type ablation (remove metric, year, table edges)
  3. Verifier ablation (with / without numerical verifier)

Usage:
    python experiments/run_ablation.py --config configs/default.yaml
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
from feg_rag.generation.verifier import NumericalVerifier
from feg_rag.graph.builder import FinancialEvidenceGraph
from feg_rag.graph.entities import extract_entities
from feg_rag.rerank.ppr import ppr_rerank
from feg_rag.retrieval.bm25 import BM25Retriever
from feg_rag.retrieval.dense import DenseRetriever
from feg_rag.retrieval.hybrid import HybridRetriever


def main() -> None:
    parser = argparse.ArgumentParser(description="Ablation experiments")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--num_samples", type=int, default=30)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    cfg.ensure_dirs()

    # ---- Setup (shared) ----
    print("[Setup] Loading data and building indices...")
    samples = load_dataset(cfg.datasets[0], cfg.data_dir)[: args.num_samples]
    corpus_chunks, gold_map = _build_corpus(samples, cfg)

    bm25 = BM25Retriever(k1=cfg.retrieval["bm25_k1"], b=cfg.retrieval["bm25_b"])
    bm25.index(corpus_chunks)
    dense = DenseRetriever(
        model_name=cfg.retrieval["dense_model"],
        query_instruction=cfg.retrieval.get("dense_query_instruction"),
        e5_max_seq_length=cfg.retrieval.get("e5_max_seq_length", 512),
        e5_batch_size=cfg.retrieval.get("e5_batch_size"),
        debug=cfg.retrieval.get("debug_dense", False),
    )
    dense.index(corpus_chunks)
    hybrid = HybridRetriever(bm25, dense, alpha=cfg.retrieval["hybrid_alpha"])

    entity_map = extract_entities(corpus_chunks)
    generator = LLMGenerator(
        model=cfg.generation["model"],
        temperature=cfg.generation["temperature"],
        max_tokens=cfg.generation["max_tokens"],
    )
    verifier = NumericalVerifier()

    # ---- Ablation 1: Graph structure ----
    print("\n[Ablation 1] Graph structure comparison...")

    # (a) Semantic-only graph
    g_semantic = FinancialEvidenceGraph()
    g_semantic.build(corpus_chunks, entity_map=entity_map, add_semantic_edges=True)

    # (b) Financial structure graph (no semantic edges)
    g_financial = FinancialEvidenceGraph()
    g_financial.build(corpus_chunks, entity_map=entity_map, add_semantic_edges=False)

    # (c) Full graph (needs embeddings for semantic edges; skip for now)
    g_full = g_financial  # placeholder

    graph_variants = {
        "semantic_graph": g_semantic,
        "financial_structure_graph": g_financial,
        "full_graph": g_full,
    }

    ablation1_results: Dict[str, Dict] = {}
    for gname, graph in graph_variants.items():
        print(f"  Evaluating {gname}...")
        results = _run_ppr_pipeline(
            samples, hybrid, graph, corpus_chunks, gold_map, generator, verifier, cfg
        )
        er = compute_all_metrics(gname, results, k_values=cfg.evaluation["recall_k_values"])
        ablation1_results[gname] = {
            "answer_accuracy": er.answer_accuracy,
            "mrr": er.mrr,
            "evidence_recall": er.evidence_recall,
            "numerical_consistency": er.numerical_consistency,
        }

    # ---- Ablation 2: Verifier ----
    print("\n[Ablation 2] Verifier ablation...")
    verifier_results: Dict[str, Dict] = {}
    for verifier_enabled in [False, True]:
        label = "with_verifier" if verifier_enabled else "without_verifier"
        print(f"  Evaluating {label}...")
        results = _run_ppr_pipeline(
            samples,
            hybrid,
            g_financial,
            corpus_chunks,
            gold_map,
            generator,
            verifier if verifier_enabled else None,
            cfg,
        )
        er = compute_all_metrics(label, results, k_values=cfg.evaluation["recall_k_values"])
        verifier_results[label] = {
            "answer_accuracy": er.answer_accuracy,
            "numerical_consistency": er.numerical_consistency,
            "hallucination_rate": er.hallucination_rate,
        }

    # ---- Print ----
    print("\n" + "=" * 60)
    print("ABLATION RESULTS")
    print("=" * 60)

    print("\n1. Graph Structure:")
    for gname, m in ablation1_results.items():
        print(f"  {gname}:")
        print(f"    Answer Accuracy: {m['answer_accuracy']:.4f}")
        print(f"    MRR:             {m['mrr']:.4f}")
        print(f"    Evidence R@5:    {m['evidence_recall'].get(5, 0):.4f}")

    print("\n2. Verifier:")
    for label, m in verifier_results.items():
        print(f"  {label}:")
        print(f"    Answer Accuracy:      {m['answer_accuracy']:.4f}")
        print(f"    Numerical Consistency: {m['numerical_consistency']:.4f}")
        print(f"    Hallucination Rate:    {m['hallucination_rate']:.4f}")

    # ---- Save ----
    output = {
        "graph_structure": ablation1_results,
        "verifier": verifier_results,
    }
    output_path = args.output or str(cfg.output_dir / "ablation_results.json")
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════

def _build_corpus(samples, cfg) -> Tuple[List[Chunk], Dict[str, List[str]]]:
    corpus: List[Chunk] = []
    gold_map: Dict[str, List[str]] = {}
    for s in samples:
        gold_ids = []
        for text in s["evidence_texts"]:
            for c in chunk_text(text, cfg.chunk_size, cfg.chunk_overlap, doc_id=s["id"]):
                corpus.append(c)
                gold_ids.append(c.chunk_id)
        gold_map[s["id"]] = gold_ids
    return corpus, gold_map


def _run_ppr_pipeline(
    samples, hybrid, graph, corpus_chunks, gold_map, generator, verifier, cfg
) -> List[Dict]:
    from feg_rag.graph.entities import EntityExtractor
    extractor = EntityExtractor()

    results: List[Dict] = []
    for s in samples:
        question = s["question"]
        q_id = s["id"]

        hybrid_results = hybrid.search(question, top_k=cfg.retrieval["top_k"])
        candidate_ids = [c.chunk_id for c, _ in hybrid_results]

        q_metrics = extractor.extract_metrics(question)
        q_years = extractor.extract_years(question)

        ppr_scores = ppr_rerank(
            graph, corpus_chunks, candidate_ids,
            seed_chunk_ids=candidate_ids[:10],
            seed_metric_names=list(q_metrics),
            seed_year_values=list(q_years),
            alpha=cfg.rerank["ppr_alpha"],
        )
        ppr_sorted = sorted(ppr_scores, key=lambda x: x[1], reverse=True)
        top_ids = [cid for cid, _ in ppr_sorted[: cfg.generation["top_k_evidence"]]]
        top_chunks = []
        for cid in top_ids:
            for c in corpus_chunks:
                if c.chunk_id == cid:
                    top_chunks.append(c)
                    break

        gen = generator.generate(question, top_chunks)

        is_consistent = False
        is_hallucination = False
        if verifier:
            vres = verifier.verify(gen, question)
            is_consistent = vres.is_consistent
            is_hallucination = not vres.evidence_fully_cited

        import re
        pred = re.sub(r"\s+", " ", gen.answer.lower().strip().rstrip("."))
        gold = re.sub(r"\s+", " ", s["answer"].lower().strip().rstrip("."))

        results.append({
            "question_id": q_id,
            "question": question,
            "gold_answer": s["answer"],
            "generated_answer": gen.answer,
            "gold_evidence_ids": gold_map.get(q_id, []),
            "retrieved_chunk_ids": [cid for cid, _ in ppr_sorted],
            "answer_is_correct": pred == gold,
            "is_consistent": is_consistent,
            "is_hallucination": is_hallucination,
        })

    return results


def _find_chunk(cid: str, chunks: List[Chunk]) -> Chunk | None:
    for c in chunks:
        if c.chunk_id == cid:
            return c
    return None


if __name__ == "__main__":
    main()
