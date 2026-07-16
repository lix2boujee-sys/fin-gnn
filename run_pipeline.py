"""End-to-end FEG-RAG pipeline.

Orchestrates the full workflow: data → chunks → retrieval → graph → rerank →
generation → evaluation.

Usage:
    python run_pipeline.py --config configs/default.yaml
    python run_pipeline.py --config configs/default.yaml --step retrieval
    python run_pipeline.py --config configs/default.yaml --skip generation
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Ensure the project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from feg_rag.config import Config
from feg_rag.data.chunker import Chunk
from feg_rag.data.corpus import build_benchmark_corpus
from feg_rag.data.hard_negatives import generate_hard_negatives
from feg_rag.data.loader import load_dataset
from feg_rag.evaluation.error_analysis import ErrorAnalyzer
from feg_rag.evaluation.metrics import compute_all_metrics
from feg_rag.generation.llm import LLMGenerator
from feg_rag.generation.verifier import NumericalVerifier
from feg_rag.graph.builder import build_financial_evidence_graph
from feg_rag.graph.entities import extract_entities, EntityExtractor
from feg_rag.graph.features import build_node_features
from feg_rag.rerank.ppr import ppr_rerank
from feg_rag.retrieval.bm25 import BM25Retriever
from feg_rag.retrieval.dense import DenseRetriever
from feg_rag.retrieval.hybrid import HybridRetriever


# ═════════════════════════════════════════════════════════════════════════════
# Pipeline steps
# ═════════════════════════════════════════════════════════════════════════════

PIPELINE_STEPS = [
    "data",
    "retrieval",
    "graph",
    "rerank",
    "generation",
    "evaluation",
]


class Pipeline:
    """Stateful pipeline that caches intermediate results to disk."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.cfg.ensure_dirs()
        self._state: Dict = {}

    # ------------------------------------------------------------------
    # Step 1: Data loading & chunking
    # ------------------------------------------------------------------

    def step_data(
        self,
        split: str | None = None,
        num_samples: int = 0,
    ) -> Tuple[List[Dict], List[Chunk], Dict[str, List[str]]]:
        """Load dataset, build chunks, map gold evidence."""
        print("[data] Loading datasets...")
        all_samples: List[Dict] = []
        for ds_name in self.cfg.datasets:
            try:
                samples = load_dataset(
                    ds_name,
                    self.cfg.data_dir,
                    split=split or self.cfg._raw.get("data_split"),
                    files=self.cfg._raw.get("data_files"),
                )
                all_samples.extend(samples)
                print(f"  {ds_name}: {len(samples)} samples")
            except FileNotFoundError as e:
                print(f"  [SKIP] {ds_name}: {e}")

        print(f"[data] Total: {len(all_samples)} QA samples")
        if num_samples > 0:
            all_samples = all_samples[:num_samples]
            print(f"[data] Limited to {len(all_samples)} QA samples")
        if not all_samples:
            raise RuntimeError("No QA samples loaded; check data_dir/data_split/data_files.")

        print("[data] Building document corpus and aligning gold evidence...")
        corpus_chunks, gold_map, alignments = build_benchmark_corpus(
            all_samples, self.cfg
        )

        print(f"[data] Corpus: {len(corpus_chunks)} total chunks")
        print(f"[data] Alignment records: {len(alignments)}")
        if not corpus_chunks:
            raise RuntimeError("Empty corpus; check edgar_dir and corpus settings.")
        if not any(gold_map.values()):
            raise RuntimeError("No gold evidence matched to corpus chunks.")

        self._save_data_cache(all_samples, corpus_chunks, gold_map)
        self._state["samples"] = all_samples
        self._state["corpus_chunks"] = corpus_chunks
        self._state["gold_map"] = gold_map
        return all_samples, corpus_chunks, gold_map

    # ------------------------------------------------------------------
    # Step 2: Retrieval indices
    # ------------------------------------------------------------------

    def step_retrieval(
        self, corpus_chunks: List[Chunk]
    ) -> Tuple[BM25Retriever, DenseRetriever, HybridRetriever]:
        print("[retrieval] Building BM25 index...")
        bm25 = BM25Retriever(
            k1=self.cfg.retrieval["bm25_k1"], b=self.cfg.retrieval["bm25_b"]
        )
        bm25.index(corpus_chunks)

        print("[retrieval] Building dense index...")
        dense = DenseRetriever(
            model_name=self.cfg.retrieval["dense_model"],
            query_instruction=self.cfg.retrieval.get("dense_query_instruction"),
            e5_max_seq_length=self.cfg.retrieval.get("e5_max_seq_length", 512),
            e5_batch_size=self.cfg.retrieval.get("e5_batch_size"),
            debug=self.cfg.retrieval.get("debug_dense", False),
        )
        dense.index(corpus_chunks)

        hybrid = HybridRetriever(bm25, dense, alpha=self.cfg.retrieval["hybrid_alpha"])
        print("[retrieval] Done.")

        self._state["bm25"] = bm25
        self._state["dense"] = dense
        self._state["hybrid"] = hybrid
        return bm25, dense, hybrid

    # ------------------------------------------------------------------
    # Step 3: Financial evidence graph
    # ------------------------------------------------------------------

    def step_graph(self, corpus_chunks: List[Chunk]):
        print("[graph] Extracting entities...")
        entity_map = extract_entities(corpus_chunks)
        metric_count = len({m for e in entity_map.values() for m in e.metrics})
        year_count = len({y for e in entity_map.values() for y in e.years})
        print(f"  Unique metrics: {metric_count}, Unique years: {year_count}")

        print("[graph] Building financial evidence graph...")
        graph = build_financial_evidence_graph(
            corpus_chunks, entity_map=entity_map, add_semantic_edges=False
        )
        print(f"  Graph: {graph.num_nodes} nodes, {graph.num_edges} edges")
        print(f"  Edge types: {graph.edge_type_counts()}")

        self._state["entity_map"] = entity_map
        self._state["graph"] = graph
        return graph, entity_map

    # ------------------------------------------------------------------
    # Step 4: Reranking (PPR baseline)
    # ------------------------------------------------------------------

    def step_rerank(
        self,
        samples: List[Dict],
        hybrid: HybridRetriever,
        graph,
        corpus_chunks: List[Chunk],
    ) -> List[Dict]:
        print("[rerank] Running PPR reranking...")
        extractor = EntityExtractor()
        top_k_retrieval = self.cfg.retrieval["top_k"]

        results: List[Dict] = []
        for i, s in enumerate(samples):
            if i % 20 == 0:
                print(f"  {i}/{len(samples)}")
            question = s["question"]

            hybrid_results = hybrid.search(question, top_k=top_k_retrieval)
            candidate_ids = [c.chunk_id for c, _ in hybrid_results]

            # Construct retrieval_scores for PPR fusion
            retrieval_scores = {c.chunk_id: float(score) for c, score in hybrid_results}

            q_metrics = extractor.extract_metrics(question)
            q_years = extractor.extract_years(question)

            ppr_scores = ppr_rerank(
                graph,
                corpus_chunks,
                candidate_ids,
                seed_chunk_ids=candidate_ids[:10],
                seed_metric_names=list(q_metrics),
                seed_year_values=list(q_years),
                alpha=self.cfg.rerank["ppr_alpha"],
                retrieval_scores=retrieval_scores,
                retrieval_weight=self.cfg.rerank.get("ppr_retrieval_weight", 0.5),
            )
            ppr_sorted = sorted(ppr_scores, key=lambda x: x[1], reverse=True)

            results.append(
                {
                    "question_id": s["id"],
                    "question": question,
                    "gold_answer": s["answer"],
                    "gold_evidence_ids": self._state.get("gold_map", {}).get(s["id"], []),
                    "hybrid_chunk_ids": candidate_ids,
                    "ppr_chunk_ids": [cid for cid, _ in ppr_sorted],
                }
            )

        print(f"[rerank] Done. {len(results)} queries reranked.")
        self._state["rerank_results"] = results
        return results

    # ------------------------------------------------------------------
    # Step 5: Answer generation
    # ------------------------------------------------------------------

    def step_generation(
        self,
        samples: List[Dict],
        rerank_results: List[Dict],
        corpus_chunks: List[Chunk],
        method: str = "ppr",  # "ppr" | "hybrid"
    ) -> List[Dict]:
        print(f"[generation] Generating answers with method={method}...")
        generator = LLMGenerator(
            model=self.cfg.generation["model"],
            temperature=self.cfg.generation["temperature"],
            max_tokens=self.cfg.generation["max_tokens"],
        )
        verifier = NumericalVerifier()
        top_k = self.cfg.generation["top_k_evidence"]

        gen_results: List[Dict] = []
        for i, (s, rr) in enumerate(zip(samples, rerank_results)):
            if i % 20 == 0:
                print(f"  {i}/{len(samples)}")

            # Pick chunk IDs based on method
            if method == "ppr":
                ranked_ids = rr.get("ppr_chunk_ids", rr.get("hybrid_chunk_ids", []))
            else:
                ranked_ids = rr.get("hybrid_chunk_ids", [])

            top_ids = ranked_ids[:top_k]

            # Resolve IDs to Chunks
            chunk_lookup = {c.chunk_id: c for c in corpus_chunks}
            top_chunks = [chunk_lookup[cid] for cid in top_ids if cid in chunk_lookup]

            gen = generator.generate(s["question"], top_chunks)
            vres = verifier.verify(gen, s["question"])

            import re
            pred = re.sub(r"\s+", " ", gen.answer.lower().strip().rstrip("."))
            gold_norm = re.sub(r"\s+", " ", s["answer"].lower().strip().rstrip("."))

            gen_results.append(
                {
                    "question_id": s["id"],
                    "question": s["question"],
                    "gold_answer": s["answer"],
                    "generated_answer": gen.answer,
                    "gold_evidence_ids": self._state.get("gold_map", {}).get(s["id"], []),
                    "retrieved_chunk_ids": ranked_ids,
                    "cited_chunk_ids": gen.cited_chunk_ids,
                    "answer_is_correct": pred == gold_norm,
                    "is_consistent": vres.is_consistent,
                    "is_hallucination": (
                        not vres.evidence_fully_cited
                        and "INSUFFICIENT_EVIDENCE" not in gen.answer.upper()
                    ),
                }
            )

        print(f"[generation] Done. {len(gen_results)} answers generated.")
        self._state["gen_results"] = gen_results
        return gen_results

    # ------------------------------------------------------------------
    # Step 6: Evaluation
    # ------------------------------------------------------------------

    def step_evaluation(self, gen_results: List[Dict], method_label: str) -> Dict:
        print(f"[evaluation] Computing metrics for {method_label}...")
        k_vals = self.cfg.evaluation["recall_k_values"]
        er = compute_all_metrics(method_label, gen_results, k_values=k_vals)

        # Error analysis
        analyzer = ErrorAnalyzer()
        error_cases, error_counts = analyzer.analyze(gen_results)
        report = analyzer.report(error_cases, error_counts)
        print(report)

        # Summary dict
        summary = {
            "method": method_label,
            "num_samples": er.num_samples,
            "answer_accuracy": er.answer_accuracy,
            "exact_match": er.exact_match,
            "f1": er.f1,
            "mrr": er.mrr,
            "evidence_recall": er.evidence_recall,
            "evidence_precision": er.evidence_precision,
            "ndcg": er.ndcg,
            "numerical_consistency": er.numerical_consistency,
            "hallucination_rate": er.hallucination_rate,
            "insufficient_evidence_rate": er.insufficient_evidence_rate,
            "error_type_counts": error_counts,
            "timestamp": datetime.now().isoformat(),
        }

        # Save
        output_path = (
            self.cfg.output_dir / f"eval_{method_label}_{_timestamp()}.json"
        )
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, default=str)
        print(f"[evaluation] Results saved to {output_path}")

        self._state["eval_summary"] = summary
        return summary

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _save_data_cache(
        self,
        samples: List[Dict],
        chunks: List[Chunk],
        gold_map: Dict[str, List[str]],
    ) -> None:
        """Save only metadata; chunk texts are too large for JSON."""
        cache = {
            "num_samples": len(samples),
            "num_chunks": len(chunks),
            "sample_ids": [s["id"] for s in samples],
            "gold_map": gold_map,
        }
        with open(self.cfg.cache_dir / "data_state.json", "w", encoding="utf-8") as fh:
            json.dump(cache, fh, indent=2)

    def _load_data_cache(self) -> Tuple[List, List, Dict]:
        """Data cache restore is intentionally disabled."""
        return self.step_data()


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="FEG-RAG end-to-end pipeline")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--steps",
        nargs="+",
        default=PIPELINE_STEPS,
        help=f"Pipeline steps to run (default: all). Available: {PIPELINE_STEPS}",
    )
    parser.add_argument(
        "--skip",
        nargs="+",
        default=[],
        help="Steps to skip.",
    )
    parser.add_argument(
        "--rerank_method",
        default="ppr",
        choices=["hybrid", "ppr"],
        help="Which reranking method to use for answer generation.",
    )
    parser.add_argument("--num_samples", type=int, default=0,
                        help="Limit QA samples (0=all).")
    parser.add_argument("--split", default=None,
                        help="Dataset split to load, e.g. train/dev/test.")
    args = parser.parse_args()

    # Determine which steps to run
    steps_to_run = [s for s in args.steps if s not in args.skip]
    print(f"Pipeline steps: {steps_to_run}")

    cfg = Config.from_yaml(args.config)
    pipeline = Pipeline(cfg)

    # ---- Run ----
    samples, corpus_chunks, gold_map = None, None, None
    bm25, dense, hybrid = None, None, None
    graph, entity_map = None, None
    rerank_results = None
    gen_results = None

    if "data" in steps_to_run:
        samples, corpus_chunks, gold_map = pipeline.step_data(
            split=args.split, num_samples=args.num_samples
        )
    else:
        raise RuntimeError("Data step is required (must be first).")

    if "retrieval" in steps_to_run:
        bm25, dense, hybrid = pipeline.step_retrieval(corpus_chunks)

    if "graph" in steps_to_run:
        graph, entity_map = pipeline.step_graph(corpus_chunks)

    if "rerank" in steps_to_run:
        if hybrid is None or graph is None:
            raise RuntimeError("Rerank step requires retrieval + graph steps.")
        rerank_results = pipeline.step_rerank(samples, hybrid, graph, corpus_chunks)

    if "generation" in steps_to_run:
        if samples is None or corpus_chunks is None:
            raise RuntimeError("Generation step requires data step.")
        # Use rerank results if available, otherwise fall back to retrieval-only
        if rerank_results is None:
            # Build minimal rerank_results from hybrid retrieval
            if hybrid is None:
                raise RuntimeError("Generation needs retrieval or rerank results.")
            rerank_results = []
            for s in samples:
                hr = hybrid.search(s["question"], top_k=cfg.retrieval["top_k"])
                rerank_results.append(
                    {
                        "question_id": s["id"],
                        "question": s["question"],
                        "gold_answer": s["answer"],
                        "gold_evidence_ids": gold_map.get(s["id"], []),
                        "hybrid_chunk_ids": [c.chunk_id for c, _ in hr],
                        "ppr_chunk_ids": [c.chunk_id for c, _ in hr],
                    }
                )
            gen_results = pipeline.step_generation(
                samples, rerank_results, corpus_chunks, method="hybrid"
            )
        else:
            gen_results = pipeline.step_generation(
                samples, rerank_results, corpus_chunks, method=args.rerank_method
            )

    if "evaluation" in steps_to_run:
        if gen_results is None:
            raise RuntimeError("Evaluation step requires generation results.")
        summary = pipeline.step_evaluation(gen_results, args.rerank_method)

        # Print final summary
        print("\n" + "=" * 60)
        print("FINAL RESULTS")
        print("=" * 60)
        print(f"Method:          {summary['method']}")
        print(f"Samples:         {summary['num_samples']}")
        print(f"Answer Accuracy: {summary['answer_accuracy']:.4f}")
        print(f"Exact Match:     {summary['exact_match']:.4f}")
        print(f"F1:              {summary['f1']:.4f}")
        print(f"MRR:             {summary['mrr']:.4f}")
        for k, v in summary["evidence_recall"].items():
            print(f"Evidence R@{k}:   {v:.4f}")
        print(f"Num Consistency: {summary['numerical_consistency']:.4f}")
        print(f"Hallucination:   {summary['hallucination_rate']:.4f}")

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
