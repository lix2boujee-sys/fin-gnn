"""Experiment 1: Plain retrieval baseline on FinDER (no graph).

Implements BM25, Dense, and Hybrid retrieval per finder_exp1_baseline_instruction.md.
Writes all artifacts to outputs/exp1_baseline/.

Usage:
    python experiments/exp1_retrieval_baseline.py
    python experiments/exp1_retrieval_baseline.py --num_samples 200 --top_k 10
    python experiments/exp1_retrieval_baseline.py --num_samples 0 --alpha 0.5
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from feg_rag.config import Config
from feg_rag.data.chunker import Chunk, chunk_text, chunk_report
from feg_rag.data.loader import load_dataset
from feg_rag.evaluation.metrics import compute_all_metrics
from feg_rag.graph.entities import EntityExtractor
from feg_rag.retrieval.bm25 import BM25Retriever
from feg_rag.retrieval.dense import DenseRetriever
from feg_rag.retrieval.hybrid import HybridRetriever


METHOD_FILES = {
    "BM25": "bm25_results.jsonl",
    "Dense Retrieval": "dense_results.jsonl",
    "Hybrid Retrieval": "hybrid_results.jsonl",
}


def build_corpus(
    samples: List[Dict],
    cfg: Config,
    max_distractor_files: int = 50,
) -> Tuple[List[Chunk], Dict[str, List[str]]]:
    """Chunk FinDER gold evidence; optionally add 10-K distractors."""
    corpus_chunks: List[Chunk] = []
    gold_map: Dict[str, List[str]] = {}

    for s in samples:
        gold_ids: List[str] = []
        for text in s["evidence_texts"]:
            for c in chunk_text(
                text,
                cfg.chunk_size,
                cfg.chunk_overlap,
                doc_id=s["id"],
            ):
                corpus_chunks.append(c)
                gold_ids.append(c.chunk_id)
        gold_map[s["id"]] = gold_ids

    edgar_dir = cfg.edgar_dir
    if edgar_dir.exists():
        files = list(edgar_dir.rglob("*.txt")) or list(edgar_dir.rglob("*.html"))
        for tf in files[:max_distractor_files]:
            try:
                corpus_chunks.extend(
                    chunk_report(tf, cfg.chunk_size, cfg.chunk_overlap)
                )
            except Exception:
                pass

    return corpus_chunks, gold_map


def _hit_at_k(retrieved_ids: List[str], gold_ids: List[str], k: int) -> bool:
    gold = set(gold_ids)
    return bool(gold & set(retrieved_ids[:k]))


def _reciprocal_rank(retrieved_ids: List[str], gold_ids: List[str]) -> float:
    gold = set(gold_ids)
    for rank, cid in enumerate(retrieved_ids, start=1):
        if cid in gold:
            return 1.0 / rank
    return 0.0


def _extract_metadata(chunk: Chunk, extractor: EntityExtractor) -> Dict[str, str]:
    text = chunk.text
    metrics = extractor.extract_metrics(text)
    years = extractor.extract_years(text)
    return {
        "company": chunk.doc_id or "",
        "year": ", ".join(sorted(years)) if years else "",
        "metric": ", ".join(sorted(metrics)) if metrics else "",
        "filing": chunk.section or "",
    }


def run_method(
    method_name: str,
    samples: List[Dict],
    retriever,
    gold_map: Dict[str, List[str]],
    chunk_lookup: Dict[str, Chunk],
    top_k: int,
    output_k: int,
    extractor: EntityExtractor,
) -> Tuple[List[Dict], List[Dict]]:
    """Run one retriever and build per-query JSONL records + metric inputs."""
    records: List[Dict] = []
    metric_inputs: List[Dict] = []

    for s in samples:
        qid = s["id"]
        question = s["question"]
        gold_ids = gold_map.get(qid, [])

        results = retriever.search(question, top_k=top_k)
        retrieved_ids = [c.chunk_id for c, _ in results]

        top_k_payload = []
        for rank, (chunk, score) in enumerate(results[:output_k], start=1):
            meta = _extract_metadata(chunk, extractor)
            top_k_payload.append(
                {
                    "rank": rank,
                    "passage_id": chunk.chunk_id,
                    "score": float(score),
                    "text": chunk.text[:500],
                    "is_gold": chunk.chunk_id in set(gold_ids),
                    **meta,
                }
            )

        rr = _reciprocal_rank(retrieved_ids, gold_ids)
        record = {
            "query_id": qid,
            "question": question,
            "method": method_name,
            "top_k": top_k_payload,
            "gold_evidence_ids": gold_ids,
            "hit_at_5": _hit_at_k(retrieved_ids, gold_ids, 5),
            "hit_at_10": _hit_at_k(retrieved_ids, gold_ids, 10),
            "rr": rr,
        }
        records.append(record)

        metric_inputs.append(
            {
                "question_id": qid,
                "question": question,
                "gold_answer": s.get("answer", ""),
                "gold_evidence_ids": gold_ids,
                "retrieved_chunk_ids": retrieved_ids,
            }
        )

    return records, metric_inputs


def write_jsonl(path: Path, records: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def write_metrics_csv(path: Path, summaries: Dict[str, Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for method, m in summaries.items():
        rows.append(
            {
                "Method": method,
                "Recall@5": m["evidence_recall"].get(5, 0),
                "Recall@10": m["evidence_recall"].get(10, 0),
                "MRR": m["mrr"],
                "nDCG@10": m["ndcg"].get(10, 0),
                "Hit@5": m["hit_at_5"],
                "Hit@10": m["hit_at_10"],
                "num_samples": m["num_samples"],
            }
        )

    fieldnames = [
        "Method", "Recall@5", "Recall@10", "MRR", "nDCG@10",
        "Hit@5", "Hit@10", "num_samples",
    ]
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def collect_failed_cases(
    all_records: Dict[str, List[Dict]],
    primary_method: str = "Hybrid Retrieval",
) -> List[Dict]:
    """Collect queries where primary method misses gold evidence in top-10."""
    failed: List[Dict] = []
    primary = all_records.get(primary_method, [])
    for rec in primary:
        if not rec.get("hit_at_10", False):
            failed.append(
                {
                    "query_id": rec["query_id"],
                    "question": rec["question"],
                    "method": primary_method,
                    "gold_evidence_ids": rec["gold_evidence_ids"],
                    "top_k_results": rec["top_k"],
                    "hit_at_5": rec["hit_at_5"],
                    "hit_at_10": rec["hit_at_10"],
                    "possible_error_type": "to_be_annotated",
                }
            )
    return failed


def write_readme(
    path: Path,
    cfg: Config,
    summaries: Dict[str, Dict],
    num_samples: int,
    alpha: float,
    top_k: int,
    dense_backend: str = "sentence-transformers",
    dense_device: str = "cpu",
    dense_batch_size: str = "auto",
    command: str = "",
) -> None:
    best = max(
        summaries.items(),
        key=lambda x: x[1]["evidence_recall"].get(10, 0),
    )
    lines = [
        "# Experiment 1: Plain Retrieval Baseline",
        "",
        "FinDER evidence retrieval without graph structure.",
        "",
        "## Run command",
        "",
        "```bash",
        f"{command or '# (see train_config.yaml for full command)'}",
        "```",
        "",
        "## Data mapping (FinDER → code)",
        "",
        "| FinDER field | Code field |",
        "|---|---|",
        "| `_id` | `query_id` / sample `id` |",
        "| `text` | `question` |",
        "| `references` | `evidence_texts` → chunked as candidate passages |",
        "| `answer` | `answer` (stored but not used in Exp1 metrics) |",
        "",
        "## Methods",
        "",
        "- **BM25**: lexical keyword retrieval",
        "- **Dense Retrieval**: sentence-transformers + FAISS",
        "- **Hybrid Retrieval**: normalized BM25 + dense fusion",
        "",
        f"- Dense model: `{cfg.retrieval['dense_model']}`",
        f"- Dense backend used: `{dense_backend}`",
        f"- Dense device: `{dense_device}`",
        f"- Dense batch size: `{dense_batch_size}`",
        f"- Hybrid alpha (BM25 weight): `{alpha}`",
        f"- Retrieval top_k: `{top_k}`",
        f"- Samples evaluated: `{num_samples}`",
        "",
        "## Output files",
        "",
        "| File | Description |",
        "|---|---|",
        "| `bm25_results.jsonl` | Per-query BM25 top-k results |",
        "| `dense_results.jsonl` | Per-query dense top-k results |",
        "| `hybrid_results.jsonl` | Per-query hybrid top-k results |",
        "| `metrics_summary.csv` | Aggregate Recall/MRR/nDCG/Hit |",
        "| `failed_cases.jsonl` | Hybrid misses at top-10 (for Exp2 error analysis) |",
        "| `metrics_full.json` | Full metric dict for all methods |",
        "",
        "## Results summary",
        "",
        "| Method | Recall@5 | Recall@10 | MRR | nDCG@10 | Hit@5 | Hit@10 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method, m in summaries.items():
        lines.append(
            f"| {method} | {m['evidence_recall'].get(5, 0):.4f} | "
            f"{m['evidence_recall'].get(10, 0):.4f} | {m['mrr']:.4f} | "
            f"{m['ndcg'].get(10, 0):.4f} | {m['hit_at_5']:.4f} | "
            f"{m['hit_at_10']:.4f} |"
        )
    lines.extend(
        [
            "",
            f"**Strongest baseline on Recall@10**: {best[0]} "
            f"({best[1]['evidence_recall'].get(10, 0):.4f})",
            "",
            "## Conclusion (Exp1 question)",
            "",
            "> Without graph structure, plain retrieval on FinDER achieves the levels above.",
            "> Hybrid generally outperforms BM25 and Dense alone on this setup.",
            "> Use `failed_cases.jsonl` for Exp2 financial structure error analysis.",
            "",
            f"Generated: {datetime.now().isoformat()}",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Exp1: plain retrieval baseline")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output_dir", default="outputs/exp1_baseline")
    parser.add_argument("--num_samples", type=int, default=0, help="0 = all FinDER")
    parser.add_argument("--top_k", type=int, default=10, help="Saved top-k per query")
    parser.add_argument("--retrieve_k", type=int, default=None,
                        help="Internal retrieval depth (default: max(top_k, config top_k))")
    parser.add_argument("--alpha", type=float, default=None,
                        help="Hybrid BM25 weight (default: config hybrid_alpha)")
    parser.add_argument("--max_distractor_files", type=int, default=50)
    parser.add_argument("--skip_dense", action="store_true",
                        help="Skip dense/hybrid (BM25 only, for debugging)")
    parser.add_argument("--dense_device", default="cpu",
                        help="Device for Dense encoding (cpu avoids GPU OOM)")
    parser.add_argument("--dense_batch_size", type=int, default=None,
                        help="Batch size for dense encoding (default: auto; "
                             "use 1-4 for E5-Mistral on CPU)")
    parser.add_argument("--overwrite_output_dir", action="store_true",
                        help="Allow overwriting existing results")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    cfg.ensure_dirs()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = cfg.root_dir / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    alpha = args.alpha if args.alpha is not None else cfg.retrieval["hybrid_alpha"]
    retrieve_k = args.retrieve_k or max(args.top_k, cfg.retrieval["top_k"])
    k_values = [1, 3, 5, 10, 20]

    print("=" * 60)
    print("  EXP1: Plain Retrieval Baseline (no graph)")
    print("=" * 60)

    # 1. Load data
    print("\n[1/4] Loading FinDER...")
    samples = load_dataset("finder", cfg.data_dir)
    if args.num_samples > 0:
        samples = samples[: args.num_samples]
    print(f"  {len(samples)} QA samples")

    # 2. Corpus
    print("\n[2/4] Building corpus...")
    corpus_chunks, gold_map = build_corpus(
        samples, cfg, max_distractor_files=args.max_distractor_files
    )
    chunk_lookup = {c.chunk_id: c for c in corpus_chunks}
    print(f"  {len(corpus_chunks)} chunks")

    # 3. Indices
    print("\n[3/4] Building indices...")
    t0 = time.time()
    bm25 = BM25Retriever(k1=cfg.retrieval["bm25_k1"], b=cfg.retrieval["bm25_b"])
    bm25.index(corpus_chunks)
    print(f"  BM25 indexed in {time.time() - t0:.1f}s")

    dense = hybrid = None
    dense_backend = "skipped"
    if not args.skip_dense:
        t0 = time.time()
        try:
            dense = DenseRetriever(
                model_name=cfg.retrieval["dense_model"],
                device=args.dense_device,
                query_instruction=cfg.retrieval.get("dense_query_instruction"),
                e5_max_seq_length=cfg.retrieval.get("e5_max_seq_length", 512),
                e5_batch_size=cfg.retrieval.get("e5_batch_size"),
                debug=cfg.retrieval.get("debug_dense", False),
            )
            dense.index(corpus_chunks, batch_size=args.dense_batch_size)
            dense_backend = getattr(dense, "backend", "sentence-transformers")
            print(f"  Dense ({dense_backend}) indexed in {time.time() - t0:.1f}s")
        except Exception as exc:
            print(f"  [WARN] Dense model load failed ({exc})")
            print("  Falling back to TF-IDF dense retriever (offline)...")
            from feg_rag.retrieval.tfidf_dense import TfidfDenseRetriever

            dense_backend = "tfidf-offline"
            t0 = time.time()
            dense = TfidfDenseRetriever()
            dense.index(corpus_chunks)
            print(f"  Dense (TF-IDF fallback) indexed in {time.time() - t0:.1f}s")
        hybrid = HybridRetriever(bm25, dense, alpha=alpha)

    # 4. Run methods
    print("\n[4/4] Running retrieval...")
    extractor = EntityExtractor()
    all_records: Dict[str, List[Dict]] = {}
    summaries: Dict[str, Dict] = {}

    runners = [("BM25", bm25)]
    if dense is not None:
        runners.extend([("Dense Retrieval", dense), ("Hybrid Retrieval", hybrid)])

    for method_name, retriever in runners:
        print(f"  -> {method_name}")
        t0 = time.time()
        records, metric_inputs = run_method(
            method_name,
            samples,
            retriever,
            gold_map,
            chunk_lookup,
            top_k=retrieve_k,
            output_k=args.top_k,
            extractor=extractor,
        )
        er = compute_all_metrics(method_name, metric_inputs, k_values=k_values)
        hit5 = float(np.mean([r["hit_at_5"] for r in records]))
        hit10 = float(np.mean([r["hit_at_10"] for r in records]))

        summaries[method_name] = {
            "method": method_name,
            "num_samples": er.num_samples,
            "evidence_recall": er.evidence_recall,
            "evidence_precision": er.evidence_precision,
            "mrr": er.mrr,
            "ndcg": er.ndcg,
            "hit_at_5": hit5,
            "hit_at_10": hit10,
        }
        all_records[method_name] = records

        out_file = output_dir / METHOD_FILES[method_name]
        write_jsonl(out_file, records)
        print(f"     {len(records)} queries -> {out_file.name} ({time.time() - t0:.1f}s)")

    # Failed cases (Hybrid primary)
    failed = collect_failed_cases(all_records)
    write_jsonl(output_dir / "failed_cases.jsonl", failed)

    # CSV + JSON summary
    write_metrics_csv(output_dir / "metrics_summary.csv", summaries)
    with open(output_dir / "metrics_full.json", "w", encoding="utf-8") as fh:
        json.dump(summaries, fh, indent=2)

    write_readme(
        output_dir / "README.md",
        cfg,
        summaries,
        len(samples),
        alpha,
        args.top_k,
        dense_backend=dense_backend if dense is not None else "skipped",
        dense_device=args.dense_device,
        dense_batch_size=str(args.dense_batch_size or "auto"),
        command=" ".join(sys.argv),
    )

    # Print table
    print("\n" + "=" * 60)
    print("  METRICS SUMMARY")
    print("=" * 60)
    print(f"{'Method':<22} {'R@5':>7} {'R@10':>7} {'MRR':>7} {'nDCG@10':>8} {'Hit@10':>8}")
    print("-" * 60)
    for method, m in summaries.items():
        print(
            f"{method:<22} "
            f"{m['evidence_recall'].get(5, 0):>7.4f} "
            f"{m['evidence_recall'].get(10, 0):>7.4f} "
            f"{m['mrr']:>7.4f} "
            f"{m['ndcg'].get(10, 0):>8.4f} "
            f"{m['hit_at_10']:>8.4f}"
        )
    print(f"\nFailed cases (Hybrid miss@10): {len(failed)}")
    print(f"Output directory: {output_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
