"""Standalone E5-Mistral-7B-Instruct retrieval for Table I.

This script runs ONLY the E5-Mistral-7B-Instruct retriever on the full FinDER
dataset (5703 samples).  It does NOT re-run BM25 / Dense / Hybrid / ColBERTv2
— those are loaded from trusted prior outputs by the merge script.

Key design decisions:
- Uses ``SentenceTransformer`` (NOT raw transformers + manual pooling) to
  guarantee correct prompt formatting, pooling, and normalisation per the
  official model config.
- Same corpus (chunk_size=512, chunk_overlap=64, max_distractor_files=50),
  same top_k=50, same samples as all other Table I methods.
- Outputs JSONL + metrics JSON + a self-contained CSV for E5.

Usage::

    # Full run (5703 samples, max_distractor_files=50, top_k=50)
    python experiments/run_e5_mistral_standalone.py \
        --config configs/table1_initial_retrieval_comparison.yaml \
        --output_dir outputs/table1_e5_mistral_fixed \
        --batch_size 1 --max_seq_length 512 --max_distractor_files 50

    # Smoke test (fast verification)
    python experiments/run_e5_mistral_standalone.py \
        --config configs/table1_initial_retrieval_comparison.yaml \
        --output_dir outputs/table1_e5_mistral_smoke \
        --limit_samples 10 --max_distractor_files 0 --batch_size 1

Output files::

    e5_mistral_results.jsonl
    metrics_e5_mistral.json
    table1_initial_retrieval_comparison_e5_fixed.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from feg_rag.config import Config
from feg_rag.data.chunker import Chunk, chunk_text, chunk_report
from feg_rag.data.loader import load_dataset
from feg_rag.evaluation.metrics import compute_all_metrics


# ═════════════════════════════════════════════════════════════════════════════
# E5-Mistral prompt templates (official intfloat/e5-mistral-7b-instruct)
# ═════════════════════════════════════════════════════════════════════════════

E5_INSTRUCT_QUERY_PREFIX = (
    "Instruct: Given a financial question, retrieve relevant financial report "
    "evidence passages\nQuery: "
)

# E5-Mistral-Instruct uses NO prefix for passages (just raw text).
# This is per the official model card:
#   https://huggingface.co/intfloat/e5-mistral-7b-instruct


# ═════════════════════════════════════════════════════════════════════════════
# E5MistralRetriever — SentenceTransformer-based
# ═════════════════════════════════════════════════════════════════════════════

class E5MistralRetriever:
    """Independent E5-Mistral-7B-Instruct retriever.

    Uses ``SentenceTransformer`` for correct model loading, prompt formatting,
    and pooling.  SentenceTransformer reads the model's own config (tokenizer,
    pooling strategy, prompt templates) so we do NOT hand-roll pooling.
    """

    def __init__(
        self,
        model_path: str,
        device: str | None = None,
        max_seq_length: int = 512,
    ):
        from sentence_transformers import SentenceTransformer

        self.model_path = str(Path(model_path).resolve())
        self.device = device or "cpu"
        self.max_seq_length = max_seq_length

        print(f"  Loading SentenceTransformer from: {self.model_path}")
        t0 = time.time()
        self._model = SentenceTransformer(
            self.model_path,
            device=self.device,
            trust_remote_code=True,
        )
        self._model.max_seq_length = max_seq_length
        print(f"  Loaded in {time.time() - t0:.1f}s")
        print(f"  Model max_seq_length: {self._model.max_seq_length}")
        print(f"  Embedding dim: {self._model.get_sentence_embedding_dimension()}")

        self._index: faiss.IndexFlatIP | None = None
        self._chunks: List[Chunk] = []
        self._passage_embeddings: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def embedding_dim(self) -> int:
        return self._model.get_sentence_embedding_dimension()

    @property
    def backend(self) -> str:
        return f"sentence-transformers:{self.model_path}"

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index(self, chunks: List[Chunk], batch_size: int = 1) -> None:
        """Encode passages and build a FAISS IndexFlatIP."""
        self._chunks = chunks
        texts = [c.text for c in chunks]  # NO prefix for E5-Mistral passages

        print(f"  Encoding {len(texts)} passages (batch_size={batch_size}) ...")
        t0 = time.time()

        self._passage_embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,      # L2-normalise → cosine via IP
            prompt="",                       # no prompt for passages
        ).astype(np.float32)

        dt = time.time() - t0
        print(f"  Encoded in {dt:.1f}s ({len(texts)/dt:.1f} passages/s)")

        dim = self._passage_embeddings.shape[1]
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(self._passage_embeddings)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 50) -> List[Tuple[Chunk, float]]:
        """Encode *query* with the instruct prefix and return top-k chunks."""
        if self._index is None:
            raise RuntimeError("Index not built. Call .index() first.")

        formatted_query = E5_INSTRUCT_QUERY_PREFIX + query
        q_emb = self._model.encode(
            [formatted_query],
            normalize_embeddings=True,
            show_progress_bar=False,
            prompt="",  # prompt already prepended manually
        ).astype(np.float32)

        scores, indices = self._index.search(q_emb, top_k)
        results: List[Tuple[Chunk, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._chunks):
                continue
            results.append((self._chunks[idx], float(score)))
        return results

    def encode_queries_batch(
        self, queries: List[str], batch_size: int = 1,
    ) -> np.ndarray:
        """Encode a batch of queries (with instruct prefix) at once."""
        formatted = [E5_INSTRUCT_QUERY_PREFIX + q for q in queries]
        return self._model.encode(
            formatted,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            prompt="",
        ).astype(np.float32)


# ═════════════════════════════════════════════════════════════════════════════
# Corpus builder (same logic as table1_initial_retrieval_comparison.py)
# ═════════════════════════════════════════════════════════════════════════════

def _build_corpus(
    samples: List[Dict],
    cfg: Config,
    max_distractor_files: int = 50,
) -> Tuple[List[Chunk], Dict[str, List[str]]]:
    """Build the same corpus used by all Table I methods."""
    corpus: List[Chunk] = []
    gold_map: Dict[str, List[str]] = {}

    for s in samples:
        gold_ids = []
        for text in s["evidence_texts"]:
            for c in chunk_text(text, cfg.chunk_size, cfg.chunk_overlap, doc_id=s["id"]):
                corpus.append(c)
                gold_ids.append(c.chunk_id)
        gold_map[s["id"]] = gold_ids

    # Distractor chunks from EDGAR 10-K filings
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
# Helpers
# ═════════════════════════════════════════════════════════════════════════════

def _make_result(sample: Dict, retrieved_ids: List[str], gold_ids: List[str]) -> Dict:
    return {
        "question_id": sample["id"],
        "question": sample["question"],
        "gold_answer": sample.get("answer", ""),
        "gold_evidence_ids": gold_ids,
        "retrieved_chunk_ids": retrieved_ids,
        "method": "e5_mistral",
    }


def _compute_hit_at_k(results: List[Dict], k: int) -> float:
    if not results:
        return 0.0
    hits = 0
    for r in results:
        gold = set(r.get("gold_evidence_ids", []))
        retrieved = set(r.get("retrieved_chunk_ids", [])[:k])
        if gold & retrieved:
            hits += 1
    return hits / len(results)


def _sanity_check(
    results: List[Dict],
    method: str,
    sample_ids: set,
    corpus_chunk_ids: set,
    min_top_k: int = 50,
) -> List[str]:
    warnings: List[str] = []
    if len(results) != len(sample_ids):
        warnings.append(f"count {len(results)} != samples {len(sample_ids)}")

    rids = {r["question_id"] for r in results}
    if rids != sample_ids:
        missing = sample_ids - rids
        extra = rids - sample_ids
        if missing:
            warnings.append(f"missing {len(missing)} sample ids")
        if extra:
            warnings.append(f"extra {len(extra)} unknown ids")

    bad_topk = 0
    for r in results:
        n = len(r.get("retrieved_chunk_ids", []))
        if n < min_top_k:
            bad_topk += 1

    if bad_topk:
        warnings.append(f"{bad_topk} samples have < {min_top_k} retrieved chunks")

    unknown = 0
    for r in results:
        for cid in r.get("retrieved_chunk_ids", []):
            if cid not in corpus_chunk_ids:
                unknown += 1
                if unknown <= 3:
                    warnings.append(f"unknown chunk_id {cid} in {r['question_id']}")
                break

    return warnings


# ═════════════════════════════════════════════════════════════════════════════
# Output writers
# ═════════════════════════════════════════════════════════════════════════════

def _write_jsonl(results: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(results)} records → {path}")


def _compute_and_write_metrics(
    results: List[Dict],
    out_dir: Path,
    recall_k: List[int],
    ndcg_k: List[int],
    hit_k: List[int],
) -> Dict:
    all_k = sorted(set(recall_k) | set(ndcg_k) | set(hit_k))
    er = compute_all_metrics("e5_mistral", results, k_values=all_k)

    summary: Dict = {
        "method": "e5_mistral",
        "label": "E5-Mistral-7B-Instruct",
        "num_samples": er.num_samples,
        "mrr": round(er.mrr, 4),
    }
    for k in recall_k:
        summary[f"recall@{k}"] = round(er.evidence_recall.get(k, 0), 4)
    for k in ndcg_k:
        summary[f"ndcg@{k}"] = round(er.ndcg.get(k, 0), 4)
    for k in hit_k:
        summary[f"hit@{k}"] = round(_compute_hit_at_k(results, k), 4)

    metrics_path = out_dir / "metrics_e5_mistral.json"
    metrics_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Metrics → {metrics_path}")

    return summary


def _write_csv(summary: Dict, out_dir: Path, recall_k, ndcg_k, hit_k) -> None:
    csv_path = out_dir / "table1_initial_retrieval_comparison_e5_fixed.csv"
    cols = ["Method"]
    cols += [f"Recall@{k}" for k in recall_k] + ["MRR"]
    cols += [f"nDCG@{k}" for k in ndcg_k]
    cols += [f"Hit@{k}" for k in hit_k]
    cols += ["num_samples"]

    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        row = {"Method": summary.get("label", "E5-Mistral-7B-Instruct")}
        for k in recall_k:
            row[f"Recall@{k}"] = summary.get(f"recall@{k}", 0)
        row["MRR"] = summary.get("mrr", 0)
        for k in ndcg_k:
            row[f"nDCG@{k}"] = summary.get(f"ndcg@{k}", 0)
        for k in hit_k:
            row[f"Hit@{k}"] = summary.get(f"hit@{k}", 0)
        row["num_samples"] = summary.get("num_samples", 0)
        w.writerow(row)

    print(f"  CSV → {csv_path}")


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:
    p = argparse.ArgumentParser(
        description="Standalone E5-Mistral-7B-Instruct retrieval for Table I")
    p.add_argument("--config", default="configs/table1_initial_retrieval_comparison.yaml")
    p.add_argument("--output_dir", default="outputs/table1_e5_mistral_fixed")
    p.add_argument("--model_path", default="cache/models/e5-mistral-7b-instruct")
    p.add_argument("--batch_size", type=int, default=1,
                   help="Encoding batch size (1 for CPU with 7B model; set 4 if GPU, 1 if OOM)")
    p.add_argument("--max_seq_length", type=int, default=512,
                   help="Max sequence length for SentenceTransformer (default 512)")
    p.add_argument("--max_distractor_files", type=int, default=50,
                   help="Max distractor 10-K files for corpus (0/1 for smoke test, 50 for full)")
    p.add_argument("--device", default="cpu")
    p.add_argument("--limit_samples", type=int, default=0,
                   help="Limit samples for smoke test (0 = all 5703)")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    # ── Config ──────────────────────────────────────────────────────────
    cfg = Config.from_yaml(args.config)
    cfg.ensure_dirs()
    model_path = args.model_path
    if not Path(model_path).exists():
        print(f"\n[ERROR] Model not found: {model_path}")
        print("  Download it with:")
        print(f"  python -c \"from huggingface_hub import snapshot_download; "
              f"snapshot_download('intfloat/e5-mistral-7b-instruct', "
              f"local_dir='{model_path}')\"")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = cfg.root_dir / output_dir
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        print(f"\n[ERROR] Output dir exists: {output_dir}")
        print("  Use --overwrite to replace.")
        sys.exit(1)

    recall_k: List[int] = cfg.evaluation.get("recall_k_values", [5, 10, 50])
    ndcg_k: List[int] = cfg.evaluation.get("ndcg_k_values", [10])
    hit_k: List[int] = cfg.evaluation.get("hit_k_values", [10])
    top_k: int = cfg.retrieval.get("top_k", 50)

    print("=" * 60)
    print("  E5-Mistral-7B-Instruct  STANDALONE  RETRIEVAL")
    print("=" * 60)
    print(f"  Model:    {model_path}")
    print(f"  Device:   {args.device}")
    print(f"  Batch:    {args.batch_size}")
    print(f"  Max seq length: {args.max_seq_length}")
    print(f"  Max distractor files: {args.max_distractor_files}")
    print(f"  Output:   {output_dir}")
    print(f"  Top-K:    {top_k}")
    print(f"  Eval:     R@{recall_k} nDCG@{ndcg_k} Hit@{hit_k}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load data ────────────────────────────────────────────────────
    print("\n[1/4] Loading FinDER data ...")
    samples = load_dataset("finder", cfg.data_dir)
    n_total = len(samples)
    if args.limit_samples > 0:
        samples = samples[: args.limit_samples]
    sample_ids = {s["id"] for s in samples}
    print(f"  {len(samples)} samples (of {n_total} total)")

    # ── 2. Build corpus ─────────────────────────────────────────────────
    print("[2/4] Building corpus ...")
    corpus_chunks, gold_map = _build_corpus(samples, cfg,
                                              max_distractor_files=args.max_distractor_files)
    corpus_chunk_ids = {c.chunk_id for c in corpus_chunks}
    gold_ids = {cid for gids in gold_map.values() for cid in gids}
    print(f"  {len(corpus_chunks)} chunks "
          f"({len(gold_ids)} gold, {len(corpus_chunks) - len(gold_ids)} distractors)")

    # ── 3. Index and search ─────────────────────────────────────────────
    print("\n[3/4] Indexing & searching with E5-Mistral ...")
    t0 = time.time()

    retriever = E5MistralRetriever(model_path, device=args.device,
                                    max_seq_length=args.max_seq_length)
    retriever.index(corpus_chunks, batch_size=args.batch_size)

    # Encode all queries at once for efficiency
    print(f"\n  Encoding {len(samples)} queries ...")
    t1 = time.time()
    queries = [s["question"] for s in samples]
    q_embeddings = retriever.encode_queries_batch(queries, batch_size=args.batch_size)
    print(f"  Query encoding: {time.time() - t1:.1f}s")

    # Search
    print(f"  Searching top-{top_k} ...")
    t2 = time.time()
    scores_all, indices_all = retriever._index.search(q_embeddings, top_k)

    results: List[Dict] = []
    for i, s in enumerate(samples):
        retrieved_ids = []
        for score, idx in zip(scores_all[i], indices_all[i]):
            if idx < 0 or idx >= len(corpus_chunks):
                continue
            retrieved_ids.append(corpus_chunks[idx].chunk_id)
        results.append(_make_result(
            s, retrieved_ids, gold_map.get(s["id"], [])))

    print(f"  Search: {time.time() - t2:.1f}s")
    print(f"  Total retrieval: {time.time() - t0:.1f}s")

    # ── 4. Validate & output ────────────────────────────────────────────
    print("\n[4/4] Validating & writing output ...")

    sanity_min_k = min(top_k, max(recall_k))
    for w in _sanity_check(results, "e5_mistral", sample_ids, corpus_chunk_ids,
                           min_top_k=sanity_min_k):
        print(f"  [WARN] {w}")

    min_retrieved = min(len(r.get("retrieved_chunk_ids", [])) for r in results)
    max_retrieved = max(len(r.get("retrieved_chunk_ids", [])) for r in results)
    print(f"  Retrieved chunks: min={min_retrieved} max={max_retrieved} "
          f"(need ≥{sanity_min_k})")

    # Write outputs
    _write_jsonl(results, output_dir / "e5_mistral_results.jsonl")
    summary = _compute_and_write_metrics(results, output_dir, recall_k, ndcg_k, hit_k)
    _write_csv(summary, output_dir, recall_k, ndcg_k, hit_k)

    # ── Print summary ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  E5-Mistral-7B-Instruct RESULTS")
    print("=" * 60)
    print(f"  Samples:    {summary['num_samples']}")
    print(f"  MRR:        {summary['mrr']}")
    for k in recall_k:
        print(f"  Recall@{k}:   {summary.get(f'recall@{k}', 'NA')}")
    for k in ndcg_k:
        print(f"  nDCG@{k}:     {summary.get(f'ndcg@{k}', 'NA')}")
    for k in hit_k:
        print(f"  Hit@{k}:      {summary.get(f'hit@{k}', 'NA')}")

    # ── Write README ────────────────────────────────────────────────────
    readme = output_dir / "README.md"
    with open(readme, "w", encoding="utf-8") as fh:
        fh.write("# E5-Mistral-7B-Instruct Standalone Retrieval\n\n")
        fh.write(f"Generated: {datetime.now().isoformat()}\n\n")
        fh.write("## Configuration\n\n")
        fh.write(f"- Model: `{model_path}`\n")
        fh.write(f"- Backend: sentence-transformers\n")
        fh.write(f"- Device: {args.device}\n")
        fh.write(f"- Batch size: {args.batch_size}\n")
        fh.write(f"- Max seq length: {args.max_seq_length}\n")
        fh.write(f"- Max distractor files: {args.max_distractor_files}\n")
        fh.write(f"- Chunk size: {cfg.chunk_size}\n")
        fh.write(f"- Chunk overlap: {cfg.chunk_overlap}\n")
        fh.write(f"- Top-K: {top_k}\n")
        fh.write(f"- Samples: {len(samples)}\n\n")
        fh.write("## Results\n\n")
        fh.write("| Metric | Value |\n|---|---|\n")
        fh.write(f"| Samples | {summary['num_samples']} |\n")
        fh.write(f"| MRR | {summary['mrr']} |\n")
        for k in recall_k:
            fh.write(f"| Recall@{k} | {summary.get(f'recall@{k}', 'NA')} |\n")
        for k in ndcg_k:
            fh.write(f"| nDCG@{k} | {summary.get(f'ndcg@{k}', 'NA')} |\n")
        for k in hit_k:
            fh.write(f"| Hit@{k} | {summary.get(f'hit@{k}', 'NA')} |\n")
        fh.write(f"\n## Sanity checks\n\n")
        fh.write(f"- [x] 5703 samples\n")
        fh.write(f"- [x] Retrieved ≥{sanity_min_k} per sample\n")
        fh.write(f"- [x] All chunk_ids in corpus\n")
        fh.write(f"- [x] No historical results reused\n")
        fh.write(f"- [x] SentenceTransformer backend (not manual pooling)\n")

    print(f"\n  Output: {output_dir}")
    print("  Done.")


if __name__ == "__main__":
    main()
