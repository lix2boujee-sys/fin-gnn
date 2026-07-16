"""Unified evaluation script for BGE-M3-Dense and SPLADE-v3 standalone retrievers.

Both retrievers use the **same** samples, corpus, gold_map, and metrics,
ensuring results are directly comparable.

Usage:
    python experiments/run_bge_m3_splade_v3.py --config configs/default.yaml

Environment:
    HF_ENDPOINT=https://hf-mirror.com  (set if behind GFW)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from feg_rag.config import Config
from feg_rag.data.chunker import Chunk
from feg_rag.data.corpus import build_benchmark_corpus
from feg_rag.data.loader import load_dataset
from feg_rag.retrieval.bge_m3 import (
    BGEM3DenseRetriever,
    build_cache_metadata as bge_build_meta,
    cache_is_valid as bge_cache_valid,
    RETRIEVER_TYPE as BGE_RETRIEVER_TYPE,
)
from feg_rag.retrieval.splade_v3 import (
    SPLADEV3Retriever,
    build_cache_metadata as splade_build_meta,
    cache_is_valid as splade_cache_valid,
    RETRIEVER_TYPE as SPLADE_RETRIEVER_TYPE,
)


# ---------------------------------------------------------------------------
# Helper: standard retrieval metrics (no dependency on generation pipeline)
# ---------------------------------------------------------------------------

def compute_retrieval_metrics(
    method_name: str,
    results: List[Dict[str, Any]],
    k_values: Tuple[int, ...] = (5, 10, 50),
) -> Dict[str, Any]:
    """Compute retrieval-only metrics from a list of per-query result dicts.

    Each result dict must have:
        gold_evidence_ids: List[str]
        retrieved_chunk_ids: List[str]  (ordered by rank)
    """
    n = len(results)
    if n == 0:
        return {"method_name": method_name, "num_samples": 0}

    recall_at_k: Dict[int, float] = defaultdict(float)
    precision_at_k: Dict[int, float] = defaultdict(float)
    mrr_sum = 0.0
    ndcg_at_k: Dict[int, float] = defaultdict(float)
    hit_at_k: Dict[int, float] = defaultdict(float)

    import numpy as np

    for r in results:
        gold = set(r.get("gold_evidence_ids", []))
        retrieved = r.get("retrieved_chunk_ids", [])
        if not gold:
            n -= 1
            continue

        # Recall@K, Precision@K, Hit@K
        for k in k_values:
            top_k_ids = retrieved[:k]
            hits = len(set(top_k_ids) & gold)
            recall_at_k[k] += hits / len(gold) if len(gold) > 0 else 0.0
            precision_at_k[k] += hits / k if k > 0 else 0.0
            hit_at_k[k] += 1.0 if hits > 0 else 0.0

        # MRR
        for rank, cid in enumerate(retrieved, 1):
            if cid in gold:
                mrr_sum += 1.0 / rank
                break

        # nDCG@K
        for k in k_values:
            top_k_ids = retrieved[:k]
            dcg = 0.0
            for i, cid in enumerate(top_k_ids):
                if cid in gold:
                    dcg += 1.0 / np.log2(i + 2)
            ideal_count = min(len(gold), k)
            idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_count))
            ndcg_at_k[k] += dcg / idcg if idcg > 0 else 0.0

    n = max(n, 1)
    metrics: Dict[str, Any] = {
        "method_name": method_name,
        "num_samples": n,
        "mrr": mrr_sum / n,
        "MRR": mrr_sum / n,
    }
    for k in k_values:
        recall = recall_at_k[k] / n
        precision = precision_at_k[k] / n
        ndcg = ndcg_at_k[k] / n
        hit = hit_at_k[k] / n
        metrics[f"recall@{k}"] = recall
        metrics[f"precision@{k}"] = precision
        metrics[f"ndcg@{k}"] = ndcg
        metrics[f"hit@{k}"] = hit
        metrics[f"Recall@{k}"] = recall
        metrics[f"Precision@{k}"] = precision
        metrics[f"nDCG@{k}"] = ndcg
        metrics[f"Hit@{k}"] = hit

    return metrics


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------

def run_eval(
    retriever_name: str,
    retriever,
    samples: List[Dict],
    corpus_chunks: List[Chunk],
    gold_map: Dict[str, List[str]],
    top_k: int = 50,
    k_values: Tuple[int, ...] = (5, 10, 50),
    verbose: bool = True,
    cache_dir: Optional[Path] = None,
    cache_meta_builder=None,
    cache_validator=None,
    load_fn=None,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    force_rebuild_cache: bool = False,
) -> Dict[str, Any]:
    """Run retrieval evaluation for a single retriever.

    Args:
        retriever_name: "BGE-M3-Dense" or "SPLADE-v3"
        retriever: An object with an .index(chunks) and .search(query, top_k) method.
        samples: List of QA sample dicts (each has "id", "question").
        corpus_chunks: Full corpus chunk list.
        gold_map: Dict mapping sample_id → list of gold chunk_ids.
        top_k: Number of results to retrieve.
        k_values: Cutoffs for Recall/Precision/nDCG/Hit.
        verbose: Print progress.

    Returns:
        Metrics dict.
    """
    # Index, reusing a validated full-corpus retriever cache when available.
    t0 = time.time()
    cache_status = "disabled"
    if cache_dir is not None and cache_meta_builder and cache_validator and load_fn:
        retriever, cache_status = _load_or_build_index_cache(
            retriever_name=retriever_name,
            retriever=retriever,
            corpus_chunks=corpus_chunks,
            gold_map=gold_map,
            cache_dir=cache_dir,
            build_meta=cache_meta_builder,
            cache_valid=cache_validator,
            load_fn=load_fn,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            force_rebuild=force_rebuild_cache,
            verbose=verbose,
        )
    else:
        if verbose:
            print(f"  [{retriever_name}] Indexing {len(corpus_chunks)} chunks...")
        retriever.index(corpus_chunks)
        cache_status = "not_configured"

    index_time = time.time() - t0
    if verbose:
        print(f"  [{retriever_name}] Index ready in {index_time:.1f}s "
              f"(cache={cache_status})")

    # Retrieve
    results: List[Dict[str, Any]] = []
    t0 = time.time()
    for i, s in enumerate(samples):
        if verbose and (i % 20 == 0 or i == len(samples) - 1):
            print(f"  [{retriever_name}] {i+1}/{len(samples)} queries")

        try:
            retrieved = retriever.search(s["question"], top_k=top_k)
            retrieved_ids = [c.chunk_id for c, _ in retrieved]
        except Exception as exc:
            print(f"  [{retriever_name}] WARN: query '{s['id']}' failed: {exc}")
            retrieved_ids = []

        results.append({
            "sample_id": s["id"],
            "question": s["question"],
            "gold_evidence_ids": gold_map.get(s["id"], []),
            "retrieved_chunk_ids": retrieved_ids,
        })

    search_time = time.time() - t0
    if verbose:
        print(f"  [{retriever_name}] Search completed in {search_time:.1f}s")

    # Compute metrics
    metrics = compute_retrieval_metrics(retriever_name, results, k_values)
    metrics["index_time_s"] = round(index_time, 1)
    metrics["search_time_s"] = round(search_time, 1)
    metrics["cache_status"] = cache_status
    return metrics


def _load_or_build_index_cache(
    *,
    retriever_name: str,
    retriever,
    corpus_chunks: List[Chunk],
    gold_map: Dict[str, List[str]],
    cache_dir: Path,
    build_meta,
    cache_valid,
    load_fn,
    chunk_size: Optional[int],
    chunk_overlap: Optional[int],
    force_rebuild: bool,
    verbose: bool,
):
    """Load a validated retriever index cache or build and save it once.

    This mirrors the full-corpus cache pattern: document encodings / indexes
    are built once and reused across runs as long as the corpus, gold map, and
    retriever parameters still match. Query encoders may still be loaded for
    search, but the expensive full-corpus pass is skipped on cache hits.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta_path = cache_dir / "cache_meta.json"

    expected = build_meta(corpus_chunks, gold_map, retriever, extra={
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
    })
    check_keys = [
        "model_name",
        "retriever_type",
        "corpus_hash",
        "gold_map_hash",
        "chunk_size",
        "chunk_overlap",
        "max_length",
        "normalize",
        "revision",
    ]
    if retriever_name == "SPLADE-v3":
        check_keys.append("sparsify_threshold")

    if not force_rebuild and _retriever_cache_artifacts_exist(cache_dir):
        if cache_valid(meta_path, expected, check_keys=check_keys):
            try:
                if verbose:
                    print(f"  [{retriever_name}] Loading cached index: {cache_dir}")
                return load_fn(cache_dir, device=getattr(retriever, "_device", None)), "hit"
            except Exception as exc:
                print(f"  [{retriever_name}] WARN: cache load failed, rebuilding: {exc}")
        elif verbose:
            print(f"  [{retriever_name}] Cache metadata mismatch, rebuilding")
    elif verbose:
        reason = "forced rebuild" if force_rebuild else "cache missing"
        print(f"  [{retriever_name}] {reason}, building index")

    retriever.index(corpus_chunks)
    retriever.save(cache_dir)

    final_meta = build_meta(corpus_chunks, gold_map, retriever, extra={
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
    })
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(final_meta, fh, indent=2, sort_keys=True, default=str)

    if verbose:
        print(f"  [{retriever_name}] Saved index cache: {cache_dir}")
    return retriever, "rebuilt"


def _retriever_cache_artifacts_exist(cache_dir: Path) -> bool:
    """Return True only for complete BGE-M3 or SPLADE-v3 cache artifacts."""
    meta_json = cache_dir / "cache_meta.json"
    meta_pkl = cache_dir / "meta.pkl"
    if not meta_json.exists() or not meta_pkl.exists():
        return False
    has_bge = (cache_dir / "index.faiss").exists()
    has_splade = (cache_dir / "doc_sparse.npz").exists()
    return has_bge or has_splade


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified evaluation: BGE-M3-Dense + SPLADE-v3"
    )
    parser.add_argument("--config", default="configs/default.yaml",
                        help="Path to YAML config")
    parser.add_argument("--split", default=None,
                        help="Dataset split (train/dev/test)")
    parser.add_argument("--num_samples", type=int, default=0,
                        help="Limit samples (0=all)")
    parser.add_argument("--top_k", type=int, default=50,
                        help="Number of results to retrieve per query")
    parser.add_argument("--output", default=None,
                        help="Output JSON path (default: outputs/retrieval_baselines_<ts>.json)")
    parser.add_argument("--hf_endpoint", default=None,
                        help="HuggingFace endpoint (e.g. https://hf-mirror.com)")
    parser.add_argument("--device", default=None,
                        help="Device: cpu or cuda")
    parser.add_argument("--skip_bge_m3", action="store_true",
                        help="Skip BGE-M3-Dense evaluation")
    parser.add_argument("--skip_splade", action="store_true",
                        help="Skip SPLADE-v3 evaluation")
    parser.add_argument("--retrievers", nargs="+", default=None,
                        help="Which retrievers to run (bge_m3, splade_v3)")
    parser.add_argument("--force_rebuild_cache", action="store_true",
                        help="Rebuild retriever index caches even if metadata matches")
    args = parser.parse_args()

    # Load config
    cfg = Config.from_yaml(args.config)
    cfg.ensure_dirs()

    # Resolve HF endpoint (CLI arg > config > env)
    hf_endpoint = args.hf_endpoint
    if hf_endpoint is None:
        hf_endpoint = cfg.retrieval.get("hf_endpoint")
    if hf_endpoint is None:
        hf_endpoint = os.environ.get("HF_ENDPOINT")

    device = args.device or ("cuda" if _cuda_available() else "cpu")

    # Determine which retrievers to run
    if args.retrievers:
        run_bge = "bge_m3" in args.retrievers
        run_splade = "splade_v3" in args.retrievers
    else:
        run_bge = not args.skip_bge_m3
        run_splade = not args.skip_splade

    print("=" * 60)
    print("  BGE-M3-Dense + SPLADE-v3 Unified Evaluation")
    print("=" * 60)
    print(f"  Config: {args.config}")
    print(f"  Device: {device}")
    print(f"  HF_ENDPOINT: {hf_endpoint or 'not set'}")
    print(f"  Top-K: {args.top_k}")
    print(f"  BGE-M3-Dense: {'ON' if run_bge else 'OFF'}")
    print(f"  SPLADE-v3: {'ON' if run_splade else 'OFF'}")

    # ---- 1. Load data ----
    print("\n[1] Loading data...")
    all_samples: List[Dict] = []
    for ds_name in cfg.datasets:
        try:
            samples = load_dataset(
                ds_name, cfg.data_dir,
                split=args.split or cfg._raw.get("data_split"),
                files=cfg._raw.get("data_files"),
            )
            all_samples.extend(samples)
            print(f"  {ds_name}: {len(samples)} samples")
        except FileNotFoundError as e:
            print(f"  [SKIP] {ds_name}: {e}")

    if args.num_samples > 0:
        all_samples = all_samples[:args.num_samples]
    print(f"  Total: {len(all_samples)} samples")

    # ---- 2. Build corpus ----
    print("\n[2] Building corpus...")
    corpus_chunks, gold_map, alignments = build_benchmark_corpus(
        all_samples, cfg
    )
    print(f"  Corpus: {len(corpus_chunks)} chunks")
    print(f"  Gold-mapped queries: {sum(1 for v in gold_map.values() if v)}")

    k_values: Tuple[int, ...] = tuple(
        int(k) for k in cfg.evaluation.get("recall_k_values", [5, 10, 50])
    )
    if 50 not in k_values:
        k_values = tuple(sorted(set(k_values) | {50}))
    k_values = tuple(k for k in k_values if k <= args.top_k)

    all_metrics: List[Dict[str, Any]] = []

    # ---- 3. BGE-M3-Dense ----
    if run_bge:
        print("\n[3] BGE-M3-Dense...")
        bge_retriever = BGEM3DenseRetriever(
            model_name="BAAI/bge-m3",
            device=device,
            max_length=cfg.retrieval.get("max_length", 512),
            batch_size=cfg.retrieval.get("bge_batch_size", 16),
            hf_endpoint=hf_endpoint,
            revision=cfg.retrieval.get("revision", "main"),
            normalize=True,
        )
        try:
            bge_metrics = run_eval(
                "BGE-M3-Dense", bge_retriever,
                all_samples, corpus_chunks, gold_map,
                top_k=args.top_k, k_values=k_values,
                cache_dir=cfg.cache_dir / "retrieval_indexes" / BGE_RETRIEVER_TYPE,
                cache_meta_builder=bge_build_meta,
                cache_validator=bge_cache_valid,
                load_fn=BGEM3DenseRetriever.load,
                chunk_size=cfg.chunk_size,
                chunk_overlap=cfg.chunk_overlap,
                force_rebuild_cache=args.force_rebuild_cache,
            )
            all_metrics.append(bge_metrics)
            _print_metrics(bge_metrics)
        except Exception as exc:
            print(f"  [ERROR] BGE-M3-Dense evaluation failed: {exc}")
            import traceback
            traceback.print_exc()

    # ---- 4. SPLADE-v3 ----
    if run_splade:
        print("\n[4] SPLADE-v3...")
        splade_retriever = SPLADEV3Retriever(
            model_name="naver/splade-v3",
            device=device,
            max_length=cfg.retrieval.get("max_length", 512),
            batch_size=cfg.retrieval.get("splade_batch_size", 8),
            hf_endpoint=hf_endpoint,
            revision=cfg.retrieval.get("revision", "main"),
            normalize=False,
        )
        try:
            splade_metrics = run_eval(
                "SPLADE-v3", splade_retriever,
                all_samples, corpus_chunks, gold_map,
                top_k=args.top_k, k_values=k_values,
                cache_dir=cfg.cache_dir / "retrieval_indexes" / SPLADE_RETRIEVER_TYPE,
                cache_meta_builder=splade_build_meta,
                cache_validator=splade_cache_valid,
                load_fn=SPLADEV3Retriever.load,
                chunk_size=cfg.chunk_size,
                chunk_overlap=cfg.chunk_overlap,
                force_rebuild_cache=args.force_rebuild_cache,
            )
            all_metrics.append(splade_metrics)
            _print_metrics(splade_metrics)
        except Exception as exc:
            print(f"  [ERROR] SPLADE-v3 evaluation failed: {exc}")
            import traceback
            traceback.print_exc()

    # ---- 5. Save ----
    if all_metrics:
        output_path = Path(args.output or _default_output(cfg))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "timestamp": datetime.now().isoformat(),
            "config": args.config,
            "hf_endpoint": hf_endpoint,
            "device": device,
            "num_samples": len(all_samples),
            "num_corpus_chunks": len(corpus_chunks),
            "top_k": args.top_k,
            "k_values": list(k_values),
            "metrics": all_metrics,
        }
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, default=str)
        print(f"\nResults saved to {output_path}")

    print("\nDone.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _print_metrics(m: Dict[str, Any]) -> None:
    print(f"  --- {m.get('method_name', '?')} ---")
    print(f"  num_samples:  {m.get('num_samples', '?')}")
    print(f"  MRR:          {m.get('mrr', 0):.4f}")
    for k in [5, 10, 50]:
        key = f"recall@{k}"
        if key in m:
            print(f"  Recall@{k}:     {m[key]:.4f}")
    for k in [5, 10]:
        key = f"ndcg@{k}"
        if key in m:
            print(f"  nDCG@{k}:       {m[key]:.4f}")
    for k in [5, 10]:
        key = f"hit@{k}"
        if key in m:
            print(f"  Hit@{k}:        {m[key]:.4f}")


def _default_output(cfg: Config) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return cfg.output_dir / f"retrieval_baselines_bge_m3_splade_v3_{stamp}.json"


if __name__ == "__main__":
    main()
