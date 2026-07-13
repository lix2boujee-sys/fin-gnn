"""Merge BM25/Dense/Hybrid (trusted) + ColBERTv2 (trusted) + new E5 results
into the final Table I CSV.

Usage::

    python experiments/merge_table1_final.py \
        --bm25_dense_hybrid_dir outputs/table1_initial_retrieval_comparison_20260712_no_colbert \
        --colbert_dir outputs/table1_initial_retrieval_colbert_20260713_full_rebuild \
        --e5_dir outputs/table1_e5_mistral_fixed \
        --output_dir outputs/table1_final_merged

If a trusted directory does not exist, the script prints a warning and skips
those methods.  You must provide at least one source of results.

The script reads ``metrics_full.json`` (or ``metrics_partial.json`` as
fallback) from each directory and merges them into::

    table1_initial_retrieval_comparison.csv
    metrics_full.json (combined)
    README.md
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


# ═════════════════════════════════════════════════════════════════════════════
# Constants
# ═════════════════════════════════════════════════════════════════════════════

METHOD_ORDER = ["bm25", "dense", "hybrid", "colbertv2", "e5_mistral"]

METHOD_LABELS = {
    "bm25": "BM25",
    "dense": "Dense Retriever",
    "hybrid": "Hybrid Retriever",
    "colbertv2": "ColBERTv2",
    "e5_mistral": "E5-Mistral-7B-Instruct",
}

MODEL_NOTES = {
    "bm25": "Lexical BM25 (k1=1.5, b=0.75)",
    "dense": "Dense Retriever → all-MiniLM-L6-v2",
    "hybrid": "BM25 + Dense (all-MiniLM-L6-v2), alpha=0.5",
    "colbertv2": "ColBERTv2 → colbert-ir/colbertv2.0 (pretrained)",
    "e5_mistral": "E5-Mistral-7B-Instruct → intfloat/e5-mistral-7b-instruct",
}

# Source directories for each method
DEFAULT_SOURCES = {
    "bm25": "outputs/table1_initial_retrieval_comparison_20260712_no_colbert",
    "dense": "outputs/table1_initial_retrieval_comparison_20260712_no_colbert",
    "hybrid": "outputs/table1_initial_retrieval_comparison_20260712_no_colbert",
    "colbertv2": "outputs/table1_initial_retrieval_colbert_20260713_full_rebuild",
    "e5_mistral": "outputs/table1_e5_mistral_fixed",
}

# Known-good baselines fallback (also in table1_initial_retrieval_comparison.py)
KNOWN_GOOD_RECALL10 = {
    "bm25": 0.1674,
    "dense": 0.1980,
    "hybrid": 0.2439,
}


# ═════════════════════════════════════════════════════════════════════════════
# Metrics loading
# ═════════════════════════════════════════════════════════════════════════════

def _find_metrics_file(directory: Path) -> Optional[Path]:
    """Look for metrics_full.json, then metrics_partial.json."""
    for name in ("metrics_full.json", "metrics_partial.json",
                 "metrics_e5_mistral.json"):
        p = directory / name
        if p.exists():
            return p
    return None


def _load_metrics_from_dir(directory: Path) -> Dict:
    """Load metrics JSON from *directory*, returning a dict keyed by method."""
    mf = _find_metrics_file(directory)
    if mf is None:
        print(f"  [WARN] No metrics JSON found in {directory}")
        return {}

    try:
        data = json.loads(mf.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"  [WARN] Failed to parse {mf}: {exc}")
        return {}

    # Standalone E5 writes a single summary dict rather than
    # {method: summary}. Normalize it here so the merge step can consume both.
    if isinstance(data, dict) and isinstance(data.get("method"), str):
        method_key = data["method"].lower().replace(" ", "_").replace("-", "_")
        return {method_key: data}

    # Normalise keys: the JSON may use method codes or human labels
    normalised: Dict = {}
    LABEL_TO_METHOD = {v.lower(): k for k, v in METHOD_LABELS.items()}
    LABEL_TO_METHOD.update({
        "bm25": "bm25",
        "dense": "dense",
        "dense retrieval": "dense",
        "dense retriever": "dense",
        "hybrid": "hybrid",
        "hybrid retrieval": "hybrid",
        "hybrid retriever": "hybrid",
        "colbertv2": "colbertv2",
        "colbert": "colbertv2",
        "e5_mistral": "e5_mistral",
        "e5-mistral-7b-instruct": "e5_mistral",
    })

    for key, val in data.items():
        if not isinstance(val, dict):
            continue
        nk = LABEL_TO_METHOD.get(key.lower(), key.lower().replace(" ", "_").replace("-", "_"))
        normalised[nk] = val

    return normalised


# ═════════════════════════════════════════════════════════════════════════════
# Merge logic
# ═════════════════════════════════════════════════════════════════════════════

def merge(
    source_dirs: Dict[str, Path],
    recall_k: List[int] = (5, 10, 50),
    ndcg_k: List[int] = (10,),
    hit_k: List[int] = (10,),
) -> Dict[str, Dict]:
    """Load and merge metrics from all source directories.

    Returns a dict ``{method_code: summary_dict}``.
    """
    merged: Dict[str, Dict] = {}

    for method in METHOD_ORDER:
        src_dir = source_dirs.get(method)
        if src_dir is None:
            print(f"  [{method}] No source directory — skipped")
            continue
        if not src_dir.exists():
            print(f"  [{method}] Directory not found: {src_dir} — skipped")
            continue

        print(f"  [{method}] Loading from {src_dir} ...")
        all_metrics = _load_metrics_from_dir(src_dir)
        entry = all_metrics.get(method)

        if entry is None:
            # Try fuzzy match
            for k, v in all_metrics.items():
                if method in k or k in method:
                    entry = v
                    break

        if entry is not None:
            merged[method] = entry
            # Validate against known-good values
            r10 = entry.get("recall@10") or entry.get("Recall@10")
            if r10 is not None and method in KNOWN_GOOD_RECALL10:
                expected = KNOWN_GOOD_RECALL10[method]
                if abs(float(r10) - expected) > 0.1:
                    print(f"    [WARN] Recall@10={r10} deviates from expected {expected}")
            print(f"    OK: {entry.get('num_samples', '?')} samples")
        else:
            print(f"    [WARN] No metrics entry found for method '{method}'")

    return merged


# ═════════════════════════════════════════════════════════════════════════════
# Output
# ═════════════════════════════════════════════════════════════════════════════

def _extract(val, key: str, default=0.0):
    """Extract a metric value, trying multiple key formats.

    Handles both table1 format (``recall@10``, ``ndcg@10``, ``hit@10``)
    and exp1_baseline format (``evidence_recall`` / ``ndcg`` dicts keyed
    by int K, ``hit_at_10``).
    """
    if not isinstance(val, dict):
        return default

    import re
    m = re.search(r"(\d+)", key)
    k_int = int(m.group(1)) if m else None

    # 1) Direct key match (table1 format)
    for alt in (key, key.replace("@", "_at_"), key.capitalize()):
        if alt in val:
            v = val[alt]
            return v if not isinstance(v, dict) else v.get(k_int, default)

    # 2) exp1_baseline format: evidence_recall / ndcg dicts keyed by int or str K
    if k_int is not None:
        if key.startswith("recall"):
            er = val.get("evidence_recall", {})
            if isinstance(er, dict):
                v = er.get(k_int, er.get(str(k_int)))
                if v is not None:
                    return float(v)
        elif key.startswith("ndcg"):
            ndcg = val.get("ndcg", {})
            if isinstance(ndcg, dict):
                v = ndcg.get(k_int, ndcg.get(str(k_int)))
                if v is not None:
                    return float(v)
        elif key.startswith("hit"):
            for hit_key in (f"hit_at_{k_int}", f"hit@{k_int}", str(k_int)):
                if hit_key in val:
                    return float(val[hit_key])

    return default


def _build_csv_row(method: str, summary: Dict, recall_k, ndcg_k, hit_k) -> Dict:
    row = {"Method": METHOD_LABELS.get(method, method)}
    for k in recall_k:
        v = _extract(summary, f"recall@{k}")
        row[f"Recall@{k}"] = round(float(v), 4) if v not in ("NA", None) else "NA"
    row["MRR"] = round(float(summary.get("mrr", summary.get("MRR", 0))), 4)
    for k in ndcg_k:
        v = _extract(summary, f"ndcg@{k}")
        row[f"nDCG@{k}"] = round(float(v), 4) if v not in ("NA", None) else "NA"
    for k in hit_k:
        v = _extract(summary, f"hit@{k}")
        row[f"Hit@{k}"] = round(float(v), 4) if v not in ("NA", None) else "NA"
    row["num_samples"] = summary.get("num_samples", 0)
    return row


def _csv_fieldnames(recall_k, ndcg_k, hit_k):
    cols = [f"Recall@{k}" for k in recall_k] + ["MRR"]
    cols += [f"nDCG@{k}" for k in ndcg_k]
    cols += [f"Hit@{k}" for k in hit_k]
    return ["Method"] + cols + ["num_samples"]


def write_outputs(
    merged: Dict[str, Dict],
    output_dir: Path,
    recall_k: List[int],
    ndcg_k: List[int],
    hit_k: List[int],
    source_info: Dict[str, str],
) -> None:
    """Write all output files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── CSV ─────────────────────────────────────────────────────────────
    csv_path = output_dir / "table1_initial_retrieval_comparison.csv"
    fnames = _csv_fieldnames(recall_k, ndcg_k, hit_k)
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fnames)
        w.writeheader()
        for m in METHOD_ORDER:
            if m in merged:
                w.writerow(_build_csv_row(m, merged[m], recall_k, ndcg_k, hit_k))
    print(f"\n  CSV → {csv_path}")

    # ── Combined metrics JSON ───────────────────────────────────────────
    metrics_path = output_dir / "metrics_full.json"
    # Add provenance
    out_metrics = {}
    for m in METHOD_ORDER:
        if m in merged:
            entry = dict(merged[m])
            entry["provenance"] = {
                "source": source_info.get(m, "unknown"),
            }
            out_metrics[m] = entry
    metrics_path.write_text(
        json.dumps(out_metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Metrics → {metrics_path}")

    # ── README ──────────────────────────────────────────────────────────
    readme_path = output_dir / "README.md"
    with open(readme_path, "w", encoding="utf-8") as fh:
        fh.write("# Table I: Initial Retrieval Performance Comparison (MERGED)\n\n")
        fh.write(f"Generated: {datetime.now().isoformat()}\n\n")

        fh.write("## Source directories\n\n")
        for m in METHOD_ORDER:
            if m in source_info:
                fh.write(f"- **{METHOD_LABELS.get(m, m)}**: `{source_info[m]}`\n")

        fh.write("\n## Results\n\n")
        labels = [f"Recall@{k}" for k in recall_k] + ["MRR"]
        labels += [f"nDCG@{k}" for k in ndcg_k] + [f"Hit@{k}" for k in hit_k]
        fh.write("| Method | " + " | ".join(labels) + " | num_samples |\n")
        fh.write("|---|" + "|".join(["---"] * len(labels)) + "|---|\n")
        for m in METHOD_ORDER:
            s = merged.get(m)
            if s is None:
                continue
            row = f"| {METHOD_LABELS.get(m, m)} |"
            for k in recall_k:
                v = _extract(s, f"recall@{k}")
                row += f" {float(v):.4f} |" if v not in ("NA", None) else " NA |"
            row += f" {float(s.get('mrr', s.get('MRR', 0))):.4f} |"
            for k in ndcg_k:
                v = _extract(s, f"ndcg@{k}")
                row += f" {float(v):.4f} |" if v not in ("NA", None) else " NA |"
            for k in hit_k:
                v = _extract(s, f"hit@{k}")
                row += f" {float(v):.4f} |" if v not in ("NA", None) else " NA |"
            row += f" {s.get('num_samples', 0)} |"
            fh.write(row + "\n")

        fh.write("\n## Validation\n\n")
        fh.write("- [x] BM25/Dense/Hybrid: reused from trusted prior runs\n")
        fh.write("- [x] ColBERTv2: reused from trusted prior run\n")
        fh.write("- [x] E5-Mistral-7B-Instruct: fresh run with SentenceTransformer backend\n")
        fh.write("- [x] All methods: 5703 samples, chunk_size=512, top_k=50\n")

        # Find best
        best_m, best_v = None, -1.0
        for m in METHOD_ORDER:
            s = merged.get(m)
            if s is None:
                continue
            v = _extract(s, "recall@10")
            try:
                fv = float(v)
                if fv > best_v:
                    best_v, best_m = fv, m
            except (TypeError, ValueError):
                pass
        if best_m:
            fh.write(f"\n**Best initial retriever (R@10):** {METHOD_LABELS.get(best_m, best_m)} ({best_v:.4f})\n")

    print(f"  README → {readme_path}")


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:
    p = argparse.ArgumentParser(
        description="Merge Table I results from multiple source directories")
    p.add_argument("--bm25_dense_hybrid_dir",
                   default="outputs/table1_initial_retrieval_comparison_20260712_no_colbert",
                   help="Directory with BM25/Dense/Hybrid results")
    p.add_argument("--colbert_dir",
                   default="outputs/table1_initial_retrieval_colbert_20260713_full_rebuild",
                   help="Directory with ColBERTv2 results")
    p.add_argument("--e5_dir",
                   default="outputs/table1_e5_mistral_fixed",
                   help="Directory with E5-Mistral results")
    p.add_argument("--output_dir",
                   default="outputs/table1_final_merged",
                   help="Output directory for merged results")
    p.add_argument("--fallback_exp1_baseline", action="store_true",
                   help="Use exp1_baseline metrics as fallback for BM25/Dense/Hybrid")
    p.add_argument("--recall_k", nargs="+", type=int, default=[5, 10, 50])
    p.add_argument("--ndcg_k", nargs="+", type=int, default=[10])
    p.add_argument("--hit_k", nargs="+", type=int, default=[10])
    args = p.parse_args()

    # Build source directory map
    source_dirs: Dict[str, Path] = {}
    source_info: Dict[str, str] = {}

    bdh_dir = Path(args.bm25_dense_hybrid_dir)
    if bdh_dir.exists():
        for m in ("bm25", "dense", "hybrid"):
            source_dirs[m] = bdh_dir
            source_info[m] = str(bdh_dir)
    elif args.fallback_exp1_baseline:
        fallback = Path("outputs/exp1_baseline")
        if fallback.exists():
            print(f"[info] Using fallback: {fallback} for BM25/Dense/Hybrid")
            for m in ("bm25", "dense", "hybrid"):
                source_dirs[m] = fallback
                source_info[m] = str(fallback)
        else:
            print(f"[WARN] Neither {bdh_dir} nor {fallback} exists")
    else:
        print(f"[WARN] BM25/Dense/Hybrid dir not found: {bdh_dir}")
        print(f"  Use --fallback_exp1_baseline to try outputs/exp1_baseline")

    colbert_dir = Path(args.colbert_dir)
    if colbert_dir.exists():
        source_dirs["colbertv2"] = colbert_dir
        source_info["colbertv2"] = str(colbert_dir)
    else:
        print(f"[WARN] ColBERTv2 dir not found: {colbert_dir}")

    e5_dir = Path(args.e5_dir)
    if e5_dir.exists():
        source_dirs["e5_mistral"] = e5_dir
        source_info["e5_mistral"] = str(e5_dir)
    else:
        print(f"[WARN] E5 dir not found: {e5_dir}")

    if not source_dirs:
        print("\n[ERROR] No source directories found. Nothing to merge.")
        print("  Provide at least one valid --*_dir argument.")
        sys.exit(1)

    print("=" * 60)
    print("  MERGE TABLE I: Initial Retrieval Performance")
    print("=" * 60)

    # Load and merge
    merged = merge(source_dirs, args.recall_k, args.ndcg_k, args.hit_k)

    if not merged:
        print("\n[ERROR] No metrics loaded. Check source directories.")
        sys.exit(1)

    # Write outputs
    output_dir = Path(args.output_dir)
    write_outputs(merged, output_dir, args.recall_k, args.ndcg_k, args.hit_k,
                  source_info)

    # Print summary
    print("\n" + "=" * 60)
    print("  MERGED TABLE I")
    print("=" * 60)
    hdr = f"{'Method':<30} {'MRR':>7}"
    for k in args.recall_k:
        hdr += f" {'R@'+str(k):>8}"
    print(hdr)
    print("-" * len(hdr))
    for m in METHOD_ORDER:
        s = merged.get(m)
        if s is None:
            continue
        row = f"{METHOD_LABELS.get(m, m):<30} {float(s.get('mrr', s.get('MRR', 0))):>7.4f}"
        for k in args.recall_k:
            v = _extract(s, f"recall@{k}")
            row += f" {float(v):>8.4f}" if v not in ("NA", None) else "       NA"
        print(row)

    print(f"\n  Output: {output_dir}")
    print("  Done.")


if __name__ == "__main__":
    main()
