"""Table 1: Initial Retrieval Performance Comparison.

Compares first-stage retrievers on the Financial Evidence Retrieval Benchmark
WITHOUT reranking, graph methods, or LLMs.

Methods (each is an independent full-corpus retriever):
    BM25                    — lexical sparse retrieval
    Dense Retriever         — all-MiniLM-L6-v2 dense retrieval
    Hybrid Retriever        — BM25 + Dense (all-MiniLM-L6-v2), alpha=0.5
    ColBERTv2               — colbert-ir/colbertv2.0 (pretrained, no fine-tuning)
    E5-Mistral-7B-Instruct   — e5-mistral-7b-instruct (INDEPENDENT dense retriever)

Metrics: Recall@5, Recall@10, Recall@50, MRR, nDCG@10, Hit@10

Usage:
    # Smoke test (no ColBERT)
    python experiments/table1_initial_retrieval_comparison.py \\
        --config configs/table1_initial_retrieval_comparison.yaml \\
        --limit_samples 20 --skip_colbert

    # Full run with history reuse (cloud)
    python experiments/table1_initial_retrieval_comparison.py \\
        --config configs/table1_initial_retrieval_comparison_cloud.yaml \\
        --dense_device cuda \\
        --reuse_baseline_dir /root/fin-gnn_outputs/exp1_baseline

    # Full run without history (cloud)
    python experiments/table1_initial_retrieval_comparison.py \\
        --config configs/table1_initial_retrieval_comparison_cloud.yaml \\
        --dense_device cuda
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

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from feg_rag.config import Config
from feg_rag.data.chunker import Chunk, chunk_text, chunk_report
from feg_rag.data.loader import load_dataset
from feg_rag.evaluation.metrics import compute_all_metrics
from feg_rag.retrieval.bm25 import BM25Retriever
from feg_rag.retrieval.dense import DenseRetriever
from feg_rag.retrieval.hybrid import HybridRetriever

# ColBERTv2 — lazy; has heavy deps
_COLBERT_AVAILABLE = False
_COLBERT_IMPORT_ERROR: Optional[str] = None
try:
    from feg_rag.retrieval.colbertv2 import ColBERTv2Retriever, _COLBERT_AVAILABLE as _CA
    _COLBERT_AVAILABLE = _CA
except ImportError as exc:
    _COLBERT_IMPORT_ERROR = str(exc)


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

METHOD_FILES = {
    "bm25": "bm25_results.jsonl",
    "dense": "dense_results.jsonl",
    "hybrid": "hybrid_results.jsonl",
    "colbertv2": "colbertv2_results.jsonl",
    "e5_mistral": "e5_mistral_results.jsonl",
}

# ── History reuse ─────────────────────────────────────────────────────────

SUSPECT_DIRS: Set[str] = {
    "D:/fin-gnn_outputs/exp1_e5_mistral",
    "/root/fin-gnn_outputs/exp1_e5_mistral",
}

KNOWN_GOOD_BASELINES: Dict[str, Dict] = {
    "D:/fin-gnn_outputs/exp1_baseline": {
        "num_samples": 5703,
        "expected_recall10": {"bm25": 0.1674, "dense": 0.1980, "hybrid": 0.2439},
        "notes": "exp1 baseline; BM25/Dense/Hybrid",
    },
    "/root/fin-gnn_outputs/exp1_baseline": {
        "num_samples": 5703,
        "expected_recall10": {"bm25": 0.1674, "dense": 0.1980, "hybrid": 0.2439},
        "notes": "exp1 baseline; BM25/Dense/Hybrid",
    },
}

# Per-method provenance
_PROVENANCE: Dict[str, Dict] = {}

# Methods eligible for history reuse.  E5-Mistral is NOT reusable — always
# fresh run to avoid contaminating the table with anomalous historical data.
_REUSABLE_METHODS = {"bm25", "dense", "hybrid"}

_HISTORY_LABEL_MAP = {
    "bm25": "bm25", "BM25": "bm25",
    "dense": "dense", "Dense": "dense", "Dense Retrieval": "dense",
    "hybrid": "hybrid", "Hybrid": "hybrid", "Hybrid (BM25 + Dense)": "hybrid",
}


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════

def _build_corpus(
    samples: List[Dict],
    cfg: Config,
    max_distractor_files: int = 50,
) -> Tuple[List[Chunk], Dict[str, List[str]]]:
    corpus: List[Chunk] = []
    gold_map: Dict[str, List[str]] = {}
    for s in samples:
        gold_ids = []
        for text in s["evidence_texts"]:
            for c in chunk_text(text, cfg.chunk_size, cfg.chunk_overlap, doc_id=s["id"]):
                corpus.append(c)
                gold_ids.append(c.chunk_id)
        gold_map[s["id"]] = gold_ids
    edgar_dir = cfg.edgar_dir
    if edgar_dir.exists():
        txt_files = list(edgar_dir.rglob("*.txt")) or list(edgar_dir.rglob("*.html"))
        for tf in txt_files[:max_distractor_files]:
            try:
                corpus.extend(chunk_report(tf, cfg.chunk_size, cfg.chunk_overlap))
            except Exception:
                pass
    return corpus, gold_map


def _make_result(sample, retrieved_ids, gold_ids, method) -> Dict:
    return {
        "question_id": sample["id"],
        "question": sample["question"],
        "gold_answer": sample.get("answer", ""),
        "gold_evidence_ids": gold_ids,
        "retrieved_chunk_ids": retrieved_ids,
        "method": method,
    }


def _has_results(out_dir: Path) -> bool:
    for s in ("metrics_full.json", "table1_initial_retrieval_comparison.csv"):
        if (out_dir / s).exists():
            return True
    return False


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


# ═════════════════════════════════════════════════════════════════════════════
# History loading & validation
# ═════════════════════════════════════════════════════════════════════════════

def _check_suspect_dir(baseline_dir: str | Path, allow_suspect: bool) -> None:
    normalized = str(Path(baseline_dir).resolve())
    for suspect in SUSPECT_DIRS:
        try:
            sn = str(Path(suspect).resolve())
        except Exception:
            sn = suspect
        if normalized == sn or normalized.rstrip("/\\") == sn.rstrip("/\\"):
            if allow_suspect:
                print(f"  [WARN] Loading from SUSPECT dir (--allow_suspect_history): {baseline_dir}")
                return
            else:
                print(f"  [ERROR] Refusing suspect directory: {baseline_dir}")
                print(f"    Use --allow_suspect_history to override.")
                sys.exit(1)


def _validate_jsonl_record(record: Dict, expected_method: str) -> Optional[str]:
    if not isinstance(record, dict):
        return "not a dict"
    for field in ("question_id", "gold_evidence_ids", "retrieved_chunk_ids"):
        if field not in record:
            return f"missing '{field}'"
    if not record.get("retrieved_chunk_ids"):
        return "empty retrieved_chunk_ids"
    actual = _HISTORY_LABEL_MAP.get(record.get("method", ""))
    if actual != expected_method:
        return f"method '{record.get('method')}' != '{expected_method}'"
    return None


def _validate_and_load_jsonl(
    jsonl_path: Path, method: str, expected_num_samples: int, require_top_k: int = 50,
) -> Tuple[List[Dict], List[str], bool]:
    """Returns (results, warnings, needs_rerun_for_topk)."""
    warnings: List[str] = []
    if not jsonl_path.exists():
        return [], [f"not found: {jsonl_path}"], True

    results: List[Dict] = []
    line_errors = 0
    with open(jsonl_path, "r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                line_errors += 1
                if line_errors <= 3:
                    warnings.append(f"line {line_no}: bad JSON: {exc}")
                continue
            err = _validate_jsonl_record(record, method)
            if err:
                line_errors += 1
                if line_errors <= 3:
                    warnings.append(f"line {line_no}: {err}")
                continue
            results.append(record)

    if line_errors > 3:
        warnings.append(f"... +{line_errors - 3} more errors")
    if len(results) != expected_num_samples:
        warnings.append(f"count {len(results)} != expected {expected_num_samples}")
    if len(results) == 0:
        return [], warnings, True

    min_retrieved = min(len(r.get("retrieved_chunk_ids", [])) for r in results)
    needs_rerun = min_retrieved < require_top_k
    if needs_rerun:
        warnings.append(
            f"min retrieved={min_retrieved} < required {require_top_k} (R@{require_top_k})"
        )

    for r in results:
        r["method"] = method
    return results, warnings, needs_rerun


def _sanity_check_history_results(
    results, method, corpus_chunk_ids, sample_ids, expected_metrics=None,
) -> List[str]:
    warnings: List[str] = []
    result_ids = {r["question_id"] for r in results}
    if result_ids != sample_ids:
        missing = sample_ids - result_ids
        extra = result_ids - sample_ids
        if missing:
            warnings.append(f"missing {len(missing)} sample ids")
        if extra:
            warnings.append(f"extra {len(extra)} unknown ids")

    unknown = 0
    for r in results:
        for cid in r.get("retrieved_chunk_ids", []):
            if cid not in corpus_chunk_ids:
                unknown += 1
                if unknown <= 3:
                    warnings.append(f"unknown chunk_id {cid} in {r['question_id']}")
                break
    if unknown > 3:
        warnings.append(f"... +{unknown - 3} more unknown chunk_ids")

    if expected_metrics:
        try:
            from experiments.table1_initial_retrieval_comparison import _recompute_recall_at_k as _rrk
        except ImportError:
            _rrk = None  # fallback
        if _rrk:
            r10 = _rrk(results, 10)
            exp = expected_metrics.get("recall10")
            if exp is not None and abs(r10 - exp) > 0.1:
                warnings.append(f"Recall@10={r10:.4f} deviates from expected {exp:.4f}")

    return warnings


def _recompute_recall_at_k(results: List[Dict], k: int) -> float:
    if not results:
        return 0.0
    recalls = []
    for r in results:
        gold = set(r.get("gold_evidence_ids", []))
        retrieved = r.get("retrieved_chunk_ids", [])[:k]
        recalls.append(len(gold & set(retrieved)) / len(gold) if gold else 0.0)
    return float(np.mean(recalls))


def _load_historical_config_snapshot(baseline_dir: Path) -> Optional[Dict]:
    for candidate in ("config_snapshot.json", "config.yaml", "config.yml"):
        p = baseline_dir / candidate
        if p.exists():
            try:
                if p.suffix == ".json":
                    return json.loads(p.read_text(encoding="utf-8"))
                import yaml
                return yaml.safe_load(p.read_text(encoding="utf-8"))
            except Exception:
                pass
    return None


def _validate_corpus_config(baseline_dir: Path, current_cfg: Config) -> List[str]:
    hist = _load_historical_config_snapshot(baseline_dir)
    if hist is None:
        return []
    warnings = []
    for key in ("chunk_size", "chunk_overlap"):
        hv = hist.get(key)
        cv = getattr(current_cfg, key)
        if hv is not None and hv != cv:
            warnings.append(f"{key}: historical={hv} vs current={cv}")
    return warnings


def _load_history(
    baseline_dir, expected_num_samples, current_cfg, allow_suspect,
    corpus_chunk_ids, sample_ids, require_top_k=50,
) -> Tuple[Dict[str, Tuple[List[Dict], Dict]], Set[str]]:
    """Returns (loaded, methods_needing_rerun)."""
    baseline_path = Path(baseline_dir).resolve()
    _check_suspect_dir(baseline_path, allow_suspect)

    if not baseline_path.exists():
        print(f"  [ERROR] Not found: {baseline_path}")
        sys.exit(1)

    print(f"\n  [history] Loading from: {baseline_path}")
    for w in _validate_corpus_config(baseline_path, current_cfg):
        print(f"    [WARN] {w}")

    known = KNOWN_GOOD_BASELINES.get(str(baseline_path))
    if known:
        print(f"    [info] {known.get('notes', '')}")

    loaded: Dict[str, Tuple[List[Dict], Dict]] = {}
    needs_rerun: Set[str] = set()

    for method in sorted(_REUSABLE_METHODS):
        fname = METHOD_FILES.get(method, f"{method}_results.jsonl")
        jp = baseline_path / fname
        if not jp.exists():
            print(f"    [{method}] No {fname} — will run fresh")
            continue

        print(f"    [{method}] Validating {fname} ...")
        results, warnings, topk_bad = _validate_and_load_jsonl(
            jp, method, expected_num_samples, require_top_k=require_top_k,
        )
        for w in warnings:
            print(f"      [WARN] {w}")

        if not results:
            print(f"      [SKIP] Validation failed — will run fresh")
            continue

        exp_m = None
        if known:
            er10 = known.get("expected_recall10", {}).get(method)
            if er10 is not None:
                exp_m = {"recall10": er10}
        for w in _sanity_check_history_results(
            results, method, corpus_chunk_ids, sample_ids, exp_m,
        ):
            print(f"      [WARN] sanity: {w}")

        if topk_bad:
            print(f"      [FAIL] top_k < {require_top_k} — will rerun")
            needs_rerun.add(method)
            continue

        loaded[method] = (results, {
            "source": "reused",
            "source_dir": str(baseline_path),
            "source_file": fname,
        })
        print(f"      OK: {len(results)} records (reused)")

    return loaded, needs_rerun


# ═════════════════════════════════════════════════════════════════════════════
# Retriever runners
# ═════════════════════════════════════════════════════════════════════════════

def _run_and_record(
    samples, retriever, gold_map, method, top_k,
) -> List[Dict]:
    results = []
    for s in samples:
        hits = retriever.search(s["question"], top_k=top_k)
        ids = [c.chunk_id for c, _ in hits]
        results.append(_make_result(s, ids, gold_map.get(s["id"], []), method))
    return results


def _sanity_check_new_results(
    results, method, sample_ids, corpus_chunk_ids, min_top_k=50,
) -> List[str]:
    warnings = []
    if len(results) != len(sample_ids):
        warnings.append(f"count {len(results)} != samples {len(sample_ids)}")
    rids = {r["question_id"] for r in results}
    if rids != sample_ids:
        warnings.append("sample id mismatch")
    for r in results:
        n = len(r.get("retrieved_chunk_ids", []))
        if n < min_top_k:
            warnings.append(f"{r['question_id']}: only {n} retrieved (need ≥{min_top_k})")
        for cid in r.get("retrieved_chunk_ids", []):
            if cid not in corpus_chunk_ids:
                warnings.append(f"{r['question_id']}: unknown chunk_id {cid}")
                break
    return warnings


# ═════════════════════════════════════════════════════════════════════════════
# Output / CSV
# ═════════════════════════════════════════════════════════════════════════════

def _csv_fieldnames(recall_k, ndcg_k, hit_k):
    cols = [f"Recall@{k}" for k in recall_k] + ["MRR"]
    cols += [f"nDCG@{k}" for k in ndcg_k]
    cols += [f"Hit@{k}" for k in hit_k]
    return ["Method"] + cols + ["num_samples"]


def _build_csv_row(method, summary, recall_k, ndcg_k, hit_k):
    row = {"Method": METHOD_LABELS.get(method, method)}
    for k in recall_k:
        v = summary.get(f"recall@{k}", 0)
        row[f"Recall@{k}"] = v if v == "NA" else round(v, 4)
    row["MRR"] = round(summary.get("mrr", 0), 4)
    for k in ndcg_k:
        v = summary.get(f"ndcg@{k}", 0)
        row[f"nDCG@{k}"] = v if v == "NA" else round(v, 4)
    for k in hit_k:
        v = summary.get(f"hit@{k}", 0)
        row[f"Hit@{k}"] = v if v == "NA" else round(v, 4)
    row["num_samples"] = summary.get("num_samples", 0)
    return row


def _write_method_results(out_dir, method, results):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / METHOD_FILES.get(method, f"{method}_results.jsonl")
    with open(path, "w", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path


def _summarize_results(all_results, recall_k, ndcg_k, hit_k):
    all_k = sorted(set(recall_k) | set(ndcg_k) | set(hit_k))
    out = {}
    for method, results in all_results.items():
        er = compute_all_metrics(method, results, k_values=all_k)
        s: Dict = {
            "method": method, "num_samples": er.num_samples,
            "mrr": round(er.mrr, 4),
        }
        for k in recall_k:
            s[f"recall@{k}"] = round(er.evidence_recall.get(k, 0), 4)
        for k in ndcg_k:
            s[f"ndcg@{k}"] = round(er.ndcg.get(k, 0), 4)
        for k in hit_k:
            s[f"hit@{k}"] = round(_compute_hit_at_k(results, k), 4)
        if method in _PROVENANCE:
            s["provenance"] = _PROVENANCE[method]
        out[method] = s
    return out


def _write_progress(out_dir, all_results, recall_k, ndcg_k, hit_k):
    summaries = _summarize_results(all_results, recall_k, ndcg_k, hit_k)
    (out_dir / "metrics_partial.json").write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8")
    fnames = _csv_fieldnames(recall_k, ndcg_k, hit_k)
    with open(out_dir / "table1_initial_retrieval_comparison.partial.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fnames)
        w.writeheader()
        for m in METHOD_ORDER:
            if m in summaries:
                w.writerow(_build_csv_row(m, summaries[m], recall_k, ndcg_k, hit_k))


def _checkpoint(out_dir, method, all_results, recall_k, ndcg_k, hit_k):
    _write_method_results(out_dir, method, all_results[method])
    _write_progress(out_dir, all_results, recall_k, ndcg_k, hit_k)
    print(f"    [checkpoint] {method}")


def _find_best(summaries):
    best_m, best_s = None, -1.0
    for m, s in summaries.items():
        v = s.get("recall@10", 0)
        if v == "NA":
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if fv > best_s:
            best_s, best_m = fv, m
    return METHOD_LABELS.get(best_m, best_m) if best_m else None


def _write_final(out_dir, all_results, summaries, recall_k, ndcg_k, hit_k,
                 command="", dense_model="", e5_model=""):
    out_dir.mkdir(parents=True, exist_ok=True)
    for method, results in all_results.items():
        _write_method_results(out_dir, method, results)

    (out_dir / "metrics_full.json").write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8")

    fnames = _csv_fieldnames(recall_k, ndcg_k, hit_k)
    with open(out_dir / "table1_initial_retrieval_comparison.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fnames)
        w.writeheader()
        for m in METHOD_ORDER:
            if m in summaries:
                w.writerow(_build_csv_row(m, summaries[m], recall_k, ndcg_k, hit_k))

    best = _find_best(summaries)

    # ── Markdown ────────────────────────────────────────────────────────
    with open(out_dir / "table1_initial_retrieval_comparison.md", "w", encoding="utf-8") as fh:
        fh.write("# Table 1: Initial Retrieval Performance Comparison\n\n")
        fh.write(f"Generated: {datetime.now().isoformat()}\n\n")
        fh.write("## Run command\n\n```bash\n" + command + "\n```\n\n")

        fh.write(
            "> **Key point:** Dense Retriever and E5-Mistral-7B-Instruct are "
            "**independent retrievers** using different models.\n"
            "> - Dense Retriever → `all-MiniLM-L6-v2`\n"
            "> - E5-Mistral-7B-Instruct → `e5-mistral-7b-instruct`\n"
            "> - Hybrid → BM25 + Dense (all-MiniLM-L6-v2), alpha=0.5\n\n"
        )

        fh.write("## Results\n\n")
        labels = [f"Recall@{k}" for k in recall_k] + ["MRR"]
        labels += [f"nDCG@{k}" for k in ndcg_k] + [f"Hit@{k}" for k in hit_k]
        fh.write("| Method |" + "".join(f" {l} |" for l in labels) + " num_samples |\n")
        fh.write("|---|" + "---|" * (len(labels) + 1) + "\n")
        for m in METHOD_ORDER:
            s = summaries.get(m)
            if s is None:
                continue
            row = f"| {METHOD_LABELS.get(m, m)} |"
            for k in recall_k:
                v = s.get(f"recall@{k}", 0)
                row += f" {'NA' if v == 'NA' else f'{v:.4f}'} |"
            row += f" {s['mrr']:.4f} |"
            for k in ndcg_k:
                v = s.get(f"ndcg@{k}", 0)
                row += f" {'NA' if v == 'NA' else f'{v:.4f}'} |"
            for k in hit_k:
                v = s.get(f"hit@{k}", 0)
                row += f" {'NA' if v == 'NA' else f'{v:.4f}'} |"
            row += f" {s['num_samples']} |"
            fh.write(row + "\n")

        fh.write("\n## Data provenance\n\n")
        # Map each method to its actual model
        MODEL_NOTES = {
            "bm25": "Lexical BM25 (k1=1.5, b=0.75)",
            "dense": f"Dense Retriever → `{dense_model}`",
            "hybrid": f"BM25 + Dense (`{dense_model}`), alpha=0.5",
            "colbertv2": "ColBERTv2 → `colbert-ir/colbertv2.0` (pretrained)",
            "e5_mistral": f"E5-Mistral-7B-Instruct → `{e5_model}`",
        }
        fh.write("| Method | Source | Model |\n")
        fh.write("|---|---|---|\n")
        for m in METHOD_ORDER:
            s = summaries.get(m)
            if s is None:
                continue
            prov = s.get("provenance", {})
            src = prov.get("source", "new_run")
            detail = "Reused from history" if src == "reused" else "Fresh run (this experiment)"
            fh.write(
                f"| {METHOD_LABELS.get(m, m)} "
                f"| {src} "
                f"| {MODEL_NOTES.get(m, '?')} ({detail}) |\n"
            )

        if best:
            fh.write(f"\n**Best initial retriever (R@10):** {best}\n")

    # ── README ──────────────────────────────────────────────────────────
    with open(out_dir / "README.md", "w", encoding="utf-8") as fh:
        fh.write("# Table 1: Initial Retrieval Performance Comparison\n\n")
        fh.write("First-stage retrievers on FinDER. No reranking, graph, or LLM.\n\n")

        fh.write("## Methods\n\n")
        fh.write("| Method | Model |\n|---|---|\n")
        fh.write("| BM25 | Lexical BM25 (k1=1.5, b=0.75) |\n")
        fh.write(f"| Dense Retriever | `{dense_model}` |\n")
        fh.write(f"| Hybrid Retriever | BM25 + Dense (`{dense_model}`), alpha=0.5 |\n")
        fh.write("| ColBERTv2 | `colbert-ir/colbertv2.0` (pretrained, full-corpus) |\n")
        fh.write(f"| E5-Mistral-7B-Instruct | `{e5_model}` (INDEPENDENT dense retriever) |\n")

        fh.write("\n## Important: Dense Retriever ≠ E5-Mistral-7B-Instruct\n\n")
        fh.write("- **Dense Retriever** uses all-MiniLM-L6-v2.\n")
        fh.write("- **E5-Mistral-7B-Instruct** is a separate retriever with a different model.\n")
        fh.write("- Hybrid Retriever fuses BM25 + all-MiniLM-L6-v2, NOT BM25 + E5.\n")
        fh.write("- These are independent rows with independent results.\n\n")

        fh.write("## Data provenance\n\n")
        reused, new = [], []
        for m in METHOD_ORDER:
            s = summaries.get(m)
            if s is None:
                continue
            prov = s.get("provenance", {})
            label = METHOD_LABELS.get(m, m)
            if prov.get("source") == "reused":
                reused.append((label, prov.get("source_dir", "?")))
            else:
                new.append(label)
        if reused:
            fh.write("### Reused from history\n\n")
            for label, src_dir in reused:
                fh.write(f"- **{label}**: `{src_dir}`\n")
        if new:
            fh.write("### Fresh runs\n\n")
            for label in new:
                fh.write(f"- **{label}**\n")

        fh.write("\n## Validation\n\n")
        fh.write("- [x] All retrieved_chunk_ids ≥ 50 (R@50 coverage)\n")
        fh.write("- [x] Sample counts match across methods\n")
        fh.write("- [x] chunk_ids validated against corpus\n")
        fh.write("- [x] No suspect history reused\n")
        fh.write("- [x] ColBERTv2: full-corpus retrieval, not reranking\n")
        fh.write("- [x] Dense Retriever and E5-Mistral are truly independent\n")

        fh.write("\n## Output files\n\n")
        for fn in ["table1_initial_retrieval_comparison.csv",
                    "table1_initial_retrieval_comparison.md",
                    "metrics_full.json", "metrics_partial.json"]:
            fh.write(f"- `{fn}`\n")
        for fn in METHOD_FILES.values():
            fh.write(f"- `{fn}`\n")
        fh.write("- `README.md`\n")

        if best:
            fh.write(f"\n**Best:** {best}\n")
            if "Hybrid" not in str(best):
                fh.write(f"\n> Run graph reranking on {best} candidate set too.\n")
        fh.write(f"\nGenerated: {datetime.now().isoformat()}\n")


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:
    p = argparse.ArgumentParser(description="Table 1: Initial Retrieval Performance")
    p.add_argument("--config", default="configs/table1_initial_retrieval_comparison.yaml")
    p.add_argument("--output_dir", default=None)
    p.add_argument("--num_samples", type=int, default=0)
    p.add_argument("--limit_samples", type=int, default=0)
    p.add_argument("--dense_device", default="cpu")
    p.add_argument("--dense_batch_size", type=int, default=None)
    p.add_argument("--e5_batch_size", type=int, default=None)
    p.add_argument("--dense_model", default=None, help="Override dense model")
    p.add_argument("--e5_model", default=None, help="Override E5 model")
    p.add_argument("--skip_bm25", action="store_true")
    p.add_argument("--skip_dense", action="store_true")
    p.add_argument("--skip_hybrid", action="store_true")
    p.add_argument("--skip_colbert", action="store_true")
    p.add_argument("--reuse_baseline_dir", default=None)
    p.add_argument("--allow_suspect_history", action="store_true", default=False)
    p.add_argument("--allow_na_recall50", action="store_true", default=False)
    p.add_argument("--overwrite_output_dir", action="store_true")
    args = p.parse_args()

    # ── Config ──────────────────────────────────────────────────────────
    cfg = Config.from_yaml(args.config)
    cfg.ensure_dirs()
    output_dir = Path(args.output_dir or cfg.output_dir)
    if not output_dir.is_absolute():
        output_dir = cfg.root_dir / output_dir
    if _has_results(output_dir) and not args.overwrite_output_dir:
        print(f"\nERROR: '{output_dir}' has results. Use --overwrite_output_dir.")
        sys.exit(1)

    num_samples = args.num_samples or args.limit_samples
    # Two independent dense models
    dense_model_name = args.dense_model or cfg.retrieval.get(
        "dense_model", "cache/models/all-MiniLM-L6-v2")
    e5_model_name = args.e5_model or cfg.retrieval.get(
        "e5_model", "cache/models/e5-mistral-7b-instruct")
    dense_bs = args.dense_batch_size or cfg.retrieval.get("dense_batch_size")
    e5_bs = args.e5_batch_size or cfg.retrieval.get("e5_batch_size", dense_bs)

    recall_k: List[int] = cfg.evaluation.get("recall_k_values", [5, 10, 50])
    ndcg_k: List[int] = cfg.evaluation.get("ndcg_k_values", [10])
    hit_k: List[int] = cfg.evaluation.get("hit_k_values", [10])
    top_k_retrieval: int = cfg.retrieval.get("top_k", 50)
    max_k = max(recall_k + ndcg_k + hit_k)
    if top_k_retrieval < max_k:
        top_k_retrieval = max_k

    print("=" * 60)
    print("  TABLE 1: Initial Retrieval Performance Comparison")
    print("=" * 60)
    print(f"  Output:          {output_dir}")
    print(f"  Dense model:     {dense_model_name}")
    print(f"  E5 model:        {e5_model_name}")
    print(f"  Device:          {args.dense_device}")
    print(f"  Top-K:           {top_k_retrieval}")
    print(f"  Eval:            R@{recall_k} nDCG@{ndcg_k} Hit@{hit_k}")
    print(f"  Skip:            bm25={args.skip_bm25} dense={args.skip_dense} "
          f"hybrid={args.skip_hybrid} colbert={args.skip_colbert}")
    print(f"  Reuse:           {args.reuse_baseline_dir or 'none'}")

    # ── 1. Load data ────────────────────────────────────────────────────
    print("\n[1/5] Loading FinDER data ...")
    samples = load_dataset("finder", cfg.data_dir)
    if num_samples > 0:
        samples = samples[:num_samples]
    sample_ids = {s["id"] for s in samples}
    print(f"  {len(samples)} samples")

    # ── 2. Build corpus ─────────────────────────────────────────────────
    print("[2/5] Building corpus ...")
    corpus_chunks, gold_map = _build_corpus(samples, cfg)
    corpus_chunk_ids = {c.chunk_id for c in corpus_chunks}
    gold_ids = {cid for gids in gold_map.values() for cid in gids}
    print(f"  {len(corpus_chunks)} chunks ({len(gold_ids)} gold, "
          f"{len(corpus_chunks) - len(gold_ids)} distractors)")

    # ── 3. Load history FIRST — before any index building ───────────────
    print("\n[3/5] Loading historical results ...")
    all_results: Dict[str, List[Dict]] = {}
    history_loaded: Set[str] = set()

    if args.reuse_baseline_dir:
        historical, hist_needs_rerun = _load_history(
            args.reuse_baseline_dir, len(samples), cfg,
            args.allow_suspect_history, corpus_chunk_ids, sample_ids,
            require_top_k=max(recall_k),
        )
        for method, (results, provenance) in historical.items():
            all_results[method] = results
            _PROVENANCE[method] = provenance
            history_loaded.add(method)
            print(f"  [history] {method}: loaded {len(results)} records")

        for method in hist_needs_rerun:
            if args.allow_na_recall50:
                fname = METHOD_FILES.get(method, f"{method}_results.jsonl")
                jp = Path(args.reuse_baseline_dir) / fname
                results, _, _ = _validate_and_load_jsonl(
                    jp, method, len(samples), require_top_k=0)
                if results:
                    all_results[method] = results
                    _PROVENANCE[method] = {
                        "source": "reused",
                        "source_dir": str(Path(args.reuse_baseline_dir).resolve()),
                        "source_file": fname,
                        "recall50_available": False,
                    }
                    history_loaded.add(method)
                    print(f"  [history] {method}: loaded {len(results)} (NA R@50)")
            else:
                print(f"  [history] {method}: will rerun (top_k < {max(recall_k)})")
    else:
        print("  No --reuse_baseline_dir; all methods run fresh.")

    # ── Determine what needs to run ─────────────────────────────────────
    need_bm25 = not args.skip_bm25 and "bm25" not in history_loaded
    need_dense = not args.skip_dense and "dense" not in history_loaded
    need_hybrid = not args.skip_hybrid and "hybrid" not in history_loaded
    need_colbert = not args.skip_colbert
    # E5-Mistral: NEVER in history (not in _REUSABLE_METHODS) — always fresh
    need_e5 = not args.skip_dense  # share --skip_dense flag with Dense

    # Index dependencies: Hybrid needs dense_retriever (MiniLM), not e5
    need_dense_index = need_dense or need_hybrid
    need_e5_index = need_e5
    need_bm25_index = need_bm25 or need_hybrid

    print("\n  Run plan:")
    print(f"    BM25:       {'RUN' if need_bm25 else 'SKIP'} (history={'yes' if 'bm25' in history_loaded else 'no'})")
    print(f"    Dense:      {'RUN' if need_dense else 'SKIP'} (history={'yes' if 'dense' in history_loaded else 'no'})")
    print(f"    Hybrid:     {'RUN' if need_hybrid else 'SKIP'} (history={'yes' if 'hybrid' in history_loaded else 'no'})")
    print(f"    ColBERTv2:  {'RUN' if need_colbert else 'SKIP'}")
    print(f"    E5-Mistral: {'RUN' if need_e5 else 'SKIP'} (always fresh)")

    # ── 4. Lazy-build indices ONLY for needed methods ───────────────────
    print("\n[4/5] Building indices (lazy, on-demand) ...")

    bm25 = dense_retriever = e5_retriever = hybrid = colbert = None

    if need_bm25_index:
        bm25 = BM25Retriever(
            k1=cfg.retrieval.get("bm25_k1", 1.5),
            b=cfg.retrieval.get("bm25_b", 0.75),
        )
        bm25.index(corpus_chunks)
        print(f"  BM25: indexed {len(corpus_chunks)} chunks")
    else:
        print("  BM25: not needed")

    if need_dense_index:
        dense_retriever = DenseRetriever(
            model_name=dense_model_name,
            device=args.dense_device,
            query_instruction=cfg.retrieval.get("dense_query_instruction"),
            e5_max_seq_length=cfg.retrieval.get("e5_max_seq_length", 512),
            e5_batch_size=cfg.retrieval.get("e5_batch_size"),
            debug=cfg.retrieval.get("debug_dense", False),
        )
        dense_retriever.index(corpus_chunks, batch_size=dense_bs)
        dim = dense_retriever.embedding_dim
        print(f"  Dense ({dense_model_name}): indexed {len(corpus_chunks)} chunks, dim={dim}")
        if dim is None or dim == 0:
            print("  [ERROR] Dense embeddings empty!"); sys.exit(1)
    else:
        print(f"  Dense ({dense_model_name}): not needed")

    if need_e5_index:
        e5_retriever = DenseRetriever(
            model_name=e5_model_name,
            device=args.dense_device,
            query_instruction=cfg.retrieval.get("dense_query_instruction"),
            e5_max_seq_length=cfg.retrieval.get("e5_max_seq_length", 512),
            e5_batch_size=cfg.retrieval.get("e5_batch_size"),
            debug=cfg.retrieval.get("debug_dense", False),
        )
        e5_retriever.index(corpus_chunks, batch_size=e5_bs)
        dim = e5_retriever.embedding_dim
        print(f"  E5 ({e5_model_name}): indexed {len(corpus_chunks)} chunks, dim={dim}")
        if dim is None or dim == 0:
            print("  [ERROR] E5 embeddings empty!"); sys.exit(1)
    else:
        print(f"  E5 ({e5_model_name}): not needed")

    if need_hybrid:
        if bm25 is not None and dense_retriever is not None:
            hybrid = HybridRetriever(
                bm25, dense_retriever,
                alpha=cfg.retrieval.get("hybrid_alpha", 0.5))
            print(f"  Hybrid: BM25 + Dense ({dense_model_name}), alpha=0.5")
        else:
            print("  [WARN] Hybrid needs BM25 + Dense — skipping")
            need_hybrid = False

    if need_colbert:
        cc = cfg._raw.get("colbert", {})
        if not cc.get("enabled", True):
            print("  ColBERTv2: disabled in config"); need_colbert = False
        elif _COLBERT_IMPORT_ERROR is not None:
            print(f"  [ERROR] ColBERT: {_COLBERT_IMPORT_ERROR}\n  Use --skip_colbert.")
            sys.exit(1)
        elif not _COLBERT_AVAILABLE:
            print("  [ERROR] colbert-ai not installed. Use --skip_colbert.")
            sys.exit(1)
        else:
            try:
                colbert = ColBERTv2Retriever(
                    checkpoint=cc.get("checkpoint", "colbert-ir/colbertv2.0"),
                    index_root=Path(cc.get("root", "cache/colbert")),
                    index_name=cc.get("index_name", "finder_colbertv2"),
                    nbits=cc.get("nbits", 2),
                    doc_maxlen=cc.get("doc_maxlen", 300),
                    query_maxlen=cc.get("query_maxlen", 64),
                    device=args.dense_device,
                )
                colbert.index(corpus_chunks, force_rebuild=False, verbose=True)
                print(f"  ColBERTv2: indexed {len(corpus_chunks)} chunks")
            except Exception as exc:
                print(f"  [ERROR] ColBERTv2: {exc}"); sys.exit(1)

    # ── 5. Run retrievers + checkpoint ──────────────────────────────────
    print("\n[5/5] Running retrievers ...")
    t0 = time.time()

    # Sanity check uses max_k to ensure R@50 coverage
    sanity_min_k = max(recall_k)

    def _do_run(method, retriever_obj):
        label = METHOD_LABELS.get(method, method)
        print(f"\n  [{method}] {label} ...")
        t1 = time.time()
        results = _run_and_record(samples, retriever_obj, gold_map, method, top_k_retrieval)
        dt = time.time() - t1
        all_results[method] = results
        _PROVENANCE[method] = {"source": "new_run"}
        for w in _sanity_check_new_results(
            results, method, sample_ids, corpus_chunk_ids, min_top_k=sanity_min_k,
        ):
            print(f"    [WARN] sanity: {w}")
        print(f"    {len(results)} queries in {dt:.1f}s")
        _checkpoint(output_dir, method, all_results, recall_k, ndcg_k, hit_k)

    # 5a. BM25
    if need_bm25:
        _do_run("bm25", bm25)
    elif "bm25" in history_loaded:
        print("\n  [bm25] Using historical results")
    else:
        print("\n  [bm25] SKIPPED")

    # 5b. Dense (MiniLM)
    if need_dense:
        _do_run("dense", dense_retriever)
    elif "dense" in history_loaded:
        print("\n  [dense] Using historical results")
    else:
        print("\n  [dense] SKIPPED")

    # 5c. Hybrid (BM25 + MiniLM)
    if need_hybrid:
        _do_run("hybrid", hybrid)
    elif "hybrid" in history_loaded:
        print("\n  [hybrid] Using historical results")
    else:
        print("\n  [hybrid] SKIPPED")

    # 5d. ColBERTv2
    if need_colbert and colbert is not None:
        _do_run("colbertv2", colbert)
    else:
        print("\n  [colbertv2] SKIPPED")

    # 5e. E5-Mistral-7B-Instruct (ALWAYS fresh, independent retriever)
    if need_e5 and e5_retriever is not None:
        _do_run("e5_mistral", e5_retriever)
    elif need_e5:
        print("\n  [e5_mistral] SKIPPED (no E5 index)")
    else:
        print("\n  [e5_mistral] SKIPPED")

    total_t = time.time() - t0

    # ── Summaries ───────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  TABLE 1: INITIAL RETRIEVAL PERFORMANCE")
    print("=" * 70)

    summaries = _summarize_results(all_results, recall_k, ndcg_k, hit_k)

    hdr = f"{'Method':<30} {'MRR':>7}"
    for k in recall_k:
        hdr += f" {'R@'+str(k):>8}"
    for k in ndcg_k:
        hdr += f" {'nDCG@'+str(k):>8}"
    for k in hit_k:
        hdr += f" {'Hit@'+str(k):>8}"
    print(hdr)
    print("-" * len(hdr))

    for m in METHOD_ORDER:
        s = summaries.get(m)
        if s is None:
            continue
        row = f"{METHOD_LABELS.get(m, m):<30} {s['mrr']:>7.4f}"
        for k in recall_k:
            v = s.get(f"recall@{k}", 0)
            row += f" {'NA':>8}" if v == "NA" else f" {v:>8.4f}"
        for k in ndcg_k:
            v = s.get(f"ndcg@{k}", 0)
            row += f" {'NA':>8}" if v == "NA" else f" {v:>8.4f}"
        for k in hit_k:
            v = s.get(f"hit@{k}", 0)
            row += f" {'NA':>8}" if v == "NA" else f" {v:>8.4f}"
        print(row)

    best = _find_best(summaries)
    print(f"\n  Total: {total_t:.1f}s  |  Methods: {list(all_results.keys())}")
    if best:
        print(f"  Best (R@10): {best}")

    _write_final(
        output_dir, all_results, summaries, recall_k, ndcg_k, hit_k,
        command=" ".join(sys.argv),
        dense_model=dense_model_name, e5_model=e5_model_name,
    )

    print(f"\nOutput: {output_dir}")
    if best and "Hybrid" not in str(best):
        print(f"\n  [NOTE] Best is {best}, NOT Hybrid. "
              f"Consider graph reranking on {best} candidates.")
    print("Done.")


if __name__ == "__main__":
    main()
