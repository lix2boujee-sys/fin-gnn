"""Answer generation using evidence selected by different retrieval/reranking methods.

Fixes a single generator and varies only the evidence source to isolate the
effect of evidence quality on downstream answer generation.

Evidence sources compared:
    1. Initial Retriever     (Hybrid BM25+Dense)
    2. Cross-Encoder         (CE re-ranker)
    3. MonoT5                (T5 pointwise re-ranker)
    4. PPR                   (Personalized PageRank on Financial Evidence Graph)
    5. R-GCN                 (Relational GCN re-ranker)
    6. GATv2                 (Graph Attention Network v2 re-ranker)
    7. FinDual-GNN (Ours)    (Dual-constraint fusion GNN)

Metrics: Accuracy, Faithfulness, Numerical Consistency, Unsupported Rate,
Evidence Hit/Recall/MRR, Token F1, Abstention handling.

Usage:
    # Smoke test (20 random queries)
    python experiments/generation_with_selected_evidence.py \\
        --num_queries 20 --sample_mode random --sample_seed 42 \\
        --method_result "Initial Retriever=/path/to/bge_m3_dense_results.jsonl"

    # Full run with resume
    python experiments/generation_with_selected_evidence.py --resume

    # Paper mode (strict alignment checks)
    python experiments/generation_with_selected_evidence.py \\
        --paper_mode --min_gold_alignment_coverage 0.95

    # Evaluate only (skip generation)
    python experiments/generation_with_selected_evidence.py --eval_only
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pickle
import random
import re
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from feg_rag.config import Config
from feg_rag.data.chunker import Chunk, chunk_text, chunk_report
from feg_rag.data.loader import load_dataset
from feg_rag.generation.evidence_schema import (
    RankedEvidence,
    convert_rich_format,
    convert_simple_format,
    convert_from_file,
    deduplicate_evidence,
    truncate_by_token_budget,
    extract_hash_from_chunk_id,
    register_chunk_meta,
)


# =============================================================================
# Constants
# =============================================================================

CANONICAL_METHODS = [
    "Initial Retriever",
    "Cross-Encoder",
    "MonoT5",
    "PPR",
    "R-GCN",
    "GATv2",
    "FinDual-GNN (Ours)",
]

METHOD_KEYS = [
    "initial_retriever",
    "cross_encoder",
    "monot5",
    "ppr",
    "rgcn",
    "gatv2",
    "findual_gnn",
]

METHOD_LABELS = {
    "initial_retriever": "Initial Retriever",
    "cross_encoder": "Cross-Encoder",
    "monot5": "MonoT5",
    "ppr": "PPR",
    "rgcn": "R-GCN",
    "gatv2": "GATv2",
    "findual_gnn": "FinDual-GNN (Ours)",
}

# Legacy glob-based method file map (used only when --method_result is not provided).
METHOD_FILE_MAP: Dict[str, List[Dict]] = {
    "Initial Retriever": [
        {"dir_glob": "outputs/v2_table1_bge_m3_correct_corpus_*", "file": "bge_m3_dense_results.jsonl", "format": "simple"},
        {"dir": "outputs/exp1_baseline", "file": "hybrid_results.jsonl", "format": "rich"},
        {"dir": "outputs/exp3_feg_ppr", "file": "hybrid_results.jsonl", "format": "simple"},
    ],
    "Cross-Encoder": [
        {"dir_glob": "outputs/v2_table2_cross_encoder_*", "file": "cross_encoder_results.jsonl", "format": "simple"},
        {"dir": "outputs/table1_non_llm_reranking", "file": "cross_encoder_results.jsonl", "format": "simple"},
    ],
    "MonoT5": [
        {"dir_glob": "outputs/v2_table2_mono_t5_bge_pool_*", "file": "mono_t5_results.jsonl", "format": "simple"},
        {"dir": "outputs/table1_non_llm_reranking", "file": "monot5_results.jsonl", "format": "simple"},
    ],
    "PPR": [
        {"dir_glob": "outputs/v2_table2_graph_bge_pool_a_ppr_sage_*", "file": "ppr_results.jsonl", "format": "simple"},
        {"dir": "outputs/exp4_gnn_fulltest", "file": "hybrid_ppr_results.jsonl", "format": "simple"},
        {"dir": "outputs/exp3_feg_ppr", "file": "ppr_results_full_graph.jsonl", "format": "simple"},
    ],
    "R-GCN": [
        {"dir_glob": "outputs/v2_table2_graph_bge_pool_rgcn_eval_fast_*", "file": "rgcn_results.jsonl", "format": "simple"},
        {"dir": "outputs/table1_non_llm_reranking", "file": "rgcn_results.jsonl", "format": "simple"},
        {"dir": "outputs/exp4_gnn_fulltest", "file": "hybrid_rgcn_results.jsonl", "format": "simple"},
    ],
    "GATv2": [
        {"dir_glob": "outputs/v2_table2_gatv2_*", "file": "gatv2_results.jsonl", "format": "simple"},
        {"dir": "outputs/table1_non_llm_reranking", "file": "gat_results.jsonl", "format": "simple"},
        {"dir": "outputs/exp4_gnn_fulltest", "file": "hybrid_gat_results.jsonl", "format": "simple"},
    ],
    "FinDual-GNN (Ours)": [
        {"dir_glob": "outputs/v2_table2_dcf_gnn_*", "file": "dcf_gnn_results.jsonl", "format": "simple"},
        {"dir_glob": "outputs/v2_table2_c2_dcf_gnn_*", "file": "c2_dcf_gnn_results.jsonl", "format": "simple"},
        {"dir": "outputs/table1_non_llm_reranking", "file": "dcf_gnn_results.jsonl", "format": "simple"},
        {"dir": "outputs/exp4_gnn_fulltest", "file": "hybrid_dcf_results.jsonl", "format": "simple"},
    ],
}

GENERATION_PROMPT_TEMPLATE = (
    "You are a financial question answering assistant.\n"
    "Answer the question using only the provided evidence.\n"
    "If the evidence is insufficient, answer \"insufficient evidence\".\n"
    "Do not use outside knowledge.\n"
    "Keep numerical values, units, company names, financial metrics, "
    "and fiscal years exactly consistent with the evidence.\n"
    "If calculation is required, show the calculation briefly.\n"
    "\n"
    "Evidence:\n"
    "{evidence_passages}\n"
    "\n"
    "Question:\n"
    "{query}\n"
    "\n"
    "Answer:"
)

DEFAULT_GENERATION_CONFIG = {
    "top_k_evidence": 5,
    "temperature": 0.0,
    "do_sample": False,
    "max_new_tokens": 128,
}

RECORD_SCHEMA_VERSION = "2.0"
PROMPT_VERSION = "financial_grounded_v2"
EVALUATOR_VERSION = "answer_evaluator_v2"
MANIFEST_SCHEMA_VERSION = "generation_eval_v2"

# Regex patterns
_NUMBER_RE = re.compile(
    r"\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b"
    r"|\b\d+\.?\d*\s*(?:million|billion|thousand|trillion)\b"
    r"|\b\d+\.?\d*\s*%"
    r"|\$\s*\d{1,3}(?:,\d{3})*(?:\.\d+)?\b",
    re.IGNORECASE,
)
_UNIT_RE = re.compile(
    r"\b(million|billion|thousand|trillion|percent|%|dollars?\$?)\b",
    re.IGNORECASE,
)
_YEAR_RE = re.compile(r"\b((?:19|20)\d{2})\b")
_PERCENTAGE_RE = re.compile(r"\b\d+\.?\d*\s*%")
_INSUFFICIENT_RE = re.compile(r"insufficient\s*evidence", re.IGNORECASE)
_INSUFFICIENT_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r"^\s*insufficient\s*evidence\s*\.?\s*$",
        r"^\s*insufficient\s*evidence\s*[.!]?\s*$",
        r"^\s*insufficient\s+evidence\s*$",
    ]
]


# =============================================================================
# Run manifest (P0-4)
# =============================================================================

@dataclass
class RunManifest:
    """Immutable record of the experiment configuration for a generation run."""

    schema_version: str = MANIFEST_SCHEMA_VERSION
    dataset_split: str = "test"
    num_split_queries: int = 0
    split_mode: str = "current_loader_default"
    corpus_cache_sha256: str = ""
    selected_query_ids_sha256: str = ""
    generator_provider: str = "local"
    generator_model: str = ""
    generator_parameters: Dict[str, Any] = field(default_factory=dict)
    prompt_version: str = PROMPT_VERSION
    prompt_sha256: str = ""
    top_k_evidence: int = 5
    max_input_tokens: Optional[int] = None
    max_tokens_per_evidence: Optional[int] = None
    tokenizer_mode: str = "approx_char_based"
    method_files: Dict[str, str] = field(default_factory=dict)
    method_sources: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    evaluator_version: str = EVALUATOR_VERSION
    min_gold_alignment_coverage: float = 0.95
    paper_mode: bool = False
    sample_mode: str = "first"
    sample_seed: int = 42
    num_queries: int = 0
    candidate_top_n: int = 50
    candidate_file_sha256: str = ""
    initial_retriever: str = ""
    created_at: str = ""

    def to_dict(self) -> Dict:
        return {
            "schema_version": self.schema_version,
            "dataset_split": self.dataset_split,
            "num_split_queries": self.num_split_queries,
            "split_mode": self.split_mode,
            "corpus_cache_sha256": self.corpus_cache_sha256,
            "selected_query_ids_sha256": self.selected_query_ids_sha256,
            "generator_provider": self.generator_provider,
            "generator_model": self.generator_model,
            "generator_parameters": self.generator_parameters,
            "prompt_version": self.prompt_version,
            "prompt_sha256": self.prompt_sha256,
            "top_k_evidence": self.top_k_evidence,
            "max_input_tokens": self.max_input_tokens,
            "max_tokens_per_evidence": self.max_tokens_per_evidence,
            "tokenizer_mode": self.tokenizer_mode,
            "method_files": self.method_files,
            "method_sources": self.method_sources,
            "evaluator_version": self.evaluator_version,
            "min_gold_alignment_coverage": self.min_gold_alignment_coverage,
            "paper_mode": self.paper_mode,
            "sample_mode": self.sample_mode,
            "sample_seed": self.sample_seed,
            "num_queries": self.num_queries,
            "candidate_top_n": self.candidate_top_n,
            "candidate_file_sha256": self.candidate_file_sha256,
            "initial_retriever": self.initial_retriever,
            "created_at": self.created_at or datetime.now().isoformat(),
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "RunManifest":
        return cls(
            schema_version=d.get("schema_version", MANIFEST_SCHEMA_VERSION),
            dataset_split=d.get("dataset_split", "test"),
            num_split_queries=d.get("num_split_queries", 0),
            split_mode=d.get("split_mode", "current_loader_default"),
            corpus_cache_sha256=d.get("corpus_cache_sha256", ""),
            selected_query_ids_sha256=d.get("selected_query_ids_sha256", ""),
            generator_provider=d.get("generator_provider", "local"),
            generator_model=d.get("generator_model", ""),
            generator_parameters=d.get("generator_parameters", {}),
            prompt_version=d.get("prompt_version", PROMPT_VERSION),
            prompt_sha256=d.get("prompt_sha256", ""),
            top_k_evidence=d.get("top_k_evidence", 5),
            max_input_tokens=d.get("max_input_tokens"),
            max_tokens_per_evidence=d.get("max_tokens_per_evidence"),
            tokenizer_mode=d.get("tokenizer_mode", "approx_char_based"),
            method_files=d.get("method_files", {}),
            method_sources=d.get("method_sources", {}),
            evaluator_version=d.get("evaluator_version", EVALUATOR_VERSION),
            min_gold_alignment_coverage=d.get("min_gold_alignment_coverage", 0.95),
            paper_mode=d.get("paper_mode", False),
            sample_mode=d.get("sample_mode", "first"),
            sample_seed=d.get("sample_seed", 42),
            num_queries=d.get("num_queries", 0),
            candidate_top_n=d.get("candidate_top_n", 50),
            candidate_file_sha256=d.get("candidate_file_sha256", ""),
            initial_retriever=d.get("initial_retriever", ""),
            created_at=d.get("created_at", ""),
        )

    def critical_fields(self) -> Dict[str, Any]:
        """Return the subset of fields that must match for resume to be allowed."""
        return {
            "schema_version": self.schema_version,
            "dataset_split": self.dataset_split,
            "corpus_cache_sha256": self.corpus_cache_sha256,
            "selected_query_ids_sha256": self.selected_query_ids_sha256,
            "generator_provider": self.generator_provider,
            "generator_model": self.generator_model,
            "generator_parameters": self.generator_parameters,
            "prompt_sha256": self.prompt_sha256,
            "top_k_evidence": self.top_k_evidence,
            "method_files": self.method_files,
            "evaluator_version": self.evaluator_version,
        }

    def check_resume_compatible(self, other: "RunManifest") -> List[str]:
        """Compare critical fields; return list of mismatches (empty = compatible)."""
        mine = self.critical_fields()
        theirs = other.critical_fields()
        mismatches = []
        for key in mine:
            if mine[key] != theirs.get(key):
                mismatches.append(f"{key}: {mine[key]} != {theirs.get(key)}")
        return mismatches


# =============================================================================
# Hashing helpers
# =============================================================================

def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_sha256(filepath: Path) -> str:
    """Compute SHA-256 of a file's contents."""
    if not filepath or not filepath.exists():
        return ""
    return _sha256_hex(filepath.read_bytes())


def _str_sha256(text: str) -> str:
    return _sha256_hex(text.encode("utf-8"))


# =============================================================================
# Result file discovery (P0-3)
# =============================================================================

def find_method_file_legacy(canonical_name: str, root_dir: Path) -> Tuple[Optional[Path], str]:
    """Find the result file via legacy METHOD_FILE_MAP glob rules.

    P0-3 behavior:
      - 0 matching candidates: return (None, "")
      - 1 matching candidate:  return it
      - 2+ matching candidates: raise an error listing all of them

    Never silently picks the first/latest/newest match.
    """
    candidates = METHOD_FILE_MAP.get(canonical_name, [])
    matched: List[Path] = []

    for candidate in candidates:
        if "dir_glob" in candidate:
            dirs = sorted(
                (p for p in root_dir.glob(candidate["dir_glob"]) if p.is_dir()),
                key=lambda p: p.name,
            )
            for d in dirs:
                fpath = d / candidate["file"]
                if fpath.exists():
                    matched.append(fpath)
        else:
            fpath = root_dir / candidate["dir"] / candidate["file"]
            if fpath.exists():
                matched.append(fpath)

    if len(matched) == 0:
        return None, ""
    if len(matched) == 1:
        # Determine format from the matching candidate
        fmt = "simple"
        for candidate in candidates:
            if "dir" in candidate:
                check = root_dir / candidate["dir"] / candidate["file"]
            elif "dir_glob" in candidate:
                # Already matched; find which one
                dirs = sorted(
                    (p for p in root_dir.glob(candidate["dir_glob"]) if p.is_dir()),
                    key=lambda p: p.name,
                )
                check = None
                for d in dirs:
                    if (d / candidate["file"]) == matched[0]:
                        check = matched[0]
                        break
                if check is None:
                    continue
            else:
                continue
            if check == matched[0]:
                fmt = candidate.get("format", "simple")
                break
        return matched[0], fmt

    # 2+ matches: error
    lines = [f"Ambiguous glob match for '{canonical_name}': {len(matched)} candidates found:"]
    for mp in matched:
        lines.append(f"  - {mp}")
    raise RuntimeError("\n".join(lines))


def parse_method_result_args(
    method_result_args: Optional[List[str]],
    root_dir: Path,
) -> Dict[str, Path]:
    """Parse explicit --method_result arguments.

    Format: "Method Label=/absolute/or/relative/path"

    Returns:
        Dict mapping method label to absolute file path.
    """
    result: Dict[str, Path] = {}
    if not method_result_args:
        return result
    for arg in method_result_args:
        if "=" not in arg:
            raise ValueError(
                f"Invalid --method_result format: '{arg}'. "
                f"Expected: 'Method Name=/path/to/results.jsonl'"
            )
        label, path_str = arg.split("=", 1)
        label = label.strip()
        path_str = path_str.strip()
        fpath = Path(path_str)
        if not fpath.is_absolute():
            fpath = root_dir / fpath
        if not fpath.exists():
            raise FileNotFoundError(
                f"Method result file for '{label}' not found: {fpath}"
            )
        result[label] = fpath.resolve()
    return result


# =============================================================================
# Chunk metadata and gold alignment (P0-1)
# =============================================================================

def load_chunk_metadata_from_corpus_cache(corpus_cache: Optional[Path]) -> Dict[str, Dict]:
    """Load chunk text metadata from the corpus cache."""
    chunk_meta: Dict[str, Dict] = {}
    if not corpus_cache or not corpus_cache.exists():
        return chunk_meta

    try:
        with open(corpus_cache, "rb") as fh:
            data = pickle.load(fh)
    except Exception as exc:
        print(f"  [WARN] Could not load corpus cache {corpus_cache}: {exc}")
        return chunk_meta

    corpus_chunks = data.get("corpus_chunks", []) if isinstance(data, dict) else []
    for chunk in corpus_chunks:
        chunk_id = getattr(chunk, "chunk_id", "")
        text = getattr(chunk, "text", "")
        doc_id = getattr(chunk, "doc_id", "")
        register_chunk_meta(chunk_meta, chunk_id, text, doc_id)

    return chunk_meta


def load_gold_map_from_corpus_cache(corpus_cache: Optional[Path]) -> Tuple[Dict[str, List[str]], bool]:
    """Load source-aligned gold evidence chunk IDs from corpus cache.

    Returns:
        (gold_map, is_source_aligned) — is_source_aligned is True when the
        cache contains a real gold_map (not fallback generated IDs).
    """
    if not corpus_cache or not corpus_cache.exists():
        return {}, False
    try:
        with open(corpus_cache, "rb") as fh:
            data = pickle.load(fh)
    except Exception as exc:
        print(f"  [WARN] Could not load gold map from corpus cache {corpus_cache}: {exc}")
        return {}, False
    if not isinstance(data, dict):
        return {}, False
    gold_map = data.get("gold_map", {})
    if not isinstance(gold_map, dict) or not gold_map:
        return {}, False
    return {str(qid): list(ids or []) for qid, ids in gold_map.items()}, True


def compute_gold_alignment(
    query_ids: List[str],
    gold_map: Dict[str, List[str]],
) -> Dict[str, Any]:
    """Compute gold alignment statistics.

    Returns dict with keys:
        gold_alignment_coverage, gold_aligned_queries, gold_unaligned_queries,
        total_queries, aligned_count, unaligned_count.
    """
    aligned = [qid for qid in query_ids if gold_map.get(qid)]
    unaligned = [qid for qid in query_ids if not gold_map.get(qid)]
    total = len(query_ids)
    coverage = len(aligned) / max(total, 1)
    return {
        "gold_alignment_coverage": coverage,
        "gold_aligned_queries": aligned,
        "gold_unaligned_queries": unaligned,
        "total_queries": total,
        "aligned_count": len(aligned),
        "unaligned_count": len(unaligned),
    }


def check_gold_alignment_threshold(
    alignment: Dict[str, Any],
    min_coverage: float,
    paper_mode: bool,
) -> None:
    """Check gold alignment coverage against threshold.

    In paper_mode, raises RuntimeError if below threshold.
    Otherwise, prints a warning.
    """
    coverage = alignment["gold_alignment_coverage"]
    if coverage >= min_coverage:
        return
    msg = (
        f"Gold alignment coverage {coverage:.2%} is below threshold {min_coverage:.2%}. "
        f"{alignment['aligned_count']}/{alignment['total_queries']} queries aligned, "
        f"{alignment['unaligned_count']} unaligned."
    )
    if paper_mode:
        raise RuntimeError(f"FATAL (--paper_mode): {msg}")
    print(f"  [WARN] {msg}")


def build_chunk_metadata_lookup(
    samples: List[Dict],
    edgar_dir: Optional[Path] = None,
    corpus_cache: Optional[Path] = None,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    max_distractor_files: int = 50,
) -> Dict[str, Dict]:
    """Build chunk_id -> {text, doc_id} lookup from gold evidence + 10-K distractors."""
    chunk_meta: Dict[str, Dict] = load_chunk_metadata_from_corpus_cache(corpus_cache)

    for s in samples:
        for text in s.get("evidence_texts", []):
            for chunk in chunk_text(
                text,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                doc_id=s["id"],
            ):
                register_chunk_meta(chunk_meta, chunk.chunk_id, chunk.text, s["id"])

    if edgar_dir and edgar_dir.exists():
        distractor_files = list(edgar_dir.rglob("*.html")) + list(edgar_dir.rglob("*.txt"))
        for tf in distractor_files[:max_distractor_files]:
            try:
                for chunk in chunk_report(tf, chunk_size, chunk_overlap):
                    register_chunk_meta(chunk_meta, chunk.chunk_id, chunk.text, chunk.doc_id)
            except Exception:
                pass

    return chunk_meta


def build_cross_reference_from_rich(rich_filepath: Path) -> Dict[str, Dict]:
    """Build passage-id -> {text} lookup from rich-format result files."""
    cross_ref: Dict[str, Dict] = {}
    with open(rich_filepath, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            for entry in rec.get("top_k", []):
                pid = entry.get("passage_id", "")
                text = entry.get("text", "")
                if pid and text and pid not in cross_ref:
                    cross_ref[pid] = {"text": text}
    return cross_ref


def _lookup_text(
    chunk_id: str,
    chunk_meta: Dict[str, Dict],
    cross_ref: Dict[str, Dict],
) -> str:
    """Look up passage text by chunk ID with fallback strategies."""
    meta = chunk_meta.get(chunk_id, {})
    if meta.get("text"):
        return meta["text"]

    xref = cross_ref.get(chunk_id, {})
    if xref.get("text"):
        return xref["text"]

    hash_id = extract_hash_from_chunk_id(chunk_id)
    if hash_id and hash_id != chunk_id:
        meta2 = chunk_meta.get(hash_id, {})
        if meta2.get("text"):
            return meta2["text"]
        xref2 = cross_ref.get(hash_id, {})
        if xref2.get("text"):
            return xref2["text"]

    return f"[Evidence text not available for chunk: {chunk_id}]"


# =============================================================================
# Evidence text coverage
# =============================================================================

def compute_text_coverage(
    all_query_data: Dict[str, List[Dict]],
) -> Dict[str, Dict]:
    """Compute evidence text coverage statistics per method."""
    coverage: Dict[str, Dict] = {}
    for method_label, qd_list in all_query_data.items():
        total = 0
        found = 0
        for qd in qd_list:
            for text in qd.get("evidence_texts", []):
                total += 1
                if text and not text.startswith("[Evidence text not available"):
                    found += 1
        coverage[method_label] = {
            "total_passages": total,
            "found_passages": found,
            "coverage_ratio": found / max(total, 1),
        }
    return coverage


# =============================================================================
# Evidence extraction with token budget (P0-12)
# =============================================================================

def get_top_k_evidence_with_budget(
    rec: Dict,
    result_format: str,
    chunk_meta: Dict[str, Dict],
    cross_ref: Dict[str, Dict],
    top_k: int = 5,
    max_input_tokens: Optional[int] = None,
    max_tokens_per_evidence: Optional[int] = None,
) -> Tuple[List[str], List[str], List[str], int, bool]:
    """Extract top-k evidence with token budget truncation.

    Steps:
      1. Extract all candidate evidence from the record.
      2. Deduplicate by chunk_id, preserving rank order.
      3. Look up text for each chunk.
      4. Truncate by token budget.
      5. Return IDs, texts, dropped IDs, token count, and truncation flag.

    Returns:
        (used_ids, used_texts, dropped_ids, prompt_token_count, truncated)
    """
    raw_evidence: List[Dict] = []

    if result_format == "rich":
        for entry in rec.get("top_k", []):
            if isinstance(entry, dict):
                cid = str(entry.get("passage_id", entry.get("chunk_id", entry.get("id", ""))))
                text = str(entry.get("text", entry.get("passage", entry.get("content", ""))))
            elif isinstance(entry, str):
                cid = entry
                text = ""
            else:
                continue
            if cid:
                raw_evidence.append({"chunk_id": cid, "text": text})
    else:
        for key in (
            "retrieved_chunk_ids", "reranked_chunk_ids", "ranked_chunk_ids",
            "candidate_chunk_ids",
        ):
            values = rec.get(key)
            if isinstance(values, list) and values:
                for v in values:
                    if isinstance(v, str):
                        raw_evidence.append({"chunk_id": v, "text": ""})
                    elif isinstance(v, dict):
                        cid = str(v.get("chunk_id", v.get("id", "")))
                        text = str(v.get("text", ""))
                        if cid:
                            raw_evidence.append({"chunk_id": cid, "text": text})
                break
        else:
            for key in ("top_k", "ranked_chunks", "retrieved_chunks", "reranked_chunks",
                         "results", "candidates"):
                values = rec.get(key)
                if isinstance(values, list) and values:
                    for v in values:
                        if isinstance(v, dict):
                            cid = str(v.get("chunk_id", v.get("passage_id", v.get("id", ""))))
                            text = str(v.get("text", v.get("passage", v.get("content", ""))))
                            if cid:
                                raw_evidence.append({"chunk_id": cid, "text": text})
                        elif isinstance(v, str):
                            raw_evidence.append({"chunk_id": v, "text": ""})
                    break

    # Deduplicate by chunk_id, keeping first occurrence (lowest rank)
    seen: Set[str] = set()
    deduped: List[Dict] = []
    for ev in raw_evidence:
        if ev["chunk_id"] not in seen:
            seen.add(ev["chunk_id"])
            deduped.append(ev)

    # Look up text for entries that lack it
    for ev in deduped:
        if not ev["text"]:
            ev["text"] = _lookup_text(ev["chunk_id"], chunk_meta, cross_ref)

    # Apply token budget (approximate: char-based)
    budget = max_input_tokens
    per_chunk = max_tokens_per_evidence
    used: List[Dict] = []
    dropped: List[str] = []
    total_est = 0

    for ev in deduped[:top_k]:
        text = ev["text"]
        est = max(1, len(text) // 4) if text else 0
        if per_chunk is not None and est > per_chunk:
            est = per_chunk
        if budget is None or total_est + est <= budget:
            used.append(ev)
            total_est += est
        else:
            dropped.append(ev["chunk_id"])

    used_ids = [ev["chunk_id"] for ev in used]
    used_texts = [ev["text"] for ev in used]
    truncated = len(dropped) > 0 or len(used) < min(top_k, len(deduped))

    return used_ids, used_texts, dropped, total_est, truncated


# =============================================================================
# Local HuggingFace Generator
# =============================================================================

class LocalHFGenerator:
    """Wrapper for local HuggingFace Qwen2.5-7B-Instruct generation."""

    def __init__(
        self,
        model_path: str = "Qwen/Qwen2.5-7B-Instruct",
        device: str = "auto",
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
        do_sample: bool = False,
        batch_size: Optional[int] = None,
    ):
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.do_sample = do_sample
        self._batch_size = batch_size
        self._device_str = device
        self.tokenizer = None
        self.model = None
        self.device = device

    def load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"  Loading tokenizer from {self.model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"  Loading model from {self.model_path}...")
        load_kwargs: Dict[str, Any] = {"trust_remote_code": True}
        if self._device_str == "auto":
            if torch.cuda.is_available():
                load_kwargs["torch_dtype"] = torch.float16
                load_kwargs["device_map"] = "auto"
                self.device = "cuda"
            else:
                self.device = "cpu"
        elif self._device_str == "cpu":
            self.device = "cpu"
        else:
            self.device = self._device_str
            load_kwargs["device_map"] = self._device_str
            load_kwargs["torch_dtype"] = torch.float16

        self.model = AutoModelForCausalLM.from_pretrained(self.model_path, **load_kwargs)
        self.model.eval()
        if self._batch_size is None:
            self._batch_size = self._detect_batch_size()
        print(f"  Model loaded on {self.device}, batch_size={self._batch_size}")

    def _detect_batch_size(self) -> int:
        import torch
        if self.device == "cpu":
            return 1
        for bs in [8, 4, 2, 1]:
            try:
                test_prompt = "test"
                inputs = self.tokenizer(
                    [test_prompt] * bs, return_tensors="pt", padding=True, truncation=True,
                )
                if self.device != "cpu":
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                with torch.no_grad():
                    self.model.generate(**inputs, max_new_tokens=1, do_sample=False)
                del inputs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return bs
            except (RuntimeError, torch.cuda.OutOfMemoryError):
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
        return 1

    @property
    def batch_size(self) -> int:
        return self._batch_size or 1

    def build_prompt(self, query: str, evidence_texts: List[str]) -> str:
        evidence_str = "\n".join(
            f"[{i+1}] {text}" for i, text in enumerate(evidence_texts)
        )
        return GENERATION_PROMPT_TEMPLATE.format(
            evidence_passages=evidence_str, query=query,
        )

    def generate(self, prompts: List[str]) -> List[str]:
        import torch
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model not loaded. Call load() first.")
        answers: List[str] = []
        bs = self.batch_size
        for i in range(0, len(prompts), bs):
            batch = prompts[i:i + bs]
            try:
                inputs = self.tokenizer(
                    batch, return_tensors="pt", padding=True, truncation=True, max_length=2048,
                )
                if self.device != "cpu":
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        temperature=self.temperature if self.do_sample else 1.0,
                        do_sample=self.do_sample,
                        top_p=self.top_p if self.do_sample else 1.0,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                input_lens = [len(ids) for ids in inputs["input_ids"]]
                for j, output_ids in enumerate(outputs):
                    new_tokens = output_ids[input_lens[j]:]
                    answer = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                    answers.append(answer.strip())
                del inputs, outputs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print(f"    [WARN] OOM at batch size {bs}, falling back to bs=1: {e}")
                for prompt in batch:
                    try:
                        answers.append(self._generate_single(prompt))
                    except Exception as e2:
                        print(f"    [ERROR] Single generation failed: {e2}")
                        answers.append("")
        return answers

    def _generate_single(self, prompt: str) -> str:
        import torch
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        if self.device != "cpu":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature if self.do_sample else 1.0,
                do_sample=self.do_sample,
                top_p=self.top_p if self.do_sample else 1.0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        input_len = len(inputs["input_ids"][0])
        new_tokens = outputs[0][input_len:]
        answer = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        del inputs, outputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return answer.strip()

    def unload(self) -> None:
        import torch
        del self.model
        del self.tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.model = None
        self.tokenizer = None


class OpenRouterGenerator:
    """OpenRouter chat-completions generator."""

    def __init__(
        self,
        model_path: str = "qwen/qwen-2.5-7b-instruct",
        api_key_env: str = "OPENROUTER_API_KEY",
        base_url: str = "https://openrouter.ai/api/v1/chat/completions",
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
        do_sample: bool = False,
        batch_size: Optional[int] = None,
    ):
        self.model_path = model_path
        self.api_key_env = api_key_env
        self.base_url = base_url
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.do_sample = do_sample
        self._batch_size = batch_size or 1
        self.api_key = ""

    def load(self) -> None:
        self.api_key = os.environ.get(self.api_key_env, "").strip()
        if not self.api_key:
            raise RuntimeError(f"Missing API key env var: {self.api_key_env}")
        print(f"  Using OpenRouter model: {self.model_path}")

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def build_prompt(self, query: str, evidence_texts: List[str]) -> str:
        evidence_str = "\n".join(
            f"[{i+1}] {text}" for i, text in enumerate(evidence_texts)
        )
        return GENERATION_PROMPT_TEMPLATE.format(
            evidence_passages=evidence_str, query=query,
        )

    def generate(self, prompts: List[str]) -> List[Dict[str, Any]]:
        """Generate answers, returning rich result dicts with API metadata (P1-3)."""
        results: List[Dict[str, Any]] = []
        for prompt in prompts:
            results.append(self._generate_single(prompt))
        return results

    def _generate_single(self, prompt: str) -> Dict[str, Any]:
        payload = {
            "model": self.model_path,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature if self.do_sample else 0.0,
            "max_tokens": self.max_new_tokens,
        }
        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com",
            "X-Title": "fin-gnn-generation-eval",
        }
        req = urllib.request.Request(self.base_url, data=data, headers=headers, method="POST")

        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    result = json.loads(resp.read().decode("utf-8"))
                choice = result.get("choices", [{}])[0]
                return {
                    "status": "completed",
                    "answer": choice.get("message", {}).get("content", "").strip(),
                    "error_type": None,
                    "error_message": None,
                    "attempt_count": attempt + 1,
                    "api_model": result.get("model", ""),
                    "api_provider": result.get("provider", ""),
                    "api_request_id": result.get("id", ""),
                    "api_created": result.get("created", ""),
                    "api_finish_reason": choice.get("finish_reason", ""),
                    "api_usage": result.get("usage", {}),
                }
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                last_error = f"HTTP {exc.code}: {body[:500]}"
            except Exception as exc:
                last_error = str(exc)
            time.sleep(2 * (attempt + 1))

        return {
            "status": "api_error",
            "answer": "",
            "error_type": "api_error",
            "error_message": last_error,
            "attempt_count": 3,
        }

    def unload(self) -> None:
        pass


# =============================================================================
# Generation runner with safe JSONL writes (P0-5)
# =============================================================================

def _is_abstention(answer: str) -> bool:
    """Check if an answer is an abstention (insufficient evidence)."""
    text = answer.strip()
    if not text:
        return False
    # Treat explicit leading abstentions as abstention even when the model adds
    # an explanation afterwards, e.g. "Insufficient evidence. The provided ...".
    # Do not match mentions later in a normal answer such as
    # "The evidence is sufficient, not insufficient."
    if re.match(r"^\s*insufficient\s+evidence\b", text, re.IGNORECASE):
        return True
    for pat in _INSUFFICIENT_PATTERNS:
        if pat.match(text):
            return True
    return False


def _safe_jsonl_append(filepath: Path, record: Dict) -> None:
    """Append a JSON record to a JSONL file with fsync (P0-5)."""
    with open(filepath, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def _safe_read_jsonl(filepath: Path) -> List[Dict]:
    """Read JSONL with tolerance for a corrupt last line (P0-5).

    If the last line is corrupt JSON, back up the original and truncate.
    If an interior line is corrupt, raise an error.
    """
    records: List[Dict] = []
    if not filepath.exists():
        return records

    lines: List[str] = []
    with open(filepath, "r", encoding="utf-8") as fh:
        lines = fh.readlines()

    corrupt_lines: List[int] = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            records.append(json.loads(stripped))
        except json.JSONDecodeError:
            corrupt_lines.append(i)

    if not corrupt_lines:
        return records

    # Only the last non-empty line(s) can be corrupt
    non_empty_indices = [j for j, ln in enumerate(lines) if ln.strip()]
    if not non_empty_indices:
        return records

    last_idx = non_empty_indices[-1]
    if all(cl == last_idx for cl in corrupt_lines):
        # Back up and truncate
        backup = filepath.with_suffix(filepath.suffix + ".corrupt_backup")
        filepath.rename(backup)
        print(f"  [WARN] Corrupt last line in {filepath.name}; backed up to {backup.name}")

        # Rewrite clean records
        clean_records = records  # json.loads already parsed the valid ones
        # But if the corrupt line was partially read... we need to be careful
        # Re-parse without the last line
        with open(backup, "r", encoding="utf-8") as fh:
            clean_lines = fh.readlines()
        clean_records2: List[Dict] = []
        for i, line in enumerate(clean_lines):
            stripped = line.strip()
            if not stripped:
                continue
            if i == last_idx:
                continue
            try:
                clean_records2.append(json.loads(stripped))
            except json.JSONDecodeError:
                pass

        with open(filepath, "w", encoding="utf-8") as fh:
            for rec in clean_records2:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        return clean_records2

    raise RuntimeError(
        f"Corrupt JSON at interior line(s) {corrupt_lines} in {filepath}. "
        f"Cannot safely recover."
    )


def run_generation(
    method_key: str,
    method_label: str,
    records: Dict[str, Dict],
    result_format: str,
    query_data: List[Dict],
    generator: Any,
    output_dir: Path,
    resume: bool = False,
    generation_config: Optional[Dict] = None,
    progress_every: int = 10,
    run_id: str = "",
    max_input_tokens: Optional[int] = None,
    max_tokens_per_evidence: Optional[int] = None,
    chunk_meta: Optional[Dict[str, Dict]] = None,
    cross_ref: Optional[Dict[str, Dict]] = None,
) -> List[Dict]:
    """Generate answers for one method with safe JSONL writes.

    Resume uses (run_id, method_label, query_id) as the unique key (P0-4).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "generated_answers.jsonl"
    generation_config = generation_config or DEFAULT_GENERATION_CONFIG
    chunk_meta = chunk_meta or {}
    cross_ref = cross_ref or {}
    top_k = generation_config.get("top_k_evidence", 5)

    # Build set of completed (run_id, method, query_id) tuples for resume
    completed: Set[Tuple[str, str, str]] = set()
    if resume and output_path.exists():
        existing = _safe_read_jsonl(output_path)
        for ex in existing:
            completed.add((ex.get("run_id", ""), ex.get("method", ""), ex.get("query_id", "")))
        print(f"    Resume: {len(completed)} existing results found")

    # Filter uncompleted queries
    pending = [
        qd for qd in query_data
        if (run_id, method_label, qd["query_id"]) not in completed
    ]
    if not pending:
        print(f"    All {len(query_data)} queries already completed, skipping")
        results = _safe_read_jsonl(output_path)
        return [r for r in results if r.get("method") == method_label]

    print(f"    Generating {len(pending)} answers "
          f"({len(query_data) - len(pending)} already completed)")

    gen_start = time.time()
    results: List[Dict] = []
    flush_every = max(int(progress_every or 1), 1)

    for start in range(0, len(pending), flush_every):
        batch_qd = pending[start:start + flush_every]
        prompts = [
            generator.build_prompt(qd["query"], qd["evidence_texts"])
            for qd in batch_qd
        ]

        # Generate
        if isinstance(generator, OpenRouterGenerator):
            raw_answers = generator.generate(prompts)
        else:
            raw_texts = generator.generate(prompts)
            raw_answers = [
                {"status": "completed", "answer": t, "error_type": None,
                 "error_message": None, "attempt_count": 1}
                for t in raw_texts
            ]

        for i, qd in enumerate(batch_qd):
            raw = raw_answers[i] if i < len(raw_answers) else {
                "status": "api_error", "answer": "", "error_type": "missing",
                "error_message": "No output from generator", "attempt_count": 0,
            }
            answer = raw.get("answer", "")
            status = raw.get("status", "completed")

            # Classify status
            if status == "completed" and not answer:
                status = "empty_response"
            elif status == "completed" and _is_abstention(answer):
                status = "completed"  # Abstention is a valid completion

            gen_result = {
                "record_schema_version": RECORD_SCHEMA_VERSION,
                "prompt_version": PROMPT_VERSION,
                "evaluator_version": EVALUATOR_VERSION,
                "run_id": run_id,
                "query_id": qd["query_id"],
                "method": method_label,
                "query": qd["query"],
                "requested_evidence_ids": qd.get("requested_evidence_ids", []),
                "used_evidence_ids": qd.get("evidence_ids", []),
                "dropped_evidence_ids": qd.get("dropped_evidence_ids", []),
                "evidence_texts": qd["evidence_texts"],
                "gold_evidence_ids": qd.get("gold_evidence_ids", []),
                "reference_answer": qd["reference_answer"],
                "generated_answer": answer,
                "model": generator.model_path,
                "generation_config": generation_config,
                "status": status,
                "error_type": raw.get("error_type"),
                "error_message": raw.get("error_message"),
                "attempt_count": raw.get("attempt_count", 1),
                "prompt_token_count": qd.get("prompt_token_count", 0),
                "truncated": qd.get("truncated", False),
            }
            # Include OpenRouter metadata if available (P1-3)
            for or_key in ("api_model", "api_provider", "api_request_id",
                           "api_created", "api_finish_reason", "api_usage"):
                if or_key in raw:
                    gen_result[or_key] = raw[or_key]

            results.append(gen_result)
            _safe_jsonl_append(output_path, gen_result)

        done = min(start + len(batch_qd), len(pending))
        elapsed = time.time() - gen_start
        eta = elapsed / max(done, 1) * (len(pending) - done)
        print(
            f"    [{method_label}] generated {done}/{len(pending)} "
            f"({done / max(len(pending), 1) * 100:.1f}%) "
            f"elapsed={elapsed:.1f}s eta={eta:.1f}s"
        )

    return results


# =============================================================================
# Evaluation data structures (P0-2, P0-9, P0-10, P0-11)
# =============================================================================

@dataclass
class GenEvalResult:
    """Per-answer evaluation result."""

    query_id: str
    method: str
    query: str = ""
    reference_answer: str = ""
    generated_answer: str = ""
    evidence_ids: List[str] = field(default_factory=list)
    evidence_texts: List[str] = field(default_factory=list)
    gold_evidence_ids: List[str] = field(default_factory=list)
    gold_aligned: bool = False

    # Evidence retrieval metrics (based on used_evidence_ids, P0-12)
    evidence_hit_at_5: int = 0
    evidence_recall_at_5: float = 0.0
    all_gold_covered_at_5: int = 0
    evidence_mrr: float = 0.0

    # Core accuracy metrics (P0-10)
    answer_accuracy: float = 0.0  # legacy, kept for compatibility
    exact_match: bool = False
    relaxed_match: bool = False
    normalized_exact_match: float = 0.0
    token_f1: float = 0.0
    numeric_exact_match: float = 0.0
    numeric_tolerance_match: float = 0.0
    unit_match: float = 0.0

    # Faithfulness and numerical consistency (P0-2: None for abstentions)
    faithfulness: Optional[float] = None
    unsupported: int = 0
    numerical_consistency: Optional[float] = None
    numerical_applicable: bool = True
    requires_arithmetic_verification: bool = False  # P0-11

    # Abstention flags (P0-2)
    is_abstention: bool = False
    answered: int = 1

    # Optional metadata
    wrong_year: int = 0
    wrong_metric: int = 0
    insufficient_evidence: int = 0
    answer_length: int = 0
    error_explanation: str = ""


@dataclass
class AggregateGenMetrics:
    """Aggregated metrics across all queries for one method."""

    method: str
    num_queries: int = 0
    num_completed: int = 0
    num_api_errors: int = 0

    # Gold alignment
    gold_alignment_coverage: float = 0.0
    gold_aligned_count: int = 0

    # Evidence retrieval (only on gold-aligned queries)
    evidence_hit_at_5: float = 0.0
    evidence_hit_at_5_n: int = 0
    evidence_recall_at_5: float = 0.0
    all_gold_covered_at_5: float = 0.0
    evidence_mrr: float = 0.0

    # Answer rate (P0-2)
    answer_rate: float = 0.0
    insufficient_evidence_rate: float = 0.0
    api_error_rate: float = 0.0

    # Accuracy (all queries, abstention = 0)
    accuracy_all: float = 0.0
    exact_match_rate: float = 0.0
    relaxed_match_rate: float = 0.0
    normalized_exact_match: float = 0.0
    token_f1: float = 0.0
    numeric_exact_match: float = 0.0
    numeric_tolerance_match: float = 0.0
    unit_match: float = 0.0

    # Accuracy (answered only)
    accuracy_answered: float = 0.0
    token_f1_answered: float = 0.0

    # Faithfulness (answered only)
    faithfulness_answered: float = 0.0
    unsupported_rate_answered: float = 0.0

    # Numerical consistency (answered, applicable only)
    numerical_consistency_answered: float = 0.0
    num_numerical_applicable: int = 0

    # Arithmetic verification (P0-11)
    arithmetic_verification_required_rate: float = 0.0

    # Legacy
    wrong_year_rate: float = 0.0
    wrong_metric_rate: float = 0.0
    avg_answer_length: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "method": self.method,
            "num_queries": self.num_queries,
            "num_completed": self.num_completed,
            "num_api_errors": self.num_api_errors,
            "gold_alignment_coverage": round(self.gold_alignment_coverage, 4),
            "gold_aligned_count": self.gold_aligned_count,
            "evidence_hit_at_5": round(self.evidence_hit_at_5, 4),
            "evidence_hit_at_5_n": self.evidence_hit_at_5_n,
            "evidence_recall_at_5": round(self.evidence_recall_at_5, 4),
            "all_gold_covered_at_5": round(self.all_gold_covered_at_5, 4),
            "evidence_mrr": round(self.evidence_mrr, 4),
            "answer_rate": round(self.answer_rate, 4),
            "insufficient_evidence_rate": round(self.insufficient_evidence_rate, 4),
            "api_error_rate": round(self.api_error_rate, 4),
            "accuracy_all": round(self.accuracy_all, 4),
            "exact_match_rate": round(self.exact_match_rate, 4),
            "relaxed_match_rate": round(self.relaxed_match_rate, 4),
            "normalized_exact_match": round(self.normalized_exact_match, 4),
            "token_f1": round(self.token_f1, 4),
            "numeric_exact_match": round(self.numeric_exact_match, 4),
            "numeric_tolerance_match": round(self.numeric_tolerance_match, 4),
            "unit_match": round(self.unit_match, 4),
            "accuracy_answered": round(self.accuracy_answered, 4),
            "token_f1_answered": round(self.token_f1_answered, 4),
            "faithfulness_answered": round(self.faithfulness_answered, 4),
            "unsupported_rate_answered": round(self.unsupported_rate_answered, 4),
            "numerical_consistency_answered": round(self.numerical_consistency_answered, 4),
            "numerical_applicable_queries": self.num_numerical_applicable,
            "arithmetic_verification_required_rate": round(self.arithmetic_verification_required_rate, 4),
            "wrong_year_rate": round(self.wrong_year_rate, 4),
            "wrong_metric_rate": round(self.wrong_metric_rate, 4),
            "avg_answer_length": round(self.avg_answer_length, 1),
        }


# =============================================================================
# Evaluation helpers
# =============================================================================

def _normalize(text: str) -> str:
    text = text.strip()
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text.lower().strip())


def _extract_numbers(text: str) -> List[str]:
    return [m.group(0).strip() for m in _NUMBER_RE.finditer(text)]


def _normalize_number(num_str: str) -> str:
    s = num_str.lower().strip()
    s = s.replace("$", "")
    s = re.sub(r"(\d),(\d)", r"\1\2", s)
    s = s.replace("million", "m").replace("billion", "b")
    s = s.replace("thousand", "k").replace("trillion", "t")
    s = re.sub(r"\s+", "", s)
    return s


def _extract_years(text: str) -> Set[str]:
    return set(m.group(1) for m in _YEAR_RE.finditer(text))


def _compute_token_f1(pred: str, gold: str) -> float:
    pred_tokens = set(pred.split())
    gold_tokens = set(gold.split())
    if not gold_tokens and not pred_tokens:
        return 1.0
    if not gold_tokens or not pred_tokens:
        return 0.0
    tp = len(pred_tokens & gold_tokens)
    prec = tp / len(pred_tokens)
    rec = tp / len(gold_tokens)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def _parse_number_value(num_str: str) -> Optional[float]:
    s = num_str.lower().strip().replace("$", "").replace(",", "").replace("%", "")
    multipliers = {"million": 1e6, "billion": 1e9, "thousand": 1e3, "trillion": 1e12}
    for word, mult in multipliers.items():
        if word in s:
            s = s.replace(word, "").strip()
            try:
                return float(s) * mult
            except ValueError:
                return None
    try:
        return float(s)
    except ValueError:
        return None


def _compute_number_set_match(gen_nums: List[str], ref_nums: List[str]) -> float:
    """Fraction of generated numbers found in reference (after normalization)."""
    if not gen_nums:
        return 1.0
    ref_set = {_normalize_number(n) for n in ref_nums}
    matched = sum(1 for n in gen_nums if _normalize_number(n) in ref_set)
    return matched / len(gen_nums)


def _compute_numeric_tolerance_match(gen_nums: List[str], ref_nums: List[str]) -> float:
    """Fraction of generated numbers that match reference within 1% tolerance."""
    if not gen_nums:
        return 1.0
    ref_vals = []
    for n in ref_nums:
        v = _parse_number_value(n)
        if v is not None:
            ref_vals.append(v)
    if not ref_vals:
        return 0.0

    matched = 0
    for n in gen_nums:
        v = _parse_number_value(n)
        if v is None:
            continue
        for rv in ref_vals:
            if abs(v - rv) / max(abs(rv), 1) < 0.01:
                matched += 1
                break
    return matched / len(gen_nums)


def _compute_unit_match(gen: str, ref: str) -> float:
    """Fraction of unit mentions in generated that appear in reference."""
    gen_units = set(m.group(1).lower() for m in _UNIT_RE.finditer(gen))
    ref_units = set(m.group(1).lower() for m in _UNIT_RE.finditer(ref))
    if not gen_units:
        return 1.0
    return len(gen_units & ref_units) / len(gen_units)


def _check_arithmetic_verification(
    gen_answer: str,
    evidence_texts: List[str],
) -> bool:
    """Check if answer contains numbers not in evidence that may come from calculation.

    P0-11: Instead of marking as unsupported, flag for arithmetic verification.
    """
    gen_numbers = _extract_numbers(gen_answer)
    if not gen_numbers:
        return False
    evidence_full = " ".join(evidence_texts)
    ev_numbers = {_normalize_number(n) for n in _extract_numbers(evidence_full)}

    for n in gen_numbers:
        n_norm = _normalize_number(n)
        if n_norm not in ev_numbers:
            # Check if close to any evidence number (within tolerance)
            try:
                n_val = _parse_number_value(n)
                found = False
                for ev_n in _extract_numbers(evidence_full):
                    ev_val = _parse_number_value(ev_n)
                    if ev_val is not None and n_val is not None:
                        if abs(n_val - ev_val) / max(abs(ev_val), 1) < 0.01:
                            found = True
                            break
                if not found:
                    return True  # Number not in evidence, may require arithmetic
            except (ValueError, ZeroDivisionError):
                return True
    return False


# =============================================================================
# Evaluator
# =============================================================================

class GenerationEvaluator:
    """Deterministic evaluator for the generation experiment.

    Key behaviors (v2):
      - P0-2: Abstentions get faithfulness=None, numerical_consistency=None.
      - P0-9: Evidence Recall@5, All-Gold-Covered@5, Evidence MRR.
      - P0-10: Token F1, numeric exact/tolerance/unit match.
      - P0-11: Arithmetic verification flag instead of unsupported for derived numbers.
    """

    def evaluate(
        self,
        query_id: str,
        method: str,
        query: str,
        reference_answer: str,
        generated_answer: str,
        evidence_ids: List[str],
        evidence_texts: List[str],
        gold_evidence_ids: Optional[List[str]] = None,
        gold_aligned: bool = False,
    ) -> GenEvalResult:
        """Evaluate a single generated answer."""
        gold_ids = gold_evidence_ids or []

        result = GenEvalResult(
            query_id=query_id,
            method=method,
            query=query,
            reference_answer=reference_answer,
            generated_answer=generated_answer,
            evidence_ids=evidence_ids,
            evidence_texts=evidence_texts,
            gold_evidence_ids=gold_ids,
            gold_aligned=gold_aligned,
        )

        gen = generated_answer.strip()
        ref = reference_answer.strip()
        result.answer_length = len(gen)

        # --- Abstention detection (P0-2) ---
        result.is_abstention = _is_abstention(gen)
        result.insufficient_evidence = int(result.is_abstention)

        if result.is_abstention:
            result.answered = 0
            result.answer_accuracy = 0.0
            result.faithfulness = None
            result.numerical_consistency = None
            # Evidence metrics still computed on abstentions
        else:
            result.answered = 1

        if not gen:
            result.error_explanation = "Empty generated answer"
            result.answered = 0
            return result

        # --- Accuracy (P0-10) ---
        gen_norm = _normalize(gen)
        ref_norm = _normalize(ref)

        result.exact_match = gen_norm == ref_norm
        result.relaxed_match = (
            ref_norm in gen_norm or gen_norm in ref_norm or
            (bool(gen_norm) and bool(ref_norm) and
             len(set(gen_norm.split()) & set(ref_norm.split())) / max(len(set(ref_norm.split())), 1) >= 0.8)
        ) if not result.exact_match else False

        if result.exact_match:
            result.answer_accuracy = 1.0
        elif result.relaxed_match:
            result.answer_accuracy = 0.5
        else:
            result.answer_accuracy = 0.0

        # Normalized exact match (case-insensitive, whitespace-normalized)
        gen_simple = re.sub(r"\s+", " ", gen.lower().strip().rstrip("."))
        ref_simple = re.sub(r"\s+", " ", ref.lower().strip().rstrip("."))
        result.normalized_exact_match = 1.0 if gen_simple == ref_simple else 0.0

        # Token F1
        result.token_f1 = _compute_token_f1(gen_norm, ref_norm)

        # Numeric metrics
        gen_numbers = _extract_numbers(gen)
        ref_numbers = _extract_numbers(ref)
        result.numeric_exact_match = _compute_number_set_match(gen_numbers, ref_numbers)
        result.numeric_tolerance_match = _compute_numeric_tolerance_match(gen_numbers, ref_numbers)
        result.unit_match = _compute_unit_match(gen, ref)

        # --- Numerical consistency (P0-2: None for abstentions) ---
        if not result.is_abstention:
            evidence_full = " ".join(evidence_texts)
            ev_numbers = {_normalize_number(n) for n in _extract_numbers(evidence_full)}
            if gen_numbers:
                matched = 0
                for n in gen_numbers:
                    n_norm = _normalize_number(n)
                    if n_norm in ev_numbers:
                        matched += 1
                    else:
                        try:
                            n_val = _parse_number_value(n)
                            for ev_n in _extract_numbers(evidence_full):
                                ev_val = _parse_number_value(ev_n)
                                if ev_val is not None and n_val is not None:
                                    if abs(n_val - ev_val) / max(abs(ev_val), 1) < 0.01:
                                        matched += 1
                                        break
                        except (ValueError, ZeroDivisionError):
                            pass
                result.numerical_consistency = matched / len(gen_numbers)
                result.numerical_applicable = True
            else:
                result.numerical_consistency = 1.0
                result.numerical_applicable = False

        # --- Faithfulness & Unsupported (P0-2: None for abstentions) ---
        if not result.is_abstention:
            result.faithfulness, result.unsupported = _compute_faithfulness_v2(
                gen, evidence_texts, query
            )
            # P0-11: Check arithmetic verification
            result.requires_arithmetic_verification = _check_arithmetic_verification(
                gen, evidence_texts
            )

        # --- Evidence retrieval metrics (P0-9) ---
        if result.gold_aligned and gold_ids:
            top5_ids = set(evidence_ids[:5])
            gold_set = set(gold_ids)

            # Hit@5
            result.evidence_hit_at_5 = int(bool(top5_ids & gold_set))

            # Recall@5
            if gold_set:
                result.evidence_recall_at_5 = len(top5_ids & gold_set) / len(gold_set)

            # All-Gold-Covered@5
            result.all_gold_covered_at_5 = int(gold_set.issubset(top5_ids))

            # MRR
            for rank, eid in enumerate(evidence_ids[:5], start=1):
                if eid in gold_set:
                    result.evidence_mrr = 1.0 / rank
                    break

        # --- Wrong year / metric ---
        if not result.is_abstention:
            result.wrong_year = _check_wrong_year(gen, evidence_texts)
            result.wrong_metric = _check_wrong_metric(gen, evidence_texts, query)

        # --- Error explanation ---
        issues = []
        if result.is_abstention:
            issues.append("abstention")
        elif not result.exact_match and not result.relaxed_match:
            issues.append("accuracy_low")
        if result.numerical_consistency is not None and result.numerical_applicable and result.numerical_consistency < 1.0:
            issues.append(f"numerical_inconsistency({result.numerical_consistency:.2f})")
        if result.unsupported:
            issues.append("unsupported_claims")
        if result.wrong_year:
            issues.append("wrong_year")
        if result.wrong_metric:
            issues.append("wrong_metric")
        if result.requires_arithmetic_verification:
            issues.append("requires_arithmetic_verification")
        result.error_explanation = "; ".join(issues) if issues else ""

        return result

    @staticmethod
    def aggregate(
        method: str,
        per_answer: List[GenEvalResult],
        gold_aligned_count: int = 0,
    ) -> AggregateGenMetrics:
        """Aggregate per-answer results into method-level metrics.

        P0-2: Abstentions do NOT contribute to faithfulness/numcon.
        P0-1: Evidence retrieval metrics only on gold-aligned queries.
        """
        n = len(per_answer)
        if n == 0:
            return AggregateGenMetrics(method=method)

        # Basic counts
        completed = [r for r in per_answer if r.answered == 1 or r.insufficient_evidence == 1]
        # Actually, completed includes all records that have a non-empty answer or abstention
        # API errors / empty have no answer
        api_errors = [r for r in per_answer if r.answer_length == 0 and not r.is_abstention]
        n_completed = n - len(api_errors)

        # Abstention / answered
        answered = [r for r in per_answer if r.answered == 1]
        n_answered = len(answered)
        answer_rate = n_answered / max(n, 1)

        # Gold-aligned subset for evidence metrics
        gold_aligned = [r for r in per_answer if r.gold_aligned]
        ga_n = len(gold_aligned)

        # Numerical applicable (answered only)
        num_applicable = [r for r in answered if r.numerical_applicable]
        n_num_app = len(num_applicable)

        # Build aggregate
        agg = AggregateGenMetrics(
            method=method,
            num_queries=n,
            num_completed=n_completed,
            num_api_errors=len(api_errors),
            gold_alignment_coverage=gold_aligned_count / max(n, 1),
            gold_aligned_count=gold_aligned_count,
            answer_rate=answer_rate,
            insufficient_evidence_rate=float(np.mean([float(r.insufficient_evidence) for r in per_answer])),
            api_error_rate=len(api_errors) / max(n, 1),
            # Accuracy (all queries)
            accuracy_all=float(np.mean([r.answer_accuracy for r in per_answer])),
            exact_match_rate=float(np.mean([float(r.exact_match) for r in per_answer])),
            relaxed_match_rate=float(np.mean([float(r.relaxed_match) for r in per_answer])),
            normalized_exact_match=float(np.mean([r.normalized_exact_match for r in per_answer])),
            token_f1=float(np.mean([r.token_f1 for r in per_answer])),
            numeric_exact_match=float(np.mean([r.numeric_exact_match for r in per_answer])),
            numeric_tolerance_match=float(np.mean([r.numeric_tolerance_match for r in per_answer])),
            unit_match=float(np.mean([r.unit_match for r in per_answer])),
            # Accuracy (answered only)
            accuracy_answered=float(np.mean([r.answer_accuracy for r in answered])) if n_answered > 0 else 0.0,
            token_f1_answered=float(np.mean([r.token_f1 for r in answered])) if n_answered > 0 else 0.0,
            # Faithfulness (answered only)
            faithfulness_answered=float(np.mean(
                [r.faithfulness for r in answered if r.faithfulness is not None]
            )) if n_answered > 0 else 0.0,
            unsupported_rate_answered=float(np.mean([float(r.unsupported) for r in answered])) if n_answered > 0 else 0.0,
            # Numerical consistency (answered, applicable only)
            numerical_consistency_answered=float(np.mean(
                [r.numerical_consistency for r in num_applicable if r.numerical_consistency is not None]
            )) if n_num_app > 0 else 0.0,
            num_numerical_applicable=n_num_app,
            # Arithmetic verification
            arithmetic_verification_required_rate=float(np.mean(
                [float(r.requires_arithmetic_verification) for r in per_answer]
            )),
            # Evidence retrieval (gold-aligned only)
            evidence_hit_at_5=float(np.mean([float(r.evidence_hit_at_5) for r in gold_aligned])) if ga_n > 0 else 0.0,
            evidence_hit_at_5_n=ga_n,
            evidence_recall_at_5=float(np.mean([r.evidence_recall_at_5 for r in gold_aligned])) if ga_n > 0 else 0.0,
            all_gold_covered_at_5=float(np.mean([float(r.all_gold_covered_at_5) for r in gold_aligned])) if ga_n > 0 else 0.0,
            evidence_mrr=float(np.mean([r.evidence_mrr for r in gold_aligned])) if ga_n > 0 else 0.0,
            # Legacy
            wrong_year_rate=float(np.mean([float(r.wrong_year) for r in per_answer])),
            wrong_metric_rate=float(np.mean([float(r.wrong_metric) for r in per_answer])),
            avg_answer_length=float(np.mean([r.answer_length for r in per_answer])),
        )
        return agg


def _compute_faithfulness_v2(
    generated_answer: str,
    evidence_texts: List[str],
    query: str = "",
) -> Tuple[float, int]:
    """Compute faithfulness score and unsupported flag (v2).

    P0-2: Abstentions handled by caller — this function should not receive them.
    P0-11: Numbers not in evidence → may require arithmetic, not automatic unsupported.
    """
    gen_numbers = _extract_numbers(generated_answer)
    evidence_full = " ".join(evidence_texts)

    if not gen_numbers:
        gen_words = set(generated_answer.lower().split())
        ev_words = set(evidence_full.lower().split())
        if gen_words:
            overlap = len(gen_words & ev_words) / len(gen_words)
            if overlap < 0.3:
                return overlap, 1
            return overlap, 0
        return 1.0, 0

    ev_numbers_normalized = {_normalize_number(n) for n in _extract_numbers(evidence_full)}
    ans_normalized = {_normalize_number(n) for n in gen_numbers}

    if ans_normalized:
        matched = len(ans_normalized & ev_numbers_normalized)
        faithfulness = matched / len(ans_normalized)
    else:
        faithfulness = 1.0

    unsupported = 0
    if ans_normalized and len(ans_normalized & ev_numbers_normalized) < len(ans_normalized) * 0.5:
        unsupported = 1

    gen_years = _extract_years(generated_answer)
    ev_years = _extract_years(evidence_full)
    if gen_years and ev_years and not (gen_years & ev_years):
        unsupported = 1

    return faithfulness, unsupported


def _check_wrong_year(generated_answer: str, evidence_texts: List[str]) -> int:
    gen_years = _extract_years(generated_answer)
    if not gen_years:
        return 0
    evidence_full = " ".join(evidence_texts)
    ev_years = _extract_years(evidence_full)
    if not ev_years:
        return 0
    if gen_years - ev_years:
        return 1
    return 0


def _check_wrong_metric(
    generated_answer: str,
    evidence_texts: List[str],
    query: str = "",
) -> int:
    _metric_terms = {
        "revenue", "revenues", "income", "profit", "loss", "eps", "ebitda",
        "ebit", "margin", "cost", "expense", "asset", "liability", "equity",
        "cash flow", "dividend", "roe", "roa", "roi", "pe ratio",
    }
    evidence_full = " ".join(evidence_texts).lower()
    gen_lower = generated_answer.lower()
    for term in _metric_terms:
        if term in gen_lower and term not in evidence_full:
            return 1
    return 0


# =============================================================================
# Output Writers
# =============================================================================

def _atomic_write_json(filepath: Path, data: Any) -> None:
    """Write JSON to a temp file then atomically replace."""
    tmp = filepath.with_suffix(filepath.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
    os.replace(tmp, filepath)


def save_run_manifest(output_dir: Path, manifest: RunManifest) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(output_dir / "run_manifest.json", manifest.to_dict())


def save_selected_query_ids(output_dir: Path, query_ids: List[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(output_dir / "selected_query_ids.json", query_ids)


def save_method_sources(output_dir: Path, method_sources: Dict[str, Dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(output_dir / "method_sources.json", method_sources)


def save_generation_metrics_csv(
    output_dir: Path,
    all_agg: Dict[str, AggregateGenMetrics],
    methods_run: List[str],
) -> None:
    """Save aggregated metrics as CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fpath = output_dir / "generation_metrics.csv"

    fieldnames = [
        "Method",
        "N Total",
        "N Completed",
        "Gold Align Coverage",
        "Evidence Hit@5",
        "Evidence Recall@5",
        "All Gold@5",
        "Evidence MRR",
        "Answer Rate",
        "Accuracy All",
        "Accuracy Answered",
        "Token F1",
        "Faithfulness Answered",
        "NumCon Answered",
        "Unsupported Answered",
        "Insufficient Evidence Rate",
        "API Error Rate",
        "Arithmetic Verify Rate",
        "Wrong Year Rate",
        "Wrong Metric Rate",
        "Avg Answer Length",
    ]

    with open(fpath, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for label in methods_run:
            m = all_agg.get(label)
            if m is None:
                writer.writerow({"Method": label})
                continue
            writer.writerow({
                "Method": label,
                "N Total": str(m.num_queries),
                "N Completed": str(m.num_completed),
                "Gold Align Coverage": f"{m.gold_alignment_coverage:.4f}",
                "Evidence Hit@5": f"{m.evidence_hit_at_5:.4f}",
                "Evidence Recall@5": f"{m.evidence_recall_at_5:.4f}",
                "All Gold@5": f"{m.all_gold_covered_at_5:.4f}",
                "Evidence MRR": f"{m.evidence_mrr:.4f}",
                "Answer Rate": f"{m.answer_rate:.4f}",
                "Accuracy All": f"{m.accuracy_all:.4f}",
                "Accuracy Answered": f"{m.accuracy_answered:.4f}",
                "Token F1": f"{m.token_f1:.4f}",
                "Faithfulness Answered": f"{m.faithfulness_answered:.4f}",
                "NumCon Answered": f"{m.numerical_consistency_answered:.4f}",
                "Unsupported Answered": f"{m.unsupported_rate_answered:.4f}",
                "Insufficient Evidence Rate": f"{m.insufficient_evidence_rate:.4f}",
                "API Error Rate": f"{m.api_error_rate:.4f}",
                "Arithmetic Verify Rate": f"{m.arithmetic_verification_required_rate:.4f}",
                "Wrong Year Rate": f"{m.wrong_year_rate:.4f}",
                "Wrong Metric Rate": f"{m.wrong_metric_rate:.4f}",
                "Avg Answer Length": f"{m.avg_answer_length:.1f}",
            })

    print(f"  CSV saved to: {fpath}")


def save_generation_metrics_json(
    output_dir: Path,
    all_agg: Dict[str, AggregateGenMetrics],
    config_info: Dict,
) -> None:
    """Save aggregated metrics as JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fpath = output_dir / "generation_metrics.json"

    output = {
        "config": config_info,
        "generated": datetime.now().isoformat(),
        "methods": {
            label: m.to_dict()
            for label, m in all_agg.items()
        },
    }

    _atomic_write_json(fpath, output)
    print(f"  JSON saved to: {fpath}")


def save_latex_table(
    output_dir: Path,
    all_agg: Dict[str, AggregateGenMetrics],
    methods_run: List[str],
) -> None:
    """Save the main results table as LaTeX."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fpath = output_dir / "table_generation_results.tex"

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Answer generation results using evidence selected "
        r"by different retrieval and reranking methods.}",
        r"\label{tab:generation_results}",
        r"\small",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Evidence Source & Accuracy $\uparrow$ & Faithfulness $\uparrow$ & "
        r"Numerical Consistency $\uparrow$ & Unsupported Rate $\downarrow$ & Hit@5 $\uparrow$ \\",
        r"\midrule",
    ]

    for label in methods_run:
        m = all_agg.get(label)
        if m is None:
            lines.append(f"{label} & N/A & N/A & N/A & N/A & N/A \\\\")
            continue
        bold = r"\textbf{" if "Ours" in label else ""
        bold_end = "}" if "Ours" in label else ""
        lines.append(
            f"{bold}{label}{bold_end} & "
            f"{bold}{m.accuracy_answered:.4f}{bold_end} & "
            f"{bold}{m.faithfulness_answered:.4f}{bold_end} & "
            f"{bold}{m.numerical_consistency_answered:.4f}{bold_end} & "
            f"{bold}{m.unsupported_rate_answered:.4f}{bold_end} & "
            f"{bold}{m.evidence_hit_at_5:.4f}{bold_end} \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    with open(fpath, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"  LaTeX table saved to: {fpath}")


def save_eval_details(
    output_dir: Path,
    all_eval: Dict[str, List[GenEvalResult]],
) -> None:
    """Save per-query-method evaluation details as JSONL."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fpath = output_dir / "eval_details.jsonl"

    with open(fpath, "w", encoding="utf-8") as fh:
        for method_label, evals in all_eval.items():
            for e in evals:
                entry = {
                    "query_id": e.query_id,
                    "method": e.method,
                    "query": e.query,
                    "reference_answer": e.reference_answer,
                    "generated_answer": e.generated_answer,
                    "evidence_ids": e.evidence_ids,
                    "gold_evidence_ids": e.gold_evidence_ids,
                    "gold_aligned": e.gold_aligned,
                    "evidence_hit_at_5": e.evidence_hit_at_5,
                    "evidence_recall_at_5": e.evidence_recall_at_5,
                    "all_gold_covered_at_5": e.all_gold_covered_at_5,
                    "evidence_mrr": e.evidence_mrr,
                    "answer_accuracy": e.answer_accuracy,
                    "exact_match": e.exact_match,
                    "relaxed_match": e.relaxed_match,
                    "normalized_exact_match": e.normalized_exact_match,
                    "token_f1": e.token_f1,
                    "numeric_exact_match": e.numeric_exact_match,
                    "numeric_tolerance_match": e.numeric_tolerance_match,
                    "unit_match": e.unit_match,
                    "faithfulness": e.faithfulness,
                    "numerical_consistency": e.numerical_consistency,
                    "numerical_applicable": e.numerical_applicable,
                    "requires_arithmetic_verification": e.requires_arithmetic_verification,
                    "unsupported": e.unsupported,
                    "wrong_year": e.wrong_year,
                    "wrong_metric": e.wrong_metric,
                    "is_abstention": e.is_abstention,
                    "answered": e.answered,
                    "answer_length": e.answer_length,
                    "error_explanation": e.error_explanation,
                    "evidence_texts": e.evidence_texts,
                }
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"  Eval details saved to: {fpath}")


def save_debug_cases(
    output_dir: Path,
    all_eval: Dict[str, List[GenEvalResult]],
) -> None:
    """Save curated debug cases as JSONL.

    Includes: normal answer, abstention, arithmetic verification samples.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    fpath = output_dir / "debug_generation_cases.jsonl"

    by_query: Dict[str, Dict[str, GenEvalResult]] = defaultdict(dict)
    for method_label, evals in all_eval.items():
        for e in evals:
            by_query[e.query_id][method_label] = e

    debug_cases: List[Dict] = []

    # Category 1: Normal correct answer
    for query_id, method_evals in by_query.items():
        for ml, e in method_evals.items():
            if e.answered == 1 and e.answer_accuracy == 1.0 and not e.requires_arithmetic_verification:
                debug_cases.append(_make_debug_entry(e, "normal_correct_answer"))
                break
        if any(d.get("debug_category") == "normal_correct_answer" for d in debug_cases):
            break

    # Category 2: Abstention
    for query_id, method_evals in by_query.items():
        for ml, e in method_evals.items():
            if e.is_abstention:
                debug_cases.append(_make_debug_entry(e, "abstention"))
                break
        if any(d.get("debug_category") == "abstention" for d in debug_cases):
            break

    # Category 3: Requires arithmetic verification
    for query_id, method_evals in by_query.items():
        for ml, e in method_evals.items():
            if e.requires_arithmetic_verification and e.answered == 1:
                debug_cases.append(_make_debug_entry(e, "requires_arithmetic_verification"))
                break
        if any(d.get("debug_category") == "requires_arithmetic_verification" for d in debug_cases):
            break

    # Category 4: FinDual-GNN correct, R-GCN wrong
    findual_label = "FinDual-GNN (Ours)"
    rgcn_label = "R-GCN"
    for query_id, method_evals in by_query.items():
        fe = method_evals.get(findual_label)
        re = method_evals.get(rgcn_label)
        if fe and re and fe.answer_accuracy == 1.0 and re.answer_accuracy == 0.0:
            debug_cases.append(_make_debug_entry(fe, "findual_correct_rgcn_wrong"))
            break

    # Category 5: All methods wrong
    for query_id, method_evals in by_query.items():
        if method_evals and all(e.answer_accuracy == 0.0 for e in method_evals.values()):
            first = next(iter(method_evals.values()))
            debug_cases.append(_make_debug_entry(first, "all_methods_wrong"))
            break

    # Category 6: Unsupported answer
    for query_id, method_evals in by_query.items():
        for ml, e in method_evals.items():
            if e.unsupported == 1 and not e.is_abstention:
                debug_cases.append(_make_debug_entry(e, "unsupported_answer"))
                break
        if any(d.get("debug_category") == "unsupported_answer" for d in debug_cases):
            break

    with open(fpath, "w", encoding="utf-8") as fh:
        for case in debug_cases:
            fh.write(json.dumps(case, ensure_ascii=False) + "\n")

    print(f"  Debug cases saved to: {fpath} ({len(debug_cases)} cases)")


def _make_debug_entry(e: GenEvalResult, category: str) -> Dict:
    return {
        "query_id": e.query_id,
        "query": e.query,
        "method": e.method,
        "evidence_ids": e.evidence_ids,
        "gold_evidence_ids": e.gold_evidence_ids,
        "evidence_hit_at_5": e.evidence_hit_at_5,
        "evidence_texts": e.evidence_texts,
        "generated_answer": e.generated_answer,
        "reference_answer": e.reference_answer,
        "answer_accuracy": e.answer_accuracy,
        "faithfulness": e.faithfulness,
        "numerical_consistency": e.numerical_consistency,
        "unsupported": e.unsupported,
        "is_abstention": e.is_abstention,
        "requires_arithmetic_verification": e.requires_arithmetic_verification,
        "error_explanation": e.error_explanation,
        "debug_category": category,
    }


def print_summary_table(
    all_agg: Dict[str, AggregateGenMetrics],
    methods_run: List[str],
) -> None:
    """Print the final results table to terminal."""
    print()
    print("=" * 130)
    print("  Answer Generation Results with Different Evidence Sources (v2)")
    print("=" * 130)
    print()
    header = (
        f"{'Method':<26} {'AnsRate':>7} {'AccAll':>7} {'AccAns':>7} "
        f"{'TokF1':>7} {'Faith':>7} {'NumCon':>7} "
        f"{'Hit@5':>7} {'Rec@5':>7} {'MRR':>7} {'N':>5}"
    )
    print(header)
    print("-" * 130)

    for label in methods_run:
        m = all_agg.get(label)
        if m is None:
            print(f"{label:<26} {'N/A':>7} {'N/A':>7} {'N/A':>7} {'N/A':>7} "
                  f"{'N/A':>7} {'N/A':>7} {'N/A':>7} {'N/A':>7} {'N/A':>7} {'-':>5}")
            continue
        print(
            f"{label:<26} {m.answer_rate:>7.4f} {m.accuracy_all:>7.4f} "
            f"{m.accuracy_answered:>7.4f} {m.token_f1:>7.4f} "
            f"{m.faithfulness_answered:>7.4f} {m.numerical_consistency_answered:>7.4f} "
            f"{m.evidence_hit_at_5:>7.4f} {m.evidence_recall_at_5:>7.4f} "
            f"{m.evidence_mrr:>7.4f} {m.num_queries:>5}"
        )

    print("-" * 130)
    print("  AnsRate = Answer Rate (non-abstention)")
    print("  AccAll = Accuracy (all queries, abstention=0)")
    print("  AccAns = Accuracy (answered only)")
    print("  TokF1 = Token F1 score")
    print("  Faith = Faithfulness (answered only)")
    print("  NumCon = Numerical Consistency (answered, applicable only)")
    print("  Hit@5 = Evidence Hit@5, Rec@5 = Evidence Recall@5, MRR = Evidence MRR")
    print()


# =============================================================================
# Main Pipeline
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Answer generation with different evidence sources (v2)"
    )
    # Data
    parser.add_argument("--config", default="configs/default.yaml",
                        help="Config file for data paths")
    parser.add_argument("--output_dir", default="outputs/generation_eval",
                        help="Output directory")
    parser.add_argument("--corpus_cache", default="cache/table1_full_corpus_seq4096.pkl",
                        help="Corpus cache with chunk text metadata")
    parser.add_argument("--split", default="test",
                        help="Dataset split (P1-2)")

    # Model
    parser.add_argument("--model_path", default="Qwen/Qwen2.5-7B-Instruct",
                        help="HuggingFace model name or local path")
    parser.add_argument("--provider", choices=["local", "openrouter"], default="local",
                        help="Generation backend")
    parser.add_argument("--openrouter_api_key_env", default="OPENROUTER_API_KEY")
    parser.add_argument("--openrouter_base_url",
                        default="https://openrouter.ai/api/v1/chat/completions")
    parser.add_argument("--device", default="auto")

    # Generation parameters
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_new_tokens", type=int, default=128,
                        help="Max generation tokens")
    parser.add_argument("--top_k_evidence", type=int, default=5,
                        help="Top-k evidence passages")

    # Token budget (P0-12)
    parser.add_argument("--max_input_tokens", type=int, default=None,
                        help="Max input tokens for evidence (None=unlimited)")
    parser.add_argument("--max_tokens_per_evidence", type=int, default=None,
                        help="Max tokens per evidence chunk (None=unlimited)")

    # Query selection
    parser.add_argument("--num_queries", type=int, default=0,
                        help="Limit number of queries (0=all)")
    parser.add_argument("--sample_mode", choices=["first", "random"], default="first")
    parser.add_argument("--sample_seed", type=int, default=42)

    # Method selection (P0-3)
    parser.add_argument("--method_result", action="append", default=None,
                        dest="method_results",
                        help="Explicit method result: 'Name=/path/to/file.jsonl' "
                             "(repeatable)")
    parser.add_argument("--methods", type=str, default="",
                        help="Comma-separated subset of methods")
    parser.add_argument("--skip_methods", type=str, default="")

    # Gold alignment (P0-1)
    parser.add_argument("--min_gold_alignment_coverage", type=float, default=0.95)
    parser.add_argument("--paper_mode", action="store_true",
                        help="Strict mode: error on low alignment coverage")

    # Run control
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing output")
    parser.add_argument("--sanity", action="store_true",
                        help="Sanity mode: 10 queries, verbose")
    parser.add_argument("--eval_only", action="store_true",
                        help="Skip generation, only evaluate")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing output directory")
    parser.add_argument("--progress_every", type=int, default=10)
    parser.add_argument("--structured_output", action="store_true",
                        help="Request JSON structured output (P1-4, experimental)")

    args = parser.parse_args()

    # Sanity mode adjustments
    if args.sanity:
        if args.num_queries == 0:
            args.num_queries = 10

    # Config
    cfg = Config.from_yaml(args.config)
    root_dir = cfg.root_dir
    output_dir = root_dir / args.output_dir

    # Use absolute corpus_cache path
    corpus_cache_path = Path(args.corpus_cache)
    if not corpus_cache_path.is_absolute():
        corpus_cache_path = root_dir / corpus_cache_path

    # Update generation config
    gen_config = dict(DEFAULT_GENERATION_CONFIG)
    gen_config["top_k_evidence"] = args.top_k_evidence
    gen_config["max_new_tokens"] = args.max_new_tokens
    gen_config["temperature"] = args.temperature
    gen_config["do_sample"] = args.do_sample if hasattr(args, 'do_sample') else (args.temperature > 0)

    # ------------------------------------------------------------------
    # Build run manifest (P0-4)
    # ------------------------------------------------------------------
    manifest = RunManifest(
        dataset_split=args.split,
        split_mode="current_loader_default",
        corpus_cache_sha256=_file_sha256(corpus_cache_path),
        generator_provider=args.provider,
        generator_model=args.model_path,
        generator_parameters={
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_new_tokens": args.max_new_tokens,
        },
        prompt_sha256=_str_sha256(GENERATION_PROMPT_TEMPLATE),
        top_k_evidence=args.top_k_evidence,
        max_input_tokens=args.max_input_tokens,
        max_tokens_per_evidence=args.max_tokens_per_evidence,
        tokenizer_mode="approx_char_based",
        min_gold_alignment_coverage=args.min_gold_alignment_coverage,
        paper_mode=args.paper_mode,
        sample_mode=args.sample_mode,
        sample_seed=args.sample_seed,
        num_queries=args.num_queries,
        created_at=datetime.now().isoformat(),
    )

    # Check for existing output. Full resume compatibility is validated after
    # method files and selected query IDs are resolved.
    manifest_path = output_dir / "run_manifest.json"
    run_id = output_dir.name
    existing_manifest: Optional[RunManifest] = None

    if output_dir.exists() and not args.overwrite and not args.eval_only:
        if manifest_path.exists():
            existing_manifest = RunManifest.from_dict(
                json.loads(manifest_path.read_text(encoding="utf-8"))
            )
            if args.resume:
                print("  Existing manifest found; resume config will be validated after inputs are resolved.")
            else:
                print(f"\nERROR: Output directory '{output_dir}' already has a manifest.")
                print(f"  Use --resume to continue or --overwrite to restart.")
                sys.exit(1)
        else:
            existing = list(output_dir.glob("generated_answers.jsonl"))
            if existing and not args.resume:
                print(f"\nERROR: Output directory '{output_dir}' already has results "
                      f"(no manifest found, possibly pre-v2).")
                print(f"  Use --resume to continue or --overwrite to restart.")
                sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Resolve method result files (P0-3)
    # ------------------------------------------------------------------
    method_sources: Dict[str, Dict[str, Any]] = {}

    if args.method_results:
        # Explicit paths
        explicit = parse_method_result_args(args.method_results, root_dir)
        target_methods = list(explicit.keys())
        for label, fpath in explicit.items():
            fmt = "simple"
            # Check if file looks like rich format
            with open(fpath, "r", encoding="utf-8") as fh:
                first = fh.readline().strip()
                if first:
                    try:
                        probe = json.loads(first)
                        if "top_k" in probe and isinstance(probe["top_k"], list):
                            if probe["top_k"] and isinstance(probe["top_k"][0], dict):
                                if "passage_id" in probe["top_k"][0] and "text" in probe["top_k"][0]:
                                    fmt = "rich"
                    except json.JSONDecodeError:
                        pass
            method_sources[label] = {
                "filepath": str(fpath.resolve()),
                "format": fmt,
                "file_sha256": _file_sha256(fpath),
                "initial_retriever": "BAAI/bge-m3",
            }
    else:
        # Legacy glob mode (P0-3 behavior: 0=error, 1=allow, 2+=error)
        target_methods = CANONICAL_METHODS[:]
        if args.methods:
            target_methods = [m.strip() for m in args.methods.split(",") if m.strip()]
        skip_set = set(m.strip() for m in args.skip_methods.split(",") if m.strip())
        target_methods = [m for m in target_methods if m not in skip_set]

        for canonical in target_methods:
            fpath, fmt = find_method_file_legacy(canonical, root_dir)
            if fpath is None:
                print(f"  [WARN] No result file found for '{canonical}' in legacy mode.")
                continue
            method_sources[canonical] = {
                "filepath": str(fpath.resolve()),
                "format": fmt,
                "file_sha256": _file_sha256(fpath),
                "initial_retriever": "",
            }
        target_methods = [m for m in target_methods if m in method_sources]

    if not method_sources:
        print("\n  ERROR: No method result files found.")
        sys.exit(1)

    # Update manifest with method info
    manifest.method_files = {
        label: ms["filepath"] for label, ms in method_sources.items()
    }
    manifest.method_sources = method_sources
    manifest.initial_retriever = "BAAI/bge-m3"

    # ------------------------------------------------------------------
    # Config info for output
    # ------------------------------------------------------------------
    config_info = {
        "model": args.model_path,
        "provider": args.provider,
        "generation_config": gen_config,
        "top_k_evidence": args.top_k_evidence,
        "max_new_tokens": args.max_new_tokens,
        "max_input_tokens": args.max_input_tokens,
        "max_tokens_per_evidence": args.max_tokens_per_evidence,
        "sample_mode": args.sample_mode,
        "sample_seed": args.sample_seed,
        "device": args.device,
        "batch_size": args.batch_size or "auto",
        "split": args.split,
        "paper_mode": args.paper_mode,
        "command": " ".join(sys.argv),
    }

    print("=" * 60)
    print("  Generation with Selected Evidence (v2)")
    print("=" * 60)
    print(f"  Model:       {args.model_path}")
    print(f"  Provider:    {args.provider}")
    print(f"  Output:      {output_dir}")
    print(f"  Top-K:       {args.top_k_evidence}")
    print(f"  Max Tokens:  {args.max_new_tokens}")
    print(f"  Max Input:   {args.max_input_tokens or 'unlimited'}")
    print(f"  CorpusCache: {corpus_cache_path}")
    print(f"  Device:      {args.device}")
    print(f"  Num Queries: {args.num_queries or 'all'}")
    print(f"  SampleMode:  {args.sample_mode}")
    print(f"  Methods:     {target_methods}")
    print(f"  Paper Mode:  {args.paper_mode}")
    print(f"  Resume:      {args.resume}")
    print(f"  Eval Only:   {args.eval_only}")

    # ------------------------------------------------------------------
    # 1. Load FinDER data
    # ------------------------------------------------------------------
    print("\n[1/5] Loading FinDER dataset...")
    samples = load_dataset("finder", cfg.data_dir)
    print(f"  Loaded {len(samples)} QA samples (split_mode={manifest.split_mode})")

    # P1-1: Deterministic random sampling with persistence
    all_query_ids = sorted([s["id"] for s in samples])
    selected_ids_path = output_dir / "selected_query_ids.json"

    if args.num_queries > 0:
        if args.eval_only and selected_ids_path.exists():
            # Read existing selection
            selected = json.loads(selected_ids_path.read_text(encoding="utf-8"))
            selected_set = set(selected)
            samples = [s for s in samples if s["id"] in selected_set]
            print(f"  Using persisted selection: {len(samples)} queries")
        elif args.sample_mode == "random":
            rng = random.Random(args.sample_seed)
            selected_ids = rng.sample(all_query_ids, min(args.num_queries, len(all_query_ids)))
            save_selected_query_ids(output_dir, selected_ids)
            selected_set = set(selected_ids)
            samples = [s for s in samples if s["id"] in selected_set]
            print(f"  Randomly selected {len(samples)} queries (seed={args.sample_seed})")
        else:
            samples = samples[:args.num_queries]
            save_selected_query_ids(output_dir, [s["id"] for s in samples])
            print(f"  Selected first {len(samples)} queries")
    else:
        save_selected_query_ids(output_dir, [s["id"] for s in samples])

    # Update manifest with query IDs hash
    selected_ids_final = sorted([s["id"] for s in samples])
    manifest.selected_query_ids_sha256 = _str_sha256(json.dumps(selected_ids_final))
    manifest.num_queries = len(samples)

    if existing_manifest is not None and args.resume:
        mismatches = manifest.check_resume_compatible(existing_manifest)
        if mismatches:
            print("\nERROR: Resume config mismatch. Cannot resume with different config.")
            print("  Mismatched fields:")
            for mm in mismatches:
                print(f"    - {mm}")
            print("  Create a new output directory to proceed.")
            sys.exit(1)
        print("  [OK] Resume config validated against existing manifest.")

    # ------------------------------------------------------------------
    # Gold alignment (P0-1)
    # ------------------------------------------------------------------
    query_ids = [s["id"] for s in samples]
    gold_map, is_source_aligned = load_gold_map_from_corpus_cache(corpus_cache_path)
    alignment = compute_gold_alignment(query_ids, gold_map)

    if is_source_aligned:
        print(f"  Gold alignment: {alignment['aligned_count']}/{alignment['total_queries']} "
              f"queries ({alignment['gold_alignment_coverage']:.2%})")
        check_gold_alignment_threshold(
            alignment, args.min_gold_alignment_coverage, args.paper_mode
        )
    else:
        print("  [WARN] No source-aligned gold map in corpus cache; "
              "evidence retrieval metrics will be unavailable.")
        alignment["gold_alignment_coverage"] = 0.0
        alignment["gold_aligned_queries"] = []
        alignment["gold_unaligned_queries"] = query_ids
        alignment["aligned_count"] = 0
        alignment["unaligned_count"] = len(query_ids)

    aligned_set = set(alignment["gold_aligned_queries"])

    # ------------------------------------------------------------------
    # 2. Build chunk metadata lookup
    # ------------------------------------------------------------------
    print("\n[2/5] Building chunk metadata lookup...")
    t0 = time.time()
    chunk_meta = build_chunk_metadata_lookup(
        samples,
        edgar_dir=cfg.edgar_dir,
        corpus_cache=corpus_cache_path,
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
    )
    print(f"  {len(chunk_meta)} chunks indexed ({time.time() - t0:.1f}s)")

    # Build cross-reference from rich-format results
    cross_ref: Dict[str, Dict] = {}
    for label, ms in method_sources.items():
        if ms["format"] == "rich":
            fpath = Path(ms["filepath"])
            if fpath.exists():
                cross_ref = build_cross_reference_from_rich(fpath)
                print(f"  Cross-ref: {len(cross_ref)} passages from {fpath.name}")
                break

    # ------------------------------------------------------------------
    # 3. Load evidence source results
    # ------------------------------------------------------------------
    print("\n[3/5] Loading evidence source results...")
    method_data: Dict[str, Dict] = {}
    actually_run_methods: List[str] = []

    for canonical in target_methods:
        ms = method_sources.get(canonical)
        if not ms:
            print(f"  [WARN] No source for '{canonical}', skipping")
            continue

        fpath = Path(ms["filepath"])
        fmt = ms["format"]
        print(f"  Loading {canonical} from {fpath.name} (format: {fmt})...")

        if fmt == "rich":
            records = {}
            with open(fpath, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    qid = rec.get("query_id", "")
                    if qid:
                        records[qid] = rec
        else:
            records = {}
            with open(fpath, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    qid = rec.get("question_id", rec.get("query_id", ""))
                    if qid:
                        records[qid] = rec

        matched = sum(1 for qid in query_ids if qid in records)
        print(f"    {matched}/{len(query_ids)} queries matched")

        method_data[canonical] = {
            "records": records,
            "format": fmt,
            "filepath": fpath,
        }
        actually_run_methods.append(canonical)

    if not method_data:
        print("\n  ERROR: No method results could be loaded.")
        sys.exit(1)

    # Build query_data with token budget (P0-12)
    print("\n  Building query evidence index (with token budget)...")
    all_query_data: Dict[str, List[Dict]] = {}

    for canonical in actually_run_methods:
        mdata = method_data[canonical]
        records = mdata["records"]
        fmt = mdata["format"]
        qd_list: List[Dict] = []

        for s in samples:
            qid = s["id"]
            rec = records.get(qid)
            if rec is None:
                continue

            used_ids, used_texts, dropped_ids, prompt_tokens, truncated = \
                get_top_k_evidence_with_budget(
                    rec, fmt, chunk_meta, cross_ref,
                    top_k=args.top_k_evidence,
                    max_input_tokens=args.max_input_tokens,
                    max_tokens_per_evidence=args.max_tokens_per_evidence,
                )

            # Get all evidence IDs before truncation for requested tracking
            all_ids, _, _, _, _ = get_top_k_evidence_with_budget(
                rec, fmt, chunk_meta, cross_ref,
                top_k=args.top_k_evidence,
                max_input_tokens=None,
                max_tokens_per_evidence=None,
            )

            qd_list.append({
                "query_id": qid,
                "query": s["question"],
                "reference_answer": s.get("answer", ""),
                "gold_evidence_ids": gold_map.get(qid, []),
                "gold_aligned": qid in aligned_set,
                "evidence_ids": used_ids,
                "evidence_texts": used_texts,
                "requested_evidence_ids": all_ids,
                "dropped_evidence_ids": dropped_ids,
                "prompt_token_count": prompt_tokens,
                "truncated": truncated,
            })

        all_query_data[canonical] = qd_list
        print(f"    {canonical}: {len(qd_list)} queries with evidence ready")

    # Print evidence text coverage
    print("\n  Evidence text coverage:")
    coverage = compute_text_coverage(all_query_data)
    for method_label, cov in coverage.items():
        pct = cov["coverage_ratio"] * 100
        flag = " [LOW]" if pct < 50 else ""
        print(f"    {method_label}: {cov['found_passages']}/{cov['total_passages']} "
              f"passages ({pct:.1f}%){flag}")

    # ------------------------------------------------------------------
    # 4. Generation
    # ------------------------------------------------------------------
    if not args.eval_only:
        print("\n[4/5] Generating answers...")

        if args.provider == "openrouter":
            generator = OpenRouterGenerator(
                model_path=args.model_path,
                api_key_env=args.openrouter_api_key_env,
                base_url=args.openrouter_base_url,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                do_sample=args.temperature > 0,
                batch_size=args.batch_size,
            )
        else:
            generator = LocalHFGenerator(
                model_path=args.model_path,
                device=args.device,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                do_sample=args.temperature > 0,
                batch_size=args.batch_size,
            )

        try:
            generator.load()
        except Exception as e:
            print(f"  ERROR: Failed to load model: {e}")
            sys.exit(1)

        all_gen: Dict[str, List[Dict]] = {}
        for canonical in actually_run_methods:
            qd_list = all_query_data.get(canonical, [])
            if not qd_list:
                print(f"\n  [{canonical}] No queries to generate, skipping")
                continue

            print(f"\n  [{canonical}] Generating...")
            t0 = time.time()

            gen_results = run_generation(
                method_key="",
                method_label=canonical,
                records=method_data[canonical]["records"],
                result_format=method_data[canonical]["format"],
                query_data=qd_list,
                generator=generator,
                output_dir=output_dir,
                resume=args.resume,
                generation_config=gen_config,
                progress_every=args.progress_every,
                run_id=run_id,
                max_input_tokens=args.max_input_tokens,
                max_tokens_per_evidence=args.max_tokens_per_evidence,
                chunk_meta=chunk_meta,
                cross_ref=cross_ref,
            )
            all_gen[canonical] = gen_results
            print(f"    {len(gen_results)} results in {time.time() - t0:.1f}s")

        generator.unload()
    else:
        print("\n[4/5] Skipped (--eval_only)")

    # Save manifest after generation
    save_run_manifest(output_dir, manifest)
    save_method_sources(output_dir, method_sources)

    # ------------------------------------------------------------------
    # 5. Evaluation
    # ------------------------------------------------------------------
    print("\n[5/5] Evaluating generated answers...")

    gen_path = output_dir / "generated_answers.jsonl"
    if not gen_path.exists():
        print(f"  ERROR: No generated answers found at {gen_path}")
        sys.exit(1)

    all_gen_loaded: Dict[str, List[Dict]] = defaultdict(list)
    for rec in _safe_read_jsonl(gen_path):
        method = rec.get("method", "")
        all_gen_loaded[method].append(rec)

    evaluator = GenerationEvaluator()
    all_eval: Dict[str, List[GenEvalResult]] = {}
    all_agg: Dict[str, AggregateGenMetrics] = {}

    for canonical in actually_run_methods:
        gen_list = all_gen_loaded.get(canonical, [])
        if not gen_list:
            print(f"  [{canonical}] No generated answers, skipping eval")
            continue

        evals = []
        for g in gen_list:
            qid = g["query_id"]
            e = evaluator.evaluate(
                query_id=qid,
                method=g["method"],
                query=g.get("query", ""),
                reference_answer=g.get("reference_answer", ""),
                generated_answer=g.get("generated_answer", ""),
                evidence_ids=g.get("used_evidence_ids", g.get("evidence_ids", [])),
                evidence_texts=g.get("evidence_texts", []),
                gold_evidence_ids=g.get("gold_evidence_ids", []),
                gold_aligned=qid in aligned_set,
            )
            evals.append(e)

        all_eval[canonical] = evals
        agg = GenerationEvaluator.aggregate(
            canonical, evals,
            gold_aligned_count=alignment["aligned_count"],
        )
        all_agg[canonical] = agg
        print(f"  [{canonical}] AccAll={agg.accuracy_all:.4f} "
              f"AccAns={agg.accuracy_answered:.4f} "
              f"Faith={agg.faithfulness_answered:.4f} "
              f"NumCon={agg.numerical_consistency_answered:.4f} "
              f"Hit@5={agg.evidence_hit_at_5:.4f} "
              f"(n={agg.num_queries})")

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  SAVING OUTPUTS")
    print("=" * 60)

    save_generation_metrics_csv(output_dir, all_agg, actually_run_methods)
    save_generation_metrics_json(output_dir, all_agg, config_info)
    save_latex_table(output_dir, all_agg, actually_run_methods)
    save_eval_details(output_dir, all_eval)
    save_debug_cases(output_dir, all_eval)

    print_summary_table(all_agg, actually_run_methods)

    # Report alignment details
    print(f"\n  Gold alignment: {alignment['aligned_count']}/{alignment['total_queries']} "
          f"queries source-aligned ({alignment['gold_alignment_coverage']:.2%})")
    if alignment["unaligned_count"] > 0:
        print(f"  Unaligned queries excluded from Evidence Hit/Recall/MRR: "
              f"{alignment['unaligned_count']}")

    print(f"\nOutput directory: {output_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
