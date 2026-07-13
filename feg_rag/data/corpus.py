"""Corpus construction and gold-evidence alignment utilities.

The benchmark retrieval corpus must come from source documents, not from the
gold evidence snippets themselves.  This module builds a stable document corpus
from SEC filings and aligns each annotated gold evidence text to those chunks.
"""

from __future__ import annotations

import re
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from feg_rag.config import Config
from feg_rag.data.chunker import Chunk, chunk_report, chunk_text


@dataclass
class AlignmentRecord:
    question_id: str
    evidence_index: int
    matched_chunk_ids: List[str]
    strategy: str
    score: float
    warning: str = ""


@dataclass
class CorpusBuildReport:
    total_doc_files: int
    chunked_doc_files: int
    failed_doc_files: int
    total_chunks: int
    failures: List[str]


def build_benchmark_corpus(
    samples: List[Dict],
    cfg: Config,
    *,
    max_doc_files: int | None = None,
    allow_gold_only_corpus: bool | None = None,
    jaccard_threshold: float = 0.35,
    max_chunk_fail_rate: float = 0.2,
    verbose: bool = True,
) -> Tuple[List[Chunk], Dict[str, List[str]], List[AlignmentRecord]]:
    """Build corpus from filings and align gold evidence to corpus chunks.

    By default this refuses to fall back to a gold-only corpus.  Gold-only mode
    is only for explicit debugging or small unit tests.
    """
    if not samples:
        raise ValueError("No QA samples loaded; cannot build benchmark corpus.")
    if allow_gold_only_corpus is None:
        allow_gold_only_corpus = bool(
            cfg._raw.get("allow_gold_only_corpus", False)
            or cfg._raw.get("corpus", {}).get("allow_gold_only_corpus", False)
        )
    if max_doc_files is None:
        max_doc_files = cfg._raw.get("corpus", {}).get("max_doc_files")

    corpus, report = build_document_corpus(
        cfg,
        max_doc_files=max_doc_files,
        max_fail_rate=max_chunk_fail_rate,
        return_report=True,
    )
    if not corpus:
        if not allow_gold_only_corpus:
            raise FileNotFoundError(
                f"No corpus documents found under {cfg.edgar_dir}. "
                "Provide full 10-K/filing documents or set "
                "allow_gold_only_corpus=true only for debug runs."
            )
        warnings.warn(
            "Using gold-only corpus because allow_gold_only_corpus is enabled. "
            "Do not use this mode for benchmark results."
        )
        corpus, gold_map = build_gold_only_corpus(samples, cfg)
        records = [
            AlignmentRecord(qid, i, ids, "gold_only_debug", 1.0)
            for qid, ids in gold_map.items()
            for i in range(len(ids))
        ]
        return corpus, gold_map, records

    gold_map, records = align_gold_evidence(
        samples, corpus, jaccard_threshold=jaccard_threshold
    )
    matched_questions = sum(1 for ids in gold_map.values() if ids)
    if samples and matched_questions == 0:
        raise RuntimeError(
            "Gold evidence alignment produced zero matched questions. "
            "Check edgar_dir, document preprocessing, and evidence text format."
        )
    if verbose:
        matched = sum(1 for r in records if r.matched_chunk_ids)
        total = len(records)
        print(
            f"  Corpus documents: {report.chunked_doc_files}/{report.total_doc_files} "
            f"chunked, {report.failed_doc_files} failed, {report.total_chunks} chunks"
        )
        print(
            f"  Gold alignment: {matched}/{total} evidence snippets matched "
            f"({matched / total:.1%})" if total else "  Gold alignment: no snippets"
        )
    return corpus, gold_map, records


def build_document_corpus(
    cfg: Config,
    *,
    max_doc_files: int | None = None,
    max_fail_rate: float = 0.2,
    return_report: bool = False,
) -> List[Chunk] | Tuple[List[Chunk], CorpusBuildReport]:
    """Chunk all source documents under cfg.edgar_dir."""
    files = _document_files(cfg.edgar_dir)
    if max_doc_files is not None:
        files = files[:max_doc_files]

    corpus: List[Chunk] = []
    failures: List[str] = []
    chunked = 0
    for path in files:
        try:
            chunks = chunk_report(path, cfg.chunk_size, cfg.chunk_overlap)
            if chunks:
                chunked += 1
                corpus.extend(chunks)
            else:
                failures.append(f"{path}: no chunks produced")
        except Exception as exc:
            failures.append(f"{path}: {exc}")
            warnings.warn(f"Failed to chunk {path}: {exc}")

    report = CorpusBuildReport(
        total_doc_files=len(files),
        chunked_doc_files=chunked,
        failed_doc_files=len(failures),
        total_chunks=len(corpus),
        failures=failures,
    )
    if files and len(failures) / len(files) > max_fail_rate:
        preview = "; ".join(failures[:5])
        raise RuntimeError(
            f"Too many document chunking failures: {len(failures)}/{len(files)} "
            f"(max_fail_rate={max_fail_rate}). Examples: {preview}"
        )
    if return_report:
        return corpus, report
    return corpus


def build_gold_only_corpus(
    samples: List[Dict],
    cfg: Config,
) -> Tuple[List[Chunk], Dict[str, List[str]]]:
    """Explicit debug-only corpus built from annotated evidence snippets."""
    corpus: List[Chunk] = []
    gold_map: Dict[str, List[str]] = {}
    for sample in samples:
        qid = sample["id"]
        ids: List[str] = []
        for i, text in enumerate(sample.get("evidence_texts", [])):
            for chunk in chunk_text(
                text,
                cfg.chunk_size,
                cfg.chunk_overlap,
                doc_id=f"gold::{qid}::{i}",
                section="gold-evidence",
            ):
                corpus.append(chunk)
                ids.append(chunk.chunk_id)
        gold_map[qid] = ids
    return corpus, gold_map


def align_gold_evidence(
    samples: List[Dict],
    corpus: List[Chunk],
    *,
    jaccard_threshold: float = 0.35,
) -> Tuple[Dict[str, List[str]], List[AlignmentRecord]]:
    """Align annotated evidence snippets to stable corpus chunk IDs."""
    normalized_chunks = [(chunk, normalize_text(chunk.text)) for chunk in corpus]
    token_index = _build_token_index(normalized_chunks)

    gold_map: Dict[str, List[str]] = {}
    records: List[AlignmentRecord] = []
    for sample in samples:
        qid = sample["id"]
        qids: List[str] = []
        for i, evidence in enumerate(sample.get("evidence_texts", [])):
            ids, strategy, score = _align_one(
                evidence,
                normalized_chunks,
                token_index,
                jaccard_threshold,
            )
            warning = ""
            if not ids:
                warning = (
                    f"Gold evidence alignment failed for question {qid} "
                    f"evidence[{i}]"
                )
                warnings.warn(warning)
            else:
                qids.extend(ids)
            records.append(AlignmentRecord(qid, i, ids, strategy, score, warning))
        gold_map[qid] = sorted(set(qids))
    return gold_map, records


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").lower()).strip()


def _align_one(
    evidence: str,
    normalized_chunks: List[Tuple[Chunk, str]],
    token_index: Dict[str, List[int]],
    jaccard_threshold: float,
) -> Tuple[List[str], str, float]:
    gold = normalize_text(evidence)
    if not gold:
        return [], "empty", 0.0

    contains_matches: List[str] = []
    for chunk, chunk_text_norm in normalized_chunks:
        if gold in chunk_text_norm or chunk_text_norm in gold:
            contains_matches.append(chunk.chunk_id)
    if contains_matches:
        return contains_matches, "contains", 1.0

    gold_tokens = _tokens(gold)
    if not gold_tokens:
        return [], "no_tokens", 0.0

    candidate_counts: Counter[int] = Counter()
    for token in gold_tokens:
        candidate_counts.update(token_index.get(token, []))

    best_ids: List[str] = []
    best_score = 0.0
    for idx, _ in candidate_counts.most_common(200):
        chunk, chunk_text_norm = normalized_chunks[idx]
        score = _jaccard(gold_tokens, _tokens(chunk_text_norm))
        if score > best_score:
            best_ids = [chunk.chunk_id]
            best_score = score
        elif score == best_score and score > 0:
            best_ids.append(chunk.chunk_id)

    if best_score >= jaccard_threshold:
        return best_ids, "jaccard", best_score
    return [], "unmatched", best_score


def _document_files(edgar_dir: Path) -> List[Path]:
    if not edgar_dir.exists():
        return []
    files = list(edgar_dir.rglob("*.txt"))
    files.extend(edgar_dir.rglob("*.html"))
    files.extend(edgar_dir.rglob("*.htm"))
    return sorted({p.resolve() for p in files})


def _build_token_index(
    normalized_chunks: List[Tuple[Chunk, str]],
) -> Dict[str, List[int]]:
    index: Dict[str, List[int]] = defaultdict(list)
    for idx, (_, text) in enumerate(normalized_chunks):
        for token in _tokens(text):
            index[token].append(idx)
    return index


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", text) if len(t) > 2}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)
