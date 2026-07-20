"""Unified evidence schema for generation experiments.

All method result files (simple and rich formats) are converted to a common
``RankedEvidence`` representation so downstream generation and evaluation code
operates on a single data model.

Provides:
  - RankedEvidence: frozen dataclass for one (query, chunk) pair
  - convert_simple / convert_rich: format-specific converters
  - deduplicate_evidence: chunk-level dedup preserving rank order
  - truncate_by_token_budget: budget-aware truncation
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Unified schema
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RankedEvidence:
    """One ranked evidence chunk for a single query.

    All fields are strings (or None for score) to avoid type ambiguity across
    different source formats.
    """

    query_id: str
    chunk_id: str
    rank: int          # 1-based rank within the method's output for this query
    score: Optional[float]
    text: Optional[str]
    doc_id: Optional[str]
    source_file: str   # path of the file this record came from


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------

def _extract_entry_id_text(entry) -> Tuple[str, str]:
    """Pull (chunk_id, text) from a single top_k entry (dict or str)."""
    if isinstance(entry, str):
        return entry, ""
    if not isinstance(entry, dict):
        return "", ""
    cid = (
        entry.get("chunk_id")
        or entry.get("passage_id")
        or entry.get("id")
        or entry.get("doc_id")
        or ""
    )
    text = entry.get("text") or entry.get("passage") or entry.get("content") or ""
    return str(cid), str(text)


def convert_rich_format(
    filepath: Path,
    query_ids: Optional[set] = None,
) -> List[RankedEvidence]:
    """Convert a rich-format (exp1-style) results file to unified schema.

    Rich format has inline ``top_k`` entries with passage_id and text.
    """
    results: List[RankedEvidence] = []
    with open(filepath, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            qid = str(rec.get("query_id", ""))
            if not qid:
                continue
            if query_ids is not None and qid not in query_ids:
                continue
            for rank, entry in enumerate(rec.get("top_k", []), start=1):
                cid, text = _extract_entry_id_text(entry)
                if cid:
                    results.append(RankedEvidence(
                        query_id=qid,
                        chunk_id=cid,
                        rank=rank,
                        score=None,
                        text=text or None,
                        doc_id=None,
                        source_file=str(filepath),
                    ))
    return results


def _extract_simple_entries(rec: dict) -> List[Tuple[str, str, Optional[float]]]:
    """Return ordered (chunk_id, text, score) pairs from simple-format JSONL."""
    # List-of-strings mode (chunk IDs only)
    for key in (
        "retrieved_chunk_ids",
        "reranked_chunk_ids",
        "ranked_chunk_ids",
        "candidate_chunk_ids",
    ):
        values = rec.get(key)
        if isinstance(values, list) and values:
            entries: List[Tuple[str, str, Optional[float]]] = []
            for v in values:
                if isinstance(v, str):
                    entries.append((v, "", None))
                elif isinstance(v, dict):
                    cid = str(v.get("chunk_id", v.get("id", "")))
                    text = str(v.get("text", ""))
                    score = v.get("score")
                    entries.append((cid, text, float(score) if score is not None else None))
            return entries

    # List-of-dicts mode (with scores)
    for key in (
        "top_k",
        "ranked_chunks",
        "retrieved_chunks",
        "reranked_chunks",
        "results",
        "candidates",
    ):
        values = rec.get(key)
        if isinstance(values, list) and values:
            entries = []
            for v in values:
                if isinstance(v, dict):
                    cid, text = _extract_entry_id_text(v)
                    score = v.get("score")
                    entries.append((cid, text, float(score) if score is not None else None))
                elif isinstance(v, str):
                    entries.append((v, "", None))
            return entries

    return []


def convert_simple_format(
    filepath: Path,
    query_ids: Optional[set] = None,
) -> List[RankedEvidence]:
    """Convert a simple-format (exp4-style) results file to unified schema.

    Simple format has chunk IDs only (via ``retrieved_chunk_ids`` etc.) and
    requires downstream text lookup from a corpus cache.
    """
    results: List[RankedEvidence] = []
    with open(filepath, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            qid = str(rec.get("question_id", rec.get("query_id", "")))
            if not qid:
                continue
            if query_ids is not None and qid not in query_ids:
                continue
            entries = _extract_simple_entries(rec)
            for rank, (cid, text, score) in enumerate(entries, start=1):
                if cid:
                    results.append(RankedEvidence(
                        query_id=qid,
                        chunk_id=cid,
                        rank=rank,
                        score=score,
                        text=text or None,
                        doc_id=None,
                        source_file=str(filepath),
                    ))
    return results


def convert_from_file(
    filepath: Path,
    result_format: str,
    query_ids: Optional[set] = None,
) -> List[RankedEvidence]:
    """Dispatch to the correct converter based on format string."""
    if result_format == "rich":
        return convert_rich_format(filepath, query_ids)
    return convert_simple_format(filepath, query_ids)


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def deduplicate_evidence(
    evidence: List[RankedEvidence],
) -> List[RankedEvidence]:
    """Deduplicate by chunk_id within a query, keeping the best (lowest) rank.

    Input must be sorted by rank (as produced by the converters). Returns a new
    list with unique chunk_ids, preserving original rank order.
    """
    seen: set = set()
    result: List[RankedEvidence] = []
    for ev in evidence:
        if ev.chunk_id not in seen:
            seen.add(ev.chunk_id)
            result.append(ev)
    return result


# ---------------------------------------------------------------------------
# Token budget truncation
# ---------------------------------------------------------------------------

def _estimate_tokens(text: str) -> int:
    """Approximate token count using char-based heuristic.

    Rough estimate: ~4 chars per token for English text.  This is intentionally
    simple — the manifest records ``tokenizer_mode: approx_char_based`` so
    readers know the limitation.
    """
    if not text:
        return 0
    return max(1, len(text) // 4)


def truncate_by_token_budget(
    evidence: List[RankedEvidence],
    max_input_tokens: Optional[int] = None,
    max_tokens_per_evidence: Optional[int] = None,
    tokenizer_mode: str = "approx_char_based",
) -> Tuple[List[RankedEvidence], List[RankedEvidence], int]:
    """Truncate evidence list to fit within a token budget.

    Evidence is assumed to already be deduplicated and in rank order.

    Args:
        evidence: Deduplicated ranked evidence list.
        max_input_tokens: Total token budget for all evidence combined.
        max_tokens_per_evidence: Per-chunk token cap (truncates text if
            exceeded, but keeps the chunk).

    Returns:
        (used, dropped, estimated_prompt_tokens) where *used* are the chunks
        that fit within budget and *dropped* are the remainder.
    """
    used: List[RankedEvidence] = []
    dropped: List[RankedEvidence] = []
    total_tokens = 0
    unlimited = max_input_tokens is None

    for ev in evidence:
        text = ev.text or ""
        est = _estimate_tokens(text)
        if max_tokens_per_evidence is not None and est > max_tokens_per_evidence:
            est = max_tokens_per_evidence
        if unlimited or total_tokens + est <= (max_input_tokens or 0):
            used.append(ev)
            total_tokens += est
        else:
            dropped.append(ev)

    return used, dropped, total_tokens


# ---------------------------------------------------------------------------
# Chunk ID helpers
# ---------------------------------------------------------------------------

def extract_hash_from_chunk_id(chunk_id: str) -> Optional[str]:
    """Extract trailing hash from a structured chunk_id.

    Chunk IDs come in two forms:
      - ``chunk::doc_id::section::offset::hash``  (10-char hex)
      - bare 10-12 char hex string
    """
    parts = chunk_id.split("::")
    if len(parts) >= 2 and len(parts[-1]) == 10:
        return parts[-1]
    if 10 <= len(chunk_id) <= 12 and all(c in "0123456789abcdef" for c in chunk_id.lower()):
        return chunk_id
    return None


def register_chunk_meta(
    chunk_meta: Dict[str, Dict],
    chunk_id: str,
    text: str,
    doc_id: str = "",
) -> None:
    """Register a chunk in the metadata lookup, including hash alias."""
    if not chunk_id:
        return
    if chunk_id not in chunk_meta:
        chunk_meta[chunk_id] = {"text": text or "", "doc_id": doc_id or ""}
    hash_id = extract_hash_from_chunk_id(chunk_id)
    if hash_id and hash_id != chunk_id and hash_id not in chunk_meta:
        chunk_meta[hash_id] = {"text": text or "", "doc_id": doc_id or ""}
