"""Text and table chunking for financial reports.

Produces uniform Chunk objects that downstream retrieval and graph modules
consume.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple


# ═════════════════════════════════════════════════════════════════════════════
# Data structures
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class Chunk:
    """A single text or table chunk from a financial document."""

    chunk_id: str
    text: str
    chunk_type: str  # "text" | "table"
    doc_id: str = ""
    company: str = ""
    filing_type: str = ""  # "10-K" | "10-Q" | "8-K" | ...
    filing_year: str = ""  # e.g. "2023"
    section: str = ""
    metadata: Dict = field(default_factory=dict)


# ═════════════════════════════════════════════════════════════════════════════
# Public API
# ═════════════════════════════════════════════════════════════════════════════

def chunk_text(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    doc_id: str = "",
    company: str = "",
    filing_type: str = "",
    filing_year: str = "",
    section: str = "",
) -> List[Chunk]:
    """Split plain text into overlapping chunks by word boundaries."""
    chunks: List[Chunk] = []
    words = text.split()
    if not words:
        return chunks

    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words)
        chunks.append(
            Chunk(
                chunk_id=_uid(),
                text=chunk_text,
                chunk_type="text",
                doc_id=doc_id,
                company=company,
                filing_type=filing_type,
                filing_year=filing_year,
                section=section,
                metadata={"word_offset": start, "word_count": len(chunk_words)},
            )
        )
        if end == len(words):
            break
        start += chunk_size - chunk_overlap
    return chunks


def chunk_table_markdown(
    markdown_table: str,
    doc_id: str = "",
    company: str = "",
    filing_type: str = "",
    filing_year: str = "",
    section: str = "",
    max_rows_per_chunk: int = 20,
) -> List[Chunk]:
    """Split a large markdown table into smaller row-bounded chunks.

    Keeps the header row in every chunk so each chunk is self-contained.
    """
    lines = markdown_table.strip().split("\n")
    if len(lines) < 2:
        return []

    header = lines[0:2]  # header + separator
    body = lines[2:]

    chunks: List[Chunk] = []
    for i in range(0, len(body), max_rows_per_chunk):
        block = body[i : i + max_rows_per_chunk]
        chunk_text = "\n".join(header + block)
        chunks.append(
            Chunk(
                chunk_id=_uid(),
                text=chunk_text,
                chunk_type="table",
                doc_id=doc_id,
                company=company,
                filing_type=filing_type,
                filing_year=filing_year,
                section=section,
                metadata={"row_offset": i, "row_count": len(block)},
            )
        )
    return chunks


def chunk_report(
    report_path: Path,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> List[Chunk]:
    """Chunk an entire 10-K / financial report text file.

    Tries to detect sections (Item 1, Item 1A, …, Item 8) and labels chunks
    accordingly. Parses company/ticker/year from filename pattern like
    ``AAPL_2023_10K.txt``.
    """
    raw = report_path.read_text(encoding="utf-8", errors="replace")
    doc_id = report_path.stem

    # Parse filing metadata from filename
    company, filing_year, filing_type = _parse_filing_filename(report_path.stem)

    # Naïve section split on SEC "Item N." headings
    section_chunks: List[Chunk] = []
    sections = _split_sections(raw)
    for sec_title, sec_text in sections:
        section_chunks.extend(
            chunk_text(
                sec_text,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                doc_id=doc_id,
                company=company,
                filing_type=filing_type,
                filing_year=filing_year,
                section=sec_title,
            )
        )
    return section_chunks


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════

_FILING_FILENAME_RE = re.compile(
    r"^(?P<ticker>[A-Z]+)_(?P<year>\d{4})_(?P<type>10-[KQ]|8-K)",
    re.IGNORECASE,
)

_SECTION_RE = re.compile(r"^(Item\s+\d+[A-Z]?\.)", re.MULTILINE | re.IGNORECASE)


def _parse_filing_filename(stem: str) -> Tuple[str, str, str]:
    """Extract company, filing_year, filing_type from filename stem.

    Returns:
        (company, filing_year, filing_type) — defaults to ("", "", "") on failure.
    """
    m = _FILING_FILENAME_RE.match(stem)
    if m:
        return m.group("ticker"), m.group("year"), m.group("type").upper()
    # Fallback: try to extract year
    year_m = re.search(r"(19|20)\d{2}", stem)
    return "", year_m.group(0) if year_m else "", ""


def _split_sections(text: str) -> List[Tuple[str, str]]:
    """Split raw 10-K text into (section_title, section_body) pairs."""
    matches = list(_SECTION_RE.finditer(text))
    if not matches:
        return [("Full Document", text)]

    sections: List[Tuple[str, str]] = []
    for i, m in enumerate(matches):
        title = m.group(1).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections.append((title, text[start:end].strip()))
    return sections


def _uid() -> str:
    return uuid.uuid4().hex[:12]
