"""Entity extraction for financial evidence graphs.

Extracts metrics, years, companies, and report sections from text chunks.
Lightweight regex-based approach — upgrade to NER or FinBERT for production.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from feg_rag.data.chunker import Chunk


# ═════════════════════════════════════════════════════════════════════════════
# Patterns (extend these lists for better coverage)
# ═════════════════════════════════════════════════════════════════════════════

_METRIC_PATTERNS: List[str] = [
    r"\b(revenue|revenues|sales|net\s+sales)\b",
    r"\b(net\s+income|net\s+earnings|net\s+loss|profit)\b",
    r"\b(operating\s+income|operating\s+earnings|operating\s+loss)\b",
    r"\b(gross\s+profit|gross\s+margin)\b",
    r"\b(eps|earnings\s+per\s+share|diluted\s+eps|basic\s+eps)\b",
    r"\b(ebitda|adjusted\s+ebitda|ebit)\b",
    r"\b(total\s+assets|total\s+liabilities|total\s+equity)\b",
    r"\b(cash\s+and\s+cash\s+equivalents|cash\s+flow)\b",
    r"\b(free\s+cash\s+flow|operating\s+cash\s+flow)\b",
    r"\b(operating\s+expenses|r\s*&?\s*d\s+expenses?|research\s+and\s+development)\b",
    r"\b(sg\s*&?\s*a|selling\s+general\s+and\s+administrative)\b",
    r"\b(cost\s+of\s+(revenue|sales|goods\s+sold)|cogs)\b",
    r"\b(shareholders?.?\s*equity|stockholders?.?\s*equity)\b",
    r"\b(working\s+capital|long[\s-]term\s+debt|short[\s-]term\s+debt)\b",
    r"\b(dividends?\s+(per\s+share)?|capital\s+expenditures?|capex)\b",
    r"\b(return\s+on\s+(equity|assets|invested\s+capital)|roe|roa|roic)\b",
    r"\b(market\s+cap(italization)?|enterprise\s+value)\b",
]

_YEAR_PATTERN = re.compile(
    r"\b((?:19|20)\d{2})\b"
    r"|(FY\s*?(?:19|20)\d{2})"
    r"|(fiscal\s+year\s+(?:19|20)\d{2})",
    re.IGNORECASE,
)

# Simple company name patterns — very approximate; use NER for production
_COMPANY_PATTERN = re.compile(
    r"\b([A-Z][a-z]+(?:\s+(?:Inc\.?|Corp\.?|Corporation|Ltd\.?|Limited|"
    r"PLC|LLC|Group|Holdings?|International|Technologies?)))\b"
)

_FILING_TYPE_RE = re.compile(
    r"\b(10-K|10-Q|8-K|20-F|40-F|S-1|S-3)\b", re.IGNORECASE
)

_METRIC_RE = re.compile("|".join(_METRIC_PATTERNS), re.IGNORECASE)


# ═════════════════════════════════════════════════════════════════════════════
# Dataclass
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class ExtractedEntities:
    """Entities found in a chunk."""

    chunk_id: str
    metrics: Set[str] = field(default_factory=set)
    years: Set[str] = field(default_factory=set)
    companies: Set[str] = field(default_factory=set)
    filing_types: Set[str] = field(default_factory=set)
    sections: Set[str] = field(default_factory=set)


# ═════════════════════════════════════════════════════════════════════════════
# Public API
# ═════════════════════════════════════════════════════════════════════════════

class EntityExtractor:
    """Extract financial entities from text using regex rules."""

    def extract(self, chunk: Chunk) -> ExtractedEntities:
        return ExtractedEntities(
            chunk_id=chunk.chunk_id,
            metrics=self.extract_metrics(chunk.text),
            years=self.extract_years(chunk.text),
            companies=self.extract_companies(chunk.text),
            filing_types=self.extract_filing_types(chunk.text),
            sections=self.extract_sections(chunk),
        )

    @staticmethod
    def extract_metrics(text: str) -> Set[str]:
        return {m.group(0).strip().lower() for m in _METRIC_RE.finditer(text)}

    @staticmethod
    def extract_years(text: str) -> Set[str]:
        years: Set[str] = set()
        for m in _YEAR_PATTERN.finditer(text):
            # pick the first non-None group
            y = next((g for g in m.groups() if g is not None), None)
            if y:
                # normalise "FY 2023" → "2023"
                digits = re.search(r"(19|20)\d{2}", y)
                if digits:
                    years.add(digits.group(0))
        return years

    @staticmethod
    def extract_companies(text: str) -> Set[str]:
        return {m.group(1).strip() for m in _COMPANY_PATTERN.finditer(text)}

    @staticmethod
    def extract_filing_types(text: str) -> Set[str]:
        return {m.group(0).strip().upper() for m in _FILING_TYPE_RE.finditer(text)}

    @staticmethod
    def extract_sections(chunk: Chunk) -> Set[str]:
        """Derive section from chunk metadata."""
        sections: Set[str] = set()
        if chunk.section:
            sections.add(chunk.section)
        if chunk.filing_type:
            sections.add(f"filing:{chunk.filing_type}")
        return sections


def extract_entities(chunks: List[Chunk]) -> Dict[str, ExtractedEntities]:
    """Batch-extract entities from a list of chunks.

    Returns:
        Mapping from chunk_id → ExtractedEntities.
    """
    extractor = EntityExtractor()
    return {c.chunk_id: extractor.extract(c) for c in chunks}
