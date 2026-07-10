"""Hard-negative generation for financial RAG.

Generates distractors that share surface features with gold evidence but are
structurally wrong (wrong year, wrong metric, wrong company, etc.).

Includes a :class:`CorpusIndex` for fast lookups on large corpora (23k+ chunks).
"""

from __future__ import annotations

import random
import re
from collections import defaultdict
from typing import Dict, List, Optional, Set

from feg_rag.data.chunker import Chunk


# ═════════════════════════════════════════════════════════════════════════════
# Regex helpers (shared)
# ═════════════════════════════════════════════════════════════════════════════

_METRIC_PATTERNS = [
    r"\b(revenue|net\s+income|operating\s+income|gross\s+profit|eps|ebitda|ebit)\b",
    r"\b(total\s+assets|total\s+liabilities|cash\s+flow|free\s+cash\s+flow)\b",
    r"\b(operating\s+expenses|r\s*&?\s*d|sg\s*&?\s*a|cost\s+of\s+revenue)\b",
    r"\b(shareholders?.?\s*equity|working\s+capital|long.term\s+debt)\b",
]

_METRIC_RE = re.compile("|".join(_METRIC_PATTERNS), re.IGNORECASE)
_YEAR_RE = re.compile(r"\b((?:19|20)\d{2})\b")


def _extract_metrics(text: str) -> Set[str]:
    return {m.group(0).strip().lower() for m in _METRIC_RE.finditer(text)}


def _extract_years(text: str) -> Set[str]:
    return set(_YEAR_RE.findall(text))


def _collect_metrics(chunks: List[Chunk]) -> Set[str]:
    out: Set[str] = set()
    for c in chunks:
        out |= _extract_metrics(c.text)
    return out


def _collect_years(chunks: List[Chunk]) -> Set[str]:
    out: Set[str] = set()
    for c in chunks:
        out |= _extract_years(c.text)
    return out


# ═════════════════════════════════════════════════════════════════════════════
# Corpus index for fast hard-negative lookups
# ═════════════════════════════════════════════════════════════════════════════

class CorpusIndex:
    """Pre-built indices over a chunk corpus for fast hard-negative mining.

    Instead of scanning the full corpus and re-extracting entities per query
    (O(Q × C × regex)), build the indices once and use set operations.

    Indices:
        chunk_metrics: chunk_id → set of metric strings
        chunk_years:   chunk_id → set of year strings
        metric_chunks: metric → list of chunk_ids
        year_chunks:   year → list of chunk_ids
        section_chunks: section → list of chunk_ids
    """

    def __init__(self, corpus: List[Chunk]):
        self.corpus = corpus
        self._chunk_by_id: Dict[str, Chunk] = {c.chunk_id: c for c in corpus}

        # chunk_id → set of metrics
        self.chunk_metrics: Dict[str, Set[str]] = {}
        # chunk_id → set of years
        self.chunk_years: Dict[str, Set[str]] = {}
        # metric → list of chunk_ids
        self.metric_chunks: Dict[str, List[str]] = defaultdict(list)
        # year → list of chunk_ids
        self.year_chunks: Dict[str, List[str]] = defaultdict(list)
        # section → list of chunk_ids
        self.section_chunks: Dict[str, List[str]] = defaultdict(list)

        self._build()

    def _build(self) -> None:
        """Scan the corpus once and populate all indices."""
        for c in self.corpus:
            metrics = _extract_metrics(c.text)
            years = _extract_years(c.text)

            self.chunk_metrics[c.chunk_id] = metrics
            self.chunk_years[c.chunk_id] = years

            for m in metrics:
                self.metric_chunks[m].append(c.chunk_id)
            for y in years:
                self.year_chunks[y].append(c.chunk_id)

            if c.section:
                self.section_chunks[c.section].append(c.chunk_id)

    # ------------------------------------------------------------------
    # Fast queries
    # ------------------------------------------------------------------

    def chunks_with_metric(self, metric: str) -> Set[str]:
        return set(self.metric_chunks.get(metric, []))

    def chunks_with_year(self, year: str) -> Set[str]:
        return set(self.year_chunks.get(year, []))

    def chunks_with_any_metric(self, metrics: Set[str]) -> Set[str]:
        out: Set[str] = set()
        for m in metrics:
            out.update(self.metric_chunks.get(m, []))
        return out

    def chunks_with_any_year(self, years: Set[str]) -> Set[str]:
        out: Set[str] = set()
        for y in years:
            out.update(self.year_chunks.get(y, []))
        return out

    def chunks_in_sections(self, sections: Set[str]) -> Set[str]:
        out: Set[str] = set()
        for s in sections:
            out.update(self.section_chunks.get(s, []))
        return out

    def get_chunk(self, chunk_id: str) -> Optional[Chunk]:
        return self._chunk_by_id.get(chunk_id)

    @property
    def num_chunks(self) -> int:
        return len(self.corpus)

    def __repr__(self) -> str:
        return (f"CorpusIndex(chunks={self.num_chunks}, "
                f"metrics={len(self.metric_chunks)}, "
                f"years={len(self.year_chunks)}, "
                f"sections={len(self.section_chunks)})")


# ═════════════════════════════════════════════════════════════════════════════
# Public API
# ═════════════════════════════════════════════════════════════════════════════

def generate_hard_negatives(
    gold_chunks: List[Chunk],
    corpus: List[Chunk],
    num_negatives: int = 10,
    seed: int = 42,
) -> List[Chunk]:
    """Create a pool of hard negatives for a given query.

    Strategies (inspired by the paper plan §5.4):
      1. Same metric, wrong year
      2. Same year, wrong metric
      3. Same section, unrelated content
      4. Semantically similar but irrelevant
      5. Random (easy negatives)

    Args:
        gold_chunks: The gold evidence chunks for this query.
        corpus: All available chunks in the retrieval corpus.
        num_negatives: Target number of hard negatives to generate.
        seed: Random seed.

    Returns:
        List of hard-negative chunks.
    """
    rng = random.Random(seed)
    negatives: List[Chunk] = []
    gold_ids = {c.chunk_id for c in gold_chunks}

    # Collect metadata from gold chunks for targeted mining
    gold_metrics = _collect_metrics(gold_chunks)
    gold_years = _collect_years(gold_chunks)
    gold_sections = {c.section for c in gold_chunks if c.section}
    gold_docs = {c.doc_id for c in gold_chunks if c.doc_id}

    # Strategy buckets
    same_metric_wrong_year: List[Chunk] = []
    same_year_wrong_metric: List[Chunk] = []
    same_section: List[Chunk] = []
    rest: List[Chunk] = []

    for c in corpus:
        if c.chunk_id in gold_ids:
            continue
        c_metrics = _extract_metrics(c.text)
        c_years = _extract_years(c.text)

        if gold_metrics & c_metrics and not (gold_years & c_years):
            same_metric_wrong_year.append(c)
        elif gold_years & c_years and not (gold_metrics & c_metrics):
            same_year_wrong_metric.append(c)
        elif c.section and c.section in gold_sections:
            same_section.append(c)
        else:
            rest.append(c)

    # Fill negatives from each bucket proportionally
    buckets = [
        same_metric_wrong_year,
        same_year_wrong_metric,
        same_section,
        rest,
    ]
    per_bucket = max(1, num_negatives // len(buckets))

    for bucket in buckets:
        rng.shuffle(bucket)
        negatives.extend(bucket[:per_bucket])

    rng.shuffle(negatives)
    return negatives[:num_negatives]


def generate_hard_negatives_fast(
    gold_chunks: List[Chunk],
    corpus_index: CorpusIndex,
    num_negatives: int = 10,
    seed: int = 42,
) -> Dict[str, List[Chunk]]:
    """Fast hard-negative generation using pre-built :class:`CorpusIndex`.

    Uses set operations on the index instead of scanning the full corpus per
    query.  O(|gold_metrics| + |gold_years|) per query instead of O(|corpus|).

    Returns a dict keyed by strategy name for per-strategy diagnostics.
    """
    rng = random.Random(seed)
    gold_ids = {c.chunk_id for c in gold_chunks}

    gold_metrics = _collect_metrics(gold_chunks)
    gold_years = _collect_years(gold_chunks)
    gold_sections = {c.section for c in gold_chunks if c.section}

    # ── strategy 1: same metric, wrong year ──
    same_metric_ids = corpus_index.chunks_with_any_metric(gold_metrics) - gold_ids
    if gold_years:
        same_year_ids = corpus_index.chunks_with_any_year(gold_years)
        swm_candidates = list(same_metric_ids - same_year_ids)
    else:
        swm_candidates = list(same_metric_ids)
    rng.shuffle(swm_candidates)

    # ── strategy 2: same year, wrong metric ──
    same_year_ids = corpus_index.chunks_with_any_year(gold_years) - gold_ids
    if gold_metrics:
        swy_candidates = list(same_year_ids - same_metric_ids)
    else:
        swy_candidates = list(same_year_ids)
    rng.shuffle(swy_candidates)

    # ── strategy 3: same section ──
    same_section_ids = corpus_index.chunks_in_sections(gold_sections) - gold_ids
    ss_candidates = list(same_section_ids)
    rng.shuffle(ss_candidates)

    # ── strategy 4: random / easy negatives (fallback) ──
    all_ids = set(corpus_index._chunk_by_id.keys())
    rest_ids = all_ids - gold_ids - set(swm_candidates) - set(swy_candidates) - set(ss_candidates)
    rest_candidates = list(rest_ids)
    rng.shuffle(rest_candidates)

    per_bucket = max(1, num_negatives // 4)

    strategies: Dict[str, List[Chunk]] = {}
    for name, candidates in [
        ("same_metric_wrong_year", swm_candidates),
        ("same_year_wrong_metric", swy_candidates),
        ("same_section", ss_candidates),
        ("random_fallback", rest_candidates),
    ]:
        result: List[Chunk] = []
        for cid in candidates[:per_bucket]:
            c = corpus_index.get_chunk(cid)
            if c is not None:
                result.append(c)
        strategies[name] = result

    return strategies
