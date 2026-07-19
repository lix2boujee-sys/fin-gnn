"""Finance-specific structural error analysis of reranked evidence.

Analyses top-k evidence from each method BEFORE generation for structural errors:
    - Wrong Company: evidence from wrong company before first gold hit
    - Wrong Year: evidence for wrong fiscal year before first gold hit
    - Wrong Metric: evidence for wrong financial metric before first gold hit
    - Missing Evidence: no gold evidence in top-k

Reads pre-computed ranked results from multiple experiment output directories,
analyses the reranked evidence passages, and produces tables/CSV/JSON/debug cases.

Usage:
    python experiments/finance_error_analysis.py
    python experiments/finance_error_analysis.py --k 10 --output_dir outputs/error_analysis
    python experiments/finance_error_analysis.py --methods "Initial Retriever,PPR,GraphSAGE"
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from feg_rag.config import Config
from feg_rag.data.chunker import Chunk, chunk_text
from feg_rag.data.loader import load_dataset
from feg_rag.graph.entities import EntityExtractor


# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

CANONICAL_METHODS = [
    "Initial Retriever",
    "Cross-Encoder",
    "MonoT5",
    "PPR",
    "GraphSAGE",
    "R-GCN",
    "GATv2",
    "FinDual-GNN (Ours)",
]

# Map canonical method names to (directory, file_pattern, format)
# format: "rich" = exp1-style with top_k metadata, "simple" = exp4-style with chunk_ids only
METHOD_FILE_MAP: Dict[str, List[Dict]] = {
    "Initial Retriever": [
        {"dir_glob": "outputs/v2_table1_bge_m3_correct_corpus_*", "file": "bge_m3_dense_results.jsonl", "format": "simple"},
        {"dir": "outputs/exp1_baseline", "file": "hybrid_results.jsonl", "format": "rich"},
        {"dir": "outputs/exp1_baseline", "file": "bm25_results.jsonl", "format": "rich"},
        {"dir": "outputs/exp3_feg_ppr", "file": "hybrid_results.jsonl", "format": "simple"},
    ],
    "Cross-Encoder": [
        {"dir_glob": "outputs/v2_table2_cross_encoder_*", "file": "cross_encoder_results.jsonl", "format": "simple"},
        {"dir": "outputs/table1_non_llm_reranking", "file": "cross_encoder_results.jsonl", "format": "simple"},
        {"dir": "outputs/exp3_verify", "file": "cross_encoder_results.jsonl", "format": "simple"},
    ],
    "MonoT5": [
        {"dir_glob": "outputs/v2_table2_mono_t5_bge_pool_*", "file": "mono_t5_results.jsonl", "format": "simple"},
        {"dir": "outputs/table1_non_llm_reranking", "file": "monot5_results.jsonl", "format": "simple"},
    ],
    "PPR": [
        {"dir_glob": "outputs/v2_table2_graph_bge_pool_a_ppr_sage_*", "file": "ppr_results.jsonl", "format": "simple"},
        {"dir": "outputs/exp4_gnn_fulltest", "file": "hybrid_ppr_results.jsonl", "format": "simple"},
        {"dir": "outputs/exp4_gnn_reranker_full_20260709_214507", "file": "hybrid_ppr_results.jsonl", "format": "simple"},
        {"dir": "outputs/exp3_feg_ppr", "file": "ppr_results_full_graph.jsonl", "format": "simple"},
        {"dir": "outputs/exp3_verify", "file": "ppr_results_full_graph.jsonl", "format": "simple"},
    ],
    "GraphSAGE": [
        {"dir_glob": "outputs/v2_table2_graph_bge_pool_checkpoint_eval_fast_*", "file": "graphsage_results.jsonl", "format": "simple"},
        {"dir": "outputs/exp4_gnn_fulltest", "file": "hybrid_sage_results.jsonl", "format": "simple"},
        {"dir": "outputs/exp4_gnn_reranker_full_20260709_214507", "file": "hybrid_sage_results.jsonl", "format": "simple"},
        {"dir": "outputs/exp4_gnn_reranker", "file": "hybrid_sage_results.jsonl", "format": "simple"},
    ],
    "R-GCN": [
        {"dir_glob": "outputs/v2_table2_graph_bge_pool_rgcn_eval_fast_*", "file": "rgcn_results.jsonl", "format": "simple"},
        {"dir": "outputs/table1_non_llm_reranking", "file": "rgcn_results.jsonl", "format": "simple"},
        {"dir": "outputs/exp4_gnn_fulltest", "file": "hybrid_rgcn_results.jsonl", "format": "simple"},
        {"dir": "outputs/exp4_gnn_reranker_full_20260709_214507", "file": "hybrid_rgcn_results.jsonl", "format": "simple"},
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
        {"dir": "outputs/table1_non_llm_reranking", "file": "c2_dcf_gnn_results.jsonl", "format": "simple"},
        {"dir": "outputs/exp4_gnn_fulltest", "file": "hybrid_c2_dcf_results.jsonl", "format": "simple"},
    ],
}

# Comparison query patterns (multiple valid years)
COMPARISON_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r"from\s+(?:FY\s*)?(?:19|20)\d{2}\s+to\s+(?:FY\s*)?(?:19|20)\d{2}",
        r"between\s+(?:FY\s*)?(?:19|20)\d{2}\s+and\s+(?:FY\s*)?(?:19|20)\d{2}",
        r"change\s+(?:from\s+)?(?:FY\s*)?(?:19|20)\d{2}",
        r"compared\s+to\s+(?:FY\s*)?(?:19|20)\d{2}",
        r"from\s+\d{2,4}\s*[-–]\s*\d{2,4}",
        r"(?:19|20)\d{2}\s*[-–]\s*(?:19|20)\d{2}",
        r"growth\s+(?:from|since)\s+(?:19|20)\d{2}",
        r"increase\s+(?:from|since)\s+(?:19|20)\d{2}",
        r"decrease\s+(?:from|since)\s+(?:19|20)\d{2}",
        r"delta\s+in\s+.*?\d{2,4}\s*[-–]\s*\d{2,4}",
        r"trend\s+(?:from|since)\s+(?:19|20)\d{2}",
        r"YoY|year[-\s]over[-\s]year|year[-\s]on[-\s]year",
    ]
]

# Company indicator patterns in queries
COMPANY_INDICATOR_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r"\bfor\s+([A-Z][A-Za-z0-9\s&.,]+?)(?:[\s,]+(?:from|in|since|for|during|as|what|how|does|did|is|are|was|were|has|have|will|can|should|would|could|calculate|compute|find|determine|analyse|analyze|estimate))",
        r"\bof\s+([A-Z][A-Za-z0-9\s&.,]+?)(?:[\s,]+(?:from|in|since|for|during|as|what|how|does|did|is|are|was|were))",
        r"\b([A-Z][A-Za-z0-9\s&.,]+?)\s*\(([A-Z]{1,5})\)",  # "Palo Alto Networks (PANW)"
        r"\bticker\s+([A-Z]{1,5})\b",
        r"\bfor\s+ticker\s+([A-Z]{1,5})\b",
        # Multi-word company name with entity suffix (handles "Cboe Global Markets Inc.")
        r"\b((?:[A-Z][a-z]+\s+){1,4}(?:Inc\.?|Corp\.?|Corporation|Ltd\.?|Limited|PLC|LLC|Group|Holdings?|International|Technologies?|Therapeutics|Pharmaceuticals?|Enterprises?|Communications?|Industries?))\b",
        # Single-word company with entity suffix
        r"\b([A-Z][a-z]+(?:Inc\.?|Corp\.?|Corporation|Ltd\.?|Limited|PLC|LLC))\b",
    ]
]

# Improved company extraction for passages: match multi-word company names
_IMPROVED_COMPANY_RE = re.compile(
    r"\b((?:[A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*){0,4})"
    r"\s+(?:Inc\.?|Corp\.?|Corporation|Ltd\.?|Limited|PLC|LLC|"
    r"Group|Holdings?|International|Technologies?|Therapeutics|"
    r"Pharmaceuticals?|Enterprises?|Communications?|Industries?|"
    r"Airlines?|Financial|Bancorp|Capital|Partners?))\b"
)


# ═══════════════════════════════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class QueryEntities:
    """Entities extracted from a query."""
    query_id: str
    companies: Set[str] = field(default_factory=set)
    years: Set[str] = field(default_factory=set)
    metrics: Set[str] = field(default_factory=set)
    canonical_metric_groups: Set[str] = field(default_factory=set)
    is_comparison: bool = False
    has_company_requirement: bool = False
    has_year_requirement: bool = False
    has_metric_requirement: bool = False


@dataclass
class PassageInfo:
    """Information about a retrieved passage."""
    rank: int
    passage_id: str
    text: str
    score: float = 0.0
    is_gold: bool = False
    companies: Set[str] = field(default_factory=set)
    years: Set[str] = field(default_factory=set)
    metrics: Set[str] = field(default_factory=set)
    canonical_metric_groups: Set[str] = field(default_factory=set)


@dataclass
class QueryErrorResult:
    """Error analysis result for one query + method."""
    query_id: str
    query_text: str
    method: str
    has_wrong_company: bool = False
    has_wrong_year: bool = False
    has_wrong_metric: bool = False
    has_missing_evidence: bool = False
    error_types: List[str] = field(default_factory=list)
    first_gold_rank: Optional[int] = None
    top_k_passages: List[PassageInfo] = field(default_factory=list)
    gold_evidence_ids: List[str] = field(default_factory=list)
    query_entities: Optional[QueryEntities] = None
    explanation: str = ""


# ═══════════════════════════════════════════════════════════════════════════════
# Financial Metric Lexicon
# ═══════════════════════════════════════════════════════════════════════════════

class MetricLexicon:
    """Load and query the financial metric lexicon."""

    def __init__(self, lexicon_path: Optional[Path] = None):
        if lexicon_path is None:
            lexicon_path = Path(__file__).resolve().parents[1] / "configs" / "financial_metric_lexicon.json"
        self.lexicon = self._load(lexicon_path)
        self._pattern_to_canonical: Dict[str, str] = {}
        self._build_index()

    @staticmethod
    def _load(path: Path) -> dict:
        if path.exists():
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        print(f"  [WARN] Metric lexicon not found at {path}, using built-in defaults")
        return {"metric_groups": {}, "canonical_to_display": {}}

    def _build_index(self) -> None:
        for group_name, group_info in self.lexicon.get("metric_groups", {}).items():
            canonical = group_info.get("canonical", group_name)
            for pattern in group_info.get("patterns", []):
                self._pattern_to_canonical[pattern.lower().strip()] = canonical

    def get_canonical_group(self, metric_text: str) -> Optional[str]:
        """Map a metric surface form to its canonical group."""
        text_lower = metric_text.lower().strip()
        # Exact match first
        if text_lower in self._pattern_to_canonical:
            return self._pattern_to_canonical[text_lower]
        # Substring match (longest first)
        for pattern, canonical in sorted(self._pattern_to_canonical.items(),
                                          key=lambda x: -len(x[0])):
            if pattern in text_lower or text_lower in pattern:
                return canonical
        return None

    def get_canonical_groups(self, metric_texts: Set[str]) -> Set[str]:
        """Get canonical groups for a set of metric strings."""
        groups: Set[str] = set()
        for mt in metric_texts:
            g = self.get_canonical_group(mt)
            if g:
                groups.add(g)
        return groups


# ═══════════════════════════════════════════════════════════════════════════════
# Query Entity Extractor
# ═══════════════════════════════════════════════════════════════════════════════

class QueryEntityExtractor:
    """Extract financial entities from query text for error analysis.

    Reuses EntityExtractor for years and basic patterns, adds:
    - Company requirement detection
    - Comparison query detection
    - Metric requirement detection using lexicon
    """

    _TICKER_RE = re.compile(r"\b\(?([A-Z]{1,5})\)?\b")
    _YEAR_RE = re.compile(r"\b((?:19|20)\d{2})\b")

    def __init__(self, metric_lexicon: Optional[MetricLexicon] = None):
        self.entity_extractor = EntityExtractor()
        self.metric_lexicon = metric_lexicon or MetricLexicon()

    def extract(self, query_id: str, query_text: str) -> QueryEntities:
        """Extract all entities from a query."""
        qe = QueryEntities(query_id=query_id)

        # Years (extract from text + handle ranges like "2021-23")
        qe.years = self._extract_query_years(query_text)

        # Companies (use both EntityExtractor and custom patterns)
        qe.companies = self._extract_query_companies(query_text)

        # Metrics (only count CLEAR financial metric mentions)
        raw_metrics = self.entity_extractor.extract_metrics(query_text)
        # Filter out overly generic terms that aren't real metric requirements
        _generic_terms = {"profit", "cost", "costs", "expense", "expenses",
                          "income", "loss", "losses", "gain", "gains",
                          "margin", "margins", "value", "values", "price",
                          "prices", "fee", "fees", "charge", "charges",
                          "performance", "health", "stability"}
        qe.metrics = {m for m in raw_metrics if m not in _generic_terms}
        qe.canonical_metric_groups = self.metric_lexicon.get_canonical_groups(qe.metrics)

        # Comparison detection
        qe.is_comparison = self._is_comparison_query(query_text)

        # Requirements
        qe.has_company_requirement = len(qe.companies) > 0
        qe.has_year_requirement = len(qe.years) > 0
        qe.has_metric_requirement = len(qe.canonical_metric_groups) > 0

        return qe

    def _extract_query_years(self, query_text: str) -> Set[str]:
        """Extract years from query text, including range expansions."""
        years = self.entity_extractor.extract_years(query_text)

        # Expand year ranges: "2021-23" → {2021, 2023}
        range_re = re.compile(r"\b((?:19|20)(\d{2}))\s*[-–]\s*(\d{2})\b")
        for m in range_re.finditer(query_text):
            prefix = m.group(1)[:2]  # "20" from "2021"
            start_year = m.group(1)   # "2021"
            end_suffix = m.group(3)   # "23"
            # Determine full end year
            if len(end_suffix) == 2:
                end_year = prefix + end_suffix  # "2023"
            else:
                end_year = end_suffix
            years.add(start_year)
            years.add(end_year)

        # Expand "from YYYY to YYYY" patterns
        from_to_re = re.compile(r"\b(?:from|between)\s+((?:19|20)\d{2})\s+(?:to|and|through)\s+((?:19|20)\d{2})\b", re.IGNORECASE)
        for m in from_to_re.finditer(query_text):
            years.add(m.group(1))
            years.add(m.group(2))

        return years

    def _extract_query_companies(self, query_text: str) -> Set[str]:
        """Extract company mentions from query text.

        Uses multiple strategies: full company names, tickers, and context patterns.
        """
        companies: Set[str] = set()

        # Strategy 1: EntityExtractor's pattern (may miss some)
        extractor_companies = self.entity_extractor.extract_companies(query_text)
        companies.update(c.strip().lower() for c in extractor_companies)

        # Strategy 2: Improved multi-word company pattern on the query text itself
        for m in _IMPROVED_COMPANY_RE.finditer(query_text):
            companies.add(m.group(1).strip().lower())

        # Strategy 3: Look for ticker patterns
        for pattern in COMPANY_INDICATOR_PATTERNS:
            for m in pattern.finditer(query_text):
                company_text = m.group(1).strip().rstrip(".,;:")
                if company_text and len(company_text) >= 2:
                    companies.add(company_text.lower())

        # Strategy 4: Extract standalone tickers (case-insensitive)
        ticker_re_cis = re.compile(r"\b([A-Za-z]{2,5})\b")
        for m in ticker_re_cis.finditer(query_text):
            t = m.group(1)
            # Only add if it looks like a ticker (all-caps or mixed-cap with context)
            if t.isupper() and len(t) >= 2 and len(t) <= 5:
                # Check if it's a known non-company word
                if t.lower() not in {"the", "and", "for", "from", "with", "that", "this",
                                      "has", "its", "rev", "cap", "cba", "eps", "roa", "roe",
                                      "yoy", "qoq", "fye", "mkt", "reg", "fee", "tax"}:
                    companies.add(t.lower())
            elif any(c.isupper() for c in t) and len(t) >= 3 and len(t) <= 6:
                # Mixed-case potential ticker like "Cboe", "Schw"
                if t[0].isupper() and t.lower() not in {"the", "from", "since", "during",
                                                         "does", "what", "how", "when", "where",
                                                         "which", "impact", "delta", "change",
                                                         "growth", "trend", "analysis"}:
                    companies.add(t.lower())

        # Strategy 5: Look for "for ticker, Full Company Name" patterns
        ticker_fullname_re = re.compile(
            r"\bfor\s+([A-Z]{1,5}),\s+([A-Z][A-Za-z\s&.]+?)(?:\s*,|\s+from|\s+in|\s+for|\s*$)",
            re.IGNORECASE
        )
        for m in ticker_fullname_re.finditer(query_text):
            companies.add(m.group(1).lower())  # ticker
            companies.add(m.group(2).strip().lower())  # full name

        # Clean up: remove common non-company words
        non_companies = {"the", "this", "that", "these", "those", "its", "their",
                         "what", "how", "when", "where", "which", "who", "why",
                         "calculate", "compute", "find", "determine", "analyse",
                         "analyze", "estimate", "implications", "impact", "analysis",
                         "change", "delta", "difference", "trend", "growth", "from",
                         "since", "during", "as", "of", "in", "for", "to", "by", "at",
                         "does", "did", "is", "are", "was", "were", "has", "have",
                         "will", "can", "should", "would", "could", "one", "two",
                         "data", "access", "solutions", "fee", "mkt", "share", "profit",
                         "cost", "market", "order", "flow", "news", "operations",
                         "authorization", "alloc", "repurchase", "litigation", "proceedings",
                         "disclosure", "disclosures", "overview"}
        companies = {c for c in companies if c not in non_companies and len(c) > 1}

        return companies

    def _is_comparison_query(self, query_text: str) -> bool:
        """Check if query is a comparison/trend query with multiple valid years."""
        qt_lower = query_text.lower()
        for pattern in COMPARISON_PATTERNS:
            if pattern.search(qt_lower):
                return True
        # Also check: multiple years mentioned
        years = self._YEAR_RE.findall(query_text)
        if len(set(years)) >= 2:
            return True
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# Passage Entity Extractor (for simple-format results without metadata)
# ═══════════════════════════════════════════════════════════════════════════════

class PassageEntityExtractor:
    """Extract entities from passage text for error analysis."""

    def __init__(self, metric_lexicon: Optional[MetricLexicon] = None):
        self.entity_extractor = EntityExtractor()
        self.metric_lexicon = metric_lexicon or MetricLexicon()

    def extract(self, text: str) -> Dict:
        """Extract entities from a passage text."""
        companies: Set[str] = set()

        # Use EntityExtractor's built-in pattern
        companies.update(self.entity_extractor.extract_companies(text))

        # Use improved multi-word company regex
        for m in _IMPROVED_COMPANY_RE.finditer(text):
            companies.add(m.group(1).strip().lower())

        years = self.entity_extractor.extract_years(text)
        metrics = self.entity_extractor.extract_metrics(text)
        canonical_groups = self.metric_lexicon.get_canonical_groups(metrics)

        return {
            "companies": companies,
            "years": years,
            "metrics": metrics,
            "canonical_metric_groups": canonical_groups,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Result Loaders
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_year_set(year_str: str) -> Set[str]:
    """Parse comma-separated year string into a set."""
    if not year_str:
        return set()
    return {y.strip() for y in year_str.split(",") if y.strip()}


def _parse_metric_set(metric_str: str) -> Set[str]:
    """Parse comma-separated metric string into a set."""
    if not metric_str:
        return set()
    return {m.strip().lower() for m in metric_str.split(",") if m.strip()}


def load_rich_format_results(filepath: Path) -> Dict[str, Dict]:
    """Load exp1-style results, indexed by query_id."""
    records: Dict[str, Dict] = {}
    with open(filepath, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            qid = rec.get("query_id", "")
            if qid:
                records[qid] = rec
    return records


def load_simple_format_results(filepath: Path) -> Dict[str, Dict]:
    """Load exp4-style results, indexed by question_id."""
    records: Dict[str, Dict] = {}
    with open(filepath, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            qid = rec.get("question_id", rec.get("query_id", ""))
            if qid:
                records[qid] = rec
    return records


def find_method_file(canonical_name: str, root_dir: Path) -> Tuple[Optional[Path], str]:
    """Find the result file for a canonical method name.

    Returns:
        (filepath, format) or (None, "") if not found.
    """
    candidates = METHOD_FILE_MAP.get(canonical_name, [])
    for candidate in candidates:
        if "dir_glob" in candidate:
            dirs = sorted(
                (p for p in root_dir.glob(candidate["dir_glob"]) if p.is_dir()),
                key=lambda p: p.name,
                reverse=True,
            )
            for d in dirs:
                fpath = d / candidate["file"]
                if fpath.exists():
                    return fpath, candidate["format"]
        else:
            fpath = root_dir / candidate["dir"] / candidate["file"]
            if fpath.exists():
                return fpath, candidate["format"]
    return None, ""


# ═══════════════════════════════════════════════════════════════════════════════
# Core Error Analysis
# ═══════════════════════════════════════════════════════════════════════════════

class FinanceErrorAnalyzer:
    """Analyse reranked evidence for financial structural errors."""

    def __init__(
        self,
        query_entities: Dict[str, QueryEntities],
        gold_map: Dict[str, List[str]],
        metric_lexicon: Optional[MetricLexicon] = None,
        chunk_metadata: Optional[Dict[str, Dict]] = None,
        cross_ref: Optional[Dict[str, Dict]] = None,
    ):
        self.query_entities = query_entities
        self.gold_map = gold_map
        self.metric_lexicon = metric_lexicon or MetricLexicon()
        self.chunk_metadata = chunk_metadata or {}
        self._cross_ref = cross_ref or {}
        self.passage_extractor = PassageEntityExtractor(self.metric_lexicon)
        self.entity_extractor = EntityExtractor()

    def analyze_method(
        self,
        method: str,
        records: Dict[str, Dict],
        result_format: str,
        query_ids: List[str],
        top_k: int = 10,
    ) -> List[QueryErrorResult]:
        """Analyse specific queries for one method.

        Args:
            method: Canonical method name.
            records: Query-id → result record dict.
            result_format: "rich" or "simple".
            query_ids: List of query IDs to analyse (in desired order).
            top_k: Number of top passages to analyse.
        """
        results: List[QueryErrorResult] = []

        for query_id in query_ids:
            rec = records.get(query_id)
            if rec is None:
                # Record not found for this query — skip
                continue

            query_text = rec.get("question", "")
            gold_ids = rec.get("gold_evidence_ids", [])

            qe = self.query_entities.get(query_id)
            if qe is None:
                # Extract on the fly
                extractor = QueryEntityExtractor(self.metric_lexicon)
                qe = extractor.extract(query_id, query_text)
                self.query_entities[query_id] = qe

            # Parse top-k passages
            passages = self._parse_passages(rec, result_format, gold_ids, top_k)

            # Analyse errors
            error_result = self._analyze_query(
                query_id, query_text, method, passages, gold_ids, qe
            )
            results.append(error_result)

        return results

    def _parse_passages(
        self,
        rec: Dict,
        result_format: str,
        gold_ids: List[str],
        top_k: int,
    ) -> List[PassageInfo]:
        """Parse top-k passages from a result record."""
        passages: List[PassageInfo] = []
        gold_set = set(gold_ids)

        if result_format == "rich":
            top_entries = rec.get("top_k", [])[:top_k]
            for entry in top_entries:
                pid = entry.get("passage_id", "")
                text = entry.get("text", "")
                is_gold = entry.get("is_gold", False) or pid in gold_set

                # Company: always extract from text since pre-computed field may be a hash
                companies: Set[str] = set()
                if text:
                    companies = self.entity_extractor.extract_companies(text)

                # Year: use pre-computed field if available, otherwise from text
                years_raw = entry.get("year", "")
                years = _parse_year_set(years_raw)
                if not years and text:
                    years = self.entity_extractor.extract_years(text)

                # Metric: use pre-computed field if available, otherwise from text
                metrics_raw = entry.get("metric", "")
                metrics = _parse_metric_set(metrics_raw)
                if not metrics and text:
                    metrics = self.entity_extractor.extract_metrics(text)
                canonical_groups = self.metric_lexicon.get_canonical_groups(metrics)

                passages.append(PassageInfo(
                    rank=entry.get("rank", len(passages) + 1),
                    passage_id=pid,
                    text=text,
                    score=float(entry.get("score", 0)),
                    is_gold=is_gold,
                    companies=companies,
                    years=years,
                    metrics=metrics,
                    canonical_metric_groups=canonical_groups,
                ))
        else:
            # Simple format: chunk_ids only — need to look up metadata
            chunk_ids = rec.get("retrieved_chunk_ids", [])[:top_k]
            for rank, cid in enumerate(chunk_ids, 1):
                is_gold = cid in gold_set
                text = ""
                companies: Set[str] = set()
                years: Set[str] = set()
                metrics: Set[str] = set()
                canonical_groups: Set[str] = set()

                # Strategy 1: Try chunk metadata lookup (gold evidence corpus)
                meta = self.chunk_metadata.get(cid, {})
                if meta:
                    text = meta.get("text", "")
                    companies = meta.get("companies", set())
                    years = meta.get("years", set())
                    metrics = meta.get("metrics", set())
                    canonical_groups = meta.get("canonical_metric_groups", set())

                # Strategy 2: Try cross-reference lookup (from rich-format results)
                if not text:
                    xref = self._cross_ref.get(cid, {})
                    if xref:
                        text = xref.get("text", "")
                        if text:
                            extracted = self.passage_extractor.extract(text)
                            companies = extracted["companies"]
                            years = extracted["years"]
                            metrics = extracted["metrics"]
                            canonical_groups = extracted["canonical_metric_groups"]

                # Strategy 3: If we have text but no entities extracted yet
                if text and not companies and not metrics:
                    extracted = self.passage_extractor.extract(text)
                    companies = companies or extracted["companies"]
                    years = years or extracted["years"]
                    metrics = metrics or extracted["metrics"]
                    canonical_groups = canonical_groups or extracted["canonical_metric_groups"]

                passages.append(PassageInfo(
                    rank=rank,
                    passage_id=cid,
                    text=text,
                    is_gold=is_gold,
                    companies=companies,
                    years=years,
                    metrics=metrics,
                    canonical_metric_groups=canonical_groups,
                ))

        return passages

    def _analyze_query(
        self,
        query_id: str,
        query_text: str,
        method: str,
        passages: List[PassageInfo],
        gold_ids: List[str],
        qe: QueryEntities,
    ) -> QueryErrorResult:
        """Analyse one query for all error types."""
        result = QueryErrorResult(
            query_id=query_id,
            query_text=query_text,
            method=method,
            gold_evidence_ids=gold_ids,
            query_entities=qe,
            top_k_passages=passages,
        )

        # Find first gold rank
        first_gold_rank = None
        for p in passages:
            if p.is_gold:
                first_gold_rank = p.rank
                break
        result.first_gold_rank = first_gold_rank

        # Passages before first gold (or all if no gold)
        if first_gold_rank is not None:
            pre_gold_passages = [p for p in passages if p.rank < first_gold_rank]
        else:
            pre_gold_passages = list(passages)

        # --- Wrong Company ---
        if qe.has_company_requirement:
            result.has_wrong_company = self._check_wrong_company(
                qe, pre_gold_passages, first_gold_rank, passages
            )
            if result.has_wrong_company:
                result.error_types.append("wrong_company")

        # --- Wrong Year ---
        if qe.has_year_requirement:
            result.has_wrong_year = self._check_wrong_year(
                qe, pre_gold_passages, first_gold_rank, passages
            )
            if result.has_wrong_year:
                result.error_types.append("wrong_year")

        # --- Wrong Metric ---
        if qe.has_metric_requirement:
            result.has_wrong_metric = self._check_wrong_metric(
                qe, pre_gold_passages, first_gold_rank, passages
            )
            if result.has_wrong_metric:
                result.error_types.append("wrong_metric")

        # --- Missing Evidence ---
        result.has_missing_evidence = not any(p.is_gold for p in passages)
        if result.has_missing_evidence:
            result.error_types.append("missing_evidence")

        # Build explanation
        result.explanation = self._build_explanation(result, qe, first_gold_rank)

        return result

    def _check_wrong_company(
        self,
        qe: QueryEntities,
        pre_gold_passages: List[PassageInfo],
        first_gold_rank: Optional[int],
        all_passages: List[PassageInfo],
    ) -> bool:
        """Check if any pre-gold passage has conflicting company."""
        # If no gold in top-k, check all passages for company conflict
        check_passages = pre_gold_passages if pre_gold_passages else all_passages

        for passage in check_passages:
            if not passage.companies:
                continue
            # A passage has wrong company if none of its companies match query companies
            company_match = self._company_overlap(qe.companies, passage.companies)
            if not company_match:
                return True
        return False

    def _check_wrong_year(
        self,
        qe: QueryEntities,
        pre_gold_passages: List[PassageInfo],
        first_gold_rank: Optional[int],
        all_passages: List[PassageInfo],
    ) -> bool:
        """Check if any pre-gold passage has conflicting year.

        For comparison queries, all mentioned years are valid.
        """
        if not qe.years:
            return False

        check_passages = pre_gold_passages if pre_gold_passages else all_passages

        for passage in check_passages:
            if not passage.years:
                continue
            # For comparison queries, any year in the query range is valid
            if qe.is_comparison:
                continue  # Don't flag wrong-year for comparison queries

            # Check if passage's years conflict with query years
            # Passage is "wrong year" if it has at least one year that's
            # clearly different from all query years AND no query year matches
            year_overlap = qe.years & passage.years
            if not year_overlap:
                # Extra check: ensure passage years are in a similar range
                # (avoid flagging passages that mention many years including query year)
                if self._has_clear_year_conflict(qe.years, passage.years):
                    return True
        return False

    @staticmethod
    def _has_clear_year_conflict(query_years: Set[str], passage_years: Set[str]) -> bool:
        """Check if there's a clear year conflict (not just different but close)."""
        qy_ints = set()
        py_ints = set()
        for y in query_years:
            try:
                qy_ints.add(int(y))
            except ValueError:
                pass
        for y in passage_years:
            try:
                py_ints.add(int(y))
            except ValueError:
                pass

        if not qy_ints or not py_ints:
            return not bool(query_years & passage_years)

        # If passage has many years (e.g., >5), it's likely a multi-year table
        if len(py_ints) > 5:
            return False

        # Clear conflict: passage years are all different from query years
        # Allow +/- 1 year tolerance (FY vs CY differences)
        for qy in qy_ints:
            for py in py_ints:
                if abs(qy - py) <= 1:
                    return False
        return not bool(qy_ints & py_ints)

    def _check_wrong_metric(
        self,
        qe: QueryEntities,
        pre_gold_passages: List[PassageInfo],
        first_gold_rank: Optional[int],
        all_passages: List[PassageInfo],
    ) -> bool:
        """Check if any pre-gold passage has conflicting metric.

        Only flags when BOTH query and passage have clear metric signals
        and they belong to clearly different canonical groups.
        """
        if not qe.canonical_metric_groups:
            return False

        check_passages = pre_gold_passages if pre_gold_passages else all_passages

        for passage in check_passages:
            if not passage.canonical_metric_groups:
                # Passage has no extracted metrics — can't determine metric match
                continue

            # Check if passage has a CLEARLY different metric than query
            # (not just different, but positively conflicting)
            metric_overlap = qe.canonical_metric_groups & passage.canonical_metric_groups
            if not metric_overlap:
                # Verify this isn't just an incomplete extraction
                # Passage must have at least one solid metric extraction
                if len(passage.canonical_metric_groups) >= 1:
                    return True
        return False

    @staticmethod
    def _company_overlap(query_companies: Set[str], passage_companies: Set[str]) -> bool:
        """Check if any query company matches any passage company.

        Uses fuzzy substring matching since company names may be abbreviated.
        """
        for qc in query_companies:
            qc_lower = qc.lower().strip()
            for pc in passage_companies:
                pc_lower = pc.lower().strip()
                # Direct substring match
                if qc_lower in pc_lower or pc_lower in qc_lower:
                    return True
                # Token overlap
                qc_tokens = set(qc_lower.split())
                pc_tokens = set(pc_lower.split())
                if qc_tokens & pc_tokens:
                    # At least 50% token overlap
                    overlap = len(qc_tokens & pc_tokens)
                    if overlap >= min(len(qc_tokens), len(pc_tokens)) * 0.5:
                        return True
        return False

    @staticmethod
    def _build_explanation(
        result: QueryErrorResult,
        qe: QueryEntities,
        first_gold_rank: Optional[int],
    ) -> str:
        """Build human-readable explanation for an error case."""
        parts = []
        if result.has_missing_evidence:
            parts.append(f"No gold evidence in top-{len(result.top_k_passages)}")
        if result.has_wrong_company:
            parts.append(f"Wrong company: query={qe.companies}")
        if result.has_wrong_year:
            parts.append(f"Wrong year: query years={qe.years}" +
                         (f" (comparison query)" if qe.is_comparison else ""))
        if result.has_wrong_metric:
            parts.append(f"Wrong metric: query metrics={qe.canonical_metric_groups}")
        if first_gold_rank is not None:
            parts.append(f"First gold at rank {first_gold_rank}")
        elif not result.has_missing_evidence:
            parts.append("Gold evidence found in top-k")
        return "; ".join(parts) if parts else "No errors detected"


# ═══════════════════════════════════════════════════════════════════════════════
# Metrics Aggregation
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MethodMetrics:
    """Aggregated error metrics for one method."""
    method: str
    top_k: int
    num_queries: int = 0
    num_company_applicable: int = 0
    num_year_applicable: int = 0
    num_metric_applicable: int = 0
    wrong_company_count: int = 0
    wrong_year_count: int = 0
    wrong_metric_count: int = 0
    missing_evidence_count: int = 0

    @property
    def wrong_company_rate(self) -> Optional[float]:
        if self.num_company_applicable < 0:
            return None  # N/A
        if self.num_company_applicable == 0:
            return 0.0
        return self.wrong_company_count / self.num_company_applicable

    @property
    def wrong_year_rate(self) -> Optional[float]:
        if self.num_year_applicable < 0:
            return None
        if self.num_year_applicable == 0:
            return 0.0
        return self.wrong_year_count / self.num_year_applicable

    @property
    def wrong_metric_rate(self) -> Optional[float]:
        if self.num_metric_applicable < 0:
            return None
        if self.num_metric_applicable == 0:
            return 0.0
        return self.wrong_metric_count / self.num_metric_applicable

    @property
    def missing_evidence_rate(self) -> float:
        if self.num_queries == 0:
            return 0.0
        return self.missing_evidence_count / self.num_queries

    def to_dict(self) -> Dict:
        wc = self.wrong_company_rate
        wy = self.wrong_year_rate
        wm = self.wrong_metric_rate
        return {
            "method": self.method,
            "top_k": self.top_k,
            "wrong_company_rate": round(wc, 4) if wc is not None else None,
            "wrong_year_rate": round(wy, 4) if wy is not None else None,
            "wrong_metric_rate": round(wm, 4) if wm is not None else None,
            "missing_evidence_rate": round(self.missing_evidence_rate, 4),
            "num_company_applicable": self.num_company_applicable,
            "num_year_applicable": self.num_year_applicable,
            "num_metric_applicable": self.num_metric_applicable,
            "num_queries": self.num_queries,
            "wrong_company_count": self.wrong_company_count,
            "wrong_year_count": self.wrong_year_count,
            "wrong_metric_count": self.wrong_metric_count,
            "missing_evidence_count": self.missing_evidence_count,
        }


def compute_metrics(
    method: str,
    error_results: List[QueryErrorResult],
    query_entities: Dict[str, QueryEntities],
    top_k: int,
    can_detect_entities: bool = True,
) -> MethodMetrics:
    """Compute aggregated metrics for one method.

    Args:
        can_detect_entities: If False, only Missing Evidence is computed;
                             entity-based errors are set to -1 (N/A).
    """
    mm = MethodMetrics(method=method, top_k=top_k)
    mm.num_queries = len(error_results)

    for er in error_results:
        qe = er.query_entities

        if can_detect_entities:
            if qe and qe.has_company_requirement:
                mm.num_company_applicable += 1
                if er.has_wrong_company:
                    mm.wrong_company_count += 1

            if qe and qe.has_year_requirement:
                mm.num_year_applicable += 1
                if er.has_wrong_year:
                    mm.wrong_year_count += 1

            if qe and qe.has_metric_requirement:
                mm.num_metric_applicable += 1
                if er.has_wrong_metric:
                    mm.wrong_metric_count += 1
        else:
            # Can't detect entity errors; set to -1 to indicate N/A
            mm.num_company_applicable = -1
            mm.num_year_applicable = -1
            mm.num_metric_applicable = -1

        if er.has_missing_evidence:
            mm.missing_evidence_count += 1

    return mm


# ═══════════════════════════════════════════════════════════════════════════════
# Output Generation
# ═══════════════════════════════════════════════════════════════════════════════

def _fmt_pct(rate: Optional[float]) -> str:
    if rate is None:
        return "N/A"
    return f"{rate * 100:.1f}%"


def print_results_table(
    all_metrics: Dict[str, MethodMetrics],
    top_k: int,
) -> None:
    """Print the main results table to terminal."""
    print()
    print("=" * 110)
    print(f"  Finance-specific structural error rates in reranked evidence (k={top_k})")
    print("=" * 110)
    print()

    header = (f"{'Method':<28} {'Wrong Company':>14} {'Wrong Year':>14} "
              f"{'Wrong Metric':>14} {'Missing Evidence':>18}")
    print(header)
    print("-" * 110)

    for canonical in CANONICAL_METHODS:
        mm = all_metrics.get(canonical)
        if mm is None:
            continue
        row = (f"{canonical:<28} "
               f"{_fmt_pct(mm.wrong_company_rate):>14} "
               f"{_fmt_pct(mm.wrong_year_rate):>14} "
               f"{_fmt_pct(mm.wrong_metric_rate):>14} "
               f"{_fmt_pct(mm.missing_evidence_rate):>18}")
        print(row)

    print("-" * 110)
    print("  All metrics lower is better (↓)")
    print()


def print_sanity_checks(
    query_entities: Dict[str, QueryEntities],
    all_metrics: Dict[str, MethodMetrics],
    all_error_results: Dict[str, List[QueryErrorResult]],
    top_k: int,
    method_formats: Optional[Dict[str, str]] = None,
) -> None:
    """Print sanity check information."""
    print()
    print("=" * 60)
    print("  SANITY CHECKS")
    print("=" * 60)

    # Query entity coverage
    total = len(query_entities)
    company_qs = sum(1 for qe in query_entities.values() if qe.has_company_requirement)
    year_qs = sum(1 for qe in query_entities.values() if qe.has_year_requirement)
    metric_qs = sum(1 for qe in query_entities.values() if qe.has_metric_requirement)
    comparison_qs = sum(1 for qe in query_entities.values() if qe.is_comparison)

    print(f"\n  Query entity extraction coverage (total={total}):")
    print(f"    With company requirement: {company_qs} ({company_qs/total*100:.1f}%)")
    print(f"    With year requirement:    {year_qs} ({year_qs/total*100:.1f}%)")
    print(f"    With metric requirement:  {metric_qs} ({metric_qs/total*100:.1f}%)")
    print(f"    Comparison queries:       {comparison_qs} ({comparison_qs/total*100:.1f}%)")

    # Method format notes
    if method_formats:
        rich_methods = [m for m, f in method_formats.items() if f == "rich"]
        simple_methods = [m for m, f in method_formats.items() if f == "simple"]
        if simple_methods and rich_methods:
            print(f"\n  Note: Entity-based errors are computed for both rich and simple")
            print(f"  formats. Simple-format methods use chunk metadata plus text")
            print(f"  extraction fallbacks: {', '.join(simple_methods)}")

    # Per-method missing gold counts
    print(f"\n  Per-method top-{top_k} missing gold evidence:")
    for canonical in CANONICAL_METHODS:
        mm = all_metrics.get(canonical)
        if mm is None:
            continue
        print(f"    {canonical:<28} {mm.missing_evidence_count}/{mm.num_queries} "
              f"({_fmt_pct(mm.missing_evidence_rate)})")

    # Per-error-type examples
    for error_type, label in [
        ("wrong_company", "Wrong Company"),
        ("wrong_year", "Wrong Year"),
        ("wrong_metric", "Wrong Metric"),
        ("missing_evidence", "Missing Evidence"),
    ]:
        print(f"\n  Example {label} cases:")
        count = 0
        for canonical in CANONICAL_METHODS:
            results = all_error_results.get(canonical, [])
            for er in results:
                if error_type in er.error_types and count < 3:
                    print(f"    [{er.method}] Q: {er.query_text[:120]}")
                    print(f"      First gold rank: {er.first_gold_rank}")
                    count += 1
            if count >= 3:
                break
        if count == 0:
            print(f"    (no examples found)")


def save_json_output(
    output_dir: Path,
    all_metrics: Dict[str, MethodMetrics],
    top_k: int,
) -> None:
    """Save error metrics as JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    methods_json = {}
    for canonical in CANONICAL_METHODS:
        mm = all_metrics.get(canonical)
        if mm is None:
            continue
        methods_json[canonical] = mm.to_dict()

    output = {
        "k": top_k,
        "methods": methods_json,
        "generated": datetime.now().isoformat(),
    }

    fpath = output_dir / f"finance_structural_errors_top{top_k}.json"
    with open(fpath, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2, ensure_ascii=False)
    print(f"  JSON saved to: {fpath}")


def save_csv_output(
    output_dir: Path,
    all_metrics: Dict[str, MethodMetrics],
    top_k: int,
) -> None:
    """Save error metrics as CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fpath = output_dir / f"finance_structural_errors_top{top_k}.csv"

    with open(fpath, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "Method",
            f"Wrong Company@{top_k}",
            f"Wrong Year@{top_k}",
            f"Wrong Metric@{top_k}",
            f"Missing Evidence@{top_k}",
            "Num Company Applicable",
            "Num Year Applicable",
            "Num Metric Applicable",
            "Num Queries",
        ])
        for canonical in CANONICAL_METHODS:
            mm = all_metrics.get(canonical)
            if mm is None:
                writer.writerow([canonical, "N/A", "N/A", "N/A", "N/A", "", "", "", ""])
                continue
            wc = mm.wrong_company_rate
            wy = mm.wrong_year_rate
            wm = mm.wrong_metric_rate
            writer.writerow([
                canonical,
                f"{wc:.4f}" if wc is not None else "N/A",
                f"{wy:.4f}" if wy is not None else "N/A",
                f"{wm:.4f}" if wm is not None else "N/A",
                f"{mm.missing_evidence_rate:.4f}",
                mm.num_company_applicable if mm.num_company_applicable >= 0 else "N/A",
                mm.num_year_applicable if mm.num_year_applicable >= 0 else "N/A",
                mm.num_metric_applicable if mm.num_metric_applicable >= 0 else "N/A",
                mm.num_queries,
            ])

    print(f"  CSV saved to: {fpath}")


def save_debug_cases(
    output_dir: Path,
    all_error_results: Dict[str, List[QueryErrorResult]],
    max_per_error_type: int = 20,
) -> None:
    """Save debug cases as JSONL."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fpath = output_dir / "debug_cases.jsonl"

    # Collect examples per error type
    error_buckets: Dict[str, List[QueryErrorResult]] = defaultdict(list)
    for method, results in all_error_results.items():
        for er in results:
            for et in er.error_types:
                if len(error_buckets[et]) < max_per_error_type:
                    error_buckets[et].append(er)

    with open(fpath, "w", encoding="utf-8") as fh:
        for error_type in ["wrong_company", "wrong_year", "wrong_metric", "missing_evidence"]:
            bucket = error_buckets.get(error_type, [])
            for er in bucket[:max_per_error_type]:
                entry = {
                    "query_id": er.query_id,
                    "query_text": er.query_text,
                    "method": er.method,
                    "detected_error_types": er.error_types,
                    "query_entities": {
                        "companies": list(er.query_entities.companies) if er.query_entities else [],
                        "years": list(er.query_entities.years) if er.query_entities else [],
                        "metrics": list(er.query_entities.metrics) if er.query_entities else [],
                        "canonical_metric_groups": list(er.query_entities.canonical_metric_groups) if er.query_entities else [],
                        "is_comparison": er.query_entities.is_comparison if er.query_entities else False,
                    },
                    "top10_passages": [
                        {
                            "rank": p.rank,
                            "passage_id": p.passage_id,
                            "text_preview": p.text[:200] if p.text else "",
                            "is_gold": p.is_gold,
                            "companies": list(p.companies),
                            "years": list(p.years),
                            "metrics": list(p.metrics),
                            "canonical_metric_groups": list(p.canonical_metric_groups),
                        }
                        for p in er.top_k_passages
                    ],
                    "gold_evidence_ids": er.gold_evidence_ids[:10],
                    "first_gold_rank": er.first_gold_rank,
                    "explanation": er.explanation,
                }
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

    total_debug = sum(len(error_buckets[et]) for et in error_buckets)
    print(f"  Debug cases saved to: {fpath} ({total_debug} cases across {len(error_buckets)} error types)")


def save_additional_k_results(
    output_dir: Path,
    all_metrics_k5: Dict[str, MethodMetrics],
    all_metrics_k10: Dict[str, MethodMetrics],
) -> None:
    """Save additional @5 metrics alongside @10."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fpath = output_dir / "finance_structural_errors_top5.json"

    methods_json = {}
    for canonical in CANONICAL_METHODS:
        mm = all_metrics_k5.get(canonical)
        if mm is None:
            continue
        methods_json[canonical] = mm.to_dict()

    output = {
        "k": 5,
        "methods": methods_json,
        "generated": datetime.now().isoformat(),
    }
    with open(fpath, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2, ensure_ascii=False)
    print(f"  @5 JSON saved to: {fpath}")

    # CSV for @5
    csv_path = output_dir / "finance_structural_errors_top5.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["Method", "Wrong Company@5", "Wrong Year@5",
                          "Wrong Metric@5", "Missing Evidence@5"])
        for canonical in CANONICAL_METHODS:
            mm = all_metrics_k5.get(canonical)
            if mm is None:
                writer.writerow([canonical, "N/A", "N/A", "N/A", "N/A"])
                continue
            wc5 = mm.wrong_company_rate
            wy5 = mm.wrong_year_rate
            wm5 = mm.wrong_metric_rate
            writer.writerow([
                canonical,
                f"{wc5:.4f}" if wc5 is not None else "N/A",
                f"{wy5:.4f}" if wy5 is not None else "N/A",
                f"{wm5:.4f}" if wm5 is not None else "N/A",
                f"{mm.missing_evidence_rate:.4f}",
            ])
    print(f"  @5 CSV saved to: {csv_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Cross-Reference Builder (bridges rich-format metadata to simple-format results)
# ═══════════════════════════════════════════════════════════════════════════════

def build_cross_reference_from_rich(rich_filepath: Path) -> Dict[str, Dict]:
    """Build passage-id → text lookup from rich-format result files.

    Extracts passage text from rich-format results so that simple-format
    results can look up passage metadata for entity extraction.
    """
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


# ═══════════════════════════════════════════════════════════════════════════════
# Corpus Builder (for simple-format results)
# ═══════════════════════════════════════════════════════════════════════════════

def build_chunk_metadata_lookup(
    samples: List[Dict],
    edgar_dir: Optional[Path] = None,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    max_distractor_files: int = 50,
) -> Dict[str, Dict]:
    """Build a chunk_id → metadata lookup from FinDER gold evidence + 10-K distractors.

    Since chunk IDs are deterministic (MD5 of text content), this matches
    the chunk IDs used in exp4-style result files that include 10-K distractors.
    """
    extractor = PassageEntityExtractor()
    chunk_meta: Dict[str, Dict] = {}

    # Gold evidence chunks
    for s in samples:
        for text in s.get("evidence_texts", []):
            for chunk in chunk_text(
                text,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                doc_id=s["id"],
            ):
                extracted = extractor.extract(chunk.text)
                chunk_meta[chunk.chunk_id] = {
                    "text": chunk.text,
                    "doc_id": s["id"],
                    "companies": extracted["companies"],
                    "years": extracted["years"],
                    "metrics": extracted["metrics"],
                    "canonical_metric_groups": extracted["canonical_metric_groups"],
                }

    # 10-K distractor chunks
    if edgar_dir and edgar_dir.exists():
        from feg_rag.data.chunker import chunk_report
        distractor_files = list(edgar_dir.rglob("*.html")) + list(edgar_dir.rglob("*.txt"))
        for tf in distractor_files[:max_distractor_files]:
            try:
                for chunk in chunk_report(tf, chunk_size, chunk_overlap):
                    if chunk.chunk_id not in chunk_meta:
                        extracted = extractor.extract(chunk.text)
                        # Merge in metadata from chunk_report (company, year from filename)
                        if chunk.company and chunk.company not in {c.lower() for c in extracted["companies"]}:
                            extracted["companies"].add(chunk.company.lower())
                        if chunk.filing_year and chunk.filing_year not in extracted["years"]:
                            extracted["years"].add(chunk.filing_year)
                        chunk_meta[chunk.chunk_id] = {
                            "text": chunk.text,
                            "doc_id": chunk.doc_id,
                            "companies": extracted["companies"],
                            "years": extracted["years"],
                            "metrics": extracted["metrics"],
                            "canonical_metric_groups": extracted["canonical_metric_groups"],
                        }
            except Exception:
                pass

    return chunk_meta


# ═══════════════════════════════════════════════════════════════════════════════
# Main Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Finance-specific structural error analysis of reranked evidence"
    )
    parser.add_argument("--config", default="configs/default.yaml",
                        help="Config file for data paths")
    parser.add_argument("--output_dir", default="outputs/error_analysis",
                        help="Output directory")
    parser.add_argument("--methods", type=str, default="",
                        help="Comma-separated list of methods to analyse (default: all)")
    parser.add_argument("--top_k", type=int, default=10,
                        help="Top-k evidence to analyse (default: 10)")
    parser.add_argument("--also_k5", action="store_true",
                        help="Also compute @5 metrics")
    parser.add_argument("--max_queries", type=int, default=0,
                        help="Max queries to analyse (0=all)")
    parser.add_argument("--debug_examples", type=int, default=20,
                        help="Max debug examples per error type (default: 20)")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    root_dir = cfg.root_dir
    output_dir = root_dir / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    top_k = args.top_k
    target_methods = [m.strip() for m in args.methods.split(",") if m.strip()] if args.methods else CANONICAL_METHODS

    print("=" * 60)
    print("  Finance Structural Error Analysis")
    print("=" * 60)
    print(f"  Config:     {args.config}")
    print(f"  Output:     {output_dir}")
    print(f"  Top-k:      {top_k}")
    print(f"  Methods:    {target_methods}")
    print(f"  Also @5:    {args.also_k5}")

    # ── 1. Load FinDER data ──
    print("\n[1/6] Loading FinDER dataset...")
    samples = load_dataset("finder", cfg.data_dir)
    if args.max_queries > 0:
        samples = samples[:args.max_queries]
    print(f"  Loaded {len(samples)} QA samples")

    # ── 2. Build query entity index ──
    print("\n[2/6] Extracting query entities...")
    metric_lexicon = MetricLexicon()
    query_extractor = QueryEntityExtractor(metric_lexicon)
    query_entities: Dict[str, QueryEntities] = {}
    for s in samples:
        qe = query_extractor.extract(s["id"], s["question"])
        query_entities[s["id"]] = qe

    total = len(query_entities)
    c_qs = sum(1 for qe in query_entities.values() if qe.has_company_requirement)
    y_qs = sum(1 for qe in query_entities.values() if qe.has_year_requirement)
    m_qs = sum(1 for qe in query_entities.values() if qe.has_metric_requirement)
    cmp_qs = sum(1 for qe in query_entities.values() if qe.is_comparison)
    print(f"  Total queries: {total}")
    print(f"  With company requirement: {c_qs} ({c_qs/total*100:.1f}%)")
    print(f"  With year requirement:    {y_qs} ({y_qs/total*100:.1f}%)")
    print(f"  With metric requirement:  {m_qs} ({m_qs/total*100:.1f}%)")
    print(f"  Comparison/trend queries: {cmp_qs} ({cmp_qs/total*100:.1f}%)")

    # ── 3. Build gold map and chunk metadata lookup ──
    print("\n[3/6] Building gold evidence map and chunk metadata...")
    gold_map: Dict[str, List[str]] = {}
    for s in samples:
        gold_ids = []
        for text in s["evidence_texts"]:
            for chunk in chunk_text(
                text,
                chunk_size=cfg.chunk_size,
                chunk_overlap=cfg.chunk_overlap,
                doc_id=s["id"],
            ):
                gold_ids.append(chunk.chunk_id)
        gold_map[s["id"]] = gold_ids

    # Build chunk metadata for simple-format results (includes 10-K distractors)
    import time as _time
    _t0 = _time.time()
    chunk_metadata = build_chunk_metadata_lookup(
        samples,
        edgar_dir=cfg.edgar_dir,
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
    )
    _elapsed = _time.time() - _t0
    print(f"  Gold map: {len(gold_map)} queries")
    print(f"  Chunk metadata: {len(chunk_metadata)} chunks indexed ({_elapsed:.1f}s)")

    # ── 4. Load method results ──
    print("\n[4/6] Loading method results...")
    all_error_results: Dict[str, List[QueryErrorResult]] = {}
    all_metrics: Dict[str, MethodMetrics] = {}
    analyzer = FinanceErrorAnalyzer(
        query_entities=query_entities,
        gold_map=gold_map,
        metric_lexicon=metric_lexicon,
        chunk_metadata=chunk_metadata,
        cross_ref={},  # cross-ref from rich format won't overlap with simple-format chunk IDs
    )

    # Determine which query IDs to analyse
    target_query_ids = [s["id"] for s in samples]
    method_formats: Dict[str, str] = {}

    for canonical in target_methods:
        fpath, fmt = find_method_file(canonical, root_dir)
        if fpath is None:
            print(f"  [WARN] No result file found for '{canonical}', skipping")
            continue

        print(f"  Loading {canonical} from {fpath.relative_to(root_dir)} (format: {fmt})...")
        method_formats[canonical] = fmt

        if fmt == "rich":
            records = load_rich_format_results(fpath)
        else:
            records = load_simple_format_results(fpath)

        # Check coverage
        matched = sum(1 for qid in target_query_ids if qid in records)
        print(f"    {matched}/{len(target_query_ids)} queries matched in result file")

        error_results = analyzer.analyze_method(
            canonical, records, fmt, target_query_ids, top_k=top_k
        )
        all_error_results[canonical] = error_results

        # Both rich and simple Table-1 JSONL can be analysed now.  For simple
        # outputs we recover passage metadata from chunk ids and fall back to
        # text-based extraction when text is present, so entity errors should
        # not be suppressed to N/A.
        can_detect = True
        metrics = compute_metrics(canonical, error_results, query_entities, top_k,
                                  can_detect_entities=can_detect)
        all_metrics[canonical] = metrics

        print(f"    {len(error_results)} queries analysed")

    if not all_metrics:
        print("\n  ERROR: No method results could be loaded. Check that result files exist.")
        sys.exit(1)

    # ── 6. Output ──
    print("\n[5/6] Generating outputs...")

    # Print table
    print_results_table(all_metrics, top_k)

    # Sanity checks
    print_sanity_checks(query_entities, all_metrics, all_error_results, top_k,
                        method_formats=method_formats)

    # Save JSON
    save_json_output(output_dir, all_metrics, top_k)

    # Save CSV
    save_csv_output(output_dir, all_metrics, top_k)

    # Save debug cases
    save_debug_cases(output_dir, all_error_results, max_per_error_type=args.debug_examples)

    # ── 6. Optional @5 metrics ──
    if args.also_k5:
        print("\n[6/6] Computing @5 metrics...")
        all_metrics_k5: Dict[str, MethodMetrics] = {}
        for canonical in target_methods:
            fpath, fmt = find_method_file(canonical, root_dir)
            if fpath is None:
                continue
            if fmt == "rich":
                records = load_rich_format_results(fpath)
            else:
                records = load_simple_format_results(fpath)
            error_results = analyzer.analyze_method(
                canonical, records, fmt, target_query_ids, top_k=5
            )
            can_detect_5 = True
            all_metrics_k5[canonical] = compute_metrics(
                canonical, error_results, query_entities, 5,
                can_detect_entities=can_detect_5,
            )

        save_additional_k_results(output_dir, all_metrics_k5, all_metrics)

        # Print @5 table
        print("\n  @5 Results:")
        print(f"  {'Method':<28} {'Wrong Co@5':>12} {'Wrong Yr@5':>12} "
              f"{'Wrong Met@5':>12} {'Missing@5':>12}")
        print(f"  {'-'*28} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")
        for canonical in CANONICAL_METHODS:
            mm = all_metrics_k5.get(canonical)
            if mm is None:
                continue
            print(f"  {canonical:<28} {_fmt_pct(mm.wrong_company_rate):>12} "
                  f"{_fmt_pct(mm.wrong_year_rate):>12} "
                  f"{_fmt_pct(mm.wrong_metric_rate):>12} "
                  f"{_fmt_pct(mm.missing_evidence_rate):>12}")
    else:
        print("\n[6/6] Skipped (use --also_k5 for @5 metrics)")

    print(f"\nOutput directory: {output_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
