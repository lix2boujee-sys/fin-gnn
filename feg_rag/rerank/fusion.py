"""Constraint-aware fusion scoring for evidence reranking.

Paper design §5.7: Final reranking score fuses retrieval, graph, GNN, and
financial constraint scores. The constraint score captures whether a chunk
matches the query's structural constraints (company, year, metric, filing type).
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Set, Tuple

from feg_rag.data.chunker import Chunk
from feg_rag.graph.builder import FinancialEvidenceGraph


# ═════════════════════════════════════════════════════════════════════════════
# Constraint scorer
# ═════════════════════════════════════════════════════════════════════════════

class ConstraintScorer:
    """Score how well a chunk satisfies the query's structural constraints.

    Checks: company match, year match, metric match, filing type match.
    """

    def __init__(
        self,
        company_weight: float = 1.0,
        year_weight: float = 1.0,
        metric_weight: float = 0.8,
        filing_type_weight: float = 0.5,
    ):
        self.company_weight = company_weight
        self.year_weight = year_weight
        self.metric_weight = metric_weight
        self.filing_type_weight = filing_type_weight

    def score(
        self,
        query: str,
        chunk: Chunk,
        query_metrics: Optional[Set[str]] = None,
        query_years: Optional[Set[str]] = None,
        query_companies: Optional[Set[str]] = None,
        query_filing_types: Optional[Set[str]] = None,
    ) -> float:
        """Compute constraint satisfaction score for a (query, chunk) pair.

        Returns a value in [0, 4] (sum of weighted matches), normalisable to [0, 1].
        """
        total = 0.0

        # Extract query entities on first call or use provided
        if query_metrics is None:
            query_metrics = _extract_metrics(query)
        if query_years is None:
            query_years = _extract_years(query)
        if query_companies is None:
            query_companies = set()
        if query_filing_types is None:
            query_filing_types = _extract_filing_types(query)

        # Company match
        if query_companies:
            chunk_companies = _extract_companies(chunk.text)
            chunk_companies.add(chunk.company.lower())  # also check metadata
            if query_companies & chunk_companies:
                total += self.company_weight

        # Year match
        if query_years:
            chunk_years = _extract_years(chunk.text)
            chunk_years.add(chunk.filing_year)  # metadata year
            if query_years & chunk_years:
                total += self.year_weight

        # Metric match
        if query_metrics:
            chunk_metrics = _extract_metrics(chunk.text)
            if query_metrics & chunk_metrics:
                total += self.metric_weight

        # Filing type match
        if query_filing_types and chunk.filing_type:
            if chunk.filing_type.upper() in {ft.upper() for ft in query_filing_types}:
                total += self.filing_type_weight

        return total

    def batch_score(
        self,
        query: str,
        chunks: List[Chunk],
        normalize: bool = True,
    ) -> Dict[str, float]:
        """Score a batch of chunks against a query.

        Returns:
            Dict[chunk_id, normalized_constraint_score].
        """
        q_metrics = _extract_metrics(query)
        q_years = _extract_years(query)
        q_companies = _extract_companies(query)
        q_filing_types = _extract_filing_types(query)

        scores: Dict[str, float] = {}
        for c in chunks:
            scores[c.chunk_id] = self.score(
                query, c,
                query_metrics=q_metrics,
                query_years=q_years,
                query_companies=q_companies,
                query_filing_types=q_filing_types,
            )

        if normalize:
            max_s = max(scores.values()) if scores else 1.0
            if max_s > 0:
                for k in scores:
                    scores[k] /= max_s

        return scores


# ═════════════════════════════════════════════════════════════════════════════
# Fusion scorer
# ═════════════════════════════════════════════════════════════════════════════

class FusionScorer:
    """Combine retrieval, graph, GNN, and constraint scores.

    final_score = α * retrieval + β * graph + γ * gnn + δ * constraint
    """

    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.3,
        gamma: float = 0.3,
        delta: float = 0.1,
        constraint_scorer: Optional[ConstraintScorer] = None,
    ):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.constraint_scorer = constraint_scorer or ConstraintScorer()

    def fuse(
        self,
        query: str,
        chunks: List[Chunk],
        retrieval_scores: Dict[str, float],
        graph_scores: Optional[Dict[str, float]] = None,
        gnn_scores: Optional[Dict[str, float]] = None,
        constraint_scores: Optional[Dict[str, float]] = None,
    ) -> List[Tuple[Chunk, float]]:
        """Compute fused score for each chunk.

        All score dicts should be normalised to [0, 1] for best results.
        """
        if graph_scores is None:
            graph_scores = {}
        if gnn_scores is None:
            gnn_scores = {}
        if constraint_scores is None:
            constraint_scores = self.constraint_scorer.batch_score(query, chunks)

        # Normalise all score sets
        ret_norm = _normalise_dict(retrieval_scores)
        graph_norm = _normalise_dict(graph_scores)
        gnn_norm = _normalise_dict(gnn_scores)
        cons_norm = _normalise_dict(constraint_scores)

        fused: List[Tuple[Chunk, float]] = []
        for c in chunks:
            cid = c.chunk_id
            score = (
                self.alpha * ret_norm.get(cid, 0.0)
                + self.beta * graph_norm.get(cid, 0.0)
                + self.gamma * gnn_norm.get(cid, 0.0)
                + self.delta * cons_norm.get(cid, 0.0)
            )
            fused.append((c, score))

        fused.sort(key=lambda x: x[1], reverse=True)
        return fused


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════

_METRIC_RE = re.compile(
    r"\b(revenue|revenues|sales|net\s+income|net\s+earnings|net\s+loss|profit|"
    r"operating\s+income|operating\s+earnings|gross\s+profit|gross\s+margin|"
    r"eps|earnings\s+per\s+share|ebitda|ebit|total\s+assets|total\s+liabilities|"
    r"cash\s+flow|free\s+cash\s+flow|r\s*&?\s*d|sg\s*&?\s*a|"
    r"cost\s+of\s+(revenue|sales|goods\s+sold)|dividends?)\b",
    re.IGNORECASE,
)

_YEAR_RE = re.compile(r"\b((?:19|20)\d{2})\b")
_FILING_TYPE_RE = re.compile(r"\b(10-K|10-Q|8-K|annual\s+report|quarterly\s+report)\b", re.IGNORECASE)
_COMPANY_RE = re.compile(
    r"\b([A-Z][a-z]+(?:\s+(?:Inc\.?|Corp\.?|Corporation|Ltd\.?|Limited|"
    r"PLC|LLC|Technologies?|Group|Holdings?)))\b"
)


def _extract_metrics(text: str) -> Set[str]:
    return {m.group(0).strip().lower() for m in _METRIC_RE.finditer(text)}


def _extract_years(text: str) -> Set[str]:
    return set(_YEAR_RE.findall(text))


def _extract_filing_types(text: str) -> Set[str]:
    return set(_FILING_TYPE_RE.findall(text))


def _extract_companies(text: str) -> Set[str]:
    return {m.group(1).strip().lower() for m in _COMPANY_RE.finditer(text)}


def _normalise_dict(d: Dict[str, float]) -> Dict[str, float]:
    if not d:
        return {}
    vals = list(d.values())
    vmin, vmax = min(vals), max(vals)
    if vmax == vmin:
        return {k: 0.5 for k in d}
    return {k: (v - vmin) / (vmax - vmin) for k, v in d.items()}
