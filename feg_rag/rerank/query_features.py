"""Query-specific feature augmentation for GNN rerankers.

Produces per-query features that are concatenated with static node features
so that both GraphSAGE and R-GCN become truly query-aware.

Shared by GraphSAGE (gnn.py) and R-GCN (rgcn.py). Training and inference MUST
use the same augmentation logic and the same ``QUERY_FEATURE_DIM`` to keep
feature dimensionality consistent.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Set

import numpy as np

from feg_rag.data.chunker import Chunk
from feg_rag.graph.entities import EntityExtractor

# Number of query-specific feature columns appended to every node vector.
# Keep in sync with the features listed in _build_augmentation_columns.
QUERY_FEATURE_DIM = 7

# Shared entity extractor (stateless, thread-safe).
_extractor = EntityExtractor()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_query_augmented_features(
    base_features: Dict[str, np.ndarray],
    node_list: List[str],
    query: str,
    chunk_lookup: Dict[str, Chunk],
    retrieval_scores: Optional[Dict[str, float]] = None,
    graph_scores: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """Build query-specific feature columns for a list of graph nodes.

    Returns a ``(N, QUERY_FEATURE_DIM)`` float32 array that the caller should
    concatenate with the base feature matrix.

    Columns (in order):
        0  is_candidate_chunk          (1 if node_id is a candidate, else 0)
        1  retrieval_score_norm        (min-max within candidates; 0 otherwise)
        2  graph_score_norm            (min-max within candidates; 0 otherwise)
        3  query_year_match            (1 if node relates to a query year)
        4  query_metric_match          (1 if node relates to a query metric)
        5  query_company_match         (1 if node relates to a query company)
        6  query_text_overlap          (|Q ∩ C| / |Q| for chunks, 0 otherwise)

    Args:
        base_features: Static node features dict (used only to validate
            node presence; shapes are not inspected).
        node_list: Ordered list of node IDs that will form the rows.
        query: The question / query text.
        chunk_lookup: Mapping from chunk_id → ``Chunk``.
        retrieval_scores: Optional retrieval scores per chunk_id (the
            presence of a key defines the candidate set).
        graph_scores: Optional graph / PPR scores per node_id.

    Returns:
        Float32 array of shape ``(len(node_list), QUERY_FEATURE_DIM)``.
    """
    retrieval_scores = retrieval_scores or {}
    graph_scores = graph_scores or {}

    # Query entity extraction (lightweight regex, no heavy deps)
    q_metrics: Set[str] = _extractor.extract_metrics(query)
    q_years: Set[str] = _extractor.extract_years(query)
    q_companies: Set[str] = _extractor.extract_companies(query)
    q_tokens: Set[str] = _tokenize(query)

    # Normalise retrieval / graph scores within *candidates* only
    ret_norm = _normalise_score_map(retrieval_scores)
    graph_norm = _normalise_score_map(graph_scores)

    N = len(node_list)
    aug = np.zeros((N, QUERY_FEATURE_DIM), dtype=np.float32)

    for i, node_id in enumerate(node_list):
        # --- col 0: is_candidate_chunk ---
        is_candidate = 1.0 if node_id in retrieval_scores else 0.0
        aug[i, 0] = is_candidate

        # --- col 1: retrieval_score_norm ---
        aug[i, 1] = ret_norm.get(node_id, 0.0)

        # --- col 2: graph_score_norm ---
        aug[i, 2] = graph_norm.get(node_id, 0.0)

        # --- col 3-5: query entity matches ---
        aug[i, 3] = _year_match(node_id, q_years, chunk_lookup)
        aug[i, 4] = _metric_match(node_id, q_metrics, chunk_lookup)
        aug[i, 5] = _company_match(node_id, q_companies, chunk_lookup)

        # --- col 6: query_text_overlap ---
        aug[i, 6] = _text_overlap(node_id, q_tokens, chunk_lookup)

    return aug


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalise_score_map(score_map: Dict[str, float]) -> Dict[str, float]:
    """Min-max normalise values in *score_map* to [0, 1].

    When all values are equal (or the map has ≤1 entry), every key receives
    0.5 — a stable neutral default that avoids division by zero while
    preserving equal treatment of all candidates.
    """
    if not score_map:
        return {}
    vals = list(score_map.values())
    vmin, vmax = min(vals), max(vals)
    if vmax == vmin:
        return {k: 0.5 for k in score_map}
    return {k: (v - vmin) / (vmax - vmin) for k, v in score_map.items()}


def _tokenize(text: str) -> Set[str]:
    """Lowercase alpha-numeric token set (fast, no NLTK dependency)."""
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _year_match(
    node_id: str,
    q_years: Set[str],
    chunk_lookup: Dict[str, Chunk],
) -> float:
    """Check whether *node_id* matches any query year."""
    if not q_years:
        return 0.0

    # year::2023 node
    if node_id.startswith("year::"):
        year_val = node_id.split("::", 1)[1]
        return 1.0 if year_val in q_years else 0.0

    # chunk node — check text + metadata
    chunk = chunk_lookup.get(node_id)
    if chunk is not None:
        if chunk.filing_year and chunk.filing_year in q_years:
            return 1.0
        chunk_years = _extractor.extract_years(chunk.text)
        if chunk_years & q_years:
            return 1.0

    return 0.0


def _metric_match(
    node_id: str,
    q_metrics: Set[str],
    chunk_lookup: Dict[str, Chunk],
) -> float:
    """Check whether *node_id* matches any query metric."""
    if not q_metrics:
        return 0.0

    # metric::revenue node
    if node_id.startswith("metric::"):
        metric_name = node_id.split("::", 1)[1]
        return 1.0 if metric_name in q_metrics else 0.0

    # chunk node — check text
    chunk = chunk_lookup.get(node_id)
    if chunk is not None:
        chunk_metrics = _extractor.extract_metrics(chunk.text)
        if chunk_metrics & q_metrics:
            return 1.0

    return 0.0


def _company_match(
    node_id: str,
    q_companies: Set[str],
    chunk_lookup: Dict[str, Chunk],
) -> float:
    """Check whether *node_id* matches any query company."""
    if not q_companies:
        return 0.0

    # company::Name node
    if node_id.startswith("company::"):
        comp_name = node_id.split("::", 1)[1]
        return 1.0 if comp_name in q_companies else 0.0

    # chunk node — check text + metadata
    chunk = chunk_lookup.get(node_id)
    if chunk is not None:
        if chunk.company:
            comp_lower = chunk.company.lower()
            for qc in q_companies:
                if qc.lower() in comp_lower or comp_lower in qc.lower():
                    return 1.0
        chunk_companies = _extractor.extract_companies(chunk.text)
        if chunk_companies & q_companies:
            return 1.0

    return 0.0


def _text_overlap(
    node_id: str,
    q_tokens: Set[str],
    chunk_lookup: Dict[str, Chunk],
) -> float:
    """Compute query-chunk token overlap: |Q ∩ C| / |Q|.

    Returns 0.0 for non-chunk nodes or when Q is empty.
    """
    if not q_tokens:
        return 0.0

    chunk = chunk_lookup.get(node_id)
    if chunk is None:
        return 0.0

    c_tokens = _tokenize(chunk.text)
    if not c_tokens:
        return 0.0

    overlap = len(q_tokens & c_tokens)
    return overlap / len(q_tokens)
