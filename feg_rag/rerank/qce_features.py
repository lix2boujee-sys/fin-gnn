"""QCE-Graph Lite: Support / Conflict Feature Extraction.

Phase 2 module — extracts counterfactual evidence features for
dual-channel support/conflict scoring without training any model.

Provides:
    - Query feature builder (10-dim query features for router input).
    - Support feature extraction (11 features).
    - Conflict feature extraction (5 features).
    - Batch feature materialization with caching.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from feg_rag.data.chunker import Chunk
from feg_rag.graph.entities import EntityExtractor
from feg_rag.rerank.qce_expansion import (
    RELATION_NAMES,
    ExpandedCandidate,
    normalize_metric,
)

# ═════════════════════════════════════════════════════════════════════════════
# Constants
# ═════════════════════════════════════════════════════════════════════════════

# Query feature dimension for router input
QUERY_FEATURE_DIM_QCE = 10

# Support feature dimension
SUPPORT_FEATURE_DIM = 11

# Conflict feature dimension
CONFLICT_FEATURE_DIM = 5

# Ordered support feature names
SUPPORT_FEATURE_NAMES: List[str] = [
    "company_match",
    "filing_year_match",
    "year_text_match",
    "metric_match",
    "filing_type_match",
    "section_match",
    "query_text_overlap",
    "same_filing_support",
    "same_section_support",
    "adjacent_support",
    "route_alignment",
]

# Ordered conflict feature names
CONFLICT_FEATURE_NAMES: List[str] = [
    "company_conflict",
    "year_conflict",
    "metric_conflict",
    "filing_type_conflict",
    "section_conflict",
]

# Keyword sets for query feature extraction
_COMPARISON_KEYWORDS = {
    "compare", "comparison", "versus", "vs", "difference", "higher", "lower",
    "better", "worse", "more than", "less than", "between",
}
_DELTA_KEYWORDS = {
    "change", "increase", "decrease", "growth", "decline", "trend",
    "quarter over quarter", "year over year", "yoy", "qoq",
}
_FILING_TYPE_KEYWORDS = {"10-k", "10-q", "8-k", "annual", "quarterly", "filing", "report"}
_SECTION_KEYWORDS = {
    "item 1", "item 1a", "item 7", "item 8", "mda", "risk factors",
    "financial statements", "notes", "business", "management discussion",
}

_entity_extractor = EntityExtractor()


# ═════════════════════════════════════════════════════════════════════════════
# Query feature builder
# ═════════════════════════════════════════════════════════════════════════════

def build_qce_query_features(query: str) -> np.ndarray:
    """Build a 10-dim query feature vector for the QueryRelationRouter.

    Feature order (fixed):
        0  num_years
        1  num_metrics
        2  num_companies
        3  query_length
        4  has_numeric_question
        5  has_comparison_keyword
        6  has_delta_keyword
        7  has_filing_type
        8  has_section_keyword
        9  is_ambiguous_short_query
    """
    q_lower = query.lower()
    q_tokens = q_lower.split()

    years = _entity_extractor.extract_years(query)
    metrics = _entity_extractor.extract_metrics(query)
    companies = _entity_extractor.extract_companies(query)

    features = np.zeros(QUERY_FEATURE_DIM_QCE, dtype=np.float32)
    features[0] = min(float(len(years)), 5.0) / 5.0  # num_years (capped)
    features[1] = min(float(len(metrics)), 5.0) / 5.0  # num_metrics (capped)
    features[2] = min(float(len(companies)), 3.0) / 3.0  # num_companies (capped)
    features[3] = min(float(len(q_tokens)), 50.0) / 50.0  # query_length (capped)

    # has_numeric_question
    has_number = any(c.isdigit() for c in query)
    has_question = "?" in query or any(
        q_lower.startswith(w) for w in ["what", "how", "which", "who", "when", "where", "why", "did", "does", "is", "are", "was", "were", "can", "could", "would", "should", "will"]
    )
    features[4] = 1.0 if (has_number and has_question) else 0.0

    # has_comparison_keyword
    features[5] = 1.0 if any(kw in q_lower for kw in _COMPARISON_KEYWORDS) else 0.0

    # has_delta_keyword
    features[6] = 1.0 if any(kw in q_lower for kw in _DELTA_KEYWORDS) else 0.0

    # has_filing_type
    features[7] = 1.0 if any(kw in q_lower for kw in _FILING_TYPE_KEYWORDS) else 0.0

    # has_section_keyword
    features[8] = 1.0 if any(kw in q_lower for kw in _SECTION_KEYWORDS) else 0.0

    # is_ambiguous_short_query
    features[9] = 1.0 if (len(q_tokens) <= 5 and len(metrics) == 0 and len(companies) == 0) else 0.0

    return features


# ═════════════════════════════════════════════════════════════════════════════
# Support feature extraction
# ═════════════════════════════════════════════════════════════════════════════

def extract_support_features(
    query: str,
    candidate: ExpandedCandidate,
    chunk_lookup: Dict[str, Chunk],
    relation_probabilities: Optional[Dict[str, float]] = None,
    seed_chunks: Optional[Dict[str, Chunk]] = None,
) -> np.ndarray:
    """Extract 11-dim support feature vector for a candidate chunk.

    Args:
        query: The question text.
        candidate: The expanded candidate.
        chunk_lookup: Mapping from chunk_id -> Chunk.
        relation_probabilities: Optional router probabilities for route_alignment.
        seed_chunks: Optional mapping from seed chunk_id -> Chunk for adjacency check.

    Returns:
        Float32 array of shape (SUPPORT_FEATURE_DIM,).
    """
    chunk = chunk_lookup.get(candidate.chunk_id)
    if chunk is None:
        return np.zeros(SUPPORT_FEATURE_DIM, dtype=np.float32)

    q_metrics = {normalize_metric(m) for m in _entity_extractor.extract_metrics(query)}
    q_years = _entity_extractor.extract_years(query)
    q_companies = {c.lower() for c in _entity_extractor.extract_companies(query)}
    q_filing_types = {ft.upper() for ft in _entity_extractor.extract_filing_types(query)}
    q_tokens = set(query.lower().split())

    feats = np.zeros(SUPPORT_FEATURE_DIM, dtype=np.float32)

    # 0: company_match
    if q_companies and chunk.company:
        chunk_company_lower = chunk.company.lower()
        feats[0] = 1.0 if any(
            qc in chunk_company_lower or chunk_company_lower in qc
            for qc in q_companies
        ) else 0.0

    # 1: filing_year_match
    if q_years and chunk.filing_year:
        feats[1] = 1.0 if chunk.filing_year in q_years else 0.0

    # 2: year_text_match
    if q_years:
        chunk_years = _entity_extractor.extract_years(chunk.text)
        feats[2] = 1.0 if bool(chunk_years & q_years) else 0.0

    # 3: metric_match
    if q_metrics:
        chunk_metrics = {normalize_metric(m) for m in _entity_extractor.extract_metrics(chunk.text)}
        feats[3] = 1.0 if bool(chunk_metrics & q_metrics) else 0.0

    # 4: filing_type_match
    if q_filing_types and chunk.filing_type:
        feats[4] = 1.0 if chunk.filing_type.upper() in q_filing_types else 0.0

    # 5: section_match
    q_sections = _extract_query_sections(query)
    if q_sections and chunk.section:
        section_lower = chunk.section.lower()
        feats[5] = 1.0 if any(qs in section_lower for qs in q_sections) else 0.0

    # 6: query_text_overlap
    if q_tokens:
        c_tokens = set(chunk.text.lower().split())
        overlap = len(q_tokens & c_tokens)
        feats[6] = min(overlap / max(len(q_tokens), 1), 1.0)

    # 7: same_filing_support
    if candidate.best_seed_chunk_id and candidate.best_seed_chunk_id in chunk_lookup:
        seed_chunk = chunk_lookup[candidate.best_seed_chunk_id]
        feats[7] = 1.0 if (seed_chunk.doc_id and chunk.doc_id == seed_chunk.doc_id) else 0.0

    # 8: same_section_support
    if candidate.best_seed_chunk_id and candidate.best_seed_chunk_id in chunk_lookup:
        seed_chunk = chunk_lookup[candidate.best_seed_chunk_id]
        feats[8] = 1.0 if (
            seed_chunk.doc_id and seed_chunk.section
            and chunk.doc_id == seed_chunk.doc_id
            and chunk.section == seed_chunk.section
        ) else 0.0

    # 9: adjacent_support
    if candidate.best_seed_chunk_id and seed_chunks:
        seed = seed_chunks.get(candidate.best_seed_chunk_id)
        if seed and seed.doc_id == chunk.doc_id:
            seed_offset = seed.metadata.get("word_offset", seed.metadata.get("row_offset", 0))
            chunk_offset = chunk.metadata.get("word_offset", chunk.metadata.get("row_offset", 0))
            dist = abs(seed_offset - chunk_offset)
            feats[9] = 1.0 / (1.0 + dist / 100.0) if dist < 1000 else 0.0

    # 10: route_alignment — weighted sum of relation probabilities for source relations
    if relation_probabilities:
        alignment = 0.0
        for sr in candidate.source_relations:
            alignment += relation_probabilities.get(sr, 0.0)
        feats[10] = min(alignment, 1.0)

    return feats


# ═════════════════════════════════════════════════════════════════════════════
# Conflict feature extraction
# ═════════════════════════════════════════════════════════════════════════════

def extract_conflict_features(
    query: str,
    candidate: ExpandedCandidate,
    chunk_lookup: Dict[str, Chunk],
) -> np.ndarray:
    """Extract 5-dim conflict feature vector for a candidate chunk.

    IMPORTANT: Missing information is NOT conflict. Only explicit entity
    mismatches count as conflict.

    Args:
        query: The question text.
        candidate: The expanded candidate.
        chunk_lookup: Mapping from chunk_id -> Chunk.

    Returns:
        Float32 array of shape (CONFLICT_FEATURE_DIM,).
    """
    chunk = chunk_lookup.get(candidate.chunk_id)
    if chunk is None:
        return np.zeros(CONFLICT_FEATURE_DIM, dtype=np.float32)

    q_metrics = {normalize_metric(m) for m in _entity_extractor.extract_metrics(query)}
    q_years = _entity_extractor.extract_years(query)
    q_companies = {c.lower() for c in _entity_extractor.extract_companies(query)}
    q_filing_types = {ft.upper() for ft in _entity_extractor.extract_filing_types(query)}

    feats = np.zeros(CONFLICT_FEATURE_DIM, dtype=np.float32)

    # 0: company_conflict — ONLY when both query and candidate have companies that don't match
    if q_companies and chunk.company:
        chunk_company_lower = chunk.company.lower()
        has_match = any(
            qc in chunk_company_lower or chunk_company_lower in qc
            for qc in q_companies
        )
        feats[0] = 1.0 if not has_match else 0.0

    # 1: year_conflict — ONLY when both query and candidate have explicit years that don't match
    if q_years:
        # Check candidate's filing year
        if chunk.filing_year:
            if chunk.filing_year not in q_years:
                feats[1] = 1.0
        else:
            # Check text for explicit years
            chunk_years = _entity_extractor.extract_years(chunk.text)
            if chunk_years and not (chunk_years & q_years):
                feats[1] = 1.0

    # 2: metric_conflict — ONLY when both have metrics that don't intersect
    if q_metrics:
        chunk_metrics = {normalize_metric(m) for m in _entity_extractor.extract_metrics(chunk.text)}
        if chunk_metrics and not (chunk_metrics & q_metrics):
            feats[2] = 1.0

    # 3: filing_type_conflict
    if q_filing_types and chunk.filing_type:
        feats[3] = 1.0 if chunk.filing_type.upper() not in q_filing_types else 0.0

    # 4: section_conflict
    q_sections = _extract_query_sections(query)
    if q_sections and chunk.section:
        section_lower = chunk.section.lower()
        feats[4] = 1.0 if not any(qs in section_lower for qs in q_sections) else 0.0

    return feats


# ═════════════════════════════════════════════════════════════════════════════
# Batch feature extraction with caching
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class QCEFeatureCache:
    """Materialized feature tensors for efficient training."""

    query_features: np.ndarray  # (num_queries, QUERY_FEATURE_DIM_QCE)
    support_features: np.ndarray  # (num_pairs, SUPPORT_FEATURE_DIM)
    conflict_features: np.ndarray  # (num_pairs, CONFLICT_FEATURE_DIM)
    base_features: np.ndarray  # (num_pairs, 3) — [retrieval_score, initial_rank, is_expanded]
    relation_origin: np.ndarray  # (num_pairs, 7) — multi-hot relation origin
    query_ids: List[str]  # question_id for each pair
    candidate_ids: List[str]  # chunk_id for each pair
    labels: np.ndarray  # (num_pairs,) — 1 for positive, 0 for negative


def materialize_qce_features(
    queries: List[Dict],
    expanded_candidates: Dict[str, List[ExpandedCandidate]],
    chunk_lookup: Dict[str, Chunk],
    gold_map: Dict[str, List[str]],
    relation_probabilities: Optional[Dict[str, Dict[str, float]]] = None,
    cache_path: Optional[str | Path] = None,
) -> QCEFeatureCache:
    """Materialize all QCE features for training/evaluation.

    Args:
        queries: List of sample dicts with 'id', 'question'.
        expanded_candidates: question_id -> list of ExpandedCandidate.
        chunk_lookup: chunk_id -> Chunk.
        gold_map: question_id -> list of gold chunk_ids.
        relation_probabilities: Optional question_id -> {rel_name: prob}.
        cache_path: Optional path to save/load cached features.

    Returns:
        QCEFeatureCache with all materialized features.
    """
    if cache_path:
        cache_path = Path(cache_path)
        if cache_path.exists():
            try:
                with open(cache_path, "rb") as fh:
                    return pickle.load(fh)
            except Exception:
                pass

    t0 = time.time()

    all_query_feats = []
    all_support_feats = []
    all_conflict_feats = []
    all_base_feats = []
    all_relation_origin = []
    all_query_ids = []
    all_candidate_ids = []
    all_labels = []

    for sample in queries:
        qid = sample["id"]
        question = sample["question"]
        gold = set(gold_map.get(qid, []))

        candidates = expanded_candidates.get(qid, [])
        if not candidates:
            continue

        qf = build_qce_query_features(question)
        rel_probs = (relation_probabilities or {}).get(qid, {})

        # Build seed chunk lookup for adjacency checks
        seed_chunks: Dict[str, Chunk] = {}
        for ec in candidates:
            if ec.best_seed_chunk_id and ec.best_seed_chunk_id in chunk_lookup:
                seed_chunks[ec.best_seed_chunk_id] = chunk_lookup[ec.best_seed_chunk_id]

        for ec in candidates:
            sf = extract_support_features(question, ec, chunk_lookup, rel_probs, seed_chunks)
            cf = extract_conflict_features(question, ec, chunk_lookup)

            # Base features: retrieval_score, initial_rank_norm, is_expanded
            rank_norm = 1.0 / max(ec.initial_rank or 1, 1) if ec.initial_rank else 0.0
            bf = np.array([
                ec.initial_score if ec.initial_score else 0.0,
                rank_norm,
                1.0 if not ec.is_initial else 0.0,
            ], dtype=np.float32)

            # Relation origin multi-hot
            ro = np.zeros(len(RELATION_NAMES), dtype=np.float32)
            for sr in ec.source_relations:
                if sr in RELATION_NAMES:
                    ro[RELATION_NAMES.index(sr)] = 1.0
            if ec.is_initial:
                # Initial candidates get no relation origin
                ro[:] = 0.0

            label = 1.0 if ec.chunk_id in gold else 0.0

            all_query_feats.append(qf)
            all_support_feats.append(sf)
            all_conflict_feats.append(cf)
            all_base_feats.append(bf)
            all_relation_origin.append(ro)
            all_query_ids.append(qid)
            all_candidate_ids.append(ec.chunk_id)
            all_labels.append(label)

    cache = QCEFeatureCache(
        query_features=np.stack(all_query_feats) if all_query_feats else np.zeros((0, QUERY_FEATURE_DIM_QCE), dtype=np.float32),
        support_features=np.stack(all_support_feats) if all_support_feats else np.zeros((0, SUPPORT_FEATURE_DIM), dtype=np.float32),
        conflict_features=np.stack(all_conflict_feats) if all_conflict_feats else np.zeros((0, CONFLICT_FEATURE_DIM), dtype=np.float32),
        base_features=np.stack(all_base_feats) if all_base_feats else np.zeros((0, 3), dtype=np.float32),
        relation_origin=np.stack(all_relation_origin) if all_relation_origin else np.zeros((0, len(RELATION_NAMES)), dtype=np.float32),
        query_ids=all_query_ids,
        candidate_ids=all_candidate_ids,
        labels=np.array(all_labels, dtype=np.float32),
    )

    elapsed = time.time() - t0
    print(f"  [QCEFeatureCache] Materialized {len(all_labels)} pairs in {elapsed:.1f}s")

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as fh:
            pickle.dump(cache, fh, protocol=pickle.HIGHEST_PROTOCOL)

    return cache


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════

def _extract_query_sections(query: str) -> Set[str]:
    """Extract section references from a query string."""
    q_lower = query.lower()
    sections: Set[str] = set()
    for kw in _SECTION_KEYWORDS:
        if kw in q_lower:
            sections.add(kw)
    return sections
