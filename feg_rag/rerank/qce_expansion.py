"""QCE-Graph Lite: Budgeted Graph Candidate Expansion.

Phase 1 module — diagnostic graph expansion that finds new evidence candidates
through query-conditioned relation routing WITHOUT training any model.

Provides:
    - GraphExpansionIndex: fast lookup structures over chunks and graph edges.
    - BudgetedGraphExpander: budget-constrained multi-relation expansion.
    - ExpandedCandidate: dataclass for expansion-aware candidate tracking.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from feg_rag.data.chunker import Chunk
from feg_rag.graph.builder import FinancialEvidenceGraph
from feg_rag.graph.entities import EntityExtractor

# ═════════════════════════════════════════════════════════════════════════════
# Constants
# ═════════════════════════════════════════════════════════════════════════════

RELATION_NAMES: List[str] = [
    "adjacent_chunk",
    "same_section",
    "same_filing",
    "same_company_year",
    "same_metric",
    "same_year",
    "semantic_similar",
]

NUM_RELATIONS: int = len(RELATION_NAMES)

# Default relation priors for expansion priority computation.
# Configurable via constructor or config dict.
DEFAULT_RELATION_PRIOR: Dict[str, float] = {
    "adjacent_chunk": 1.00,
    "same_section": 0.90,
    "same_filing": 0.75,
    "same_company_year": 0.90,
    "same_metric": 0.85,
    "same_year": 0.65,
    "semantic_similar": 0.50,
}

# Schema version for cache fingerprinting — bump when index structure changes.
INDEX_SCHEMA_VERSION = 1

# Default expansion budget constants
DEFAULT_INITIAL_TOP_N = 50
DEFAULT_SEED_TOP_M = 15
DEFAULT_EXPANSION_BUDGET = 30
DEFAULT_MAX_BUDGET_PER_RELATION = 10
DEFAULT_MAX_TOTAL_CANDIDATES = 80
DEFAULT_RELATION_THRESHOLD = 0.10
DEFAULT_SEMANTIC_MAX_PER_QUERY = 5

_entity_extractor = EntityExtractor()


# ═════════════════════════════════════════════════════════════════════════════
# Data structures
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class ExpandedCandidate:
    """A candidate chunk that may come from initial retrieval or graph expansion."""

    chunk_id: str
    is_initial: bool
    initial_score: float
    initial_rank: Optional[int]
    source_relations: List[str] = field(default_factory=list)
    best_relation: Optional[str] = None
    best_seed_chunk_id: Optional[str] = None
    best_seed_rank: Optional[int] = None
    graph_distance: int = 1
    expansion_priority: float = 0.0


# ═════════════════════════════════════════════════════════════════════════════
# Graph Expansion Index
# ═════════════════════════════════════════════════════════════════════════════

class GraphExpansionIndex:
    """Pre-built lookup structures for fast relation-based candidate expansion.

    Indexes chunks by doc, section, company+year, metric, year, adjacency,
    and semantic neighbours.  Designed to be built once and cached.
    """

    def __init__(self):
        # chunk_id -> Chunk
        self.chunk_lookup: Dict[str, Chunk] = {}
        # doc_id -> [chunk_id, ...]
        self.chunks_by_doc: Dict[str, List[str]] = defaultdict(list)
        # (doc_id, section) -> [chunk_id, ...]
        self.chunks_by_section: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        # (company, filing_year) -> [chunk_id, ...]
        self.chunks_by_company_year: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        # normalized_metric -> [chunk_id, ...]
        self.chunks_by_metric: Dict[str, List[str]] = defaultdict(list)
        # year -> [chunk_id, ...]
        self.chunks_by_year: Dict[str, List[str]] = defaultdict(list)
        # chunk_id -> [adjacent_chunk_id, ...] (same doc, nearby offsets)
        self.adjacent_chunks: Dict[str, List[str]] = defaultdict(list)
        # chunk_id -> [(neighbor_chunk_id, similarity), ...]
        self.semantic_neighbors: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
        # chunk_id -> word offset within document
        self._chunk_offset: Dict[str, int] = {}
        # doc_id -> sorted list of (offset, chunk_id)
        self._doc_order: Dict[str, List[Tuple[int, str]]] = defaultdict(list)

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(
        self,
        chunks: List[Chunk],
        graph: Optional[FinancialEvidenceGraph] = None,
    ) -> "GraphExpansionIndex":
        """Build all lookup structures from a list of chunks and optional graph.

        Args:
            chunks: All corpus chunks.
            graph: Optional FinancialEvidenceGraph for semantic edges and
                chunk metadata.
        """
        t0 = time.time()

        for c in chunks:
            self.chunk_lookup[c.chunk_id] = c
            # doc index
            if c.doc_id:
                self.chunks_by_doc[c.doc_id].append(c.chunk_id)
            # section index
            if c.doc_id and c.section:
                self.chunks_by_section[(c.doc_id, c.section)].append(c.chunk_id)
            # company+year index
            if c.company and c.filing_year:
                self.chunks_by_company_year[(c.company, c.filing_year)].append(c.chunk_id)
            # year index
            if c.filing_year:
                self.chunks_by_year[c.filing_year].append(c.chunk_id)
            # metric index (from extracted entities)
            ents = _entity_extractor.extract(c)
            for m in ents.metrics:
                normalized = _normalize_metric(m)
                self.chunks_by_metric[normalized].append(c.chunk_id)
            # offset tracking
            offset = c.metadata.get("word_offset", c.metadata.get("row_offset", 0))
            self._chunk_offset[c.chunk_id] = offset
            if c.doc_id:
                self._doc_order[c.doc_id].append((offset, c.chunk_id))

        # Sort per-doc order for adjacent chunk lookup
        for doc_id in self._doc_order:
            self._doc_order[doc_id].sort(key=lambda x: x[0])

        # Build adjacent chunk links
        for doc_id, ordered in self._doc_order.items():
            cids = [cid for _, cid in ordered]
            for i, cid in enumerate(cids):
                neighbors: List[str] = []
                if i > 0:
                    neighbors.append(cids[i - 1])
                if i > 1:
                    neighbors.append(cids[i - 2])
                if i < len(cids) - 1:
                    neighbors.append(cids[i + 1])
                if i < len(cids) - 2:
                    neighbors.append(cids[i + 2])
                self.adjacent_chunks[cid] = neighbors

        # Extract semantic neighbors from graph
        if graph is not None:
            for u, v, _k, etype in graph.graph.edges(keys=True, data="edge_type"):
                if etype == "semantic-similar":
                    w = graph.graph.edges[u, v, _k].get("weight", 0.5)
                    if graph.node_types.get(u) == "chunk" and graph.node_types.get(v) == "chunk":
                        self.semantic_neighbors[u].append((v, float(w)))
                        self.semantic_neighbors[v].append((u, float(w)))

        elapsed = time.time() - t0
        n_chunks = len(self.chunk_lookup)
        n_semantic = sum(len(v) for v in self.semantic_neighbors.values())
        print(
            f"  [GraphExpansionIndex] Built in {elapsed:.1f}s: "
            f"{n_chunks} chunks, {len(self.chunks_by_doc)} docs, "
            f"{len(self.chunks_by_section)} sections, "
            f"{len(self.chunks_by_company_year)} company-years, "
            f"{len(self.chunks_by_metric)} metrics, "
            f"{n_semantic} semantic edges"
        )
        return self

    # ------------------------------------------------------------------
    # Cache
    # ------------------------------------------------------------------

    def compute_fingerprint(
        self,
        corpus_cache_path: Optional[str] = None,
        graph_cache_path: Optional[str] = None,
    ) -> str:
        """Compute a stable fingerprint for cache validation."""
        parts: List[str] = [f"schema_v{INDEX_SCHEMA_VERSION}"]

        if corpus_cache_path:
            p = Path(corpus_cache_path)
            if p.exists():
                st = p.stat()
                parts.append(f"corpus:{p}:{st.st_size}:{st.st_mtime}")
        if graph_cache_path:
            p = Path(graph_cache_path)
            if p.exists():
                st = p.stat()
                parts.append(f"graph:{p}:{st.st_size}:{st.st_mtime}")

        parts.append(f"relations:{','.join(sorted(RELATION_NAMES))}")
        parts.append(f"n_chunks:{len(self.chunk_lookup)}")

        h = hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]
        return h

    def save(self, path: str | Path) -> None:
        """Pickle the index to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str | Path) -> "GraphExpansionIndex":
        """Load a pickled index from disk."""
        with open(path, "rb") as fh:
            return pickle.load(fh)


# ═════════════════════════════════════════════════════════════════════════════
# Budgeted Graph Expander
# ═════════════════════════════════════════════════════════════════════════════

class BudgetedGraphExpander:
    """Budget-constrained multi-relation candidate expansion.

    Given initial top-N candidates and (optionally) relation probabilities
    from a router, expands the candidate pool by following graph relations
    within a fixed budget.
    """

    def __init__(
        self,
        index: GraphExpansionIndex,
        relation_prior: Optional[Dict[str, float]] = None,
        initial_top_n: int = DEFAULT_INITIAL_TOP_N,
        seed_top_m: int = DEFAULT_SEED_TOP_M,
        expansion_budget: int = DEFAULT_EXPANSION_BUDGET,
        max_budget_per_relation: int = DEFAULT_MAX_BUDGET_PER_RELATION,
        max_total_candidates: int = DEFAULT_MAX_TOTAL_CANDIDATES,
        relation_threshold: float = DEFAULT_RELATION_THRESHOLD,
        semantic_max_per_query: int = DEFAULT_SEMANTIC_MAX_PER_QUERY,
    ):
        self.index = index
        self.relation_prior = relation_prior or dict(DEFAULT_RELATION_PRIOR)
        self.initial_top_n = initial_top_n
        self.seed_top_m = seed_top_m
        self.expansion_budget = expansion_budget
        self.max_budget_per_relation = max_budget_per_relation
        self.max_total_candidates = max_total_candidates
        self.relation_threshold = relation_threshold
        self.semantic_max_per_query = semantic_max_per_query

    # ------------------------------------------------------------------
    # Main expansion entry point
    # ------------------------------------------------------------------

    def expand(
        self,
        query: str,
        initial_candidates: List[Tuple[Chunk, float]],
        relation_probabilities: Optional[Dict[str, float]] = None,
        seed_random: Optional[Any] = None,
    ) -> Tuple[List[ExpandedCandidate], Dict[str, Any]]:
        """Expand the candidate pool using graph relations.

        Args:
            query: The question text.
            initial_candidates: List of (Chunk, score) from initial retrieval,
                ordered by descending score.
            relation_probabilities: Optional dict from relation_name -> prob
                (e.g., from QueryRelationRouter). If None, uses uniform/prior
                probabilities for fixed-budget expansion.
            seed_random: Random state for deterministic tie-breaking.

        Returns:
            (merged_candidates, expansion_stats) where merged_candidates is
            a deduplicated list of ExpandedCandidate sorted by priority, and
            expansion_stats is a dict with diagnostic information.
        """
        t0 = time.time()

        # Default uniform probabilities if no router
        if relation_probabilities is None:
            relation_probabilities = {r: 0.5 for r in RELATION_NAMES}

        # Collect seed chunks from initial top-M
        seeds = initial_candidates[: self.seed_top_m]
        seed_chunk_ids: Set[str] = set()
        seed_by_id: Dict[str, Tuple[Chunk, float, int]] = {}
        for rank, (chunk, score) in enumerate(seeds, start=1):
            seed_chunk_ids.add(chunk.chunk_id)
            seed_by_id[chunk.chunk_id] = (chunk, score, rank)

        # Build initial candidate tracking
        initial_candidates_map: Dict[str, ExpandedCandidate] = {}
        for rank, (chunk, score) in enumerate(initial_candidates[: self.initial_top_n], start=1):
            initial_candidates_map[chunk.chunk_id] = ExpandedCandidate(
                chunk_id=chunk.chunk_id,
                is_initial=True,
                initial_score=score,
                initial_rank=rank,
            )

        # Filter relations below threshold and compute budgets
        active_relations = {
            r: p for r, p in relation_probabilities.items() if p >= self.relation_threshold
        }
        budgets = self._allocate_budgets(active_relations, self.expansion_budget)

        # Expand per relation
        expanded: Dict[str, ExpandedCandidate] = {}
        relation_stats: Dict[str, Dict[str, int]] = {}

        for rel_name, budget in budgets.items():
            if budget <= 0:
                continue
            candidates = self._expand_relation(
                rel_name, query, seeds, seed_by_id, budget, seed_random
            )
            rel_added = 0
            for ec in candidates:
                if ec.chunk_id in initial_candidates_map:
                    continue
                if ec.chunk_id in expanded:
                    # Merge source relations
                    existing = expanded[ec.chunk_id]
                    for sr in ec.source_relations:
                        if sr not in existing.source_relations:
                            existing.source_relations.append(sr)
                    if ec.expansion_priority > existing.expansion_priority:
                        existing.expansion_priority = ec.expansion_priority
                        existing.best_relation = ec.best_relation
                        existing.best_seed_chunk_id = ec.best_seed_chunk_id
                        existing.best_seed_rank = ec.best_seed_rank
                else:
                    expanded[ec.chunk_id] = ec
                    rel_added += 1
            relation_stats[rel_name] = {
                "budget": budget,
                "candidates_found": len(candidates),
                "new_added": rel_added,
                "router_prob": relation_probabilities.get(rel_name, 0.5),
            }

        # Merge initial + expanded, cap at max_total_candidates
        all_candidates: Dict[str, ExpandedCandidate] = {}
        all_candidates.update(initial_candidates_map)
        all_candidates.update(expanded)

        # Sort by priority (expanded first), then by initial score for ties
        sorted_candidates = sorted(
            all_candidates.values(),
            key=lambda ec: (
                not ec.is_initial,  # expanded first for diagnostic visibility
                -ec.expansion_priority,
                -ec.initial_score,
                ec.chunk_id,  # stable tie-break
            ),
        )

        if len(sorted_candidates) > self.max_total_candidates:
            sorted_candidates = sorted_candidates[: self.max_total_candidates]

        elapsed = time.time() - t0
        stats = {
            "num_initial": len(initial_candidates_map),
            "num_expanded": len(expanded),
            "num_total": len(sorted_candidates),
            "active_relations": list(active_relations.keys()),
            "budgets": budgets,
            "relation_stats": relation_stats,
            "expansion_time_s": round(elapsed, 3),
        }

        return sorted_candidates, stats

    # ------------------------------------------------------------------
    # Budget allocation
    # ------------------------------------------------------------------

    def _allocate_budgets(
        self,
        relation_probs: Dict[str, float],
        total_budget: int,
    ) -> Dict[str, int]:
        """Allocate integer budgets proportional to relation probabilities.

        Rules:
        1. Relations below threshold are excluded (caller handles this).
        2. Remaining probabilities are normalized.
        3. Budgets allocated proportionally.
        4. Single relation capped at max_budget_per_relation.
        5. Excess redistributed.
        """
        if not relation_probs:
            return {}

        total_prob = sum(relation_probs.values())
        if total_prob <= 0:
            return {}

        # Normalize and compute raw allocation
        raw: Dict[str, float] = {}
        for r, p in relation_probs.items():
            raw[r] = (p / total_prob) * total_budget

        # Cap per-relation
        budgets: Dict[str, int] = {}
        excess = 0.0
        for r, v in raw.items():
            capped = min(v, self.max_budget_per_relation)
            budgets[r] = max(1, int(round(capped))) if capped >= 0.5 else 0
            excess += v - capped

        # Redistribute excess to uncapped relations
        if excess > 0.5:
            uncapped = [r for r in budgets if raw[r] < self.max_budget_per_relation and budgets[r] > 0]
            if uncapped:
                extra_per = excess / len(uncapped)
                for r in uncapped:
                    budgets[r] = min(
                        self.max_budget_per_relation,
                        budgets[r] + max(1, int(round(extra_per))),
                    )

        # Ensure total doesn't exceed budget
        while sum(budgets.values()) > total_budget:
            max_r = max(budgets, key=lambda r: budgets[r])
            if budgets[max_r] > 1:
                budgets[max_r] -= 1
            else:
                break

        return {r: b for r, b in budgets.items() if b > 0}

    # ------------------------------------------------------------------
    # Per-relation expansion
    # ------------------------------------------------------------------

    def _expand_relation(
        self,
        rel_name: str,
        query: str,
        seeds: List[Tuple[Chunk, float]],
        seed_by_id: Dict[str, Tuple[Chunk, float, int]],
        budget: int,
        seed_random: Optional[Any] = None,
    ) -> List[ExpandedCandidate]:
        """Expand candidates along a single relation type.

        Args:
            rel_name: One of RELATION_NAMES.
            query: The question text.
            seeds: Seed chunks (top-M of initial retrieval).
            seed_by_id: Mapping from chunk_id -> (chunk, score, rank).
            budget: Maximum number of candidates to return from this relation.
            seed_random: Optional random state.

        Returns:
            List of ExpandedCandidate, sorted by expansion_priority descending.
        """
        if rel_name == "adjacent_chunk":
            return self._expand_adjacent(seeds, seed_by_id, budget)
        elif rel_name == "same_section":
            return self._expand_same_section(seeds, seed_by_id, budget)
        elif rel_name == "same_filing":
            return self._expand_same_filing(query, seeds, seed_by_id, budget)
        elif rel_name == "same_company_year":
            return self._expand_same_company_year(query, seeds, seed_by_id, budget)
        elif rel_name == "same_metric":
            return self._expand_same_metric(query, seeds, seed_by_id, budget)
        elif rel_name == "same_year":
            return self._expand_same_year(query, seeds, seed_by_id, budget)
        elif rel_name == "semantic_similar":
            return self._expand_semantic(seeds, seed_by_id, budget)
        else:
            return []

    # ------------------------------------------------------------------
    # Relation: adjacent_chunk
    # ------------------------------------------------------------------

    def _expand_adjacent(
        self,
        seeds: List[Tuple[Chunk, float]],
        seed_by_id: Dict[str, Tuple[Chunk, float, int]],
        budget: int,
    ) -> List[ExpandedCandidate]:
        """Adjacent chunks in same document, same section preferred."""
        candidates: List[ExpandedCandidate] = []
        seen: Set[str] = set()

        for chunk, score in seeds:
            seed_cid = chunk.chunk_id
            neighbors = self.index.adjacent_chunks.get(seed_cid, [])
            for nid in neighbors:
                if nid in seen:
                    continue
                nc = self.index.chunk_lookup.get(nid)
                if nc is None:
                    continue
                # Must be same doc
                if nc.doc_id != chunk.doc_id:
                    continue
                seen.add(nid)
                _, seed_score, seed_rank = seed_by_id.get(seed_cid, (chunk, score, 0))
                priority = self._compute_priority(
                    rel_name="adjacent_chunk",
                    router_prob=1.0,
                    seed_score=seed_score,
                    seed_rank=seed_rank or 1,
                    graph_distance=1,
                    local_match_bonus=1.0 if nc.section == chunk.section else 0.5,
                )
                candidates.append(
                    ExpandedCandidate(
                        chunk_id=nid,
                        is_initial=False,
                        initial_score=score,
                        initial_rank=None,
                        source_relations=["adjacent_chunk"],
                        best_relation="adjacent_chunk",
                        best_seed_chunk_id=seed_cid,
                        best_seed_rank=seed_rank,
                        graph_distance=1,
                        expansion_priority=priority,
                    )
                )

        candidates.sort(key=lambda c: -c.expansion_priority)
        return candidates[:budget]

    # ------------------------------------------------------------------
    # Relation: same_section
    # ------------------------------------------------------------------

    def _expand_same_section(
        self,
        seeds: List[Tuple[Chunk, float]],
        seed_by_id: Dict[str, Tuple[Chunk, float, int]],
        budget: int,
    ) -> List[ExpandedCandidate]:
        """Chunks in same (doc_id, section)."""
        candidates: List[ExpandedCandidate] = []
        seen: Set[str] = set()

        for chunk, score in seeds:
            if not chunk.doc_id or not chunk.section:
                continue
            section_key = (chunk.doc_id, chunk.section)
            section_chunks = self.index.chunks_by_section.get(section_key, [])
            seed_offset = self.index._chunk_offset.get(chunk.chunk_id, 0)

            for cid in section_chunks:
                if cid == chunk.chunk_id or cid in seen:
                    continue
                seen.add(cid)
                nc = self.index.chunk_lookup.get(cid)
                if nc is None:
                    continue
                offset_dist = abs(self.index._chunk_offset.get(cid, 0) - seed_offset)
                _, seed_score, seed_rank = seed_by_id.get(chunk.chunk_id, (chunk, score, 0))
                priority = self._compute_priority(
                    rel_name="same_section",
                    router_prob=1.0,
                    seed_score=seed_score,
                    seed_rank=seed_rank or 1,
                    graph_distance=1,
                    local_match_bonus=1.0 / (1.0 + offset_dist / 1000.0),
                )
                candidates.append(
                    ExpandedCandidate(
                        chunk_id=cid,
                        is_initial=False,
                        initial_score=score,
                        initial_rank=None,
                        source_relations=["same_section"],
                        best_relation="same_section",
                        best_seed_chunk_id=chunk.chunk_id,
                        best_seed_rank=seed_rank,
                        graph_distance=1,
                        expansion_priority=priority,
                    )
                )

        candidates.sort(key=lambda c: -c.expansion_priority)
        return candidates[:budget]

    # ------------------------------------------------------------------
    # Relation: same_filing
    # ------------------------------------------------------------------

    def _expand_same_filing(
        self,
        query: str,
        seeds: List[Tuple[Chunk, float]],
        seed_by_id: Dict[str, Tuple[Chunk, float, int]],
        budget: int,
    ) -> List[ExpandedCandidate]:
        """Chunks from same doc_id, preferring tables, adjacent sections, query tokens."""
        candidates: List[ExpandedCandidate] = []
        seen: Set[str] = {c.chunk_id for c, _ in seeds}
        q_tokens = set(query.lower().split())

        for chunk, score in seeds:
            if not chunk.doc_id:
                continue
            doc_chunks = self.index.chunks_by_doc.get(chunk.doc_id, [])
            _, seed_score, seed_rank = seed_by_id.get(chunk.chunk_id, (chunk, score, 0))

            for cid in doc_chunks:
                if cid in seen:
                    continue
                nc = self.index.chunk_lookup.get(cid)
                if nc is None:
                    continue

                # Compute local bonuses
                bonus = 1.0
                if nc.chunk_type == "table":
                    bonus *= 1.2
                if nc.section != chunk.section:
                    # Prefer adjacent sections
                    bonus *= 0.8
                c_tokens = set(nc.text.lower().split())
                token_overlap = len(q_tokens & c_tokens)
                if token_overlap > 0:
                    bonus *= 1.0 + min(token_overlap / max(len(q_tokens), 1), 0.5)

                priority = self._compute_priority(
                    rel_name="same_filing",
                    router_prob=1.0,
                    seed_score=seed_score,
                    seed_rank=seed_rank or 1,
                    graph_distance=1,
                    local_match_bonus=bonus,
                )
                candidates.append(
                    ExpandedCandidate(
                        chunk_id=cid,
                        is_initial=False,
                        initial_score=score,
                        initial_rank=None,
                        source_relations=["same_filing"],
                        best_relation="same_filing",
                        best_seed_chunk_id=chunk.chunk_id,
                        best_seed_rank=seed_rank,
                        graph_distance=1,
                        expansion_priority=priority,
                    )
                )

        # Deduplicate by chunk_id keeping highest priority
        best: Dict[str, ExpandedCandidate] = {}
        for ec in candidates:
            if ec.chunk_id not in best or ec.expansion_priority > best[ec.chunk_id].expansion_priority:
                best[ec.chunk_id] = ec

        sorted_candidates = sorted(best.values(), key=lambda c: -c.expansion_priority)
        return sorted_candidates[:budget]

    # ------------------------------------------------------------------
    # Relation: same_company_year
    # ------------------------------------------------------------------

    def _expand_same_company_year(
        self,
        query: str,
        seeds: List[Tuple[Chunk, float]],
        seed_by_id: Dict[str, Tuple[Chunk, float, int]],
        budget: int,
    ) -> List[ExpandedCandidate]:
        """Chunks with same company and filing_year; prefer query metric matches."""
        candidates: List[ExpandedCandidate] = []
        seen: Set[str] = {c.chunk_id for c, _ in seeds}

        q_metrics = _entity_extractor.extract_metrics(query)
        q_metrics_norm = {_normalize_metric(m) for m in q_metrics}

        for chunk, score in seeds:
            if not chunk.company or not chunk.filing_year:
                continue
            cy_key = (chunk.company, chunk.filing_year)
            cy_chunks = self.index.chunks_by_company_year.get(cy_key, [])
            _, seed_score, seed_rank = seed_by_id.get(chunk.chunk_id, (chunk, score, 0))

            for cid in cy_chunks:
                if cid in seen:
                    continue
                nc = self.index.chunk_lookup.get(cid)
                if nc is None:
                    continue

                bonus = 1.0
                if q_metrics_norm:
                    c_metrics = {_normalize_metric(m) for m in _entity_extractor.extract_metrics(nc.text)}
                    if c_metrics & q_metrics_norm:
                        bonus *= 1.5

                priority = self._compute_priority(
                    rel_name="same_company_year",
                    router_prob=1.0,
                    seed_score=seed_score,
                    seed_rank=seed_rank or 1,
                    graph_distance=1,
                    local_match_bonus=bonus,
                )
                candidates.append(
                    ExpandedCandidate(
                        chunk_id=cid,
                        is_initial=False,
                        initial_score=score,
                        initial_rank=None,
                        source_relations=["same_company_year"],
                        best_relation="same_company_year",
                        best_seed_chunk_id=chunk.chunk_id,
                        best_seed_rank=seed_rank,
                        graph_distance=1,
                        expansion_priority=priority,
                    )
                )

        best: Dict[str, ExpandedCandidate] = {}
        for ec in candidates:
            if ec.chunk_id not in best or ec.expansion_priority > best[ec.chunk_id].expansion_priority:
                best[ec.chunk_id] = ec

        sorted_candidates = sorted(best.values(), key=lambda c: -c.expansion_priority)
        return sorted_candidates[:budget]

    # ------------------------------------------------------------------
    # Relation: same_metric
    # ------------------------------------------------------------------

    def _expand_same_metric(
        self,
        query: str,
        seeds: List[Tuple[Chunk, float]],
        seed_by_id: Dict[str, Tuple[Chunk, float, int]],
        budget: int,
    ) -> List[ExpandedCandidate]:
        """Chunks sharing at least one normalized metric with query; prefer same company/year/filing."""
        candidates: List[ExpandedCandidate] = []
        seen: Set[str] = {c.chunk_id for c, _ in seeds}

        q_metrics = _entity_extractor.extract_metrics(query)
        q_metrics_norm = {_normalize_metric(m) for m in q_metrics}
        if not q_metrics_norm:
            return []

        for chunk, score in seeds:
            _, seed_score, seed_rank = seed_by_id.get(chunk.chunk_id, (chunk, score, 0))

            for qm in q_metrics_norm:
                metric_chunks = self.index.chunks_by_metric.get(qm, [])
                for cid in metric_chunks:
                    if cid in seen:
                        continue
                    nc = self.index.chunk_lookup.get(cid)
                    if nc is None:
                        continue

                    # Bonus for same company, same year, same filing
                    bonus = 1.0
                    if chunk.company and nc.company == chunk.company:
                        bonus *= 1.3
                    if chunk.filing_year and nc.filing_year == chunk.filing_year:
                        bonus *= 1.2
                    if chunk.doc_id and nc.doc_id == chunk.doc_id:
                        bonus *= 1.1

                    priority = self._compute_priority(
                        rel_name="same_metric",
                        router_prob=1.0,
                        seed_score=seed_score,
                        seed_rank=seed_rank or 1,
                        graph_distance=2,
                        local_match_bonus=bonus,
                    )
                    candidates.append(
                        ExpandedCandidate(
                            chunk_id=cid,
                            is_initial=False,
                            initial_score=score,
                            initial_rank=None,
                            source_relations=["same_metric"],
                            best_relation="same_metric",
                            best_seed_chunk_id=chunk.chunk_id,
                            best_seed_rank=seed_rank,
                            graph_distance=2,
                            expansion_priority=priority,
                        )
                    )

        # Deduplicate
        best: Dict[str, ExpandedCandidate] = {}
        for ec in candidates:
            if ec.chunk_id not in best or ec.expansion_priority > best[ec.chunk_id].expansion_priority:
                best[ec.chunk_id] = ec

        # Don't flood with unrelated-company same-metric candidates
        sorted_candidates = sorted(best.values(), key=lambda c: -c.expansion_priority)
        return sorted_candidates[:budget]

    # ------------------------------------------------------------------
    # Relation: same_year
    # ------------------------------------------------------------------

    def _expand_same_year(
        self,
        query: str,
        seeds: List[Tuple[Chunk, float]],
        seed_by_id: Dict[str, Tuple[Chunk, float, int]],
        budget: int,
    ) -> List[ExpandedCandidate]:
        """Chunks with same filing_year, only when query mentions a year; prefer same company/filing."""
        candidates: List[ExpandedCandidate] = []
        seen: Set[str] = {c.chunk_id for c, _ in seeds}

        q_years = _entity_extractor.extract_years(query)
        if not q_years:
            return []

        for chunk, score in seeds:
            if not chunk.filing_year:
                continue
            _, seed_score, seed_rank = seed_by_id.get(chunk.chunk_id, (chunk, score, 0))
            year_chunks = self.index.chunks_by_year.get(chunk.filing_year, [])

            for cid in year_chunks:
                if cid in seen:
                    continue
                nc = self.index.chunk_lookup.get(cid)
                if nc is None:
                    continue

                # Prefer same company or same filing
                bonus = 1.0
                if chunk.company and nc.company == chunk.company:
                    bonus *= 1.5
                elif chunk.doc_id and nc.doc_id == chunk.doc_id:
                    bonus *= 1.2

                priority = self._compute_priority(
                    rel_name="same_year",
                    router_prob=1.0,
                    seed_score=seed_score,
                    seed_rank=seed_rank or 1,
                    graph_distance=2,
                    local_match_bonus=bonus,
                )
                candidates.append(
                    ExpandedCandidate(
                        chunk_id=cid,
                        is_initial=False,
                        initial_score=score,
                        initial_rank=None,
                        source_relations=["same_year"],
                        best_relation="same_year",
                        best_seed_chunk_id=chunk.chunk_id,
                        best_seed_rank=seed_rank,
                        graph_distance=2,
                        expansion_priority=priority,
                    )
                )

        best: Dict[str, ExpandedCandidate] = {}
        for ec in candidates:
            if ec.chunk_id not in best or ec.expansion_priority > best[ec.chunk_id].expansion_priority:
                best[ec.chunk_id] = ec

        sorted_candidates = sorted(best.values(), key=lambda c: -c.expansion_priority)
        return sorted_candidates[:budget]

    # ------------------------------------------------------------------
    # Relation: semantic_similar
    # ------------------------------------------------------------------

    def _expand_semantic(
        self,
        seeds: List[Tuple[Chunk, float]],
        seed_by_id: Dict[str, Tuple[Chunk, float, int]],
        budget: int,
    ) -> List[ExpandedCandidate]:
        """Semantic neighbours from graph edges, limited per query."""
        candidates: List[ExpandedCandidate] = []
        seen: Set[str] = {c.chunk_id for c, _ in seeds}
        effective_budget = min(budget, self.semantic_max_per_query)

        for chunk, score in seeds:
            neighbors = self.index.semantic_neighbors.get(chunk.chunk_id, [])
            _, seed_score, seed_rank = seed_by_id.get(chunk.chunk_id, (chunk, score, 0))
            for nid, sim in sorted(neighbors, key=lambda x: -x[1]):
                if nid in seen:
                    continue
                seen.add(nid)
                nc = self.index.chunk_lookup.get(nid)
                if nc is None:
                    continue
                priority = self._compute_priority(
                    rel_name="semantic_similar",
                    router_prob=0.5,
                    seed_score=seed_score,
                    seed_rank=seed_rank or 1,
                    graph_distance=1,
                    local_match_bonus=sim,
                )
                candidates.append(
                    ExpandedCandidate(
                        chunk_id=nid,
                        is_initial=False,
                        initial_score=score,
                        initial_rank=None,
                        source_relations=["semantic_similar"],
                        best_relation="semantic_similar",
                        best_seed_chunk_id=chunk.chunk_id,
                        best_seed_rank=seed_rank,
                        graph_distance=1,
                        expansion_priority=priority,
                    )
                )

        candidates.sort(key=lambda c: -c.expansion_priority)
        return candidates[:effective_budget]

    # ------------------------------------------------------------------
    # Priority computation
    # ------------------------------------------------------------------

    def _compute_priority(
        self,
        rel_name: str,
        router_prob: float,
        seed_score: float,
        seed_rank: int,
        graph_distance: int,
        local_match_bonus: float = 1.0,
    ) -> float:
        """Compute expansion priority per the paper formula.

        priority = router_prob * relation_prior * seed_score * (1/seed_rank)
                 * (1/(1 + graph_distance)) * local_match_bonus
        """
        rel_prior = self.relation_prior.get(rel_name, 0.5)
        # Normalize seed_score to [0, 1] range if needed
        norm_score = max(0.0, min(1.0, seed_score / max(abs(seed_score), 1e-8)))
        rank_factor = 1.0 / max(seed_rank, 1)

        priority = (
            router_prob
            * rel_prior
            * norm_score
            * rank_factor
            * (1.0 / (1.0 + graph_distance))
            * local_match_bonus
        )
        return priority


# ═════════════════════════════════════════════════════════════════════════════
# Diagnostic helpers
# ═════════════════════════════════════════════════════════════════════════════

def compute_expansion_diagnostics(
    per_query_results: List[Dict],
    gold_map: Dict[str, List[str]],
) -> Dict[str, Any]:
    """Compute before/after expansion candidate recall diagnostics.

    Args:
        per_query_results: List of per-query result dicts, each containing:
            - gold_evidence_ids
            - initial_chunk_ids
            - expanded_chunk_ids
        gold_map: question_id -> [gold_chunk_id, ...]

    Returns:
        Dict with aggregate expansion statistics.
    """
    total_queries = len(per_query_results)
    if total_queries == 0:
        return {}

    before_recalls = []
    after_recalls = []
    new_gold_count = 0
    total_expanded = 0
    total_initial = 0
    queries_no_relation = 0
    relation_usage: Dict[str, int] = defaultdict(int)
    relation_gold_recovery: Dict[str, int] = defaultdict(int)

    for r in per_query_results:
        gold = set(r.get("gold_evidence_ids", []))
        initial = set(r.get("initial_chunk_ids", []))
        expanded = set(r.get("expanded_chunk_ids", []))
        all_ids = initial | expanded

        total_initial += len(initial)
        total_expanded += len(expanded)

        # Recall before/after
        if gold:
            before_recalls.append(len(gold & initial) / len(gold))
            after_recalls.append(len(gold & all_ids) / len(gold))
        else:
            before_recalls.append(0.0)
            after_recalls.append(0.0)

        # New gold recovered
        new_gold = gold & (expanded - initial)
        if new_gold:
            new_gold_count += 1

        # Relation usage from relation_probabilities
        rel_probs = r.get("relation_probabilities", {})
        active = [rn for rn, p in rel_probs.items() if p > 0]
        if not active:
            queries_no_relation += 1
        for rn in active:
            relation_usage[rn] += 1

        # Relation gold recovery
        for rn in active:
            # Check if any expanded chunk from this relation recovers gold
            # (simplified: if any new gold was found, credit all active relations)
            if new_gold:
                relation_gold_recovery[rn] += 1

    avg_before = float(np.mean(before_recalls)) if before_recalls else 0.0
    avg_after = float(np.mean(after_recalls)) if after_recalls else 0.0

    return {
        "num_queries": total_queries,
        "avg_initial_candidates": total_initial / total_queries if total_queries else 0,
        "avg_expanded_candidates": total_expanded / total_queries if total_queries else 0,
        "avg_total_candidates": (total_initial + total_expanded) / total_queries if total_queries else 0,
        "candidate_recall_before_expansion": round(avg_before, 4),
        "candidate_recall_after_expansion": round(avg_after, 4),
        "num_queries_with_new_gold": new_gold_count,
        "new_gold_recovery_rate": round(new_gold_count / total_queries, 4) if total_queries else 0,
        "num_queries_no_relation_active": queries_no_relation,
        "relation_usage": dict(relation_usage),
        "relation_gold_recovery": dict(relation_gold_recovery),
    }


# ═════════════════════════════════════════════════════════════════════════════
# Metric normalization
# ═════════════════════════════════════════════════════════════════════════════

_METRIC_ALIASES: Dict[str, str] = {
    "revenues": "revenue",
    "net sales": "revenue",
    "sales": "revenue",
    "net earnings": "net income",
    "profit": "net income",
    "net loss": "net income",
    "earnings per share": "eps",
    "eps": "eps",
    "diluted eps": "eps",
    "basic eps": "eps",
    "operating earnings": "operating income",
    "operating loss": "operating income",
    "ebitda": "ebitda",
    "adjusted ebitda": "ebitda",
    "ebit": "ebit",
    "gross profit": "gross profit",
    "gross margin": "gross margin",
    "total assets": "total assets",
    "total liabilities": "total liabilities",
    "total equity": "total equity",
    "shareholders equity": "total equity",
    "stockholders equity": "total equity",
    "cash and cash equivalents": "cash",
    "cash flow": "cash flow",
    "free cash flow": "free cash flow",
    "operating cash flow": "operating cash flow",
    "operating expenses": "operating expenses",
    "r&d expenses": "r&d",
    "research and development": "r&d",
    "sg&a": "sga",
    "selling general and administrative": "sga",
    "cost of revenue": "cogs",
    "cost of sales": "cogs",
    "cost of goods sold": "cogs",
    "cogs": "cogs",
    "working capital": "working capital",
    "long-term debt": "long term debt",
    "short-term debt": "short term debt",
    "dividends per share": "dividends",
    "capital expenditures": "capex",
    "capex": "capex",
    "roe": "roe",
    "roa": "roa",
    "roic": "roic",
    "return on equity": "roe",
    "return on assets": "roa",
    "return on invested capital": "roic",
    "market cap": "market cap",
    "market capitalization": "market cap",
    "enterprise value": "enterprise value",
}


def _normalize_metric(metric: str) -> str:
    """Normalize a metric string to its canonical form."""
    cleaned = metric.strip().lower()
    # Remove extra whitespace
    cleaned = " ".join(cleaned.split())
    return _METRIC_ALIASES.get(cleaned, cleaned)


def normalize_metric(metric: str) -> str:
    """Public API for metric normalization."""
    return _normalize_metric(metric)
