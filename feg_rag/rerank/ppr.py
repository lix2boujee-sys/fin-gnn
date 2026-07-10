"""Personalized PageRank (PPR) evidence reranking.

Paper plan §8.2: PPR serves as the graph-algorithm baseline before GNN.
Seed nodes: question entities + initial retrieval chunks.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np

from feg_rag.data.chunker import Chunk
from feg_rag.graph.builder import FinancialEvidenceGraph


def ppr_rerank(
    graph: FinancialEvidenceGraph,
    chunks: List[Chunk],
    candidate_chunk_ids: List[str],
    seed_chunk_ids: List[str],
    seed_metric_names: Optional[List[str]] = None,
    seed_year_values: Optional[List[str]] = None,
    alpha: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> List[Tuple[str, float]]:
    """Run Personalized PageRank and return reranked chunk scores.

    Seeds are constructed from:
        - Initial retrieval candidate chunks
        - Question-matched metric nodes
        - Question-matched year nodes

    Args:
        graph: The financial evidence graph.
        chunks: All chunks (unused; kept for API consistency).
        candidate_chunk_ids: Chunk IDs from initial retrieval.
        seed_chunk_ids: Seed chunk IDs (e.g., top-k from initial retrieval).
        seed_metric_names: Metric names extracted from the question.
        seed_year_values: Year values extracted from the question.
        alpha: PageRank damping factor.
        max_iter: Maximum iterations.
        tol: Convergence tolerance.

    Returns:
        List of (chunk_id, ppr_score) sorted descending.
    """
    nxg = graph.graph

    # Build personalisation vector
    all_nodes = list(nxg.nodes())
    personalization: Dict[str, float] = {n: 0.0 for n in all_nodes}

    # Seed weight distribution
    total_seeds = 0
    for cid in seed_chunk_ids:
        if cid in personalization:
            personalization[cid] += 1.0
            total_seeds += 1

    if seed_metric_names:
        for m in seed_metric_names:
            m_node = f"metric::{m}"
            if m_node in personalization:
                personalization[m_node] += 1.0
                total_seeds += 1

    if seed_year_values:
        for y in seed_year_values:
            y_node = f"year::{y}"
            if y_node in personalization:
                personalization[y_node] += 1.0
                total_seeds += 1

    if total_seeds == 0:
        # Fall back to uniform
        return [(cid, 1.0 / len(candidate_chunk_ids)) for cid in candidate_chunk_ids]

    # Normalize
    for n in personalization:
        personalization[n] /= total_seeds

    # Run PPR (use edge weights if present)
    ppr_scores = nx.pagerank(
        nxg,
        alpha=alpha,
        personalization=personalization,
        max_iter=max_iter,
        tol=tol,
        weight="weight",  # will be ignored if no 'weight' attr on edges
    )

    # Keep only chunk nodes, sorted
    chunk_scores = [
        (cid, ppr_scores.get(cid, 0.0))
        for cid in candidate_chunk_ids
        if cid in ppr_scores
    ]
    chunk_scores.sort(key=lambda x: x[1], reverse=True)
    return chunk_scores
