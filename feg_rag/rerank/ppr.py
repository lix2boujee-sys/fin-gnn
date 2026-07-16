"""Personalized PageRank (PPR) evidence reranking.

Paper plan §8.2: PPR serves as the graph-algorithm baseline before GNN.
Seed nodes: question entities + initial retrieval chunks.

Runs PPR on a candidate-local subgraph (not the full corpus graph) and
fuses graph scores with initial retrieval scores to avoid destroying
Hybrid ranking (which caused MRR drops in Exp3/Exp4).

Internally converts the directed MultiDiGraph to a bidirectional DiGraph
so that PageRank can propagate score from entity seeds (metric/year) back
to candidate chunks.  The original FinancialEvidenceGraph is **never**
modified.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import networkx as nx

from feg_rag.data.chunker import Chunk
from feg_rag.graph.builder import FinancialEvidenceGraph


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_bidirectional_ppr_graph(
    nxg: Union[nx.MultiDiGraph, nx.DiGraph],
) -> nx.DiGraph:
    """Convert a (possibly multi-)directed graph into a simple **bidirectional**
    DiGraph suitable for Personalized PageRank.

    For every directed edge  u → v  in the input graph:

        * add  u → v  with weight *w*
        * add  v → u  with the same weight *w*

    When multiple edges exist for the same ordered pair (u, v) — common in
    ``MultiDiGraph`` — we take the **maximum** weight rather than summing.
    This prevents high-frequency relations (e.g. many chunks mentioning the
    same metric) from dominating PageRank, while preserving the strongest
    signal for each direction.

    All nodes are copied so that isolated seed / candidate nodes are not
    lost.
    """
    ppr_graph = nx.DiGraph()

    # Copy every node first (preserves isolated nodes)
    for node_id in nxg.nodes():
        ppr_graph.add_node(node_id, **nxg.nodes[node_id])

    # Collect edge weights: (u, v) → max weight
    edge_max_weight: Dict[Tuple[str, str], float] = {}
    for u, v, data in nxg.edges(data=True):
        w = data.get("weight", 1.0)
        key = (u, v)
        if key not in edge_max_weight or w > edge_max_weight[key]:
            edge_max_weight[key] = w

    # Add bidirectional edges
    for (u, v), w in edge_max_weight.items():
        ppr_graph.add_edge(u, v, weight=w)
        ppr_graph.add_edge(v, u, weight=w)

    return ppr_graph


def _normalise_dict(d: Dict[str, float]) -> Dict[str, float]:
    if not d:
        return {}
    vals = list(d.values())
    vmin, vmax = min(vals), max(vals)
    if vmax == vmin:
        return {k: 0.5 for k in d}
    return {k: (v - vmin) / (vmax - vmin) for k, v in d.items()}


def _build_subgraph(
    nxg: Union[nx.DiGraph, nx.MultiDiGraph],
    candidate_chunk_ids: List[str],
    seed_chunk_ids: List[str],
    seed_metric_names: Optional[List[str]] = None,
    seed_year_values: Optional[List[str]] = None,
) -> nx.DiGraph:
    """Restrict PPR to candidates, seeds, and their 1-hop neighbours."""
    nodes: set[str] = set(candidate_chunk_ids) | set(seed_chunk_ids)
    if seed_metric_names:
        for m in seed_metric_names:
            nodes.add(f"metric::{m}")
    if seed_year_values:
        for y in seed_year_values:
            nodes.add(f"year::{y}")

    expanded = set(nodes)
    for n in list(nodes):
        if n not in nxg:
            continue
        expanded.update(nxg.predecessors(n))
        expanded.update(nxg.successors(n))

    return nxg.subgraph(expanded).copy()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

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
    retrieval_scores: Optional[Dict[str, float]] = None,
    retrieval_weight: float = 0.5,
) -> List[Tuple[str, float]]:
    """Run Personalized PageRank and return reranked chunk scores.

    When ``retrieval_scores`` is provided, final score is a convex combination
    of normalised retrieval and PPR scores among candidates only.
    """
    del chunks  # API compatibility

    if not candidate_chunk_ids:
        return []

    # Build bidirectional PPR graph (does NOT modify the original graph)
    ppr_graph = _to_bidirectional_ppr_graph(graph.graph)

    subg = _build_subgraph(
        ppr_graph, candidate_chunk_ids, seed_chunk_ids,
        seed_metric_names, seed_year_values,
    )

    personalization: Dict[str, float] = {n: 0.0 for n in subg.nodes()}
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
        # No seeds: return all candidates with uniform score
        return [(cid, 1.0 / len(candidate_chunk_ids)) for cid in candidate_chunk_ids]

    for n in personalization:
        personalization[n] /= total_seeds

    # If subgraph is empty or has no edges, return uniform scores
    if subg.number_of_nodes() == 0:
        return [(cid, 1.0 / len(candidate_chunk_ids)) for cid in candidate_chunk_ids]

    has_weight = any(
        subg.edges[e].get("weight") is not None for e in subg.edges
    )
    ppr_scores = nx.pagerank(
        subg,
        alpha=alpha,
        personalization=personalization,
        max_iter=max_iter,
        tol=tol,
        weight="weight" if has_weight else None,
    )

    # Build ppr_only: all candidates get their PPR score (default 0.0 if
    # PageRank didn't reach them).  This ensures every candidate appears in
    # the final output even when the subgraph omits some.
    ppr_only = {
        cid: ppr_scores.get(cid, 0.0)
        for cid in candidate_chunk_ids
    }

    if not retrieval_scores:
        ranked = sorted(ppr_only.items(), key=lambda x: x[1], reverse=True)
        return ranked

    # Fusion: convex combination of normalised retrieval and PPR scores
    ret_subset = {cid: retrieval_scores.get(cid, 0.0) for cid in candidate_chunk_ids}
    ret_norm = _normalise_dict(ret_subset)
    ppr_norm = _normalise_dict(ppr_only)
    rw = min(max(retrieval_weight, 0.0), 1.0)

    combined = [
        (cid, rw * ret_norm.get(cid, 0.0) + (1.0 - rw) * ppr_norm.get(cid, 0.0))
        for cid in candidate_chunk_ids
    ]
    combined.sort(key=lambda x: x[1], reverse=True)
    return combined
