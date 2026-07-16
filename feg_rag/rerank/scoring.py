"""Shared score normalisation utilities for GNN reranker fusion.

Used by both GraphSAGE (gnn.py) and R-GCN (rgcn.py) to normalise retrieval,
graph/PPR, and GNN logit scores before linear fusion.
"""

from __future__ import annotations

from typing import Dict


def normalise_score_map(score_map: Dict[str, float]) -> Dict[str, float]:
    """Min-max normalise values in *score_map* to [0, 1].

    When all values are equal (or the map has at most one entry), every key
    receives **0.5** — a stable neutral default that avoids division-by-zero
    and treats all candidates equally when the score provides no
    discrimination.

    Missing keys in downstream lookups should default to **0.0** (the caller
    is responsible for this fallback).
    """
    if not score_map:
        return {}
    vals = list(score_map.values())
    vmin, vmax = min(vals), max(vals)
    if vmax == vmin:
        return {k: 0.5 for k in score_map}
    return {k: (v - vmin) / (vmax - vmin) for k, v in score_map.items()}
