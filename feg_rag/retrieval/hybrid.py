"""Hybrid retrieval: combines BM25 and dense scores."""

from __future__ import annotations

from typing import Dict, List, Tuple

from feg_rag.data.chunker import Chunk
from feg_rag.retrieval.bm25 import BM25Retriever
from feg_rag.retrieval.dense import DenseRetriever


class HybridRetriever:
    """Combines BM25 and dense retrieval with linear score fusion.

    final_score = alpha * norm_bm25_score + (1 - alpha) * norm_dense_score
    """

    def __init__(
        self,
        bm25: BM25Retriever,
        dense: DenseRetriever,
        alpha: float = 0.5,
    ):
        self.bm25 = bm25
        self.dense = dense
        self.alpha = alpha

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self, query: str, top_k: int = 50
    ) -> List[Tuple[Chunk, float]]:
        """Run both retrievers and fuse scores."""
        bm25_results = self.bm25.search(query, top_k=top_k * 2)
        dense_results = self.dense.search(query, top_k=top_k * 2)

        # Build score maps
        bm25_map = {c.chunk_id: s for c, s in bm25_results}
        dense_map = {c.chunk_id: s for c, s in dense_results}

        # Normalise each set to [0,1]
        bm25_norm = _normalise_scores(bm25_map)
        dense_norm = _normalise_scores(dense_map)

        # Fuse
        all_ids = set(bm25_map) | set(dense_map)
        fused: Dict[str, Tuple[Chunk, float]] = {}
        for cid in all_ids:
            s_bm25 = bm25_norm.get(cid, 0.0)
            s_dense = dense_norm.get(cid, 0.0)
            score = self.alpha * s_bm25 + (1 - self.alpha) * s_dense
            # the Chunk object lives in either map; pick bm25 first, then dense
            chunk = None
            for r in bm25_results:
                if r[0].chunk_id == cid:
                    chunk = r[0]
                    break
            if chunk is None:
                for r in dense_results:
                    if r[0].chunk_id == cid:
                        chunk = r[0]
                        break
            if chunk is not None:
                fused[cid] = (chunk, score)

        # Sort and truncate
        sorted_results = sorted(fused.values(), key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]


def _normalise_scores(score_map: Dict[str, float]) -> Dict[str, float]:
    if not score_map:
        return {}
    vals = list(score_map.values())
    vmin, vmax = min(vals), max(vals)
    if vmax == vmin:
        return {k: 0.5 for k in score_map}
    return {k: (v - vmin) / (vmax - vmin) for k, v in score_map.items()}
