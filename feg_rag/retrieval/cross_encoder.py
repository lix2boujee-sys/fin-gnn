"""Cross-Encoder evidence reranker.

Paper design §7.1: Strong non-graph reranker baseline.
Re-ranks initial retrieval candidates using a cross-encoder model that scores
(query, chunk) pairs jointly.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from feg_rag.data.chunker import Chunk


class CrossEncoderReranker:
    """Rerank candidate chunks with a cross-encoder model.

    Uses sentence-transformers CrossEncoder (e.g. ms-marco-MiniLM-L-6-v2)
    for pairwise relevance scoring.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        batch_size: int = 32,
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None
        self._device = device

    # ------------------------------------------------------------------
    # Rerank
    # ------------------------------------------------------------------

    def rerank(
        self,
        query: str,
        candidate_chunks: List[Tuple[Chunk, float]],
        top_k: int = 50,
    ) -> List[Tuple[Chunk, float]]:
        """Score and rerank candidate chunks against the query.

        Args:
            query: The question text.
            candidate_chunks: List of (Chunk, retrieval_score) from initial retrieval.
            top_k: Number of top results to return.

        Returns:
            Reranked list of (Chunk, cross_encoder_score).
        """
        if not candidate_chunks:
            return []

        model = self._get_model()
        pairs = [(query, chunk.text) for chunk, _ in candidate_chunks]
        scores = model.predict(pairs, batch_size=self.batch_size,
                               show_progress_bar=False)

        # Pair scores with chunks
        scored = [
            (chunk, float(score))
            for (chunk, _), score in zip(candidate_chunks, scores)
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            kwargs = {}
            if self._device:
                kwargs["device"] = self._device
            self._model = CrossEncoder(self.model_name, **kwargs)
        return self._model
