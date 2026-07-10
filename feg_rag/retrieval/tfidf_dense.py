"""Offline dense-like retriever using TF-IDF cosine similarity.

Used when sentence-transformers model cannot be downloaded.
Same API as DenseRetriever for drop-in replacement in Hybrid.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from feg_rag.data.chunker import Chunk


class TfidfDenseRetriever:
    """TF-IDF vector retrieval (local, no network)."""

    def __init__(self) -> None:
        self._vectorizer: TfidfVectorizer | None = None
        self._matrix = None
        self._chunks: List[Chunk] = []

    def index(self, chunks: List[Chunk]) -> None:
        self._chunks = chunks
        texts = [c.text[:3000] for c in chunks]
        self._vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words="english",
            ngram_range=(1, 2),
        )
        self._matrix = self._vectorizer.fit_transform(texts)

    def search(self, query: str, top_k: int = 50) -> List[Tuple[Chunk, float]]:
        if self._vectorizer is None or self._matrix is None:
            raise RuntimeError("Index not built. Call index() first.")
        q_vec = self._vectorizer.transform([query[:3000]])
        scores = linear_kernel(q_vec, self._matrix).ravel()
        if top_k >= len(scores):
            top_idx = np.argsort(scores)[::-1]
        else:
            top_idx = np.argpartition(scores, -top_k)[-top_k:]
            top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
        return [(self._chunks[i], float(scores[i])) for i in top_idx if scores[i] > 0]
