"""BM25 (sparse) retrieval via rank_bm25."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from rank_bm25 import BM25Okapi

from feg_rag.data.chunker import Chunk


class BM25Retriever:
    """Sparse keyword retrieval using BM25."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._bm25: BM25Okapi | None = None
        self._chunks: List[Chunk] = []
        self._tokenized: List[List[str]] = []

    # ------------------------------------------------------------------
    # Index
    # ------------------------------------------------------------------

    def index(self, chunks: List[Chunk]) -> None:
        """Build BM25 index over *chunks*."""
        self._chunks = chunks
        self._tokenized = [_tokenize(c.text) for c in chunks]
        self._bm25 = BM25Okapi(self._tokenized, k1=self.k1, b=self.b)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self, query: str, top_k: int = 50
    ) -> List[Tuple[Chunk, float]]:
        """Return top-k chunks with BM25 scores."""
        if self._bm25 is None:
            raise RuntimeError("Index not built. Call .index() first.")
        tokenized_query = _tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self._chunks[i], float(scores[i])) for i in top_indices]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as fh:
            pickle.dump(
                {"k1": self.k1, "b": self.b, "chunks": self._chunks},
                fh,
            )

    @classmethod
    def load(cls, path: str | Path) -> "BM25Retriever":
        with open(path, "rb") as fh:
            data = pickle.load(fh)
        obj = cls(k1=data["k1"], b=data["b"])
        obj.index(data["chunks"])
        return obj


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    """Simple lowercase word tokenizer."""
    return text.lower().split()
