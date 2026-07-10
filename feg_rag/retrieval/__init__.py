"""Retrieval methods: BM25, dense, hybrid, and cross-encoder.

Heavy backends (sentence-transformers) are imported lazily to keep startup fast.
"""

from feg_rag.retrieval.bm25 import BM25Retriever

__all__ = ["BM25Retriever", "DenseRetriever", "HybridRetriever", "CrossEncoderReranker"]


def __getattr__(name: str):
    if name == "DenseRetriever":
        from feg_rag.retrieval.dense import DenseRetriever
        return DenseRetriever
    if name == "HybridRetriever":
        from feg_rag.retrieval.hybrid import HybridRetriever
        return HybridRetriever
    if name == "CrossEncoderReranker":
        from feg_rag.retrieval.cross_encoder import CrossEncoderReranker
        return CrossEncoderReranker
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
