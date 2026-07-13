"""Retrieval methods: BM25, dense, hybrid, cross-encoder, and ColBERTv2.

Heavy backends (sentence-transformers, colbert) are imported lazily to keep
startup fast.
"""

from feg_rag.retrieval.bm25 import BM25Retriever

__all__ = [
    "BM25Retriever",
    "DenseRetriever",
    "HybridRetriever",
    "CrossEncoderReranker",
    "ColBERTv2Retriever",
]


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
    if name == "ColBERTv2Retriever":
        from feg_rag.retrieval.colbertv2 import ColBERTv2Retriever
        return ColBERTv2Retriever
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
