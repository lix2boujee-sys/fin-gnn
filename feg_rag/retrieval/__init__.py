"""Retrieval methods: BM25, dense, hybrid, cross-encoder, ColBERTv2,
BGE-M3-Dense, and SPLADE-v3.

Heavy backends (sentence-transformers, colbert, FlagEmbedding) are imported
lazily to keep startup fast.
"""

from feg_rag.retrieval.bm25 import BM25Retriever

__all__ = [
    "BM25Retriever",
    "DenseRetriever",
    "HybridRetriever",
    "CrossEncoderReranker",
    "ColBERTv2Retriever",
    "BGEM3DenseRetriever",
    "SPLADEV3Retriever",
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
    if name == "BGEM3DenseRetriever":
        from feg_rag.retrieval.bge_m3 import BGEM3DenseRetriever
        return BGEM3DenseRetriever
    if name == "SPLADEV3Retriever":
        from feg_rag.retrieval.splade_v3 import SPLADEV3Retriever
        return SPLADEV3Retriever
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
