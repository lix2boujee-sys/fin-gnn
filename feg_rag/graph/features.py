"""Node feature construction for the financial evidence graph.

Produces a feature matrix that can be fed into GNN rerankers.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from feg_rag.data.chunker import Chunk
from feg_rag.graph.builder import FinancialEvidenceGraph
from feg_rag.graph.entities import ExtractedEntities


def _compute_chunk_embeddings(
    chunks: List[Chunk],
    model_name: str = "tfidf",
    device: str = "cpu",
    dim: int = 256,
) -> Dict[str, np.ndarray]:
    """Compute text embeddings for chunks.

    Supports two backends:
    - ``"tfidf"`` (default): sklearn TfidfVectorizer + TruncatedSVD (no download).
    - Any sentence-transformers model name: neural embeddings.
    """
    if model_name == "tfidf":
        return _compute_tfidf_embeddings(chunks, dim=dim)
    else:
        return _compute_sbert_embeddings(chunks, model_name=model_name, device=device)


def _compute_tfidf_embeddings(
    chunks: List[Chunk],
    dim: int = 256,
) -> Dict[str, np.ndarray]:
    """Compute TF-IDF + SVD embeddings (pure local, no network required)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD

    texts = [c.text[:3000] for c in chunks]
    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words="english",
        ngram_range=(1, 2),
    )
    tfidf = vectorizer.fit_transform(texts)
    n_components = min(dim, tfidf.shape[1] - 1, tfidf.shape[0] - 1)
    if n_components > 5:
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        reduced = svd.fit_transform(tfidf)
    else:
        reduced = tfidf.toarray().astype(np.float32)
        n_components = reduced.shape[1]
    # Normalize to unit length
    norms = np.linalg.norm(reduced, axis=1, keepdims=True) + 1e-8
    reduced = reduced / norms
    actual_dim = reduced.shape[1]
    # Pad to dim if needed
    if actual_dim < dim:
        padded = np.zeros((len(chunks), dim), dtype=np.float32)
        padded[:, :actual_dim] = reduced.astype(np.float32)
        reduced = padded
    return {c.chunk_id: reduced[i].astype(np.float32) for i, c in enumerate(chunks)}


def _compute_sbert_embeddings(
    chunks: List[Chunk],
    model_name: str,
    device: str = "cpu",
) -> Dict[str, np.ndarray]:
    """Compute neural embeddings via sentence-transformers (needs network)."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device)
    texts = [c.text[:2000] for c in chunks]
    embeddings = model.encode(
        texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True
    )
    return {c.chunk_id: emb.astype(np.float32) for c, emb in zip(chunks, embeddings)}


def build_node_features(
    graph: FinancialEvidenceGraph,
    chunks: List[Chunk],
    entity_map: Dict[str, ExtractedEntities],
    retrieval_scores: Optional[Dict[str, float]] = None,
    embedding_dim: int = 256,
    chunk_embeddings: Optional[Dict[str, np.ndarray]] = None,
    compute_embeddings: bool = True,
    embedding_model: str = "tfidf",
    embedding_device: str = "cpu",
) -> Dict[str, np.ndarray]:
    """Build a feature dictionary {node_id: feature_vector}.

    Feature components (paper plan S8.3):
        - text embedding (computed via sentence-transformers)
        - BM25 retrieval score
        - node type one-hot (chunk / metric / year)
        - query match flags (metric, year)
        - graph centrality (log1p degree)

    Args:
        graph: The financial evidence graph.
        chunks: All chunks.
        entity_map: Extracted entities per chunk.
        retrieval_scores: Optional dict of chunk_id -> retrieval score.
        embedding_dim: Fallback dim if no embeddings provided/computed.
        chunk_embeddings: Pre-computed embeddings dict.
        compute_embeddings: If True and chunk_embeddings is None, compute on the fly.
        embedding_model: Model for computing embeddings.
        embedding_device: Device for embedding computation.

    Returns:
        Dict mapping node_id -> numpy feature vector.
    """
    # --- Resolve embeddings ---
    embedding_source = "none"
    if chunk_embeddings is not None:
        if chunk_embeddings:
            embedding_dim = next(iter(chunk_embeddings.values())).shape[0]
            embedding_source = "precomputed"
            print(f"  Using pre-computed chunk embeddings: dim={embedding_dim}")
    elif compute_embeddings and chunks:
        print(f"  Computing text embeddings for {len(chunks)} chunks "
              f"(model={embedding_model})...")
        chunk_embeddings = _compute_chunk_embeddings(
            chunks, model_name=embedding_model, device=embedding_device
        )
        if chunk_embeddings:
            embedding_dim = next(iter(chunk_embeddings.values())).shape[0]
            embedding_source = embedding_model
            print(f"  Computed embeddings: model={embedding_model}, dim={embedding_dim}")

    # Node type one-hot: chunk, metric, year
    type_to_idx = {"chunk": 0, "metric": 1, "year": 2}
    num_types = len(type_to_idx)

    # Total dim: embedding_dim + 1 (retrieval score) + num_types + 2 (match flags) + 1 (degree)
    feat_dim = embedding_dim + 1 + num_types + 2 + 1

    print(f"  Feature dim: {feat_dim}  "
          f"(embedding={embedding_dim} + retrieval_score=1 + node_types={num_types} "
          f"+ match_flags=2 + degree=1)")
    print(f"  Embedding source: {embedding_source}")

    # Warn if the expected neural dim (384 → 391) doesn't match
    _EXPECTED_NEURAL_DIM = 391
    if embedding_source not in ("none", "tfidf") and feat_dim != _EXPECTED_NEURAL_DIM:
        print(f"  ⚠ WARNING: Feature dim is {feat_dim}, expected {_EXPECTED_NEURAL_DIM} "
              f"for local MiniLM (384-dim). Check your embedding backend!")

    # Pre-compute degree
    degrees = dict(graph.graph.degree())

    features: Dict[str, np.ndarray] = {}

    for node_id in graph.graph.nodes():
        ntype = graph.node_types.get(node_id, "chunk")
        vec = np.zeros(feat_dim, dtype=np.float32)

        offset = 0

        # text embedding
        if chunk_embeddings and node_id in chunk_embeddings:
            emb = chunk_embeddings[node_id]
            vec[offset:offset + len(emb)] = emb
        offset += embedding_dim

        # retrieval score
        if retrieval_scores and node_id in retrieval_scores:
            vec[offset] = retrieval_scores[node_id]
        offset += 1

        # node type one-hot
        t_idx = type_to_idx.get(ntype, 0)
        vec[offset + t_idx] = 1.0
        offset += num_types

        # query match flags (set externally during reranking; 0 for now)
        offset += 2

        # degree (log1p normalised)
        vec[offset] = np.log1p(degrees.get(node_id, 0))

        features[node_id] = vec

    return features
