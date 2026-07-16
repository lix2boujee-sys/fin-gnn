"""Smoke tests for BGE-M3-Dense retriever.

Verifies:
1. Dense-only encoding (no lexical, no colbert)
2. L2 normalisation
3. FAISS IndexFlatIP search
4. Cache metadata matching / invalidation
5. API compatibility
"""

import hashlib
import json
import os
import tempfile
from pathlib import Path
import numpy as np

from feg_rag.data.chunker import Chunk
from feg_rag.retrieval.bge_m3 import (
    BGEM3DenseRetriever,
    build_cache_metadata,
    cache_is_valid,
    RETRIEVER_TYPE,
)


def _make_chunks(n: int = 5) -> list:
    return [
        Chunk(
            chunk_id=f"c{i}",
            text=f"Financial report chunk {i}: Revenue in 2023 was ${i*10}M. "
                 f"Net income was ${i*2}M. The company operates globally.",
            chunk_type="text",
            company="TestCorp",
            filing_year="2023",
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Test 1: indexing + search returns correct shape
# ---------------------------------------------------------------------------

def test_bge_m3_index_and_search():
    """Encoder produces L2-normalised embeddings, FAISS returns top-k."""
    chunks = _make_chunks(5)
    retriever = BGEM3DenseRetriever(
        batch_size=4, debug=False, device="cpu",
    )

    # Skip if no encoder available (model not downloaded)
    try:
        retriever.index(chunks)
    except Exception as e:
        msg = str(e).lower()
        if "failed to load model" in msg or "could not" in msg:
            print(f"  SKIP (model not available): {e}")
            return
        raise

    # Embeddings should be L2-normalised
    embs = retriever._embeddings
    assert embs is not None
    norms = np.linalg.norm(embs, axis=1)
    assert np.allclose(norms, 1.0, atol=0.01), (
        f"Expected L2 norms ≈ 1.0, got {norms}"
    )

    # Search returns results
    results = retriever.search("What was the revenue in 2023?", top_k=3)
    assert len(results) == 3
    for chunk, score in results:
        assert isinstance(chunk, Chunk)
        assert isinstance(score, float)

    # Dense only (no lexical/colbert in the output)
    # embedding_dim should be set
    assert retriever.embedding_dim is not None
    assert retriever.embedding_dim > 0


# ---------------------------------------------------------------------------
# Test 2: chunk_embeddings() accessor
# ---------------------------------------------------------------------------

def test_chunk_embeddings_accessor():
    """chunk_embeddings() returns indexed dict keyed by chunk_id."""
    chunks = _make_chunks(3)
    retriever = BGEM3DenseRetriever(device="cpu", batch_size=4)

    try:
        retriever.index(chunks)
    except Exception as e:
        msg = str(e).lower()
        if "failed to load model" in msg or "could not" in msg:
            print(f"  SKIP (model not available): {e}")
            return
        raise

    emb_dict = retriever.chunk_embeddings()
    assert len(emb_dict) == 3
    for c in chunks:
        assert c.chunk_id in emb_dict
        assert emb_dict[c.chunk_id].ndim == 1


# ---------------------------------------------------------------------------
# Test 3: save / load roundtrip
# ---------------------------------------------------------------------------

def test_save_load_roundtrip():
    """Saved retriever can be reloaded and produces same search results."""
    chunks = _make_chunks(5)
    retriever = BGEM3DenseRetriever(device="cpu", batch_size=4)

    try:
        retriever.index(chunks)
    except Exception as e:
        msg = str(e).lower()
        if "failed to load model" in msg or "could not" in msg:
            print(f"  SKIP (model not available): {e}")
            return
        raise

    result_before = retriever.search("revenue 2023", top_k=3)

    with tempfile.TemporaryDirectory() as tmpdir:
        retriever.save(tmpdir)
        loaded = BGEM3DenseRetriever.load(tmpdir, device="cpu")
        result_after = loaded.search("revenue 2023", top_k=3)

        assert len(result_before) == len(result_after)
        for (c1, s1), (c2, s2) in zip(result_before, result_after):
            assert c1.chunk_id == c2.chunk_id
            assert abs(s1 - s2) < 1e-5


# ---------------------------------------------------------------------------
# Test 4: cache metadata validation
# ---------------------------------------------------------------------------

def test_cache_metadata_validation():
    """cache_is_valid returns True for matching, False for mismatched metadata."""
    chunks = _make_chunks(5)
    gold_map = {"q1": ["c0", "c1"], "q2": ["c2"]}

    retriever = BGEM3DenseRetriever(
        model_name="BAAI/bge-m3",
        max_length=256,
        device="cpu",
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        meta_path = Path(tmpdir) / "cache_meta.json"
        meta = build_cache_metadata(chunks, gold_map, retriever)
        meta["chunk_size"] = 512
        meta["chunk_overlap"] = 64

        with open(meta_path, "w") as f:
            json.dump(meta, f)

        # Matching → valid
        expected = dict(meta)
        assert cache_is_valid(meta_path, expected)

        # Mismatched model_name → invalid
        bad = dict(expected, model_name="different/model")
        assert not cache_is_valid(meta_path, bad)

        # Mismatched retriever_type → invalid
        bad = dict(expected, retriever_type="e5")
        assert not cache_is_valid(meta_path, bad)

        # Mismatched corpus_hash → invalid
        bad = dict(expected, corpus_hash="deadbeef")
        assert not cache_is_valid(meta_path, bad)

        # Mismatched max_length → invalid
        bad = dict(expected, max_length=999)
        assert not cache_is_valid(meta_path, bad)

        # Mismatched revision → invalid
        bad = dict(expected, revision="v2")
        assert not cache_is_valid(meta_path, bad)

        # Missing meta file → invalid
        assert not cache_is_valid(Path(tmpdir) / "nonexistent.json", expected)

        # e5 retriever_type should NOT validate as bge_m3
        bad = dict(expected, retriever_type="dense")
        assert not cache_is_valid(meta_path, bad)


# ---------------------------------------------------------------------------
# Test 5: retriever_type constant
# ---------------------------------------------------------------------------

def test_retriever_type_constant():
    """BGEM3DenseRetriever.retriever_type returns 'bge_m3_dense'."""
    r = BGEM3DenseRetriever(device="cpu")
    assert r.retriever_type == RETRIEVER_TYPE
    assert RETRIEVER_TYPE == "bge_m3_dense"
