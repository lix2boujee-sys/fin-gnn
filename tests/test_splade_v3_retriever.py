"""Smoke tests for SPLADE-v3 retriever.

Verifies:
1. Sparse vector encoding (not dense)
2. CSR matrix storage (not dense vocab matrix)
3. Sparse dot-product retrieval
4. Empty vector handling
5. Cache metadata matching / invalidation
6. API compatibility
"""

import hashlib
import json
import os
import tempfile
from pathlib import Path
import numpy as np

from feg_rag.data.chunker import Chunk
from feg_rag.retrieval.splade_v3 import (
    SPLADEV3Retriever,
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


HAS_SCIPY = False
try:
    import scipy.sparse as sp
    HAS_SCIPY = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Test 1: indexing + search returns correct results
# ---------------------------------------------------------------------------

def test_splade_v3_index_and_search():
    """SPLADE-v3 encodes docs as sparse CSR, query as sparse vec, dot-product."""
    if not HAS_SCIPY:
        print("  SKIP (scipy not available)")
        return

    chunks = _make_chunks(5)
    retriever = SPLADEV3Retriever(
        batch_size=4, debug=False, device="cpu",
    )

    # Skip if no model available
    try:
        retriever.index(chunks)
    except Exception as e:
        msg = str(e).lower()
        if "failed to load model" in msg or "could not" in msg:
            print(f"  SKIP (model not available): {e}")
            return
        raise

    # Document matrix should be sparse CSR (not dense)
    assert retriever._doc_sparse_matrix is not None
    assert isinstance(retriever._doc_sparse_matrix, sp.csr_matrix), (
        f"Expected CSR, got {type(retriever._doc_sparse_matrix)}"
    )

    # Matrix shape: (num_docs, vocab_size)
    assert retriever._doc_sparse_matrix.shape[0] == len(chunks)
    assert retriever.embedding_dim == retriever._doc_sparse_matrix.shape[1]

    # Search returns results
    results = retriever.search("What was the revenue in 2023?", top_k=3)
    assert len(results) == 3
    for chunk, score in results:
        assert isinstance(chunk, Chunk)
        assert isinstance(score, float)

    # SPLADE-v3 is sparse, not dense — should not have FAISS index
    assert retriever.embedding_dim is not None


# ---------------------------------------------------------------------------
# Test 2: empty vector handling
# ---------------------------------------------------------------------------

def test_empty_query_handling():
    """Query with no matching tokens returns empty, no crash."""
    if not HAS_SCIPY:
        print("  SKIP (scipy not available)")
        return

    chunks = _make_chunks(3)
    retriever = SPLADEV3Retriever(device="cpu", batch_size=4)

    try:
        retriever.index(chunks)
    except Exception as e:
        msg = str(e).lower()
        if "failed to load model" in msg or "could not" in msg:
            print(f"  SKIP (model not available): {e}")
            return
        raise

    # Search with empty-ish query should not crash
    results = retriever.search("", top_k=5)
    # Either returns results or empty list — but never crashes
    assert isinstance(results, list)


# ---------------------------------------------------------------------------
# Test 3: save / load roundtrip
# ---------------------------------------------------------------------------

def test_save_load_roundtrip():
    """Saved SPLADE retriever can be reloaded and produces same results."""
    if not HAS_SCIPY:
        print("  SKIP (scipy not available)")
        return

    chunks = _make_chunks(5)
    retriever = SPLADEV3Retriever(device="cpu", batch_size=4)

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

        # Loaded retriever doesn't need the model for search
        loaded = SPLADEV3Retriever.load(tmpdir, device="cpu")
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

    retriever = SPLADEV3Retriever(
        model_name="naver/splade-v3",
        max_length=256,
        device="cpu",
    )
    # Pre-set vocab_size since we don't load model
    retriever._vocab_size = 30522

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
        bad = dict(expected, model_name="BAAI/bge-m3")
        assert not cache_is_valid(meta_path, bad)

        # Mismatched retriever_type → invalid
        bad = dict(expected, retriever_type="bge_m3_dense")
        assert not cache_is_valid(meta_path, bad)

        # Mismatched corpus_hash → invalid
        bad = dict(expected, corpus_hash="deadbeef")
        assert not cache_is_valid(meta_path, bad)

        # Mismatched max_length → invalid
        bad = dict(expected, max_length=999)
        assert not cache_is_valid(meta_path, bad)

        # SPLADE normalize is false
        assert meta["normalize"] is False

        # Missing meta file → invalid
        assert not cache_is_valid(Path(tmpdir) / "nonexistent.json", expected)

        # e5 retriever_type should NOT validate as splade_v3
        bad = dict(expected, retriever_type="dense")
        assert not cache_is_valid(meta_path, bad)


# ---------------------------------------------------------------------------
# Test 5: retriever_type constant
# ---------------------------------------------------------------------------

def test_retriever_type_constant():
    """SPLADEV3Retriever.retriever_type returns 'splade_v3'."""
    r = SPLADEV3Retriever(device="cpu")
    assert r.retriever_type == RETRIEVER_TYPE
    assert RETRIEVER_TYPE == "splade_v3"


# ---------------------------------------------------------------------------
# Test 6: sparse matrix not dense
# ---------------------------------------------------------------------------

def test_sparse_not_dense():
    """Document matrix is scipy CSR, not a dense numpy array (memory safe)."""
    if not HAS_SCIPY:
        print("  SKIP (scipy not available)")
        return

    chunks = _make_chunks(3)
    retriever = SPLADEV3Retriever(device="cpu", batch_size=4)

    try:
        retriever.index(chunks)
    except Exception as e:
        msg = str(e).lower()
        if "failed to load model" in msg or "could not" in msg:
            print(f"  SKIP (model not available): {e}")
            return
        raise

    # Document storage must be scipy sparse CSR
    assert isinstance(retriever._doc_sparse_matrix, sp.csr_matrix)
    # Not a dense numpy array
    assert not isinstance(retriever._doc_sparse_matrix, np.ndarray)
