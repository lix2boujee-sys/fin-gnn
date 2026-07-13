"""Smoke test for E5MistralEncoder and DenseRetriever with E5-Mistral model.

Validates:
  - Model loads (ST primary or transformers fallback)
  - Embedding dim == 4096
  - Encode → index → search round-trip
  - Search returns non-empty top-k
  - No NaN / Inf in embeddings
  - L2 norms ≈ 1.0
  - Backend string contains expected identifiers
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from feg_rag.data.chunker import Chunk
from feg_rag.retrieval.dense import (
    DenseRetriever,
    E5MistralEncoder,
    _is_e5_mistral_instruct,
)

MODEL_PATH = "D:/fin-gnn/cache/models/e5-mistral-7b-instruct"


def make_chunks() -> list[Chunk]:
    """Create 3 synthetic Chunk objects."""
    return [
        Chunk(
            chunk_id="test-chunk-1",
            doc_id="test-doc-1",
            text=(
                "The company reported revenue of $10.2 billion for fiscal year "
                "2025, representing a 15% increase compared to the prior year. "
                "Operating income was $2.1 billion with a net profit margin of "
                "12.3%."
            ),
            chunk_type="text",
            section="Financial Highlights",
        ),
        Chunk(
            chunk_id="test-chunk-2",
            doc_id="test-doc-1",
            text=(
                "Capital expenditures totaled $3.5 billion in 2025, primarily "
                "driven by investments in data center infrastructure and cloud "
                "computing capabilities. The company expects capex to decline "
                "to $2.8 billion in 2026."
            ),
            chunk_type="text",
            section="Capital Investment",
        ),
        Chunk(
            chunk_id="test-chunk-3",
            doc_id="test-doc-2",
            text=(
                "Risk factors include foreign exchange fluctuations, regulatory "
                "changes in key markets, and potential supply chain disruptions. "
                "The company maintains a diversified supplier base across 12 "
                "countries to mitigate concentration risk."
            ),
            chunk_type="text",
            section="Risk Factors",
        ),
    ]


def test_model_name_detection():
    """Verify _is_e5_mistral_instruct detection."""
    print("\n── Test 1: Model name detection ──")
    assert _is_e5_mistral_instruct("e5-mistral-7b-instruct"), "short name"
    assert _is_e5_mistral_instruct("intfloat/e5-mistral-7b-instruct"), "hub name"
    assert _is_e5_mistral_instruct("D:/path/e5-mistral-7b-instruct"), "local path"
    assert not _is_e5_mistral_instruct("all-MiniLM-L6-v2"), "MiniLM"
    assert not _is_e5_mistral_instruct("intfloat/e5-base-v2"), "E5 base"
    print("  PASS")


def test_encoder_direct():
    """Test E5MistralEncoder directly."""
    print("\n── Test 2: E5MistralEncoder direct ──")
    encoder = E5MistralEncoder(
        model_path=MODEL_PATH,
        device="cpu",
        max_seq_length=512,
        debug=True,
    )

    # Check properties
    print(f"  backend: {encoder.backend}")
    print(f"  embedding_dim: {encoder.embedding_dim}")
    print(f"  device: {encoder.device}")

    assert encoder.embedding_dim == 4096, (
        f"Expected dim=4096, got {encoder.embedding_dim}"
    )
    assert "e5-mistral" in encoder.backend, (
        f"Backend should mention e5-mistral: {encoder.backend}"
    )
    assert encoder.backend.startswith("sentence-transformers:") or \
        encoder.backend.startswith("transformers-fallback:"), (
        f"Unexpected backend format: {encoder.backend}"
    )

    # Encode passages (raw text)
    chunks = make_chunks()
    texts = [c.text for c in chunks]
    embeddings = encoder.encode(texts, batch_size=2, show_progress_bar=False)

    # Shape checks
    assert embeddings.shape == (3, 4096), f"Shape: {embeddings.shape}"
    assert embeddings.dtype == np.float32, f"dtype: {embeddings.dtype}"

    # NaN / Inf checks
    assert not np.isnan(embeddings).any(), "NaN found"
    assert not np.isinf(embeddings).any(), "Inf found"

    # Norm checks (should be ~1.0 after L2 normalize)
    norms = np.linalg.norm(embeddings, axis=1)
    print(f"  L2 norms: {norms}")
    assert np.allclose(norms, 1.0, atol=1e-4), f"Norms not ~1.0: {norms}"

    # Encode query with instruction format
    instruction = "Given a financial question, retrieve relevant evidence passages from SEC filings that directly support the answer."
    query_text = f"Instruct: {instruction}\nQuery: What was the revenue in 2025?"
    q_emb = encoder.encode([query_text], batch_size=1)
    assert q_emb.shape == (1, 4096), f"Query shape: {q_emb.shape}"
    q_norm = float(np.linalg.norm(q_emb))
    assert 0.99 <= q_norm <= 1.01, f"Query norm: {q_norm}"

    print("  PASS")


def test_retriever_roundtrip():
    """Test DenseRetriever with E5-Mistral: index → search."""
    print("\n── Test 3: DenseRetriever round-trip ──")
    chunks = make_chunks()
    retriever = DenseRetriever(
        model_name=MODEL_PATH,
        device="cpu",
        debug=True,
    )

    # Verify model detection
    assert retriever.is_e5_mistral, "Should be detected as E5-Mistral"
    assert retriever.is_e5, "Should be detected as E5-family"

    # Index
    retriever.index(chunks, batch_size=2)

    # Check properties after indexing
    assert retriever.embedding_dim == 4096, (
        f"Retriever dim: {retriever.embedding_dim}"
    )

    # Check embeddings
    emb_dict = retriever.chunk_embeddings()
    assert len(emb_dict) == 3, f"Expected 3 embeddings, got {len(emb_dict)}"
    for cid, emb in emb_dict.items():
        assert emb.shape == (4096,), f"{cid} shape: {emb.shape}"
        assert not np.isnan(emb).any(), f"{cid}: NaN"
        norm = float(np.linalg.norm(emb))
        assert 0.99 <= norm <= 1.01, f"{cid} norm: {norm}"

    # Search
    results = retriever.search(
        "What was the company's revenue in fiscal year 2025?", top_k=3
    )
    assert len(results) > 0, "Search returned no results!"
    print(f"  Search returned {len(results)} results")
    for rank, (chunk, score) in enumerate(results, 1):
        print(f"    rank={rank} score={score:.6f} chunk_id={chunk.chunk_id}")
        print(f"      text: {chunk.text[:100]}...")

    # Verify the most relevant chunk (about revenue) ranks highest
    top_chunk = results[0][0]
    assert "revenue" in top_chunk.text.lower(), (
        f"Top result should be about revenue, got: {top_chunk.text[:100]}"
    )

    # Check backend string
    assert "e5-mistral" in retriever.backend, (
        f"Backend: {retriever.backend}"
    )

    print("  PASS")


def test_minilm_unchanged():
    """Verify MiniLM path is not broken."""
    print("\n── Test 4: MiniLM behavior unchanged ──")
    chunks = make_chunks()
    retriever = DenseRetriever(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        device="cpu",
        debug=False,
    )
    assert not retriever.is_e5_mistral, "MiniLM should not be E5-Mistral"
    assert not retriever.is_e5, "MiniLM should not be E5-family"

    # Check that _fmt_passage and _fmt_query don't add E5 prefixes
    assert retriever._fmt_passage("test text") == "test text"
    assert retriever._fmt_query("test query") == "test query"

    print("  PASS (MiniLM identity checks only — no model load)")


def main():
    print("=" * 70)
    print("E5MistralEncoder / DenseRetriever Smoke Test")
    print(f"Model: {MODEL_PATH}")
    print("=" * 70)

    test_model_name_detection()

    model_path = Path(MODEL_PATH)
    if not model_path.exists():
        print(f"\n  [SKIP] Model directory not found at {MODEL_PATH}")
        print("  Skipping encoder tests.")
        test_minilm_unchanged()
        print("\n  All available tests passed!")
        return

    # Check if model weight files actually exist (not just configs)
    has_weights = (
        list(model_path.glob("*.safetensors")) or
        list(model_path.glob("*.bin"))
    )
    if not has_weights:
        print(f"\n  [SKIP] Model config exists but weight files (*.safetensors "
              f"or *.bin) are missing.")
        print(f"  The model at {MODEL_PATH} appears to be a partial download.")
        print(f"  Run 'huggingface-cli download intfloat/e5-mistral-7b-instruct "
              f"--local-dir {MODEL_PATH}' to complete.")
        print("  Running code-path tests only (no model load).")
        test_minilm_unchanged()
        # Also verify the encoder raises a clear error (not a raw FileNotFoundError)
        print("\n── Test: E5MistralEncoder clear error on missing weights ──")
        try:
            E5MistralEncoder(model_path=str(model_path), device="cpu",
                             max_seq_length=512, debug=False)
            print("  FAIL: Should have raised RuntimeError")
            sys.exit(1)
        except RuntimeError as e:
            msg = str(e)
            assert "Failed to load model" in msg, f"Unexpected error: {msg}"
            assert "SentenceTransformer error" in msg, f"Missing ST error: {msg}"
            assert "AutoModel error" in msg, f"Missing AM error: {msg}"
            print(f"  PASS (got clear RuntimeError with both failure reasons)")
        print("\n  All available tests passed!")
        return

    test_encoder_direct()
    test_retriever_roundtrip()
    test_minilm_unchanged()

    print("\n" + "=" * 70)
    print("All tests PASSED!")
    print("=" * 70)


if __name__ == "__main__":
    main()
