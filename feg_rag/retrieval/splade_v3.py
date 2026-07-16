"""SPLADE-v3 standalone learned sparse retriever.

Uses naver/splade-v3 to produce sparse lexical vectors.  Document vectors are
stored as a scipy sparse CSR matrix; query vectors are sparse dot-producted
against the document matrix for retrieval.

Does NOT fuse with BM25, Dense, or Hybrid by default.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from feg_rag.data.chunker import Chunk

logger = logging.getLogger(__name__)

RETRIEVER_TYPE = "splade_v3"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class SPLADEV3Retriever:
    """SPLADE-v3 learned sparse retriever.

    Encodes documents and queries into sparse vectors using
    ``log(1 + ReLU(logits))`` activation followed by max-pooling over the
    sequence dimension.  Document vectors are stored as a scipy CSR matrix
    for memory efficiency and fast sparse dot-product retrieval.
    """

    def __init__(
        self,
        model_name: str = "naver/splade-v3",
        device: str | None = None,
        max_length: int = 512,
        batch_size: int = 8,
        hf_endpoint: str | None = None,
        revision: str = "main",
        sparsify_threshold: float = 0.0,
        normalize: bool = False,
        debug: bool = False,
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.hf_endpoint = hf_endpoint
        self.revision = revision
        self.sparsify_threshold = sparsify_threshold
        self.normalize = normalize
        self.debug = debug

        # Resolve device
        import torch
        if device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        elif device == "cuda" and not torch.cuda.is_available():
            self._device = "cpu"
        else:
            self._device = device

        self._model: Any = None
        self._tokenizer: Any = None
        self._vocab_size: int | None = None
        self._chunks: List[Chunk] = []
        self._doc_sparse_matrix: Any = None  # scipy.sparse.csr_matrix
        self._backend: str = ""

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def embedding_dim(self) -> int | None:
        """Vocabulary size (the dimension of sparse vectors)."""
        return self._vocab_size

    @property
    def retriever_type(self) -> str:
        return RETRIEVER_TYPE

    # ------------------------------------------------------------------
    # Index
    # ------------------------------------------------------------------

    def index(self, chunks: List[Chunk]) -> None:
        """Encode chunks as sparse vectors and build sparse document matrix."""
        self._chunks = chunks
        self._load_model()

        if self.debug:
            print(f"  [SPLADE-v3] Indexing {len(chunks)} chunks")
            print(f"  [SPLADE-v3] backend={self._backend}")
            print(f"  [SPLADE-v3] device={self._device}")

        t0 = time.time()
        self._doc_sparse_matrix = self._encode_documents(
            chunks, show_progress=True
        )
        elapsed = time.time() - t0
        if self.debug:
            nnz = self._doc_sparse_matrix.nnz
            shape = self._doc_sparse_matrix.shape
            density = nnz / (shape[0] * shape[1]) * 100 if shape[0] > 0 else 0
            print(f"  [SPLADE-v3] Encoding took {elapsed:.1f}s")
            print(f"  [SPLADE-v3] sparse matrix shape={shape}")
            print(f"  [SPLADE-v3] nnz={nnz} density={density:.2f}%")

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self, query: str, top_k: int = 50
    ) -> List[Tuple[Chunk, float]]:
        """Return top-k chunks ranked by sparse dot-product score."""
        if self._doc_sparse_matrix is None:
            raise RuntimeError("Index not built. Call .index() first.")
        if self._doc_sparse_matrix.shape[0] == 0:
            return []

        q_vec = self._encode_query(query)

        if q_vec.nnz == 0:
            # Query produced empty sparse vector — return empty
            return []

        # Sparse dot product: scores = doc_matrix @ query_vector.T
        scores = self._doc_sparse_matrix @ q_vec.T
        if hasattr(scores, "toarray"):
            scores = np.asarray(scores.toarray()).flatten()
        else:
            scores = np.asarray(scores).flatten()

        # Top-k indices
        n = len(scores)
        k = min(top_k, n)
        if k == 0:
            return []

        # Use argpartition for efficiency
        if k < n:
            idx = np.argpartition(-scores, k - 1)[:k]
            idx = idx[np.argsort(-scores[idx])]
        else:
            idx = np.argsort(-scores)

        results: List[Tuple[Chunk, float]] = []
        for i in idx:
            results.append((self._chunks[i], float(scores[i])))
        return results

    # ------------------------------------------------------------------
    # Cache / I/O
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Persist sparse matrix and metadata to *path*."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save metadata
        with open(path / "meta.pkl", "wb") as fh:
            pickle.dump(
                {
                    "model_name": self.model_name,
                    "retriever_type": RETRIEVER_TYPE,
                    "backend": self._backend,
                    "vocab_size": self._vocab_size,
                    "max_length": self.max_length,
                    "batch_size": self.batch_size,
                    "sparsify_threshold": self.sparsify_threshold,
                    "normalize": self.normalize,
                    "hf_endpoint": self.hf_endpoint,
                    "revision": self.revision,
                    "chunks": self._chunks,
                },
                fh,
            )

        # Save sparse matrix (scipy sparse format)
        if self._doc_sparse_matrix is not None:
            import scipy.sparse as sp
            sp.save_npz(str(path / "doc_sparse.npz"), self._doc_sparse_matrix)

    @classmethod
    def load(cls, path: str | Path, device: str | None = None) -> "SPLADEV3Retriever":
        """Load a previously saved retriever from *path*."""
        import scipy.sparse as sp

        path = Path(path)
        with open(path / "meta.pkl", "rb") as fh:
            data = pickle.load(fh)

        obj = cls(
            model_name=data["model_name"],
            max_length=data.get("max_length", 512),
            batch_size=data.get("batch_size", 8),
            hf_endpoint=data.get("hf_endpoint"),
            revision=data.get("revision", "main"),
            sparsify_threshold=data.get("sparsify_threshold", 0.0),
            normalize=data.get("normalize", False),
            device=device,
        )
        obj._backend = data.get("backend", "")
        obj._vocab_size = data.get("vocab_size")
        obj._chunks = data["chunks"]

        sparse_path = path / "doc_sparse.npz"
        if sparse_path.exists():
            obj._doc_sparse_matrix = sp.load_npz(str(sparse_path))
        return obj

    # ------------------------------------------------------------------
    # Internal: model loading
    # ------------------------------------------------------------------

    def _resolve_hf_endpoint(self) -> None:
        """Set HF_ENDPOINT environment variable if configured."""
        if self.hf_endpoint:
            os.environ["HF_ENDPOINT"] = self.hf_endpoint
            if self.debug:
                print(f"  [SPLADE-v3] HF_ENDPOINT={self.hf_endpoint}")

    def _load_model(self) -> None:
        """Load SPLADE-v3 tokenizer and model."""
        if self._model is not None:
            return

        self._resolve_hf_endpoint()

        import torch
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        try:
            if self.debug:
                print(f"  [SPLADE-v3] Loading tokenizer: {self.model_name}")
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                revision=self.revision,
            )

            if self.debug:
                print(f"  [SPLADE-v3] Loading model: {self.model_name}")
            self._model = AutoModelForMaskedLM.from_pretrained(
                self.model_name,
                revision=self.revision,
            ).to(self._device)
            self._model.eval()

            self._vocab_size = self._model.config.vocab_size
            self._backend = f"transformers-splade:{self.model_name}"

        except Exception as exc:
            raise RuntimeError(
                f"SPLADEV3Retriever: Failed to load model "
                f"'{self.model_name}'.\n"
                f"  HF_ENDPOINT={os.environ.get('HF_ENDPOINT', 'not set')}\n"
                f"  Please ensure the model is accessible.\n"
                f"  Tip: try setting HF_ENDPOINT=https://hf-mirror.com\n"
                f"  Error: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Internal: encoding
    # ------------------------------------------------------------------

    def _encode_documents(
        self, chunks: List[Chunk], show_progress: bool = False
    ):
        """Encode chunks into a scipy CSR sparse matrix."""
        import scipy.sparse as sp
        from tqdm import tqdm

        texts = [c.text[:2000] for c in chunks]
        bs = self.batch_size

        rows: List[sp.csr_matrix] = []
        batches = range(0, len(texts), bs)
        if show_progress:
            batches = tqdm(
                list(batches), desc="SPLADE-v3 doc encode", leave=False
            )

        for start in batches:
            batch = texts[start:start + bs]
            batch_sparse = self._encode_texts_to_sparse(batch)
            rows.append(batch_sparse)

        if rows:
            return sp.vstack(rows, format="csr")
        return sp.csr_matrix((0, self._vocab_size or 30522), dtype=np.float32)

    def _encode_query(self, query: str):
        """Encode a single query into a sparse vector (row CSR)."""
        import scipy.sparse as sp

        self._load_model()
        q_vec = self._encode_texts_to_sparse([query])
        # L2 normalize the query vector
        if self.normalize and q_vec.nnz > 0:
            norm = sp.linalg.norm(q_vec)
            if norm > 0:
                q_vec = q_vec / norm
        return q_vec

    def _encode_texts_to_sparse(self, texts: List[str]):
        """Core SPLADE encoding: tokenize → logits → log(1+relu) → max-pool."""
        import torch
        import scipy.sparse as sp

        inputs = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self._device)

        with torch.no_grad():
            outputs = self._model(**inputs)
            logits = outputs.logits  # (batch, seq_len, vocab_size)

        # SPLADE activation: max pooling over sequence dim
        # weights = log(1 + relu(logits))
        weights = torch.log1p(torch.relu(logits))

        # Max pooling over sequence dimension → (batch, vocab_size)
        sparse_weights, _ = torch.max(weights, dim=1)

        # Apply threshold to keep only meaningful weights
        if self.sparsify_threshold > 0:
            sparse_weights[sparse_weights < self.sparsify_threshold] = 0.0

        sparse_weights = sparse_weights.cpu().numpy().astype(np.float32)

        # Convert to sparse CSR row(s)
        rows_list: List[sp.csr_matrix] = []
        for i in range(len(texts)):
            row_vec = sparse_weights[i]
            nonzero = np.nonzero(row_vec)[0]
            if len(nonzero) > 0:
                data = row_vec[nonzero]
                row_csr = sp.csr_matrix(
                    (data, (np.zeros_like(nonzero), nonzero)),
                    shape=(1, row_vec.shape[0]),
                    dtype=np.float32,
                )
            else:
                row_csr = sp.csr_matrix(
                    (1, row_vec.shape[0]), dtype=np.float32
                )
            rows_list.append(row_csr)

        if len(rows_list) > 1:
            return sp.vstack(rows_list, format="csr")
        elif len(rows_list) == 1:
            return rows_list[0]
        else:
            return sp.csr_matrix(
                (0, self._vocab_size or 30522), dtype=np.float32
            )


# ---------------------------------------------------------------------------
# Cache metadata helper
# ---------------------------------------------------------------------------

def build_cache_metadata(
    corpus_chunks: List[Chunk],
    gold_map: Dict[str, List[str]],
    retriever: SPLADEV3Retriever,
    extra: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Build a cache metadata dict for reproducibility checks."""
    corpus_text = json.dumps(
        [
            {
                "chunk_id": c.chunk_id,
                "text": c.text,
                "company": c.company,
                "filing_year": c.filing_year,
                "section": c.section,
            }
            for c in corpus_chunks
        ],
        sort_keys=True,
        ensure_ascii=False,
    )
    gold_text = json.dumps(gold_map, sort_keys=True)

    meta: Dict[str, Any] = {
        "model_name": retriever.model_name,
        "retriever_type": RETRIEVER_TYPE,
        "corpus_hash": hashlib.sha1(corpus_text.encode()).hexdigest(),
        "gold_map_hash": hashlib.sha1(gold_text.encode()).hexdigest(),
        "max_length": retriever.max_length,
        "embedding_dim": retriever.embedding_dim,
        "normalize": retriever.normalize,
        "sparsify_threshold": retriever.sparsify_threshold,
        "hf_endpoint": retriever.hf_endpoint,
        "revision": retriever.revision,
    }
    if extra:
        meta.update(extra)
    return meta


def cache_is_valid(
    meta_path: Path,
    expected: Dict[str, Any],
    check_keys: List[str] | None = None,
) -> bool:
    """Check whether a cached index matches *expected* metadata.

    Args:
        meta_path: Path to ``cache_meta.json``.
        expected: Dict of expected values.
        check_keys: Keys to compare.  If None, compares all of:
            model_name, retriever_type, corpus_hash, gold_map_hash,
            max_length, embedding_dim, normalize, revision.

    Returns:
        True if all checked keys match.
    """
    if check_keys is None:
        check_keys = [
            "model_name", "retriever_type", "corpus_hash",
            "gold_map_hash", "max_length", "embedding_dim",
            "normalize", "revision",
        ]

    if not meta_path.exists():
        return False

    try:
        cached = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return False

    for key in check_keys:
        if cached.get(key) != expected.get(key):
            return False
    return True
