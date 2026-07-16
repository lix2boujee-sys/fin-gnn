"""BGE-M3-Dense standalone retriever.

Uses BAAI/bge-m3 dense embeddings only (no lexical weights, no colbert vectors,
no hybrid scoring).  L2-normalised embeddings are indexed with FAISS IndexFlatIP
for cosine-equivalent inner-product search.

Supports HuggingFace mirror via ``HF_ENDPOINT`` environment variable or an
explicit ``hf_endpoint`` parameter.
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

import faiss
import numpy as np

from feg_rag.data.chunker import Chunk

logger = logging.getLogger(__name__)

RETRIEVER_TYPE = "bge_m3_dense"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class BGEM3DenseRetriever:
    """BGE-M3 dense-only retriever.

    Encodes documents and queries with BAAI/bge-m3, extracts **only** the
    ``dense_vecs``, L2-normalises them, and indexes with FAISS IndexFlatIP
    (inner product = cosine similarity for normalised vectors).
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: str | None = None,
        max_length: int = 512,
        batch_size: int = 16,
        hf_endpoint: str | None = None,
        revision: str = "main",
        normalize: bool = True,
        debug: bool = False,
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.hf_endpoint = hf_endpoint
        self.revision = revision
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

        self._encoder: Any = None
        self._embedding_dim: int | None = None
        self._index: faiss.IndexFlatIP | None = None
        self._chunks: List[Chunk] = []
        self._embeddings: np.ndarray | None = None
        self._backend: str = ""

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def embedding_dim(self) -> int | None:
        return self._embedding_dim

    @property
    def retriever_type(self) -> str:
        return RETRIEVER_TYPE

    # ------------------------------------------------------------------
    # Index
    # ------------------------------------------------------------------

    def index(self, chunks: List[Chunk]) -> None:
        """Encode chunks and build FAISS IndexFlatIP."""
        self._chunks = chunks
        encoder = self._get_encoder()

        if self.debug:
            print(f"  [BGEM3Dense] Indexing {len(chunks)} chunks")
            print(f"  [BGEM3Dense] backend={self._backend}")
            print(f"  [BGEM3Dense] device={self._device}")

        t0 = time.time()
        embeddings = self._encode_documents(
            encoder, chunks, show_progress=True
        )
        elapsed = time.time() - t0
        if self.debug:
            print(f"  [BGEM3Dense] Encoding took {elapsed:.1f}s")

        dim = embeddings.shape[1]
        self._embedding_dim = dim
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(embeddings.astype(np.float32))
        self._embeddings = embeddings.astype(np.float32)

        if self.debug:
            norms = np.linalg.norm(self._embeddings, axis=1)
            print(f"  [BGEM3Dense] embeddings shape={self._embeddings.shape}")
            print(f"  [BGEM3Dense] mean L2 norm={norms.mean():.6f}")
            print(f"  [BGEM3Dense] FAISS ntotal={self._index.ntotal}")

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self, query: str, top_k: int = 50
    ) -> List[Tuple[Chunk, float]]:
        """Return top-k chunks ranked by inner product (cosine similarity)."""
        if self._index is None:
            raise RuntimeError("Index not built. Call .index() first.")

        encoder = self._get_encoder()
        q_emb = self._encode_query(encoder, query)

        if q_emb.ndim == 1:
            q_emb = q_emb.reshape(1, -1)

        scores, indices = self._index.search(
            q_emb.astype(np.float32), top_k
        )

        results: List[Tuple[Chunk, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if 0 <= idx < len(self._chunks):
                results.append((self._chunks[idx], float(score)))
        return results

    # ------------------------------------------------------------------
    # Embeddings accessor
    # ------------------------------------------------------------------

    def chunk_embeddings(self) -> Dict[str, np.ndarray]:
        """Return indexed chunk embeddings keyed by chunk_id."""
        if self._embeddings is None:
            raise RuntimeError("Index not built. Call .index() first.")
        return {
            c.chunk_id: self._embeddings[i]
            for i, c in enumerate(self._chunks)
        }

    # ------------------------------------------------------------------
    # Cache / I/O
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Persist FAISS index, embeddings, and metadata to *path*."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        with open(path / "meta.pkl", "wb") as fh:
            pickle.dump(
                {
                    "model_name": self.model_name,
                    "retriever_type": RETRIEVER_TYPE,
                    "backend": self._backend,
                    "embedding_dim": self._embedding_dim,
                    "max_length": self.max_length,
                    "normalize": self.normalize,
                    "hf_endpoint": self.hf_endpoint,
                    "revision": self.revision,
                    "chunks": self._chunks,
                },
                fh,
            )

        if self._index is not None:
            faiss.write_index(self._index, str(path / "index.faiss"))
        if self._embeddings is not None:
            np.save(str(path / "embeddings.npy"), self._embeddings)

    @classmethod
    def load(cls, path: str | Path, device: str | None = None) -> "BGEM3DenseRetriever":
        """Load a previously saved retriever from *path*."""
        path = Path(path)
        with open(path / "meta.pkl", "rb") as fh:
            data = pickle.load(fh)

        obj = cls(
            model_name=data["model_name"],
            max_length=data.get("max_length", 512),
            hf_endpoint=data.get("hf_endpoint"),
            revision=data.get("revision", "main"),
            normalize=data.get("normalize", True),
            device=device,
        )
        obj._backend = data.get("backend", "")
        obj._embedding_dim = data.get("embedding_dim")
        obj._chunks = data["chunks"]

        idx_path = path / "index.faiss"
        if idx_path.exists():
            obj._index = faiss.read_index(str(idx_path))
        emb_path = path / "embeddings.npy"
        if emb_path.exists():
            obj._embeddings = np.load(str(emb_path))
        return obj

    # ------------------------------------------------------------------
    # Internal: encoder
    # ------------------------------------------------------------------

    def _resolve_hf_endpoint(self) -> None:
        """Set HF_ENDPOINT environment variable if configured."""
        if self.hf_endpoint:
            os.environ["HF_ENDPOINT"] = self.hf_endpoint
            if self.debug:
                print(f"  [BGEM3Dense] HF_ENDPOINT={self.hf_endpoint}")

    def _get_encoder(self):
        """Return (and cache) the BGE-M3 encoder.

        Tries FlagEmbedding first (most compatible with BGE-M3),
        falls back to sentence-transformers, then AutoModel.
        """
        if self._encoder is not None:
            return self._encoder

        self._resolve_hf_endpoint()

        # ── Path 1: FlagEmbedding ──────────────────────────────────
        try:
            from FlagEmbedding import BGEM3FlagModel

            if self.debug:
                print(f"  [BGEM3Dense] Loading via FlagEmbedding: "
                      f"{self.model_name}")

            self._encoder = BGEM3FlagModel(
                self.model_name,
                use_fp16=(self._device == "cuda"),
                device=self._device,
            )
            self._backend = f"FlagEmbedding:{self.model_name}"
            # Determine dim from a test encoding
            test_vecs = self._encoder.encode(
                ["test"], batch_size=1, max_length=8
            )
            self._embedding_dim = test_vecs["dense_vecs"].shape[1]
            return self._encoder

        except ImportError:
            if self.debug:
                print(f"  [BGEM3Dense] FlagEmbedding not available, "
                      f"trying sentence-transformers")
        except Exception as exc:
            if self.debug:
                print(f"  [BGEM3Dense] FlagEmbedding failed: {exc}")

        # ── Path 2: sentence-transformers ───────────────────────────
        try:
            from sentence_transformers import SentenceTransformer

            if self.debug:
                print(f"  [BGEM3Dense] Loading via sentence-transformers: "
                      f"{self.model_name}")

            self._encoder = SentenceTransformer(
                self.model_name,
                device=self._device,
                trust_remote_code=True,
                revision=self.revision,
            )
            self._encoder.max_seq_length = self.max_length
            self._backend = f"sentence-transformers:{self.model_name}"
            # get_sentence_embedding_dimension → get_embedding_dimension (newer API)
            if hasattr(self._encoder, "get_embedding_dimension"):
                self._embedding_dim = (
                    self._encoder.get_embedding_dimension()
                )
            else:
                self._embedding_dim = (
                    self._encoder.get_sentence_embedding_dimension()
                )
            return self._encoder

        except Exception as exc_st:
            if self.debug:
                print(f"  [BGEM3Dense] sentence-transformers failed: {exc_st}")

        # ── Path 3: transformers AutoModel (mean pooling) ───────────
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer

            if self.debug:
                print(f"  [BGEM3Dense] Loading via transformers: "
                      f"{self.model_name}")

            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, revision=self.revision
            )
            model = AutoModel.from_pretrained(
                self.model_name, revision=self.revision
            ).to(self._device)
            model.eval()

            class _TransformersWrapper:
                def __init__(self, model, tokenizer, max_length, device, debug=False):
                    self.model = model
                    self.tokenizer = tokenizer
                    self.max_length = max_length
                    self.device = device
                    self.debug = debug

                def encode(self, texts, batch_size=16, max_length=None,
                          show_progress_bar=False):
                    ml = max_length or self.max_length
                    all_embs = []
                    for start in range(0, len(texts), batch_size):
                        batch = texts[start:start + batch_size]
                        inputs = self.tokenizer(
                            batch, padding=True, truncation=True,
                            max_length=ml, return_tensors="pt",
                        ).to(self.device)
                        with torch.no_grad():
                            outputs = self.model(**inputs)
                            emb = outputs.last_hidden_state
                            mask = (inputs["attention_mask"]
                                    .unsqueeze(-1).expand(emb.size()).float())
                            summed = torch.sum(emb * mask, dim=1)
                            counts = torch.clamp(mask.sum(dim=1), min=1e-9)
                            emb = summed / counts
                            emb = torch.nn.functional.normalize(
                                emb, p=2, dim=1)
                            all_embs.append(emb.cpu().numpy())
                    return np.vstack(all_embs) if all_embs else np.zeros((0, model.config.hidden_size), dtype=np.float32)

            self._encoder = _TransformersWrapper(
                model, tokenizer, self.max_length, self._device, debug=self.debug
            )
            self._backend = f"transformers-mean-pool:{self.model_name}"
            self._embedding_dim = model.config.hidden_size
            return self._encoder

        except Exception as exc_tf:
            raise RuntimeError(
                f"BGEM3DenseRetriever: Failed to load model "
                f"'{self.model_name}'.\n"
                f"  HF_ENDPOINT={os.environ.get('HF_ENDPOINT', 'not set')}\n"
                f"  Please ensure the model is accessible.\n"
                f"  Tip: try setting HF_ENDPOINT=https://hf-mirror.com\n"
                f"  Underlying errors:\n"
                f"    FlagEmbedding: not available or failed\n"
                f"    sentence-transformers: {exc_st if 'exc_st' in dir() else 'not tried'}\n"
                f"    transformers: {exc_tf}"
            ) from exc_tf

    # ------------------------------------------------------------------
    # Internal: encoding helpers
    # ------------------------------------------------------------------

    def _encode_documents(
        self, encoder, chunks: List[Chunk], show_progress: bool = False
    ) -> np.ndarray:
        """Encode chunk texts, returning L2-normalised dense embeddings."""
        texts = [c.text[:3000] for c in chunks]
        return self._encode_batched(encoder, texts, show_progress)

    def _encode_query(self, encoder, query: str) -> np.ndarray:
        """Encode a single query, returning L2-normalised dense embedding."""
        return self._encode_batched(encoder, [query], show_progress=False)

    def _encode_batched(
        self, encoder, texts: List[str], show_progress: bool
    ) -> np.ndarray:
        """Encode texts through the encoder, extracting dense_vecs only.

        Handles three encoder backends transparently:
        - FlagEmbedding BGEM3FlagModel: dict output with dense_vecs key
        - sentence-transformers: numpy array output, max_seq_length via model attr
        - transformers fallback: numpy array, max_length via tokenizer
        """
        import torch
        from tqdm import tqdm

        bs = self.batch_size
        all_embs: List[np.ndarray] = []

        batches = range(0, len(texts), bs)
        if show_progress:
            batches = tqdm(
                list(batches), desc="BGE-M3 dense encode", leave=False
            )

        for start in batches:
            batch = texts[start:start + bs]

            # Build keyword args based on backend
            encode_kwargs: Dict[str, Any] = dict(
                batch_size=bs,
                show_progress_bar=False,
            )
            # FlagEmbedding accepts max_length, sentence-transformers uses
            # model.max_seq_length (already set), transformers uses tokenizer.
            if self._backend.startswith("FlagEmbedding"):
                encode_kwargs["max_length"] = self.max_length

            output = encoder.encode(batch, **encode_kwargs)

            # Extract dense_vecs only — ignore lexical_weights and colbert_vecs
            if isinstance(output, dict):
                emb = np.asarray(output["dense_vecs"], dtype=np.float32)
            elif isinstance(output, np.ndarray):
                emb = np.asarray(output, dtype=np.float32)
            elif isinstance(output, torch.Tensor):
                emb = output.cpu().numpy().astype(np.float32)
            else:
                raise TypeError(
                    f"Unexpected encoder output type: {type(output)}"
                )

            if self.normalize:
                emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)

            all_embs.append(emb)

        if not all_embs:
            dim = self._embedding_dim or 1024
            return np.zeros((0, dim), dtype=np.float32)

        result = np.vstack(all_embs)

        # Sanity checks
        if self.debug:
            norms = np.linalg.norm(result, axis=1)
            print(f"  [BGEM3Dense] Encoded shape={result.shape}")
            print(f"  [BGEM3Dense] Mean L2 norm={norms.mean():.6f}")
            print(f"  [BGEM3Dense] NaN count={np.isnan(result).sum()}")
            print(f"  [BGEM3Dense] Inf count={np.isinf(result).sum()}")

        return result


# ---------------------------------------------------------------------------
# Cache metadata helper
# ---------------------------------------------------------------------------

def build_cache_metadata(
    corpus_chunks: List[Chunk],
    gold_map: Dict[str, List[str]],
    retriever: BGEM3DenseRetriever,
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
