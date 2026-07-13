"""Dense (vector) retrieval via embeddings + faiss.

Supports:
  - Local HuggingFace model dir via transformers (recommended on this machine)
  - sentence-transformers hub name (optional, if import works)
  - E5-style models (e.g. intfloat/e5-mistral-7b-instruct) with a dedicated
    encoding path (E5MistralEncoder) that handles decoder-only LLMs correctly.
  - Non-E5 dense models (all-MiniLM-L6-v2, etc.) keep their existing
    mean-pooling behaviour unchanged.
"""

from __future__ import annotations

import logging
import pickle
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import faiss
import numpy as np

from feg_rag.data.chunker import Chunk

logger = logging.getLogger(__name__)

DEFAULT_LOCAL_MODEL = Path("D:/fin-gnn/cache/models/all-MiniLM-L6-v2")

# ---------------------------------------------------------------------------
# Constants / helpers
# ---------------------------------------------------------------------------

# Default query instruction for e5-mistral-7b-instruct.
# Configurable via ``retrieval.dense_query_instruction`` in YAML.
DEFAULT_E5_QUERY_INSTRUCTION = (
    "Given a financial question, retrieve relevant evidence passages "
    "from SEC filings that directly support the answer."
)


def _is_e5_model_name(model_name: str) -> bool:
    """Return True if *model_name* refers to any E5-family model."""
    return "e5" in model_name.lower()


def _is_e5_mistral_instruct(model_name: str) -> bool:
    """Return True if *model_name* refers specifically to an E5-Mistral model.

    Matches:
      - e5-mistral
      - e5-mistral-7b-instruct
      - intfloat/e5-mistral-7b-instruct
    """
    return "e5-mistral" in model_name.lower()


# ---------------------------------------------------------------------------
# Legacy formatting helpers (used for non-Mistral E5 models)
# ---------------------------------------------------------------------------

def format_e5_passage(text: str) -> str:
    """Format a passage / chunk for E5-family encoders (non-Mistral)."""
    return f"passage: {text}"


def format_e5_query(query: str, model_name: str) -> str:
    """Format a query for E5-family encoders (non-Mistral).

    Note: E5-Mistral query formatting is handled inside DenseRetriever._fmt_query
    using the configurable instruction, not via this function.
    """
    return f"query: {query}"


# ===================================================================
# E5MistralEncoder — dedicated encoder for e5-mistral-7b-instruct
# ===================================================================

class E5MistralEncoder:
    """Dedicated encoder for intfloat/e5-mistral-7b-instruct.

    **Primary path** (always tried first):
      ``SentenceTransformer(model_path, device=device, trust_remote_code=True)``
      using the model's own ``modules.json`` / ``1_Pooling`` / ``2_Normalize``
      configuration.  We set ``max_seq_length`` and then call ``encode()``
      with ``prompt=""`` so that the wrapper does **not** auto-apply its own
      prompt on top of the manual query/passage formatting done by the caller.

    **Fallback path** (only if SentenceTransformer fails to load):
      ``AutoModel + AutoTokenizer`` with last-token pooling, suitable for
      decoder-only LLMs.  A warning is printed when the fallback is active.

    All embeddings are L2-normalised (required for FAISS IndexFlatIP =
    cosine similarity).
    """

    def __init__(
        self,
        model_path: str,
        device: str | None = None,
        max_seq_length: int = 512,
        debug: bool = False,
    ):
        import torch

        self.model_path = model_path
        self.max_seq_length = max_seq_length
        self.debug = debug
        self._st_has_prompts: bool = False

        # ── Resolve device ─────────────────────────────────────────
        if device is None:
            self._device = "cpu"
        elif device == "cuda" and not torch.cuda.is_available():
            self._device = "cpu"
        else:
            self._device = device

        self._use_st = False
        self._dim: int | None = None
        self._model: Any = None
        self._tokenizer: Any = None
        self._backend: str = ""

        st_error: Exception | None = None
        ft_error: Exception | None = None

        # ── Primary: SentenceTransformer ──────────────────────────
        try:
            from sentence_transformers import SentenceTransformer

            if self.debug:
                print(f"  [E5MistralEncoder] Loading via SentenceTransformer: "
                      f"{model_path}")
            self._model = SentenceTransformer(
                model_path,
                device=self._device,
                trust_remote_code=True,
            )
            self._model.max_seq_length = max_seq_length
            self._use_st = True
            self._backend = f"sentence-transformers:e5-mistral:{model_path}"
            self._dim = self._model.get_sentence_embedding_dimension()

            if self.debug:
                print(f"  [E5MistralEncoder] backend={self._backend}")
                print(f"  [E5MistralEncoder] embedding_dim={self._dim}")
                print(f"  [E5MistralEncoder] max_seq_length={max_seq_length}")
                print(f"  [E5MistralEncoder] device={self._device}")
            return
        except Exception as exc:
            st_error = exc
            warnings.warn(
                f"E5MistralEncoder: SentenceTransformer load failed ({exc}). "
                f"Falling back to transformers AutoModel with last-token pooling."
            )

        # ── Fallback: transformers AutoModel + AutoTokenizer ──────
        try:
            from transformers import AutoModel, AutoTokenizer

            if self.debug:
                print(f"  [E5MistralEncoder] Loading via transformers "
                      f"AutoModel: {model_path}")

            self._tokenizer = AutoTokenizer.from_pretrained(model_path)
            self._model = AutoModel.from_pretrained(model_path).to(
                self._device)
            self._model.eval()
            self._use_st = False
            self._backend = f"transformers-fallback:e5-mistral:{model_path}"
            self._dim = self._model.config.hidden_size

            if self.debug:
                print(f"  [E5MistralEncoder] backend={self._backend}")
                print(f"  [E5MistralEncoder] embedding_dim={self._dim}")
                print(f"  [E5MistralEncoder] max_seq_length={max_seq_length}")
                print(f"  [E5MistralEncoder] device={self._device}")
        except Exception as exc:
            ft_error = exc
            raise RuntimeError(
                f"E5MistralEncoder: Failed to load model from '{model_path}'. "
                f"Both SentenceTransformer and AutoModel paths failed.\n"
                f"  SentenceTransformer error: {st_error}\n"
                f"  AutoModel error: {ft_error}\n"
                f"  Hint: Ensure all model weight files (e.g. *.safetensors) "
                f"are present in the model directory."
            ) from ft_error

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def backend(self) -> str:
        """Backend identifier string (e.g. sentence-transformers:e5-mistral:...)."""
        return self._backend

    @property
    def embedding_dim(self) -> int:
        """Embedding dimension (4096 for e5-mistral-7b-instruct)."""
        if self._dim is None:
            raise RuntimeError("E5MistralEncoder: embedding dimension not set")
        return self._dim

    @property
    def device(self) -> str:
        """Device string for compatibility with DenseRetriever helpers."""
        return self._device

    # ------------------------------------------------------------------
    # encode  (public API)
    # ------------------------------------------------------------------

    def encode(
        self,
        texts: List[str],
        batch_size: int = 4,
        show_progress_bar: bool = False,
        normalize_embeddings: bool = True,
        prompt: str | None = None,
    ) -> np.ndarray:
        """Encode *texts* into L2-normalised float32 embeddings.

        Parameters
        ----------
        texts:
            Already-formatted strings (query with instruction, or raw passage).
        batch_size:
            Batch size for encoding.
        show_progress_bar:
            Whether to show a tqdm progress bar.
        normalize_embeddings:
            Must be ``True`` for FAISS ``IndexFlatIP`` (cosine similarity).
            The fallback path always normalises; the ST path respects this flag.
        prompt:
            Passed through to ``SentenceTransformer.encode()``.
            Set to ``""`` to prevent the ST wrapper from auto-applying its
            own prompt templates.  Ignored in the fallback path.

        Returns
        -------
        np.ndarray  shape ``(len(texts), embedding_dim)``, dtype ``float32``.
        """
        if self._use_st:
            embeddings = self._encode_st(
                texts, batch_size, show_progress_bar,
                normalize_embeddings, prompt,
            )
        else:
            embeddings = self._encode_fallback(
                texts, batch_size, show_progress_bar,
            )

        # Ensure float32
        embeddings = np.asarray(embeddings, dtype=np.float32)

        # ── Sanity checks ─────────────────────────────────────────
        self._sanity_check(embeddings, texts)

        if self.debug:
            norms = np.linalg.norm(embeddings, axis=1)
            print(f"  [E5MistralEncoder.encode] shape={embeddings.shape}")
            print(f"  [E5MistralEncoder.encode] mean L2 norm={norms.mean():.6f}")
            print(f"  [E5MistralEncoder.encode] NaN count={np.isnan(embeddings).sum()}")
            print(f"  [E5MistralEncoder.encode] Inf count={np.isinf(embeddings).sum()}")

        return embeddings

    # ------------------------------------------------------------------
    # Internal: SentenceTransformer path
    # ------------------------------------------------------------------

    def _encode_st(
        self,
        texts: List[str],
        batch_size: int,
        show_progress_bar: bool,
        normalize_embeddings: bool,
        prompt: str | None,
    ) -> np.ndarray:
        encode_kwargs: Dict[str, Any] = dict(
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            normalize_embeddings=normalize_embeddings,
        )
        # Always pass an explicit prompt to prevent ST from layering
        # its own prompt templates on top of our manual formatting.
        if prompt is not None:
            encode_kwargs["prompt"] = prompt
        else:
            encode_kwargs["prompt"] = ""

        return self._model.encode(texts, **encode_kwargs)

    # ------------------------------------------------------------------
    # Internal: transformers AutoModel fallback
    # ------------------------------------------------------------------

    def _encode_fallback(
        self,
        texts: List[str],
        batch_size: int,
        show_progress_bar: bool,
    ) -> np.ndarray:
        """Encode using AutoModel + last-token pooling (decoder-only safe)."""
        import torch
        from tqdm import tqdm

        batches = range(0, len(texts), batch_size)
        if show_progress_bar:
            batches = tqdm(batches, desc="E5-Mistral (fallback)", leave=False)

        parts: List[np.ndarray] = []
        for start in batches:
            batch = texts[start:start + batch_size]
            inputs = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_seq_length,
                return_tensors="pt",
            ).to(self._device)

            with torch.no_grad():
                outputs = self._model(**inputs)
                token_emb = outputs.last_hidden_state
                emb = self._last_token_pool(
                    token_emb, inputs["attention_mask"])
                emb = torch.nn.functional.normalize(emb, p=2, dim=1)
                parts.append(emb.cpu().numpy())

        if parts:
            return np.vstack(parts)
        return np.zeros((0, self._dim or 0), dtype=np.float32)

    @staticmethod
    def _last_token_pool(last_hidden_states, attention_mask):
        """Pool the final non-padding token (decoder-only LLM compatible)."""
        import torch

        left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
        if left_padding:
            return last_hidden_states[:, -1]
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size_val = last_hidden_states.shape[0]
        return last_hidden_states[
            torch.arange(batch_size_val, device=last_hidden_states.device),
            sequence_lengths,
        ]

    # ------------------------------------------------------------------
    # Sanity checks
    # ------------------------------------------------------------------

    def _sanity_check(self, embeddings: np.ndarray, texts: List[str]) -> None:
        """Validate embeddings and raise clear errors on failure."""
        if embeddings.size == 0:
            raise RuntimeError(
                f"E5MistralEncoder: produced empty embeddings for "
                f"{len(texts)} texts"
            )
        if embeddings.ndim != 2:
            raise RuntimeError(
                f"E5MistralEncoder: expected 2D embeddings, "
                f"got shape {embeddings.shape}"
            )
        if embeddings.shape[0] != len(texts):
            raise RuntimeError(
                f"E5MistralEncoder: expected {len(texts)} embeddings, "
                f"got {embeddings.shape[0]}"
            )
        if self._dim is not None and embeddings.shape[1] != self._dim:
            raise RuntimeError(
                f"E5MistralEncoder: expected embedding dim {self._dim}, "
                f"got {embeddings.shape[1]}"
            )
        if np.isnan(embeddings).any():
            nan_count = int(np.isnan(embeddings).sum())
            raise RuntimeError(
                f"E5MistralEncoder: {nan_count} NaN values in embeddings"
            )
        if np.isinf(embeddings).any():
            inf_count = int(np.isinf(embeddings).sum())
            raise RuntimeError(
                f"E5MistralEncoder: {inf_count} Inf values in embeddings"
            )

        # Check normalisation
        norms = np.linalg.norm(embeddings, axis=1)
        mean_norm = float(norms.mean())
        if not (0.99 <= mean_norm <= 1.01):
            warnings.warn(
                f"E5MistralEncoder: mean L2 norm after normalisation is "
                f"{mean_norm:.6f} (expected ≈1.0). Embeddings may not be "
                f"normalised."
            )


# ===================================================================
# _TransformersEncoder — generic encoder for non-E5 models
# ===================================================================

class _TransformersEncoder:
    """Embeddings using transformers only (no sentence-transformers).

    Used for non-E5 local models such as all-MiniLM-L6-v2.
    Default pooling is **mean** pooling, which is correct for BERT-family
    encoder-only models.
    """

    def __init__(
        self,
        model_path: str,
        device: str | None = None,
        pooling: str = "mean",
        max_length: int = 512,
    ):
        import torch
        from transformers import AutoModel, AutoTokenizer

        # Default CPU: batch indexing on a 4 GB GPU easily OOMs
        # (batch 256 × seq 512).
        if device is None:
            self.device = "cpu"
        elif device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"
        else:
            self.device = device
        self.pooling = pooling
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path).to(self.device)
        self.model.eval()

        # Determine embedding dimension from the model config so we never
        # hard-code a MiniLM-specific fallback.
        self._dim: int = self.model.config.hidden_size

    @property
    def embedding_dim(self) -> int:
        return self._dim

    def encode(
        self,
        texts: List[str],
        batch_size: int = 256,
        show_progress_bar: bool = False,
        normalize_embeddings: bool = True,
    ) -> np.ndarray:
        import torch
        from tqdm import tqdm

        batches = range(0, len(texts), batch_size)
        if show_progress_bar:
            batches = tqdm(batches, desc="Encoding", leave=False)

        parts: List[np.ndarray] = []
        for start in batches:
            batch = texts[start:start + batch_size]
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                token_emb = outputs.last_hidden_state
                if self.pooling == "last":
                    emb = self._last_token_pool(
                        token_emb, inputs["attention_mask"])
                else:
                    mask = (inputs["attention_mask"]
                            .unsqueeze(-1)
                            .expand(token_emb.size())
                            .float())
                    summed = torch.sum(token_emb * mask, dim=1)
                    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
                    emb = summed / counts
                if normalize_embeddings:
                    emb = torch.nn.functional.normalize(emb, p=2, dim=1)
                parts.append(emb.cpu().numpy())
        if parts:
            return np.vstack(parts)
        # Never hard-code dimension — use the model's actual hidden size.
        return np.zeros((0, self._dim), dtype=np.float32)

    @staticmethod
    def _last_token_pool(last_hidden_states, attention_mask):
        """Pool the final non-padding token, matching E5-Mistral usage."""
        import torch

        left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
        if left_padding:
            return last_hidden_states[:, -1]
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size_val = last_hidden_states.shape[0]
        return last_hidden_states[
            torch.arange(batch_size_val, device=last_hidden_states.device),
            sequence_lengths,
        ]


# ===================================================================
# DenseRetriever
# ===================================================================

class DenseRetriever:
    """Dense retrieval using neural embeddings + FAISS index.

    Automatically routes to the correct encoder backend:

    * **E5-Mistral** (``"e5-mistral"`` in model name)
      → ``E5MistralEncoder`` (dedicated, last-token-pooling or ST)

    * **Other E5** (``"e5"`` in model name, non-Mistral)
      → ``SentenceTransformer`` (via hub or local path)

    * **Non-E5 local** (e.g. ``all-MiniLM-L6-v2``)
      → ``_TransformersEncoder`` with mean pooling

    * **Non-E5 hub name**
      → ``SentenceTransformer`` (standard path)

    For E5-Mistral, the query is formatted as::

        Instruct: {query_instruction}
        Query: {raw_query}

    Passages are used as raw text (no prefix).
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str | None = None,
        query_instruction: str | None = None,
        e5_max_seq_length: int = 512,
        e5_batch_size: int | None = None,
        debug: bool = False,
    ):
        self.model_name = model_name
        self._encoding_device = device
        self._encoder = None
        self._backend = ""
        self._index: faiss.IndexFlatIP | None = None
        self._chunks: List[Chunk] = []
        self._embeddings: np.ndarray | None = None

        # E5-Mistral specific settings
        self._e5_query_instruction: str = (
            query_instruction or DEFAULT_E5_QUERY_INSTRUCTION
        )
        self._e5_max_seq_length: int = e5_max_seq_length
        self._e5_batch_size: int | None = e5_batch_size
        self._debug: bool = debug

        # Set by _get_encoder()
        self._embedding_dim: int | None = None

    # ------------------------------------------------------------------
    # Model identity helpers
    # ------------------------------------------------------------------

    @property
    def is_e5(self) -> bool:
        """True for any E5-family model name."""
        return _is_e5_model_name(self.model_name)

    @property
    def is_e5_mistral(self) -> bool:
        """True specifically for e5-mistral-7b-instruct variants."""
        return _is_e5_mistral_instruct(self.model_name)

    # ------------------------------------------------------------------
    # Text formatting (E5-aware, no-op for non-E5 models)
    # ------------------------------------------------------------------

    def _fmt_passage(self, text: str) -> str:
        """Format a passage / chunk text for the current model."""
        if self.is_e5_mistral:
            # E5-Mistral: raw passage text, no prefix
            return text
        if self.is_e5:
            # Non-Mistral E5: "passage: {text}"
            return format_e5_passage(text)
        # Non-E5: raw text
        return text

    def _fmt_query(self, query: str) -> str:
        """Format a query string for the current model."""
        if self.is_e5_mistral:
            # E5-Mistral instruct format
            return (
                f"Instruct: {self._e5_query_instruction}\nQuery: {query}"
            )
        if self.is_e5:
            # Non-Mistral E5: "query: {query}"
            return format_e5_query(query, self.model_name)
        # Non-E5: raw query
        return query

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def index(self, chunks: List[Chunk], batch_size: int | None = None) -> None:
        """Encode chunks and build FAISS ``IndexFlatIP``."""
        self._chunks = chunks
        encoder = self._get_encoder()
        effective_bs = batch_size or self._default_batch_size(encoder)

        # Apply per-model passage formatting
        texts = [self._fmt_passage(c.text) for c in chunks]

        encode_kwargs: Dict[str, Any] = dict(
            batch_size=effective_bs,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        # For E5 models with SentenceTransformer backend, explicitly set
        # prompt="" so the ST wrapper does NOT auto-apply its own prompt
        # on top of our manual formatting.
        if self.is_e5 and self._backend.startswith("sentence-transformers"):
            encode_kwargs["prompt"] = ""

        if self._debug:
            print(f"  [DenseRetriever.index] backend={self._backend}")
            print(f"  [DenseRetriever.index] model_path={self.model_name}")
            print(f"  [DenseRetriever.index] num_chunks={len(chunks)}")
            print(f"  [DenseRetriever.index] batch_size={effective_bs}")

        embeddings = encoder.encode(texts, **encode_kwargs)
        dim = embeddings.shape[1]
        self._embedding_dim = dim
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(embeddings.astype(np.float32))
        self._embeddings = embeddings.astype(np.float32)

        if self._debug:
            norms = np.linalg.norm(self._embeddings, axis=1)
            print(f"  [DenseRetriever.index] passage embeddings "
                  f"shape={self._embeddings.shape}")
            print(f"  [DenseRetriever.index] passage mean L2 norm="
                  f"{norms.mean():.6f}")
            print(f"  [DenseRetriever.index] passage NaN count="
                  f"{np.isnan(self._embeddings).sum()}")
            print(f"  [DenseRetriever.index] passage Inf count="
                  f"{np.isinf(self._embeddings).sum()}")
            print(f"  [DenseRetriever.index] FAISS ntotal={self._index.ntotal}")
            print(f"  [DenseRetriever.index] embedding_dim={dim}")

        # Suppress INFO-level messages from FAISS
        if logger.isEnabledFor(logging.INFO):
            logger.info("FAISS index built: ntotal=%d, dim=%d",
                        self._index.ntotal, dim)

    def search(
        self, query: str, top_k: int = 50
    ) -> List[Tuple[Chunk, float]]:
        """Return top-k chunks with cosine similarity scores (via IP)."""
        if self._index is None:
            raise RuntimeError("Index not built. Call .index() first.")

        encoder = self._get_encoder()
        formatted_query = self._fmt_query(query)

        encode_kwargs: Dict[str, Any] = dict(
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        if self.is_e5 and self._backend.startswith("sentence-transformers"):
            encode_kwargs["prompt"] = ""

        q_emb = encoder.encode([formatted_query], **encode_kwargs)

        if self._debug:
            q_norm = float(np.linalg.norm(q_emb))
            print(f"  [DenseRetriever.search] query embedding "
                  f"shape={q_emb.shape}")
            print(f"  [DenseRetriever.search] query L2 norm={q_norm:.6f}")
            print(f"  [DenseRetriever.search] query NaN count="
                  f"{np.isnan(q_emb).sum()}")
            print(f"  [DenseRetriever.search] query Inf count="
                  f"{np.isinf(q_emb).sum()}")

        scores, indices = self._index.search(
            q_emb.astype(np.float32), top_k)

        results: List[Tuple[Chunk, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._chunks):
                continue
            results.append((self._chunks[idx], float(score)))

        if self._debug and results:
            print(f"  [DenseRetriever.search] top-{min(5, len(results))} "
                  f"results:")
            print(f"    query: {query[:120]}")
            for rank, (chunk, score) in enumerate(results[:5], 1):
                text_preview = chunk.text[:120].replace("\n", " ")
                print(f"    rank={rank}  score={score:.6f}  "
                      f"chunk_id={chunk.chunk_id}  doc_id={chunk.doc_id}")
                print(f"      text: {text_preview}...")

        return results

    def save(self, path: str | Path) -> None:
        """Persist FAISS index, embeddings, and metadata to *path*."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "meta.pkl", "wb") as fh:
            pickle.dump(
                {
                    "model_name": self.model_name,
                    "backend": self._backend,
                    "chunks": self._chunks,
                },
                fh,
            )
        if self._index is not None:
            faiss.write_index(self._index, str(path / "index.faiss"))
        if self._embeddings is not None:
            np.save(str(path / "embeddings.npy"), self._embeddings)

    @classmethod
    def load(cls, path: str | Path) -> "DenseRetriever":
        """Load a previously saved DenseRetriever from *path*."""
        path = Path(path)
        with open(path / "meta.pkl", "rb") as fh:
            data = pickle.load(fh)
        obj = cls(model_name=data["model_name"])
        obj._backend = data.get("backend", "")
        obj._chunks = data["chunks"]
        index_path = path / "index.faiss"
        if index_path.exists():
            obj._index = faiss.read_index(str(index_path))
        emb_path = path / "embeddings.npy"
        if emb_path.exists():
            obj._embeddings = np.load(str(emb_path))
        return obj

    @property
    def backend(self) -> str:
        """Backend identifier string."""
        return self._backend

    @property
    def embedding_dim(self) -> int | None:
        """Return the (cached) embedding dimension, or None if not indexed."""
        if self._embedding_dim is not None:
            return self._embedding_dim
        # Try to get it from the encoder even before indexing
        try:
            encoder = self._get_encoder()
            return encoder.embedding_dim
        except Exception:
            return None

    def chunk_embeddings(self) -> dict[str, np.ndarray]:
        """Return indexed chunk embeddings keyed by chunk_id."""
        if self._embeddings is None:
            raise RuntimeError("Index not built. Call .index() first.")
        return {
            c.chunk_id: self._embeddings[i]
            for i, c in enumerate(self._chunks)
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _resolve_model_path(self) -> str:
        """Resolve *model_name* to a local path when possible.

        Important: if the caller explicitly asked for an E5 model we do NOT
        silently fall back to the MiniLM default — that would corrupt the
        comparison.  For non-E5 names the legacy fallback is preserved so
        existing MiniLM runs stay unchanged.
        """
        p = Path(self.model_name)
        if p.exists():
            return str(p)

        # Only fall back to the default local MiniLM when the user did NOT
        # explicitly request an E5 variant.
        if not self.is_e5 and DEFAULT_LOCAL_MODEL.exists():
            return str(DEFAULT_LOCAL_MODEL)

        return self.model_name

    def _default_batch_size(self, encoder) -> int:
        """Return a reasonable default batch size for *encoder*."""
        device = getattr(encoder, "device", "cpu")
        # E5-Mistral: use configured e5_batch_size, fallback to heuristics
        if self.is_e5_mistral:
            if self._e5_batch_size is not None:
                return self._e5_batch_size
            return 4 if device == "cuda" else 2
        # Other E5 models (smaller, but still careful)
        if self.is_e5:
            return 2 if device == "cuda" else 1
        # Non-E5 (MiniLM etc.)
        return 16 if device == "cuda" else 128

    def _get_encoder(self):
        """Return (and cache) the appropriate encoder for ``self.model_name``.

        Routing logic
        -------------
        1. E5-Mistral → ``E5MistralEncoder`` (dedicated)
        2. E5 (non-Mistral) local → ``SentenceTransformer``
        3. Non-E5 local → ``_TransformersEncoder`` (mean pooling)
        4. Hub name → ``SentenceTransformer``
        """
        if self._encoder is not None:
            return self._encoder

        model_path = self._resolve_model_path()

        # ── 1. E5-Mistral: dedicated encoder ──────────────────────
        if self.is_e5_mistral:
            self._encoder = E5MistralEncoder(
                model_path=model_path,
                device=self._encoding_device,
                max_seq_length=self._e5_max_seq_length,
                debug=self._debug,
            )
            self._backend = self._encoder.backend
            self._embedding_dim = self._encoder.embedding_dim
            return self._encoder

        # ── 2. E5 (non-Mistral): SentenceTransformer ──────────────
        if self.is_e5 and Path(model_path).exists():
            try:
                from sentence_transformers import SentenceTransformer

                print(f"  [dense] Loading E5 model via SentenceTransformer: "
                      f"{model_path}")
                self._encoder = SentenceTransformer(
                    model_path,
                    device=self._encoding_device or "cpu",
                    trust_remote_code=True,
                )
                self._backend = f"sentence-transformers:{model_path}"
                try:
                    self._embedding_dim = \
                        self._encoder.get_sentence_embedding_dimension()
                except Exception:
                    self._embedding_dim = None
                return self._encoder
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load E5 model '{model_path}' via "
                    f"SentenceTransformer: {exc}"
                ) from exc

        # ── 3. Non-E5 local model ─────────────────────────────────
        if Path(model_path).exists():
            self._encoder = _TransformersEncoder(
                model_path,
                device=self._encoding_device,
                pooling="mean",
                max_length=512,
            )
            self._backend = f"transformers-local:{model_path}"
            self._embedding_dim = self._encoder.embedding_dim
            return self._encoder

        # ── 4. Hub name (non-E5) — try sentence-transformers ─────
        try:
            from sentence_transformers import SentenceTransformer

            self._encoder = SentenceTransformer(
                self.model_name,
                device=self._encoding_device or "cpu",
            )
            self._backend = f"sentence-transformers:{self.model_name}"
            try:
                self._embedding_dim = \
                    self._encoder.get_sentence_embedding_dimension()
            except Exception:
                self._embedding_dim = None
        except Exception as exc:
            raise RuntimeError(
                f"Could not load dense model '{self.model_name}' "
                f"and no local model found at the given path: {exc}"
            ) from exc
        return self._encoder
