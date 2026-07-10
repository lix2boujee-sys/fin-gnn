"""Dense (vector) retrieval via embeddings + faiss.

Supports:
  - Local HuggingFace model dir via transformers (recommended on this machine)
  - sentence-transformers hub name (optional, if import works)
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np

from feg_rag.data.chunker import Chunk

DEFAULT_LOCAL_MODEL = Path("D:/fin-gnn/cache/models/all-MiniLM-L6-v2")


class _TransformersEncoder:
    """Mean-pooled embeddings using transformers only (no sentence-transformers)."""

    def __init__(self, model_path: str, device: str | None = None):
        import torch
        from transformers import AutoModel, AutoTokenizer

        # Default CPU: batch indexing on a 4 GB GPU easily OOMs (batch 256 × seq 512).
        if device is None:
            self.device = "cpu"
        elif device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"
        else:
            self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path).to(self.device)
        self.model.eval()

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
                max_length=512,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                token_emb = outputs.last_hidden_state
                mask = inputs["attention_mask"].unsqueeze(-1).expand(token_emb.size()).float()
                summed = torch.sum(token_emb * mask, dim=1)
                counts = torch.clamp(mask.sum(dim=1), min=1e-9)
                emb = summed / counts
                if normalize_embeddings:
                    emb = torch.nn.functional.normalize(emb, p=2, dim=1)
                parts.append(emb.cpu().numpy())
        return np.vstack(parts) if parts else np.zeros((0, 384), dtype=np.float32)


class DenseRetriever:
    """Dense retrieval using neural embeddings + FAISS index."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str | None = None,
    ):
        self.model_name = model_name
        self._encoding_device = device
        self._encoder = None
        self._backend = ""
        self._index: faiss.IndexFlatIP | None = None
        self._chunks: List[Chunk] = []
        self._embeddings: np.ndarray | None = None

    def index(self, chunks: List[Chunk], batch_size: int | None = None) -> None:
        """Encode chunks and build FAISS index."""
        self._chunks = chunks
        encoder = self._get_encoder()
        effective_bs = batch_size or self._default_batch_size(encoder)
        texts = [c.text for c in chunks]
        embeddings = encoder.encode(
            texts,
            batch_size=effective_bs,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        dim = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(embeddings.astype(np.float32))
        self._embeddings = embeddings.astype(np.float32)

    def search(self, query: str, top_k: int = 50) -> List[Tuple[Chunk, float]]:
        """Return top-k chunks with cosine similarity scores."""
        if self._index is None:
            raise RuntimeError("Index not built. Call .index() first.")
        encoder = self._get_encoder()
        q_emb = encoder.encode(
            [query], normalize_embeddings=True, show_progress_bar=False
        )
        scores, indices = self._index.search(q_emb.astype(np.float32), top_k)
        results: List[Tuple[Chunk, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._chunks):
                continue
            results.append((self._chunks[idx], float(score)))
        return results

    def save(self, path: str | Path) -> None:
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
        return self._backend

    def chunk_embeddings(self) -> dict[str, np.ndarray]:
        """Return indexed chunk embeddings keyed by chunk_id."""
        if self._embeddings is None:
            raise RuntimeError("Index not built. Call .index() first.")
        return {
            c.chunk_id: self._embeddings[i]
            for i, c in enumerate(self._chunks)
        }

    def _resolve_model_path(self) -> str:
        p = Path(self.model_name)
        if p.exists():
            return str(p)
        if DEFAULT_LOCAL_MODEL.exists():
            return str(DEFAULT_LOCAL_MODEL)
        return self.model_name

    @staticmethod
    def _default_batch_size(encoder) -> int:
        device = getattr(encoder, "device", "cpu")
        return 16 if device == "cuda" else 128

    def _get_encoder(self):
        if self._encoder is not None:
            return self._encoder

        model_path = self._resolve_model_path()
        if Path(model_path).exists():
            self._encoder = _TransformersEncoder(
                model_path, device=self._encoding_device
            )
            self._backend = f"transformers-local:{model_path}"
            return self._encoder

        try:
            from sentence_transformers import SentenceTransformer

            self._encoder = SentenceTransformer(self.model_name)
            self._backend = f"sentence-transformers:{self.model_name}"
        except Exception as exc:
            raise RuntimeError(
                f"Could not load dense model '{self.model_name}' "
                f"and no local model at {DEFAULT_LOCAL_MODEL}: {exc}"
            ) from exc
        return self._encoder
