"""ColBERTv2 full-corpus retrieval.

Uses the stanford-futuredata/ColBERT package (``colbert-ai``) to index the
entire corpus and perform end-to-end retrieval — *not* reranking on top of
another retriever.

Index layout (under ``index_root/INDEX_NAME/``)::

    collection.tsv          pid \\t passage_text
    pid_to_chunk_id.json    {pid: chunk_id}
    ... (ColBERT index files written by the Indexer in nested subdirectories)

ColBERT may create nested index directories.  After building, the actual
index path is recorded so subsequent runs can locate and reuse the index.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from feg_rag.data.chunker import Chunk

# ---------------------------------------------------------------------------
# ColBERT import helpers (lazy — only fail when actually called)
# ---------------------------------------------------------------------------

_COLBERT_AVAILABLE = False
_COLBERT_ERROR: str = ""

try:
    from colbert.infra import Run, RunConfig
    from colbert.infra.config import ColBERTConfig
    from colbert import Indexer, Searcher

    _COLBERT_AVAILABLE = True
except ImportError as exc:
    _COLBERT_ERROR = (
        f"ColBERT package not installed: {exc}\n"
        "Install with:  pip install colbert-ai  (or  pip install ragatouille)\n"
        "See: https://github.com/stanford-futuredata/ColBERT"
    )


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------


class ColBERTv2Retriever:
    """ColBERTv2 full-corpus dense retrieval.

    Parameters
    ----------
    checkpoint:
        HuggingFace model name or local path, e.g. ``"colbert-ir/colbertv2.0"``.
    index_root:
        Directory under which all ColBERT indices live (e.g. ``cache/colbert``).
    index_name:
        Sub-directory name for this specific index (e.g. ``finder_colbertv2``).
    nbits:
        Number of bits for FAISS PQ compression (2 = recommended for ColBERTv2).
    doc_maxlen:
        Maximum number of tokens per document (passage).
    query_maxlen:
        Maximum number of tokens per query.
    device:
        Device string (``"cuda"``, ``"cpu"``, ``"cuda:0"``, …).  Defaults to
        ``"cuda"`` when available, otherwise ``"cpu"``.
    """

    _INDEX_MARKERS = ("ivf.pid.pt", "ivf.metadata.pt", "centroids.pt", "avg_residual.pt")

    def __init__(
        self,
        checkpoint: str = "colbert-ir/colbertv2.0",
        index_root: str | Path = "cache/colbert",
        index_name: str = "finder_colbertv2",
        nbits: int = 2,
        doc_maxlen: int = 300,
        query_maxlen: int = 64,
        device: str | None = None,
    ):
        if not _COLBERT_AVAILABLE:
            raise ImportError(_COLBERT_ERROR)

        self.checkpoint = checkpoint
        self.index_root = Path(index_root)
        self.index_name = index_name
        self.nbits = nbits
        self.doc_maxlen = doc_maxlen
        self.query_maxlen = query_maxlen

        if device is None:
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self._chunks: List[Chunk] = []
        self._pid_to_chunk_id: Dict[int, str] = {}
        self._chunk_id_to_pid: Dict[str, int] = {}
        self._searcher: Optional[Searcher] = None

        # Resolved after building: actual directory containing index files.
        self._resolved_index_path: Optional[Path] = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def index_dir(self) -> Path:
        """Top-level index directory (collection + pid map live here)."""
        return (self.index_root / self.index_name).resolve()

    @property
    def collection_path(self) -> Path:
        return self.index_dir / "collection.tsv"

    @property
    def pid_map_path(self) -> Path:
        return self.index_dir / "pid_to_chunk_id.json"

    @property
    def manifest_path(self) -> Path:
        return self.index_dir / "collection_manifest.json"

    @property
    def is_indexed(self) -> bool:
        """Check whether a complete ColBERT index exists on disk.

        Scans recursively under ``index_dir`` for ColBERT index markers
        (e.g. ``ivf.pid.pt``) rather than assuming a fixed path, because
        ColBERT may nest the actual index files in subdirectories.
        """
        if not self.index_dir.exists():
            return False
        if not self.collection_path.exists():
            return False
        if not self.pid_map_path.exists():
            return False
        # Scan recursively for at least one index marker file
        found = self._find_index_marker()
        if found is not None:
            self._resolved_index_path = found.parent
            return True
        return False

    @property
    def resolved_index_path(self) -> Optional[Path]:
        """The actual directory containing ColBERT index files, if known."""
        if self._resolved_index_path is not None:
            return self._resolved_index_path
        found = self._find_index_marker()
        if found is not None:
            self._resolved_index_path = found.parent
        return self._resolved_index_path

    # ------------------------------------------------------------------
    # Index
    # ------------------------------------------------------------------

    def index(
        self,
        chunks: List[Chunk],
        force_rebuild: bool = False,
        verbose: bool = True,
    ) -> None:
        """Build (or load) the ColBERTv2 index for *chunks*.

        If a complete index already exists on disk and *force_rebuild* is
        ``False``, the index and pid mapping are loaded from disk without
        recomputation.

        Parameters
        ----------
        chunks:
            Corpus chunks to index.
        force_rebuild:
            If ``True``, delete any existing index and rebuild from scratch.
        verbose:
            Print progress messages.
        """
        self._chunks = chunks

        if self.is_indexed and not force_rebuild:
            if self._cache_matches_chunks(chunks):
                if verbose:
                    rpath = self.resolved_index_path or self.index_dir
                    print(f"  [ColBERT] Loading cached index from {rpath}")
                self._load_pid_map()
                self._init_searcher()
                return

            if verbose:
                print(
                    "  [ColBERT] Existing index does not match current corpus "
                    "or ColBERT settings; rebuilding."
                )
            shutil.rmtree(self.index_dir)

        if force_rebuild and self.index_dir.exists():
            if verbose:
                print(f"  [ColBERT] Removing existing index at {self.index_dir}")
            shutil.rmtree(self.index_dir)

        self.index_dir.mkdir(parents=True, exist_ok=True)

        # --- write collection.tsv ---
        if verbose:
            print(f"  [ColBERT] Writing collection.tsv ({len(chunks)} passages)")
        self._pid_to_chunk_id = {}
        self._chunk_id_to_pid = {}
        with open(self.collection_path, "w", encoding="utf-8") as fh:
            for pid, chunk in enumerate(chunks):
                text = chunk.text.replace("\t", " ").replace("\n", " ")
                fh.write(f"{pid}\t{text}\n")
                self._pid_to_chunk_id[pid] = chunk.chunk_id
                self._chunk_id_to_pid[chunk.chunk_id] = pid

        # --- persist pid mapping ---
        with open(self.pid_map_path, "w", encoding="utf-8") as fh:
            json.dump(
                {str(k): v for k, v in self._pid_to_chunk_id.items()},
                fh,
                ensure_ascii=False,
            )

        # --- build index ---
        if verbose:
            print(
                f"  [ColBERT] Building index: nbits={self.nbits}, "
                f"doc_maxlen={self.doc_maxlen}, device={self.device}"
            )

        with Run().context(
            RunConfig(
                nranks=1,
                root=str(self.index_root.resolve()),
                experiment=self.index_name,
                index_root=str(self.index_root.resolve()),
            )
        ):
            config = ColBERTConfig(
                doc_maxlen=self.doc_maxlen,
                query_maxlen=self.query_maxlen,
                nbits=self.nbits,
                root=str(self.index_root.resolve()),
                experiment=self.index_name,
                index_root=str(self.index_root.resolve()),
                checkpoint=self.checkpoint,
                gpus=1 if self.device.startswith("cuda") else 0,
                rank=0,
            )

            indexer = Indexer(
                checkpoint=self.checkpoint,
                config=config,
            )
            indexer.index(
                name=self.index_name,
                collection=str(self.collection_path),
                overwrite=True,
            )

        # --- resolve the actual index path (ColBERT may have nested it) ---
        found = self._find_index_marker()
        if found is not None:
            self._resolved_index_path = found.parent
            if verbose:
                print(f"  [ColBERT] Resolved index path: {self._resolved_index_path}")
        else:
            if verbose:
                print(
                    f"  [ColBERT] WARNING: Could not locate index marker files "
                    f"under {self.index_dir}.  Searcher may fail."
                )

        self._write_manifest(chunks)
        self._init_searcher()
        if verbose:
            print(f"  [ColBERT] Index ready: {self.index_dir}")

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 50,
        verbose: bool = False,
    ) -> List[Tuple[Chunk, float]]:
        """Search the full corpus with ColBERTv2.

        Returns top-k *(Chunk, score)* pairs sorted by descending score.
        """
        if self._searcher is None:
            self._init_searcher()

        if self._searcher is None:
            raise RuntimeError(
                "ColBERTv2 index not built or loaded. Call .index() first."
            )

        # ColBERT versions differ: some return an iterable of
        # (pid, rank, score), while others return (pids, ranks, scores).
        raw_results = self._searcher.search(query, k=top_k)
        def _is_scalar(value) -> bool:
            if hasattr(value, "ndim"):
                try:
                    return int(value.ndim) == 0
                except Exception:
                    pass
            return isinstance(value, (int, float, str))

        if isinstance(raw_results, tuple) and len(raw_results) == 3 and not _is_scalar(raw_results[0]):
            raw_iter = zip(raw_results[0], raw_results[1], raw_results[2])
        else:
            raw_iter = raw_results

        # Build chunk lookup
        chunk_by_id = {c.chunk_id: c for c in self._chunks}

        results: List[Tuple[Chunk, float]] = []
        for item in raw_iter:
            pid, _rank, score = item[:3]
            if hasattr(pid, "item"):
                pid = pid.item()
            pid = int(pid)
            if hasattr(score, "item"):
                score = score.item()
            chunk_id = self._pid_to_chunk_id.get(pid)
            if chunk_id is not None:
                chunk = chunk_by_id.get(chunk_id)
                if chunk is not None:
                    results.append((chunk, float(score)))
                    continue
            if 0 <= pid < len(self._chunks):
                results.append((self._chunks[pid], float(score)))

        if not results and raw_results:
            sample = raw_results
            try:
                if isinstance(raw_results, tuple):
                    sample = tuple(x[:3] if hasattr(x, "__getitem__") else x for x in raw_results)
                else:
                    sample = list(raw_results)[:3]
            except Exception:
                pass
            print(f"  [ColBERT] WARNING: search returned no mapped chunks; raw sample={sample}")

        return results

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _find_index_marker(self) -> Optional[Path]:
        """Scan recursively for a ColBERT index marker file under ``index_dir``."""
        if not self.index_dir.exists():
            return None
        for marker in self._INDEX_MARKERS:
            hits = list(self.index_dir.rglob(marker))
            if hits:
                return hits[0]
        # Fallback: look for any .pt file
        hits = list(self.index_dir.rglob("*.pt"))
        return hits[0] if hits else None

    def _load_pid_map(self) -> None:
        """Load pid→chunk_id mapping from disk."""
        with open(self.pid_map_path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        self._pid_to_chunk_id = {int(k): v for k, v in raw.items()}
        self._chunk_id_to_pid = {v: int(k) for k, v in raw.items()}

    @staticmethod
    def _normalize_text(text: str) -> str:
        return " ".join((text or "").split())

    @staticmethod
    def _sha1(text: str) -> str:
        return hashlib.sha1(text.encode("utf-8")).hexdigest()

    def _manifest_for_chunks(self, chunks: List[Chunk]) -> Dict[str, Any]:
        chunk_hasher = hashlib.sha1()
        text_hasher = hashlib.sha1()
        for pid, chunk in enumerate(chunks):
            text_hash = self._sha1(self._normalize_text(chunk.text))
            chunk_hasher.update(
                f"{pid}\t{chunk.chunk_id}\t{text_hash}\n".encode("utf-8")
            )
            text_hasher.update(f"{pid}\t{text_hash}\n".encode("utf-8"))

        return {
            "version": 1,
            "chunk_count": len(chunks),
            "first_chunk_id": chunks[0].chunk_id if chunks else None,
            "last_chunk_id": chunks[-1].chunk_id if chunks else None,
            "chunk_sequence_sha1": chunk_hasher.hexdigest(),
            "text_sequence_sha1": text_hasher.hexdigest(),
            "checkpoint": str(self.checkpoint),
            "nbits": self.nbits,
            "doc_maxlen": self.doc_maxlen,
            "query_maxlen": self.query_maxlen,
        }

    def _write_manifest(self, chunks: List[Chunk]) -> None:
        with open(self.manifest_path, "w", encoding="utf-8") as fh:
            json.dump(self._manifest_for_chunks(chunks), fh, indent=2, ensure_ascii=False)

    def _cache_matches_chunks(self, chunks: List[Chunk]) -> bool:
        """Return True only when the cached index was built for these chunks."""
        if not self.manifest_path.exists():
            return False
        try:
            with open(self.manifest_path, "r", encoding="utf-8") as fh:
                cached = json.load(fh)
        except (OSError, json.JSONDecodeError):
            return False

        expected = self._manifest_for_chunks(chunks)
        keys = (
            "version",
            "chunk_count",
            "first_chunk_id",
            "last_chunk_id",
            "chunk_sequence_sha1",
            "text_sequence_sha1",
            "checkpoint",
            "nbits",
            "doc_maxlen",
            "query_maxlen",
        )
        return all(cached.get(k) == expected.get(k) for k in keys)

    def _init_searcher(self) -> None:
        """Initialize the ColBERT Searcher for an existing index."""
        # Ensure we have located the actual index files
        if self.is_indexed:
            resolved = self.resolved_index_path
        else:
            resolved = None

        if resolved is None:
            raise RuntimeError(
                f"ColBERTv2 index not found under {self.index_dir}. "
                "Run .index() first."
            )

        with Run().context(
            RunConfig(
                nranks=1,
                root=str(self.index_root.resolve()),
                experiment=self.index_name,
                index_root=str(self.index_root.resolve()),
            )
        ):
            config = ColBERTConfig(
                doc_maxlen=self.doc_maxlen,
                query_maxlen=self.query_maxlen,
                nbits=self.nbits,
                root=str(self.index_root.resolve()),
                experiment=self.index_name,
                checkpoint=self.checkpoint,
                index_root=str(self.index_root.resolve()),
                gpus=1 if self.device.startswith("cuda") else 0,
            )
            self._searcher = Searcher(
                index=self.index_name,
                config=config,
                collection=str(self.collection_path),
                checkpoint=self.checkpoint,
            )

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def pid_for_chunk(self, chunk_id: str) -> Optional[int]:
        """Return the ColBERT pid for a chunk_id, or None."""
        return self._chunk_id_to_pid.get(chunk_id)

    def chunk_id_for_pid(self, pid: int) -> Optional[str]:
        """Return the chunk_id for a ColBERT pid, or None."""
        return self._pid_to_chunk_id.get(pid)
