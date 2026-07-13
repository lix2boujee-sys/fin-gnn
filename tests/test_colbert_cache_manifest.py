from __future__ import annotations

import json
import importlib.util
from pathlib import Path

from feg_rag.data.chunker import Chunk


RUNTIME_ROOT = Path(__file__).resolve().parent / "_colbert_runtime"
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _case_dir(name: str) -> Path:
    path = RUNTIME_ROOT / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def _chunks():
    return [
        Chunk(chunk_id="chunk::doc1::item7::0::aaa", text="Revenue increased.", chunk_type="text"),
        Chunk(chunk_id="chunk::doc1::item7::2::bbb", text="Operating income rose.", chunk_type="text"),
    ]


def _retriever(root, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "colbertv2_under_test",
        PROJECT_ROOT / "feg_rag" / "retrieval" / "colbertv2.py",
    )
    colbertv2 = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(colbertv2)

    monkeypatch.setattr(colbertv2, "_COLBERT_AVAILABLE", True)
    return colbertv2.ColBERTv2Retriever(
        checkpoint="local-colbert",
        index_root=root,
        index_name="idx",
        nbits=2,
        doc_maxlen=300,
        query_maxlen=64,
        device="cpu",
    )


def test_colbert_cache_requires_manifest(monkeypatch):
    retriever = _retriever(_case_dir("missing_manifest"), monkeypatch)
    retriever.index_dir.mkdir(parents=True)
    retriever.collection_path.write_text("0\tRevenue increased.\n", encoding="utf-8")
    retriever.pid_map_path.write_text(json.dumps({"0": "chunk::doc1::item7::0::aaa"}), encoding="utf-8")
    (retriever.index_dir / "ivf.pid.pt").write_bytes(b"marker")

    assert retriever.is_indexed
    assert not retriever._cache_matches_chunks(_chunks())


def test_colbert_cache_manifest_detects_corpus_change(monkeypatch):
    retriever = _retriever(_case_dir("corpus_change"), monkeypatch)
    chunks = _chunks()
    retriever.index_dir.mkdir(parents=True)
    retriever._write_manifest(chunks)

    assert retriever._cache_matches_chunks(chunks)

    changed = [
        Chunk(chunk_id=chunks[0].chunk_id, text=chunks[0].text, chunk_type="text"),
        Chunk(chunk_id=chunks[1].chunk_id, text="Different text.", chunk_type="text"),
    ]
    assert not retriever._cache_matches_chunks(changed)


def test_colbert_cache_manifest_detects_parameter_change(monkeypatch):
    root = _case_dir("parameter_change")
    retriever = _retriever(root, monkeypatch)
    chunks = _chunks()
    retriever.index_dir.mkdir(parents=True)
    retriever._write_manifest(chunks)

    other = _retriever(root, monkeypatch)
    other.doc_maxlen = 180

    assert not other._cache_matches_chunks(chunks)
