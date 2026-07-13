from __future__ import annotations

import json
from pathlib import Path

import pytest

from feg_rag.config import Config
from feg_rag.data.chunker import chunk_table_markdown, chunk_text
from feg_rag.data.corpus import build_benchmark_corpus
from feg_rag.data.loader import load_finder


RUNTIME_ROOT = Path(__file__).resolve().parent / "_runtime"


def _case_dir(name: str) -> Path:
    path = RUNTIME_ROOT / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_config(root: Path, *, allow_gold_only: bool = False) -> Path:
    cfg_path = root / "config.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                f'root_dir: "{root.as_posix()}"',
                'data_dir: "FinDER/data"',
                'edgar_dir: "10-k"',
                'output_dir: "outputs"',
                'cache_dir: "cache"',
                "datasets: [finder]",
                "chunk_size: 30",
                "chunk_overlap: 5",
                f"allow_gold_only_corpus: {str(allow_gold_only).lower()}",
            ]
        ),
        encoding="utf-8",
    )
    return cfg_path


def _sample() -> dict:
    evidence = "Revenue increased to 10.2 billion dollars in fiscal 2025."
    return {
        "id": "q1",
        "question": "What was revenue in 2025?",
        "answer": "10.2 billion dollars",
        "evidence_texts": [evidence],
    }


def test_chunk_text_ids_are_deterministic():
    text = " ".join(f"token{i}" for i in range(80))
    a = chunk_text(text, chunk_size=20, chunk_overlap=5, doc_id="DOC-1", section="Item 7")
    b = chunk_text(text, chunk_size=20, chunk_overlap=5, doc_id="DOC-1", section="Item 7")

    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]
    assert len({c.chunk_id for c in a}) == len(a)
    assert a[0].chunk_id != a[1].chunk_id


def test_chunk_table_ids_are_deterministic():
    table = "| A | B |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |"
    a = chunk_table_markdown(table, doc_id="DOC-1", section="Item 8", max_rows_per_chunk=1)
    b = chunk_table_markdown(table, doc_id="DOC-1", section="Item 8", max_rows_per_chunk=1)

    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]
    assert a[0].chunk_id != a[1].chunk_id


def test_chunk_text_rejects_invalid_overlap():
    with pytest.raises(ValueError):
        chunk_text("a b c", chunk_size=3, chunk_overlap=3)
    with pytest.raises(ValueError):
        chunk_text("a b c", chunk_size=0, chunk_overlap=0)


def test_load_finder_split_filters_parquet_files(monkeypatch):
    root = _case_dir("finder_split")
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "train-00000.parquet").write_text("", encoding="utf-8")
    (data_dir / "test-00000.parquet").write_text("", encoding="utf-8")

    seen = []

    class _Refs:
        def tolist(self):
            return ["Revenue increased."]

    class _Row(dict):
        def __getitem__(self, key):
            return self.get(key)

    class _DF:
        def iterrows(self):
            yield 0, _Row(
                {
                    "_id": "q-test",
                    "text": "Question?",
                    "answer": "Answer",
                    "references": _Refs(),
                }
            )

    def fake_read_parquet(path):
        seen.append(Path(path).name)
        return _DF()

    monkeypatch.setattr("feg_rag.data.loader.pd.read_parquet", fake_read_parquet)

    samples = load_finder(data_dir, split="test")

    assert seen == ["test-00000.parquet"]
    assert samples[0]["id"] == "q-test"


def test_corpus_builder_does_not_default_to_gold_only():
    root = _case_dir("no_gold_only")
    cfg = Config.from_yaml(_write_config(root))

    with pytest.raises(FileNotFoundError):
        build_benchmark_corpus([_sample()], cfg)


def test_gold_evidence_aligns_to_document_chunk():
    root = _case_dir("alignment")
    cfg = Config.from_yaml(_write_config(root))
    edgar = root / "10-k"
    edgar.mkdir(exist_ok=True)
    (edgar / "ACME_2025_10-K.txt").write_text(
        "Item 7. Management discussion. Revenue increased to 10.2 billion "
        "dollars in fiscal 2025. Operating income also increased.",
        encoding="utf-8",
    )

    corpus, gold_map, records = build_benchmark_corpus([_sample()], cfg)

    assert corpus
    assert gold_map["q1"]
    assert records[0].matched_chunk_ids == gold_map["q1"]
    assert set(gold_map["q1"]).issubset({c.chunk_id for c in corpus})


def test_run_pipeline_step_data_ignores_incomplete_cache(monkeypatch):
    pytest.importorskip("rank_bm25")
    import run_pipeline

    root = _case_dir("pipeline_cache")
    cfg = Config.from_yaml(_write_config(root))
    edgar = root / "10-k"
    edgar.mkdir(exist_ok=True)
    (edgar / "ACME_2025_10-K.txt").write_text(
        "Item 7. Revenue increased to 10.2 billion dollars in fiscal 2025.",
        encoding="utf-8",
    )
    cfg.cache_dir.mkdir(parents=True)
    (cfg.cache_dir / "data_state.json").write_text(
        json.dumps({"incomplete": True}), encoding="utf-8"
    )

    monkeypatch.setattr(run_pipeline, "load_dataset", lambda *_args, **_kwargs: [_sample()])

    pipeline = run_pipeline.Pipeline(cfg)
    samples, corpus, gold_map = pipeline.step_data()

    assert len(samples) == 1
    assert corpus
    assert gold_map["q1"]
