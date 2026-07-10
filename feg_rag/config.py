"""Centralised configuration for FEG-RAG experiments.

All paths, model settings, and experiment parameters are read from a single YAML
file so that every experiment script shares the same source of truth.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


# ---------------------------------------------------------------------------
# Defaults (used when the YAML file is missing a key)
# ---------------------------------------------------------------------------

DEFAULTS: Dict[str, Any] = {
    # ---- paths ----
    "root_dir": ".",
    "data_dir": "FinDER/data",
    "edgar_dir": "10-k",
    "output_dir": "outputs",
    "cache_dir": "cache",
    # ---- data ----
    "datasets": ["finder"],  # finder, tatqa, finqa
    "chunk_size": 512,
    "chunk_overlap": 64,
    "max_chunks_per_doc": 200,
    # ---- retrieval ----
    "retrieval": {
        "bm25_k1": 1.5,
        "bm25_b": 0.75,
        "dense_model": "sentence-transformers/all-MiniLM-L6-v2",
        "top_k": 50,
        "hybrid_alpha": 0.5,  # weight for BM25 vs dense
    },
    # ---- graph ----
    "graph": {
        "node_types": ["company", "filing", "section", "chunk", "metric", "year"],
        "edge_types": [
            "company-has-filing",
            "filing-has-section",
            "section-has-chunk",
            "chunk-mentions-metric",
            "chunk-mentions-year",
            "same-metric",
            "same-year",
            "same-company",
            "same-filing-year",
            "semantic-similar",
        ],
        "semantic_threshold": 0.7,
        "max_semantic_edges_per_node": 10,
        "use_edge_weights": True,
        # Edge weight defaults (from experiment design §4.3)
        "edge_weights": {
            "query-matches-company": 1.0,
            "query-matches-metric": 1.0,
            "query-matches-year": 1.0,
            "chunk-mentions-metric": 0.8,
            "chunk-mentions-year": 0.8,
            "company-has-filing": 0.7,
            "filing-has-section": 0.6,
            "section-has-chunk": 0.6,
            "same-metric": 0.5,
            "same-year": 0.5,
            "same-company": 0.5,
            "same-filing-year": 0.5,
            "semantic-similar": 0.3,
        },
    },
    # ---- cross-encoder (non-graph reranker baseline) ----
    "cross_encoder": {
        "model_name": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "top_k_rerank": 100,
        "batch_size": 32,
    },
    # ---- rerank ----
    "rerank": {
        "ppr_alpha": 0.85,
        "ppr_max_iter": 100,
        "gnn_model": "sage",  # sage | rgcn
        "gnn_hidden": 128,
        "gnn_layers": 2,
        "gnn_epochs": 50,
        "gnn_lr": 0.001,
        "gnn_dropout": 0.3,
        # fusion weights (α ret + β graph + γ gnn + δ constraint)
        "fusion_alpha": 0.3,   # retrieval score weight
        "fusion_beta": 0.3,    # graph/PPR score weight
        "fusion_gamma": 0.3,   # GNN score weight
        "fusion_delta": 0.1,   # constraint score weight
    },
    # ---- constraint scoring ----
    "constraint": {
        "enabled": True,
        "company_match_weight": 1.0,
        "year_match_weight": 1.0,
        "metric_match_weight": 0.8,
        "filing_type_match_weight": 0.5,
    },
    # ---- generation ----
    "generation": {
        "model": "gpt-4o",
        "temperature": 0.0,
        "max_tokens": 512,
        "top_k_evidence": 5,
        # Alternative local models (for robustness check Table 5)
        "alt_models": [
            "Qwen/Qwen2.5-7B-Instruct",
            "meta-llama/Llama-3.1-8B-Instruct",
        ],
        "use_local_model": False,
        "local_model_endpoint": "http://localhost:8000/v1",
    },
    # ---- evaluation ----
    "evaluation": {
        "recall_k_values": [1, 3, 5, 10, 20],
        "metrics": [
            "evidence_recall",
            "evidence_precision",
            "answer_accuracy",
            "numerical_consistency",
            "hallucination_rate",
        ],
    },
}


# ---------------------------------------------------------------------------
# Config object
# ---------------------------------------------------------------------------

@dataclass
class Config:
    """Typed configuration holder built from a YAML file."""

    # paths
    root_dir: Path
    data_dir: Path
    edgar_dir: Path
    output_dir: Path
    cache_dir: Path

    # data
    datasets: List[str]
    chunk_size: int
    chunk_overlap: int
    max_chunks_per_doc: int

    # retrieval
    retrieval: Dict[str, Any]

    # cross-encoder
    cross_encoder: Dict[str, Any]

    # graph
    graph: Dict[str, Any]

    # rerank
    rerank: Dict[str, Any]

    # constraint
    constraint: Dict[str, Any]

    # generation
    generation: Dict[str, Any]

    # evaluation
    evaluation: Dict[str, Any]

    # raw dict for ad-hoc access
    _raw: Dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """Load config from a YAML file, falling back to DEFAULTS."""
        path = Path(path)
        raw = dict(DEFAULTS)
        if path.exists():
            with open(path, "r", encoding="utf-8") as fh:
                user = yaml.safe_load(fh) or {}
            _deep_update(raw, user)

        # resolve paths relative to root_dir
        root = Path(raw["root_dir"]).resolve()
        return cls(
            root_dir=root,
            data_dir=_resolve(root, raw["data_dir"]),
            edgar_dir=_resolve(root, raw["edgar_dir"]),
            output_dir=_resolve(root, raw["output_dir"]),
            cache_dir=_resolve(root, raw["cache_dir"]),
            datasets=raw["datasets"],
            chunk_size=raw["chunk_size"],
            chunk_overlap=raw["chunk_overlap"],
            max_chunks_per_doc=raw["max_chunks_per_doc"],
            retrieval=raw["retrieval"],
            cross_encoder=raw["cross_encoder"],
            graph=raw["graph"],
            rerank=raw["rerank"],
            constraint=raw["constraint"],
            generation=raw["generation"],
            evaluation=raw["evaluation"],
            _raw=raw,
        )

    def ensure_dirs(self) -> None:
        """Create output and cache directories if they don't exist."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> Dict[str, Any]:
        return self._raw

    def dumps(self) -> str:
        return json.dumps(self._raw, indent=2, default=str)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _deep_update(base: dict, overlay: dict) -> None:
    """Recursively update *base* in-place with values from *overlay*."""
    for k, v in overlay.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v


def _resolve(root: Path, p: str) -> Path:
    """Resolve *p* relative to *root* if not already absolute."""
    path = Path(p)
    if not path.is_absolute():
        path = root / path
    return path
