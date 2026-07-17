"""Typed financial evidence path extraction and encoding.

FinPath-RGCN keeps the vanilla R-GCN backbone unchanged and adds a separate
path-aware branch.  This module contains the graph-side path extractor plus
small neural components for encoding and aggregating typed financial paths.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import torch
import torch.nn as nn


MATCH_KEYS = [
    "company_match",
    "year_match",
    "metric_match",
    "filing_match",
    "section_match",
]

CONFLICT_KEYS = [
    "company_conflict",
    "year_conflict",
    "metric_conflict",
    "filing_conflict",
]

PATH_FEATURE_KEYS = [
    "path_count",
    "matched_path_count",
    "conflict_path_count",
    "path_coverage_ratio",
    "company_path_exists",
    "year_path_exists",
    "metric_path_exists",
    "semantic_support_path_exists",
    "company_conflict_exists",
    "year_conflict_exists",
    "metric_conflict_exists",
]

METRIC_ALIASES = {
    "revenue": {
        "revenue", "revenues", "sales", "net sales", "net revenue", "total revenue",
        "data and access solutions rev", "rev",
    },
    "income": {
        "net income", "net earnings", "net loss", "earnings", "profit", "profits",
    },
    "operating_income": {
        "operating income", "operating earnings", "operating loss",
    },
    "gross_profit": {"gross profit", "gross margin"},
    "eps": {"eps", "earnings per share", "diluted eps", "basic eps"},
    "ebitda": {"ebitda", "adjusted ebitda", "ebit"},
    "cash_flow": {
        "cash flow", "operating cash flow", "free cash flow", "cash and cash equivalents",
    },
    "assets": {"total assets", "assets"},
    "liabilities": {"total liabilities", "liabilities"},
    "equity": {"total equity", "shareholders equity", "stockholders equity", "equity"},
    "debt": {"long-term debt", "short-term debt", "debt"},
    "expenses": {
        "operating expenses", "r&d expenses", "research and development",
        "sg&a", "selling general and administrative", "cost of revenue", "cogs",
    },
}

_METRIC_ALIAS_LOOKUP = {
    alias: canonical
    for canonical, aliases in METRIC_ALIASES.items()
    for alias in aliases
}


@dataclass
class FinancialPath:
    """A typed financial evidence path ending at a candidate chunk."""

    path_nodes: List[str]
    path_relations: List[str]
    path_node_types: List[str]
    path_type: str
    target_chunk_id: str
    match_flags: Dict[str, int] = field(default_factory=dict)
    conflict_flags: Dict[str, int] = field(default_factory=dict)
    path_length: int = 0

    def __post_init__(self) -> None:
        if not self.path_length:
            self.path_length = len(self.path_relations)
        self.match_flags = {k: int(self.match_flags.get(k, 0)) for k in MATCH_KEYS}
        self.conflict_flags = {k: int(self.conflict_flags.get(k, 0)) for k in CONFLICT_KEYS}

    @property
    def has_match(self) -> bool:
        return any(self.match_flags.values())

    @property
    def has_conflict(self) -> bool:
        return any(self.conflict_flags.values())

    def to_string(self) -> str:
        pieces: List[str] = []
        for i, node in enumerate(self.path_nodes):
            pieces.append(node)
            if i < len(self.path_relations):
                pieces.append(f"-[{self.path_relations[i]}]->")
        return " ".join(pieces)


def _as_nx_graph(graph: Any) -> nx.MultiDiGraph:
    return graph.graph if hasattr(graph, "graph") and isinstance(graph.graph, nx.MultiDiGraph) else graph


def _node_types(graph: Any) -> Dict[str, str]:
    return getattr(graph, "node_types", {})


def _node_type(graph: Any, node_id: str) -> str:
    types = _node_types(graph)
    if node_id in types:
        return types[node_id]
    nxg = _as_nx_graph(graph)
    return str(nxg.nodes.get(node_id, {}).get("node_type", _infer_node_type(node_id)))


def _infer_node_type(node_id: str) -> str:
    if "::" in node_id:
        return node_id.split("::", 1)[0]
    return "chunk"


def _norm_text(value: Any) -> str:
    return str(value or "").strip().lower()


def _norm_set(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, (list, tuple, set)):
        return {_norm_text(v) for v in value if _norm_text(v)}
    text = _norm_text(value)
    return {text} if text else set()


def canonical_metric(value: Any) -> str:
    text = _norm_text(value).replace("_", " ")
    text = re.sub(r"\s+", " ", text)
    if text in _METRIC_ALIAS_LOOKUP:
        return _METRIC_ALIAS_LOOKUP[text]
    for alias, canonical in _METRIC_ALIAS_LOOKUP.items():
        if alias and alias in text:
            return canonical
    return text


def canonical_metric_set(values: Iterable[Any]) -> set[str]:
    return {canonical_metric(v) for v in values if canonical_metric(v)}


def _metric_aliases_in_text(text: str) -> set[str]:
    norm = _norm_text(text).replace("_", " ")
    found: set[str] = set()
    for alias in _METRIC_ALIAS_LOOKUP:
        if re.search(rf"\b{re.escape(alias)}\b", norm):
            found.add(alias)
    return found


def expand_years_from_text(text: str) -> set[str]:
    """Extract standalone years and compact ranges like 2021-23."""

    text = str(text or "")
    years = {m.group(0) for m in re.finditer(r"\b(?:19|20)\d{2}\b", text)}
    for m in re.finditer(r"\b((?:19|20)\d{2})\s*(?:-|to|through|–|—)\s*(\d{2}|(?:19|20)\d{2})\b", text, re.I):
        start = int(m.group(1))
        end_raw = m.group(2)
        end = int(end_raw) if len(end_raw) == 4 else int(str(start)[:2] + end_raw)
        if start <= end and end - start <= 10:
            years.update(str(y) for y in range(start, end + 1))
    return years


def _entity_value(node_id: str) -> str:
    return _norm_text(node_id.split("::", 1)[1] if "::" in node_id else node_id)


def _company_from_filing_node(nxg: nx.MultiDiGraph, filing_id: str) -> str:
    attrs = nxg.nodes.get(filing_id, {})
    if attrs.get("company"):
        return _norm_text(attrs.get("company"))
    raw = filing_id.split("::", 1)[1] if "::" in filing_id else filing_id
    return _norm_text(raw.split("_", 1)[0])


def _edge_type(data: Any) -> str:
    if isinstance(data, dict):
        return str(data.get("edge_type", ""))
    return ""


def _out_edges_of_type(nxg: nx.MultiDiGraph, node_id: str, edge_type: str) -> Iterable[Tuple[str, str]]:
    if node_id not in nxg:
        return []
    out: List[Tuple[str, str]] = []
    for _, dst, data in nxg.out_edges(node_id, data=True):
        if _edge_type(data) == edge_type:
            out.append((dst, edge_type))
    return out


def _in_edges_of_type(nxg: nx.MultiDiGraph, node_id: str, edge_type: str) -> Iterable[Tuple[str, str]]:
    if node_id not in nxg:
        return []
    out: List[Tuple[str, str]] = []
    for src, _, data in nxg.in_edges(node_id, data=True):
        if _edge_type(data) == edge_type:
            out.append((src, edge_type))
    return out


class FinancialPathExtractor:
    """Extract typed evidence paths for query entities and candidate chunks."""

    def __init__(self, max_paths_per_chunk: int = 8, max_path_len: int = 4):
        self.max_paths_per_chunk = max_paths_per_chunk
        self.max_path_len = max_path_len

    def extract_paths(
        self,
        graph: Any,
        candidate_chunk_ids: Sequence[str],
        query_entities: Optional[Dict[str, Any]] = None,
        max_paths_per_chunk: Optional[int] = None,
        max_path_len: Optional[int] = None,
    ) -> Dict[str, List[FinancialPath]]:
        nxg = _as_nx_graph(graph)
        max_paths = max_paths_per_chunk or self.max_paths_per_chunk
        max_len = max_path_len or self.max_path_len
        entities = self._normalise_query_entities(query_entities or {})
        out: Dict[str, List[FinancialPath]] = {}

        for cid in candidate_chunk_ids:
            paths: List[FinancialPath] = []
            if cid in nxg:
                paths.extend(self._company_filing_section_paths(graph, cid, entities))
                paths.extend(self._filing_section_paths(graph, cid, entities))
                paths.extend(self._chunk_entity_paths(graph, cid, entities, "metric"))
                paths.extend(self._chunk_entity_paths(graph, cid, entities, "year"))
                paths.extend(self._semantic_support_paths(graph, cid, entities))
                paths.extend(self._conflict_paths(graph, cid, entities))
            paths = [p for p in paths if p.path_length <= max_len]
            out[cid] = paths[:max_paths]
        return out

    @staticmethod
    def _normalise_query_entities(query_entities: Dict[str, Any]) -> Dict[str, set[str]]:
        companies = _norm_set(query_entities.get("company")) | _norm_set(query_entities.get("companies"))
        years = _norm_set(query_entities.get("year")) | _norm_set(query_entities.get("years"))
        years |= expand_years_from_text(str(query_entities.get("query_text", "")))
        metrics_raw = _norm_set(query_entities.get("metric")) | _norm_set(query_entities.get("metrics"))
        metrics_raw |= _metric_aliases_in_text(str(query_entities.get("query_text", "")))
        metrics = canonical_metric_set(metrics_raw)
        filings = _norm_set(query_entities.get("filing_type")) | _norm_set(query_entities.get("filing_types"))
        sections = _norm_set(query_entities.get("section_hint")) | _norm_set(query_entities.get("sections"))
        return {
            "company": companies,
            "year": years,
            "metric": metrics,
            "filing": filings,
            "section": sections,
        }

    def _make_path(
        self,
        graph: Any,
        nodes: List[str],
        relations: List[str],
        path_type: str,
        target_chunk_id: str,
        match_flags: Optional[Dict[str, int]] = None,
        conflict_flags: Optional[Dict[str, int]] = None,
    ) -> FinancialPath:
        return FinancialPath(
            path_nodes=nodes,
            path_relations=relations,
            path_node_types=[_node_type(graph, n) for n in nodes],
            path_type=path_type,
            target_chunk_id=target_chunk_id,
            match_flags=match_flags or {},
            conflict_flags=conflict_flags or {},
            path_length=len(relations),
        )

    def _company_filing_section_paths(
        self, graph: Any, cid: str, entities: Dict[str, set[str]]
    ) -> List[FinancialPath]:
        if not entities["company"]:
            return []
        nxg = _as_nx_graph(graph)
        paths: List[FinancialPath] = []
        for section, _ in _in_edges_of_type(nxg, cid, "section-has-chunk"):
            for filing, _ in _in_edges_of_type(nxg, section, "filing-has-section"):
                filing_attrs = nxg.nodes.get(filing, {})
                filing_match = int(
                    not entities["filing"]
                    or _norm_text(filing_attrs.get("filing_type")) in entities["filing"]
                    or any(f in _norm_text(filing) for f in entities["filing"])
                )
                section_match = int(
                    not entities["section"]
                    or _entity_value(section) in entities["section"]
                    or any(s in _entity_value(section) for s in entities["section"])
                )
                for company, _ in _in_edges_of_type(nxg, filing, "company-has-filing"):
                    if _entity_value(company) not in entities["company"]:
                        continue
                    paths.append(
                        self._make_path(
                            graph,
                            [company, filing, section, cid],
                            ["company-has-filing", "filing-has-section", "section-has-chunk"],
                            "company_filing_section_chunk",
                            cid,
                            {
                                "company_match": 1,
                                "filing_match": filing_match,
                                "section_match": section_match,
                            },
                        )
                    )
        if paths:
            return paths

        # Fallback for corpora where chunks are linked directly to filings but
        # section nodes are missing or too coarse.
        for filing, _ in _out_edges_of_type(nxg, cid, "chunk-belongs-to-filing"):
            if _company_from_filing_node(nxg, filing) not in entities["company"]:
                continue
            filing_attrs = nxg.nodes.get(filing, {})
            filing_match = int(
                not entities["filing"]
                or _norm_text(filing_attrs.get("filing_type")) in entities["filing"]
                or any(f in _norm_text(filing) for f in entities["filing"])
            )
            for company, _ in _in_edges_of_type(nxg, filing, "company-has-filing"):
                if _entity_value(company) not in entities["company"]:
                    continue
                paths.append(
                    self._make_path(
                        graph,
                        [company, filing, cid],
                        ["company-has-filing", "chunk-belongs-to-filing"],
                        "company_filing_chunk",
                        cid,
                        {"company_match": 1, "filing_match": filing_match},
                    )
                )
        return paths

    def _filing_section_paths(
        self, graph: Any, cid: str, entities: Dict[str, set[str]]
    ) -> List[FinancialPath]:
        nxg = _as_nx_graph(graph)
        paths: List[FinancialPath] = []
        for section, _ in _in_edges_of_type(nxg, cid, "section-has-chunk"):
            for filing, _ in _in_edges_of_type(nxg, section, "filing-has-section"):
                filing_attrs = nxg.nodes.get(filing, {})
                filing_match = int(
                    not entities["filing"]
                    or _norm_text(filing_attrs.get("filing_type")) in entities["filing"]
                    or any(f in _norm_text(filing) for f in entities["filing"])
                )
                section_match = int(
                    not entities["section"]
                    or _entity_value(section) in entities["section"]
                    or any(s in _entity_value(section) for s in entities["section"])
                )
                paths.append(
                    self._make_path(
                        graph,
                        [filing, section, cid],
                        ["filing-has-section", "section-has-chunk"],
                        "filing_section_chunk",
                        cid,
                        {"filing_match": filing_match, "section_match": section_match},
                    )
                )
        if paths:
            return paths

        for filing, _ in _out_edges_of_type(nxg, cid, "chunk-belongs-to-filing"):
            filing_attrs = nxg.nodes.get(filing, {})
            filing_match = int(
                not entities["filing"]
                or _norm_text(filing_attrs.get("filing_type")) in entities["filing"]
                or any(f in _norm_text(filing) for f in entities["filing"])
            )
            paths.append(
                self._make_path(
                    graph,
                    [filing, cid],
                    ["chunk-belongs-to-filing"],
                    "filing_chunk",
                    cid,
                    {"filing_match": filing_match},
                )
            )
        return paths

    def _chunk_entity_paths(
        self, graph: Any, cid: str, entities: Dict[str, set[str]], entity_type: str
    ) -> List[FinancialPath]:
        key = "metric" if entity_type == "metric" else "year"
        if not entities[key]:
            return []
        relation = f"chunk-mentions-{key}"
        nxg = _as_nx_graph(graph)
        paths: List[FinancialPath] = []
        for node, _ in _out_edges_of_type(nxg, cid, relation):
            if _node_type(graph, node) != key:
                continue
            value = _entity_value(node)
            comparable_value = canonical_metric(value) if key == "metric" else value
            if comparable_value not in entities[key]:
                continue
            paths.append(
                self._make_path(
                    graph,
                    [cid, node],
                    [relation],
                    f"chunk_{key}",
                    cid,
                    {f"{key}_match": 1},
                )
            )
        return paths

    def _semantic_support_paths(
        self, graph: Any, cid: str, entities: Dict[str, set[str]]
    ) -> List[FinancialPath]:
        nxg = _as_nx_graph(graph)
        paths: List[FinancialPath] = []
        neighbors: List[str] = []
        neighbors.extend([dst for dst, _ in _out_edges_of_type(nxg, cid, "semantic-similar")])
        neighbors.extend([src for src, _ in _in_edges_of_type(nxg, cid, "semantic-similar")])
        for neighbor in dict.fromkeys(neighbors):
            flags = self._chunk_match_flags(graph, neighbor, entities)
            if not any(flags.values()):
                continue
            paths.append(
                self._make_path(
                    graph,
                    [cid, neighbor],
                    ["semantic-similar"],
                    "semantic_support",
                    cid,
                    flags,
                )
            )
        return paths

    def _conflict_paths(
        self, graph: Any, cid: str, entities: Dict[str, set[str]]
    ) -> List[FinancialPath]:
        nxg = _as_nx_graph(graph)
        paths: List[FinancialPath] = []
        if entities["year"]:
            for node, _ in _out_edges_of_type(nxg, cid, "chunk-mentions-year"):
                if _node_type(graph, node) == "year" and _entity_value(node) not in entities["year"]:
                    paths.append(
                        self._make_path(
                            graph,
                            [cid, node],
                            ["chunk-mentions-year"],
                            "year_conflict",
                            cid,
                            conflict_flags={"year_conflict": 1},
                        )
                    )
        if entities["metric"]:
            for node, _ in _out_edges_of_type(nxg, cid, "chunk-mentions-metric"):
                if _node_type(graph, node) == "metric" and canonical_metric(_entity_value(node)) not in entities["metric"]:
                    paths.append(
                        self._make_path(
                            graph,
                            [cid, node],
                            ["chunk-mentions-metric"],
                            "metric_conflict",
                            cid,
                            conflict_flags={"metric_conflict": 1},
                        )
                    )
        if entities["company"]:
            for section, _ in _in_edges_of_type(nxg, cid, "section-has-chunk"):
                for filing, _ in _in_edges_of_type(nxg, section, "filing-has-section"):
                    for company, _ in _in_edges_of_type(nxg, filing, "company-has-filing"):
                        if _entity_value(company) in entities["company"]:
                            continue
                        paths.append(
                            self._make_path(
                                graph,
                                [company, filing, section, cid],
                                ["company-has-filing", "filing-has-section", "section-has-chunk"],
                                "company_conflict",
                                cid,
                                conflict_flags={"company_conflict": 1},
                            )
                        )
        return paths

    def _chunk_match_flags(
        self, graph: Any, cid: str, entities: Dict[str, set[str]]
    ) -> Dict[str, int]:
        nxg = _as_nx_graph(graph)
        flags = {k: 0 for k in MATCH_KEYS}
        if cid not in nxg:
            return flags
        if entities["year"]:
            for node, _ in _out_edges_of_type(nxg, cid, "chunk-mentions-year"):
                flags["year_match"] = int(flags["year_match"] or _entity_value(node) in entities["year"])
        if entities["metric"]:
            for node, _ in _out_edges_of_type(nxg, cid, "chunk-mentions-metric"):
                flags["metric_match"] = int(flags["metric_match"] or canonical_metric(_entity_value(node)) in entities["metric"])
        attrs = nxg.nodes.get(cid, {})
        if entities["company"]:
            flags["company_match"] = int(_norm_text(attrs.get("company")) in entities["company"])
            if not flags["company_match"]:
                for filing, _ in _out_edges_of_type(nxg, cid, "chunk-belongs-to-filing"):
                    flags["company_match"] = int(_company_from_filing_node(nxg, filing) in entities["company"])
                    if flags["company_match"]:
                        break
        return flags


def compute_path_features(paths: Sequence[FinancialPath], max_paths: int = 8) -> np.ndarray:
    """Return normalized/binary chunk-level path features."""

    n = len(paths)
    matched = sum(1 for p in paths if p.has_match)
    conflicts = sum(1 for p in paths if p.has_conflict)
    denom = float(max(max_paths, 1))
    values = {
        "path_count": min(n / denom, 1.0),
        "matched_path_count": min(matched / denom, 1.0),
        "conflict_path_count": min(conflicts / denom, 1.0),
        "path_coverage_ratio": matched / max(n, 1),
        "company_path_exists": float(any(p.match_flags.get("company_match", 0) for p in paths)),
        "year_path_exists": float(any(p.match_flags.get("year_match", 0) for p in paths)),
        "metric_path_exists": float(any(p.match_flags.get("metric_match", 0) for p in paths)),
        "semantic_support_path_exists": float(any(p.path_type == "semantic_support" for p in paths)),
        "company_conflict_exists": float(any(p.conflict_flags.get("company_conflict", 0) for p in paths)),
        "year_conflict_exists": float(any(p.conflict_flags.get("year_conflict", 0) for p in paths)),
        "metric_conflict_exists": float(any(p.conflict_flags.get("metric_conflict", 0) for p in paths)),
    }
    return np.array([values[k] for k in PATH_FEATURE_KEYS], dtype=np.float32)


def build_path_vocab(paths_by_chunk: Dict[str, Sequence[FinancialPath]]) -> Dict[str, Dict[str, int]]:
    """Build stable vocabularies for path relation, node type, and path type IDs."""

    rels = set()
    node_types = set()
    path_types = set()
    for paths in paths_by_chunk.values():
        for path in paths:
            rels.update(path.path_relations)
            node_types.update(path.path_node_types)
            path_types.add(path.path_type)

    def ordered_vocab(values: set[str], specials: List[str]) -> Dict[str, int]:
        vocab = {token: idx for idx, token in enumerate(specials)}
        for value in sorted(values):
            if value not in vocab:
                vocab[value] = len(vocab)
        return vocab

    return {
        "relation": ordered_vocab(rels, ["<pad>", "<unk>"]),
        "node_type": ordered_vocab(node_types, ["<pad>", "<unk>"]),
        "path_type": ordered_vocab(path_types, ["<pad>", "<unk>", "<none>"]),
    }


def tensorize_paths(
    paths: Sequence[FinancialPath],
    vocab: Dict[str, Dict[str, int]],
    max_paths: int = 8,
    max_path_len: int = 4,
) -> Dict[str, torch.Tensor]:
    """Convert variable-length paths into padded tensors."""

    rel_pad = vocab["relation"].get("<pad>", 0)
    node_pad = vocab["node_type"].get("<pad>", 0)
    path_pad = vocab["path_type"].get("<pad>", 0)
    rel_unk = vocab["relation"].get("<unk>", rel_pad)
    node_unk = vocab["node_type"].get("<unk>", node_pad)
    path_unk = vocab["path_type"].get("<unk>", path_pad)

    rel_ids = torch.full((max_paths, max_path_len), rel_pad, dtype=torch.long)
    src_type_ids = torch.full((max_paths, max_path_len), node_pad, dtype=torch.long)
    dst_type_ids = torch.full((max_paths, max_path_len), node_pad, dtype=torch.long)
    path_type_ids = torch.full((max_paths,), path_pad, dtype=torch.long)
    flag_feats = torch.zeros((max_paths, len(MATCH_KEYS) + len(CONFLICT_KEYS)), dtype=torch.float32)
    path_mask = torch.zeros((max_paths,), dtype=torch.bool)
    step_mask = torch.zeros((max_paths, max_path_len), dtype=torch.bool)

    for i, path in enumerate(paths[:max_paths]):
        path_mask[i] = True
        path_type_ids[i] = vocab["path_type"].get(path.path_type, path_unk)
        for j, rel in enumerate(path.path_relations[:max_path_len]):
            rel_ids[i, j] = vocab["relation"].get(rel, rel_unk)
            src_type = path.path_node_types[j] if j < len(path.path_node_types) else "<unk>"
            dst_type = path.path_node_types[j + 1] if j + 1 < len(path.path_node_types) else "<unk>"
            src_type_ids[i, j] = vocab["node_type"].get(src_type, node_unk)
            dst_type_ids[i, j] = vocab["node_type"].get(dst_type, node_unk)
            step_mask[i, j] = True
        flag_feats[i] = torch.tensor(
            [path.match_flags.get(k, 0) for k in MATCH_KEYS]
            + [path.conflict_flags.get(k, 0) for k in CONFLICT_KEYS],
            dtype=torch.float32,
        )

    return {
        "relation_ids": rel_ids,
        "src_node_type_ids": src_type_ids,
        "dst_node_type_ids": dst_type_ids,
        "path_type_ids": path_type_ids,
        "flag_features": flag_feats,
        "path_mask": path_mask,
        "step_mask": step_mask,
    }


class LearnablePathEncoder(nn.Module):
    """Encode short typed financial evidence paths with a lightweight GRU."""

    def __init__(
        self,
        num_relations: int,
        num_node_types: int,
        num_path_types: int,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.1,
        flag_dim: int = len(MATCH_KEYS) + len(CONFLICT_KEYS),
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.relation_emb = nn.Embedding(num_relations, hidden_dim, padding_idx=0)
        self.node_type_emb = nn.Embedding(num_node_types, hidden_dim, padding_idx=0)
        self.path_type_emb = nn.Embedding(num_path_types, hidden_dim, padding_idx=0)
        self.gru = nn.GRU(
            hidden_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.out = nn.Sequential(
            nn.Linear(hidden_dim * 2 + flag_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        relation_ids: torch.Tensor,
        src_node_type_ids: torch.Tensor,
        dst_node_type_ids: torch.Tensor,
        path_type_ids: torch.Tensor,
        flag_features: torch.Tensor,
        step_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        tokens = (
            self.relation_emb(relation_ids)
            + self.node_type_emb(src_node_type_ids)
            + self.node_type_emb(dst_node_type_ids)
        )
        if step_mask is not None:
            tokens = tokens * step_mask.unsqueeze(-1).float()
        _, hidden = self.gru(tokens)
        seq_emb = hidden[-1]
        type_emb = self.path_type_emb(path_type_ids)
        return self.out(torch.cat([seq_emb, type_emb, flag_features.float()], dim=-1))


class PathAggregator(nn.Module):
    """Query-conditioned attention over multiple path embeddings for one chunk."""

    def __init__(self, hidden_dim: int = 128, query_dim: Optional[int] = None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.query_proj = nn.Linear(query_dim or hidden_dim, hidden_dim)
        self.path_proj = nn.Linear(hidden_dim, hidden_dim)
        self.attn = nn.Linear(hidden_dim, 1, bias=False)
        self.no_path = nn.Parameter(torch.zeros(hidden_dim))

    def forward(
        self,
        path_embeddings: torch.Tensor,
        query_embedding: torch.Tensor,
        path_mask: Optional[torch.Tensor] = None,
        path_type_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.unsqueeze(0)
        valid_mask = path_mask.bool() if path_mask is not None else torch.ones(
            path_embeddings.shape[0], dtype=torch.bool, device=path_embeddings.device
        )
        if path_embeddings.numel() == 0 or not bool(valid_mask.any().item()):
            return self.no_path, {
                "attention_weights": [],
                "max_attention_index": None,
                "max_attention_path_type": None,
            }

        q = self.query_proj(query_embedding.to(path_embeddings.device))[0]
        logits = self.attn(torch.tanh(self.path_proj(path_embeddings) + q)).squeeze(-1)
        logits = logits.masked_fill(~valid_mask.to(logits.device), -1e9)
        weights = torch.softmax(logits, dim=0)
        repr_vec = torch.sum(weights.unsqueeze(-1) * path_embeddings, dim=0)
        max_idx = int(torch.argmax(weights).item())
        max_type = None
        if path_type_ids is not None:
            max_type = int(path_type_ids[max_idx].item())
        return repr_vec, {
            "attention_weights": weights.detach().cpu().tolist(),
            "max_attention_index": max_idx,
            "max_attention_path_type": max_type,
        }
