"""DCF-GNN: Dual-Channel Financial Evidence Graph Neural Reranker.

DCF-GNN keeps the vanilla R-GCN baseline untouched and adds a separate
dual-channel backbone:

* Structural Constraint Channel for precise financial graph relations.
* Conflict-Suppressed Semantic Channel for weak semantic/entity-similarity
  relations, with query-level conflict downweighting.
* Query-type-aware channel fusion to shift weight between the two channels.

The public reranker API mirrors :mod:`feg_rag.rerank.rgcn` so existing Table 1
experiments can train and evaluate it as another GNN reranker.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from feg_rag.data.chunker import Chunk
from feg_rag.graph.builder import FinancialEvidenceGraph
from feg_rag.graph.entities import EntityExtractor
from feg_rag.rerank.query_features import (
    QUERY_FEATURE_DIM,
    build_query_augmented_features,
)
from feg_rag.rerank.scoring import normalise_score_map


IncidentEdgeMap = Dict[str, List[Tuple[str, str, float]]]


STRUCTURAL_RELATION_HINTS = (
    "company-has-filing",
    "filing-has-section",
    "section-has-chunk",
    "chunk-belongs-to-filing",
    "chunk-mentions-metric",
    "chunk-mentions-year",
    "filing",
    "section",
    "metric",
    "year",
)

SEMANTIC_RELATION_HINTS = (
    "semantic-similar",
    "same-company",
    "same-year",
    "same-metric",
    "same-section",
    "same-filing-year",
    "near-duplicate",
    "duplicate",
    "similar",
)

QUERY_TYPE_NAMES = ("numeric_fact", "comparison_trend", "explanation", "general")

_EXTRACTOR = EntityExtractor()


def split_relation_channels(relation_map: Dict[str, int]) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Split relation types into structural and semantic channel maps.

    The function is intentionally conservative: known weak/similarity edges go
    to the semantic channel, while precise entity/filing/section edges go to
    the structural channel. Unknown relations default to structural so they
    cannot silently become noisy semantic shortcuts.
    """
    structural: Dict[str, int] = {}
    semantic: Dict[str, int] = {}
    for rel in sorted(relation_map):
        rel_l = rel.lower()
        if any(h in rel_l for h in SEMANTIC_RELATION_HINTS):
            semantic[rel] = len(semantic)
        elif any(h in rel_l for h in STRUCTURAL_RELATION_HINTS):
            structural[rel] = len(structural)
        else:
            structural[rel] = len(structural)
    if not structural:
        structural["__empty_structural__"] = 0
    if not semantic:
        semantic["__empty_semantic__"] = 0
    return structural, semantic


def infer_query_type_features(query: str) -> np.ndarray:
    """Return a 4-dim coarse query-type vector.

    Order: numeric fact, comparison/trend, explanation/risk/disclosure,
    general. The vector is multi-hot but always has at least one active value.
    """
    q = query.lower()
    feats = np.zeros(len(QUERY_TYPE_NAMES), dtype=np.float32)

    has_year = bool(_EXTRACTOR.extract_years(query))
    has_metric = bool(_EXTRACTOR.extract_metrics(query))
    has_number = any(ch.isdigit() for ch in query)
    if has_year or has_metric or has_number:
        feats[0] = 1.0

    if any(k in q for k in (
        "compare", "comparison", "versus", " vs ", "change", "increase",
        "decrease", "growth", "decline", "trend", "delta", "between",
        "year over year", "yoy", "qoq",
    )):
        feats[1] = 1.0

    if any(k in q for k in (
        "why", "explain", "reason", "risk", "disclosure", "describe",
        "discussion", "management", "factor", "outlook",
    )):
        feats[2] = 1.0

    if not feats.any():
        feats[3] = 1.0
    return feats


def financial_match_features(query: str, chunk: Optional[Chunk]) -> np.ndarray:
    """Build simple query-candidate financial match/conflict features.

    Columns:
    company_match, year_match, metric_match, filing_match, section_match,
    company_conflict, year_conflict, metric_conflict.
    """
    feats = np.zeros(8, dtype=np.float32)
    if chunk is None:
        return feats

    q_companies = {c.lower() for c in _EXTRACTOR.extract_companies(query)}
    q_years = _EXTRACTOR.extract_years(query)
    q_metrics = {_norm_metric(m) for m in _EXTRACTOR.extract_metrics(query)}
    q_filings = {f.upper() for f in _EXTRACTOR.extract_filing_types(query)}
    q_sections = _extract_section_hints(query)

    company = (chunk.company or "").lower()
    if q_companies and company:
        feats[0] = 1.0 if any(qc in company or company in qc for qc in q_companies) else 0.0
        feats[5] = 1.0 - feats[0]

    chunk_years = _EXTRACTOR.extract_years(chunk.text)
    if chunk.filing_year:
        chunk_years = set(chunk_years) | {str(chunk.filing_year)}
    if q_years and chunk_years:
        feats[1] = 1.0 if bool(q_years & chunk_years) else 0.0
        feats[6] = 1.0 - feats[1]

    chunk_metrics = {_norm_metric(m) for m in _EXTRACTOR.extract_metrics(chunk.text)}
    if q_metrics and chunk_metrics:
        feats[2] = 1.0 if bool(q_metrics & chunk_metrics) else 0.0
        feats[7] = 1.0 - feats[2]

    if q_filings and chunk.filing_type:
        feats[3] = 1.0 if chunk.filing_type.upper() in q_filings else 0.0

    section = (chunk.section or "").lower()
    if q_sections and section:
        feats[4] = 1.0 if any(s in section for s in q_sections) else 0.0

    return feats


class DCFChannelLayer(nn.Module):
    """Relation-aware channel layer with one weight per channel relation."""

    def __init__(self, in_dim: int, out_dim: int, num_relations: int, dropout: float = 0.3):
        super().__init__()
        self.self_loop = nn.Linear(in_dim, out_dim, bias=False)
        self.rel_weights = nn.ModuleList([
            nn.Linear(in_dim, out_dim, bias=False)
            for _ in range(max(num_relations, 1))
        ])
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj_list: List[torch.Tensor]) -> torch.Tensor:
        out = self.self_loop(x)
        for r, adj in enumerate(adj_list):
            if r >= len(self.rel_weights) or adj.numel() == 0:
                continue
            out = out + adj @ self.rel_weights[r](x)
        out = F.relu(out + self.bias)
        return self.dropout(out)


class DCFGNNReranker(nn.Module):
    """Dual-channel financial evidence GNN backbone."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        out_dim: int = 64,
        num_structural_relations: int = 1,
        num_semantic_relations: int = 1,
        match_feature_dim: int = 8,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.structural1 = DCFChannelLayer(in_dim, hidden_dim, num_structural_relations, dropout)
        self.structural2 = DCFChannelLayer(hidden_dim, out_dim, num_structural_relations, dropout)
        self.semantic1 = DCFChannelLayer(in_dim, hidden_dim, num_semantic_relations, dropout)
        self.semantic2 = DCFChannelLayer(hidden_dim, out_dim, num_semantic_relations, dropout)

        self.channel_gate = nn.Sequential(
            nn.Linear(len(QUERY_TYPE_NAMES), 16),
            nn.ReLU(),
            nn.Linear(16, 2),
        )
        self.scorer = nn.Sequential(
            nn.Linear(out_dim + match_feature_dim + 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        structural_adj: List[torch.Tensor],
        semantic_adj: List[torch.Tensor],
        query_type_features: torch.Tensor,
        match_features: torch.Tensor,
        return_intermediate: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        h_s = self.structural2(self.structural1(x, structural_adj), structural_adj)
        h_m = self.semantic2(self.semantic1(x, semantic_adj), semantic_adj)

        gate_logits = self.channel_gate(query_type_features)
        gate = torch.softmax(gate_logits, dim=-1)
        h = gate[:, :1] * h_s + gate[:, 1:] * h_m

        scores = self.scorer(torch.cat([h, match_features, gate], dim=-1))
        if not return_intermediate:
            return scores
        return scores, {
            "structural_gate": gate[:, 0],
            "semantic_gate": gate[:, 1],
            "structural_embedding": h_s,
            "semantic_embedding": h_m,
        }


class DCFRerankDataset(Dataset):
    """Pairwise ranking dataset for DCF-GNN."""

    def __init__(
        self,
        samples: List[Dict],
        graph: FinancialEvidenceGraph,
        features: Dict[str, np.ndarray],
        relation_map: Optional[Dict[str, int]] = None,
        chunk_lookup: Optional[Dict[str, Chunk]] = None,
    ):
        self.samples = samples
        self.graph = graph
        self.features = features
        self._chunk_lookup = chunk_lookup or _build_chunk_lookup(graph)
        self._incident_edges = _build_incident_edges(graph)

        if relation_map is None:
            etypes = {
                etype
                for _u, _v, _k, etype in graph.graph.edges(keys=True, data="edge_type")
            }
            self.relation_map = {et: i for i, et in enumerate(sorted(etypes))}
        else:
            self.relation_map = relation_map
        self.structural_relation_map, self.semantic_relation_map = split_relation_channels(
            self.relation_map
        )

    @property
    def num_structural_relations(self) -> int:
        return max(self.structural_relation_map.values()) + 1 if self.structural_relation_map else 1

    @property
    def num_semantic_relations(self) -> int:
        return max(self.semantic_relation_map.values()) + 1 if self.semantic_relation_map else 1

    @property
    def chunk_lookup(self) -> Dict[str, Chunk]:
        return self._chunk_lookup

    @property
    def incident_edges(self) -> IncidentEdgeMap:
        return self._incident_edges

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        pos_id = s.get("positive", s.get("chunk_id", ""))
        neg_id = s.get("negative", "")
        question = s.get("question", "")

        sub_nodes: Set[str] = set()
        for seed in [pos_id, neg_id]:
            if seed:
                sub_nodes.add(seed)
                sub_nodes |= self.graph.get_chunk_neighbors(seed, max_hops=2)

        node_list = list(sub_nodes)
        node2idx = {n: i for i, n in enumerate(node_list)}
        n_nodes = len(node_list)

        base_dim = next(iter(self.features.values())).shape[0]
        x_base = np.zeros((n_nodes, base_dim), dtype=np.float32)
        for node_id in node_list:
            if node_id in self.features:
                x_base[node2idx[node_id]] = self.features[node_id]

        retrieval_scores = s.get("retrieval_scores", None) or {}
        graph_scores = s.get("graph_scores", None) or {}
        x_aug = build_query_augmented_features(
            self.features,
            node_list,
            question,
            chunk_lookup=self._chunk_lookup,
            retrieval_scores=retrieval_scores,
            graph_scores=graph_scores,
        )
        x = np.concatenate([x_base, x_aug], axis=1)

        qtype = infer_query_type_features(question)
        qtype_nodes = np.repeat(qtype[None, :], n_nodes, axis=0).astype(np.float32)

        match_feats = np.zeros((n_nodes, 8), dtype=np.float32)
        for node_id in node_list:
            match_feats[node2idx[node_id]] = financial_match_features(
                question, self._chunk_lookup.get(node_id)
            )

        structural_adj, semantic_adj, downweight_stats = _build_channel_adj(
            self.graph,
            node2idx,
            question,
            self._chunk_lookup,
            self.structural_relation_map,
            self.semantic_relation_map,
            incident_edges=self._incident_edges,
        )

        return (
            torch.from_numpy(x),
            torch.stack([torch.from_numpy(a) for a in structural_adj]),
            torch.stack([torch.from_numpy(a) for a in semantic_adj]),
            torch.from_numpy(qtype_nodes),
            torch.from_numpy(match_feats),
            torch.tensor(node2idx.get(pos_id, 0), dtype=torch.long),
            torch.tensor(node2idx.get(neg_id, 0), dtype=torch.long),
            downweight_stats,
        )


class DCFGNNFusionReranker:
    """DCF-GNN reranker with candidate-level score fusion."""

    def __init__(
        self,
        model: DCFGNNReranker,
        structural_relation_map: Dict[str, int],
        semantic_relation_map: Dict[str, int],
        alpha: float = 0.35,
        beta: float = 0.10,
        gamma: float = 0.55,
        device: str = "cpu",
        chunk_lookup: Optional[Dict[str, Chunk]] = None,
        incident_edges: Optional[IncidentEdgeMap] = None,
    ):
        self.model = model.to(device)
        self.structural_relation_map = structural_relation_map
        self.semantic_relation_map = semantic_relation_map
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.device = device
        self._chunk_lookup = chunk_lookup
        self._incident_edges = incident_edges
        self._eval_cache_dir: Optional[Path] = None
        self.eval_cache_hits = 0
        self.eval_cache_misses = 0
        self.last_diagnostics: Dict[str, Any] = {}

    def set_eval_cache(self, cache_dir: Optional[str | Path]) -> None:
        """Enable persistent per-query eval tensor caching.

        The cache stores CPU numpy tensors for the expensive query/candidate
        subgraph build. It is intentionally model-agnostic: changing weights
        does not invalidate it, but changing candidates, query text, feature
        shape, or relation maps does.
        """
        if cache_dir is None:
            self._eval_cache_dir = None
            return
        self._eval_cache_dir = Path(cache_dir)
        self._eval_cache_dir.mkdir(parents=True, exist_ok=True)
        self.eval_cache_hits = 0
        self.eval_cache_misses = 0

    def fit(
        self,
        train_dataset: DCFRerankDataset,
        val_dataset: Optional[DCFRerankDataset] = None,
        epochs: int = 50,
        lr: float = 0.001,
        batch_size: int = 32,
        verbose: bool = True,
    ) -> List[float]:
        loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, collate_fn=_collate_dcf
        )
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        loss_fn = nn.MarginRankingLoss(margin=0.5)
        history: List[float] = []
        self.model.train()

        t_start = time.time()
        n_batches = len(loader)
        print(f"\n{'=' * 55}")
        print("  Training DCF-GNN Reranker")
        print(f"  Samples: {len(train_dataset)}  |  Epochs: {epochs}  |  "
              f"Batches/epoch: {n_batches}")
        print(f"  Structural rels: {len(self.structural_relation_map)}  |  "
              f"Semantic rels: {len(self.semantic_relation_map)}  |  "
              f"Batch size: {batch_size}  |  Device: {self.device}  |  LR: {lr}")
        print(f"{'=' * 55}")

        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_t0 = time.time()
            for batch in loader:
                x, s_adj, m_adj, qtype, match, pos_idx, neg_idx, _diag = batch
                x = x.to(self.device)
                s_adj = [a.to(self.device) for a in s_adj]
                m_adj = [a.to(self.device) for a in m_adj]
                qtype = qtype.to(self.device)
                match = match.to(self.device)
                pos_idx = pos_idx.to(self.device)
                neg_idx = neg_idx.to(self.device)

                scores = self.model(x, s_adj, m_adj, qtype, match).squeeze(-1)
                pos_scores = scores[pos_idx]
                neg_scores = scores[neg_idx]
                target = torch.ones_like(pos_scores)
                loss = loss_fn(pos_scores, neg_scores, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / max(n_batches, 1)
            history.append(avg_loss)

            pct_done = (epoch + 1) / epochs
            bar_len = 20
            filled = int(pct_done * bar_len)
            bar = "#" * filled + "-" * (bar_len - filled)
            delta_str = ""
            if epoch > 0:
                delta = avg_loss - history[epoch - 1]
                sign = "v" if delta < 0 else "^"
                delta_str = f"  d={sign}{abs(delta):.4f}"
            elapsed = time.time() - t_start
            epoch_time = time.time() - epoch_t0
            eta = elapsed / (epoch + 1) * (epochs - epoch - 1) if epoch + 1 < epochs else 0
            if verbose:
                print(
                    f"  [{bar}] {pct_done:3.0%}  |  "
                    f"Epoch {epoch+1:>3}/{epochs}  |  "
                    f"loss={avg_loss:.4f}{delta_str}"
                    f"  |  {epoch_time:.1f}s/ep  |  "
                    f"elapsed={elapsed:.0f}s  |  eta={eta:.0f}s"
                )

        total_time = time.time() - t_start
        print(f"{'-' * 55}")
        print(f"  Training finished in {total_time:.1f}s")
        print(f"  Initial loss: {history[0]:.4f}  -->  Final loss: {history[-1]:.4f}"
              f"  (d: {history[-1] - history[0]:+.4f})")
        print(f"{'=' * 55}\n")
        return history

    def rerank(
        self,
        query: str,
        candidate_chunks: List[Tuple[Chunk, float]],
        graph: FinancialEvidenceGraph,
        features: Dict[str, np.ndarray],
        ppr_scores: Optional[Dict[str, float]] = None,
    ) -> List[Tuple[Chunk, float]]:
        self.model.eval()
        candidate_ids = [c.chunk_id for c, _ in candidate_chunks]
        sub_nodes = set(candidate_ids)
        for cid in candidate_ids:
            sub_nodes |= graph.get_chunk_neighbors(cid, max_hops=1)

        node_list = list(sub_nodes)
        node2idx = {n: i for i, n in enumerate(node_list)}
        n_nodes = len(node_list)

        if self._chunk_lookup is None:
            self._chunk_lookup = _build_chunk_lookup(graph)
        if self._incident_edges is None:
            self._incident_edges = _build_incident_edges(graph)
        chunk_lookup = dict(self._chunk_lookup)
        for chunk, _ in candidate_chunks:
            chunk_lookup[chunk.chunk_id] = chunk

        retrieval_scores = {c.chunk_id: float(s) for c, s in candidate_chunks}
        base_dim = next(iter(features.values())).shape[0]
        cache_key = _make_eval_tensor_cache_key(
            method="dcf_gnn",
            query=query,
            candidate_chunks=candidate_chunks,
            retrieval_scores=retrieval_scores,
            ppr_scores=ppr_scores or {},
            base_dim=base_dim,
            structural_relation_map=self.structural_relation_map,
            semantic_relation_map=self.semantic_relation_map,
        )
        cached = _load_eval_tensor_cache(self._eval_cache_dir, cache_key)
        if cached is not None:
            self.eval_cache_hits += 1
            node_list = list(cached["node_list"])
            node2idx = {n: i for i, n in enumerate(node_list)}
            x = cached["x"]
            qtype_nodes = cached["qtype_nodes"]
            match_feats = cached["match_feats"]
            s_adj = [a for a in cached["structural_adj"]]
            m_adj = [a for a in cached["semantic_adj"]]
            down_diag = cached["diag"]
        else:
            self.eval_cache_misses += 1
            x_base = np.zeros((n_nodes, base_dim), dtype=np.float32)
            for n in node_list:
                if n in features:
                    x_base[node2idx[n]] = features[n]
            x_aug = build_query_augmented_features(
                features,
                node_list,
                query,
                chunk_lookup=chunk_lookup,
                retrieval_scores=retrieval_scores,
                graph_scores=ppr_scores or {},
            )
            x = np.concatenate([x_base, x_aug], axis=1)

            qtype = infer_query_type_features(query)
            qtype_nodes = np.repeat(qtype[None, :], n_nodes, axis=0).astype(np.float32)
            match_feats = np.zeros((n_nodes, 8), dtype=np.float32)
            for node_id in node_list:
                match_feats[node2idx[node_id]] = financial_match_features(
                    query, chunk_lookup.get(node_id)
                )

            s_adj, m_adj, down_diag = _build_channel_adj(
                graph,
                node2idx,
                query,
                chunk_lookup,
                self.structural_relation_map,
                self.semantic_relation_map,
                incident_edges=self._incident_edges,
            )
            _save_eval_tensor_cache(
                self._eval_cache_dir,
                cache_key,
                {
                    "node_list": node_list,
                    "x": x,
                    "qtype_nodes": qtype_nodes,
                    "match_feats": match_feats,
                    "structural_adj": np.stack(s_adj).astype(np.float32),
                    "semantic_adj": np.stack(m_adj).astype(np.float32),
                    "diag": down_diag,
                },
            )

        with torch.no_grad():
            x_t = torch.from_numpy(x).to(self.device)
            s_t = [torch.from_numpy(a).to(self.device) for a in s_adj]
            m_t = [torch.from_numpy(a).to(self.device) for a in m_adj]
            q_t = torch.from_numpy(qtype_nodes).to(self.device)
            match_t = torch.from_numpy(match_feats).to(self.device)
            dcf_logits, inter = self.model(
                x_t, s_t, m_t, q_t, match_t, return_intermediate=True
            )
            dcf_logits_np = dcf_logits.squeeze(-1).cpu().numpy()

        ret_map = {c.chunk_id: s for c, s in candidate_chunks}
        graph_map = ppr_scores or {}
        dcf_map = {
            node_list[i]: float(dcf_logits_np[i])
            for i in range(n_nodes) if node_list[i] in ret_map
        }

        ret_norm = normalise_score_map(ret_map)
        graph_norm = normalise_score_map(graph_map)
        dcf_norm = normalise_score_map(dcf_map)

        reranked: List[Tuple[Chunk, float]] = []
        for chunk, _ in candidate_chunks:
            cid = chunk.chunk_id
            final = (
                self.alpha * ret_norm.get(cid, 0.0)
                + self.beta * graph_norm.get(cid, 0.0)
                + self.gamma * dcf_norm.get(cid, 0.0)
            )
            reranked.append((chunk, final))
        reranked.sort(key=lambda x: x[1], reverse=True)

        cand_indices = [node2idx[cid] for cid in candidate_ids if cid in node2idx]
        if cand_indices:
            sg = inter["structural_gate"].detach().cpu().numpy()[cand_indices]
            mg = inter["semantic_gate"].detach().cpu().numpy()[cand_indices]
            self.last_diagnostics = {
                "query_type": QUERY_TYPE_NAMES[int(np.argmax(qtype_nodes[0]))],
                "structural_gate_mean": float(np.mean(sg)),
                "semantic_gate_mean": float(np.mean(mg)),
                **down_diag,
            }
        else:
            self.last_diagnostics = {}

        return reranked

    def save(self, path: str | Path) -> None:
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "structural_relation_map": self.structural_relation_map,
                "semantic_relation_map": self.semantic_relation_map,
                "alpha": self.alpha,
                "beta": self.beta,
                "gamma": self.gamma,
            },
            path,
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        model: DCFGNNReranker,
        device: str = "cpu",
    ) -> "DCFGNNFusionReranker":
        ckpt = torch.load(path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        return cls(
            model=model,
            structural_relation_map=ckpt.get("structural_relation_map", {}),
            semantic_relation_map=ckpt.get("semantic_relation_map", {}),
            alpha=ckpt.get("alpha", 0.35),
            beta=ckpt.get("beta", 0.10),
            gamma=ckpt.get("gamma", 0.55),
            device=device,
        )


def _collate_dcf(batch):
    xs, s_adjs, m_adjs, qtypes, matches, pos_idxs, neg_idxs, diags = zip(*batch)
    feat_dim = xs[0].shape[1]
    match_dim = matches[0].shape[1]
    qtype_dim = qtypes[0].shape[1]
    ns = [x.shape[0] for x in xs]
    total_n = sum(ns)
    offsets = [0] + list(np.cumsum(ns))

    x_big = torch.zeros(total_n, feat_dim, dtype=xs[0].dtype)
    q_big = torch.zeros(total_n, qtype_dim, dtype=qtypes[0].dtype)
    match_big = torch.zeros(total_n, match_dim, dtype=matches[0].dtype)
    for i, x in enumerate(xs):
        start, end = offsets[i], offsets[i + 1]
        x_big[start:end] = x
        q_big[start:end] = qtypes[i]
        match_big[start:end] = matches[i]

    s_big = _block_diag_relation_adjs(s_adjs, ns, offsets)
    m_big = _block_diag_relation_adjs(m_adjs, ns, offsets)
    pos_global = torch.stack([pos_idxs[i] + offsets[i] for i in range(len(batch))])
    neg_global = torch.stack([neg_idxs[i] + offsets[i] for i in range(len(batch))])

    return x_big, s_big, m_big, q_big, match_big, pos_global, neg_global, list(diags)


def _block_diag_relation_adjs(adj_stacks, ns: List[int], offsets: List[int]) -> List[torch.Tensor]:
    num_rels = adj_stacks[0].shape[0]
    total_n = sum(ns)
    out: List[torch.Tensor] = []
    for r in range(num_rels):
        a_r = torch.zeros(total_n, total_n)
        for i, adj in enumerate(adj_stacks):
            n = ns[i]
            a_r[offsets[i]:offsets[i] + n, offsets[i]:offsets[i] + n] = adj[r]
        out.append(a_r)
    return out


def _build_channel_adj(
    graph: FinancialEvidenceGraph,
    node2idx: Dict[str, int],
    query: str,
    chunk_lookup: Dict[str, Chunk],
    structural_relation_map: Dict[str, int],
    semantic_relation_map: Dict[str, int],
    incident_edges: Optional[IncidentEdgeMap] = None,
) -> Tuple[List[np.ndarray], List[np.ndarray], Dict[str, float]]:
    n_nodes = len(node2idx)
    structural_adj = [
        np.zeros((n_nodes, n_nodes), dtype=np.float32)
        for _ in range(max(structural_relation_map.values()) + 1 if structural_relation_map else 1)
    ]
    semantic_adj = [
        np.zeros((n_nodes, n_nodes), dtype=np.float32)
        for _ in range(max(semantic_relation_map.values()) + 1 if semantic_relation_map else 1)
    ]
    downweights: Dict[str, List[float]] = defaultdict(list)

    edge_iter = _iter_subgraph_edges(graph, node2idx, incident_edges)
    for u, v, etype, weight in edge_iter:
        if u not in node2idx or v not in node2idx:
            continue
        i, j = node2idx[u], node2idx[v]
        if etype in semantic_relation_map:
            factor, flags = _semantic_conflict_factor(
                query, chunk_lookup.get(u), chunk_lookup.get(v)
            )
            r = semantic_relation_map[etype]
            w = weight * factor
            semantic_adj[r][i, j] = max(semantic_adj[r][i, j], w)
            semantic_adj[r][j, i] = max(semantic_adj[r][j, i], w)
            for name, active in flags.items():
                if active:
                    downweights[name].append(factor)
        elif etype in structural_relation_map:
            r = structural_relation_map[etype]
            structural_adj[r][i, j] = max(structural_adj[r][i, j], weight)
            structural_adj[r][j, i] = max(structural_adj[r][j, i], weight)

    diag = {
        f"semantic_downweight_{name}": float(np.mean(vals)) if vals else 1.0
        for name, vals in downweights.items()
    }
    return [_normalise_adj(a) for a in structural_adj], [_normalise_adj(a) for a in semantic_adj], diag


def _build_incident_edges(graph: FinancialEvidenceGraph) -> IncidentEdgeMap:
    """Index graph edges once so per-query subgraph builds avoid full scans."""
    incident: IncidentEdgeMap = defaultdict(list)
    for u, v, _k, data in graph.graph.edges(keys=True, data=True):
        etype = str(data.get("edge_type", ""))
        weight = float(data.get("weight", 1.0))
        incident[u].append((v, etype, weight))
        incident[v].append((u, etype, weight))
    return dict(incident)


def _iter_subgraph_edges(
    graph: FinancialEvidenceGraph,
    node2idx: Dict[str, int],
    incident_edges: Optional[IncidentEdgeMap],
):
    if incident_edges is None:
        for u, v, k, etype in graph.graph.edges(keys=True, data="edge_type"):
            if u not in node2idx or v not in node2idx:
                continue
            weight = float(graph.graph.edges[u, v, k].get("weight", 1.0))
            yield u, v, etype, weight
        return

    seen: Set[Tuple[str, str, str]] = set()
    for u in node2idx:
        for v, etype, weight in incident_edges.get(u, []):
            if v not in node2idx:
                continue
            key = (u, v, etype) if u <= v else (v, u, etype)
            if key in seen:
                continue
            seen.add(key)
            yield u, v, etype, weight


def _semantic_conflict_factor(
    query: str,
    chunk_a: Optional[Chunk],
    chunk_b: Optional[Chunk],
) -> Tuple[float, Dict[str, bool]]:
    flags = {"wrong_company": False, "wrong_year": False, "wrong_metric": False}
    if chunk_a is None and chunk_b is None:
        return 1.0, flags
    factor = 1.0
    for chunk in (chunk_a, chunk_b):
        feats = financial_match_features(query, chunk)
        if feats[5] > 0:
            flags["wrong_company"] = True
            factor *= 0.35
        if feats[6] > 0:
            flags["wrong_year"] = True
            factor *= 0.40
        if feats[7] > 0:
            flags["wrong_metric"] = True
            factor *= 0.50
    return max(0.05, factor), flags


def _normalise_adj(adj: np.ndarray) -> np.ndarray:
    if adj.size == 0:
        return adj
    deg = adj.sum(axis=1) + 1e-8
    d_inv_sqrt = np.diag(1.0 / np.sqrt(deg))
    return (d_inv_sqrt @ adj @ d_inv_sqrt).astype(np.float32)


def _build_chunk_lookup(graph: FinancialEvidenceGraph) -> Dict[str, Chunk]:
    lookup: Dict[str, Chunk] = {}
    for node_id in graph.graph.nodes():
        if graph.node_types.get(node_id) != "chunk":
            continue
        attrs = graph.graph.nodes[node_id]
        lookup[node_id] = Chunk(
            chunk_id=node_id,
            text=attrs.get("text", ""),
            chunk_type="text",
            doc_id=attrs.get("doc_id", ""),
            company=attrs.get("company", ""),
            filing_type=attrs.get("filing_type", ""),
            filing_year=str(attrs.get("filing_year", "")),
            section=attrs.get("section", ""),
        )
    return lookup


def _norm_metric(metric: str) -> str:
    return " ".join(metric.lower().replace("_", " ").replace("-", " ").split())


def _extract_section_hints(query: str) -> Set[str]:
    q = query.lower()
    hints = set()
    for token in (
        "item 1", "item 1a", "item 7", "item 8", "risk factors",
        "business", "management discussion", "mda", "financial statements",
        "notes",
    ):
        if token in q:
            hints.add(token)
    return hints


def _make_eval_tensor_cache_key(
    *,
    method: str,
    query: str,
    candidate_chunks: List[Tuple[Chunk, float]],
    retrieval_scores: Dict[str, float],
    ppr_scores: Dict[str, float],
    base_dim: int,
    structural_relation_map: Dict[str, int],
    semantic_relation_map: Dict[str, int],
) -> str:
    candidate_payload = [
        [chunk.chunk_id, round(float(retrieval_scores.get(chunk.chunk_id, score)), 8)]
        for chunk, score in candidate_chunks
    ]
    ppr_payload = sorted(
        [[cid, round(float(score), 8)] for cid, score in ppr_scores.items()]
    )
    payload = {
        "version": 1,
        "method": method,
        "query": query,
        "candidates": candidate_payload,
        "ppr": ppr_payload,
        "base_dim": int(base_dim),
        "structural_relation_map": sorted(structural_relation_map.items()),
        "semantic_relation_map": sorted(semantic_relation_map.items()),
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _load_eval_tensor_cache(
    cache_dir: Optional[Path],
    cache_key: str,
) -> Optional[Dict[str, Any]]:
    if cache_dir is None:
        return None
    npz_path = cache_dir / f"{cache_key}.npz"
    meta_path = cache_dir / f"{cache_key}.json"
    if not npz_path.exists() or not meta_path.exists():
        return None
    try:
        with open(meta_path, "r", encoding="utf-8") as fh:
            meta = json.load(fh)
        data = np.load(npz_path, allow_pickle=False)
        return {
            "node_list": list(meta["node_list"]),
            "diag": dict(meta.get("diag", {})),
            "x": data["x"],
            "qtype_nodes": data["qtype_nodes"],
            "match_feats": data["match_feats"],
            "structural_adj": data["structural_adj"],
            "semantic_adj": data["semantic_adj"],
            **({
                "conflict_feats": data["conflict_feats"],
                "ret_feats": data["ret_feats"],
                "qent_nodes": data["qent_nodes"],
            } if "conflict_feats" in data.files else {}),
        }
    except Exception:
        return None


def _save_eval_tensor_cache(
    cache_dir: Optional[Path],
    cache_key: str,
    payload: Dict[str, Any],
) -> None:
    if cache_dir is None:
        return
    cache_dir.mkdir(parents=True, exist_ok=True)
    npz_path = cache_dir / f"{cache_key}.npz"
    meta_path = cache_dir / f"{cache_key}.json"
    tmp_npz = cache_dir / f"{cache_key}.tmp.npz"
    tmp_meta = cache_dir / f"{cache_key}.json.tmp"
    try:
        with open(tmp_npz, "wb") as fh:
            np.savez_compressed(
                fh,
                x=np.asarray(payload["x"], dtype=np.float32),
                qtype_nodes=np.asarray(payload["qtype_nodes"], dtype=np.float32),
                match_feats=np.asarray(payload["match_feats"], dtype=np.float32),
                structural_adj=np.asarray(payload["structural_adj"], dtype=np.float32),
                semantic_adj=np.asarray(payload["semantic_adj"], dtype=np.float32),
                **({
                    "conflict_feats": np.asarray(payload["conflict_feats"], dtype=np.float32),
                    "ret_feats": np.asarray(payload["ret_feats"], dtype=np.float32),
                    "qent_nodes": np.asarray(payload["qent_nodes"], dtype=np.float32),
                } if "conflict_feats" in payload else {}),
            )
        with open(tmp_meta, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "node_list": list(payload["node_list"]),
                    "diag": dict(payload.get("diag", {})),
                },
                fh,
                ensure_ascii=False,
            )
        tmp_npz.replace(npz_path)
        tmp_meta.replace(meta_path)
    except Exception:
        for path in (tmp_npz, tmp_meta):
            try:
                if path.exists():
                    path.unlink()
            except OSError:
                pass
