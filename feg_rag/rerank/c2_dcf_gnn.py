"""C2-DCF-GNN: Contrastive Confidence-aware Dual-Channel Financial GNN Reranker.

C2-DCF-GNN adapts reliable expert routing and fusion ideas to financial RAG
with domain-specific structural, semantic, conflict, and retrieval experts.

Key design:
- 4 experts: structural, semantic, conflict (GNN-based), retrieval (MLP-based).
- Query-type router with type-specific learnable bias table and top-k sparse routing.
- Contrastive routing loss groups queries by type and aligns router weights.
- Confidence-aware fusion: final = base + tau * confidence * correction.
- Score-preserving: tau=0 exactly preserves the base score ordering.
- Semantic channel reuses DCF-GNN conflict suppression on weak edges.

Does NOT replace R-GCN or DCF-GNN — added as a new comparable reranker.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from feg_rag.data.chunker import Chunk
from feg_rag.graph.builder import FinancialEvidenceGraph
from feg_rag.graph.entities import EntityExtractor
from feg_rag.rerank.dcf_gnn import (
    IncidentEdgeMap,
    STRUCTURAL_RELATION_HINTS,
    SEMANTIC_RELATION_HINTS,
    DCFChannelLayer,
    financial_match_features,
    infer_query_type_features,
    split_relation_channels,
    _build_chunk_lookup,
    _build_incident_edges,
    _iter_subgraph_edges,
    _load_eval_tensor_cache,
    _make_eval_tensor_cache_key,
    _normalise_adj,
    _save_eval_tensor_cache,
    _semantic_conflict_factor,
)
from feg_rag.rerank.query_features import (
    QUERY_FEATURE_DIM,
    build_query_augmented_features,
)
from feg_rag.rerank.scoring import normalise_score_map

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

QUERY_TYPES = ["numeric_fact", "comparison_trend", "explanation_risk", "general"]

RETRIEVAL_FEAT_DIM: int = 7
CONFLICT_FEAT_DIM: int = 3
NUM_EXPERTS: int = 4

_EXTRACTOR = EntityExtractor()

# Preferred expert indices for rule-initialized router bias
# structural=0, semantic=1, conflict=2, retrieval=3
ROUTER_BIAS_INIT: Dict[int, List[int]] = {
    0: [0, 3],  # numeric_fact      -> structural + retrieval
    1: [0, 2],  # comparison_trend  -> structural + conflict
    2: [1, 3],  # explanation_risk  -> semantic + retrieval
    3: [3, 0],  # general           -> retrieval + structural
}


# ---------------------------------------------------------------------------
# Helper: router bias
# ---------------------------------------------------------------------------

def build_router_bias(num_experts: int = 4, bias_scale: float = 1.0) -> np.ndarray:
    """Build initial router bias table from sensible financial priors.

    Returns (num_query_types, num_experts) float32 array.
    """
    bias = np.zeros((len(QUERY_TYPES), num_experts), dtype=np.float32)
    for qtype_idx, expert_indices in ROUTER_BIAS_INIT.items():
        for expert_idx in expert_indices:
            bias[qtype_idx, expert_idx] = bias_scale
    return bias


# ---------------------------------------------------------------------------
# Helpers: query entity / retrieval / conflict features
# ---------------------------------------------------------------------------

def infer_query_entity_features(query: str) -> np.ndarray:
    """Return [has_company, has_year, has_metric] binary vector."""
    feats = np.zeros(3, dtype=np.float32)
    if _EXTRACTOR.extract_companies(query):
        feats[0] = 1.0
    if _EXTRACTOR.extract_years(query):
        feats[1] = 1.0
    if _EXTRACTOR.extract_metrics(query):
        feats[2] = 1.0
    return feats


def build_retrieval_features(
    node_list: List[str],
    retrieval_scores: Dict[str, float],
    chunk_lookup: Dict[str, Chunk],
    query: str,
) -> np.ndarray:
    """Build per-node retrieval features: [norm_score, company, year, metric,
    filing, section, rank_placeholder].

    The rank placeholder is zero during dataset construction (not available
    per-node) and is filled during inference if needed.
    """
    n = len(node_list)
    feats = np.zeros((n, RETRIEVAL_FEAT_DIM), dtype=np.float32)

    max_score = max(retrieval_scores.values()) if retrieval_scores else 1.0
    if max_score <= 0:
        max_score = 1.0

    for i, node_id in enumerate(node_list):
        feats[i, 0] = retrieval_scores.get(node_id, 0.0) / max(max_score, 1e-8)
        chunk = chunk_lookup.get(node_id)
        if chunk is not None:
            mf = financial_match_features(query, chunk)
            feats[i, 1:6] = mf[:5]  # company, year, metric, filing, section match
        # rank placeholder stays 0

    return feats


def build_conflict_features(
    node_list: List[str],
    chunk_lookup: Dict[str, Chunk],
    query: str,
) -> np.ndarray:
    """Build per-node conflict indicators: wrong_company, wrong_year, wrong_metric."""
    n = len(node_list)
    feats = np.zeros((n, CONFLICT_FEAT_DIM), dtype=np.float32)
    for i, node_id in enumerate(node_list):
        chunk = chunk_lookup.get(node_id)
        if chunk is not None:
            mf = financial_match_features(query, chunk)
            feats[i, 0] = mf[5]  # wrong_company
            feats[i, 1] = mf[6]  # wrong_year
            feats[i, 2] = mf[7]  # wrong_metric
    return feats


def _query_type_index(qtype_vec: np.ndarray) -> int:
    """Return the integer query type index from the 4-dim feature vector."""
    return int(np.argmax(qtype_vec))


def _query_type_name(qtype_vec: np.ndarray) -> str:
    return QUERY_TYPES[_query_type_index(qtype_vec)]


# ---------------------------------------------------------------------------
# Expert modules
# ---------------------------------------------------------------------------

class ExpertGNN(nn.Module):
    """Two-layer relation-aware GNN backbone for one expert."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_relations: int,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.conv1 = DCFChannelLayer(in_dim, hidden_dim, num_relations, dropout)
        self.conv2 = DCFChannelLayer(hidden_dim, out_dim, num_relations, dropout)
        self.out_dim = out_dim

    def forward(
        self, x: torch.Tensor, adj_list: List[torch.Tensor]
    ) -> torch.Tensor:
        h = self.conv1(x, adj_list)
        h = self.conv2(h, adj_list)
        return h


class RetrievalMLP(nn.Module):
    """Lightweight MLP expert operating purely on retrieval features."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class QueryTypeRouter(nn.Module):
    """Query-type router with type-specific learnable bias table and top-k
    sparse routing.

    The router computes logits via a small MLP, then adds a type-specific bias
    looked up from a learnable ``bias_table`` of shape (num_query_types,
    num_experts).  The bias table is initialised from financial priors but can
    be updated during training.
    """

    def __init__(
        self,
        num_experts: int = 4,
        top_k: int = 2,
        bias_scale: float = 1.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        router_in_dim = len(QUERY_TYPES) + 3  # query_type (4) + entity (3)
        self.router_mlp = nn.Sequential(
            nn.Linear(router_in_dim, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, num_experts),
        )

        # Type-specific bias table: (num_query_types, num_experts), learnable
        init_bias = build_router_bias(num_experts, bias_scale)
        self.bias_table = nn.Parameter(
            torch.from_numpy(init_bias), requires_grad=True
        )

    def forward(
        self,
        query_type_features: torch.Tensor,
        query_entity_features: torch.Tensor,
        return_logits: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Return sparse router weights of shape (N, num_experts).

        Logits = MLP([qtype, entity]) + qtype @ bias_table.
        """
        router_in = torch.cat([query_type_features, query_entity_features], dim=-1)
        logits = self.router_mlp(router_in)

        # Type-specific bias: each query type row contributes its bias
        # query_type_features: (N, 4)   bias_table: (4, num_experts)
        type_bias = torch.matmul(query_type_features, self.bias_table)  # (N, E)
        logits = logits + type_bias

        # Top-k sparse routing
        top_k_vals, top_k_indices = torch.topk(logits, self.top_k, dim=-1)
        mask = torch.zeros_like(logits)
        mask.scatter_(-1, top_k_indices, 1.0)
        masked_logits = logits * mask + (1.0 - mask) * (-1e9)
        weights = torch.softmax(masked_logits, dim=-1)

        if return_logits:
            return weights, logits
        return weights


class ExpertScoringHead(nn.Module):
    """Small MLP that maps (expert_embed, match_features) -> score."""

    def __init__(
        self,
        embed_dim: int,
        match_dim: int,
        hidden_dim: int,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim + match_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self, expert_embed: torch.Tensor, match_features: torch.Tensor
    ) -> torch.Tensor:
        return self.mlp(torch.cat([expert_embed, match_features], dim=-1))


class ConfidenceHead(nn.Module):
    """Per-expert confidence head with sigmoid output in [0, 1]."""

    def __init__(
        self,
        embed_dim: int,
        match_dim: int,
        conflict_dim: int,
        hidden_dim: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        in_dim = embed_dim + match_dim + conflict_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        expert_embed: torch.Tensor,
        match_features: torch.Tensor,
        conflict_features: torch.Tensor,
    ) -> torch.Tensor:
        in_vec = torch.cat([expert_embed, match_features, conflict_features], dim=-1)
        return torch.sigmoid(self.mlp(in_vec)).squeeze(-1)


# ---------------------------------------------------------------------------
# Main GNN backbone
# ---------------------------------------------------------------------------

class C2DCFGNNReranker(nn.Module):
    """C2-DCF-GNN: 4-expert backbone with contrastive router and confidence fusion.

    Experts:
      0 - Structural (GNN over precise relations)
      1 - Semantic   (GNN over conflict-suppressed weak similarity relations)
      2 - Conflict   (GNN + conflict features)
      3 - Retrieval  (MLP over retrieval features)

    Score-preserving fusion:
      final = base_score + tau * confidence_total * tanh(correction)
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        out_dim: int = 64,
        num_structural_relations: int = 1,
        num_semantic_relations: int = 1,
        match_feature_dim: int = 8,
        retrieval_feat_dim: int = RETRIEVAL_FEAT_DIM,
        conflict_feat_dim: int = CONFLICT_FEAT_DIM,
        num_experts: int = 4,
        top_k: int = 2,
        tau: float = 0.15,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.tau = tau
        self.out_dim = out_dim
        self.match_feature_dim = match_feature_dim

        # --- 3 GNN experts ---
        self.structural_expert = ExpertGNN(
            in_dim, hidden_dim, out_dim, num_structural_relations, dropout
        )
        self.semantic_expert = ExpertGNN(
            in_dim, hidden_dim, out_dim, num_semantic_relations, dropout
        )
        self.conflict_expert = ExpertGNN(
            in_dim + conflict_feat_dim, hidden_dim, out_dim,
            num_semantic_relations, dropout,
        )

        # --- 1 retrieval expert ---
        self.retrieval_expert = RetrievalMLP(
            retrieval_feat_dim, hidden_dim, out_dim, dropout
        )

        # --- Router ---
        self.router = QueryTypeRouter(num_experts, top_k, dropout=dropout)

        # --- Per-expert scoring heads (for correction term) ---
        self.expert_scorers = nn.ModuleList([
            ExpertScoringHead(out_dim, match_feature_dim, hidden_dim, dropout)
            for _ in range(num_experts)
        ])

        # --- Base scorer (R-GCN style: all embeddings + match features) ---
        self.base_scorer = nn.Sequential(
            nn.Linear(out_dim * num_experts + match_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        # --- Confidence heads ---
        self.confidence_heads = nn.ModuleList([
            ConfidenceHead(out_dim, match_feature_dim, conflict_feat_dim,
                           hidden_dim=32, dropout=dropout)
            for _ in range(num_experts)
        ])

    def _expert_forward(
        self,
        x: torch.Tensor,
        structural_adj: List[torch.Tensor],
        semantic_adj: List[torch.Tensor],
        conflict_features: torch.Tensor,
        retrieval_features: torch.Tensor,
    ) -> List[torch.Tensor]:
        """Run all 4 experts; return list of embeddings."""
        # Expert 0: structural
        h0 = self.structural_expert(x, structural_adj)
        # Expert 1: semantic (conflict-suppressed edges)
        h1 = self.semantic_expert(x, semantic_adj)
        # Expert 2: conflict (uses semantic adj + conflict features)
        x_conf = torch.cat([x, conflict_features], dim=-1)
        h2 = self.conflict_expert(x_conf, semantic_adj)
        # Expert 3: retrieval
        h3 = self.retrieval_expert(retrieval_features)

        return [h0, h1, h2, h3]

    def forward(
        self,
        x: torch.Tensor,
        structural_adj: List[torch.Tensor],
        semantic_adj: List[torch.Tensor],
        query_type_features: torch.Tensor,
        match_features: torch.Tensor,
        conflict_features: torch.Tensor,
        retrieval_features: torch.Tensor,
        query_entity_features: Optional[torch.Tensor] = None,
        return_intermediate: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        n_nodes = x.shape[0]

        if query_entity_features is None:
            query_entity_features = torch.zeros(
                n_nodes, 3, device=x.device, dtype=x.dtype
            )

        # 1. Expert embeddings
        embeddings = self._expert_forward(
            x, structural_adj, semantic_adj,
            conflict_features, retrieval_features,
        )

        # 2. Per-expert scores
        expert_scores_list: List[torch.Tensor] = []
        for i, h in enumerate(embeddings):
            s = self.expert_scorers[i](h, match_features)  # (N, 1)
            expert_scores_list.append(s)
        expert_scores = torch.cat(expert_scores_list, dim=-1)  # (N, num_experts)

        # 3. Per-expert confidence
        expert_confs_list: List[torch.Tensor] = []
        for i, h in enumerate(embeddings):
            c = self.confidence_heads[i](h, match_features, conflict_features)  # (N,)
            expert_confs_list.append(c)
        expert_confs = torch.stack(expert_confs_list, dim=-1)  # (N, num_experts)

        # 4. Router weights (per-node, sparse top-k, type-specific bias)
        router_weights = self.router(query_type_features, query_entity_features)  # (N, K)

        # 5. Base score (R-GCN style)
        base_in = torch.cat(embeddings + [match_features], dim=-1)
        base_score = self.base_scorer(base_in)  # (N, 1)

        # 6. Score-preserving fusion
        # correction = sum_i w_i * c_i * s_i
        correction = (router_weights * expert_confs * expert_scores).sum(dim=-1)  # (N,)
        confidence_total = (router_weights * expert_confs).sum(dim=-1)  # (N,)

        final = (
            base_score.squeeze(-1)
            + self.tau * confidence_total * torch.tanh(correction)
        )  # (N,)

        if not return_intermediate:
            return final.unsqueeze(-1)

        intermediates: Dict[str, torch.Tensor] = {
            "base_score": base_score,
            "correction": correction,
            "confidence_total": confidence_total,
            "router_weights": router_weights,
            "expert_confidence": expert_confs,
            "expert_scores": expert_scores,
        }
        return final.unsqueeze(-1), intermediates


# ---------------------------------------------------------------------------
# Loss helpers (standalone functions for testability)
# ---------------------------------------------------------------------------

def compute_contrastive_routing_loss(
    router_weights: torch.Tensor,
    query_type_indices: torch.Tensor,
    temperature: float = 0.2,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Group-wise contrastive routing loss.

    Args:
        router_weights: (B, num_experts) — one router vector per sample.
        query_type_indices: (B,) long tensor — group label per sample.
        temperature: softmax temperature.

    Returns:
        scalar loss. Returns 0.0 if fewer than 2 groups present.
    """
    unique_groups = query_type_indices.unique()
    if len(unique_groups) < 2:
        return torch.tensor(0.0, device=router_weights.device)

    # Prototype per group = mean router weights
    prototypes: Dict[int, torch.Tensor] = {}
    for g in unique_groups:
        gid = int(g.item())
        mask = query_type_indices == gid
        prototypes[gid] = router_weights[mask].mean(dim=0)  # (num_experts,)

    proto_list = [prototypes[int(g.item())] for g in unique_groups]
    proto_stack = torch.stack(proto_list, dim=0)  # (G, num_experts)
    proto_stack = F.normalize(proto_stack, dim=-1)

    loss = torch.tensor(0.0, device=router_weights.device)
    count = 0
    for g in unique_groups:
        gid = int(g.item())
        g_vec = F.normalize(prototypes[gid], dim=-1)  # (num_experts,)
        sim = torch.matmul(g_vec, proto_stack.T) / temperature  # (G,)
        # Positive is the group itself
        pos_idx = (unique_groups == gid).nonzero(as_tuple=True)[0]
        if pos_idx.numel() == 0:
            continue
        numerator = sim[pos_idx[0]].exp()
        denominator = sim.exp().sum()
        loss = loss - torch.log(numerator / (denominator + eps) + eps)
        count += 1

    if count == 0:
        return torch.tensor(0.0, device=router_weights.device)
    return loss / count


def compute_load_balance_loss(
    router_weights: torch.Tensor,
) -> torch.Tensor:
    """Load-balance regularisation: penalise low variance across experts.

    Encourages the router to use all experts rather than collapsing to one.
    """
    mean_usage = router_weights.mean(dim=0)  # (num_experts,)
    # Maximize entropy of mean usage -> minimize negative entropy
    mean_usage = mean_usage.clamp(min=1e-8)
    entropy = -(mean_usage * torch.log(mean_usage)).sum()
    # Normalise by log(num_experts) so max is 1
    max_entropy = np.log(router_weights.shape[-1])
    return 1.0 - entropy / max_entropy


def compute_confidence_loss(
    expert_scores: torch.Tensor,
    expert_confs: torch.Tensor,
    pos_idx: torch.Tensor,
    neg_idx: torch.Tensor,
) -> torch.Tensor:
    """Confidence supervision: train confidence to predict expert correctness.

    For each expert, target = sigmoid(score_pos - score_neg), detached.
    Loss = MSE(confidence_pos, target).
    """
    num_experts = expert_scores.shape[-1]

    loss = torch.tensor(0.0, device=expert_scores.device)
    for e in range(num_experts):
        s_pos = expert_scores[pos_idx, e]  # (B,)
        s_neg = expert_scores[neg_idx, e]  # (B,)
        margin = s_pos - s_neg
        target = torch.sigmoid(margin.detach())  # (B,)
        conf_pos = expert_confs[pos_idx, e]  # (B,)
        loss = loss + F.mse_loss(conf_pos, target)
    return loss / num_experts


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class C2DCFDataset(Dataset):
    """Pairwise ranking dataset for C2-DCF-GNN.

    Mirrors :class:`~feg_rag.rerank.dcf_gnn.DCFRerankDataset` but adds
    retrieval features, conflict features, query entity features, and
    DCF-GNN-style semantic conflict suppression.
    """

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
        self.structural_relation_map, self.semantic_relation_map = (
            split_relation_channels(self.relation_map)
        )

    @property
    def num_structural_relations(self) -> int:
        n = max(self.structural_relation_map.values()) + 1 if self.structural_relation_map else 0
        return max(n, 1)

    @property
    def num_semantic_relations(self) -> int:
        n = max(self.semantic_relation_map.values()) + 1 if self.semantic_relation_map else 0
        return max(n, 1)

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

        # Base + query-augmented features
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
        x = np.concatenate([x_base, x_aug], axis=1).astype(np.float32)

        # Query type features (per-node; all same for a given query)
        qtype = infer_query_type_features(question)
        qtype_nodes = np.repeat(qtype[None, :], n_nodes, axis=0).astype(np.float32)

        # Query entity features
        qent = infer_query_entity_features(question)
        qent_nodes = np.repeat(qent[None, :], n_nodes, axis=0).astype(np.float32)

        # Match features
        match_feats = np.zeros((n_nodes, 8), dtype=np.float32)
        for node_id in node_list:
            match_feats[node2idx[node_id]] = financial_match_features(
                question, self._chunk_lookup.get(node_id)
            )

        # Conflict & retrieval features
        conflict_feats = build_conflict_features(
            node_list, self._chunk_lookup, question
        )
        ret_feats = build_retrieval_features(
            node_list, retrieval_scores, self._chunk_lookup, question
        )

        # Adjacency matrices with DCF-GNN conflict suppression on semantic edges
        s_adj, m_adj, _down_diag = _build_channel_adjs(
            self.graph,
            node2idx,
            question,
            self._chunk_lookup,
            self.structural_relation_map,
            self.semantic_relation_map,
            incident_edges=self._incident_edges,
        )

        pos_idx = node2idx.get(pos_id, 0)
        neg_idx = node2idx.get(neg_id, 0)

        return (
            torch.from_numpy(x),
            _adj_stack(s_adj, n_nodes),
            _adj_stack(m_adj, n_nodes),
            torch.from_numpy(qtype_nodes),
            torch.from_numpy(match_feats),
            torch.from_numpy(conflict_feats),
            torch.from_numpy(ret_feats),
            torch.from_numpy(qent_nodes),
            torch.tensor(pos_idx, dtype=torch.long),
            torch.tensor(neg_idx, dtype=torch.long),
            question,
        )


def _adj_stack(adj_list: List[np.ndarray], n_nodes: int) -> torch.Tensor:
    """Stack adjacency matrices; return a (1, n, n) placeholder if empty."""
    if not adj_list:
        return torch.zeros(1, n_nodes, n_nodes)
    return torch.stack([torch.from_numpy(a) for a in adj_list])


def _build_channel_adjs(
    graph: FinancialEvidenceGraph,
    node2idx: Dict[str, int],
    query: str,
    chunk_lookup: Dict[str, Chunk],
    structural_relation_map: Dict[str, int],
    semantic_relation_map: Dict[str, int],
    incident_edges: Optional[IncidentEdgeMap] = None,
) -> Tuple[List[np.ndarray], List[np.ndarray], Dict[str, float]]:
    """Build structural and semantic adjacency lists for the subgraph.

    Semantic edges are downweighted via DCF-GNN-style conflict suppression
    based on query-level company/year/metric mismatches.
    """
    n_nodes = len(node2idx)

    s_adj = [
        np.zeros((n_nodes, n_nodes), dtype=np.float32)
        for _ in range(
            max(structural_relation_map.values()) + 1 if structural_relation_map else 1
        )
    ]
    m_adj = [
        np.zeros((n_nodes, n_nodes), dtype=np.float32)
        for _ in range(
            max(semantic_relation_map.values()) + 1 if semantic_relation_map else 1
        )
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
            m_adj[r][i, j] = max(m_adj[r][i, j], w)
            m_adj[r][j, i] = max(m_adj[r][j, i], w)
            for name, active in flags.items():
                if active:
                    downweights[name].append(factor)
        elif etype in structural_relation_map:
            r = structural_relation_map[etype]
            s_adj[r][i, j] = max(s_adj[r][i, j], weight)
            s_adj[r][j, i] = max(s_adj[r][j, i], weight)

    diag = {
        f"semantic_downweight_{name}": float(np.mean(vals)) if vals else 1.0
        for name, vals in downweights.items()
    }
    return (
        [_normalise_adj(a) for a in s_adj],
        [_normalise_adj(a) for a in m_adj],
        diag,
    )


# ---------------------------------------------------------------------------
# Collation
# ---------------------------------------------------------------------------

def _collate_c2(batch):
    """Collate function: block-diagonal batching like DCF-GNN."""
    (
        xs, s_adjs, m_adjs, qtypes, matches,
        conflicts, rets, qents, pos_idxs, neg_idxs, questions,
    ) = zip(*batch)

    feat_dim = xs[0].shape[1]
    match_dim = matches[0].shape[1]
    qtype_dim = qtypes[0].shape[1]
    conflict_dim = conflicts[0].shape[1]
    ret_dim = rets[0].shape[1]
    qent_dim = qents[0].shape[1]
    ns = [x.shape[0] for x in xs]
    total_n = sum(ns)
    offsets = [0] + list(np.cumsum(ns))

    x_big = torch.zeros(total_n, feat_dim, dtype=xs[0].dtype)
    q_big = torch.zeros(total_n, qtype_dim, dtype=qtypes[0].dtype)
    match_big = torch.zeros(total_n, match_dim, dtype=matches[0].dtype)
    conflict_big = torch.zeros(total_n, conflict_dim, dtype=conflicts[0].dtype)
    ret_big = torch.zeros(total_n, ret_dim, dtype=rets[0].dtype)
    qent_big = torch.zeros(total_n, qent_dim, dtype=qents[0].dtype)

    for i in range(len(batch)):
        start, end = offsets[i], offsets[i + 1]
        x_big[start:end] = xs[i]
        q_big[start:end] = qtypes[i]
        match_big[start:end] = matches[i]
        conflict_big[start:end] = conflicts[i]
        ret_big[start:end] = rets[i]
        qent_big[start:end] = qents[i]

    s_big = _block_diag_adjs(s_adjs, ns, offsets)
    m_big = _block_diag_adjs(m_adjs, ns, offsets)

    pos_global = torch.stack([pos_idxs[i] + offsets[i] for i in range(len(batch))])
    neg_global = torch.stack([neg_idxs[i] + offsets[i] for i in range(len(batch))])

    return (
        x_big, s_big, m_big, q_big, match_big,
        conflict_big, ret_big, qent_big,
        pos_global, neg_global, list(questions),
    )


def _block_diag_adjs(
    adj_stacks: Tuple[torch.Tensor, ...], ns: List[int], offsets: List[int]
) -> List[torch.Tensor]:
    """Build block-diagonal adjacency per relation."""
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


# ---------------------------------------------------------------------------
# Fusion Reranker (training + inference)
# ---------------------------------------------------------------------------

class C2DCFFusionReranker:
    """C2-DCF-GNN reranker that uses the model's internal score-preserving fusion.

    The model already computes::

        final = base_score + tau * confidence_total * tanh(correction)

    When ``tau=0`` the output is exactly the base score (R-GCN style), so the
    ranking is fully determined by the graph-based base scorer.  The external
    ``rerank()`` method uses the model output **directly** — it does not apply
    a separate alpha/beta/gamma fusion on top, because the model already
    handles fusion internally.

    For evaluation purposes, PPR/graph auxiliary scores can be blended by
    setting ``alpha`` / ``beta``. The default keeps most weight on the original
    retriever and uses C2 as a small learned correction.
    """

    def __init__(
        self,
        model: C2DCFGNNReranker,
        structural_relation_map: Dict[str, int],
        semantic_relation_map: Dict[str, int],
        alpha: float = 0.85,
        beta: float = 0.00,
        gamma: float = 0.15,
        device: str = "cpu",
        chunk_lookup: Optional[Dict[str, Chunk]] = None,
        incident_edges: Optional[IncidentEdgeMap] = None,
        # C2-specific
        route_contrastive_lambda: float = 0.05,
        confidence_lambda: float = 0.05,
        load_balance_weight: float = 0.1,
        margin: float = 0.1,
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
        self.route_contrastive_lambda = route_contrastive_lambda
        self.confidence_lambda = confidence_lambda
        self.load_balance_weight = load_balance_weight
        self.margin = margin
        self.last_diagnostics: Dict[str, Any] = {}

    def set_eval_cache(self, cache_dir: Optional[Union[str, Path]]) -> None:
        if cache_dir is None:
            self._eval_cache_dir = None
            return
        self._eval_cache_dir = Path(cache_dir)
        self._eval_cache_dir.mkdir(parents=True, exist_ok=True)
        self.eval_cache_hits = 0
        self.eval_cache_misses = 0

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        train_dataset: C2DCFDataset,
        val_dataset: Optional[C2DCFDataset] = None,
        epochs: int = 50,
        lr: float = 0.001,
        batch_size: int = 32,
        verbose: bool = True,
        checkpoint_dir: Optional[Union[str, Path]] = None,
        checkpoint_prefix: str = "c2_dcf_gnn",
        checkpoint_every: int = 1,
    ) -> List[float]:
        loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            collate_fn=_collate_c2,
        )
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        history: List[float] = []

        t_start = time.time()
        n_batches = len(loader)
        model = self.model
        model.train()

        # Resolve checkpoint directory
        ckpt_dir: Optional[Path] = None
        if checkpoint_dir is not None:
            ckpt_dir = Path(checkpoint_dir)
            ckpt_dir.mkdir(parents=True, exist_ok=True)

        if verbose:
            print(f"\n{'=' * 55}")
            print("  Training C2-DCF-GNN Reranker")
            print(f"  Samples: {len(train_dataset)}  |  Epochs: {epochs}  |  "
                  f"Batches/epoch: {n_batches}")
            print(f"  Structural rels: {len(self.structural_relation_map)}  |  "
                  f"Semantic rels: {len(self.semantic_relation_map)}")
            print(f"  Batch size: {batch_size}  |  Device: {self.device}  |  "
                  f"LR: {lr}")
            print(f"  Top-K experts: {model.top_k}  |  Tau: {model.tau}")
            print(f"  Route lambda: {self.route_contrastive_lambda}  |  "
                  f"Conf lambda: {self.confidence_lambda}")
            if ckpt_dir is not None:
                print(f"  Checkpoint dir: {ckpt_dir}  |  "
                      f"Every: {checkpoint_every} epoch(s)")
            print(f"{'=' * 55}")

        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_t0 = time.time()
            for batch in loader:
                (
                    x, s_adj, m_adj, qtype, match,
                    conflict, ret_feat, qent,
                    pos_idx, neg_idx, questions,
                ) = batch

                x = x.to(self.device)
                s_adj = [a.to(self.device) for a in s_adj]
                m_adj = [a.to(self.device) for a in m_adj]
                qtype = qtype.to(self.device)
                match = match.to(self.device)
                conflict = conflict.to(self.device)
                ret_feat = ret_feat.to(self.device)
                qent = qent.to(self.device)
                pos_idx = pos_idx.to(self.device)
                neg_idx = neg_idx.to(self.device)

                scores, inter = model(
                    x, s_adj, m_adj, qtype, match, conflict, ret_feat, qent,
                    return_intermediate=True,
                )
                scores = scores.squeeze(-1)

                pos_scores = scores[pos_idx]
                neg_scores = scores[neg_idx]

                # 1. Pairwise ranking loss
                target = torch.ones_like(pos_scores)
                L_rank = F.margin_ranking_loss(
                    pos_scores, neg_scores, target, margin=self.margin,
                )

                # 2. Contrastive routing loss
                router_w = inter["router_weights"]  # (total_N, num_experts)
                qtype_per_node = qtype  # (total_N, 4)
                L_route = torch.tensor(0.0, device=self.device)
                L_load = torch.tensor(0.0, device=self.device)

                if self.route_contrastive_lambda > 0 and len(questions) > 0:
                    per_query_rw = router_w[pos_idx]  # (B, num_experts)
                    per_query_qtype = qtype_per_node[pos_idx]  # (B, 4)
                    qtype_indices = per_query_qtype.argmax(dim=-1)  # (B,)

                    L_route = compute_contrastive_routing_loss(
                        per_query_rw, qtype_indices,
                    )
                    L_load = compute_load_balance_loss(per_query_rw)

                L_router = L_route + self.load_balance_weight * L_load

                # 3. Confidence loss
                L_conf = torch.tensor(0.0, device=self.device)
                if self.confidence_lambda > 0:
                    L_conf = compute_confidence_loss(
                        inter["expert_scores"],
                        inter["expert_confidence"],
                        pos_idx, neg_idx,
                    )

                # Total loss
                loss = (
                    L_rank
                    + self.route_contrastive_lambda * L_router
                    + self.confidence_lambda * L_conf
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / max(n_batches, 1)
            history.append(avg_loss)

            # Per-epoch checkpoint
            if ckpt_dir is not None and checkpoint_every > 0:
                if (epoch + 1) % checkpoint_every == 0 or (epoch + 1) == epochs:
                    ckpt_path = ckpt_dir / f"{checkpoint_prefix}_epoch_{epoch + 1:04d}.pt"
                    latest_path = ckpt_dir / f"{checkpoint_prefix}_latest.pt"
                    self.save(ckpt_path)
                    # Symlink / copy latest
                    try:
                        if latest_path.exists() or latest_path.is_symlink():
                            latest_path.unlink()
                        latest_path.symlink_to(ckpt_path.name)
                    except OSError:
                        import shutil
                        shutil.copy2(str(ckpt_path), str(latest_path))
                    if verbose:
                        print(f"    [ckpt] Saved: {ckpt_path.name}")

            if verbose:
                pct = (epoch + 1) / epochs
                bar = "#" * int(pct * 20) + "-" * (20 - int(pct * 20))
                delta_str = ""
                if epoch > 0:
                    d = avg_loss - history[epoch - 1]
                    delta_str = f"  d={'v' if d < 0 else '^'}{abs(d):.4f}"
                elapsed = time.time() - t_start
                epoch_time = time.time() - epoch_t0
                eta = (
                    elapsed / (epoch + 1) * (epochs - epoch - 1)
                    if epoch + 1 < epochs else 0
                )
                print(
                    f"  [{bar}] {pct:3.0%}  |  "
                    f"Epoch {epoch + 1:>3}/{epochs}  |  "
                    f"loss={avg_loss:.4f}{delta_str}"
                    f"  |  {epoch_time:.1f}s/ep  |  "
                    f"elapsed={elapsed:.0f}s  |  eta={eta:.0f}s"
                )

        total_time = time.time() - t_start
        if verbose and history:
            print(f"{'-' * 55}")
            print(f"  Training finished in {total_time:.1f}s")
            print(f"  Initial loss: {history[0]:.4f}  -->  "
                  f"Final loss: {history[-1]:.4f}"
                  f"  (d: {history[-1] - history[0]:+.4f})")
            print(f"{'=' * 55}\n")

        return history

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def rerank(
        self,
        query: str,
        candidate_chunks: List[Tuple[Chunk, float]],
        graph: FinancialEvidenceGraph,
        features: Dict[str, np.ndarray],
        ppr_scores: Optional[Dict[str, float]] = None,
    ) -> List[Tuple[Chunk, float]]:
        """Rerank candidates with retrieval-preserved C2 score fusion."""
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
            method="c2_dcf_gnn",
            query=query,
            candidate_chunks=candidate_chunks,
            retrieval_scores=retrieval_scores,
            ppr_scores=ppr_scores or {},
            base_dim=base_dim,
            structural_relation_map=self.structural_relation_map,
            semantic_relation_map=self.semantic_relation_map,
        )
        cached = _load_eval_tensor_cache(self._eval_cache_dir, cache_key)
        if cached is not None and "conflict_feats" in cached:
            self.eval_cache_hits += 1
            node_list = list(cached["node_list"])
            node2idx = {n: i for i, n in enumerate(node_list)}
            x = cached["x"]
            qtype_nodes = cached["qtype_nodes"]
            qent_nodes = cached["qent_nodes"]
            match_feats = cached["match_feats"]
            conflict_feats = cached["conflict_feats"]
            ret_feats = cached["ret_feats"]
            s_adj = [a for a in cached["structural_adj"]]
            m_adj = [a for a in cached["semantic_adj"]]
            _down_diag = cached["diag"]
            qtype = qtype_nodes[0]
        else:
            self.eval_cache_misses += 1
            x_base = np.zeros((n_nodes, base_dim), dtype=np.float32)
            for n in node_list:
                if n in features:
                    x_base[node2idx[n]] = features[n]
            x_aug = build_query_augmented_features(
                features, node_list, query,
                chunk_lookup=chunk_lookup,
                retrieval_scores=retrieval_scores,
                graph_scores=ppr_scores or {},
            )
            x = np.concatenate([x_base, x_aug], axis=1).astype(np.float32)

            qtype = infer_query_type_features(query)
            qtype_nodes = np.repeat(qtype[None, :], n_nodes, axis=0).astype(np.float32)

            qent = infer_query_entity_features(query)
            qent_nodes = np.repeat(qent[None, :], n_nodes, axis=0).astype(np.float32)

            match_feats = np.zeros((n_nodes, 8), dtype=np.float32)
            for node_id in node_list:
                match_feats[node2idx[node_id]] = financial_match_features(
                    query, chunk_lookup.get(node_id)
                )

            conflict_feats = build_conflict_features(
                node_list, chunk_lookup, query
            )
            ret_feats = build_retrieval_features(
                node_list, retrieval_scores, chunk_lookup, query
            )

            # Build adjacencies with DCF-GNN conflict suppression
            s_adj, m_adj, _down_diag = _build_channel_adjs(
                graph, node2idx, query, chunk_lookup,
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
                    "qent_nodes": qent_nodes,
                    "match_feats": match_feats,
                    "conflict_feats": conflict_feats,
                    "ret_feats": ret_feats,
                    "structural_adj": np.stack(s_adj).astype(np.float32),
                    "semantic_adj": np.stack(m_adj).astype(np.float32),
                    "diag": _down_diag,
                },
            )

        with torch.no_grad():
            x_t = torch.from_numpy(x).to(self.device)
            s_t = [torch.from_numpy(a).to(self.device) for a in s_adj]
            m_t = [torch.from_numpy(a).to(self.device) for a in m_adj]
            q_t = torch.from_numpy(qtype_nodes).to(self.device)
            match_t = torch.from_numpy(match_feats).to(self.device)
            conflict_t = torch.from_numpy(conflict_feats).to(self.device)
            ret_t = torch.from_numpy(ret_feats).to(self.device)
            qent_t = torch.from_numpy(qent_nodes).to(self.device)

            scores, inter = self.model(
                x_t, s_t, m_t, q_t, match_t, conflict_t, ret_t, qent_t,
                return_intermediate=True,
            )
            scores_np = scores.squeeze(-1).cpu().numpy()

        c2_map = {
            node_list[i]: float(scores_np[i])
            for i in range(n_nodes) if node_list[i] in {c.chunk_id for c, _ in candidate_chunks}
        }

        ret_norm = normalise_score_map(retrieval_scores)
        graph_norm = normalise_score_map(ppr_scores or {})
        c2_norm = normalise_score_map(c2_map)

        graph_weight = self.beta if graph_norm else 0.0
        total_weight = self.alpha + graph_weight + self.gamma
        if total_weight <= 0:
            total_weight = 1.0
        alpha = self.alpha / total_weight
        beta = graph_weight / total_weight
        gamma = self.gamma / total_weight

        reranked: List[Tuple[Chunk, float]] = []
        for chunk, _ in candidate_chunks:
            cid = chunk.chunk_id
            final = (
                alpha * ret_norm.get(cid, 0.0)
                + beta * graph_norm.get(cid, 0.0)
                + gamma * c2_norm.get(cid, 0.0)
            )
            reranked.append((chunk, final))
        reranked.sort(key=lambda x: x[1], reverse=True)

        # Build rich diagnostics
        cand_indices = [node2idx[cid] for cid in candidate_ids if cid in node2idx]
        if cand_indices:
            rw = inter["router_weights"].detach().cpu().numpy()
            ec = inter["expert_confidence"].detach().cpu().numpy()
            ct = inter["confidence_total"].detach().cpu().numpy()

            rw_cand = rw[cand_indices]
            ec_cand = ec[cand_indices]
            ct_cand = ct[cand_indices]

            expert_names = ["structural", "semantic", "conflict", "retrieval"]

            # Routing entropy
            rw_mean = rw_cand.mean(axis=0)
            rw_clipped = np.clip(rw_mean, 1e-8, 1.0)
            routing_entropy = float(-np.sum(rw_clipped * np.log(rw_clipped)))

            self.last_diagnostics = {
                "query_type": _query_type_name(qtype),
                "router_weights": {
                    name: float(rw_mean[i])
                    for i, name in enumerate(expert_names)
                },
                "expert_confidence": {
                    name: float(ec_cand[:, i].mean())
                    for i, name in enumerate(expert_names)
                },
                "confidence_total": float(ct_cand.mean()),
                "routing_entropy": routing_entropy,
                "semantic_downweight_wrong_company": _down_diag.get(
                    "semantic_downweight_wrong_company", 1.0
                ),
                "semantic_downweight_wrong_year": _down_diag.get(
                    "semantic_downweight_wrong_year", 1.0
                ),
                "semantic_downweight_wrong_metric": _down_diag.get(
                    "semantic_downweight_wrong_metric", 1.0
                ),
            }
        else:
            self.last_diagnostics = {}

        return reranked

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Union[str, Path]) -> None:
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "structural_relation_map": self.structural_relation_map,
                "semantic_relation_map": self.semantic_relation_map,
                "alpha": self.alpha,
                "beta": self.beta,
                "gamma": self.gamma,
                "route_contrastive_lambda": self.route_contrastive_lambda,
                "confidence_lambda": self.confidence_lambda,
                "margin": self.margin,
            },
            path,
        )

    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        model: C2DCFGNNReranker,
        device: str = "cpu",
    ) -> "C2DCFFusionReranker":
        ckpt = torch.load(path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        return cls(
            model=model,
            structural_relation_map=ckpt.get("structural_relation_map", {}),
            semantic_relation_map=ckpt.get("semantic_relation_map", {}),
            alpha=ckpt.get("alpha", 0.85),
            beta=ckpt.get("beta", 0.0),
            gamma=ckpt.get("gamma", 0.15),
            device=device,
            route_contrastive_lambda=ckpt.get("route_contrastive_lambda", 0.05),
            confidence_lambda=ckpt.get("confidence_lambda", 0.05),
            margin=ckpt.get("margin", 0.1),
        )


# ---------------------------------------------------------------------------
# Diagnostics writer
# ---------------------------------------------------------------------------

def write_c2_diagnostics(
    output_dir: Path,
    results: List[Dict],
    filename: str = "c2_dcf_gnn_diagnostics.json",
) -> Dict[str, Any]:
    """Aggregate C2-DCF-GNN diagnostics across all queries and write JSON."""
    diag_list = [r.get("diagnostics") or {} for r in results]
    diag_list = [d for d in diag_list if d]

    if not diag_list:
        empty: Dict[str, Any] = {
            "num_queries_with_diagnostics": 0,
            "avg_router_weight_by_query_type": {},
            "avg_confidence_by_expert": {},
            "expert_load": {},
            "routing_entropy_mean": 0.0,
            "confidence_total_mean": 0.0,
            "semantic_conflict_downweight": {},
        }
        out_path = output_dir / filename
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(empty, fh, indent=2, ensure_ascii=False)
        return empty

    by_type: Dict[str, List[Dict]] = defaultdict(list)
    for d in diag_list:
        by_type[d.get("query_type", "unknown")].append(d)

    expert_names = ["structural", "semantic", "conflict", "retrieval"]

    router_by_qtype: Dict[str, Dict[str, float]] = {}
    conf_by_qtype: Dict[str, Dict[str, float]] = {}
    for qtype, rows in sorted(by_type.items()):
        rw_agg = {name: 0.0 for name in expert_names}
        ec_agg = {name: 0.0 for name in expert_names}
        for r in rows:
            for name in expert_names:
                rw_agg[name] += r.get("router_weights", {}).get(name, 0.0)
                ec_agg[name] += r.get("expert_confidence", {}).get(name, 0.0)
        n = len(rows)
        router_by_qtype[qtype] = {k: round(v / n, 4) for k, v in rw_agg.items()}
        conf_by_qtype[qtype] = {k: round(v / n, 4) for k, v in ec_agg.items()}

    # Expert load: average weight across all queries
    expert_load: Dict[str, float] = {}
    for name in expert_names:
        vals = [
            d.get("router_weights", {}).get(name, 0.0)
            for d in diag_list
        ]
        expert_load[name] = round(float(np.mean(vals)), 4) if vals else 0.0

    routing_entropy_mean = float(np.mean([
        d.get("routing_entropy", 0.0) for d in diag_list
    ]))
    confidence_total_mean = float(np.mean([
        d.get("confidence_total", 0.0) for d in diag_list
    ]))

    # Semantic conflict downweight averages
    _mean_dw = lambda key: round(float(np.mean([
        d.get(key, 1.0) for d in diag_list
    ])), 4)

    summary: Dict[str, Any] = {
        "num_queries_with_diagnostics": len(diag_list),
        "avg_router_weight_by_query_type": router_by_qtype,
        "avg_confidence_by_expert": conf_by_qtype,
        "expert_load": expert_load,
        "routing_entropy_mean": round(routing_entropy_mean, 4),
        "confidence_total_mean": round(confidence_total_mean, 4),
        "semantic_conflict_downweight": {
            "wrong_company": _mean_dw("semantic_downweight_wrong_company"),
            "wrong_year": _mean_dw("semantic_downweight_wrong_year"),
            "wrong_metric": _mean_dw("semantic_downweight_wrong_metric"),
        },
    }

    out_path = output_dir / filename
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    return summary
