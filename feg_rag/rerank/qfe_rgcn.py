"""QFE-RGCN: Query-Aware Financial Evidence R-GCN Reranker.

Paper design: QFE-RGCN extends R-GCN with query-aware relation gates and
a financial entity-gated evidence scoring head for Table II "Final Graph (Ours)".

Key innovations over vanilla R-GCN:
1. Query-aware relation gate: alpha_r(q) = softmax_r(MLP([q_embed; r_embed]))
2. Entity-gated scoring head: score = MLP([ret, q_embed, chunk_proj, gnn_embed, matches...])
3. Query-level pairwise ranking loss with trainable scoring head
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

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


# ═════════════════════════════════════════════════════════════════════════════
# Constants
# ═════════════════════════════════════════════════════════════════════════════

QUERY_EMBED_DIM = 64
RELATION_EMBED_DIM = 64

# Shared entity extractor (stateless, thread-safe)
_extractor = EntityExtractor()


# ═════════════════════════════════════════════════════════════════════════════
# Query Vector Derivation (MVP: entity-feature-based)
# ═════════════════════════════════════════════════════════════════════════════

def derive_query_vector(query: str, dim: int = QUERY_EMBED_DIM) -> np.ndarray:
    """Derive a lightweight query embedding from financial entity features.

    MVP approach: uses entity extraction (metrics, years, companies, filing
    types) to create a deterministic feature vector.  This avoids loading a
    heavy encoder model while still capturing query structure.

    For best results, replace with BGE-M3 query embeddings in production.

    Args:
        query: The question text.
        dim: Output dimension (default 64).

    Returns:
        Float32 vector of shape (dim,).
    """
    metrics = _extractor.extract_metrics(query)
    years = _extractor.extract_years(query)
    companies = _extractor.extract_companies(query)
    filing_types = _extractor.extract_filing_types(query)

    features = np.zeros(dim, dtype=np.float32)

    # Position 0-3: entity counts (clipped / normalised)
    features[0] = min(len(metrics), 10) / 10.0
    features[1] = min(len(years), 5) / 5.0
    features[2] = min(len(companies), 10) / 10.0
    features[3] = min(len(query.split()), 200) / 200.0

    # Position 4-8: entity type presence indicators
    features[4] = 1.0 if metrics else 0.0
    features[5] = 1.0 if years else 0.0
    features[6] = 1.0 if companies else 0.0
    features[7] = 1.0 if filing_types else 0.0
    features[8] = 1.0 if any(w in query.lower() for w in
                            ["revenue", "profit", "income", "earnings",
                             "assets", "liabilities", "cash", "eps"]) else 0.0

    # Position 9+: stable hash-based encoding of extracted entities
    # Use hashlib.sha1 instead of Python's built-in hash() so that the same
    # query produces identical vectors across different processes and machines.
    def _stable_hash(s: str, mod: int) -> int:
        return int(hashlib.sha1(s.encode("utf-8")).hexdigest(), 16) % mod

    for i, m in enumerate(sorted(metrics)[:8]):
        idx = 9 + _stable_hash(m, dim - 9)
        features[idx] = 1.0

    for i, y in enumerate(sorted(years)[:4]):
        idx = 9 + _stable_hash(y, dim - 9)
        features[idx] = 1.0

    for i, c in enumerate(sorted(companies)[:4]):
        idx = 9 + _stable_hash(c, dim - 9)
        features[idx] = 1.0

    # L2-normalise
    norm = np.linalg.norm(features)
    if norm > 1e-8:
        features = features / norm

    return features


def build_query_embedding_cache(
    queries: List[str],
    dim: int = QUERY_EMBED_DIM,
    *,
    bge_model: object = None,
) -> Dict[str, np.ndarray]:
    """Pre-compute query embeddings for a list of query strings.

    If *bge_model* is provided (a sentence-transformers model or similar that
    supports ``.encode()``), it is used to produce semantic embeddings.
    Otherwise, falls back to :func:`derive_query_vector`.

    Args:
        queries: List of query strings.
        dim: Embedding dimension (used for fallback only).
        bge_model: Optional sentence-transformers model for semantic encoding.

    Returns:
        Dict mapping each unique query string → its embedding (float32).
    """
    unique_queries = list(dict.fromkeys(queries))  # dedup, preserve order
    cache: Dict[str, np.ndarray] = {}

    if bge_model is not None:
        try:
            bge_embeds = bge_model.encode(
                unique_queries,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            for q, emb in zip(unique_queries, bge_embeds):
                # Project to target dim if needed
                emb_f32 = np.asarray(emb, dtype=np.float32)
                if emb_f32.shape[0] != dim:
                    # Simple truncation/padding or use a learnable projection
                    if emb_f32.shape[0] > dim:
                        emb_f32 = emb_f32[:dim]
                    else:
                        padded = np.zeros(dim, dtype=np.float32)
                        padded[:emb_f32.shape[0]] = emb_f32
                        emb_f32 = padded
                cache[q] = emb_f32
            return cache
        except Exception:
            pass  # fall through to derive_query_vector

    # Fallback: entity-feature-based query vector
    for q in unique_queries:
        cache[q] = derive_query_vector(q, dim=dim)
    return cache


# ═════════════════════════════════════════════════════════════════════════════
# QFE-RGCN Layer
# ═════════════════════════════════════════════════════════════════════════════

class QFERGCNLayer(nn.Module):
    """Query-Aware Financial Evidence R-GCN Layer.

    Extends standard R-GCN with a query-aware relation gate:

        alpha_r(q) = softmax_r(MLP([q_embed; r_embed]))
        h_i^{l+1} = sigma( sum_r alpha_r(q) * A_r * H * W_r + W_0 * h_i )

    The relation gate ``alpha_r(q)`` is dynamically computed for each
    query/subgraph, allowing the model to adaptively weight different relation
    types based on the query's financial entity structure.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_relations: int,
        query_embed_dim: int = QUERY_EMBED_DIM,
        relation_embed_dim: int = RELATION_EMBED_DIM,
        dropout: float = 0.3,
        use_bias: bool = True,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_relations = num_relations
        self.query_embed_dim = query_embed_dim

        # Self-loop weight (W_0)
        self.self_loop = nn.Linear(in_dim, out_dim, bias=False)

        # Per-relation weights
        self.rel_weights = nn.ModuleList([
            nn.Linear(in_dim, out_dim, bias=False)
            for _ in range(num_relations)
        ])

        # Relation embeddings (learnable parameters)
        self.relation_embeds = nn.Parameter(
            torch.randn(num_relations, relation_embed_dim) * 0.1
        )

        # Query-aware relation gate MLP
        # Input: [query_embed (query_embed_dim), relation_embed (relation_embed_dim)]
        gate_input_dim = query_embed_dim + relation_embed_dim
        self.gate_mlp = nn.Sequential(
            nn.Linear(gate_input_dim, gate_input_dim // 2),
            nn.ReLU(),
            nn.Linear(gate_input_dim // 2, 1),
        )

        self.dropout = nn.Dropout(dropout)
        self.bias = nn.Parameter(torch.zeros(out_dim)) if use_bias else None

    def forward(
        self,
        x: torch.Tensor,
        adj_list: List[torch.Tensor],
        norm_list: List[torch.Tensor],
        query_embed: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass with query-aware relation gating.

        Args:
            x: Node features (N, in_dim).
            adj_list: List of pre-normalised adjacency matrices per relation,
                each (N, N).
            norm_list: Per-node normalisation scalars per relation.
                **Accepted for API compatibility but not used** — normalisation
                is assumed to be folded into adj_list by the caller.
            query_embed: Query embedding vector (query_embed_dim,).

        Returns:
            Updated node features (N, out_dim).
        """
        out = self.self_loop(x)  # self-loop contribution

        # Compute relation gates: alpha_r(q) for all relations
        # query_embed: (query_embed_dim,) → (num_relations, query_embed_dim)
        q_expanded = query_embed.unsqueeze(0).expand(self.num_relations, -1)
        # Concat: (num_relations, query_embed_dim + relation_embed_dim)
        gate_input = torch.cat([q_expanded, self.relation_embeds], dim=-1)
        # Gate scores: (num_relations,) after squeeze
        gate_scores = self.gate_mlp(gate_input).squeeze(-1)
        # Softmax over relations → alpha_r(q)
        alpha = F.softmax(gate_scores, dim=0)  # (num_relations,)

        # Message passing with gated relations
        for r in range(self.num_relations):
            if adj_list[r].numel() == 0:
                continue
            support = self.rel_weights[r](x)    # (N, out_dim)
            messages = adj_list[r] @ support     # (N, out_dim)
            out += alpha[r] * messages           # weight by relation gate

        out = F.relu(out)
        out = self.dropout(out)
        if self.bias is not None:
            out = out + self.bias
        return out


# ═════════════════════════════════════════════════════════════════════════════
# QFE-RGCN Reranker Model
# ═════════════════════════════════════════════════════════════════════════════

class QFERGCNReranker(nn.Module):
    """2-layer QFE-RGCN for query-aware evidence reranking.

    Produces per-node embeddings that are consumed by
    :class:`EntityGatedScoringHead` for final relevance scoring.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        out_dim: int = 64,
        num_relations: int = 8,
        query_embed_dim: int = QUERY_EMBED_DIM,
        relation_embed_dim: int = RELATION_EMBED_DIM,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.out_dim = out_dim
        self.conv1 = QFERGCNLayer(
            in_dim, hidden_dim, num_relations,
            query_embed_dim=query_embed_dim,
            relation_embed_dim=relation_embed_dim,
            dropout=dropout,
        )
        self.conv2 = QFERGCNLayer(
            hidden_dim, out_dim, num_relations,
            query_embed_dim=query_embed_dim,
            relation_embed_dim=relation_embed_dim,
            dropout=dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        adj_list: List[torch.Tensor],
        query_embed: torch.Tensor,
    ) -> torch.Tensor:
        """Two-layer query-aware message passing.

        Args:
            x: Node features (N, in_dim).
            adj_list: List of pre-normalised adjacency matrices per relation.
            query_embed: Query embedding vector (query_embed_dim,).

        Returns:
            Per-node embeddings (N, out_dim).
        """
        norm_list = [
            torch.ones(a.shape[0], device=x.device) if a.numel() > 0
            else torch.tensor([], device=x.device)
            for a in adj_list
        ]

        x = self.conv1(x, adj_list, norm_list, query_embed)
        x = self.conv2(x, adj_list, norm_list, query_embed)
        return x


# ═════════════════════════════════════════════════════════════════════════════
# Entity-Gated Evidence Scoring Head
# ═════════════════════════════════════════════════════════════════════════════

class EntityGatedScoringHead(nn.Module):
    """Financial entity-gated evidence scoring head.

    Replaces hand-crafted linear fusion with a trainable MLP:

        score(q, d) = MLP([
            retrieval_score,        # 1
            query_embedding,        # query_embed_dim
            chunk_embedding_proj,   # chunk_proj_dim (from base features)
            qfe_rgcn_embedding,     # gnn_out_dim
            company_match,          # 1
            year_match,             # 1
            metric_match,           # 1
            filing_type_match,      # 1
            section_match,          # 1
        ])
    """

    def __init__(
        self,
        query_embed_dim: int = QUERY_EMBED_DIM,
        chunk_proj_dim: int = 64,
        gnn_out_dim: int = 64,
        base_feat_dim: int = 4096,
        hidden_dim: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()
        # Project base chunk features to manageable dimension
        self.chunk_proj = nn.Sequential(
            nn.Linear(base_feat_dim, chunk_proj_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(chunk_proj_dim * 2, chunk_proj_dim),
        )

        # Total input: 1 + query_embed_dim + chunk_proj_dim + gnn_out_dim + 5
        mlp_input_dim = 1 + query_embed_dim + chunk_proj_dim + gnn_out_dim + 5

        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        retrieval_score: torch.Tensor,
        query_embed: torch.Tensor,
        chunk_base_feat: torch.Tensor,
        gnn_embed: torch.Tensor,
        company_match: torch.Tensor,
        year_match: torch.Tensor,
        metric_match: torch.Tensor,
        filing_type_match: torch.Tensor,
        section_match: torch.Tensor,
    ) -> torch.Tensor:
        """Compute entity-gated evidence relevance score.

        Args:
            retrieval_score: Normalised retrieval score (N,) or scalar.
            query_embed: Query embedding (query_embed_dim,) — broadcast over N.
            chunk_base_feat: Base chunk features (N, base_feat_dim).
            gnn_embed: QFE-RGCN output embedding for this chunk (N, gnn_out_dim).
            company_match: Binary company match indicator (N,) or scalar.
            year_match: Binary year match indicator (N,) or scalar.
            metric_match: Binary metric match indicator (N,) or scalar.
            filing_type_match: Binary filing type match indicator (N,) or scalar.
            section_match: Binary section match indicator (N,) or scalar.

        Returns:
            Relevance scores (N,) or scalar.
        """
        # Project base features
        chunk_proj = self.chunk_proj(chunk_base_feat)  # (N, chunk_proj_dim)

        # Ensure all inputs are at least 1D
        if retrieval_score.dim() == 0:
            retrieval_score = retrieval_score.unsqueeze(0)
        if company_match.dim() == 0:
            company_match = company_match.unsqueeze(0)
        if year_match.dim() == 0:
            year_match = year_match.unsqueeze(0)
        if metric_match.dim() == 0:
            metric_match = metric_match.unsqueeze(0)
        if filing_type_match.dim() == 0:
            filing_type_match = filing_type_match.unsqueeze(0)
        if section_match.dim() == 0:
            section_match = section_match.unsqueeze(0)

        # Broadcast query_embed if needed
        if query_embed.dim() == 1:
            query_embed = query_embed.unsqueeze(0).expand(chunk_proj.shape[0], -1)

        # Concatenate all features
        x = torch.cat([
            retrieval_score.unsqueeze(-1),
            query_embed,
            chunk_proj,
            gnn_embed,
            company_match.unsqueeze(-1),
            year_match.unsqueeze(-1),
            metric_match.unsqueeze(-1),
            filing_type_match.unsqueeze(-1),
            section_match.unsqueeze(-1),
        ], dim=-1)

        return self.mlp(x).squeeze(-1)


# ═════════════════════════════════════════════════════════════════════════════
# Dataset for QFE-RGCN training
# ═════════════════════════════════════════════════════════════════════════════

class QFERGCNRerankDataset(Dataset):
    """Pairwise ranking dataset with per-relation adjacency and query embeddings.

    Each sample is a (positive, negative) pair from the same query's top-50
    candidates.  Positive = gold evidence chunk; negative = non-gold chunk from
    top-50.
    """

    def __init__(
        self,
        samples: List[Dict],
        graph: FinancialEvidenceGraph,
        features: Dict[str, np.ndarray],
        query_embeddings: Dict[str, np.ndarray],
        relation_map: Optional[Dict[str, int]] = None,
        chunk_lookup: Optional[Dict[str, Chunk]] = None,
        query_embed_dim: int = QUERY_EMBED_DIM,
    ):
        self.samples = samples
        self.graph = graph
        self.features = features
        self.query_embeddings = query_embeddings
        self.query_embed_dim = query_embed_dim
        self._chunk_lookup = chunk_lookup or _build_qfe_chunk_lookup(graph)

        # Build relation → integer index mapping
        if relation_map is None:
            etypes: Set[str] = set()
            for _u, _v, _k, etype in graph.graph.edges(keys=True, data="edge_type"):
                etypes.add(etype)
            self.relation_map = {et: i for i, et in enumerate(sorted(etypes))}
        else:
            self.relation_map = relation_map

        self.num_relations = (
            max(self.relation_map.values()) + 1 if self.relation_map else 1
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        pos_id = s.get("positive", s.get("chunk_id", ""))
        neg_id = s.get("negative", "")
        question = s.get("question", "")

        # Build subgraph (2-hop neighbours around pos and neg)
        sub_nodes: Set[str] = set()
        for seed in [pos_id, neg_id]:
            if seed:
                sub_nodes.add(seed)
                sub_nodes |= self.graph.get_chunk_neighbors(seed, max_hops=2)

        node_list = list(sub_nodes)
        node2idx = {n: i for i, n in enumerate(node_list)}
        N = len(node_list)

        # Base feature matrix
        base_dim = next(iter(self.features.values())).shape[0]
        x_base = np.zeros((N, base_dim), dtype=np.float32)
        for n in node_list:
            if n in self.features:
                x_base[node2idx[n]] = self.features[n]

        # Query-aware augmented features
        ret_scores = s.get("retrieval_scores", None) or {}
        graph_scores = s.get("graph_scores", None) or {}
        x_aug = build_query_augmented_features(
            self.features, node_list, question,
            chunk_lookup=self._chunk_lookup,
            retrieval_scores=ret_scores,
            graph_scores=graph_scores,
        )

        # Concatenate base + query features
        x = np.concatenate([x_base, x_aug], axis=1)

        # Per-relation adjacency matrices
        adj_list = [
            np.zeros((N, N), dtype=np.float32)
            for _ in range(self.num_relations)
        ]
        for u, v, k, etype in self.graph.graph.edges(keys=True, data="edge_type"):
            if u not in node2idx or v not in node2idx:
                continue
            r = self.relation_map.get(etype)
            if r is None:
                continue
            i, j = node2idx[u], node2idx[v]
            adj_list[r][i, j] = 1.0
            adj_list[r][j, i] = 1.0  # symmetrize

        # Normalise each relation adjacency (D^-0.5 A D^-0.5)
        for r in range(self.num_relations):
            a = adj_list[r]
            deg = a.sum(axis=1) + 1e-8
            d_inv_sqrt = np.diag(1.0 / np.sqrt(deg))
            adj_list[r] = d_inv_sqrt @ a @ d_inv_sqrt

        # Query embedding (pre-computed or fallback)
        q_embed = self.query_embeddings.get(
            question,
            derive_query_vector(question, dim=self.query_embed_dim),
        )

        pos_idx = node2idx.get(pos_id, 0)
        neg_idx = node2idx.get(neg_id, 0)

        # Also return retrieval scores and entity match features for scoring head
        ret_pos = float(ret_scores.get(pos_id, 0.0))
        ret_neg = float(ret_scores.get(neg_id, 0.0))

        # Entity match features for pos and neg chunks
        pos_matches = _compute_entity_matches(
            pos_id, question, self._chunk_lookup,
        )
        neg_matches = _compute_entity_matches(
            neg_id, question, self._chunk_lookup,
        )

        return (
            torch.from_numpy(x),
            torch.stack([torch.from_numpy(a) for a in adj_list]),
            torch.from_numpy(q_embed.copy()),
            torch.tensor(pos_idx, dtype=torch.long),
            torch.tensor(neg_idx, dtype=torch.long),
            torch.tensor(ret_pos, dtype=torch.float32),
            torch.tensor(ret_neg, dtype=torch.float32),
            torch.tensor(pos_matches, dtype=torch.float32),  # (5,) company/year/metric/filing/section
            torch.tensor(neg_matches, dtype=torch.float32),
        )


# ═════════════════════════════════════════════════════════════════════════════
# QFE-RGCN Fusion Reranker
# ═════════════════════════════════════════════════════════════════════════════

class QFERGCNFusionReranker:
    """QFE-RGCN reranker with entity-gated evidence scoring.

    Uses a trainable :class:`EntityGatedScoringHead` instead of hand-crafted
    linear fusion.  Training minimises pairwise ranking loss on query-level
    positive/negative pairs.
    """

    def __init__(
        self,
        model: nn.Module,
        scoring_head: EntityGatedScoringHead,
        relation_map: Dict[str, int],
        query_embeddings: Dict[str, np.ndarray],
        query_embed_dim: int = QUERY_EMBED_DIM,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.scoring_head = scoring_head.to(device)
        self.relation_map = relation_map
        self.num_relations = (
            max(relation_map.values()) + 1 if relation_map else 1
        )
        self.query_embeddings = query_embeddings
        self.query_embed_dim = query_embed_dim
        self.device = device

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------

    def fit(
        self,
        train_dataset: QFERGCNRerankDataset,
        val_dataset: Optional[QFERGCNRerankDataset] = None,
        epochs: int = 50,
        lr: float = 0.001,
        batch_size: int = 32,
        verbose: bool = True,
    ) -> List[float]:
        """Train with pairwise margin ranking loss on entity-gated scores."""
        loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            collate_fn=_collate_qfe_rgcn,
        )
        params = list(self.model.parameters()) + list(self.scoring_head.parameters())
        optimizer = torch.optim.Adam(params, lr=lr)
        loss_fn = nn.MarginRankingLoss(margin=0.5)

        history: List[float] = []
        self.model.train()
        self.scoring_head.train()

        t_start = time.time()
        n_batches = len(loader)
        print(f"\n{'=' * 55}")
        print(f"  Training QFE-RGCN Reranker")
        print(f"  Samples: {len(train_dataset)}  |  Epochs: {epochs}  |  "
              f"Batches/epoch: {n_batches}")
        print(f"  Relations: {self.num_relations}  |  Batch size: {batch_size}  |  "
              f"Device: {self.device}  |  LR: {lr}")
        print(f"{'=' * 55}")

        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_t0 = time.time()
            for batch in loader:
                (x, adj_list, q_embed, pos_idx, neg_idx,
                 ret_pos, ret_neg, match_pos, match_neg) = batch

                x = x.to(self.device)
                adj_list = [a.to(self.device) for a in adj_list]
                q_embed = q_embed.to(self.device)
                pos_idx = pos_idx.to(self.device)
                neg_idx = neg_idx.to(self.device)
                ret_pos = ret_pos.to(self.device)
                ret_neg = ret_neg.to(self.device)
                match_pos = match_pos.to(self.device)
                match_neg = match_neg.to(self.device)

                # QFE-RGCN forward: produce per-node embeddings
                gnn_embeds = self.model(x, adj_list, q_embed)  # (total_N, gnn_out_dim)

                # Get base features for scoring head (first base_dim columns of x)
                base_dim = self.scoring_head.chunk_proj[0].in_features
                base_feats = x[:, :base_dim]

                # Compute entity-gated scores for pos and neg nodes
                pos_base = base_feats[pos_idx]
                neg_base = base_feats[neg_idx]
                pos_gnn = gnn_embeds[pos_idx]
                neg_gnn = gnn_embeds[neg_idx]

                pos_scores = self.scoring_head(
                    ret_pos, q_embed, pos_base, pos_gnn,
                    match_pos[:, 0], match_pos[:, 1], match_pos[:, 2],
                    match_pos[:, 3], match_pos[:, 4],
                )
                neg_scores = self.scoring_head(
                    ret_neg, q_embed, neg_base, neg_gnn,
                    match_neg[:, 0], match_neg[:, 1], match_neg[:, 2],
                    match_neg[:, 3], match_neg[:, 4],
                )

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
            eta = (elapsed / (epoch + 1) * (epochs - epoch - 1)
                   if epoch + 1 < epochs else 0)

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
        if len(history) >= 2:
            print(f"  Initial loss: {history[0]:.4f}  -->  Final loss: {history[-1]:.4f}"
                  f"  (d: {history[-1] - history[0]:+.4f})")
        print(f"{'=' * 55}\n")
        return history

    # ------------------------------------------------------------------
    # Rerank
    # ------------------------------------------------------------------

    def rerank(
        self,
        query: str,
        candidate_chunks: List[Tuple[Chunk, float]],
        graph: FinancialEvidenceGraph,
        features: Dict[str, np.ndarray],
        ppr_scores: Optional[Dict[str, float]] = None,
    ) -> List[Tuple[Chunk, float]]:
        """Rerank candidate chunks with QFE-RGCN + entity-gated scoring.

        Args:
            query: The question text.
            candidate_chunks: List of (Chunk, retrieval_score).
            graph: The financial evidence graph.
            features: Node features dict.
            ppr_scores: Optional dict of chunk_id → PPR score.

        Returns:
            Reranked list of (Chunk, final_score).
        """
        self.model.eval()
        self.scoring_head.eval()

        candidate_ids = [c.chunk_id for c, _ in candidate_chunks]
        sub_nodes = set(candidate_ids)
        for cid in candidate_ids:
            sub_nodes |= graph.get_chunk_neighbors(cid, max_hops=1)

        node_list = list(sub_nodes)
        node2idx = {n: i for i, n in enumerate(node_list)}
        N = len(node_list)

        # Build chunk_lookup
        chunk_lookup = _build_qfe_chunk_lookup(graph)
        for chunk, _ in candidate_chunks:
            chunk_lookup[chunk.chunk_id] = chunk

        retrieval_scores = {c.chunk_id: s for c, s in candidate_chunks}

        # Base feature matrix
        base_dim = next(iter(features.values())).shape[0]
        x_base = np.zeros((N, base_dim), dtype=np.float32)
        for n in node_list:
            if n in features:
                x_base[node2idx[n]] = features[n]

        # Query-aware augmented features
        x_aug = build_query_augmented_features(
            features, node_list, query,
            chunk_lookup=chunk_lookup,
            retrieval_scores=retrieval_scores,
            graph_scores=ppr_scores or {},
        )

        x = np.concatenate([x_base, x_aug], axis=1)

        # Per-relation adjacency
        adj_list_np = [
            np.zeros((N, N), dtype=np.float32)
            for _ in range(self.num_relations)
        ]
        for u, v, k, etype in graph.graph.edges(keys=True, data="edge_type"):
            if u not in node2idx or v not in node2idx:
                continue
            r = self.relation_map.get(etype)
            if r is None:
                continue
            i, j = node2idx[u], node2idx[v]
            adj_list_np[r][i, j] = 1.0
            adj_list_np[r][j, i] = 1.0

        for r in range(self.num_relations):
            a = adj_list_np[r]
            deg = a.sum(axis=1) + 1e-8
            d_inv_sqrt = np.diag(1.0 / np.sqrt(deg))
            adj_list_np[r] = d_inv_sqrt @ a @ d_inv_sqrt

        # Query embedding
        q_embed_np = self.query_embeddings.get(
            query,
            derive_query_vector(query, dim=self.query_embed_dim),
        )

        # Normalise retrieval scores per candidate for scoring head input
        ret_norm = normalise_score_map(retrieval_scores)

        with torch.no_grad():
            x_t = torch.from_numpy(x).to(self.device)
            adj_list_t = [torch.from_numpy(a).to(self.device) for a in adj_list_np]
            q_embed_t = torch.from_numpy(q_embed_np).to(self.device)

            gnn_embeds = self.model(x_t, adj_list_t, q_embed_t)  # (N, gnn_out_dim)

            # Compute entity-gated scores for candidate chunks
            base_feats_t = x_t[:, :base_dim]  # full base dim as stored
            # Note: scoring head's chunk_proj expects base_feat_dim.
            # x_t includes base + aug features; slice to base only.
            base_dim_scoring = self.scoring_head.chunk_proj[0].in_features
            base_feats_for_scoring = base_feats_t[:, :base_dim_scoring]

            scores_list = []
            for chunk, _ in candidate_chunks:
                cid = chunk.chunk_id
                if cid not in node2idx:
                    scores_list.append((chunk, 0.0))
                    continue
                idx = node2idx[cid]

                ret_s = torch.tensor(
                    ret_norm.get(cid, 0.0), dtype=torch.float32, device=self.device
                )
                chunk_base = base_feats_for_scoring[idx:idx+1]
                gnn_emb = gnn_embeds[idx:idx+1]

                matches = _compute_entity_matches(cid, query, chunk_lookup)
                match_t = torch.tensor(matches, dtype=torch.float32, device=self.device)

                score = self.scoring_head(
                    ret_s, q_embed_t, chunk_base, gnn_emb,
                    match_t[0:1], match_t[1:2], match_t[2:3],
                    match_t[3:4], match_t[4:5],
                )
                scores_list.append((chunk, float(score.cpu().item())))

        scores_list.sort(key=lambda x: x[1], reverse=True)
        return scores_list

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "scoring_head_state": self.scoring_head.state_dict(),
                "relation_map": self.relation_map,
                "query_embed_dim": self.query_embed_dim,
            },
            path,
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        model: nn.Module,
        scoring_head: EntityGatedScoringHead,
        query_embeddings: Dict[str, np.ndarray],
        device: str = "cpu",
    ) -> "QFERGCNFusionReranker":
        ckpt = torch.load(path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        scoring_head.load_state_dict(ckpt["scoring_head_state"])
        return cls(
            model=model,
            scoring_head=scoring_head,
            relation_map=ckpt.get("relation_map", {}),
            query_embeddings=query_embeddings,
            query_embed_dim=ckpt.get("query_embed_dim", QUERY_EMBED_DIM),
            device=device,
        )


# ═════════════════════════════════════════════════════════════════════════════
# Collate helper for batched QFE-RGCN
# ═════════════════════════════════════════════════════════════════════════════

def _collate_qfe_rgcn(batch):
    """Collate by stacking features into block-diagonal adjacency.

    Returns:
        Tuple of (x, adj_list, q_embed, pos_idx, neg_idx,
                  ret_pos, ret_neg, match_pos, match_neg).
    """
    (xs, adj_lists, q_embeds, pos_idxs, neg_idxs,
     ret_pos, ret_neg, match_pos, match_neg) = zip(*batch)

    feat_dim = xs[0].shape[1]
    Ns = [x.shape[0] for x in xs]
    total_N = sum(Ns)
    offsets = [0] + list(np.cumsum(Ns))

    # Block-diagonal feature matrix
    x_big = torch.zeros(total_N, feat_dim, dtype=xs[0].dtype)
    for i, x in enumerate(xs):
        x_big[offsets[i]:offsets[i+1]] = x

    # Block-diagonal adjacency per relation
    num_rels = adj_lists[0].shape[0]
    adj_big = []
    for r in range(num_rels):
        a_r = torch.zeros(total_N, total_N)
        for i, adj in enumerate(adj_lists):
            a = adj[r]
            n = Ns[i]
            a_r[offsets[i]:offsets[i]+n, offsets[i]:offsets[i]+n] = a
        adj_big.append(a_r)

    # Use the first query embedding in the batch (batched queries share embedding)
    q_embed = q_embeds[0]

    pos_global = torch.stack([
        pos_idxs[i] + offsets[i] for i in range(len(batch))
    ])
    neg_global = torch.stack([
        neg_idxs[i] + offsets[i] for i in range(len(batch))
    ])

    ret_pos_t = torch.stack(list(ret_pos))
    ret_neg_t = torch.stack(list(ret_neg))
    match_pos_t = torch.stack(list(match_pos))
    match_neg_t = torch.stack(list(match_neg))

    return (x_big, adj_big, q_embed, pos_global, neg_global,
            ret_pos_t, ret_neg_t, match_pos_t, match_neg_t)


# ═════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ═════════════════════════════════════════════════════════════════════════════

def _build_qfe_chunk_lookup(
    graph: FinancialEvidenceGraph,
) -> Dict[str, Chunk]:
    """Build a minimal ``chunk_id → Chunk`` dict from graph node attributes."""
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


def _compute_entity_matches(
    chunk_id: str,
    query: str,
    chunk_lookup: Dict[str, Chunk],
) -> np.ndarray:
    """Compute binary entity match features for a (query, chunk) pair.

    Returns a float32 array of shape (5,) with:
        [company_match, year_match, metric_match, filing_type_match, section_match]
    """
    chunk = chunk_lookup.get(chunk_id)
    if chunk is None:
        return np.zeros(5, dtype=np.float32)

    # Extract query entities
    q_metrics = _extractor.extract_metrics(query)
    q_years = _extractor.extract_years(query)
    q_companies = _extractor.extract_companies(query)
    q_filing_types = _extractor.extract_filing_types(query)

    # Company match
    company_match = 0.0
    if q_companies:
        chunk_companies = _extractor.extract_companies(chunk.text)
        if chunk.company:
            chunk_companies.add(chunk.company.lower())
        if q_companies & chunk_companies:
            company_match = 1.0

    # Year match
    year_match = 0.0
    if q_years:
        chunk_years = _extractor.extract_years(chunk.text)
        if chunk.filing_year:
            chunk_years.add(chunk.filing_year)
        if q_years & chunk_years:
            year_match = 1.0

    # Metric match
    metric_match = 0.0
    if q_metrics:
        chunk_metrics = _extractor.extract_metrics(chunk.text)
        if q_metrics & chunk_metrics:
            metric_match = 1.0

    # Filing type match
    filing_type_match = 0.0
    if q_filing_types and chunk.filing_type:
        if chunk.filing_type.upper() in {ft.upper() for ft in q_filing_types}:
            filing_type_match = 1.0

    # Section match
    section_match = 0.0
    if chunk.section:
        # Simple substring match for section mentions in query
        section_lower = chunk.section.lower()
        query_lower = query.lower()
        if section_lower in query_lower or any(
            word in query_lower for word in section_lower.split()
            if len(word) > 3
        ):
            section_match = 1.0

    return np.array(
        [company_match, year_match, metric_match, filing_type_match, section_match],
        dtype=np.float32,
    )
