"""R-GCN (Relational Graph Convolutional Network) evidence reranker.

Paper design §5.4: R-GCN handles multiple edge types in the financial evidence
graph, making it suitable for heterogeneous graphs with distinct relations like
chunk-mentions-metric, chunk-mentions-year, same-metric, etc.
"""

from __future__ import annotations

import pickle
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from feg_rag.data.chunker import Chunk
from feg_rag.graph.builder import FinancialEvidenceGraph
from feg_rag.rerank.query_features import (
    QUERY_FEATURE_DIM,
    build_query_augmented_features,
)
from feg_rag.rerank.scoring import normalise_score_map


# ═════════════════════════════════════════════════════════════════════════════
# R-GCN Layer
# ═════════════════════════════════════════════════════════════════════════════

class RGCNLayer(nn.Module):
    """Single R-GCN layer: one weight matrix per relation type.

    h_i^(l+1) = σ( W_0^(l) h_i^(l) + Σ_r Σ_j∈N_i^r (1/c_i,r) W_r^(l) h_j^(l) )

    Notes:
        * The adjacency matrices in ``adj_list`` are expected to be
          **pre-normalised** (e.g. D^-0.5 A D^-0.5) by the caller.
        * ``norm_list`` is accepted for API compatibility but is **not used**
          internally — normalisation is folded into the adjacency.
        * Self-loop is handled by ``self_loop`` (a dedicated ``nn.Linear``),
          not by an identity relation in ``adj_list``.  This keeps the
          implementation stable regardless of how relation adjacencies are
          constructed.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_relations: int,
        dropout: float = 0.3,
        use_bias: bool = True,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_relations = num_relations

        # Self-loop weight (W_0)
        self.self_loop = nn.Linear(in_dim, out_dim, bias=False)

        # Per-relation weights
        self.rel_weights = nn.ModuleList([
            nn.Linear(in_dim, out_dim, bias=False)
            for _ in range(num_relations)
        ])

        self.dropout = nn.Dropout(dropout)
        self.bias = nn.Parameter(torch.zeros(out_dim)) if use_bias else None

    def forward(
        self,
        x: torch.Tensor,
        adj_list: List[torch.Tensor],  # [num_relations * (N, N)] — pre-normalised
        norm_list: List[torch.Tensor],  # [num_relations * (N,)] — accepted but NOT used
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Node features (N, in_dim).
            adj_list: List of **pre-normalised** adjacency matrices, one per
                relation type, each (N, N).
            norm_list: Per-node normalisation scalars per relation.
                **Accepted for API compatibility but not used** —
                normalisation is assumed to be folded into adj_list by the
                dataset / rerank methods.

        Returns:
            Updated node features (N, out_dim).
        """
        out = self.self_loop(x)  # self-loop contribution

        for r in range(self.num_relations):
            if adj_list[r].numel() == 0:
                continue
            # Message passing: aggregate neighbours per relation
            support = self.rel_weights[r](x)  # (N, out_dim)
            messages = adj_list[r] @ support     # (N, out_dim)
            out += messages

        out = F.relu(out)
        out = self.dropout(out)
        if self.bias is not None:
            out = out + self.bias
        return out


# ═════════════════════════════════════════════════════════════════════════════
# R-GCN Reranker Model
# ═════════════════════════════════════════════════════════════════════════════

class RGCNReranker(nn.Module):
    """2-layer R-GCN for binary chunk relevance classification."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        out_dim: int = 64,
        num_relations: int = 8,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.conv1 = RGCNLayer(in_dim, hidden_dim, num_relations, dropout)
        self.conv2 = RGCNLayer(hidden_dim, out_dim, num_relations, dropout)
        self.classifier = nn.Linear(out_dim, 1)  # scalar relevance score

    def forward(
        self,
        x: torch.Tensor,
        adj_list: List[torch.Tensor],
    ) -> torch.Tensor:
        """Two-layer message passing.

        Args:
            x: Node features (N, in_dim).
            adj_list: List of **pre-normalised** adjacency matrices per
                relation type.

        Returns:
            Per-node scores (N, 1).
        """
        # Use identity norm_list (normalisation folded into adj_list by caller).
        # See RGCNLayer.forward docstring for rationale.
        norm_list = [torch.ones(a.shape[0], device=x.device) if a.numel() > 0
                     else torch.tensor([], device=x.device) for a in adj_list]

        x = self.conv1(x, adj_list, norm_list)
        x = self.conv2(x, adj_list, norm_list)
        return self.classifier(x)


# ═════════════════════════════════════════════════════════════════════════════
# R-GCN Lite (shared-weight + scalar-gate, compute-friendly baseline)
# ═════════════════════════════════════════════════════════════════════════════

class LiteRGCNLayer(nn.Module):
    """Lightweight R-GCN layer with shared transformation + per-relation scalar gates.

    Instead of one weight matrix per relation (vanilla R-GCN):

        h' = W_self h + sum_r A_r W_r h

    Lite uses a single shared weight plus a learnable scalar gate per relation:

        h' = W_self h + sum_r g_r * A_r W_shared h

    This cuts parameters from O(num_relations * in_dim * out_dim) down to
    O(in_dim * out_dim + num_relations), making training and inference
    substantially faster while preserving multi-relational modelling capacity.

    Notes:
        * Adjacency matrices in ``adj_list`` are expected to be
          **pre-normalised** (e.g. D^-0.5 A D^-0.5) by the caller.
        * ``norm_list`` is accepted for API compatibility but is **not used**
          internally.
        * ``use_bias`` controls an optional bias term; the per-relation gates
          ``g_r`` are always present.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_relations: int,
        dropout: float = 0.3,
        use_bias: bool = True,
        use_relation_bias: bool = False,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_relations = num_relations

        # Self-loop weight (W_0) — kept separate for stability
        self.self_loop = nn.Linear(in_dim, out_dim, bias=False)

        # Shared transformation applied to all relations
        self.shared = nn.Linear(in_dim, out_dim, bias=False)

        # Per-relation learnable scalar gates
        self.gates = nn.Parameter(torch.ones(num_relations))

        # Optional per-relation bias (kept small to limit parameter growth)
        self.relation_bias = (
            nn.Parameter(torch.zeros(num_relations, out_dim))
            if use_relation_bias
            else None
        )

        self.dropout = nn.Dropout(dropout)
        self.bias = nn.Parameter(torch.zeros(out_dim)) if use_bias else None

    def forward(
        self,
        x: torch.Tensor,
        adj_list: List[torch.Tensor],  # [num_relations * (N, N)] — pre-normalised
        norm_list: List[torch.Tensor],  # [num_relations * (N,)] — accepted but NOT used
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Node features (N, in_dim).
            adj_list: List of **pre-normalised** adjacency matrices, one per
                relation type, each (N, N).
            norm_list: Per-node normalisation scalars per relation.
                **Accepted for API compatibility but not used**.

        Returns:
            Updated node features (N, out_dim).
        """
        out = self.self_loop(x)  # self-loop contribution
        shared_support = self.shared(x)  # (N, out_dim) — computed once

        for r in range(self.num_relations):
            if adj_list[r].numel() == 0:
                continue
            # g_r * A_r @ W_shared h
            gate = self.gates[r]
            messages = gate * (adj_list[r] @ shared_support)  # (N, out_dim)
            if self.relation_bias is not None:
                messages = messages + self.relation_bias[r]
            out = out + messages

        out = F.relu(out)
        out = self.dropout(out)
        if self.bias is not None:
            out = out + self.bias
        return out


class LiteRGCNReranker(nn.Module):
    """2-layer Lite R-GCN for binary chunk relevance classification.

    A lightweight variant of :class:`RGCNReranker` that uses
    :class:`LiteRGCNLayer` (shared transformation + scalar gates) instead of
    per-relation weight matrices.  Designed as a compute-friendly efficiency
    baseline that keeps the same evaluation protocol as vanilla R-GCN.

    Default hidden/out dims are smaller than vanilla R-GCN (96/48 vs 128/64)
    to keep the model lean.  Both are configurable via the constructor.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 96,
        out_dim: int = 48,
        num_relations: int = 8,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.conv1 = LiteRGCNLayer(in_dim, hidden_dim, num_relations, dropout)
        self.conv2 = LiteRGCNLayer(hidden_dim, out_dim, num_relations, dropout)
        self.classifier = nn.Linear(out_dim, 1)

        # Store for introspection
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

    def forward(
        self,
        x: torch.Tensor,
        adj_list: List[torch.Tensor],
    ) -> torch.Tensor:
        """Two-layer message passing.

        Args:
            x: Node features (N, in_dim).
            adj_list: List of **pre-normalised** adjacency matrices per
                relation type.

        Returns:
            Per-node scores (N, 1).
        """
        norm_list = [torch.ones(a.shape[0], device=x.device) if a.numel() > 0
                     else torch.tensor([], device=x.device) for a in adj_list]

        x = self.conv1(x, adj_list, norm_list)
        x = self.conv2(x, adj_list, norm_list)
        return self.classifier(x)


# ═════════════════════════════════════════════════════════════════════════════
# Dataset for R-GCN training
# ═════════════════════════════════════════════════════════════════════════════

class RGCNRerankDataset(Dataset):
    """Pairwise ranking dataset with per-relation adjacency matrices."""

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
        self._chunk_lookup = chunk_lookup or _build_rgcn_chunk_lookup(graph)

        # Build relation → integer index mapping
        if relation_map is None:
            etypes = set()
            for _u, _v, _k, etype in graph.graph.edges(keys=True, data="edge_type"):
                etypes.add(etype)
            self.relation_map = {et: i for i, et in enumerate(sorted(etypes))}
        else:
            self.relation_map = relation_map

        self.num_relations = max(self.relation_map.values()) + 1 if self.relation_map else 1

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        pos_id = s.get("positive", s.get("chunk_id", ""))
        neg_id = s.get("negative", "")

        # Build subgraph
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

        # Query-aware augmented features (same logic as GraphSAGE)
        question = s.get("question", "")
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
            # Unknown relation types (not in relation_map) are skipped to
            # avoid silently polluting relation 0.  They do not contribute
            # to any relation adjacency.
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

        pos_idx = node2idx.get(pos_id, 0)
        neg_idx = node2idx.get(neg_id, 0)

        return (
            torch.from_numpy(x),
            torch.stack([torch.from_numpy(a) for a in adj_list]),
            torch.tensor(pos_idx, dtype=torch.long),
            torch.tensor(neg_idx, dtype=torch.long),
        )


# ═════════════════════════════════════════════════════════════════════════════
# R-GCN Fusion Reranker (same API as GNNFusionReranker)
# ═════════════════════════════════════════════════════════════════════════════

class RGCNFusionReranker:
    """R-GCN reranker with score fusion.

    final_score = α * retrieval_norm + β * graph_norm + γ * gnn_norm

    Each score component is min-max normalised within the current query's
    candidate set before fusion.  alpha/beta/gamma are linear fusion weights
    and are NOT required to sum to 1.
    """

    def __init__(
        self,
        model: nn.Module,
        relation_map: Dict[str, int],
        alpha: float = 0.3,
        beta: float = 0.3,
        gamma: float = 0.4,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.relation_map = relation_map
        self.num_relations = max(relation_map.values()) + 1 if relation_map else 1
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.device = device

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------

    def fit(
        self,
        train_dataset: RGCNRerankDataset,
        val_dataset: Optional[RGCNRerankDataset] = None,
        epochs: int = 50,
        lr: float = 0.001,
        batch_size: int = 32,
        verbose: bool = True,
    ) -> List[float]:
        """Train with pairwise margin ranking loss."""
        loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, collate_fn=_collate_rgcn
        )
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        loss_fn = nn.MarginRankingLoss(margin=0.5)

        history: List[float] = []
        self.model.train()

        t_start = time.time()
        n_batches = len(loader)
        print(f"\n{'=' * 55}")
        print(f"  Training R-GCN Reranker")
        print(f"  Samples: {len(train_dataset)}  |  Epochs: {epochs}  |  "
              f"Batches/epoch: {n_batches}")
        print(f"  Relations: {self.num_relations}  |  Batch size: {batch_size}  |  "
              f"Device: {self.device}  |  LR: {lr}")
        print(f"{'=' * 55}")

        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_t0 = time.time()
            for batch_idx, (x, adj_list, pos_idx, neg_idx) in enumerate(loader):
                x = x.to(self.device)
                adj_list = [a.to(self.device) for a in adj_list]
                pos_idx = pos_idx.to(self.device)
                neg_idx = neg_idx.to(self.device)

                scores = self.model(x, adj_list).squeeze(-1)  # (total_N,)

                # pos_idx/neg_idx are already global indices (collate adds offsets)
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

            # Build progress bar (text only)
            pct_done = (epoch + 1) / epochs
            bar_len = 20
            filled = int(pct_done * bar_len)
            bar = "#" * filled + "-" * (bar_len - filled)

            # Delta from previous epoch
            delta_str = ""
            if epoch > 0:
                delta = avg_loss - history[epoch - 1]
                sign = "v" if delta < 0 else "^"
                delta_str = f"  d={sign}{abs(delta):.4f}"

            # Timing
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
        """Rerank candidate chunks with R-GCN + fusion scoring.

        All three score components are min-max normalised within the candidate
        set before fusion, preventing raw-logit dominance.

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

        candidate_ids = [c.chunk_id for c, _ in candidate_chunks]
        sub_nodes = set(candidate_ids)
        for cid in candidate_ids:
            sub_nodes |= graph.get_chunk_neighbors(cid, max_hops=1)

        node_list = list(sub_nodes)
        node2idx = {n: i for i, n in enumerate(node_list)}
        N = len(node_list)

        # Build chunk_lookup
        chunk_lookup = _build_rgcn_chunk_lookup(graph)
        for chunk, _ in candidate_chunks:
            chunk_lookup[chunk.chunk_id] = chunk

        retrieval_scores = {c.chunk_id: s for c, s in candidate_chunks}

        # Base feature matrix
        base_dim = next(iter(features.values())).shape[0]
        x_base = np.zeros((N, base_dim), dtype=np.float32)
        for n in node_list:
            if n in features:
                x_base[node2idx[n]] = features[n]

        # Query-aware augmented features (same logic as training)
        x_aug = build_query_augmented_features(
            features, node_list, query,
            chunk_lookup=chunk_lookup,
            retrieval_scores=retrieval_scores,
            graph_scores=ppr_scores or {},
        )

        # Concatenate base + query features
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
            # Unknown relation types are skipped — they do NOT silently map to
            # relation 0, preserving the integrity of known relations.
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

        with torch.no_grad():
            x_t = torch.from_numpy(x).to(self.device)
            adj_list_t = [torch.from_numpy(a).to(self.device) for a in adj_list_np]
            gnn_logits = self.model(x_t, adj_list_t).squeeze(-1).cpu().numpy()

        # Build per-candidate score maps
        ret_map = {c.chunk_id: s for c, s in candidate_chunks}
        graph_map = ppr_scores or {}
        gnn_map = {
            node_list[i]: float(gnn_logits[i])
            for i in range(N) if node_list[i] in ret_map
        }

        # Normalise each score component within candidates
        ret_norm = normalise_score_map(ret_map)
        graph_norm = normalise_score_map(graph_map)
        gnn_norm = normalise_score_map(gnn_map)

        # Fusion with normalised scores
        reranked: List[Tuple[Chunk, float]] = []
        for chunk, _ in candidate_chunks:
            cid = chunk.chunk_id
            rn = ret_norm.get(cid, 0.0)
            gn = graph_norm.get(cid, 0.0)
            gnn_n = gnn_norm.get(cid, 0.0)
            final = self.alpha * rn + self.beta * gn + self.gamma * gnn_n
            reranked.append((chunk, final))

        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "relation_map": self.relation_map,
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
        model: nn.Module,
        device: str = "cpu",
    ) -> "RGCNFusionReranker":
        ckpt = torch.load(path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        return cls(
            model=model,
            relation_map=ckpt.get("relation_map", {}),
            alpha=ckpt.get("alpha", 0.3),
            beta=ckpt.get("beta", 0.3),
            gamma=ckpt.get("gamma", 0.4),
            device=device,
        )


# ═════════════════════════════════════════════════════════════════════════════
# Collate helper for batched R-GCN
# ═════════════════════════════════════════════════════════════════════════════

def _collate_rgcn(batch):
    """Collate by stacking features into a large block-diagonal adjacency."""
    xs, adj_lists, pos_idxs, neg_idxs = zip(*batch)

    # Pad to same feature dim
    feat_dim = xs[0].shape[1]
    Ns = [x.shape[0] for x in xs]
    total_N = sum(Ns)
    offsets = [0] + list(np.cumsum(Ns))

    # Build block-diagonal feature matrix
    x_big = torch.zeros(total_N, feat_dim, dtype=xs[0].dtype)
    for i, x in enumerate(xs):
        x_big[offsets[i]:offsets[i+1]] = x

    # Build block-diagonal adjacency per relation
    num_rels = adj_lists[0].shape[0]
    adj_big = []
    for r in range(num_rels):
        a_r = torch.zeros(total_N, total_N)
        for i, adj in enumerate(adj_lists):
            a = adj[r]
            n = Ns[i]
            a_r[offsets[i]:offsets[i]+n, offsets[i]:offsets[i]+n] = a
        adj_big.append(a_r)

    # Adjust pos/neg indices for global offset
    pos_global = torch.stack([
        pos_idxs[i] + offsets[i] for i in range(len(batch))
    ])
    neg_global = torch.stack([
        neg_idxs[i] + offsets[i] for i in range(len(batch))
    ])

    return x_big, adj_big, pos_global, neg_global


# ═════════════════════════════════════════════════════════════════════════════
# Internal helper
# ═════════════════════════════════════════════════════════════════════════════

def _build_rgcn_chunk_lookup(
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
