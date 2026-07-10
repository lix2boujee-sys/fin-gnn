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
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from feg_rag.data.chunker import Chunk
from feg_rag.graph.builder import FinancialEvidenceGraph


# ═════════════════════════════════════════════════════════════════════════════
# R-GCN Layer
# ═════════════════════════════════════════════════════════════════════════════

class RGCNLayer(nn.Module):
    """Single R-GCN layer: one weight matrix per relation type.

    h_i^(l+1) = σ( W_0^(l) h_i^(l) + Σ_r Σ_j∈N_i^r (1/c_i,r) W_r^(l) h_j^(l) )
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
        adj_list: List[torch.Tensor],  # [num_relations * (N, N)] — sparse or dense
        norm_list: List[torch.Tensor],  # [num_relations * (N,)]
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Node features (N, in_dim).
            adj_list: List of adjacency matrices, one per relation type, each (N, N).
            norm_list: Per-node normalisation scalars per relation.

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
            adj_list: List of adjacency matrices per relation type.

        Returns:
            Per-node scores (N, 1).
        """
        # Use identity norm_list (normalisation folded into adj_list)
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
    ):
        self.samples = samples
        self.graph = graph
        self.features = features

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

        # Feature matrix
        feat_dim = next(iter(self.features.values())).shape[0]
        x = np.zeros((N, feat_dim), dtype=np.float32)
        for n in node_list:
            if n in self.features:
                x[node2idx[n]] = self.features[n]

        # Per-relation adjacency matrices
        adj_list = [
            np.zeros((N, N), dtype=np.float32)
            for _ in range(self.num_relations)
        ]
        for u, v, k, etype in self.graph.graph.edges(keys=True, data="edge_type"):
            if u in node2idx and v in node2idx:
                r = self.relation_map.get(etype, 0)
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

    final_score = α * retrieval_score + β * graph_score + γ * gnn_score
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

        # Feature matrix
        feat_dim = next(iter(features.values())).shape[0]
        x = np.zeros((N, feat_dim), dtype=np.float32)
        for n in node_list:
            if n in features:
                x[node2idx[n]] = features[n]

        # Per-relation adjacency
        adj_list_np = [
            np.zeros((N, N), dtype=np.float32)
            for _ in range(self.num_relations)
        ]
        for u, v, k, etype in graph.graph.edges(keys=True, data="edge_type"):
            if u in node2idx and v in node2idx:
                r = self.relation_map.get(etype, 0)
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
            gnn_scores = self.model(x_t, adj_list_t).squeeze(-1).cpu().numpy()

        # Fuse scores
        reranked: List[Tuple[Chunk, float]] = []
        for chunk, ret_score in candidate_chunks:
            cid = chunk.chunk_id
            gnn_s = float(gnn_scores[node2idx[cid]]) if cid in node2idx else 0.0
            graph_s = ppr_scores.get(cid, 0.0) if ppr_scores else 0.0

            final = (
                self.alpha * ret_score
                + self.beta * graph_s
                + self.gamma * gnn_s
            )
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
