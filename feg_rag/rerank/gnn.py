"""GNN-based evidence reranker.

Paper plan §8.3: GraphSAGE or R-GCN for evidence reranking.
The GNN scores chunks; the final ranking fuses retrieval, graph, and GNN scores.
"""

from __future__ import annotations

import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from feg_rag.data.chunker import Chunk
from feg_rag.graph.builder import FinancialEvidenceGraph


# ═════════════════════════════════════════════════════════════════════════════
# GNN Model
# ═════════════════════════════════════════════════════════════════════════════

class GraphSAGEReranker(nn.Module):
    """Simple 2-layer GraphSAGE for binary chunk relevance classification.

    This is the recommended first model (§8.3). Upgrade to R-GCN if edge types
    prove important.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        out_dim: int = 64,
        dropout: float = 0.3,
    ):
        super().__init__()
        # We use a simple GCN here as a stand-in for GraphSAGE;
        # swap to SAGEConv from torch_geometric in production.
        self.conv1 = nn.Linear(in_dim, hidden_dim)
        self.conv2 = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(out_dim, 1)  # scalar relevance score

    def forward(
        self, x: torch.Tensor, adj: torch.Tensor
    ) -> torch.Tensor:
        """Message passing over adjacency matrix.

        Args:
            x: Node features (N, in_dim).
            adj: Normalised adjacency (N, N).

        Returns:
            Per-node scores (N, 1).
        """
        # Layer 1: aggregate + transform
        x = adj @ x  # simple mean aggregation
        x = self.conv1(x)
        x = F.relu(x)
        x = self.dropout(x)

        # Layer 2
        x = adj @ x
        x = self.conv2(x)
        x = F.relu(x)
        x = self.dropout(x)

        # Score
        return self.classifier(x)


# ═════════════════════════════════════════════════════════════════════════════
# Dataset
# ═════════════════════════════════════════════════════════════════════════════

class RerankDataset(Dataset):
    """Pairwise ranking dataset: (query_subgraph, pos_chunk_idx, neg_chunk_idx)."""

    def __init__(
        self,
        samples: List[Dict],
        graph: FinancialEvidenceGraph,
        features: Dict[str, np.ndarray],
    ):
        self.samples = samples
        self.graph = graph
        self.features = features

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        # Build subgraph feature matrix for this query (max_hops=2 for richer context)
        sub_nodes = list(
            self.graph.get_chunk_neighbors(s["positive"], max_hops=2)
            | self.graph.get_chunk_neighbors(s["negative"], max_hops=2)
        )
        # Safety cap for any remaining edge cases
        max_sub_nodes = 2000
        if len(sub_nodes) > max_sub_nodes:
            sub_nodes = [s["positive"], s["negative"]] + sub_nodes[:max_sub_nodes - 2]
        node2idx = {n: i for i, n in enumerate(sub_nodes)}
        N = len(sub_nodes)

        # Feature matrix
        feat_dim = next(iter(self.features.values())).shape[0]
        x = np.zeros((N, feat_dim), dtype=np.float32)
        for n in sub_nodes:
            if n in self.features:
                x[node2idx[n]] = self.features[n]

        # Adjacency (undirected, self-loops)
        adj = np.zeros((N, N), dtype=np.float32)
        for n in sub_nodes:
            for _, neighbor in self.graph.graph.edges(n):
                if neighbor in node2idx:
                    i, j = node2idx[n], node2idx[neighbor]
                    adj[i, j] = 1.0
                    adj[j, i] = 1.0
        # Normalise (D^-0.5 A D^-0.5)
        deg = adj.sum(axis=1) + 1e-8
        d_inv_sqrt = np.diag(1.0 / np.sqrt(deg))
        adj_norm = d_inv_sqrt @ adj @ d_inv_sqrt

        pos_idx = node2idx.get(s["positive"], 0)
        neg_idx = node2idx.get(s["negative"], 0)

        return (
            torch.from_numpy(x),
            torch.from_numpy(adj_norm),
            torch.tensor(pos_idx, dtype=torch.long),
            torch.tensor(neg_idx, dtype=torch.long),
        )


# ═════════════════════════════════════════════════════════════════════════════
# Trainer / Reranker
# ═════════════════════════════════════════════════════════════════════════════

class GNNFusionReranker:
    """GNN reranker with score fusion.

    final_score = α * retrieval_score + β * graph_score + γ * gnn_score
    """

    def __init__(
        self,
        model: nn.Module,
        alpha: float = 0.3,
        beta: float = 0.3,
        gamma: float = 0.4,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.device = device

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------

    def fit(
        self,
        train_dataset: RerankDataset,
        val_dataset: Optional[RerankDataset] = None,
        epochs: int = 50,
        lr: float = 0.001,
        batch_size: int = 32,
        verbose: bool = True,
    ) -> List[float]:
        """Train with pairwise margin ranking loss."""
        loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=_collate_gnn,
        )
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        loss_fn = nn.MarginRankingLoss(margin=0.5)

        history: List[float] = []
        self.model.train()

        t_start = time.time()
        n_batches = len(loader)
        print(f"\n{'=' * 55}")
        print(f"  Training GraphSAGE Reranker")
        print(f"  Samples: {len(train_dataset)}  |  Epochs: {epochs}  |  "
              f"Batches/epoch: {n_batches}")
        print(f"  Batch size: {batch_size}  |  Device: {self.device}  |  LR: {lr}")
        print(f"{'=' * 55}")

        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_t0 = time.time()
            for batch_idx, (x, adj, pos_idx, neg_idx) in enumerate(loader):
                x = x.to(self.device)
                adj = adj.to(self.device)
                pos_idx = pos_idx.to(self.device)
                neg_idx = neg_idx.to(self.device)

                scores = self.model(x, adj).squeeze(-1)
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

            # Build progress bar (text only, no tqdm)
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
        """Rerank candidate chunks using fusion of retrieval + graph + GNN scores.

        Args:
            query: The question text.
            candidate_chunks: List of (Chunk, retrieval_score) from initial retrieval.
            graph: The financial evidence graph.
            features: Node features dict.
            ppr_scores: Optional dict of chunk_id → PPR score.

        Returns:
            Reranked list of (Chunk, final_score).
        """
        self.model.eval()

        # Build subgraph from candidate chunks
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

        # Adjacency
        adj = np.zeros((N, N), dtype=np.float32)
        for n in node_list:
            for _, neighbor in graph.graph.edges(n):
                if neighbor in node2idx:
                    i, j = node2idx[n], node2idx[neighbor]
                    adj[i, j] = 1.0
                    adj[j, i] = 1.0
        deg = adj.sum(axis=1) + 1e-8
        d_inv_sqrt = np.diag(1.0 / np.sqrt(deg))
        adj_norm = d_inv_sqrt @ adj @ d_inv_sqrt

        with torch.no_grad():
            x_t = torch.from_numpy(x).to(self.device)
            adj_t = torch.from_numpy(adj_norm).to(self.device)
            gnn_scores = self.model(x_t, adj_t).squeeze(-1).cpu().numpy()

        # Fuse scores
        retrieval_map = {c.chunk_id: s for c, s in candidate_chunks}
        reranked: List[Tuple[Chunk, float]] = []
        for chunk, ret_score in candidate_chunks:
            cid = chunk.chunk_id
            gnn_s = float(gnn_scores[node2idx[cid]]) if cid in node2idx else 0.0
            graph_s = ppr_scores.get(cid, 0.0) if ppr_scores else 0.0

            # Min-max normalise each to [0,1] (done per-query; approximate here)
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
    ) -> "GNNFusionReranker":
        ckpt = torch.load(path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        return cls(
            model=model,
            alpha=ckpt.get("alpha", 0.3),
            beta=ckpt.get("beta", 0.3),
            gamma=ckpt.get("gamma", 0.4),
            device=device,
        )


# ═════════════════════════════════════════════════════════════════════════════
# Collate helper for batched GNN
# ═════════════════════════════════════════════════════════════════════════════

def _collate_gnn(batch):
    """Collate variable-size subgraphs into a block-diagonal batch."""
    xs, adjs, pos_idxs, neg_idxs = zip(*batch)

    feat_dim = xs[0].shape[1]
    Ns = [x.shape[0] for x in xs]
    total_N = sum(Ns)
    offsets = [0] + list(np.cumsum(Ns))

    x_big = torch.zeros(total_N, feat_dim, dtype=xs[0].dtype)
    for i, x in enumerate(xs):
        x_big[offsets[i]:offsets[i + 1]] = x

    adj_big = torch.zeros(total_N, total_N)
    for i, adj in enumerate(adjs):
        n = Ns[i]
        adj_big[offsets[i]:offsets[i] + n, offsets[i]:offsets[i] + n] = adj

    pos_global = torch.stack([
        pos_idxs[i] + offsets[i] for i in range(len(batch))
    ])
    neg_global = torch.stack([
        neg_idxs[i] + offsets[i] for i in range(len(batch))
    ])

    return x_big, adj_big, pos_global, neg_global
