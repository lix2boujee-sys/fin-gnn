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
from feg_rag.rerank.query_features import (
    QUERY_FEATURE_DIM,
    build_query_augmented_features,
)
from feg_rag.rerank.scoring import normalise_score_map


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


class DenseGATv2Layer(nn.Module):
    """Dense GATv2-style attention layer for small query subgraphs.

    This avoids a torch-geometric dependency while keeping the key GATv2 idea:
    attention depends on a non-linear combination of source and target node
    projections. The implementation is chunked over source rows to avoid
    constructing an enormous N x N x heads x dim tensor at once.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        heads: int = 4,
        dropout: float = 0.3,
        concat: bool = True,
        row_chunk_size: int = 256,
    ):
        super().__init__()
        self.out_dim = out_dim
        self.heads = heads
        self.concat = concat
        self.row_chunk_size = row_chunk_size
        self.lin_src = nn.Linear(in_dim, heads * out_dim, bias=False)
        self.lin_dst = nn.Linear(in_dim, heads * out_dim, bias=False)
        self.att = nn.Parameter(torch.empty(heads, out_dim))
        self.bias = nn.Parameter(torch.zeros(heads * out_dim if concat else out_dim))
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.lin_src.weight)
        nn.init.xavier_uniform_(self.lin_dst.weight)
        nn.init.xavier_uniform_(self.att)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        n_nodes = x.shape[0]
        src = self.lin_src(x).view(n_nodes, self.heads, self.out_dim)
        dst = self.lin_dst(x).view(n_nodes, self.heads, self.out_dim)
        mask = adj > 0

        outputs = []
        for start in range(0, n_nodes, self.row_chunk_size):
            end = min(start + self.row_chunk_size, n_nodes)
            pair = self.leaky_relu(src[start:end, None, :, :] + dst[None, :, :, :])
            logits = torch.einsum("bnhd,hd->bnh", pair, self.att)
            logits = logits.masked_fill(~mask[start:end, :, None], torch.finfo(logits.dtype).min)
            alpha = torch.softmax(logits, dim=1)
            alpha = self.dropout(alpha)
            out = torch.einsum("bnh,nhd->bhd", alpha, dst)
            outputs.append(out)

        h = torch.cat(outputs, dim=0)
        if self.concat:
            h = h.reshape(n_nodes, self.heads * self.out_dim)
        else:
            h = h.mean(dim=1)
        return h + self.bias


class GATv2Reranker(nn.Module):
    """2-layer dense GATv2 baseline for evidence reranking.

    The residual projections keep each candidate passage's own retrieval/entity
    features available after attention aggregation.  This makes GATv2 a stronger
    baseline than a pure neighbourhood smoother while still remaining a generic
    untyped graph-attention model.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        out_dim: int = 32,
        heads: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        hidden_per_head = max(1, hidden_dim // heads)
        out_per_head = max(1, out_dim // heads)
        self.conv1 = DenseGATv2Layer(
            in_dim, hidden_per_head, heads=heads, dropout=dropout, concat=True
        )
        self.conv2 = DenseGATv2Layer(
            hidden_per_head * heads, out_per_head, heads=heads, dropout=dropout, concat=True
        )
        self.res1 = nn.Linear(in_dim, hidden_per_head * heads, bias=False)
        self.res2 = nn.Linear(hidden_per_head * heads, out_per_head * heads, bias=False)
        self.norm1 = nn.LayerNorm(hidden_per_head * heads)
        self.norm2 = nn.LayerNorm(out_per_head * heads)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(out_per_head * heads, 1)
        self.in_dim = in_dim
        self.hidden_dim = hidden_per_head * heads
        self.out_dim = out_per_head * heads
        self.heads = heads

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x, adj) + self.res1(x)
        x = self.norm1(F.elu(h))
        x = self.dropout(x)
        h = self.conv2(x, adj) + self.res2(x)
        x = self.norm2(F.elu(h))
        x = self.dropout(x)
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
        chunk_lookup: Optional[Dict[str, Chunk]] = None,
    ):
        self.samples = samples
        self.graph = graph
        self.features = features
        self._chunk_lookup = chunk_lookup or _build_chunk_lookup_from_graph(graph)

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
            priority = [s["positive"], s["negative"]]
            seen = set()
            capped = []
            for n in priority + sub_nodes:
                if n in seen:
                    continue
                seen.add(n)
                capped.append(n)
                if len(capped) >= max_sub_nodes:
                    break
            sub_nodes = capped
        node2idx = {n: i for i, n in enumerate(sub_nodes)}
        N = len(sub_nodes)

        # Base feature matrix
        base_dim = next(iter(self.features.values())).shape[0]
        x_base = np.zeros((N, base_dim), dtype=np.float32)
        for n in sub_nodes:
            if n in self.features:
                x_base[node2idx[n]] = self.features[n]

        # Query-aware augmented features (training/inference use same logic)
        question = s.get("question", "")
        ret_scores = s.get("retrieval_scores", None) or {}
        graph_scores = s.get("graph_scores", None) or {}
        x_aug = build_query_augmented_features(
            self.features, sub_nodes, question,
            chunk_lookup=self._chunk_lookup,
            retrieval_scores=ret_scores,
            graph_scores=graph_scores,
        )

        # Concatenate base + query features
        x = np.concatenate([x_base, x_aug], axis=1)

        # Adjacency (undirected, with explicit self-loops)
        adj = np.zeros((N, N), dtype=np.float32)
        for n in sub_nodes:
            for _, neighbor in self.graph.graph.edges(n):
                if neighbor in node2idx:
                    i, j = node2idx[n], node2idx[neighbor]
                    adj[i, j] = 1.0
                    adj[j, i] = 1.0

        # Explicit self-loop: preserve chunk's own features during message passing
        np.fill_diagonal(adj, 1.0)

        # Normalise (D^-0.5 A D^-0.5) — self-loop ensures diagonal > 0
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

    final_score = α * retrieval_norm + β * graph_norm + γ * gnn_norm

    Each score component is min-max normalised within the current query's
    candidate set before fusion.  alpha/beta/gamma are linear fusion weights
    and are NOT required to sum to 1.
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

        All three score components are min-max normalised within the candidate
        set before fusion, preventing raw-logit dominance.

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

        # Build chunk_lookup from candidate_chunks + graph nodes
        chunk_lookup = _build_chunk_lookup_from_graph(graph)
        # Override with actual Chunk objects from candidates
        for chunk, _ in candidate_chunks:
            chunk_lookup[chunk.chunk_id] = chunk

        # Retrieval scores from candidate_chunks
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

        # Adjacency (undirected, with explicit self-loops — same as training)
        adj = np.zeros((N, N), dtype=np.float32)
        for n in node_list:
            for _, neighbor in graph.graph.edges(n):
                if neighbor in node2idx:
                    i, j = node2idx[n], node2idx[neighbor]
                    adj[i, j] = 1.0
                    adj[j, i] = 1.0
        np.fill_diagonal(adj, 1.0)
        deg = adj.sum(axis=1) + 1e-8
        d_inv_sqrt = np.diag(1.0 / np.sqrt(deg))
        adj_norm = d_inv_sqrt @ adj @ d_inv_sqrt

        with torch.no_grad():
            x_t = torch.from_numpy(x).to(self.device)
            adj_t = torch.from_numpy(adj_norm).to(self.device)
            gnn_logits = self.model(x_t, adj_t).squeeze(-1).cpu().numpy()

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

        # Fusion with normalised scores (alpha/beta/gamma are linear weights)
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


# ═════════════════════════════════════════════════════════════════════════════
# Internal helper: build minimal chunk lookup from graph node attributes
# ═════════════════════════════════════════════════════════════════════════════

def _build_chunk_lookup_from_graph(
    graph: FinancialEvidenceGraph,
) -> Dict[str, Chunk]:
    """Build a minimal ``chunk_id → Chunk`` dict from graph node attributes.

    Used as a fallback when the caller does not provide a full chunk list.
    Node attributes for non-chunk types may be incomplete; query feature
    matching degrades gracefully in that case.
    """
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
