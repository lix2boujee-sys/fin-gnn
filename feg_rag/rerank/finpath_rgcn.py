"""FinPath-RGCN: path-augmented financial evidence reranking.

This module intentionally leaves the vanilla R-GCN message passing untouched.
It consumes R-GCN scores/embeddings as backbone signals and adds a learnable
typed-path correction branch.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from feg_rag.rerank.path_encoder import (
    PATH_FEATURE_KEYS,
    FinancialPath,
    FinancialPathExtractor,
    LearnablePathEncoder,
    PathAggregator,
    build_path_vocab,
    compute_path_features,
    tensorize_paths,
)


QUERY_ENTITY_KEYS = ["company", "year", "metric", "filing", "section"]


@dataclass
class FinPathScoreBreakdown:
    chunk_id: str
    final_score: float
    rgcn_score: float
    retrieval_score: float
    path_correction: float
    top_attended_path_type: Optional[str]
    match_flags: Dict[str, int]
    conflict_flags: Dict[str, int]
    path_string: str


def set_finpath_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def query_entities_to_embedding(
    query_entities: Optional[Dict[str, Any]],
    hidden_dim: int,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Small deterministic query embedding from entity presence/counts.

    This keeps the FinPath branch usable without requiring a text encoder.
    External query embeddings can still be passed directly to ``score_tensors``.
    """

    entities = query_entities or {}

    def _count(*keys: str) -> float:
        vals: List[Any] = []
        for key in keys:
            value = entities.get(key)
            if value is None:
                continue
            if isinstance(value, (list, tuple, set)):
                vals.extend(value)
            elif value:
                vals.append(value)
        return float(len([v for v in vals if str(v).strip()]))

    base = torch.tensor(
        [
            float(_count("company", "companies") > 0),
            min(_count("year", "years"), 5.0) / 5.0,
            min(_count("metric", "metrics"), 5.0) / 5.0,
            float(_count("filing_type", "filing_types") > 0),
            float(_count("section_hint", "sections") > 0),
        ],
        dtype=torch.float32,
        device=device,
    )
    if hidden_dim <= len(base):
        return base[:hidden_dim]
    return torch.cat([base, torch.zeros(hidden_dim - len(base), device=device)])


def _as_tensor(values: Sequence[float], device: str | torch.device) -> torch.Tensor:
    return torch.tensor([float(v) for v in values], dtype=torch.float32, device=device)


def _path_flags(paths: Sequence[FinancialPath]) -> Tuple[Dict[str, int], Dict[str, int], str, str]:
    match: Dict[str, int] = {}
    conflict: Dict[str, int] = {}
    for path in paths:
        for k, v in path.match_flags.items():
            match[k] = int(match.get(k, 0) or v)
        for k, v in path.conflict_flags.items():
            conflict[k] = int(conflict.get(k, 0) or v)
    top_path = paths[0] if paths else None
    return match, conflict, (top_path.path_type if top_path else "none"), (top_path.to_string() if top_path else "")


class FinPathRGCNReranker(nn.Module):
    """R-GCN backbone plus learnable typed financial path correction."""

    def __init__(
        self,
        vocab: Dict[str, Dict[str, int]],
        hidden_dim: int = 128,
        dropout: float = 0.1,
        fusion_mode: str = "residual",
        tau: float = 0.2,
        rgcn_embedding_dim: int = 0,
        max_paths_per_chunk: int = 8,
        max_path_len: int = 4,
        device: str = "cpu",
    ):
        super().__init__()
        if fusion_mode not in {"residual", "concat_mlp"}:
            raise ValueError("fusion_mode must be 'residual' or 'concat_mlp'")
        self.vocab = vocab
        self.hidden_dim = hidden_dim
        self.fusion_mode = fusion_mode
        self.tau = float(tau)
        self.rgcn_embedding_dim = int(rgcn_embedding_dim)
        self.max_paths_per_chunk = max_paths_per_chunk
        self.max_path_len = max_path_len
        self.device_name = device

        self.path_encoder = LearnablePathEncoder(
            num_relations=max(vocab["relation"].values()) + 1,
            num_node_types=max(vocab["node_type"].values()) + 1,
            num_path_types=max(vocab["path_type"].values()) + 1,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.aggregator = PathAggregator(hidden_dim=hidden_dim, query_dim=hidden_dim)

        correction_in = 1 + hidden_dim + len(PATH_FEATURE_KEYS) + self.rgcn_embedding_dim
        self.correction_mlp = nn.Sequential(
            nn.Linear(correction_in, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        concat_in = correction_in + 1
        self.concat_mlp = nn.Sequential(
            nn.Linear(concat_in, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self._init_score_preserving_residual()
        self.to(device)

    def _init_score_preserving_residual(self) -> None:
        """Start residual mode exactly at the R-GCN backbone score."""

        last = self.correction_mlp[-1]
        if isinstance(last, nn.Linear):
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    @staticmethod
    def _validate_score_inputs(
        candidate_chunk_ids: Sequence[str],
        retrieval_scores: Sequence[float],
        rgcn_scores: Sequence[float],
    ) -> None:
        n = len(candidate_chunk_ids)
        if len(retrieval_scores) != n or len(rgcn_scores) != n:
            raise ValueError(
                "FinPath input length mismatch: "
                f"{n} candidates, {len(retrieval_scores)} retrieval scores, "
                f"{len(rgcn_scores)} R-GCN scores"
            )

    def score_tensors(
        self,
        retrieval_scores: torch.Tensor,
        rgcn_scores: torch.Tensor,
        paths_by_chunk: Sequence[Sequence[FinancialPath]],
        query_embedding: torch.Tensor,
        rgcn_embeddings: Optional[torch.Tensor] = None,
        use_path_features: bool = True,
        return_debug: bool = False,
    ) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
        device = rgcn_scores.device
        if len(paths_by_chunk) == 0:
            return torch.empty(0, dtype=torch.float32, device=device), []
        query_embedding = query_embedding.to(device)
        if rgcn_embeddings is None and self.rgcn_embedding_dim > 0:
            rgcn_embeddings = torch.zeros(
                (len(paths_by_chunk), self.rgcn_embedding_dim), dtype=torch.float32, device=device
            )
        elif rgcn_embeddings is not None:
            rgcn_embeddings = rgcn_embeddings.to(device)

        finals: List[torch.Tensor] = []
        debug_rows: List[Dict[str, Any]] = []
        inv_path_type = {v: k for k, v in self.vocab["path_type"].items()}

        for i, paths in enumerate(paths_by_chunk):
            tensors = tensorize_paths(paths, self.vocab, self.max_paths_per_chunk, self.max_path_len)
            tensors = {k: v.to(device) for k, v in tensors.items()}
            path_embs = self.path_encoder(
                tensors["relation_ids"],
                tensors["src_node_type_ids"],
                tensors["dst_node_type_ids"],
                tensors["path_type_ids"],
                tensors["flag_features"],
                tensors["step_mask"],
            )
            path_repr, attn_debug = self.aggregator(
                path_embs,
                query_embedding,
                tensors["path_mask"],
                tensors["path_type_ids"],
            )
            path_feats_np = compute_path_features(paths, self.max_paths_per_chunk)
            if not use_path_features:
                path_feats_np = np.zeros_like(path_feats_np)
            path_feats = torch.from_numpy(path_feats_np).to(device)
            parts = [retrieval_scores[i].view(1), path_repr, path_feats]
            if self.rgcn_embedding_dim > 0 and rgcn_embeddings is not None:
                parts.append(rgcn_embeddings[i].view(-1))
            corr_in = torch.cat(parts).float()

            if self.fusion_mode == "residual":
                correction = self.tau * torch.tanh(self.correction_mlp(corr_in).squeeze(-1))
                final = rgcn_scores[i] + correction
            else:
                final = self.concat_mlp(torch.cat([rgcn_scores[i].view(1), corr_in])).squeeze(-1)
                correction = final - rgcn_scores[i]
            finals.append(final)

            if return_debug:
                type_id = attn_debug.get("max_attention_path_type")
                debug_rows.append(
                    {
                        "path_correction": float(correction.detach().cpu().item()),
                        "path_features": {k: float(v) for k, v in zip(PATH_FEATURE_KEYS, path_feats_np)},
                        "top_attended_path_type": inv_path_type.get(type_id, None) if type_id is not None else None,
                        "attention_weights": attn_debug.get("attention_weights", []),
                    }
                )

        return torch.stack(finals), debug_rows

    def rerank(
        self,
        query: str,
        query_id: str,
        candidate_chunk_ids: Sequence[str],
        retrieval_scores: Sequence[float],
        graph: Any,
        query_entities: Optional[Dict[str, Any]],
        rgcn_scores: Sequence[float],
        rgcn_embeddings: Optional[np.ndarray | torch.Tensor] = None,
        extractor: Optional[FinancialPathExtractor] = None,
        paths_map: Optional[Dict[str, Sequence[FinancialPath]]] = None,
        use_path_features: bool = True,
        return_debug: bool = False,
    ) -> List[Tuple[str, float]] | Tuple[List[Tuple[str, float]], List[FinPathScoreBreakdown]]:
        del query, query_id
        self._validate_score_inputs(candidate_chunk_ids, retrieval_scores, rgcn_scores)
        if not candidate_chunk_ids:
            return ([], []) if return_debug else []
        extractor = extractor or FinancialPathExtractor(
            max_paths_per_chunk=self.max_paths_per_chunk,
            max_path_len=self.max_path_len,
        )
        if paths_map is None:
            paths_map = extractor.extract_paths(graph, candidate_chunk_ids, query_entities)
        paths_by_chunk = [paths_map.get(cid, []) for cid in candidate_chunk_ids]
        device = next(self.parameters()).device
        ret_t = _as_tensor(retrieval_scores, device)
        rgcn_t = _as_tensor(rgcn_scores, device)
        emb_t = None
        if rgcn_embeddings is not None:
            emb_t = torch.as_tensor(rgcn_embeddings, dtype=torch.float32, device=device)
        query_emb = query_entities_to_embedding(query_entities, self.hidden_dim, device)
        with torch.no_grad():
            scores, debug_rows = self.score_tensors(
                ret_t,
                rgcn_t,
                paths_by_chunk,
                query_emb,
                emb_t,
                use_path_features=use_path_features,
                return_debug=return_debug,
            )
        rows = list(zip(candidate_chunk_ids, scores.detach().cpu().tolist()))
        order = sorted(range(len(rows)), key=lambda idx: rows[idx][1], reverse=True)
        reranked = [(rows[i][0], float(rows[i][1])) for i in order]
        if not return_debug:
            return reranked

        breakdowns: List[FinPathScoreBreakdown] = []
        for i in order:
            paths = paths_by_chunk[i]
            match, conflict, top_type, path_string = _path_flags(paths)
            dbg = debug_rows[i] if i < len(debug_rows) else {}
            breakdowns.append(
                FinPathScoreBreakdown(
                    chunk_id=candidate_chunk_ids[i],
                    final_score=float(rows[i][1]),
                    rgcn_score=float(rgcn_scores[i]),
                    retrieval_score=float(retrieval_scores[i]),
                    path_correction=float(dbg.get("path_correction", rows[i][1] - rgcn_scores[i])),
                    top_attended_path_type=dbg.get("top_attended_path_type") or top_type,
                    match_flags=match,
                    conflict_flags=conflict,
                    path_string=path_string,
                )
            )
        return reranked, breakdowns

    def save(self, path: str) -> None:
        torch.save(
            {
                "model_state": self.state_dict(),
                "vocab": self.vocab,
                "hidden_dim": self.hidden_dim,
                "fusion_mode": self.fusion_mode,
                "tau": self.tau,
                "rgcn_embedding_dim": self.rgcn_embedding_dim,
                "max_paths_per_chunk": self.max_paths_per_chunk,
                "max_path_len": self.max_path_len,
            },
            path,
        )

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "FinPathRGCNReranker":
        ckpt = torch.load(path, map_location=device)
        model = cls(
            vocab=ckpt["vocab"],
            hidden_dim=int(ckpt.get("hidden_dim", 128)),
            fusion_mode=ckpt.get("fusion_mode", "residual"),
            tau=float(ckpt.get("tau", 0.2)),
            rgcn_embedding_dim=int(ckpt.get("rgcn_embedding_dim", 0)),
            max_paths_per_chunk=int(ckpt.get("max_paths_per_chunk", 8)),
            max_path_len=int(ckpt.get("max_path_len", 4)),
            device=device,
        )
        model.load_state_dict(ckpt["model_state"])
        return model


def make_finpath_reranker_from_paths(
    paths_by_chunk: Dict[str, Sequence[FinancialPath]],
    hidden_dim: int = 128,
    dropout: float = 0.1,
    fusion_mode: str = "residual",
    tau: float = 0.2,
    device: str = "cpu",
) -> FinPathRGCNReranker:
    vocab = build_path_vocab(paths_by_chunk)
    return FinPathRGCNReranker(
        vocab=vocab,
        hidden_dim=hidden_dim,
        dropout=dropout,
        fusion_mode=fusion_mode,
        tau=tau,
        device=device,
    )


def rule_based_path_score(
    rgcn_score: float,
    paths: Sequence[FinancialPath],
    tau: float = 0.2,
    max_paths: int = 8,
) -> float:
    """A deterministic path-feature ablation over a vanilla R-GCN score."""

    f = compute_path_features(paths, max_paths=max_paths)
    features = dict(zip(PATH_FEATURE_KEYS, f))
    support = (
        0.40 * features["path_coverage_ratio"]
        + 0.15 * features["company_path_exists"]
        + 0.15 * features["year_path_exists"]
        + 0.15 * features["metric_path_exists"]
        + 0.10 * features["semantic_support_path_exists"]
    )
    conflict = (
        0.30 * features["company_conflict_exists"]
        + 0.30 * features["year_conflict_exists"]
        + 0.30 * features["metric_conflict_exists"]
    )
    return float(rgcn_score + tau * np.tanh(support - conflict))


def train_finpath_pairwise(
    model: FinPathRGCNReranker,
    train_queries: Sequence[Dict[str, Any]],
    graph: Any,
    extractor: Optional[FinancialPathExtractor] = None,
    epochs: int = 5,
    lr: float = 1e-3,
    margin: float = 0.1,
    beta_year: float = 0.5,
    beta_metric: float = 0.5,
    beta_company: float = 0.5,
    use_hard_negative_loss: bool = True,
    seed: int = 42,
    verbose: bool = True,
) -> List[float]:
    """Train FinPath with query-level pairwise ranking loss.

    Each train query dict should contain:
      candidate_chunk_ids, retrieval_scores, rgcn_scores, gold_evidence_ids,
      query_entities, and optionally query/query_id.
    """

    set_finpath_seed(seed)
    extractor = extractor or FinancialPathExtractor(
        max_paths_per_chunk=model.max_paths_per_chunk,
        max_path_len=model.max_path_len,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    device = next(model.parameters()).device
    history: List[float] = []

    for epoch in range(epochs):
        total = 0.0
        count = 0
        for item in train_queries:
            cids = list(item.get("candidate_chunk_ids", []))
            gold = set(item.get("gold_evidence_ids", []))
            if not cids or not gold:
                continue
            pos_idx = [i for i, cid in enumerate(cids) if cid in gold]
            neg_idx = [i for i, cid in enumerate(cids) if cid not in gold]
            if not pos_idx or not neg_idx:
                continue

            paths_map = extractor.extract_paths(graph, cids, item.get("query_entities", {}))
            paths = [paths_map.get(cid, []) for cid in cids]
            ret_t = _as_tensor(item.get("retrieval_scores", [0.0] * len(cids)), device)
            rgcn_t = _as_tensor(item.get("rgcn_scores", [0.0] * len(cids)), device)
            q_emb = query_entities_to_embedding(item.get("query_entities", {}), model.hidden_dim, device)
            scores, _ = model.score_tensors(ret_t, rgcn_t, paths, q_emb)

            losses: List[torch.Tensor] = []
            for pi in pos_idx:
                neg_choices = neg_idx[:10]
                for ni in neg_choices:
                    weight = 1.0
                    if use_hard_negative_loss:
                        feats = dict(zip(PATH_FEATURE_KEYS, compute_path_features(paths[ni], model.max_paths_per_chunk)))
                        weight += beta_year * feats["year_conflict_exists"]
                        weight += beta_metric * feats["metric_conflict_exists"]
                        weight += beta_company * feats["company_conflict_exists"]
                    losses.append(weight * F.relu(margin - scores[pi] + scores[ni]))
            if not losses:
                continue
            loss = torch.stack(losses).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach().cpu().item())
            count += 1
        avg = total / max(count, 1)
        history.append(avg)
        if verbose:
            print(f"  Epoch {epoch + 1:>3}/{epochs}  finpath_loss={avg:.6f}")
    return history
