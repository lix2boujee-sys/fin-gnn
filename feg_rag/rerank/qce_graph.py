"""QCE-Graph Lite: Query-Conditioned Counterfactual Evidence Graph Reranker.

Project-level contributions:
1. Query-conditioned multi-label relation routing (feature gating).
2. Budget-constrained graph candidate expansion (ablation only).
3. Dual-channel support/conflict residual evidence scoring.

Primary mode: strict top-50 reranker (no candidate expansion).
Expansion mode: available as ablation (qce_expansion_* methods).

The model is intentionally lightweight and may optionally consume a
pre-computed R-GCN score. It does not modify the vanilla R-GCN baseline.

Residual scoring:
    final_score = base_score + correction
    correction = support_scale * support_score - conflict_scale * conflict_score
                 + context_scale * context_features

Scales are initialised near zero (sigmoid(-5.0) ≈ 0.007) so the model
starts close to the retrieval baseline.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from feg_rag.rerank.qce_expansion import (
    RELATION_NAMES,
    DEFAULT_RELATION_PRIOR,
    DEFAULT_INITIAL_TOP_N,
    DEFAULT_SEED_TOP_M,
    DEFAULT_EXPANSION_BUDGET,
    DEFAULT_MAX_BUDGET_PER_RELATION,
    DEFAULT_MAX_TOTAL_CANDIDATES,
    DEFAULT_RELATION_THRESHOLD,
    DEFAULT_SEMANTIC_MAX_PER_QUERY,
    GraphExpansionIndex,
    BudgetedGraphExpander,
    ExpandedCandidate,
)
from feg_rag.rerank.qce_features import (
    QUERY_FEATURE_DIM_QCE,
    SUPPORT_FEATURE_DIM,
    CONFLICT_FEATURE_DIM,
    SUPPORT_FEATURE_NAMES,
    CONFLICT_FEATURE_NAMES,
    build_qce_query_features,
    extract_support_features,
    extract_conflict_features,
)

NUM_RELATIONS = len(RELATION_NAMES)

# ── Scale helpers ────────────────────────────────────────────────────────────

def _init_sigmoid_scale(target: float, max_value: float) -> torch.Tensor:
    """Initialise raw scale so that sigmoid(raw) * max ≈ target.

    raw = logit(target / max) = ln(p / (1-p))  where p = target / max
    """
    import math
    p = target / max_value
    # Clamp p away from 0 and 1 for numerical stability
    p = max(min(p, 0.999), 0.001)
    raw = math.log(p / (1.0 - p))
    return torch.tensor(raw, dtype=torch.float32)


def _sigmoid_scale(raw: torch.Tensor, max_value: float) -> torch.Tensor:
    """Apply constrained sigmoid: sigmoid(raw) * max_value."""
    return max_value * torch.sigmoid(raw)


# ═════════════════════════════════════════════════════════════════════════════
# Query Relation Router
# ═════════════════════════════════════════════════════════════════════════════

class QueryRelationRouter(nn.Module):
    """Multi-label relation router: query features → per-relation probabilities.

    Architecture:
        query_features (10-dim)
        → Linear(10, 32) → ReLU → Dropout(0.1)
        → Linear(32, 7) → sigmoid
        → relation_probabilities

    In rerank-only mode, the router is used for query-conditioned feature
    gating (route_alignment), NOT for candidate expansion.
    """

    def __init__(
        self,
        query_feature_dim: int = QUERY_FEATURE_DIM_QCE,
        hidden_dim: int = 32,
        num_relations: int = NUM_RELATIONS,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.query_feature_dim = query_feature_dim
        self.hidden_dim = hidden_dim
        self.num_relations = num_relations

        self.encoder = nn.Sequential(
            nn.Linear(query_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_relations),
        )

    def forward(self, query_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        logits = self.encoder(query_features)  # (B, num_relations)
        probs = torch.sigmoid(logits)
        return {"logits": logits, "probs": probs}

    def get_relation_probabilities(
        self, query_features: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward(query_features)["probs"]


# ═════════════════════════════════════════════════════════════════════════════
# Counterfactual Evidence Scorer
# ═════════════════════════════════════════════════════════════════════════════

class CounterfactualEvidenceScorer(nn.Module):
    """Dual-channel support/conflict evidence scorer.

    Support head: query_embed + support_features → support_score ∈ [0, 1]
    Conflict head: query_embed + conflict_features → conflict_score ∈ [0, 1]
    """

    def __init__(
        self,
        query_feature_dim: int = QUERY_FEATURE_DIM_QCE,
        support_feature_dim: int = SUPPORT_FEATURE_DIM,
        conflict_feature_dim: int = CONFLICT_FEATURE_DIM,
        query_embed_dim: int = 32,
        hidden_dim: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.query_embed_dim = query_embed_dim

        self.query_encoder = nn.Sequential(
            nn.Linear(query_feature_dim, query_embed_dim),
            nn.ReLU(),
        )

        support_input_dim = query_embed_dim + support_feature_dim
        self.support_head = nn.Sequential(
            nn.Linear(support_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        conflict_input_dim = query_embed_dim + conflict_feature_dim
        self.conflict_head = nn.Sequential(
            nn.Linear(conflict_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        query_features: torch.Tensor,
        support_features: torch.Tensor,
        conflict_features: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        q_embed = self.query_encoder(query_features)  # (B, query_embed_dim)

        support_input = torch.cat([q_embed, support_features], dim=-1)
        support_score = self.support_head(support_input).squeeze(-1)  # (B,)

        conflict_input = torch.cat([q_embed, conflict_features], dim=-1)
        conflict_score = self.conflict_head(conflict_input).squeeze(-1)  # (B,)

        return {
            "support_score": support_score,
            "conflict_score": conflict_score,
            "query_embed": q_embed,
        }


# ═════════════════════════════════════════════════════════════════════════════
# QCE Graph Lite Reranker (Residual)
# ═════════════════════════════════════════════════════════════════════════════

class QCEGraphLiteReranker(nn.Module):
    """Residual QCE-Graph Lite reranker.

    Core formula:
        final_score = base_score + correction

    where:
        base_score = retrieval_score_norm          (no R-GCN)
        base_score = w_ret * ret_norm + w_rgcn * rgcn_norm   (with R-GCN)

        correction = support_scale * support_score
                   - conflict_scale * conflict_score
                   + context_scale * context_features

    All scales are initialised near zero (≈0.007–0.01) so the model
    starts close to the retrieval baseline and learns small corrections.
    """

    def __init__(
        self,
        query_feature_dim: int = QUERY_FEATURE_DIM_QCE,
        support_feature_dim: int = SUPPORT_FEATURE_DIM,
        conflict_feature_dim: int = CONFLICT_FEATURE_DIM,
        num_relations: int = NUM_RELATIONS,
        router_hidden_dim: int = 32,
        scorer_hidden_dim: int = 32,
        query_embed_dim: int = 32,
        dropout: float = 0.1,
        use_rgcn_score: bool = True,
        support_scale_max: float = 0.20,
        conflict_scale_max: float = 0.25,
        context_scale_max: float = 0.10,
        # Initial target values (≈ sigmoid(-5.0) * max)
        support_scale_init: float = 0.01,
        conflict_scale_init: float = 0.01,
        context_scale_init: float = 0.005,
    ):
        super().__init__()

        self.num_relations = num_relations
        self.use_rgcn_score = use_rgcn_score

        # Router — used for query-conditioned feature gating
        self.router = QueryRelationRouter(
            query_feature_dim=query_feature_dim,
            hidden_dim=router_hidden_dim,
            num_relations=num_relations,
            dropout=dropout,
        )

        # Counterfactual scorer
        self.scorer = CounterfactualEvidenceScorer(
            query_feature_dim=query_feature_dim,
            support_feature_dim=support_feature_dim,
            conflict_feature_dim=conflict_feature_dim,
            query_embed_dim=query_embed_dim,
            hidden_dim=scorer_hidden_dim,
            dropout=dropout,
        )

        # Learnable scale parameters — initialised near zero for residual behaviour
        self._support_scale_raw = nn.Parameter(
            _init_sigmoid_scale(support_scale_init, support_scale_max)
        )
        self._conflict_scale_raw = nn.Parameter(
            _init_sigmoid_scale(conflict_scale_init, conflict_scale_max)
        )
        self._context_scale_raw = nn.Parameter(
            _init_sigmoid_scale(context_scale_init, context_scale_max)
        )

        self.support_scale_max = support_scale_max
        self.conflict_scale_max = conflict_scale_max
        self.context_scale_max = context_scale_max

        # Base score fusion: learnable weights for retrieval + R-GCN
        if use_rgcn_score:
            # Both start at 0.0 → softmax gives 0.5 each
            self._base_retrieval_raw = nn.Parameter(torch.tensor(0.0))
            self._base_rgcn_raw = nn.Parameter(torch.tensor(0.0))

    # -- Scale properties --------------------------------------------------

    @property
    def support_scale(self) -> torch.Tensor:
        return _sigmoid_scale(self._support_scale_raw, self.support_scale_max)

    @property
    def conflict_scale(self) -> torch.Tensor:
        return _sigmoid_scale(self._conflict_scale_raw, self.conflict_scale_max)

    @property
    def context_scale(self) -> torch.Tensor:
        """Context / expansion scale (used for graph context features)."""
        return _sigmoid_scale(self._context_scale_raw, self.context_scale_max)

    # Backward-compat alias
    @property
    def expansion_scale(self) -> torch.Tensor:
        return self.context_scale

    @property
    def base_weights(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.use_rgcn_score:
            return torch.tensor(1.0), torch.tensor(0.0)
        w = torch.softmax(
            torch.stack([self._base_retrieval_raw, self._base_rgcn_raw]), dim=0
        )
        return w[0], w[1]

    # -- Forward -----------------------------------------------------------

    def forward(
        self,
        query_features: torch.Tensor,
        support_features: torch.Tensor,
        conflict_features: torch.Tensor,
        base_features: torch.Tensor,
        relation_origin: torch.Tensor,
        rgcn_scores: Optional[torch.Tensor] = None,
        return_intermediate: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Residual forward pass.

        Args:
            query_features: (B, query_feature_dim).
            support_features: (B, support_feature_dim).
            conflict_features: (B, conflict_feature_dim).
            base_features: (B, 3) — [retrieval_score_norm, rank_norm, is_expanded].
            relation_origin: (B, num_relations) multi-hot relation origin.
            rgcn_scores: (B,) per-candidate R-GCN scores, **already per-query
                min-max normalised** by the caller.
            return_intermediate: Include all intermediate tensors.

        Returns:
            Dict with 'score' (B,).  If return_intermediate, also
            includes 'base_score', 'support_score', 'conflict_score',
            'relation_probs', 'route_alignment', 'correction'.
        """
        B = query_features.shape[0]

        # 1. Router: query-conditioned relation gating
        router_out = self.router(query_features)
        relation_probs = router_out["probs"]  # (B, num_relations)
        relation_logits = router_out["logits"]

        # 2. Scorer: support and conflict scores
        scorer_out = self.scorer(query_features, support_features, conflict_features)
        support_score = scorer_out["support_score"]  # (B,)
        conflict_score = scorer_out["conflict_score"]  # (B,)

        # 3. Base score: retrieval_score (and optionally R-GCN)
        #    RGCN scores MUST be pre-normalised per-query by the caller.
        retrieval_score = base_features[:, 0]  # (B,)

        if self.use_rgcn_score and rgcn_scores is not None:
            w_ret, w_rgcn = self.base_weights
            base_score = w_ret * retrieval_score + w_rgcn * rgcn_scores
        else:
            base_score = retrieval_score

        # 4. Route alignment: weighted sum of relation probabilities
        route_alignment = (relation_probs * relation_origin).sum(dim=-1)  # (B,)

        # 5. Context features: rank_norm serves as positional context
        context_feat = base_features[:, 1]  # rank_norm (B,)

        # 6. Residual correction
        correction = (
            self.support_scale * support_score
            - self.conflict_scale * conflict_score
            + self.context_scale * context_feat
        )

        # 7. Final score = base + residual correction
        final_score = base_score + correction

        result = {"score": final_score}

        if return_intermediate:
            result.update({
                "base_score": base_score,
                "correction": correction,
                "support_score": support_score,
                "conflict_score": conflict_score,
                "relation_probs": relation_probs,
                "relation_logits": relation_logits,
                "route_alignment": route_alignment,
                "context_feat": context_feat,
            })

        return result

    def get_relation_probabilities(
        self, query_features: torch.Tensor,
    ) -> torch.Tensor:
        return self.router.get_relation_probabilities(query_features)

    @property
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ═════════════════════════════════════════════════════════════════════════════
# Training utilities
# ═════════════════════════════════════════════════════════════════════════════

def compute_qce_loss(
    model: QCEGraphLiteReranker,
    batch: Dict[str, torch.Tensor],
    lambda_router: float = 0.0,   # default 0: no router loss in rerank mode
    lambda_scale: float = 0.001,
    use_router_loss: bool = False,  # only True for expansion ablation
) -> Dict[str, torch.Tensor]:
    """Compute QCE loss: pairwise ranking + optional router + scale reg.

    L_total = L_rank + lambda_router * L_router + lambda_scale * L_scale

    In rerank-only mode (default), router_loss is disabled (lambda_router=0).
    The router still runs and produces relation_probs for route_alignment,
    but is not trained against expansion-based relation targets.
    """
    device = next(model.parameters()).device

    qf = batch["query_features"].to(device)
    sf = batch["support_features"].to(device)
    cf = batch["conflict_features"].to(device)
    bf = batch["base_features"].to(device)
    ro = batch["relation_origin"].to(device)
    rgcn = batch.get("rgcn_scores")
    if rgcn is not None:
        rgcn = rgcn.to(device)

    out = model(qf, sf, cf, bf, ro, rgcn, return_intermediate=True)
    scores = out["score"]  # (B,)

    # --- Pairwise ranking loss ---
    pos_mask = batch["positive_mask"].to(device).bool()
    neg_mask = batch["negative_mask"].to(device).bool()

    if pos_mask.any() and neg_mask.any():
        pos_scores = scores[pos_mask]
        neg_scores = scores[neg_mask]

        n_pairs = min(pos_scores.shape[0], neg_scores.shape[0])
        if n_pairs > 0:
            pos_scores = pos_scores[:n_pairs]
            neg_scores = neg_scores[:n_pairs]

            # Conflict-weighted negative penalty
            neg_weight = torch.ones(n_pairs, device=device)
            if neg_mask.any():
                neg_cf = cf[neg_mask][:n_pairs]
                neg_weight += 0.5 * neg_cf[:, 0]  # company_conflict
                neg_weight += 0.5 * neg_cf[:, 1]  # year_conflict
                neg_weight += 0.5 * neg_cf[:, 2]  # metric_conflict

            margin = pos_scores - neg_scores
            rank_loss = (neg_weight * F.softplus(-margin)).mean()
        else:
            rank_loss = torch.tensor(0.0, device=device)
    else:
        rank_loss = torch.tensor(0.0, device=device)

    # --- Router auxiliary loss (expansion ablation only) ---
    router_loss = torch.tensor(0.0, device=device)
    if use_router_loss and "relation_targets" in batch and batch["relation_targets"] is not None:
        targets = batch["relation_targets"].to(device)
        if out["relation_logits"].shape[0] == targets.shape[0] * 2:
            targets = targets.repeat_interleave(2, dim=0)
        router_loss = F.binary_cross_entropy_with_logits(
            out["relation_logits"], targets
        )

    # --- Scale regularization ---
    ss = model.support_scale
    cs = model.conflict_scale
    es = model.context_scale
    scale_loss = ss ** 2 + cs ** 2 + es ** 2

    # --- Total loss ---
    total_loss = rank_loss + lambda_router * router_loss + lambda_scale * scale_loss

    return {
        "loss": total_loss,
        "rank_loss": rank_loss,
        "router_loss": router_loss,
        "scale_loss": scale_loss,
        "support_scale": ss.detach(),
        "conflict_scale": cs.detach(),
        "context_scale": es.detach(),
    }


# ═════════════════════════════════════════════════════════════════════════════
# Fixed-candidate inference pipeline (primary: no expansion)
# ═════════════════════════════════════════════════════════════════════════════

class QCEFixedCandidatePipeline:
    """Strict top-50 reranker — no candidate expansion.

    Only reranks candidates within the given BGE top-50 pool.
    Does NOT add or remove candidates.
    """

    def __init__(
        self,
        model: QCEGraphLiteReranker,
        index: GraphExpansionIndex,
        chunk_lookup: Dict[str, Any],
        device: str = "cpu",
        initial_top_n: int = 50,
        use_rgcn_score: bool = True,
    ):
        self.model = model.to(device)
        self.index = index
        self.chunk_lookup = chunk_lookup
        self.device = device
        self.initial_top_n = initial_top_n
        self.use_rgcn_score = use_rgcn_score

    @torch.no_grad()
    def rerank(
        self,
        query: str,
        initial_candidates: List[Tuple[Any, float]],
        rgcn_scores: Optional[Dict[str, float]] = None,
        output_k: int = 10,
    ) -> Tuple[List[Tuple[str, float]], Dict[str, Any]]:
        """Rerank within BGE top-50 only.  No expansion.

        Args:
            query: The question text.
            initial_candidates: List of (Chunk, score) from BGE top-50.
            rgcn_scores: Optional dict of chunk_id → R-GCN score (raw).
            output_k: Number of top candidates to return.

        Returns:
            (ranked_chunk_ids_with_scores, metadata_dict)
        """
        self.model.eval()

        # Clamp to top-N
        candidates = initial_candidates[:self.initial_top_n]
        n_cands = len(candidates)

        # 1. Build query features
        qf = build_qce_query_features(query)

        # 2. Router: get relation probabilities
        qf_t = torch.from_numpy(qf).unsqueeze(0).to(self.device)
        rel_probs_t = self.model.get_relation_probabilities(qf_t)
        rel_probs = rel_probs_t.squeeze(0).cpu().numpy()
        relation_probabilities = {
            RELATION_NAMES[i]: float(rel_probs[i])
            for i in range(len(RELATION_NAMES))
        }

        # 3. Build ExpandedCandidate wrappers for ALL top-50 (no expansion)
        expanded_candidates: List[ExpandedCandidate] = []
        for rank, (chunk, score) in enumerate(candidates, start=1):
            ec = ExpandedCandidate(
                chunk_id=chunk.chunk_id,
                is_initial=True,
                initial_score=float(score),
                initial_rank=rank,
                source_relations=[],
            )
            expanded_candidates.append(ec)

        # 4. Per-query R-GCN normalisation (min-max within this query's candidates)
        rgcn_norm_map: Dict[str, float] = {}
        if rgcn_scores:
            rgcn_vals = [rgcn_scores.get(ec.chunk_id, 0.0) for ec in expanded_candidates]
            rgcn_min = min(rgcn_vals)
            rgcn_max = max(rgcn_vals)
            if rgcn_max > rgcn_min:
                for ec in expanded_candidates:
                    raw = rgcn_scores.get(ec.chunk_id, 0.0)
                    rgcn_norm_map[ec.chunk_id] = (raw - rgcn_min) / (rgcn_max - rgcn_min)
            else:
                for ec in expanded_candidates:
                    rgcn_norm_map[ec.chunk_id] = 0.5
        else:
            for ec in expanded_candidates:
                rgcn_norm_map[ec.chunk_id] = 0.5  # neutral default

        # 5. Extract features for each candidate
        seed_chunks_map: Dict[str, Any] = {}
        for chunk, _ in candidates[:DEFAULT_SEED_TOP_M]:
            seed_chunks_map[chunk.chunk_id] = chunk

        # Also build retrieval score norm per-query
        ret_scores = [ec.initial_score for ec in expanded_candidates]
        ret_min = min(ret_scores)
        ret_max = max(ret_scores)
        if ret_max > ret_min:
            ret_norm_map = {
                ec.chunk_id: (ec.initial_score - ret_min) / (ret_max - ret_min)
                for ec in expanded_candidates
            }
        else:
            ret_norm_map = {ec.chunk_id: 0.5 for ec in expanded_candidates}

        all_qf = []
        all_sf = []
        all_cf = []
        all_bf = []
        all_ro = []
        all_rgcn_norm = []
        candidate_ids = []

        for ec in expanded_candidates:
            sf = extract_support_features(
                query, ec, self.chunk_lookup, relation_probabilities, seed_chunks_map,
            )
            cf = extract_conflict_features(query, ec, self.chunk_lookup)

            rank_norm = 1.0 / max(ec.initial_rank or 1, 1)
            ret_norm = ret_norm_map.get(ec.chunk_id, 0.5)
            bf = np.array([ret_norm, rank_norm, 0.0], dtype=np.float32)  # is_expanded=0

            ro = np.zeros(NUM_RELATIONS, dtype=np.float32)
            for sr in ec.source_relations:
                if sr in RELATION_NAMES:
                    ro[RELATION_NAMES.index(sr)] = 1.0

            all_qf.append(qf)
            all_sf.append(sf)
            all_cf.append(cf)
            all_bf.append(bf)
            all_ro.append(ro)
            candidate_ids.append(ec.chunk_id)
            all_rgcn_norm.append(rgcn_norm_map.get(ec.chunk_id, 0.5))

        # 6. Batch forward
        qf_batch = torch.from_numpy(np.stack(all_qf)).to(self.device)
        sf_batch = torch.from_numpy(np.stack(all_sf)).to(self.device)
        cf_batch = torch.from_numpy(np.stack(all_cf)).to(self.device)
        bf_batch = torch.from_numpy(np.stack(all_bf)).to(self.device)
        ro_batch = torch.from_numpy(np.stack(all_ro)).to(self.device)
        rgcn_batch = torch.tensor(all_rgcn_norm, dtype=torch.float32).to(self.device)

        out = self.model(
            qf_batch, sf_batch, cf_batch, bf_batch, ro_batch,
            rgcn_scores=rgcn_batch if self.use_rgcn_score else None,
            return_intermediate=True,
        )

        scores = out["score"].cpu().numpy()
        support_scores = out.get("support_score", torch.zeros(n_cands)).cpu().numpy()
        conflict_scores = out.get("conflict_score", torch.zeros(n_cands)).cpu().numpy()
        base_scores = out.get("base_score", torch.zeros(n_cands)).cpu().numpy()
        corrections = out.get("correction", torch.zeros(n_cands)).cpu().numpy()

        # 7. Sort and return top-k
        ranked_all = sorted(
            zip(candidate_ids, scores, support_scores, conflict_scores, base_scores, corrections),
            key=lambda x: -x[1],
        )

        top_k = [(cid, float(s)) for cid, s, _, _, _, _ in ranked_all[:output_k]]

        # 8. Build debug-rich metadata
        initial_ids = [c.chunk_id for c, _ in candidates]
        debug_candidates = []
        for cid, final_s, sup_s, con_s, base_s, corr in ranked_all[:output_k]:
            chunk = self.chunk_lookup.get(cid)
            initial_rank = next(
                (ec.initial_rank for ec in expanded_candidates if ec.chunk_id == cid), None
            )
            ret_score = next(
                (ec.initial_score for ec in expanded_candidates if ec.chunk_id == cid), 0.0
            )
            debug_candidates.append({
                "chunk_id": cid,
                "initial_rank": initial_rank,
                "final_rank": None,  # filled below
                "retrieval_score": round(float(ret_score), 4),
                "rgcn_score": round(rgcn_norm_map.get(cid, 0.5), 4),
                "support_score": round(float(sup_s), 4),
                "conflict_score": round(float(con_s), 4),
                "base_score": round(float(base_s), 4),
                "correction": round(float(corr), 4),
                "final_score": round(float(final_s), 4),
            })
        # Fill final ranks
        for i, dc in enumerate(debug_candidates, start=1):
            dc["final_rank"] = i

        meta = {
            "relation_probabilities": relation_probabilities,
            "initial_chunk_ids": initial_ids,
            "expanded_chunk_ids": [],  # no expansion
            "num_candidates": len(expanded_candidates),
            "debug_candidates": debug_candidates,
        }

        return top_k, meta


# ═════════════════════════════════════════════════════════════════════════════
# Expansion pipeline (ablation only — kept for backward compatibility)
# ═════════════════════════════════════════════════════════════════════════════

class QCEInferencePipeline:
    """End-to-end QCE with expansion (ablation only).

    Prefer QCEFixedCandidatePipeline for primary reranking.
    """

    def __init__(
        self,
        model: QCEGraphLiteReranker,
        expander: BudgetedGraphExpander,
        index: GraphExpansionIndex,
        chunk_lookup: Dict[str, Any],
        device: str = "cpu",
        initial_top_n: int = DEFAULT_INITIAL_TOP_N,
        use_rgcn_score: bool = True,
    ):
        self.model = model.to(device)
        self.expander = expander
        self.index = index
        self.chunk_lookup = chunk_lookup
        self.device = device
        self.initial_top_n = initial_top_n
        self.use_rgcn_score = use_rgcn_score

    @torch.no_grad()
    def rerank(
        self,
        query: str,
        initial_candidates: List[Tuple[Any, float]],
        rgcn_scores: Optional[Dict[str, float]] = None,
        output_k: int = 10,
    ) -> Tuple[List[Tuple[str, float]], Dict[str, Any]]:
        """Full inference with expansion (ablation)."""
        self.model.eval()

        qf = build_qce_query_features(query)
        qf_t = torch.from_numpy(qf).unsqueeze(0).to(self.device)

        rel_probs_t = self.model.get_relation_probabilities(qf_t)
        rel_probs = rel_probs_t.squeeze(0).cpu().numpy()
        relation_probabilities = {
            RELATION_NAMES[i]: float(rel_probs[i])
            for i in range(len(RELATION_NAMES))
        }

        expanded, exp_stats = self.expander.expand(
            query, initial_candidates, relation_probabilities,
        )

        seed_chunks_map: Dict[str, Any] = {}
        for chunk, score in initial_candidates[:self.expander.seed_top_m]:
            seed_chunks_map[chunk.chunk_id] = chunk

        # Per-query norm
        all_rgcn_raw = []
        for ec in expanded:
            val = rgcn_scores.get(ec.chunk_id, 0.0) if rgcn_scores else 0.0
            all_rgcn_raw.append(val)
        rgcn_min, rgcn_max = min(all_rgcn_raw), max(all_rgcn_raw)
        if rgcn_max > rgcn_min:
            rgcn_norm_vals = [(v - rgcn_min) / (rgcn_max - rgcn_min) for v in all_rgcn_raw]
        else:
            rgcn_norm_vals = [0.5] * len(all_rgcn_raw)

        all_qf, all_sf, all_cf, all_bf, all_ro, all_rgcn_norm = [], [], [], [], [], []
        candidate_ids = []

        for ec, rgcn_n in zip(expanded, rgcn_norm_vals):
            sf = extract_support_features(
                query, ec, self.chunk_lookup, relation_probabilities, seed_chunks_map,
            )
            cf = extract_conflict_features(query, ec, self.chunk_lookup)

            rank_norm = 1.0 / max(ec.initial_rank or 1, 1) if ec.initial_rank else 0.0
            bf = np.array([
                ec.initial_score if ec.initial_score else 0.0,
                rank_norm,
                1.0 if not ec.is_initial else 0.0,
            ], dtype=np.float32)

            ro = np.zeros(NUM_RELATIONS, dtype=np.float32)
            for sr in ec.source_relations:
                if sr in RELATION_NAMES:
                    ro[RELATION_NAMES.index(sr)] = 1.0

            all_qf.append(qf)
            all_sf.append(sf)
            all_cf.append(cf)
            all_bf.append(bf)
            all_ro.append(ro)
            candidate_ids.append(ec.chunk_id)
            all_rgcn_norm.append(rgcn_n)

        qf_batch = torch.from_numpy(np.stack(all_qf)).to(self.device)
        sf_batch = torch.from_numpy(np.stack(all_sf)).to(self.device)
        cf_batch = torch.from_numpy(np.stack(all_cf)).to(self.device)
        bf_batch = torch.from_numpy(np.stack(all_bf)).to(self.device)
        ro_batch = torch.from_numpy(np.stack(all_ro)).to(self.device)
        rgcn_batch = torch.tensor(all_rgcn_norm, dtype=torch.float32).to(self.device)

        out = self.model(
            qf_batch, sf_batch, cf_batch, bf_batch, ro_batch,
            rgcn_scores=rgcn_batch if self.use_rgcn_score else None,
            return_intermediate=True,
        )

        scores = out["score"].cpu().numpy()
        ranked = sorted(zip(candidate_ids, scores), key=lambda x: -x[1])[:output_k]

        initial_ids = [c.chunk_id for c, _ in initial_candidates[:self.initial_top_n]]
        expanded_ids = [ec.chunk_id for ec in expanded if not ec.is_initial]

        meta = {
            "relation_probabilities": relation_probabilities,
            "expansion_stats": exp_stats,
            "initial_chunk_ids": initial_ids,
            "expanded_chunk_ids": expanded_ids,
            "num_candidates": len(expanded),
        }

        return ranked, meta


# ═════════════════════════════════════════════════════════════════════════════
# Model I/O
# ═════════════════════════════════════════════════════════════════════════════

def save_qce_checkpoint(
    model: QCEGraphLiteReranker,
    path: str | Path,
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "meta": meta or {}}, path)


def load_qce_checkpoint(
    path: str | Path,
    model: Optional[QCEGraphLiteReranker] = None,
    device: str = "cpu",
    **model_kwargs,
) -> Tuple[QCEGraphLiteReranker, Dict[str, Any]]:
    ckpt = torch.load(path, map_location=device)
    if model is None:
        model = QCEGraphLiteReranker(**model_kwargs)
    model.load_state_dict(ckpt["model_state"])
    return model.to(device), ckpt.get("meta", {})
