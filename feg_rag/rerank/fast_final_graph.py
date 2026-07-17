"""Fast Final Graph: Lightweight Query-Adaptive Fusion Reranker.

A lightweight alternative to QFE-RGCN that achieves competitive performance
without expensive graph neural network training.  Uses pre-computed scores from
BGE, R-GCN, and MonoT5, combined with entity features through a small
query-adaptive MLP.

Key design
----------
1. Feature extraction per query-candidate pair — no GNN needed at inference.
2. Query-adaptive weight prediction via small MLP.
3. Weighted-sum fusion with minimum-weight floors to preserve strong baselines.
4. Optional delta MLP for residual scoring on full pair features.
5. Pairwise margin ranking loss for training.

Target: train + eval in < 2 hours on the full FinDER dataset.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from feg_rag.data.chunker import Chunk
from feg_rag.graph.entities import EntityExtractor
from feg_rag.rerank.scoring import normalise_score_map

# ═════════════════════════════════════════════════════════════════════════════
# Constants
# ═════════════════════════════════════════════════════════════════════════════

PAIR_FEATURE_DIM = 14
QUERY_ONLY_FEATURE_DIM = 7
NUM_SOURCES = 5  # bge, rgcn, monot5, entity, graph(ppr)

# Shared entity extractor (stateless, thread-safe)
_extractor = EntityExtractor()


# ═════════════════════════════════════════════════════════════════════════════
# JSONL I/O helpers
# ═════════════════════════════════════════════════════════════════════════════

def load_jsonl_results(path: str | Path) -> Dict[str, Dict]:
    """Load results from a JSONL file, keyed by question text.

    Each row must have ``"question"`` and ``"retrieved_chunk_ids"``.

    Returns
    -------
    dict
        ``question_text → {"ids": [...], "scores": {chunk_id: rank_score}}``
        where *rank_score* is ``len(ids) - rank`` so that higher-ranked
        chunks receive higher scores.
    """
    results: Dict[str, Dict] = {}
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            question = row.get("question", "")
            ids = list(row.get("retrieved_chunk_ids", []))
            if not question:
                continue
            scores: Dict[str, float] = {}
            n = len(ids)
            for rank, cid in enumerate(ids):
                scores[cid] = float(n - rank)
            results[question] = {"ids": ids, "scores": scores}
    if not results:
        raise ValueError(f"No results loaded from {path}")
    return results


# ═════════════════════════════════════════════════════════════════════════════
# Feature extraction
# ═════════════════════════════════════════════════════════════════════════════

def extract_query_features(query: str) -> np.ndarray:
    """Extract query-only features (dim = ``QUERY_ONLY_FEATURE_DIM``).

    Columns
    -------
    0  num_years           normalised by /5
    1  num_metrics         normalised by /10
    2  num_companies       normalised by /10
    3  query_length        normalised by /200  (word count)
    4  has_numeric_question  0/1
    5  has_comparison_keyword  0/1
    6  has_delta_keyword   0/1  (change / growth / decline / ...)
    """
    metrics = _extractor.extract_metrics(query)
    years = _extractor.extract_years(query)
    companies = _extractor.extract_companies(query)
    query_lower = query.lower()
    tokens = query.split()

    feats = np.zeros(QUERY_ONLY_FEATURE_DIM, dtype=np.float32)
    feats[0] = min(len(years), 5) / 5.0
    feats[1] = min(len(metrics), 10) / 10.0
    feats[2] = min(len(companies), 10) / 10.0
    feats[3] = min(len(tokens), 200) / 200.0

    _numeric = [
        "how many", "how much", "what is the", "what was the",
        "amount", "value", "number of", "percentage", "percent",
        "ratio", "what are the",
    ]
    feats[4] = 1.0 if any(kw in query_lower for kw in _numeric) else 0.0

    _cmp = [
        "compare", "comparison", "versus", "vs", "higher", "lower",
        "more than", "less than", "greater", "difference", "between",
        "than", "relative to",
    ]
    feats[5] = 1.0 if any(kw in query_lower for kw in _cmp) else 0.0

    _delta = [
        "change", "growth", "decline", "increase", "decrease",
        "delta", "trend", "rising", "falling", "improved", "deteriorated",
    ]
    feats[6] = 1.0 if any(kw in query_lower for kw in _delta) else 0.0

    return feats


def extract_pair_features(
    query: str,
    chunk_id: str,
    chunk: Optional[Chunk],
    bge_scores_norm: Dict[str, float],
    rgcn_scores_norm: Dict[str, float],
    monot5_scores_norm: Dict[str, float],
    ppr_scores_norm: Dict[str, float],
    bge_ranks: Dict[str, float],
    rgcn_ranks: Dict[str, float],
    monot5_ranks: Dict[str, float],
) -> np.ndarray:
    """Extract per (query, candidate) pair features (dim = ``PAIR_FEATURE_DIM``).

    Columns
    -------
    0   bge_score_norm
    1   rgcn_score_norm
    2   monot5_score_norm
    3   entity_score         (mean of cols 8-12)
    4   ppr_score_norm
    5   candidate_rank_bge   (1 / rank)
    6   candidate_rank_rgcn  (1 / rank)
    7   candidate_rank_monot5 (1 / rank)
    8   company_match        0/1
    9   year_match           0/1
    10  metric_match         0/1
    11  filing_type_match    0/1
    12  section_match        0/1
    13  reserved (0)
    """
    feats = np.zeros(PAIR_FEATURE_DIM, dtype=np.float32)

    # -- source scores (cols 0-2, 4) --
    feats[0] = bge_scores_norm.get(chunk_id, 0.0)
    feats[1] = rgcn_scores_norm.get(chunk_id, 0.0)
    feats[2] = monot5_scores_norm.get(chunk_id, 0.0)
    feats[4] = ppr_scores_norm.get(chunk_id, 0.0)

    # -- reciprocal ranks (cols 5-7) --
    feats[5] = bge_ranks.get(chunk_id, 0.0)
    feats[6] = rgcn_ranks.get(chunk_id, 0.0)
    feats[7] = monot5_ranks.get(chunk_id, 0.0)

    # -- entity match features (cols 8-12) --
    if chunk is not None:
        q_metrics = _extractor.extract_metrics(query)
        q_years = _extractor.extract_years(query)
        q_companies = _extractor.extract_companies(query)
        q_filing_types = _extractor.extract_filing_types(query)

        # company_match
        cm = 0.0
        if q_companies and chunk.company:
            cl = chunk.company.lower()
            for qc in q_companies:
                if qc.lower() in cl or cl in qc.lower():
                    cm = 1.0
                    break
        feats[8] = cm

        # year_match
        ym = 0.0
        if q_years:
            if chunk.filing_year and chunk.filing_year in q_years:
                ym = 1.0
            else:
                c_years = _extractor.extract_years(chunk.text)
                if c_years & q_years:
                    ym = 1.0
        feats[9] = ym

        # metric_match
        mm = 0.0
        if q_metrics:
            c_metrics = _extractor.extract_metrics(chunk.text)
            if c_metrics & q_metrics:
                mm = 1.0
        feats[10] = mm

        # filing_type_match
        fm = 0.0
        if q_filing_types and chunk.filing_type:
            if chunk.filing_type.upper() in q_filing_types:
                fm = 1.0
        feats[11] = fm

        # section_match
        sm = 0.0
        if chunk.section:
            ql = query.lower()
            sl = chunk.section.lower()
            if sl in ql:
                sm = 1.0
            else:
                _section_kw: Dict[str, List[str]] = {
                    "risk": ["risk", "risks"],
                    "mda": ["management discussion", "mda", "md&a"],
                    "business": ["business", "overview"],
                    "financial": ["financial statement", "financial condition"],
                    "notes": ["notes to financial", "footnote"],
                }
                for sec_type, keywords in _section_kw.items():
                    if sec_type in sl:
                        if any(kw in ql for kw in keywords):
                            sm = 1.0
                        break
        feats[12] = sm

        # entity_score (col 3) = mean of match features
        feats[3] = float(np.mean(feats[8:13]))

    return feats


# ═════════════════════════════════════════════════════════════════════════════
# Dataset
# ═════════════════════════════════════════════════════════════════════════════

class FastFinalGraphDataset(Dataset):
    """Pre-computed (positive, negative) pair dataset for fusion training.

    Each item is a triple of *(query_features, positive_pair_feat,
    negative_pair_feat)*.  All features are materialised eagerly in
    ``__init__`` so that training loops are pure tensor operations.
    """

    def __init__(
        self,
        samples: List[Dict],
        gold_map: Dict[str, List[str]],
        chunk_by_id: Dict[str, Chunk],
        bge_results: Dict[str, Dict],
        rgcn_results: Dict[str, Dict],
        monot5_results: Dict[str, Dict],
        ppr_results: Optional[Dict[str, Dict]] = None,
        hard_negatives: int = 10,
        top_n: int = 50,
    ):
        self.pairs: List[Dict[str, np.ndarray]] = []
        self.query_features: Dict[str, np.ndarray] = {}

        skipped_no_gold = 0
        skipped_no_candidates = 0

        for s in samples:
            question = s["question"]
            qid = s["id"]
            gold = set(gold_map.get(qid, []))
            if not gold:
                skipped_no_gold += 1
                continue

            # BGE candidate pool
            bge = bge_results.get(question)
            if bge is None:
                skipped_no_candidates += 1
                continue
            candidate_ids = list(bge.get("ids", []))[:top_n]
            if not candidate_ids:
                skipped_no_candidates += 1
                continue

            # Positive / negative split
            pos_ids = [cid for cid in candidate_ids if cid in gold]
            neg_ids = [cid for cid in candidate_ids if cid not in gold]
            if not pos_ids or not neg_ids:
                skipped_no_gold += 1
                continue

            # ---- normalise scores within this query's candidate set ----
            bge_raw = bge.get("scores", {})
            rgcn = rgcn_results.get(question, {})
            monot5 = monot5_results.get(question, {})
            ppr = ppr_results.get(question, {}) if ppr_results else {}

            def _cand_scores(source: Dict, cids: List[str]) -> Dict[str, float]:
                return {cid: source.get("scores", {}).get(cid, 0.0) for cid in cids}

            bge_norm = normalise_score_map(_cand_scores(bge, candidate_ids))
            rgcn_norm = normalise_score_map(_cand_scores(rgcn, candidate_ids))
            monot5_norm = normalise_score_map(_cand_scores(monot5, candidate_ids))
            ppr_norm: Dict[str, float] = {}
            if ppr_results:
                ppr_norm = normalise_score_map(_cand_scores(ppr, candidate_ids))

            # ---- reciprocal ranks ----
            bge_ranks = {cid: 1.0 / r for r, cid in enumerate(candidate_ids, 1)}
            rgcn_sorted = sorted(
                candidate_ids,
                key=lambda cid: rgcn.get("scores", {}).get(cid, 0.0),
                reverse=True,
            )
            rgcn_ranks = {cid: 1.0 / r for r, cid in enumerate(rgcn_sorted, 1)}
            monot5_sorted = sorted(
                candidate_ids,
                key=lambda cid: monot5.get("scores", {}).get(cid, 0.0),
                reverse=True,
            )
            monot5_ranks = {cid: 1.0 / r for r, cid in enumerate(monot5_sorted, 1)}

            # ---- query features (cache once per question) ----
            if question not in self.query_features:
                self.query_features[question] = extract_query_features(question)

            # ---- build pairs ----
            for pos_id in pos_ids:
                pos_feat = extract_pair_features(
                    question, pos_id, chunk_by_id.get(pos_id),
                    bge_norm, rgcn_norm, monot5_norm, ppr_norm,
                    bge_ranks, rgcn_ranks, monot5_ranks,
                )
                for neg_id in neg_ids[:hard_negatives]:
                    neg_feat = extract_pair_features(
                        question, neg_id, chunk_by_id.get(neg_id),
                        bge_norm, rgcn_norm, monot5_norm, ppr_norm,
                        bge_ranks, rgcn_ranks, monot5_ranks,
                    )
                    self.pairs.append({
                        "question": question,
                        "positive_feat": pos_feat,
                        "negative_feat": neg_feat,
                    })

        if not self.pairs:
            raise ValueError(
                f"No training pairs could be built. "
                f"(skipped_no_gold={skipped_no_gold}, "
                f"skipped_no_candidates={skipped_no_candidates})"
            )

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        pair = self.pairs[idx]
        return {
            "query_features": torch.from_numpy(
                self.query_features[pair["question"]]
            ),
            "positive_pair_feat": torch.from_numpy(pair["positive_feat"]),
            "negative_pair_feat": torch.from_numpy(pair["negative_feat"]),
        }

    @property
    def num_pairs(self) -> int:
        return len(self.pairs)

    @classmethod
    def from_cached(
        cls,
        pairs: List[Dict[str, np.ndarray]],
        query_features: Dict[str, np.ndarray],
    ) -> "FastFinalGraphDataset":
        """Reconstruct a dataset from cached materialised features."""
        obj = cls.__new__(cls)
        obj.pairs = pairs
        obj.query_features = query_features
        return obj


# ═════════════════════════════════════════════════════════════════════════════
# Dataset cache fingerprinting & I/O
# ═════════════════════════════════════════════════════════════════════════════

def _file_fingerprint_segment(file_path: Optional[str | Path]) -> str:
    """Return a stable fingerprint segment for a single input file.

    Uses the resolved absolute path, mtime, and file size so that the same
    file always maps to the same hash regardless of the working directory.
    Returns ``"none"`` if *file_path* is None/empty.
    """
    if not file_path:
        return "none"
    p = Path(file_path).resolve()
    if not p.exists():
        return f"missing:{p}"
    try:
        stat = p.stat()
        return f"{p}|{stat.st_mtime}|{stat.st_size}"
    except OSError:
        return f"unreadable:{p}"


def compute_config_fingerprint(
    bge_results_jsonl: str | Path,
    rgcn_results_jsonl: str | Path,
    monot5_results_jsonl: str | Path,
    ppr_results_jsonl: Optional[str | Path] = None,
    *,
    train_sample_ids: Optional[List[str]] = None,
    val_sample_ids: Optional[List[str]] = None,
    top_n: int = 50,
    hard_negatives: int = 10,
    min_rgcn_weight: float = 0.35,
    min_bge_weight: float = 0.15,
    split_seed: int = 42,
    val_split: float = 0.2,
) -> str:
    """Compute a stable SHA256 fingerprint for the dataset configuration.

    The fingerprint includes resolved paths + mtime/size of every input
    file plus all hyperparameters that affect the feature content.
    When any input changes the fingerprint will differ, triggering a
    cache rebuild.
    """
    def _ids_hash(ids: Optional[List[str]]) -> str:
        if ids is None:
            return "none"
        h = hashlib.sha256()
        for item in ids:
            h.update(str(item).encode("utf-8"))
            h.update(b"\0")
        return h.hexdigest()[:24]

    segments = [
        f"bge:{_file_fingerprint_segment(bge_results_jsonl)}",
        f"rgcn:{_file_fingerprint_segment(rgcn_results_jsonl)}",
        f"monot5:{_file_fingerprint_segment(monot5_results_jsonl)}",
        f"ppr:{_file_fingerprint_segment(ppr_results_jsonl)}",
        f"train_sample_ids:{_ids_hash(train_sample_ids)}",
        f"val_sample_ids:{_ids_hash(val_sample_ids)}",
        f"top_n:{top_n}",
        f"hard_negatives:{hard_negatives}",
        f"min_rgcn_weight:{min_rgcn_weight}",
        f"min_bge_weight:{min_bge_weight}",
        f"split_seed:{split_seed}",
        f"val_split:{val_split}",
        f"format:v2",
    ]
    payload = "\n".join(segments)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def save_dataset_cache(
    cache_path: Path,
    train_dataset: FastFinalGraphDataset,
    val_dataset: Optional[FastFinalGraphDataset],
    fingerprint: str,
    meta: Optional[Dict] = None,
) -> None:
    """Save materialised dataset features to a pickle cache.

    Uses atomic write (tmp → rename) to avoid corruption on interrupt.
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    data: Dict = {
        "fingerprint": fingerprint,
        "train_pairs": train_dataset.pairs,
        "train_query_features": train_dataset.query_features,
        "meta": meta or {},
    }
    if val_dataset is not None:
        data["val_pairs"] = val_dataset.pairs
        data["val_query_features"] = val_dataset.query_features
    else:
        data["val_pairs"] = None
        data["val_query_features"] = None

    tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with tmp.open("wb") as fh:
        pickle.dump(data, fh, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(cache_path)


def load_dataset_cache(
    cache_path: Path,
    expected_fingerprint: str,
) -> Tuple[Optional[FastFinalGraphDataset], Optional[FastFinalGraphDataset], str]:
    """Load cached dataset features from a pickle file.

    Returns
    -------
    train_dataset : FastFinalGraphDataset or None
    val_dataset : FastFinalGraphDataset or None
    status : str
        ``"ok"`` on success, or a mismatch reason on failure
        (caller should rebuild).
    """
    if not cache_path.exists():
        return None, None, "cache file does not exist"

    try:
        with cache_path.open("rb") as fh:
            data = pickle.load(fh)
    except Exception as exc:
        return None, None, f"cache load error: {exc}"

    cached_fp = data.get("fingerprint", "")
    if cached_fp != expected_fingerprint:
        # Build a human-readable summary of what changed
        return None, None, (
            f"fingerprint mismatch: "
            f"cached={cached_fp} expected={expected_fingerprint}"
        )

    train_pairs = data.get("train_pairs")
    train_qf = data.get("train_query_features")
    if not train_pairs or not train_qf:
        return None, None, "cache missing train_pairs or train_query_features"

    train_dataset = FastFinalGraphDataset.from_cached(train_pairs, train_qf)

    val_dataset: Optional[FastFinalGraphDataset] = None
    val_pairs = data.get("val_pairs")
    val_qf = data.get("val_query_features")
    if val_pairs and val_qf:
        val_dataset = FastFinalGraphDataset.from_cached(val_pairs, val_qf)

    return train_dataset, val_dataset, "ok"


# ═════════════════════════════════════════════════════════════════════════════
# Model
# ═════════════════════════════════════════════════════════════════════════════

class QueryAdaptiveFusionReranker(nn.Module):
    """Lightweight query-adaptive fusion model.

    Architecture
    ------------
    - ``query_encoder``:  small MLP over query-only features → query embedding
    - ``weight_head``:    predicts dynamic source weights from query embedding
    - ``delta_mlp``:      optional residual MLP over full pair features

    Scoring
    -------
    ::

        q_embed   = query_encoder(query_feats)
        raw_w     = weight_head(q_embed)           # (batch, 5)
        weights   = softmax(raw_w) with floors     # bge ≥ 0.15, rgcn ≥ 0.35
        base      = Σ weights[i] * pair_feats[i]   # i ∈ {bge,rgcn,monot5,entity,ppr}
        delta     = delta_mlp(pair_feats)          # optional
        score     = base + delta
    """

    def __init__(
        self,
        pair_feat_dim: int = PAIR_FEATURE_DIM,
        query_feat_dim: int = QUERY_ONLY_FEATURE_DIM,
        hidden_dim: int = 64,
        num_sources: int = NUM_SOURCES,
        min_rgcn_weight: float = 0.35,
        min_bge_weight: float = 0.15,
        max_entity_weight: float = 0.10,
        delta_scale: float = 0.05,
        use_delta_mlp: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.pair_feat_dim = pair_feat_dim
        self.query_feat_dim = query_feat_dim
        self.hidden_dim = hidden_dim
        self.num_sources = num_sources
        self.min_rgcn_weight = min_rgcn_weight
        self.min_bge_weight = min_bge_weight
        self.max_entity_weight = max_entity_weight
        self.delta_scale = delta_scale
        self.use_delta_mlp = use_delta_mlp

        # Query encoder: query_feat_dim → hidden_dim
        self.query_encoder = nn.Sequential(
            nn.Linear(query_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Weight prediction head: hidden_dim → num_sources
        self.weight_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_sources),
        )

        # Optional delta MLP over full pair features
        if use_delta_mlp:
            self.delta_mlp = nn.Sequential(
                nn.Linear(pair_feat_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
            )
        else:
            self.delta_mlp = None

    # ------------------------------------------------------------------

    def forward(
        self,
        pair_features: torch.Tensor,
        query_features: torch.Tensor,
        return_weights: bool = False,
    ):
        """Score a batch of (query, candidate) pairs.

        Parameters
        ----------
        pair_features : (B, pair_feat_dim)
        query_features : (B, query_feat_dim)
        return_weights : bool
            If True, also return the learned source weights.

        Returns
        -------
        scores : (B, 1)
        weights : (B, num_sources), optional
        """
        # 1. Query encoding
        q_embed = self.query_encoder(query_features)  # (B, H)

        # 2. Dynamic source weights with guaranteed floors
        raw_w = self.weight_head(q_embed)              # (B, num_sources)
        w = F.softmax(raw_w, dim=-1)                   # (B, num_sources), sums to 1

        # Enforce floors as guaranteed-minimum shares:
        #   w_bge = min_bge + free_budget * softmax_bge
        #   w_rgcn = min_rgcn + free_budget * softmax_rgcn
        #   w_other = free_budget * softmax_other
        # where free_budget = 1 - min_bge - min_rgcn
        # This guarantees the floor AND sums to exactly 1.
        free_budget = max(0.0, 1.0 - self.min_bge_weight - self.min_rgcn_weight)
        w = torch.stack([
            self.min_bge_weight + free_budget * w[:, 0],   # bge  ≥ 0.15
            self.min_rgcn_weight + free_budget * w[:, 1],  # rgcn ≥ 0.35
            free_budget * w[:, 2],  # monot5
            free_budget * w[:, 3],  # entity
            free_budget * w[:, 4],  # graph / ppr
        ], dim=-1)

        # Entity features are sparse heuristics and can become an easy shortcut
        # under pairwise training. Cap their contribution, then redistribute
        # excess mass to retrieval/model sources so strong rankers stay intact.
        if self.max_entity_weight is not None and self.max_entity_weight >= 0:
            entity = w[:, 3]
            capped_entity = torch.clamp(entity, max=self.max_entity_weight)
            excess = entity - capped_entity
            if torch.any(excess > 0):
                w = w.clone()
                w[:, 3] = capped_entity
                redistribute_idx = [0, 1, 2, 4]
                base_weights = w[:, redistribute_idx]
                denom = base_weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                w[:, redistribute_idx] = (
                    base_weights + excess.unsqueeze(-1) * base_weights / denom
                )
        # Sum = min_bge + min_rgcn + free_budget * Σ w_i = min_bge + min_rgcn + free_budget = 1 ✓

        # 3. Weighted sum over source scores (first num_sources columns)
        source_scores = pair_features[:, :self.num_sources]  # (B, 5)
        base = (w * source_scores).sum(dim=-1, keepdim=True)  # (B, 1)

        # 4. Optional delta
        if self.delta_mlp is not None:
            delta = self.delta_scale * torch.tanh(self.delta_mlp(pair_features))
            score = base + delta
        else:
            score = base

        if return_weights:
            return score, w
        return score

    # ------------------------------------------------------------------

    @torch.no_grad()
    def compute_average_weights(
        self,
        dataloader: DataLoader,
        device: str = "cpu",
    ) -> Dict[str, float]:
        """Compute average learned source weights over a dataset."""
        self.eval()
        all_w: List[np.ndarray] = []
        for batch in dataloader:
            qf = batch["query_features"].to(device)
            pf = batch["positive_pair_feat"].to(device)
            _, w = self.forward(pf, qf, return_weights=True)
            all_w.append(w.cpu().numpy())

        if not all_w:
            return {}

        avg = np.concatenate(all_w, axis=0).mean(axis=0)
        return {
            "avg_w_bge": float(avg[0]),
            "avg_w_rgcn": float(avg[1]),
            "avg_w_monot5": float(avg[2]),
            "avg_w_entity": float(avg[3]),
            "avg_w_graph": float(avg[4]),
        }


# ═════════════════════════════════════════════════════════════════════════════
# Training
# ═════════════════════════════════════════════════════════════════════════════

def train_fast_final_graph(
    model: QueryAdaptiveFusionReranker,
    train_dataset: FastFinalGraphDataset,
    val_dataset: Optional[FastFinalGraphDataset] = None,
    *,
    epochs: int = 20,
    batch_size: int = 512,
    lr: float = 1e-3,
    device: str = "cpu",
    margin: float = 0.1,
    weight_decay: float = 1e-5,
    early_stopping_patience: int = 5,
    verbose: bool = True,
) -> Tuple[List[float], Optional[List[float]]]:
    """Train the fusion model with pairwise margin ranking loss.

    Parameters
    ----------
    model : QueryAdaptiveFusionReranker
    train_dataset : FastFinalGraphDataset
    val_dataset : FastFinalGraphDataset or None
    epochs : int
    batch_size : int
    lr : float
    device : str
    margin : float
        Margin for ``margin_ranking_loss``.
    weight_decay : float
    early_stopping_patience : int
        Number of epochs without val-loss improvement before stopping.
    verbose : bool

    Returns
    -------
    train_losses : list[float]
    val_losses : list[float] or None
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.01,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        drop_last=False,
    )
    val_loader: Optional[DataLoader] = None
    if val_dataset is not None and len(val_dataset) > 0:
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            drop_last=False,
        )

    train_losses: List[float] = []
    val_losses: List[float] = []
    best_val_loss = float("inf")
    best_state: Optional[Dict] = None
    patience_counter = 0

    for epoch in range(epochs):
        # ---- train ----
        model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            qf = batch["query_features"].to(device)
            pos_pf = batch["positive_pair_feat"].to(device)
            neg_pf = batch["negative_pair_feat"].to(device)

            pos_score = model(pos_pf, qf)
            neg_score = model(neg_pf, qf)

            loss = F.margin_ranking_loss(
                pos_score, neg_score,
                torch.ones_like(pos_score),
                margin=margin,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_train = total_loss / max(n_batches, 1)
        train_losses.append(avg_train)
        scheduler.step()

        # ---- validation ----
        avg_val: Optional[float] = None
        if val_loader is not None:
            model.eval()
            val_total = 0.0
            val_n = 0
            with torch.no_grad():
                for batch in val_loader:
                    qf = batch["query_features"].to(device)
                    pos_pf = batch["positive_pair_feat"].to(device)
                    neg_pf = batch["negative_pair_feat"].to(device)

                    pos_score = model(pos_pf, qf)
                    neg_score = model(neg_pf, qf)

                    loss = F.margin_ranking_loss(
                        pos_score, neg_score,
                        torch.ones_like(pos_score),
                        margin=margin,
                    )
                    val_total += loss.item()
                    val_n += 1
            avg_val = val_total / max(val_n, 1)
            val_losses.append(avg_val)

        if verbose:
            val_str = f"  val_loss={avg_val:.6f}" if avg_val is not None else ""
            lr_str = f"  lr={scheduler.get_last_lr()[0]:.2e}"
            print(
                f"  Epoch {epoch + 1:>3}/{epochs}  "
                f"train_loss={avg_train:.6f}{val_str}{lr_str}"
            )

        # ---- early stopping ----
        monitor = avg_val if avg_val is not None else avg_train
        if monitor < best_val_loss - 1e-6:
            best_val_loss = monitor
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                if verbose:
                    print(f"  Early stopping at epoch {epoch + 1}")
                break

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    return train_losses, val_losses if val_loader is not None else None


# ═════════════════════════════════════════════════════════════════════════════
# Evaluation
# ═════════════════════════════════════════════════════════════════════════════

def evaluate_fast_final_graph(
    model: QueryAdaptiveFusionReranker,
    samples: List[Dict],
    gold_map: Dict[str, List[str]],
    chunk_by_id: Dict[str, Chunk],
    bge_results: Dict[str, Dict],
    rgcn_results: Dict[str, Dict],
    monot5_results: Dict[str, Dict],
    ppr_results: Optional[Dict[str, Dict]] = None,
    *,
    top_n: int = 50,
    output_k: int = 10,
    device: str = "cpu",
    batch_size: int = 512,
    progress_every: int = 100,
    partial_output_path: Optional[Path] = None,
) -> List[Dict]:
    """Evaluate the trained fusion model on a list of samples.

    Returns
    -------
    list[dict]
        Standard result dicts with keys ``question_id``, ``question``,
        ``gold_evidence_ids``, ``retrieved_chunk_ids``, ``method``.
    """
    model.eval()
    model = model.to(device)

    results: List[Dict] = []
    t_start = time.time()

    for idx, s in enumerate(samples, 1):
        question = s["question"]
        qid = s["id"]
        gold = gold_map.get(qid, [])

        bge = bge_results.get(question)
        if bge is None:
            results.append({
                "question_id": qid, "question": question,
                "gold_evidence_ids": gold, "retrieved_chunk_ids": [],
                "method": "fast_final_graph",
            })
            continue

        candidate_ids = list(bge.get("ids", []))[:top_n]
        if not candidate_ids:
            results.append({
                "question_id": qid, "question": question,
                "gold_evidence_ids": gold, "retrieved_chunk_ids": [],
                "method": "fast_final_graph",
            })
            continue

        # ---- normalised scores & ranks (same as in dataset construction) ----
        rgcn = rgcn_results.get(question, {})
        monot5 = monot5_results.get(question, {})

        def _cand_scores(source: Dict, cids: List[str]) -> Dict[str, float]:
            return {cid: source.get("scores", {}).get(cid, 0.0) for cid in cids}

        bge_norm = normalise_score_map(_cand_scores(bge, candidate_ids))
        rgcn_norm = normalise_score_map(_cand_scores(rgcn, candidate_ids))
        monot5_norm = normalise_score_map(_cand_scores(monot5, candidate_ids))

        ppr_norm: Dict[str, float] = {}
        if ppr_results:
            ppr_norm = normalise_score_map(
                _cand_scores(ppr_results.get(question, {}), candidate_ids)
            )

        bge_ranks = {cid: 1.0 / r for r, cid in enumerate(candidate_ids, 1)}
        rgcn_sorted = sorted(
            candidate_ids,
            key=lambda cid: rgcn.get("scores", {}).get(cid, 0.0),
            reverse=True,
        )
        rgcn_ranks = {cid: 1.0 / r for r, cid in enumerate(rgcn_sorted, 1)}
        monot5_sorted = sorted(
            candidate_ids,
            key=lambda cid: monot5.get("scores", {}).get(cid, 0.0),
            reverse=True,
        )
        monot5_ranks = {cid: 1.0 / r for r, cid in enumerate(monot5_sorted, 1)}

        # ---- build features for all candidates ----
        qf_arr = extract_query_features(question)
        pair_list = []
        for cid in candidate_ids:
            pf = extract_pair_features(
                question, cid, chunk_by_id.get(cid),
                bge_norm, rgcn_norm, monot5_norm, ppr_norm,
                bge_ranks, rgcn_ranks, monot5_ranks,
            )
            pair_list.append(pf)

        qf_tensor = torch.from_numpy(qf_arr).unsqueeze(0).repeat(
            len(candidate_ids), 1,
        ).to(device)
        pf_tensor = torch.from_numpy(np.stack(pair_list, axis=0)).to(device)

        # ---- score in batches ----
        all_scores: List[float] = []
        for start in range(0, len(candidate_ids), batch_size):
            end = min(start + batch_size, len(candidate_ids))
            with torch.no_grad():
                batch_scores = model(pf_tensor[start:end], qf_tensor[start:end])
            all_scores.extend(batch_scores.cpu().numpy().flatten().tolist())

        # ---- sort by descending score ----
        scored = list(zip(candidate_ids, all_scores))
        scored.sort(key=lambda x: x[1], reverse=True)
        ranked_ids = [cid for cid, _ in scored[:output_k]]

        results.append({
            "question_id": qid,
            "question": question,
            "gold_evidence_ids": gold,
            "retrieved_chunk_ids": ranked_ids,
            "method": "fast_final_graph",
        })

        # ---- progress ----
        if progress_every > 0 and (idx % progress_every == 0 or idx == len(samples)):
            elapsed = time.time() - t_start
            rate = idx / max(elapsed, 1e-6)
            eta = (len(samples) - idx) / max(rate, 1e-6)
            pct = idx / max(len(samples), 1) * 100
            print(
                f"    [fast_final_graph] eval {idx}/{len(samples)} "
                f"({pct:.1f}%) elapsed={elapsed:.1f}s eta={eta:.1f}s",
                flush=True,
            )
            if partial_output_path is not None:
                partial_output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(partial_output_path, "w", encoding="utf-8") as fh:
                    for r in results:
                        fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    total = time.time() - t_start
    print(f"    [fast_final_graph] Evaluation done in {total:.1f}s")
    return results


# ═════════════════════════════════════════════════════════════════════════════
# Save / Load
# ═════════════════════════════════════════════════════════════════════════════

def save_checkpoint(
    model: QueryAdaptiveFusionReranker,
    path: Path,
    meta: Optional[Dict] = None,
) -> None:
    """Save model checkpoint to *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state": model.state_dict(),
        "pair_feat_dim": model.pair_feat_dim,
        "query_feat_dim": model.query_feat_dim,
        "hidden_dim": model.hidden_dim,
        "num_sources": model.num_sources,
        "min_rgcn_weight": model.min_rgcn_weight,
        "min_bge_weight": model.min_bge_weight,
        "max_entity_weight": model.max_entity_weight,
        "delta_scale": model.delta_scale,
        "use_delta_mlp": model.use_delta_mlp,
        "meta": meta or {},
    }
    torch.save(checkpoint, path)


def load_checkpoint(
    path: str | Path,
    device: str = "cpu",
) -> Tuple[QueryAdaptiveFusionReranker, Dict]:
    """Load model checkpoint from *path*.

    Returns
    -------
    model : QueryAdaptiveFusionReranker
    meta : dict
        Metadata stored in the checkpoint.
    """
    ckpt = torch.load(path, map_location=device)
    model = QueryAdaptiveFusionReranker(
        pair_feat_dim=ckpt.get("pair_feat_dim", PAIR_FEATURE_DIM),
        query_feat_dim=ckpt.get("query_feat_dim", QUERY_ONLY_FEATURE_DIM),
        hidden_dim=ckpt.get("hidden_dim", 64),
        num_sources=ckpt.get("num_sources", NUM_SOURCES),
        min_rgcn_weight=ckpt.get("min_rgcn_weight", 0.35),
        min_bge_weight=ckpt.get("min_bge_weight", 0.15),
        max_entity_weight=ckpt.get("max_entity_weight", 0.10),
        delta_scale=ckpt.get("delta_scale", 0.05),
        use_delta_mlp=ckpt.get("use_delta_mlp", True),
    )
    model.load_state_dict(ckpt["model_state"])
    return model, ckpt.get("meta", {})
