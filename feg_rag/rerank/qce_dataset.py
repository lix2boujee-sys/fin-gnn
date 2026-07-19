"""QCE-Graph Lite: Training Dataset Construction.

Provides two training builders:
1. build_qce_rerank_candidates — strict top-50 reranker (primary).
2. build_qce_training_candidates — expansion-based training (ablation).

Both construct positive/negative pairs WITHOUT using gold during
candidate pool construction.
"""

from __future__ import annotations

import random
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from feg_rag.data.chunker import Chunk
from feg_rag.rerank.qce_expansion import (
    RELATION_NAMES,
    GraphExpansionIndex,
    BudgetedGraphExpander,
    ExpandedCandidate,
    normalize_metric,
    DEFAULT_SEED_TOP_M,
)

NUM_RELATIONS = len(RELATION_NAMES)
from feg_rag.rerank.qce_features import (
    QUERY_FEATURE_DIM_QCE,
    SUPPORT_FEATURE_DIM,
    CONFLICT_FEATURE_DIM,
    build_qce_query_features,
    extract_support_features,
    extract_conflict_features,
)
from feg_rag.graph.entities import EntityExtractor

_entity_extractor = EntityExtractor()


# ═════════════════════════════════════════════════════════════════════════════
# Training pair construction — strict top-50 reranker (PRIMARY)
# ═════════════════════════════════════════════════════════════════════════════

def build_qce_rerank_candidates(
    samples: List[Dict],
    retriever,
    chunk_lookup: Dict[str, Chunk],
    gold_map: Dict[str, List[str]],
    *,
    top_n: int = 50,
    hard_negatives_per_positive: int = 10,
    random_negatives_per_positive: int = 2,
    rgcn_scores: Optional[Dict[str, Dict[str, float]]] = None,
    seed: int = 42,
    verbose: bool = True,
) -> Tuple[List[Dict], Dict[str, Any]]:
    """Build training pairs from BGE top-50 ONLY — no candidate expansion.

    CRITICAL:
    - Each query reads exactly BGE top-50.
    - Gold is ONLY used for training pair labels, NOT for candidate construction.
    - No expansion, no new candidates, no deletion.
    - R-GCN scores are per-query min-max normalised.

    Args:
        samples: List of training sample dicts (with 'id', 'question').
        retriever: Candidate pool retriever with .search(query, top_k) method.
        chunk_lookup: chunk_id -> Chunk mapping.
        gold_map: question_id -> list of gold chunk_ids.
        top_n: Number of BGE candidates (default 50).
        hard_negatives_per_positive: Hard negatives per positive.
        random_negatives_per_positive: Random negatives per positive.
        rgcn_scores: Optional question_id -> {chunk_id: score} for R-GCN.
        seed: Random seed.
        verbose: Print progress.

    Returns:
        (training_pairs, meta) where each pair has all features pre-computed.
    """
    rng = random.Random(seed)

    all_pairs: List[Dict] = []
    stats = {
        "total_queries": len(samples),
        "queries_with_gold_in_topn": 0,
        "queries_without_gold_in_topn": 0,
        "total_positive": 0,
        "total_hard_negatives": 0,
        "total_random_negatives": 0,
    }

    t0 = time.time()

    for idx, sample in enumerate(samples):
        qid = sample["id"]
        question = sample["question"]
        gold = set(gold_map.get(qid, []))

        if not gold:
            stats["queries_without_gold_in_topn"] += 1
            continue

        # Step 1: BGE top-50 retrieval
        retrieved = retriever.search(question, top_k=top_n)
        if not retrieved:
            continue

        # Step 2: Build ExpandedCandidate wrappers for ALL top-50 (no expansion)
        candidates: Dict[str, ExpandedCandidate] = {}
        for rank, (chunk, score) in enumerate(retrieved, start=1):
            candidates[chunk.chunk_id] = ExpandedCandidate(
                chunk_id=chunk.chunk_id,
                is_initial=True,
                initial_score=float(score),
                initial_rank=rank,
                source_relations=[],
            )

        # Step 3: Per-query R-GCN normalisation
        rgcn_norm_map: Dict[str, float] = {}
        if rgcn_scores:
            q_rgcn = rgcn_scores.get(qid, {})
            rgcn_vals = [q_rgcn.get(cid, 0.0) for cid in candidates]
            rgcn_min = min(rgcn_vals)
            rgcn_max = max(rgcn_vals)
            if rgcn_max > rgcn_min:
                for cid in candidates:
                    raw = q_rgcn.get(cid, 0.0)
                    rgcn_norm_map[cid] = (raw - rgcn_min) / (rgcn_max - rgcn_min)
            else:
                for cid in candidates:
                    rgcn_norm_map[cid] = 0.5
        else:
            for cid in candidates:
                rgcn_norm_map[cid] = 0.5

        # Step 4: Per-query retrieval score normalisation
        ret_scores = [ec.initial_score for ec in candidates.values()]
        ret_min = min(ret_scores)
        ret_max = max(ret_scores)
        if ret_max > ret_min:
            ret_norm_map = {
                cid: (ec.initial_score - ret_min) / (ret_max - ret_min)
                for cid, ec in candidates.items()
            }
        else:
            ret_norm_map = {cid: 0.5 for cid in candidates}

        # Step 5: Find positives and negatives
        positives = [cid for cid in candidates if cid in gold]
        if not positives:
            stats["queries_without_gold_in_topn"] += 1
            continue

        stats["queries_with_gold_in_topn"] += 1

        # Negatives: ordered by rank
        negatives_all = [
            cid for cid in candidates if cid not in gold
        ]

        # Priority negatives:
        # 1. Top-ranked non-gold (hard negatives by rank)
        hard_negs = [cid for cid in negatives_all[:10]]

        # 2. Conflict candidates
        conflict_negs = _identify_conflict_negatives(
            question, candidates, chunk_lookup, gold,
        )

        # 3. Remaining random
        remaining = [cid for cid in negatives_all if cid not in hard_negs and cid not in conflict_negs]
        rng.shuffle(remaining)

        selected_negatives: List[str] = []
        seen_neg: Set[str] = set()
        for n in hard_negs + conflict_negs + remaining:
            if n not in seen_neg and n not in gold:
                seen_neg.add(n)
                selected_negatives.append(n)

        n_hard = min(hard_negatives_per_positive, len(selected_negatives))
        n_random = min(random_negatives_per_positive, max(0, len(selected_negatives) - n_hard))
        hard_neg_sampled = selected_negatives[:n_hard]
        random_pool = selected_negatives[n_hard:]
        rng.shuffle(random_pool)
        random_neg_sampled = random_pool[:n_random]

        # Step 6: Build seed chunk lookup for feature extraction
        seed_chunks: Dict[str, Chunk] = {}
        for chunk, _ in retrieved[:DEFAULT_SEED_TOP_M]:
            seed_chunks[chunk.chunk_id] = chunk

        # Step 7: Extract features and create pairs
        qf = build_qce_query_features(question)
        rel_prob_dict = {rn: 0.5 for rn in RELATION_NAMES}  # uniform for feature extraction

        for pos_id in positives:
            pos_ec = candidates[pos_id]
            pos_sf = extract_support_features(question, pos_ec, chunk_lookup, rel_prob_dict, seed_chunks)
            pos_cf = extract_conflict_features(question, pos_ec, chunk_lookup)
            pos_bf = _build_base_features_fixed(pos_ec, ret_norm_map)
            pos_ro = _build_relation_origin(pos_ec)
            pos_rgcn_norm = rgcn_norm_map.get(pos_id, 0.5)

            stats["total_positive"] += 1

            for neg_id in hard_neg_sampled + random_neg_sampled:
                neg_ec = candidates.get(neg_id)
                if neg_ec is None:
                    continue
                neg_sf = extract_support_features(question, neg_ec, chunk_lookup, rel_prob_dict, seed_chunks)
                neg_cf = extract_conflict_features(question, neg_ec, chunk_lookup)
                neg_bf = _build_base_features_fixed(neg_ec, ret_norm_map)
                neg_ro = _build_relation_origin(neg_ec)
                neg_rgcn_norm = rgcn_norm_map.get(neg_id, 0.5)

                all_pairs.append({
                    "positive_id": pos_id,
                    "negative_id": neg_id,
                    "question_id": qid,
                    "question": question,
                    "query_features": qf.copy(),
                    "support_features_pos": pos_sf.copy(),
                    "conflict_features_pos": pos_cf.copy(),
                    "base_features_pos": pos_bf.copy(),
                    "relation_origin_pos": pos_ro.copy(),
                    "rgcn_score_pos": pos_rgcn_norm,
                    "support_features_neg": neg_sf.copy(),
                    "conflict_features_neg": neg_cf.copy(),
                    "base_features_neg": neg_bf.copy(),
                    "relation_origin_neg": neg_ro.copy(),
                    "rgcn_score_neg": neg_rgcn_norm,
                })
                if neg_id in hard_neg_sampled:
                    stats["total_hard_negatives"] += 1
                else:
                    stats["total_random_negatives"] += 1

        if verbose and (idx + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(
                f"  [QCERerank] {idx + 1}/{len(samples)} queries, "
                f"{len(all_pairs)} pairs, {elapsed:.1f}s"
            )

    elapsed = time.time() - t0
    if verbose:
        print(
            f"  [QCERerank] Built {len(all_pairs)} training pairs "
            f"from {stats['queries_with_gold_in_topn']} queries in {elapsed:.1f}s"
        )
        print(f"    Positives: {stats['total_positive']}, "
              f"Hard negs: {stats['total_hard_negatives']}, "
              f"Random negs: {stats['total_random_negatives']}")

    return all_pairs, stats


# ═════════════════════════════════════════════════════════════════════════════
# Training pair construction — with expansion (ABLATION ONLY)
# ═════════════════════════════════════════════════════════════════════════════

def build_qce_training_candidates(
    samples: List[Dict],
    expander: BudgetedGraphExpander,
    index: GraphExpansionIndex,
    chunk_lookup: Dict[str, Chunk],
    gold_map: Dict[str, List[str]],
    *,
    initial_top_n: int = 50,
    train_max_per_relation: int = 10,
    train_pool_cap: int = 120,
    hard_negatives_per_positive: int = 10,
    random_negatives_per_positive: int = 2,
    seed: int = 42,
    verbose: bool = True,
) -> Tuple[List[Dict], Dict[str, Any]]:
    """Build training pairs with pre-expanded candidate pools.

    CRITICAL: Pre-expansion uses ONLY query, seed candidates, chunk metadata,
    and graph — NEVER gold evidence.  Gold is only used AFTER candidate pool
    construction to create positive/negative labels and relation targets.

    Args:
        samples: List of training sample dicts (with 'id', 'question').
        expander: BudgetedGraphExpander instance.
        index: GraphExpansionIndex instance.
        chunk_lookup: chunk_id -> Chunk mapping.
        gold_map: question_id -> list of gold chunk_ids.
        initial_top_n: Top-N initial candidates.
        train_max_per_relation: Max candidates per relation for pre-expansion.
        train_pool_cap: Max total candidates in training pool.
        hard_negatives_per_positive: Number of hard negatives per positive.
        random_negatives_per_positive: Number of random negatives per positive.
        seed: Random seed for reproducibility.
        verbose: Print progress.

    Returns:
        (training_pairs, meta) where training_pairs is a list of dicts with
        keys: positive_id, negative_id, question_id, question,
        query_features, support_features (pos), conflict_features (pos),
        support_features_neg, conflict_features_neg, base_features (pos),
        base_features_neg, relation_origin (pos), relation_origin (neg),
        relation_targets.
    """
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)

    all_pairs: List[Dict] = []
    stats = {
        "total_queries": len(samples),
        "queries_with_gold_in_pool": 0,
        "queries_without_gold": 0,
        "total_positive": 0,
        "total_hard_negatives": 0,
        "total_random_negatives": 0,
        "queries_with_no_relation_recovery": 0,
    }

    t0 = time.time()

    for idx, sample in enumerate(samples):
        qid = sample["id"]
        question = sample["question"]
        gold = set(gold_map.get(qid, []))

        if not gold:
            stats["queries_without_gold"] += 1
            continue

        # Step 1: Initial retrieval (simulated from sample's retrieval_scores)
        retrieval_scores = sample.get("retrieval_scores", {})
        if not retrieval_scores:
            continue

        # Build initial candidates list
        sorted_initial = sorted(
            retrieval_scores.items(), key=lambda x: -x[1]
        )[:initial_top_n]
        initial_chunks = [
            (chunk_lookup[cid], score)
            for cid, score in sorted_initial
            if cid in chunk_lookup
        ]

        if not initial_chunks:
            continue

        # Step 2: Pre-expand — ONE relation at a time for relation targets
        # Use uniform relation probabilities for pre-expansion
        uniform_probs = {r: 1.0 for r in RELATION_NAMES}

        expanded_all: Dict[str, ExpandedCandidate] = {}
        relation_recovery: Dict[str, bool] = {}

        for rel_name in RELATION_NAMES:
            # Expand with just this single relation
            single_rel_candidates, _ = expander.expand(
                question,
                initial_chunks,
                relation_probabilities={rel_name: 1.0},
            )

            # Take top train_max_per_relation
            rel_expanded = [
                ec for ec in single_rel_candidates
                if not ec.is_initial and ec.best_relation == rel_name
            ][:train_max_per_relation]

            # Check if any of these recover gold
            recovers_gold = any(
                ec.chunk_id in gold for ec in rel_expanded
            )
            relation_recovery[rel_name] = recovers_gold

            # Add to pool (dedup)
            for ec in rel_expanded:
                if ec.chunk_id not in expanded_all:
                    expanded_all[ec.chunk_id] = ec

        # Step 3: Merge initial + expanded candidates
        all_candidates: Dict[str, ExpandedCandidate] = {}

        # Add initial candidates
        for rank, (chunk, score) in enumerate(initial_chunks, start=1):
            all_candidates[chunk.chunk_id] = ExpandedCandidate(
                chunk_id=chunk.chunk_id,
                is_initial=True,
                initial_score=score,
                initial_rank=rank,
            )

        # Add expanded (only up to cap)
        sorted_expanded = sorted(
            expanded_all.values(),
            key=lambda ec: -ec.expansion_priority,
        )
        for ec in sorted_expanded:
            if ec.chunk_id not in all_candidates:
                if len(all_candidates) >= train_pool_cap:
                    break
                all_candidates[ec.chunk_id] = ec

        # Step 4: Identify positives and negatives
        positives = [
            cid for cid in all_candidates if cid in gold
        ]
        if not positives:
            stats["queries_without_gold"] += 1
            continue

        stats["queries_with_gold_in_pool"] += 1

        # Build relation targets (multi-label)
        relation_targets = np.zeros(len(RELATION_NAMES), dtype=np.float32)
        for i, rn in enumerate(RELATION_NAMES):
            relation_targets[i] = 1.0 if relation_recovery.get(rn, False) else 0.0

        if not relation_targets.any():
            stats["queries_with_no_relation_recovery"] += 1

        # Negative candidates
        negatives_all = [cid for cid in all_candidates if cid not in gold]

        # Priority negatives:
        # 1. Top-10 initial non-gold (hard negatives)
        hard_negs = [
            cid for cid in list(all_candidates.keys())[:10]
            if cid not in gold
        ]

        # 2. High-priority expanded non-gold
        expanded_negs = [
            cid for cid, ec in all_candidates.items()
            if cid not in gold and not ec.is_initial
        ]
        expanded_negs.sort(
            key=lambda cid: -all_candidates[cid].expansion_priority
        )

        # 3. Conflict candidates (wrong company/year/metric)
        conflict_negs = _identify_conflict_negatives(
            question, all_candidates, chunk_lookup, gold,
        )

        # Merge negatives with priority
        selected_negatives: List[str] = []
        neg_pool = []
        neg_pool.extend(hard_negs)
        neg_pool.extend([n for n in expanded_negs if n not in hard_negs])
        neg_pool.extend([n for n in conflict_negs if n not in neg_pool])
        # Fill remaining with random
        remaining = [n for n in negatives_all if n not in neg_pool]
        rng.shuffle(remaining)
        neg_pool.extend(remaining)

        # Deduplicate while preserving order
        seen_neg: Set[str] = set()
        for n in neg_pool:
            if n not in seen_neg and n not in gold:
                seen_neg.add(n)
                selected_negatives.append(n)

        # Sample negatives
        n_hard = min(hard_negatives_per_positive, len(selected_negatives))
        n_random = min(random_negatives_per_positive, max(0, len(selected_negatives) - n_hard))

        hard_neg_sampled = selected_negatives[:n_hard]
        random_pool = selected_negatives[n_hard:]
        rng.shuffle(random_pool)
        random_neg_sampled = random_pool[:n_random]

        # Build seed chunk lookup for feature extraction
        seed_chunks: Dict[str, Chunk] = {}
        for chunk, score in initial_chunks[:expander.seed_top_m]:
            seed_chunks[chunk.chunk_id] = chunk

        # Build relation probabilities dict for route alignment
        rel_prob_dict = {rn: 1.0 for rn in RELATION_NAMES}

        # Create positive-negative pairs
        qf = build_qce_query_features(question)

        for pos_id in positives:
            pos_ec = all_candidates.get(pos_id)
            if pos_ec is None:
                continue

            pos_sf = extract_support_features(question, pos_ec, chunk_lookup, rel_prob_dict, seed_chunks)
            pos_cf = extract_conflict_features(question, pos_ec, chunk_lookup)
            pos_bf = _build_base_features(pos_ec)
            pos_ro = _build_relation_origin(pos_ec)

            stats["total_positive"] += 1

            # Hard negatives
            for neg_id in hard_neg_sampled:
                neg_ec = all_candidates.get(neg_id)
                if neg_ec is None:
                    continue
                neg_sf = extract_support_features(question, neg_ec, chunk_lookup, rel_prob_dict, seed_chunks)
                neg_cf = extract_conflict_features(question, neg_ec, chunk_lookup)
                neg_bf = _build_base_features(neg_ec)
                neg_ro = _build_relation_origin(neg_ec)

                all_pairs.append({
                    "positive_id": pos_id,
                    "negative_id": neg_id,
                    "question_id": qid,
                    "question": question,
                    "query_features": qf.copy(),
                    "support_features_pos": pos_sf.copy(),
                    "conflict_features_pos": pos_cf.copy(),
                    "base_features_pos": pos_bf.copy(),
                    "relation_origin_pos": pos_ro.copy(),
                    "support_features_neg": neg_sf.copy(),
                    "conflict_features_neg": neg_cf.copy(),
                    "base_features_neg": neg_bf.copy(),
                    "relation_origin_neg": neg_ro.copy(),
                    "relation_targets": relation_targets.copy(),
                })
                stats["total_hard_negatives"] += 1

            # Random negatives
            for neg_id in random_neg_sampled:
                neg_ec = all_candidates.get(neg_id)
                if neg_ec is None:
                    continue
                neg_sf = extract_support_features(question, neg_ec, chunk_lookup, rel_prob_dict, seed_chunks)
                neg_cf = extract_conflict_features(question, neg_ec, chunk_lookup)
                neg_bf = _build_base_features(neg_ec)
                neg_ro = _build_relation_origin(neg_ec)

                all_pairs.append({
                    "positive_id": pos_id,
                    "negative_id": neg_id,
                    "question_id": qid,
                    "question": question,
                    "query_features": qf.copy(),
                    "support_features_pos": pos_sf.copy(),
                    "conflict_features_pos": pos_cf.copy(),
                    "base_features_pos": pos_bf.copy(),
                    "relation_origin_pos": pos_ro.copy(),
                    "support_features_neg": neg_sf.copy(),
                    "conflict_features_neg": neg_cf.copy(),
                    "base_features_neg": neg_bf.copy(),
                    "relation_origin_neg": neg_ro.copy(),
                    "relation_targets": relation_targets.copy(),
                })
                stats["total_random_negatives"] += 1

        if verbose and (idx + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(
                f"  [QCEDataset] {idx + 1}/{len(samples)} queries, "
                f"{len(all_pairs)} pairs, {elapsed:.1f}s"
            )

    elapsed = time.time() - t0
    if verbose:
        print(
            f"  [QCEDataset] Built {len(all_pairs)} training pairs "
            f"from {stats['queries_with_gold_in_pool']} queries in {elapsed:.1f}s"
        )
        print(f"    Positives: {stats['total_positive']}, "
              f"Hard negs: {stats['total_hard_negatives']}, "
              f"Random negs: {stats['total_random_negatives']}")
        print(f"    No relation recovery: {stats['queries_with_no_relation_recovery']}")

    return all_pairs, stats


# ═════════════════════════════════════════════════════════════════════════════
# PyTorch Dataset
# ═════════════════════════════════════════════════════════════════════════════

class QCERerankDataset(Dataset):
    """Pairwise ranking dataset for QCE-Graph Lite training.

    Supports both expansion-based and fixed-candidate pairs.
    RGCN scores are already per-query normalised by the training builder.
    """

    def __init__(
        self,
        pairs: List[Dict],
        use_rgcn_score: bool = False,
        rgcn_scores: Optional[Dict[str, Dict[str, float]]] = None,
    ):
        self.pairs = pairs
        self.use_rgcn_score = use_rgcn_score
        self.rgcn_scores = rgcn_scores or {}

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        pair = self.pairs[idx]
        qid = pair["question_id"]
        pos_id = pair["positive_id"]
        neg_id = pair["negative_id"]

        # Query features
        qf = torch.from_numpy(pair["query_features"]).float()

        # Support features
        sf_pos = torch.from_numpy(pair["support_features_pos"]).float()
        sf_neg = torch.from_numpy(pair["support_features_neg"]).float()

        # Conflict features
        cf_pos = torch.from_numpy(pair["conflict_features_pos"]).float()
        cf_neg = torch.from_numpy(pair["conflict_features_neg"]).float()

        # Base features
        bf_pos = torch.from_numpy(pair["base_features_pos"]).float()
        bf_neg = torch.from_numpy(pair["base_features_neg"]).float()

        # Relation origin
        ro_pos = torch.from_numpy(pair["relation_origin_pos"]).float()
        ro_neg = torch.from_numpy(pair["relation_origin_neg"]).float()

        # Conflict indicators for negative weighting (from conflict features)
        cc = cf_neg[0]  # company_conflict
        yc = cf_neg[1]  # year_conflict
        mc = cf_neg[2]  # metric_conflict

        # R-GCN scores — already per-query normalised if from build_qce_rerank_candidates
        rgcn_pos = pair.get("rgcn_score_pos", 0.5)
        rgcn_neg = pair.get("rgcn_score_neg", 0.5)
        if self.use_rgcn_score and self.rgcn_scores:
            q_scores = self.rgcn_scores.get(qid, {})
            rgcn_pos = q_scores.get(pos_id, rgcn_pos)
            rgcn_neg = q_scores.get(neg_id, rgcn_neg)

        # Relation targets (may be absent for rerank-only mode)
        rt = pair.get("relation_targets", np.zeros(NUM_RELATIONS, dtype=np.float32))
        if isinstance(rt, np.ndarray):
            rt = torch.from_numpy(rt).float()
        else:
            rt = torch.tensor(rt, dtype=torch.float32)

        return {
            "query_features": qf,
            "support_features": torch.stack([sf_pos, sf_neg]),
            "conflict_features": torch.stack([cf_pos, cf_neg]),
            "base_features": torch.stack([bf_pos, bf_neg]),
            "relation_origin": torch.stack([ro_pos, ro_neg]),
            "rgcn_scores": torch.tensor([rgcn_pos, rgcn_neg], dtype=torch.float32),
            "relation_targets": rt,
            "positive_mask": torch.tensor([1, 0], dtype=torch.float32),
            "negative_mask": torch.tensor([0, 1], dtype=torch.float32),
            "company_conflict": cc,
            "year_conflict": yc,
            "metric_conflict": mc,
            "question_id": qid,
            "positive_id": pos_id,
            "negative_id": neg_id,
        }


def collate_qce_pairs(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate function for QCERerankDataset.

    Stacks all pair tensors into a single batch where the first half
    of each feature is the positive and the second half is the negative.
    """
    B = len(batch)

    # Stack all pair features
    qf = torch.stack([item["query_features"] for item in batch])  # (B, QF_DIM)
    sf = torch.cat([item["support_features"] for item in batch], dim=0)  # (2B, SF_DIM)
    cf = torch.cat([item["conflict_features"] for item in batch], dim=0)  # (2B, CF_DIM)
    bf = torch.cat([item["base_features"] for item in batch], dim=0)  # (2B, 3)
    ro = torch.cat([item["relation_origin"] for item in batch], dim=0)  # (2B, NR)
    rgcn = torch.cat([item["rgcn_scores"] for item in batch], dim=0)  # (2B,)

    # Repeat query features for each positive/negative
    qf_repeated = qf.repeat_interleave(2, dim=0)  # (2B, QF_DIM)

    # Masks
    pos_mask = torch.cat([item["positive_mask"] for item in batch], dim=0)  # (2B,)
    neg_mask = torch.cat([item["negative_mask"] for item in batch], dim=0)  # (2B,)

    # Relation targets (average across pairs)
    rt = torch.stack([item["relation_targets"] for item in batch])  # (B, NR)

    # Conflict indicators for negatives only
    cc = torch.stack([item["company_conflict"].unsqueeze(0) for item in batch])
    yc = torch.stack([item["year_conflict"].unsqueeze(0) for item in batch])
    mc = torch.stack([item["metric_conflict"].unsqueeze(0) for item in batch])

    return {
        "query_features": qf_repeated,
        "support_features": sf,
        "conflict_features": cf,
        "base_features": bf,
        "relation_origin": ro,
        "rgcn_scores": rgcn,
        "relation_targets": rt,
        "positive_mask": pos_mask,
        "negative_mask": neg_mask,
        "company_conflict": cc,
        "year_conflict": yc,
        "metric_conflict": mc,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ═════════════════════════════════════════════════════════════════════════════

def _identify_conflict_negatives(
    query: str,
    candidates: Dict[str, ExpandedCandidate],
    chunk_lookup: Dict[str, Chunk],
    gold: Set[str],
) -> List[str]:
    """Identify conflict negatives: wrong company/year/metric."""
    q_companies = {c.lower() for c in _entity_extractor.extract_companies(query)}
    q_years = _entity_extractor.extract_years(query)
    q_metrics = {normalize_metric(m) for m in _entity_extractor.extract_metrics(query)}

    conflicts = []
    for cid, ec in candidates.items():
        if cid in gold:
            continue
        chunk = chunk_lookup.get(cid)
        if chunk is None:
            continue

        is_conflict = False

        # Company conflict
        if q_companies and chunk.company:
            chunk_company_lower = chunk.company.lower()
            if not any(
                qc in chunk_company_lower or chunk_company_lower in qc
                for qc in q_companies
            ):
                is_conflict = True

        # Year conflict
        if q_years and chunk.filing_year and chunk.filing_year not in q_years:
            is_conflict = True

        # Metric conflict
        if q_metrics:
            c_metrics = {normalize_metric(m) for m in _entity_extractor.extract_metrics(chunk.text)}
            if c_metrics and not (c_metrics & q_metrics):
                is_conflict = True

        if is_conflict:
            conflicts.append(cid)

    return conflicts


def _build_base_features(ec: ExpandedCandidate) -> np.ndarray:
    """Build 3-dim base feature vector for an ExpandedCandidate."""
    rank_norm = 1.0 / max(ec.initial_rank or 1, 1) if ec.initial_rank else 0.0
    return np.array([
        ec.initial_score if ec.initial_score else 0.0,
        rank_norm,
        1.0 if not ec.is_initial else 0.0,
    ], dtype=np.float32)


def _build_relation_origin(ec: ExpandedCandidate) -> np.ndarray:
    """Build multi-hot relation origin vector."""
    ro = np.zeros(len(RELATION_NAMES), dtype=np.float32)
    for sr in ec.source_relations:
        if sr in RELATION_NAMES:
            ro[RELATION_NAMES.index(sr)] = 1.0
    return ro


def _build_base_features_fixed(
    ec: ExpandedCandidate,
    ret_norm_map: Dict[str, float],
) -> np.ndarray:
    """Build 3-dim base features for fixed-candidate (no expansion).

    Uses per-query normalised retrieval score.
    """
    rank_norm = 1.0 / max(ec.initial_rank or 1, 1) if ec.initial_rank else 0.0
    ret_norm = ret_norm_map.get(ec.chunk_id, 0.5)
    return np.array([ret_norm, rank_norm, 0.0], dtype=np.float32)  # is_expanded = 0
