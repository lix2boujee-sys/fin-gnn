"""Evaluation metrics for FEG-RAG.

Paper plan §10.2:
  - Evidence Recall@K / Precision@K
  - MRR / nDCG
  - Answer Accuracy / Exact Match / F1
  - Numerical Consistency
  - Hallucination Rate
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from feg_rag.data.chunker import Chunk
from feg_rag.generation.llm import GeneratedAnswer


# ═════════════════════════════════════════════════════════════════════════════
# Data structures
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class EvalResult:
    """Aggregated evaluation results for one method."""

    method_name: str
    # Evidence quality
    evidence_recall: Dict[int, float] = field(default_factory=dict)  # K → recall
    evidence_precision: Dict[int, float] = field(default_factory=dict)  # K → precision
    mrr: float = 0.0
    ndcg: Dict[int, float] = field(default_factory=dict)
    # Answer quality
    answer_accuracy: float = 0.0
    exact_match: float = 0.0
    f1: float = 0.0
    # Reliability
    numerical_consistency: float = 0.0
    hallucination_rate: float = 0.0
    insufficient_evidence_rate: float = 0.0
    # Counts
    num_samples: int = 0


# ═════════════════════════════════════════════════════════════════════════════
# Core metrics
# ═════════════════════════════════════════════════════════════════════════════

def compute_all_metrics(
    method_name: str,
    results: List[Dict],
    k_values: List[int] = (1, 3, 5, 10, 20),
) -> EvalResult:
    """Compute all metrics from a list of per-sample result dicts.

    Each dict must have:
        gold_evidence_ids: List[str]
        retrieved_chunk_ids: List[str]  (ordered)
        gold_answer: str
        generated_answer: str
        answer_is_correct: bool (optional; computed if absent)
        is_hallucination: bool (optional)
        is_consistent: bool (optional)
    """
    out = EvalResult(method_name=method_name)
    n = len(results)
    if n == 0:
        return out
    out.num_samples = n

    # --- evidence recall & precision ---
    for k in k_values:
        recalls, precisions = [], []
        for r in results:
            gold = set(r.get("gold_evidence_ids", []))
            retrieved = r.get("retrieved_chunk_ids", [])[:k]
            if gold:
                recalls.append(len(gold & set(retrieved)) / len(gold))
                precisions.append(len(gold & set(retrieved)) / len(retrieved))
            else:
                recalls.append(0.0)
                precisions.append(0.0)
        out.evidence_recall[k] = float(np.mean(recalls))
        out.evidence_precision[k] = float(np.mean(precisions))

    # --- MRR ---
    mrr_vals = []
    for r in results:
        gold = set(r.get("gold_evidence_ids", []))
        retrieved = r.get("retrieved_chunk_ids", [])
        for rank, cid in enumerate(retrieved, start=1):
            if cid in gold:
                mrr_vals.append(1.0 / rank)
                break
        else:
            mrr_vals.append(0.0)
    out.mrr = float(np.mean(mrr_vals))

    # --- nDCG ---
    for k in k_values:
        ndcg_vals = []
        for r in results:
            gold = set(r.get("gold_evidence_ids", []))
            retrieved = r.get("retrieved_chunk_ids", [])[:k]
            dcg = 0.0
            idcg = 0.0
            for i, cid in enumerate(retrieved):
                rel = 1.0 if cid in gold else 0.0
                dcg += rel / np.log2(i + 2)
            for i in range(min(len(gold), k)):
                idcg += 1.0 / np.log2(i + 2)
            ndcg_vals.append(dcg / idcg if idcg > 0 else 0.0)
        out.ndcg[k] = float(np.mean(ndcg_vals))

    # --- answer accuracy ---
    acc_vals = []
    for r in results:
        if "answer_is_correct" in r:
            acc_vals.append(float(r["answer_is_correct"]))
        else:
            acc_vals.append(
                float(_normalize(r.get("generated_answer", "")) == _normalize(r.get("gold_answer", "")))
            )
    out.answer_accuracy = float(np.mean(acc_vals))
    out.exact_match = float(
        np.mean(
            [
                float(
                    _normalize(r.get("generated_answer", ""))
                    == _normalize(r.get("gold_answer", ""))
                )
                for r in results
            ]
        )
    )

    # --- F1 ---
    f1_vals = []
    for r in results:
        f1_vals.append(_compute_f1(r.get("generated_answer", ""), r.get("gold_answer", "")))
    out.f1 = float(np.mean(f1_vals))

    # --- reliability ---
    consistency_vals = [float(r.get("is_consistent", False)) for r in results]
    out.numerical_consistency = float(np.mean(consistency_vals))

    hall_vals = [float(r.get("is_hallucination", False)) for r in results]
    out.hallucination_rate = float(np.mean(hall_vals))

    ie_vals = [
        float("INSUFFICIENT_EVIDENCE" in r.get("generated_answer", "").upper())
        for r in results
    ]
    out.insufficient_evidence_rate = float(np.mean(ie_vals))

    return out


def evaluate(
    method_name: str,
    results: List[Dict],
    k_values: List[int] = (1, 3, 5, 10, 20),
) -> EvalResult:
    """Alias for compute_all_metrics."""
    return compute_all_metrics(method_name, results, k_values)


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════

def _normalize(text: str) -> str:
    """Normalise answer text for comparison."""
    return re.sub(r"\s+", " ", text.lower().strip().rstrip("."))


def _compute_f1(pred: str, gold: str) -> float:
    """Token-level F1 between prediction and gold."""
    pred_tokens = set(_normalize(pred).split())
    gold_tokens = set(_normalize(gold).split())
    if not pred_tokens or not gold_tokens:
        return 0.0
    tp = len(pred_tokens & gold_tokens)
    precision = tp / len(pred_tokens)
    recall = tp / len(gold_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)
