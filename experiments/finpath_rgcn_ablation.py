"""Run FinPath-RGCN ablations from cached graph/retrieval artifacts.

This script is intentionally separate from the main Table 1 runner so the
vanilla R-GCN implementation remains unchanged.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch

from feg_rag.graph.entities import EntityExtractor
from feg_rag.rerank.finpath_rgcn import (
    FinPathRGCNReranker,
    rule_based_path_score,
    set_finpath_seed,
    train_finpath_pairwise,
)
from feg_rag.rerank.path_encoder import (
    FinancialPathExtractor,
    PATH_FEATURE_KEYS,
    build_path_vocab,
    compute_path_features,
)


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_results_by_qid(path: str | Path) -> Dict[str, Dict[str, Any]]:
    return {str(r.get("question_id") or r.get("id")): r for r in load_jsonl(path)}


def ids_and_rank_scores(row: Dict[str, Any], top_n: int) -> tuple[List[str], List[float]]:
    ids = list(row.get("retrieved_chunk_ids") or row.get("chunk_ids") or [])[:top_n]
    scores = row.get("scores") or row.get("retrieved_scores") or row.get("retrieval_scores")
    if isinstance(scores, list) and len(scores) >= len(ids):
        return ids, [float(s) for s in scores[: len(ids)]]
    n = max(len(ids), 1)
    return ids, [1.0 - i / n for i in range(len(ids))]


def load_graph_cache(path: str | Path):
    with Path(path).open("rb") as fh:
        data = pickle.load(fh)
    if isinstance(data, dict):
        if "graph" in data:
            return data["graph"]
        if "financial_graph" in data:
            return data["financial_graph"]
    return data


def sample_query_entities(question: str, sample: Dict[str, Any]) -> Dict[str, Any]:
    extractor = EntityExtractor()
    fake_chunk = type("QuestionChunk", (), {"chunk_id": "query", "text": question, "section": "", "filing_type": ""})()
    ents = extractor.extract(fake_chunk)
    md = sample.get("metadata", {}) or {}
    companies = set(ents.companies)
    if md.get("company"):
        companies.add(str(md["company"]))
    return {
        "company": list(companies),
        "years": list(ents.years),
        "metrics": list(ents.metrics),
        "filing_type": list(ents.filing_types),
        "section_hint": list(ents.sections),
    }


def build_items(
    bge_rows: Sequence[Dict[str, Any]],
    rgcn_by_qid: Dict[str, Dict[str, Any]],
    top_n: int,
    num_samples: int | None = None,
) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for row in bge_rows[: num_samples or len(bge_rows)]:
        qid = str(row.get("question_id") or row.get("id"))
        bge_ids, bge_scores = ids_and_rank_scores(row, top_n)
        rgcn_row = rgcn_by_qid.get(qid, {})
        rgcn_ids, rgcn_rank_scores = ids_and_rank_scores(rgcn_row, max(top_n, len(bge_ids)))
        rgcn_score_map = {cid: score for cid, score in zip(rgcn_ids, rgcn_rank_scores)}
        rgcn_scores = [rgcn_score_map.get(cid, 0.0) for cid in bge_ids]
        item = {
            "query_id": qid,
            "query": row.get("question", ""),
            "gold_answer": row.get("gold_answer", ""),
            "gold_evidence_ids": list(row.get("gold_evidence_ids") or []),
            "candidate_chunk_ids": bge_ids,
            "retrieval_scores": bge_scores,
            "rgcn_scores": rgcn_scores,
        }
        item["query_entities"] = sample_query_entities(item["query"], item)
        items.append(item)
    return items


def make_result(item: Dict[str, Any], retrieved_ids: List[str], method: str) -> Dict[str, Any]:
    return {
        "question_id": item["query_id"],
        "question": item["query"],
        "gold_answer": item.get("gold_answer", ""),
        "gold_evidence_ids": item.get("gold_evidence_ids", []),
        "retrieved_chunk_ids": retrieved_ids,
        "method": method,
    }


def metrics(method: str, results: Sequence[Dict[str, Any]]) -> Dict[str, float | str | int]:
    n = len(results)
    out: Dict[str, float | str | int] = {"Method": method, "num_samples": n}
    if n == 0:
        return out
    for k in (5, 10):
        vals = []
        ndcg_vals = []
        hit_vals = []
        for row in results:
            gold = set(row.get("gold_evidence_ids", []))
            retrieved = row.get("retrieved_chunk_ids", [])[:k]
            vals.append(len(gold & set(retrieved)) / len(gold) if gold else 0.0)
            hit_vals.append(float(bool(gold & set(retrieved))) if gold else 0.0)
            dcg = sum((1.0 if cid in gold else 0.0) / np.log2(i + 2) for i, cid in enumerate(retrieved))
            idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(gold), k)))
            ndcg_vals.append(dcg / idcg if idcg else 0.0)
        out[f"R@{k}"] = float(np.mean(vals))
        out[f"nDCG@{k}"] = float(np.mean(ndcg_vals))
        if k == 10:
            out["Hit@10"] = float(np.mean(hit_vals))
    mrr_vals = []
    for row in results:
        gold = set(row.get("gold_evidence_ids", []))
        for rank, cid in enumerate(row.get("retrieved_chunk_ids", []), start=1):
            if cid in gold:
                mrr_vals.append(1.0 / rank)
                break
        else:
            mrr_vals.append(0.0)
    out["MRR"] = float(np.mean(mrr_vals))
    return out


def split_train_eval(items: List[Dict[str, Any]], train_ratio: float, seed: int):
    rng = random.Random(seed)
    idxs = list(range(len(items)))
    rng.shuffle(idxs)
    cut = int(len(idxs) * train_ratio)
    train = [items[i] for i in idxs[:cut]]
    eval_items = [items[i] for i in idxs[cut:]]
    return train, eval_items


def diagnostics(items: Sequence[Dict[str, Any]], graph: Any, extractor: FinancialPathExtractor) -> Dict[str, Any]:
    cand_counts = []
    pos_counts = []
    neg_counts = []
    no_paths = 0
    positives_with = defaultdict(int)
    negative_conflicts = 0
    total_candidates = 0
    total_pos = 0
    total_neg = 0
    path_type_counts = defaultdict(int)

    for item in items:
        paths_map = extractor.extract_paths(graph, item["candidate_chunk_ids"], item.get("query_entities", {}))
        gold = set(item.get("gold_evidence_ids", []))
        for cid in item["candidate_chunk_ids"]:
            paths = paths_map.get(cid, [])
            total_candidates += 1
            cand_counts.append(len(paths))
            if not paths:
                no_paths += 1
            for path in paths:
                path_type_counts[path.path_type] += 1
            feats = dict(zip(PATH_FEATURE_KEYS, compute_path_features(paths, extractor.max_paths_per_chunk)))
            if cid in gold:
                total_pos += 1
                pos_counts.append(len(paths))
                for key in ("company_path_exists", "year_path_exists", "metric_path_exists"):
                    positives_with[key] += int(feats[key] > 0)
            else:
                total_neg += 1
                neg_counts.append(len(paths))
                negative_conflicts += int(
                    feats["company_conflict_exists"] or feats["year_conflict_exists"] or feats["metric_conflict_exists"]
                )

    return {
        "avg_paths_per_candidate": float(np.mean(cand_counts)) if cand_counts else 0.0,
        "avg_paths_per_positive_candidate": float(np.mean(pos_counts)) if pos_counts else 0.0,
        "avg_paths_per_negative_candidate": float(np.mean(neg_counts)) if neg_counts else 0.0,
        "percent_candidates_with_no_paths": no_paths / max(total_candidates, 1),
        "percent_positives_with_company_path": positives_with["company_path_exists"] / max(total_pos, 1),
        "percent_positives_with_year_path": positives_with["year_path_exists"] / max(total_pos, 1),
        "percent_positives_with_metric_path": positives_with["metric_path_exists"] / max(total_pos, 1),
        "percent_negatives_with_conflict_paths": negative_conflicts / max(total_neg, 1),
        "path_type_counts": dict(path_type_counts),
    }


def run_ablation(args: argparse.Namespace) -> None:
    set_finpath_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bge_rows = load_jsonl(args.candidate_results_jsonl)
    rgcn_by_qid = load_results_by_qid(args.rgcn_results_jsonl)
    graph = load_graph_cache(args.graph_cache)
    items = build_items(bge_rows, rgcn_by_qid, args.top_n, args.num_samples)
    train_items, eval_items = split_train_eval(items, args.train_ratio, args.seed)

    extractor = FinancialPathExtractor(args.max_paths_per_chunk, args.max_path_len)
    all_paths = {}
    for item in train_items[: min(len(train_items), args.vocab_queries)]:
        all_paths.update(extractor.extract_paths(graph, item["candidate_chunk_ids"], item.get("query_entities", {})))
    vocab = build_path_vocab(all_paths)
    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"

    model = FinPathRGCNReranker(
        vocab=vocab,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        fusion_mode=args.fusion_mode,
        tau=args.tau,
        max_paths_per_chunk=args.max_paths_per_chunk,
        max_path_len=args.max_path_len,
        device=device,
    )
    if args.epochs > 0:
        train_finpath_pairwise(
            model,
            train_items,
            graph,
            extractor=extractor,
            epochs=args.epochs,
            lr=args.lr,
            margin=args.margin,
            beta_year=args.beta_year,
            beta_metric=args.beta_metric,
            beta_company=args.beta_company,
            use_hard_negative_loss=args.use_hard_negative_loss,
            seed=args.seed,
        )

    result_sets: Dict[str, List[Dict[str, Any]]] = {
        "Initial Retriever": [],
        "R-GCN": [],
        "R-GCN + rule-based path features": [],
        "R-GCN + LearnablePathEncoder": [],
        "FinPath-RGCN full": [],
    }
    debug_examples: List[Dict[str, Any]] = []

    for item_idx, item in enumerate(eval_items):
        cids = item["candidate_chunk_ids"]
        result_sets["Initial Retriever"].append(make_result(item, cids, "Initial Retriever"))
        rgcn_order = [cid for _, cid in sorted(zip(item["rgcn_scores"], cids), reverse=True)]
        result_sets["R-GCN"].append(make_result(item, rgcn_order, "R-GCN"))

        paths_map = extractor.extract_paths(graph, cids, item.get("query_entities", {}))
        rule_scores = [
            rule_based_path_score(s, paths_map.get(cid, []), tau=args.tau, max_paths=args.max_paths_per_chunk)
            for cid, s in zip(cids, item["rgcn_scores"])
        ]
        rule_order = [cid for _, cid in sorted(zip(rule_scores, cids), reverse=True)]
        result_sets["R-GCN + rule-based path features"].append(
            make_result(item, rule_order, "R-GCN + rule-based path features")
        )

        learnable_out = model.rerank(
            query=item["query"],
            query_id=item["query_id"],
            candidate_chunk_ids=cids,
            retrieval_scores=item["retrieval_scores"],
            graph=graph,
            query_entities=item.get("query_entities", {}),
            rgcn_scores=item["rgcn_scores"],
            extractor=extractor,
            use_path_features=False,
            return_debug=False,
        )
        learnable_ids = [cid for cid, _ in learnable_out]  # type: ignore[misc]
        result_sets["R-GCN + LearnablePathEncoder"].append(
            make_result(item, learnable_ids, "R-GCN + LearnablePathEncoder")
        )

        finpath_out = model.rerank(
            query=item["query"],
            query_id=item["query_id"],
            candidate_chunk_ids=cids,
            retrieval_scores=item["retrieval_scores"],
            graph=graph,
            query_entities=item.get("query_entities", {}),
            rgcn_scores=item["rgcn_scores"],
            extractor=extractor,
            use_path_features=True,
            return_debug=item_idx < args.debug_examples,
        )
        if item_idx < args.debug_examples:
            ranked, breakdowns = finpath_out  # type: ignore[misc]
            debug_examples.append(
                {
                    "query": item["query"],
                    "gold_chunk_ids": item.get("gold_evidence_ids", []),
                    "top5_rgcn": rgcn_order[:5],
                    "top5_finpath": [cid for cid, _ in ranked[:5]],
                    "details": [b.__dict__ for b in breakdowns[:5]],
                }
            )
        else:
            ranked = finpath_out  # type: ignore[assignment]
        fin_ids = [cid for cid, _ in ranked]
        result_sets["FinPath-RGCN full"].append(make_result(item, fin_ids, "FinPath-RGCN full"))

    summary = [metrics(name, rows) for name, rows in result_sets.items()]
    with (out_dir / "finpath_rgcn_ablation_metrics.csv").open("w", newline="", encoding="utf-8") as fh:
        fields = ["Method", "R@5", "R@10", "MRR", "nDCG@5", "nDCG@10", "Hit@10", "num_samples"]
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in summary:
            writer.writerow({k: row.get(k, "") for k in fields})

    for name, rows in result_sets.items():
        safe = name.lower().replace(" ", "_").replace("+", "plus").replace("-", "_")
        with (out_dir / f"{safe}_results.jsonl").open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    diag = diagnostics(eval_items, graph, extractor)
    with (out_dir / "finpath_path_diagnostics.json").open("w", encoding="utf-8") as fh:
        json.dump(diag, fh, indent=2)
    with (out_dir / "finpath_debug_examples.json").open("w", encoding="utf-8") as fh:
        json.dump(debug_examples, fh, indent=2, ensure_ascii=False)

    print("\nFinPath-RGCN ablation metrics")
    for row in summary:
        print(
            f"{row['Method']:<38} "
            f"MRR={row.get('MRR', 0):.4f} R@5={row.get('R@5', 0):.4f} "
            f"R@10={row.get('R@10', 0):.4f} nDCG@10={row.get('nDCG@10', 0):.4f}"
        )
    print(f"\nOutput: {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FinPath-RGCN ablation runner")
    parser.add_argument("--candidate_results_jsonl", required=True)
    parser.add_argument("--rgcn_results_jsonl", required=True)
    parser.add_argument("--graph_cache", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--top_n", type=int, default=50)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vocab_queries", type=int, default=1000)
    parser.add_argument("--max_paths_per_chunk", type=int, default=8)
    parser.add_argument("--max_path_len", type=int, default=4)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--fusion_mode", choices=["residual", "concat_mlp"], default="residual")
    parser.add_argument("--tau", type=float, default=0.2)
    parser.add_argument("--margin", type=float, default=0.1)
    parser.add_argument("--beta_year", type=float, default=0.5)
    parser.add_argument("--beta_metric", type=float, default=0.5)
    parser.add_argument("--beta_company", type=float, default=0.5)
    parser.add_argument("--use_hard_negative_loss", action="store_true", default=True)
    parser.add_argument("--debug_examples", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    run_ablation(parse_args())
