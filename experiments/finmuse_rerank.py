"""FinMUSE evidence-set reranking over BGE top-N candidates.

This experiment intentionally does not consume cached R-GCN outputs.  FinMUSE
uses its own typed financial graph propagation backbone inside the same top-N
candidate pool as the other rerankers, preserving comparison fairness while
avoiding a dependency on the R-GCN ranking.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

from feg_rag.graph.entities import EntityExtractor
from feg_rag.rerank.finmuse import (
    FinMUSESetReranker,
    _as_nx_graph,
    _minmax_score_map,
    evidence_set_metrics,
    normalise_query_entities,
)
from feg_rag.rerank.path_encoder import canonical_metric, expand_years_from_text


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows = []
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
        return data.get("graph") or data.get("financial_graph") or data
    return data


def query_entities(question: str) -> Dict[str, Any]:
    extractor = EntityExtractor()
    fake = type("QuestionChunk", (), {"chunk_id": "query", "text": question, "section": "", "filing_type": ""})()
    ents = extractor.extract(fake)
    metrics = {canonical_metric(m) for m in ents.metrics}
    return {
        "company": list(ents.companies),
        "years": list(set(ents.years) | expand_years_from_text(question)),
        "metrics": list(metrics),
        "filing_type": list(ents.filing_types),
        "section_hint": list(ents.sections),
    }


def make_result(item: Dict[str, Any], retrieved_ids: Sequence[str], method: str) -> Dict[str, Any]:
    return {
        "question_id": item["query_id"],
        "question": item["query"],
        "gold_answer": item.get("gold_answer", ""),
        "gold_evidence_ids": item.get("gold_evidence_ids", []),
        "retrieved_chunk_ids": list(retrieved_ids),
        "method": method,
    }


def metrics(method: str, results: Sequence[Dict[str, Any]]) -> Dict[str, float | str | int]:
    n = len(results)
    out: Dict[str, float | str | int] = {"Method": method, "num_samples": n}
    if n == 0:
        return out
    for k in (5, 10):
        recalls = []
        ndcgs = []
        hits = []
        for row in results:
            gold = set(row.get("gold_evidence_ids", []))
            retrieved = row.get("retrieved_chunk_ids", [])[:k]
            recalls.append(len(gold & set(retrieved)) / len(gold) if gold else 0.0)
            hits.append(float(bool(gold & set(retrieved))) if gold else 0.0)
            dcg = sum((1.0 if cid in gold else 0.0) / np.log2(i + 2) for i, cid in enumerate(retrieved))
            idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(gold), k)))
            ndcgs.append(dcg / idcg if idcg else 0.0)
        out[f"R@{k}"] = float(np.mean(recalls))
        out[f"nDCG@{k}"] = float(np.mean(ndcgs))
        if k == 10:
            out["Hit@10"] = float(np.mean(hits))
    mrrs = []
    for row in results:
        gold = set(row.get("gold_evidence_ids", []))
        for rank, cid in enumerate(row.get("retrieved_chunk_ids", []), start=1):
            if cid in gold:
                mrrs.append(1.0 / rank)
                break
        else:
            mrrs.append(0.0)
    out["MRR"] = float(np.mean(mrrs))
    return out


def build_items(bge_rows: Sequence[Dict[str, Any]], top_n: int, num_samples: int | None):
    items = []
    for row in bge_rows[: num_samples or len(bge_rows)]:
        qid = str(row.get("question_id") or row.get("id"))
        bge_ids, bge_scores = ids_and_rank_scores(row, top_n)
        items.append(
            {
                "query_id": qid,
                "query": row.get("question", ""),
                "gold_answer": row.get("gold_answer", ""),
                "gold_evidence_ids": list(row.get("gold_evidence_ids") or []),
                "candidate_chunk_ids": bge_ids,
                "bge_scores": bge_scores,
            }
        )
    return items


def run(args: argparse.Namespace) -> None:
    start_time = time.time()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print("============================================================", flush=True)
    print("  FinMUSE: Reliability-Guided Evidence Set Reranking", flush=True)
    print("============================================================", flush=True)
    print(f"  Output:       {out_dir}", flush=True)
    print(f"  Top-N:        {args.top_n}", flush=True)
    print(f"  Max set size: {args.max_set_size}", flush=True)
    print(f"  Seed top-k:   {args.seed_top_k}", flush=True)
    print(f"  Progress:     every {args.progress_every} queries", flush=True)

    print("\n[1/4] Loading candidate results...", flush=True)
    bge_rows = load_jsonl(args.candidate_results_jsonl)
    print(f"  BGE rows:     {len(bge_rows)}", flush=True)
    if args.rgcn_results_jsonl:
        print("  R-GCN input:  ignored (FinMUSE no longer consumes cached R-GCN)", flush=True)

    print("[2/4] Loading graph cache...", flush=True)
    graph = load_graph_cache(args.graph_cache)
    print("  Graph cache loaded", flush=True)

    print("[3/4] Building eval items...", flush=True)
    items = build_items(bge_rows, args.top_n, args.num_samples)
    print(f"  Eval queries: {len(items)}", flush=True)

    reranker = FinMUSESetReranker(
        max_set_size=args.max_set_size,
        seed_top_k=args.seed_top_k,
        companion_pool_k=args.companion_pool_k,
        min_reliability=args.min_reliability,
        delta_cap=args.delta_cap,
    )

    initial_results = []
    backbone_results = []
    finmuse_results = []
    selected_sets = []
    debug_rows = []

    print("[4/4] Running FinMUSE reranking...", flush=True)
    eval_start = time.time()
    for idx, item in enumerate(items):
        initial_results.append(make_result(item, item["candidate_chunk_ids"], "Initial Retriever"))
        q_ents = query_entities(item["query"])
        ranked, best_set, _all_sets = reranker.rerank(
            item["query"],
            item["candidate_chunk_ids"],
            item["bge_scores"],
            graph,
            q_ents,
        )
        # Recompute the graph-backbone-only ranking for an explicit ablation row.
        nxg = _as_nx_graph(graph)
        q_norm = normalise_query_entities(item["query"], q_ents)
        cids = list(item["candidate_chunk_ids"][: args.companion_pool_k])
        profiles = {cid: reranker.chunk_profile(graph, cid) for cid in cids}
        retrieval_map = _minmax_score_map({cid: float(s) for cid, s in zip(item["candidate_chunk_ids"], item["bge_scores"])})
        rank_prior = {
            cid: 1.0 - (r / max(len(item["candidate_chunk_ids"]) - 1, 1))
            for r, cid in enumerate(item["candidate_chunk_ids"])
        }
        backbone_scores = reranker.score_passages(cids, profiles, q_norm, nxg, retrieval_map, rank_prior)
        backbone_ranked = [cid for cid, _ in sorted(backbone_scores.items(), key=lambda x: x[1], reverse=True)]
        backbone_results.append(make_result(item, backbone_ranked, "FinMUSE Graph Backbone"))
        selected_sets.append(best_set)
        finmuse_results.append(make_result(item, ranked, "FinMUSE"))
        if idx < args.debug_examples:
            debug_rows.append(
                {
                    "query": item["query"],
                    "gold_chunk_ids": item.get("gold_evidence_ids", []),
                    "initial_top5": item["candidate_chunk_ids"][:5],
                    "graph_backbone_top5": backbone_ranked[:5],
                    "selected_evidence_set": best_set.passage_ids,
                    "score_breakdown": best_set.breakdown.__dict__,
                    "conflicts": best_set.conflicts,
                    "coverage": best_set.coverage,
                    "companion_reasons": best_set.companion_reasons,
                    "finmuse_top10": ranked[:10],
                }
            )
        done = idx + 1
        if args.progress_every > 0 and (done % args.progress_every == 0 or done == len(items)):
            elapsed = time.time() - eval_start
            rate = done / elapsed if elapsed > 0 else 0.0
            eta = (len(items) - done) / rate if rate > 0 else 0.0
            print(
                f"  [FinMUSE] eval {done}/{len(items)} ({done / max(len(items), 1) * 100:.1f}%) "
                f"elapsed={elapsed:.1f}s eta={eta:.1f}s",
                flush=True,
            )

    result_sets = {
        "Initial Retriever": initial_results,
        "FinMUSE Graph Backbone": backbone_results,
        "FinMUSE": finmuse_results,
    }
    summary = [metrics(name, rows) for name, rows in result_sets.items()]
    set_metrics = evidence_set_metrics(items, selected_sets)

    metric_path = out_dir / "finmuse_metrics.csv"
    fields = [
        "Method", "R@5", "R@10", "MRR", "nDCG@5", "nDCG@10", "Hit@10", "num_samples",
        "evidence_set_gold_coverage", "query_entity_coverage", "conflict_rate", "redundancy_rate",
    ]
    with metric_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in summary:
            if row["Method"] == "FinMUSE":
                row.update(set_metrics)
            writer.writerow({k: row.get(k, "") for k in fields})

    for name, rows in result_sets.items():
        safe = name.lower().replace(" ", "_").replace("-", "_")
        with (out_dir / f"{safe}_results.jsonl").open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (out_dir / "finmuse_debug_examples.json").open("w", encoding="utf-8") as fh:
        json.dump(debug_rows, fh, ensure_ascii=False, indent=2)

    print("\nFinMUSE metrics", flush=True)
    for row in summary:
        print(
            f"{row['Method']:<20} MRR={row.get('MRR', 0):.4f} "
            f"R@5={row.get('R@5', 0):.4f} R@10={row.get('R@10', 0):.4f} "
            f"nDCG@10={row.get('nDCG@10', 0):.4f}",
            flush=True,
        )
    print(f"Set metrics: {set_metrics}", flush=True)
    print(f"Total time: {time.time() - start_time:.1f}s", flush=True)
    print(f"Output: {out_dir}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FinMUSE evidence-set reranking")
    parser.add_argument("--candidate_results_jsonl", required=True)
    parser.add_argument("--rgcn_results_jsonl", default=None)
    parser.add_argument("--graph_cache", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--top_n", type=int, default=50)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--max_set_size", type=int, default=5)
    parser.add_argument("--seed_top_k", type=int, default=10)
    parser.add_argument("--companion_pool_k", type=int, default=50)
    parser.add_argument("--min_reliability", type=float, default=0.15)
    parser.add_argument("--delta_cap", type=float, default=0.15)
    parser.add_argument("--debug_examples", type=int, default=10)
    parser.add_argument("--progress_every", type=int, default=100)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
