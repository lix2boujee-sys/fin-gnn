"""Generate experiment tables (LaTeX & Markdown) from JSON results.

Usage:
    python experiments/tables.py --results outputs/experiment_results_20260708_120000.json
    python experiments/tables.py --results-dir outputs/  (aggregates all JSON files)
    python experiments/tables.py --results outputs/xxx.json --format latex
    python experiments/tables.py --results outputs/xxx.json --format markdown
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

# ═════════════════════════════════════════════════════════════════════════════
# Table generators
# ═════════════════════════════════════════════════════════════════════════════

# Table 1: Retrieval Performance
TABLE1_METHODS = [
    "bm25", "dense", "hybrid", "hybrid+cross_encoder",
    "hybrid+ppr", "hybrid+ppr+constraint",
]
TABLE1_METRICS = ["Recall@5", "Recall@10", "MRR", "nDCG@10"]


def table1_retrieval(results: Dict[str, Dict], fmt: str = "markdown") -> str:
    """Table 1: Evidence retrieval performance on FinDER."""
    header = ["Method"] + TABLE1_METRICS
    rows: List[List[str]] = []

    method_labels = {
        "bm25": "BM25",
        "dense": "Dense Retrieval",
        "hybrid": "Hybrid (BM25 + Dense)",
        "hybrid+cross_encoder": "Hybrid + Cross-Encoder",
        "hybrid+ppr": "Hybrid + PPR",
        "hybrid+ppr+constraint": "Hybrid + PPR + Constraint (FEG-Rerank)",
    }

    aliases = {
        "bm25": ["bm25", "BM25"],
        "dense": ["dense", "Dense"],
        "hybrid": ["hybrid", "Hybrid"],
        "hybrid+cross_encoder": [
            "hybrid+cross_encoder",
            "hybrid+ce",
            "Hybrid+CE",
            "Hybrid+CrossEncoder",
            "Hybrid+Cross-Encoder",
        ],
        "hybrid+ppr": ["hybrid+ppr", "Hybrid+PPR"],
        "hybrid+ppr+constraint": [
            "hybrid+ppr+constraint",
            "Hybrid+PPR+Constraint",
            "Hybrid+PPR+ConstraintScore",
        ],
    }

    for method in TABLE1_METHODS:
        r = _get_result(results, aliases.get(method, [method]))
        if not r:
            rows.append([method_labels.get(method, method)] + ["—"] * len(TABLE1_METRICS))
            continue

        recall = r.get("evidence_recall", {})
        ndcg = r.get("ndcg", {})
        row = [
            method_labels.get(method, method),
            _fmt_pct(recall.get(5, recall.get("5", 0))),
            _fmt_pct(recall.get(10, recall.get("10", 0))),
            _fmt_num(r.get("mrr", 0)),
            _fmt_num(ndcg.get("10", ndcg.get(10, 0))),
        ]
        rows.append(row)

    return _render_table(header, rows, fmt, caption="Evidence Retrieval Performance on FinDER",
                         label="tab:retrieval")


# Table 2: Answer Reliability
TABLE2_METHODS = [
    "bm25", "dense", "hybrid", "hybrid+cross_encoder",
    "hybrid+ppr", "hybrid+ppr+constraint",
]
TABLE2_METRICS = ["Accuracy", "EM", "Faithfulness", "Num Consistency", "Unsup. Rate"]


def table2_reliability(results: Dict[str, Dict], fmt: str = "markdown") -> str:
    """Table 2: Answer reliability with fixed LLM (Qwen2.5-7B)."""
    header = ["Retrieval Setting"] + TABLE2_METRICS
    rows: List[List[str]] = []

    labels = {
        "bm25": "BM25-RAG",
        "dense": "Dense-RAG",
        "hybrid": "Hybrid-RAG",
        "hybrid+cross_encoder": "Hybrid + Cross-Encoder",
        "hybrid+ppr": "FEG-PPR-RAG",
        "hybrid+ppr+constraint": "FEG-GNN-RAG + Verifier",
    }

    for method in TABLE2_METHODS:
        r = results.get(method, {})
        if not r:
            rows.append([labels.get(method, method)] + ["—"] * len(TABLE2_METRICS))
            continue

        row = [
            labels.get(method, method),
            _fmt_pct(r.get("answer_accuracy", 0)),
            _fmt_pct(r.get("exact_match", r.get("answer_accuracy", 0))),
            _fmt_pct(r.get("faithfulness", 0)),
            _fmt_pct(r.get("numerical_consistency", 0)),
            _fmt_pct(r.get("unsupported_rate", r.get("insufficient_evidence_rate", 0))),
        ]
        rows.append(row)

    caption = "Answer Reliability with Fixed LLM (Qwen2.5-7B-Instruct)"
    return _render_table(header, rows, fmt, caption=caption, label="tab:reliability")


# Table 3: Graph Structure Ablation
TABLE3_METHODS = ["no_graph", "semantic_only", "financial_only", "financial+semantic", "full_weighted"]
TABLE3_METRICS = ["Recall@10", "MRR", "Accuracy", "Wrong-Year Err", "Wrong-Metric Err"]


def table3_graph_ablation(results: Dict[str, Dict], fmt: str = "markdown") -> str:
    """Table 3: Ablation on graph structure."""
    header = ["Graph Setting"] + TABLE3_METRICS
    labels = {
        "no_graph": "No Graph",
        "semantic_only": "Semantic Edges Only",
        "financial_only": "Financial Edges Only",
        "financial+semantic": "Financial + Semantic Edges",
        "full_weighted": "Full Weighted Graph",
    }

    rows: List[List[str]] = []
    for method in TABLE3_METHODS:
        r = results.get(method, {})
        if not r:
            rows.append([labels.get(method, method)] + ["—"] * len(TABLE3_METRICS))
            continue
        recall = r.get("evidence_recall", {})
        row = [
            labels.get(method, method),
            _fmt_pct(recall.get(10, recall.get("10", 0))),
            _fmt_num(r.get("mrr", 0)),
            _fmt_pct(r.get("answer_accuracy", 0)),
            _fmt_pct((r.get("error_type_counts", {}) or {}).get("wrong_year", 0)),
            _fmt_pct((r.get("error_type_counts", {}) or {}).get("wrong_metric", 0)),
        ]
        rows.append(row)

    caption = "Ablation on Graph Structure"
    return _render_table(header, rows, fmt, caption=caption, label="tab:graph_ablation")


# Table 4: Edge Type Ablation
TABLE4_METHODS = [
    "full_graph", "wo_company", "wo_filing", "wo_section",
    "wo_metric", "wo_year", "wo_semantic",
]
TABLE4_METRICS = ["Recall@10", "MRR", "nDCG@10"]


def table4_edge_ablation(results: Dict[str, Dict], fmt: str = "markdown") -> str:
    """Table 4: Edge type ablation."""
    header = ["Setting"] + TABLE4_METRICS
    labels = {
        "full_graph": "Full Graph",
        "wo_company": "w/o Company Edges",
        "wo_filing": "w/o Filing Edges",
        "wo_section": "w/o Section Edges",
        "wo_metric": "w/o Metric Edges",
        "wo_year": "w/o Year Edges",
        "wo_semantic": "w/o Semantic Edges",
    }

    rows: List[List[str]] = []
    for method in TABLE4_METHODS:
        r = results.get(method, {})
        if not r:
            rows.append([labels.get(method, method)] + ["—"] * len(TABLE4_METRICS))
            continue
        recall = r.get("evidence_recall", {})
        ndcg = r.get("ndcg", {})
        row = [
            labels.get(method, method),
            _fmt_pct(recall.get(10, recall.get("10", 0))),
            _fmt_num(r.get("mrr", 0)),
            _fmt_num(ndcg.get("10", ndcg.get(10, 0))),
        ]
        rows.append(row)

    caption = "Edge Type Ablation"
    return _render_table(header, rows, fmt, caption=caption, label="tab:edge_ablation")


# Table 5: Generator Robustness
TABLE5_METHODS = ["qwen_hybrid", "qwen_feg", "llama_hybrid", "llama_feg"]
TABLE5_METRICS = ["Accuracy", "Faithfulness", "Num Consistency"]


def table5_robustness(results: Dict[str, Dict], fmt: str = "markdown") -> str:
    """Table 5: Generator robustness across LLMs."""
    header = ["Generator", "Retrieval Method"] + TABLE5_METRICS
    labels = {
        "qwen_hybrid": ("Qwen2.5-7B-Instruct", "Hybrid"),
        "qwen_feg": ("Qwen2.5-7B-Instruct", "FEG-Rerank"),
        "llama_hybrid": ("Llama-3.1-8B-Instruct", "Hybrid"),
        "llama_feg": ("Llama-3.1-8B-Instruct", "FEG-Rerank"),
    }

    rows: List[List[str]] = []
    for method in TABLE5_METHODS:
        r = results.get(method, {})
        gen, ret = labels.get(method, (method, ""))
        if not r:
            rows.append([gen, ret] + ["—"] * len(TABLE5_METRICS))
            continue
        row = [
            gen, ret,
            _fmt_pct(r.get("answer_accuracy", 0)),
            _fmt_pct(r.get("faithfulness", 0)),
            _fmt_pct(r.get("numerical_consistency", 0)),
        ]
        rows.append(row)

    caption = "Generator Robustness Across LLMs"
    return _render_table(header, rows, fmt, caption=caption, label="tab:robustness")


# ═════════════════════════════════════════════════════════════════════════════
# Render helpers
# ═════════════════════════════════════════════════════════════════════════════

def _render_table(
    header: List[str],
    rows: List[List[str]],
    fmt: str,
    caption: str = "",
    label: str = "",
) -> str:
    if fmt == "csv":
        return _render_csv(header, rows)
    if fmt == "latex":
        return _render_latex(header, rows, caption, label)
    return _render_markdown(header, rows, caption)


def _render_markdown(header: List[str], rows: List[List[str]], caption: str) -> str:
    lines: List[str] = []
    if caption:
        lines.append(f"### {caption}")
        lines.append("")
    # Header
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---:" for _ in header]) + "|")
    # Rows
    for row in rows:
        lines.append("| " + " | ".join(_clean_cell(cell) for cell in row) + " |")
    return "\n".join(lines)


def _render_csv(header: List[str], rows: List[List[str]]) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(header)
    for row in rows:
        writer.writerow([_clean_cell(cell) for cell in row])
    return buf.getvalue().rstrip("\n")


def _render_latex(
    header: List[str],
    rows: List[List[str]],
    caption: str,
    label: str,
) -> str:
    ncol = len(header)
    align = "l" + "r" * (ncol - 1)
    lines: List[str] = [
        r"\begin{table}[htbp]",
        r"  \centering",
        rf"  \caption{{{caption}}}",
        rf"  \label{{{label}}}",
        rf"  \begin{{tabular}}{{{align}}}",
        r"    \toprule",
    ]
    lines.append("    " + " & ".join(rf"\textbf{{{h}}}" for h in header) + r" \\")
    lines.append(r"    \midrule")
    for row in rows:
        lines.append("    " + " & ".join(_clean_cell(cell) for cell in row) + r" \\")
    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def _get_result(results: Dict[str, Dict], keys: List[str]) -> Dict:
    """Return the first matching result, accepting case-only key differences."""
    for key in keys:
        if key in results:
            return results[key]

    lower_results = {str(key).lower(): value for key, value in results.items()}
    for key in keys:
        value = lower_results.get(str(key).lower())
        if value is not None:
            return value
    return {}


def _clean_cell(cell: str) -> str:
    if cell in {"\u2014", "鈥?", "бк"}:
        return "-"
    return cell


# ═════════════════════════════════════════════════════════════════════════════
# Format helpers
# ═════════════════════════════════════════════════════════════════════════════

def _fmt_pct(val) -> str:
    if isinstance(val, (int, float)):
        return f"{val * 100:.1f}"
    return str(val)


def _fmt_num(val) -> str:
    if isinstance(val, (int, float)):
        return f"{val:.4f}"
    return str(val)


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════

def load_results(path: Path) -> Dict[str, Dict]:
    """Load experiment results from a JSON file."""
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    # Handle both direct dict and wrapped {"results": ...} format
    if "results" in data:
        return data["results"]
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate experiment tables")
    parser.add_argument("--results", type=str, default="",
                        help="Path to single results JSON file")
    parser.add_argument("--results-dir", type=str, default="",
                        help="Directory of result JSON files to aggregate")
    parser.add_argument("--format", choices=["markdown", "latex", "csv"], default="markdown",
                        help="Output format")
    parser.add_argument("--table", type=str, default="all",
                        choices=["all", "1", "2", "3", "4", "5"],
                        help="Which table(s) to generate")
    parser.add_argument("--output", type=str, default="",
                        help="Optional output file for the generated table text")
    args = parser.parse_args()

    # Load results
    all_results: Dict[str, Dict] = {}

    if args.results:
        all_results = load_results(Path(args.results))
    elif args.results_dir:
        for jf in sorted(Path(args.results_dir).glob("*.json")):
            res = load_results(jf)
            all_results.update(res)
        print(f"Aggregated {len(all_results)} method results from {args.results_dir}")
    else:
        print("ERROR: Provide --results or --results-dir")
        sys.exit(1)

    # Generate tables
    generators = {
        "1": table1_retrieval,
        "2": table2_reliability,
        "3": table3_graph_ablation,
        "4": table4_edge_ablation,
        "5": table5_robustness,
    }

    tables_to_generate = (
        ["1", "2", "3", "4", "5"] if args.table == "all" else [args.table]
    )

    chunks: List[str] = []
    for t in tables_to_generate:
        table_text = generators[t](all_results, fmt=args.format)
        if args.format == "csv" and len(tables_to_generate) == 1:
            chunks.append(table_text)
        else:
            chunks.append(f"{'=' * 60}\nTable {t}\n{'=' * 60}")
            chunks.append(table_text)

    output = "\n\n".join(chunks)
    print(output)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output + "\n", encoding="utf-8")
        print(f"\nWrote table to: {output_path}")


if __name__ == "__main__":
    main()
