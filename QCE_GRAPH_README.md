# QCE-Graph Lite

**Query-Conditioned Counterfactual Evidence Graph Reranker**

查询条件化反事实金融证据图轻量重排器

## Overview

QCE-Graph Lite is a lightweight, trainable evidence reranker for financial question answering. Unlike the vanilla R-GCN baseline which only reranks existing candidates, QCE-Graph Lite:

1. **Dynamically routes queries** — A learnable router selects which graph relations to use based on query features (company mentions, year references, metric keywords, etc.).

2. **Expands the candidate pool** — Budget-constrained graph expansion finds evidence chunks that the initial retriever may have missed, using relations like same-company-year, same-metric, adjacent chunks, etc.

3. **Dual-channel scoring** — Separately models **support** (evidence that matches the query) and **conflict** (evidence that explicitly contradicts the query on company, year, or metric), preventing wrong-entity chunks from ranking highly.

4. **Optionally consumes R-GCN scores** — Can use pre-computed R-GCN scores as an additional signal, but does not require them. The vanilla R-GCN baseline remains unchanged.

## Architecture

```
Query → [QueryRelationRouter] → relation probabilities
  → [BudgetedGraphExpander] → expanded candidate pool
  → [CounterfactualEvidenceScorer] → support/conflict scores
  → Final score = base_score + α·support - β·conflict + γ·expansion
```

### Key design choices

- **Sigmoid router** (not softmax): a single query may need multiple relations.
- **Budget-constrained**: expansion is capped per-relation and globally to prevent candidate explosion (default max 80).
- **Conflict ≠ missing**: only explicit entity mismatches count as conflict. Missing information is neutral.
- **Lightweight**: < 100K parameters, no transformers, no large GNNs, supports CPU inference.
- **Trainable scales**: support/conflict/expansion contributions are learned but constrained to small maximum values so the model starts near the retrieval baseline.

## Files

| File | Purpose |
|---|---|
| `feg_rag/rerank/qce_expansion.py` | Graph expansion index and budgeted expander |
| `feg_rag/rerank/qce_features.py` | Query, support, and conflict feature extraction |
| `feg_rag/rerank/qce_graph.py` | Router, scorer, and full QCE-Graph Lite model |
| `feg_rag/rerank/qce_dataset.py` | Training data construction and PyTorch dataset |
| `experiments/qce_graph_ablation.py` | Full ablation experiment script |
| `tests/test_qce_expansion.py` | Expansion unit tests |
| `tests/test_qce_features.py` | Feature extraction unit tests |
| `tests/test_qce_model.py` | Model unit tests |

## Methods (Ablation)

| Method | Description |
|---|---|
| `initial_retriever` | Baseline retrieval, no reranking |
| `rgcn` | Existing R-GCN results (read-only, no retraining) |
| `qce_fixed` | Fixed uniform relation budgets + expansion + fixed scoring |
| `qce_router` | Learnable router + budgeted expansion, no conflict channel |
| `qce_counterfactual` | Support/conflict scoring only, no expansion |
| `qce_full_no_rgcn` | Full model without R-GCN score |
| `qce_full` | Full model with optional R-GCN score |

## Quick Start

### Smoke Test

```bash
PYTHONPATH=. python experiments/qce_graph_ablation.py \
  --sanity \
  --methods initial_retriever,qce_fixed,qce_counterfactual \
  --device cpu \
  --epochs 2 \
  --progress_every 20
```

### Full Experiment

```bash
PYTHONPATH=. python experiments/qce_graph_ablation.py \
  --methods initial_retriever,rgcn,qce_fixed,qce_router,qce_counterfactual,qce_full_no_rgcn,qce_full \
  --initial_results_jsonl outputs/<bge_results>.jsonl \
  --rgcn_results_jsonl outputs/<rgcn_results>.jsonl \
  --graph_cache cache/<graph_cache>.pkl \
  --corpus_cache cache/<corpus_cache>.pkl \
  --top_n 50 \
  --expansion_budget 30 \
  --max_total_candidates 80 \
  --device cuda \
  --seeds 42,43,44 \
  --output_dir outputs/qce_graph/heldout_main \
  --progress_every 100
```

### Run Tests

```bash
pytest tests/test_qce_expansion.py tests/test_qce_features.py tests/test_qce_model.py -v
```

## Configuration

All parameters are in `configs/default.yaml` under the `qce_graph` key and can be overridden via CLI:

| Parameter | Default | Description |
|---|---|---|
| `initial_top_n` | 50 | Top-N initial candidates |
| `expansion_budget` | 30 | Max expanded candidates per query |
| `max_total_candidates` | 80 | Max total candidates after expansion |
| `relation_threshold` | 0.10 | Min router probability to activate a relation |
| `support_scale_max` | 0.20 | Max support contribution to final score |
| `conflict_scale_max` | 0.25 | Max conflict penalty to final score |
| `expansion_scale_max` | 0.10 | Max expansion priority contribution |
| `lambda_router` | 0.20 | Router auxiliary loss weight |
| `lambda_scale` | 0.001 | Scale regularization weight |

## Output Files

Each run produces in `output_dir`:

- `config_snapshot.yaml` — Full configuration used
- `per_query_results.jsonl` — Per-query rankings with relation probabilities
- `metrics_summary.csv` — MRR, Recall@K, nDCG@K per method
- `ablation_summary.csv` — Expansion diagnostics (before/after recall)
- `expansion_stats.json` — Aggregate expansion statistics
- `relation_usage.csv` — Per-relation activation frequency
- `relation_recovery.csv` — Per-relation gold recovery frequency
- `run.log` — Full run log

## Requirements

- No modifications to vanilla R-GCN baseline
- No gold evidence in candidate expansion
- All methods use same train/val/test split and gold mapping
- 3 random seeds for mean ± std reporting
- Configurable via CLI for all parameters

## Relationship to R-GCN

R-GCN: `fixed candidates → multi-relation message passing → rerank existing candidates`

QCE-Graph Lite: `initial candidates → query-conditioned routing → budgeted expansion → support/conflict scoring → rerank`

R-GCN is a strong passage-level reranker. QCE-Graph Lite is not intended to replace it but to validate whether:

1. Query-conditioned relation routing improves candidate selection
2. Graph expansion recovers missed gold evidence within a fixed budget
3. Support/conflict dual-channel scoring improves financial evidence ranking

Results must be reported honestly, including negative or null results from ablation studies.
