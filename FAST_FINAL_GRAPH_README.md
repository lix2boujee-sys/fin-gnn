# Fast Final Graph

This release adds the final lightweight Fast Final Graph reranker used in the
held-out evidence ranking experiment.

## Files

- `feg_rag/rerank/fast_final_graph.py`
  - Query-adaptive fusion model.
  - Uses BGE-M3, R-GCN, MonoT5, optional PPR/graph scores, and financial entity
    match features.
  - Includes dataset caching, checkpoint save/load, bounded residual scoring,
    and entity weight capping.

- `experiments/table1_non_llm_reranking_comparison.py`
  - Main experiment runner.
  - Adds `--methods fast_final_graph`.
  - Adds held-out evaluation with `--fast_graph_eval_split heldout`.

## Method

Fast Final Graph is a supervised late-fusion reranker. It does not rerun BGE,
MonoT5, PPR, or R-GCN. It reads their saved JSONL outputs and trains a small MLP
to combine them.

```text
score =
  w_bge    * BGE_score
+ w_rgcn   * RGCN_score
+ w_monot5 * MonoT5_score
+ w_entity * Entity_score
+ w_graph  * PPR_score
+ delta_scale * tanh(delta_mlp(features))
```

The final version uses:

- `--fast_graph_min_rgcn_weight 0.45`
- `--fast_graph_min_bge_weight 0.20`
- `--fast_graph_max_entity_weight 0.10`
- `--fast_graph_delta_scale 0.05`

These constraints reduce shortcut overfitting while preserving strong R-GCN and
BGE rankings.

## Held-Out Result

Recommended reporting result:

```text
Method                   MRR     R@5     nDCG@5   R@10    nDCG@10
Fast Final Graph (Ours)  0.1250  0.1583  0.1254   0.2031  0.1399
```

Interpretation:

- Best observed held-out MRR, Recall@10, and nDCG@10.
- Recall@5 is slightly below standalone R-GCN.
- Use held-out results for paper reporting, not full-data eval.

## Required Existing Artifacts

The full cloud command assumes these files already exist:

```text
cache/table1_full_corpus_seq4096.pkl
cache/table2_graph_features_bge_pool_seq4096.pkl
cache/retrieval_indexes/bge_m3_dense
outputs/v2_table1_bge_m3_correct_corpus_20260715_123130/bge_m3_dense_results.jsonl
outputs/v2_table2_graph_bge_pool_rgcn_eval_fast_20260716_032108/rgcn_results.jsonl
outputs/v2_table2_mono_t5_bge_pool_20260716_042622/mono_t5_results.jsonl
outputs/v2_table2_graph_bge_pool_a_ppr_sage_20260715_143613/ppr_results.jsonl
```

## Validation

```bash
python -m py_compile \
  feg_rag/rerank/fast_final_graph.py \
  experiments/table1_non_llm_reranking_comparison.py
```

