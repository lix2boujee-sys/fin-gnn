# Fast Final Graph Release Notes

## Added

- `feg_rag/rerank/fast_final_graph.py`
  - Lightweight query-adaptive fusion reranker.
  - Dataset cache for materialized training features.
  - Checkpoint save/load.
  - PPR/graph feature support.

- `experiments/table1_non_llm_reranking_comparison.py`
  - `--methods fast_final_graph`
  - `--fast_graph_eval_split {all,heldout}`
  - `--fast_graph_ppr_results_jsonl`
  - `--fast_graph_model_cache`
  - `--fast_graph_max_entity_weight`
  - `--fast_graph_delta_scale`

## Final Recommended Setting

```text
--fast_graph_eval_split heldout
--fast_graph_epochs 20
--fast_graph_batch_size 512
--fast_graph_min_rgcn_weight 0.45
--fast_graph_min_bge_weight 0.20
--fast_graph_max_entity_weight 0.10
--fast_graph_delta_scale 0.05
```

## Held-Out Result

```text
MRR=0.1250
R@5=0.1583
R@10=0.2031
nDCG@5=0.1254
nDCG@10=0.1399
```

