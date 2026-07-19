# GATv2 Strong Baseline and Finance Error Analysis

This release update adds two experiment-facing components:

1. Strong GATv2 baseline for evidence reranking.
   - Residual dense GATv2 layers with LayerNorm.
   - Stronger default capacity: hidden 96, output 48.
   - Stronger retrieval-preserved fusion: retrieval 0.70, GATv2 0.30.

2. Finance-specific structural error analysis.
   - Computes Wrong Company, Wrong Year, Wrong Metric, and Missing Evidence.
   - Supports simple Table-1 JSONL result files by using chunk metadata and text extraction fallback.
   - Reports both top-10 and top-5 error rates.

Useful cloud commands:

```bash
PYTHONPATH=/root/fin-gnn python -m py_compile \
  feg_rag/rerank/gnn.py \
  feg_rag/rerank/train.py \
  feg_rag/rerank/__init__.py \
  experiments/table1_non_llm_reranking_comparison.py \
  experiments/finance_error_analysis.py \
  tests/test_gnn_reranker.py
```

```bash
PYTHONPATH=/root/fin-gnn python experiments/finance_error_analysis.py \
  --config configs/table1_non_llm_reranking_e5_mistral_cloud.yaml \
  --output_dir outputs/error_analysis \
  --top_k 10 \
  --also_k5
```
