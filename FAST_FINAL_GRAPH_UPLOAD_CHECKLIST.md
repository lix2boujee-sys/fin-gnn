# Fast Final Graph Upload Checklist

- [ ] Include `feg_rag/rerank/fast_final_graph.py`.
- [ ] Include updated `experiments/table1_non_llm_reranking_comparison.py`.
- [ ] Include `FAST_FINAL_GRAPH_README.md`.
- [ ] Include `FAST_FINAL_GRAPH_RELEASE_NOTES.md`.
- [ ] Include `RUN_FAST_FINAL_GRAPH_HELDOUT.sh`.
- [ ] Include `CHECK_FAST_FINAL_GRAPH_PROGRESS.sh`.
- [ ] Do not upload large runtime artifacts:
  - `cache/`
  - `outputs/`
  - `*.pkl`
  - `*.pt`
  - `*.bin`
  - `*.safetensors`
  - `*.pyc`
  - `__pycache__/`

