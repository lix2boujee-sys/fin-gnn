#!/usr/bin/env bash
set -euo pipefail

cd /root/fin-gnn
mkdir -p outputs/logs cache/rerank_scores

FFG_OUT=/root/fin-gnn/outputs/v2_table2_fast_final_graph_bounded_heldout_$(date +%Y%m%d_%H%M%S)
FFG_LOG=/root/fin-gnn/outputs/logs/v2_table2_fast_final_graph_bounded_heldout.log

CUDA_VISIBLE_DEVICES=0 nohup python -u experiments/table1_non_llm_reranking_comparison.py \
  --config configs/table1_non_llm_reranking_e5_mistral_cloud.yaml \
  --output_dir "$FFG_OUT" \
  --corpus_cache cache/table1_full_corpus_seq4096.pkl \
  --graph_cache cache/table2_graph_features_bge_pool_seq4096.pkl \
  --graph_feature_retriever_cache cache/retrieval_indexes/bge_m3_dense \
  --candidate_results_jsonl outputs/v2_table1_bge_m3_correct_corpus_20260715_123130/bge_m3_dense_results.jsonl \
  --candidate_pool_name BGE-M3-Dense \
  --methods fast_final_graph \
  --fast_graph_eval_split heldout \
  --fast_graph_epochs 20 \
  --fast_graph_batch_size 512 \
  --fast_graph_min_rgcn_weight 0.45 \
  --fast_graph_min_bge_weight 0.20 \
  --fast_graph_max_entity_weight 0.10 \
  --fast_graph_delta_scale 0.05 \
  --fast_graph_rgcn_results_jsonl outputs/v2_table2_graph_bge_pool_rgcn_eval_fast_20260716_032108/rgcn_results.jsonl \
  --fast_graph_monot5_results_jsonl outputs/v2_table2_mono_t5_bge_pool_20260716_042622/mono_t5_results.jsonl \
  --fast_graph_ppr_results_jsonl outputs/v2_table2_graph_bge_pool_a_ppr_sage_20260715_143613/ppr_results.jsonl \
  --fast_graph_model_cache cache/rerank_scores/fast_final_graph_bounded_heldout_dataset.pkl \
  --rerank_checkpoint_every 200 \
  --device cuda \
  --dense_device cuda \
  > "$FFG_LOG" 2>&1 &

echo "FFG_PID=$!"
echo "FFG_OUT=$FFG_OUT"
echo "FFG_LOG=$FFG_LOG"

