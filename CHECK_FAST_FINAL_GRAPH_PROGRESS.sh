#!/usr/bin/env bash
set -euo pipefail

cd /root/fin-gnn

echo "== Fast Final Graph jobs =="
pgrep -af "table1_non_llm_reranking_comparison.py.*fast_final_graph" || echo "no Fast Final Graph job running"

echo
echo "== Fast Final Graph log =="
grep -E "eval_split|Dataset cache|Train pairs|Val pairs|Epoch|avg_w|Evaluation done|TABLE 1 RESULTS|Fast Final|Traceback|Error" \
  outputs/logs/v2_table2_fast_final_graph_bounded_heldout.log 2>/dev/null | tail -120

echo
echo "== Latest held-out output =="
LATEST=$(ls -dt outputs/v2_table2_fast_final_graph_bounded_heldout_* 2>/dev/null | head -1 || true)
echo "${LATEST:-none}"
if [ -n "${LATEST:-}" ]; then
  find "$LATEST" -maxdepth 2 -type f -printf "%TY-%Tm-%Td %TH:%TM %10s %p\n" 2>/dev/null | sort
  echo
  cat "$LATEST/table1_non_llm_reranking_comparison.csv" 2>/dev/null || true
fi

