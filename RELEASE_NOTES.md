# Release v1.0.0 — FinDER Exp1–4 完整实验结果

**发布日期**：2026-07-10

FinDER 全量 **5703** 样本上的 Financial Evidence Graph RAG 完整实验流水线与可复现代码。

---

## Highlights

- ✅ **Exp1–4 全部完成**，指标均在 **5703 全测试集** 上报告（Exp4 用 `--eval_on all` 与 Exp1 对齐）
- 🔧 **PPR 修复**：子图 PPR + 检索分融合，不再系统性损害 MRR
- 📈 **GraphSAGE R@10=0.253**，首次超过 Hybrid baseline（0.244）
- 🖥️ 支持 **4 GB 笔记本 GPU**（Dense CPU + GNN CUDA）

---

## 实验结果摘要

### Exp1 — Retrieval Baseline

| Method | R@10 | MRR |
|--------|------|-----|
| BM25 | 0.167 | 0.131 |
| Dense | 0.198 | 0.145 |
| **Hybrid** | **0.244** | **0.184** |

### Exp2 — Error Analysis（Hybrid 失败案例）

| Error Type | Count |
|------------|------:|
| Missing Evidence | 4030 |
| Wrong Metric | 423 |
| Wrong Company | 155 |
| Wrong Year | 98 |

### Exp3 — Graph + PPR

| Method | R@10 | MRR |
|--------|------|-----|
| Hybrid | 0.244 | 0.179 |
| Full + PPR | 0.244 | 0.177 |

PPR 不改变 Recall@10；MRR 与 Hybrid 基本持平。

### Exp4 — GNN Reranker（GraphSAGE, 50 epoch）

| Method | R@10 | MRR |
|--------|------|-----|
| Hybrid | 0.244 | 0.179 |
| Hybrid + PPR | 0.244 | 0.178 |
| **Hybrid + GraphSAGE** | **0.253** | 0.172 |

Training loss: 0.479 → 0.278 (−42%).

---

## 最佳结果

| Metric | Best | Value |
|--------|------|-------|
| **Recall@10** | Hybrid + GraphSAGE | **0.253** |
| **MRR** | Hybrid | **0.184** |

---

## What's Included

- `feg_rag/` — retrieval, graph builder, PPR reranker, GNN reranker, evaluation
- `experiments/` — Exp1–4 scripts + `train_gnn.py`
- `configs/default.yaml` — hyperparameters
- Experiment design docs (CN) + FinDER instruction specs
- `setup_env.ps1` / `setup_env.bat` — environment setup

**Not included** (download separately):

- FinDER dataset → `python download_finder.py`
- Dense model `all-MiniLM-L6-v2` → HuggingFace or local cache
- Experiment outputs → run scripts locally

---

## Reproduce

```powershell
pip install -r requirements.txt
python download_finder.py

# Full pipeline
python experiments/exp1_retrieval_baseline.py --num_samples 0 --top_k 10
python experiments/exp2_error_analysis.py
python experiments/exp3_feg_ppr.py --num_samples 0 --top_n 50 --dense_device cpu
python experiments/exp4_gnn_reranker.py --num_samples 0 --epochs 50 --no_ablation `
  --eval_on all --dense_device cpu --device cuda
```

**Estimated runtime** (RTX 3050 Ti 4GB): Exp1 ~1h, Exp3 ~2h, Exp4 ~2.5h.

---

## Changes since initial release

- Fix PPR reranking (`feg_rag/rerank/ppr.py`): subgraph + retrieval score fusion
- Exp4: add `--eval_on all` for Exp1-aligned full-test evaluation
- Exp4: `--dense_device cpu` default for 4GB GPU compatibility
- Complete Exp3 full run (5703) and Exp4 full test evaluation

---

## Full Changelog

See [README.md](README.md) for detailed tables and hardware notes.
