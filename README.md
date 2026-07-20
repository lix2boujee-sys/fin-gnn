# FEG-RAG

Financial Evidence Graph RAG — 在 [FinDER](https://huggingface.co/datasets) 金融 QA 基准上，对比纯检索、图 + PPR 重排、GNN 重排（GCN-style GNN / R-GCN）。

构建 **Financial Evidence Graph**（company / filing / section / chunk / metric / year），系统评估 BM25、Dense、Hybrid 及图方法对证据检索的影响。

---

## 实验概览

| 实验 | 脚本 | 研究问题 |
|------|------|----------|
| **Exp1** | `experiments/exp1_retrieval_baseline.py` | 无图结构下，纯检索 baseline 上限是多少？ |
| **Exp2** | `experiments/exp2_error_analysis.py` | 检索失败是金融结构错误还是完全未命中？ |
| **Exp3** | `experiments/exp3_feg_ppr.py` | 建图 + PPR 能否在不训练的情况下提升排序？ |
| **Exp4** | `experiments/exp4_gnn_reranker.py` | 训练的 GNN 能否超过 PPR / Hybrid？ |

**数据集**：FinDER 全量 **5703** QA 样本，语料 **31607** chunks（含 10-K 干扰项）  
**Dense 模型**：本地 `all-MiniLM-L6-v2`（`cache/models/all-MiniLM-L6-v2`）  
**评估协议**：Exp1 / Exp3 / Exp4 均在 **5703 全测试集** 上报告指标（Exp4 用 80% 训练 GNN，全量评估）

---

## 实验结果

### Exp1 — 纯检索 Baseline

| 方法 | R@5 | **R@10** | MRR | nDCG@10 |
|------|-----|---------|-----|---------|
| BM25 | 0.135 | 0.167 | 0.131 | 0.122 |
| Dense | 0.160 | 0.198 | 0.145 | 0.143 |
| **Hybrid** | **0.200** | **0.244** | **0.184** | **0.179** |

**结论**：Hybrid 为后续实验的对照上限；无图结构下 R@10 约 24.4%。

输出：`outputs/exp1_baseline/metrics_summary.csv`

### Table I — 全量检索对比 (First-Stage Retrieval, 5 methods)

完整 Table I 包含 BM25 / Dense / Hybrid / ColBERTv2 / E5-Mistral-7B-Instruct。
运行方式见 `outputs/E5_MISTRAL_RUN_GUIDE.md`。

> **⚠️ E5-Mistral-7B-Instruct underperforms on FinDER.** 之前基于
> `transformers.AutoModel` + 手动 last-token pooling 的结果 R@10 ≈ 0.06 异常低。
> 当前代码已修复为 `SentenceTransformer` 后端（`feg_rag/retrieval/dense.py`），
> 正确的 prompt 格式和 pooling 策略由模型自带配置处理。具体 R@10 值待重新运行后填入。
> 如修复后 E5 仍然很低，也要如实保留结果，并在 README 标注
> "E5 underperforms; do not fabricate or tune by test labels."

---

### Exp2 — 错误类型分析

基于 Exp1 Hybrid top-10 失败案例标注：

| 错误类型 | Hybrid 数量 | 占比（约） |
|----------|------------:|-----------:|
| **Missing Evidence** | 4030 | 84% |
| Wrong Metric | 423 | 9% |
| Wrong Company | 155 | 3% |
| Wrong Year | 98 | 2% |

**结论**：失败以「top-10 完全未命中 gold 证据」为主；Wrong Year / Metric / Company 支持建图动机，但占比远小于 Missing Evidence。

输出：`outputs/exp2_error_analysis/error_type_summary.csv`

---

### Exp3 — Financial Evidence Graph + PPR

PPR 在候选子图上运行，并与 Hybrid 检索分融合（`retrieval_weight=0.5`）。

| 方法 | MRR | **R@10** |
|------|-----|---------|
| Hybrid | 0.179 | 0.244 |
| Semantic + PPR | 0.179 | 0.244 |
| Financial + PPR | 0.177 | 0.244 |
| Full + PPR | 0.177 | 0.244 |
| Full + PPR + Constraint | 0.176 | 0.244 |

**结论**：PPR 重排不改变 R@10；MRR 与 Hybrid 基本持平，无显著提升。

输出：`outputs/exp3_feg_ppr/metrics_summary.csv`

---

### Exp4 — GNN Reranker（GCN-style GNN，50 epoch）

| 方法 | MRR | R@5 | **R@10** | nDCG@10 |
|------|-----|-----|---------|---------|
| Hybrid | 0.179 | 0.200 | 0.244 | 0.179 |
| Hybrid + PPR | 0.178 | 0.200 | 0.244 | 0.178 |
| **Hybrid + GCN-style GNN** | 0.172 | 0.206 | **0.253** | 0.178 |

- **训练**：4563 样本（80%），Loss 0.479 → 0.278（−42%），GNN 训练 ~19 min
- **总耗时**：~133 min（Dense CPU 编码 + 5703 全量评估）
- **Checkpoint**：`outputs/exp4_gnn_fulltest/model_checkpoints/exp4_gnn_reranker_20260710_183019.pt`

**结论**：GCN-style GNN 在全测试集上 **R@10=0.253**，比 Hybrid 高 **+0.9pp**；MRR 略低于 Hybrid。

输出：`outputs/exp4_gnn_fulltest/metrics_summary.csv`

---

### 跨实验总表（5703 全测试，R@10 可直接对比）

| 方法 | MRR | **R@10** | 实验 |
|------|-----|---------|------|
| BM25 | 0.131 | 0.167 | Exp1 |
| Dense | 0.145 | 0.198 | Exp1 |
| Hybrid | **0.184** | 0.244 | Exp1 |
| Hybrid + Full PPR | 0.177 | 0.244 | Exp3 |
| **Hybrid + GCN-style GNN** | 0.172 | **0.253** | Exp4 |

| 指标 | 最佳方法 |
|------|----------|
| **Recall@10** | Hybrid + GCN-style GNN（0.253） |
| **MRR** | Hybrid（0.184） |

更详细汇总见 [`outputs/experiments_summary.md`](outputs/experiments_summary.md)。

---

## 快速开始

### Benchmark corpus policy

By default, benchmark retrieval builds the corpus from source filing documents
under `edgar_dir` / `10-k`. Gold evidence snippets from FinDER are aligned back
to stable source-document chunk IDs; they are not inserted into the retrieval
corpus. If no source documents are found, the code raises an error.

`allow_gold_only_corpus: true` or `--allow_gold_only_corpus` is for explicit
debug/smoke tests only and must not be used for paper results.

Use `experiments/table1_non_llm_reranking_comparison.py` for current benchmark
results. Older legacy scripts such as `experiments/exp1_retrieval_baseline.py`
and `experiments/table1_retrieval_rerank.py` originally chunked FinDER annotated
gold snippets directly into the candidate pool; those outputs should be treated
as legacy/debug diagnostics, not as the paper's full-filing retrieval setting.

The historical method key `graphsage` is also a compatibility name. Its current
implementation is a dense GCN-style adjacency propagation baseline, not a strict
Hamilton et al. GraphSAGE/SAGEConv model. Paper tables should label it as
`GCN-style GNN` unless a true SAGEConv implementation is added.

### 1. 环境（推荐 D 盘，少占 C 盘）

```powershell
cd D:\fin-gnn
.\setup_env.ps1          # conda-env + cache 指向 D:
```

解释器：`D:\fin-gnn\conda-env\python.exe`（见 `.vscode/settings.json`）

缓存环境变量：

```powershell
$env:HF_HOME='D:\fin-gnn\cache\huggingface'
$env:PIP_CACHE_DIR='D:\fin-gnn\cache\pip'
$env:TMP='D:\fin-gnn\.tmp'
$env:TEMP='D:\fin-gnn\.tmp'
```

### 2. 数据

```powershell
python download_finder.py   # FinDER parquet → FinDER/data/
python extract_10k.py         # 可选：10-K 干扰项 → 10-k/
python check_finder.py
```

Dense 模型目录（或修改 `configs/default.yaml` → `retrieval.dense_model`）：

```
cache/models/all-MiniLM-L6-v2/
```

### 3. 运行实验

```powershell
# Exp1 — 检索 baseline（全量）
python experiments/exp1_retrieval_baseline.py --num_samples 0 --top_k 10

# Exp2 — 错误分析（需 Exp1 输出）
python experiments/exp2_error_analysis.py

# Exp3 — 建图 + PPR（全量）
python experiments/exp3_feg_ppr.py --num_samples 0 --top_n 50 --dense_device cpu

# Exp4 — GNN 重排（全量，评估与 Exp1 对齐）
python experiments/exp4_gnn_reranker.py --num_samples 0 --epochs 50 --no_ablation `
  --eval_on all --dense_device cpu --device cuda `
  --output_dir outputs/exp4_gnn_fulltest

# 独立 GNN 训练（不含完整 Exp4 评估流程）
python experiments/train_gnn.py --device cuda --epochs 50 --num_samples 0
```

Smoke test：

```powershell
python experiments/exp4_gnn_reranker.py --sanity --device cuda
```

---

## 项目结构

```
feg_rag/              # 核心库（retrieval, graph, rerank, eval）
experiments/          # Exp1–4 脚本
configs/default.yaml  # 路径与超参
FinDER/data/          # 数据集（不入库）
10-k/                 # EDGAR 干扰文档（不入库）
cache/                # 模型与 HF 缓存（不入库）
outputs/              # 实验输出（不入库）
conda-env/            # 本地 Python 环境（不入库）
```

---

## 硬件说明

| 配置 | 建议 |
|------|------|
| **RTX 3050 Ti 4GB**（当前） | Dense 编码用 `--dense_device cpu`；GNN 用 `--device cuda` |
| **RTX 3080 Ti 12GB** | Dense + GNN 均可 GPU，全量 Exp4 约 60–80 min |
| **云 A100** | Dense batch 加大，全量 Exp4 约 35–55 min |

Exp4 全量（5703，`eval_on all`）在本机约 **2–2.5 小时**，瓶颈在 CPU Dense 编码与 PPR 全量评估。

---

## 依赖

```powershell
pip install -r requirements.txt
```

PyTorch CUDA 版通过 `setup_env.ps1` 安装在 `conda-env/`。

---

## 文档

- 实验说明：`finder_exp1_baseline_instruction.md`、`finder_all_experiments_instruction.md`
- 设计文档：`financial_graph_rag_experiment_design_cn.md`、`financial_graph_rag_paper_plan_cn.md`
- 结果汇总：`outputs/experiments_summary.md`

---

## License

Research / academic use. 公开发布前请补充 License。
