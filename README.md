# FEG-RAG

Financial Evidence Graph RAG — 在 [FinDER](https://huggingface.co/datasets) 金融 QA 基准上，对比纯检索、图 + PPR 重排、GNN 重排（GraphSAGE / R-GCN）。

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
**Dense 模型**：`all-MiniLM-L6-v2`（本地或 HuggingFace）  
**评估协议**：Exp1 / Exp3 / Exp4 均在 **5703 全测试集** 上报告指标

---

## 实验结果

### Exp1 — 纯检索 Baseline

| 方法 | R@5 | **R@10** | MRR | nDCG@10 |
|------|-----|---------|-----|---------|
| BM25 | 0.135 | 0.167 | 0.131 | 0.122 |
| Dense | 0.160 | 0.198 | 0.145 | 0.143 |
| **Hybrid** | **0.200** | **0.244** | **0.184** | **0.179** |

---

### Exp2 — 错误类型分析

| 错误类型 | Hybrid 数量 |
|----------|------------:|
| **Missing Evidence** | 4030 |
| Wrong Metric | 423 |
| Wrong Company | 155 |
| Wrong Year | 98 |

失败以「top-10 完全未命中 gold 证据」为主（约 84%）。

---

### Exp3 — Financial Evidence Graph + PPR

| 方法 | MRR | **R@10** |
|------|-----|---------|
| Hybrid | 0.179 | 0.244 |
| Semantic + PPR | 0.179 | 0.244 |
| Financial + PPR | 0.177 | 0.244 |
| Full + PPR | 0.177 | 0.244 |
| Full + PPR + Constraint | 0.176 | 0.244 |

PPR 在候选子图上运行并与 Hybrid 检索分融合；R@10 与 Hybrid 一致，MRR 基本持平。

---

### Exp4 — GNN Reranker（GraphSAGE，50 epoch，5703 全测试）

| 方法 | MRR | R@5 | **R@10** | nDCG@10 |
|------|-----|-----|---------|---------|
| Hybrid | 0.179 | 0.200 | 0.244 | 0.179 |
| Hybrid + PPR | 0.178 | 0.200 | 0.244 | 0.178 |
| **Hybrid + GraphSAGE** | 0.172 | 0.206 | **0.253** | 0.178 |

训练 Loss：0.479 → 0.278（−42%）。GraphSAGE **R@10=0.253**，比 Hybrid 高 +0.9pp。

---

### 跨实验总表（5703 全测试）

| 方法 | MRR | **R@10** | 实验 |
|------|-----|---------|------|
| BM25 | 0.131 | 0.167 | Exp1 |
| Dense | 0.145 | 0.198 | Exp1 |
| Hybrid | **0.184** | 0.244 | Exp1 |
| Hybrid + Full PPR | 0.177 | 0.244 | Exp3 |
| **Hybrid + GraphSAGE** | 0.172 | **0.253** | Exp4 |

| 指标 | 最佳方法 |
|------|----------|
| **Recall@10** | Hybrid + GraphSAGE（0.253） |
| **MRR** | Hybrid（0.184） |

---

## 快速开始

### 1. 环境

```powershell
conda create -p ./conda-env python=3.10
./conda-env/python.exe -m pip install -r requirements.txt
./setup_env.ps1
```

```powershell
$env:HF_HOME='./cache/huggingface'
$env:PIP_CACHE_DIR='./cache/pip'
$env:TMP='./.tmp'
$env:TEMP='./.tmp'
```

### 2. 数据

```powershell
python download_finder.py
python extract_10k.py          # 可选：10-K 干扰项
python check_finder.py
```

将 `all-MiniLM-L6-v2` 放到 `cache/models/`，或在 `configs/default.yaml` 中配置 `retrieval.dense_model`。

### 3. 运行实验

```powershell
python experiments/exp1_retrieval_baseline.py --num_samples 0 --top_k 10
python experiments/exp2_error_analysis.py
python experiments/exp3_feg_ppr.py --num_samples 0 --top_n 50 --dense_device cpu
python experiments/exp4_gnn_reranker.py --num_samples 0 --epochs 50 --no_ablation `
  --eval_on all --dense_device cpu --device cuda
```

---

## 项目结构

```
feg_rag/              # 核心库
experiments/          # Exp1–4
configs/default.yaml
FinDER/data/          # 数据集（不入库）
cache/                # 模型缓存（不入库）
outputs/              # 实验输出（不入库）
```

---

## 硬件说明

| GPU | 建议 |
|-----|------|
| 4 GB（如 3050 Ti） | `--dense_device cpu`，GNN 用 `--device cuda` |
| 12 GB+（如 3080 Ti） | Dense + GNN 均可 GPU |
| 云 A100 | 全 pipeline 约 35–55 min |

Exp4 全量在本机（CPU Dense）约 2–2.5 小时。

---

## 文档

- `finder_exp1_baseline_instruction.md` / `finder_all_experiments_instruction.md`
- `financial_graph_rag_experiment_design_cn.md` / `financial_graph_rag_paper_plan_cn.md`

---

## License

Research / academic use. 公开发布前请补充 License。
