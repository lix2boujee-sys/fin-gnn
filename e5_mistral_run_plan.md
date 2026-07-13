# E5-Mistral 实验运行计划

> 日期：2026-07-11
> 目标：用 `intfloat/e5-mistral-7b-instruct` (4096-dim) 替代 MiniLM (384-dim) 跑全量实验，做 apples-to-apples 对比

---

## 前置条件

- [x] 代码已改完（E5 格式化、动态维度、batch size 控制、配置分离）
- [ ] 云 GPU 租好（建议 RTX 4090 24GB，约 ¥1.5-2/时）
- [ ] 模型下载到云机器：`intfloat/e5-mistral-7b-instruct`（~14GB）
- [ ] 项目代码 + FinDER 数据 + 10-K 数据上传到云机器

---

## 已有 MiniLM Baseline（用于对比）

| 实验 | 方法 | MRR | R@5 | R@10 |
|---|---|---|---|---|
| Exp1 | BM25 | 0.1306 | 0.1347 | 0.1674 |
| Exp1 | Dense (MiniLM 384d) | 0.1450 | 0.1603 | 0.1980 |
| Exp1 | Hybrid | 0.1841 | 0.2001 | 0.2439 |
| Exp4 | Hybrid | 0.1787 | 0.2002 | 0.2439 |
| Exp4 | Hybrid + PPR | 0.1775 | 0.2002 | 0.2439 |
| Exp4 | Hybrid + **GraphSAGE** | 0.1723 | **0.2056** | **0.2529** |

---

## Step 1: Smoke Test（验证 Pipeline）

> 目标：确认 E5 模型加载、文本格式化、维度正确、输出写入正常
> 预计：10 分钟

```powershell
# Exp1 smoke — 20 samples, batch=1, CPU
python experiments/exp1_retrieval_baseline.py `
  --config configs/e5_mistral.yaml `
  --num_samples 20 `
  --top_k 10 `
  --dense_device cuda `
  --dense_batch_size 2 `
  --output_dir outputs/smoke_exp1_e5_mistral `
  --overwrite_output_dir

# Exp4 smoke — sanity mode
python experiments/exp4_gnn_reranker.py `
  --config configs/e5_mistral.yaml `
  --sanity `
  --dense_device cuda `
  --dense_batch_size 2 `
  --device cuda `
  --output_dir outputs/smoke_exp4_e5_mistral `
  --overwrite_output_dir
```

### Smoke 检查清单

- [ ] Dense backend 打印了 E5-Mistral，不是 MiniLM
- [ ] Dense embedding dim = **4096**
- [ ] Exp4 feature dim = **4103** (4096 + 1 + 3 + 2 + 1)
- [ ] 没有 "expected 391" 之类的 MiniLM 警告
- [ ] `metrics_summary.csv` 和 JSONL 文件正常写出
- [ ] 日志中能看到 E5 查询被格式化成 `Instruct: ... Query: ...`

---

## Step 2: Exp1 全量（Retrieval Baseline）

> 预计：2-3 小时

```powershell
python experiments/exp1_retrieval_baseline.py `
  --config configs/e5_mistral.yaml `
  --num_samples 0 `
  --top_k 10 `
  --dense_device cuda `
  --dense_batch_size 4 `
  --output_dir outputs/exp1_e5_mistral `
  --overwrite_output_dir
```

### 输出
- `outputs/exp1_e5_mistral/bm25_results.jsonl`
- `outputs/exp1_e5_mistral/dense_results.jsonl`
- `outputs/exp1_e5_mistral/hybrid_results.jsonl`
- `outputs/exp1_e5_mistral/metrics_summary.csv`
- `outputs/exp1_e5_mistral/failed_cases.jsonl`
- `outputs/exp1_e5_mistral/README.md`

---

## Step 3: Exp4 全量（GNN Reranker）

> 预计：3-5 小时（编码 2h + GNN 训练 1h + 评估 1h）

```powershell
python experiments/exp4_gnn_reranker.py `
  --config configs/e5_mistral.yaml `
  --num_samples 0 `
  --epochs 50 `
  --no_ablation `
  --eval_on all `
  --dense_device cuda `
  --dense_batch_size 4 `
  --device cuda `
  --output_dir outputs/exp4_e5_mistral_gnn `
  --overwrite_output_dir
```

> **注意**：如果云 GPU 显存不够同时跑 E5 编码 + GNN 训练，可以先跑完编码后用 `--dense_device cpu` 跑 GNN（编码结果已在 FAISS index 里），但 4096-dim × 31k 的 FAISS index 约 500MB，不会占太多显存。

### 输出
- `outputs/exp4_e5_mistral_gnn/hybrid_results.jsonl`
- `outputs/exp4_e5_mistral_gnn/hybrid_ppr_results.jsonl`
- `outputs/exp4_e5_mistral_gnn/hybrid_sage_results.jsonl`
- `outputs/exp4_e5_mistral_gnn/metrics_summary.csv`
- `outputs/exp4_e5_mistral_gnn/train_config.yaml`
- `outputs/exp4_e5_mistral_gnn/README.md`

---

## Step 4: 结果对比

### Exp1 对比

| 方法 | MiniLM MRR | E5-Mistral MRR | MiniLM R@10 | E5-Mistral R@10 |
|---|---|---|---|---|
| BM25 | 0.1306 | （同） | 0.1674 | （同） |
| Dense | 0.1450 | ？ | 0.1980 | ？ |
| Hybrid | 0.1841 | ？ | 0.2439 | ？ |

> BM25 不受 dense model 影响，两列数值应完全一致

### Exp4 对比

| 方法 | MiniLM MRR | E5-Mistral MRR | MiniLM R@10 | E5-Mistral R@10 |
|---|---|---|---|---|
| Hybrid | 0.1787 | ？ | 0.2439 | ？ |
| Hybrid + PPR | 0.1775 | ？ | 0.2439 | ？ |
| Hybrid + GraphSAGE | 0.1723 | ？ | 0.2529 | ？ |

### 预期

根据 FinDER 原论文，E5-Mistral 在金融文本上的表现显著优于 MiniLM：
- Dense R@10 可能从 0.198 → **0.25+**
- Hybrid R@10 可能从 0.244 → **0.30+**
- GNN 在此基础上可能进一步受益（更大的特征空间）

---

## 可选：Exp3 PPR（如果 Exp4 效果好的话）

```powershell
python experiments/exp3_feg_ppr.py `
  --config configs/e5_mistral.yaml `
  --num_samples 0 `
  --top_n 50 `
  --dense_device cuda `
  --dense_batch_size 4 `
  --output_dir outputs/exp3_e5_mistral_ppr `
  --overwrite_output_dir
```

---

## 云 GPU 推荐配置

| 需求 | 规格 |
|---|---|
| GPU | RTX 4090 / A5000 / A10（≥24GB VRAM） |
| CPU | 8 核+ |
| RAM | 32GB+ |
| 磁盘 | 50GB+（模型 14GB + 数据 ~500MB + 依赖） |
| 平台 | AutoDL（国内 ¥1.5/时）、恒源云、RunPod |

---

## 云 GPU 详细操作指南（AutoDL 推荐）

> 第一次用云 GPU？下面每一步都写清楚了，跟着做就行。

### 1. 注册 & 充值

1. 打开 https://www.autodl.com （国内平台，中文界面）
2. 注册账号 → 实名认证（需要身份证，几分钟）
3. 充值 ¥20-30 就够（RTX 4090 约 ¥1.5/时，8 小时约 ¥12）

### 2. 租机器

1. 首页点「算力市场」
2. 筛选：
   - GPU：**RTX 4090**（24GB，性价比最高）
   - 地区：选离你近的（北京/上海/广州）
   - 镜像：选 **PyTorch 2.x + Python 3.10+** 的官方镜像
3. 点「租用」，创建实例（1 卡就够，不用多卡）

### 3. 连接机器

租好后会看到一个实例卡片，上面有：
- **JupyterLab 地址**（点开就能用终端）
- **SSH 命令**（本地终端连接）

推荐直接用 JupyterLab：点开 → 打开 Terminal（终端），后续所有命令在这里执行。

### 4. 上传项目

**方法 A：JupyterLab 网页上传（推荐）**
- 在 JupyterLab 左侧文件管理器中，把项目文件夹拖进去
- 大文件（10-K HTML、模型）上传慢，用方法 B

**方法 B：用 AutoDL 的「数据存储」**
- AutoDL 提供 `/root/autodl-tmp/` 目录，是高速 SSD
- 也可以挂载网盘（阿里云盘等）

**你需要上传的：**
```
fin-gnn/
├── configs/          # 配置文件（含 e5_mistral.yaml）
├── experiments/      # 实验脚本
├── feg_rag/          # 核心代码
├── FinDER/           # 数据集（~10MB，parquet）
├── 10-k/             # 10-K HTML 文件（你本地有，需上传）
├── requirements.txt
└── run_pipeline.py
```

> ⚠️ `outputs/` 和 `cache/models/` 太大（MiniLM 模型 ~90MB），不用上传，在云端重新下载即可。

### 5. 安装依赖 & 下载模型

```bash
# 进入项目目录
cd /root/autodl-tmp/fin-gnn   # 或你上传到的路径

# 安装依赖（PyTorch 镜像已经装了 torch，只需装其他）
pip install -r requirements.txt

# 下载 E5-Mistral 模型（~14GB，云上很快，5-10分钟）
# 放到 configs/e5_mistral.yaml 指定的路径
mkdir -p /root/autodl-tmp/fin-gnn/cache/models/e5-mistral-7b-instruct
python -c "
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('intfloat/e5-mistral-7b-instruct')
model.save('/root/autodl-tmp/fin-gnn/cache/models/e5-mistral-7b-instruct')
"

# 也下载 MiniLM（Exp4 可能需要对比，而且很小 ~90MB）
mkdir -p /root/autodl-tmp/fin-gnn/cache/models/all-MiniLM-L6-v2
python -c "
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
model.save('/root/autodl-tmp/fin-gnn/cache/models/all-MiniLM-L6-v2')
"
```

### 6. 修改配置中的路径

云机器上路径不同，需要改两个 yaml 文件：

```bash
# 快速替换：把 D:/fin-gnn 替换为云机器路径
sed -i 's|D:/fin-gnn|/root/autodl-tmp/fin-gnn|g' configs/default.yaml
sed -i 's|D:/fin-gnn|/root/autodl-tmp/fin-gnn|g' configs/e5_mistral.yaml
```

### 7. 开 tmux 跑实验（防断线）

```bash
# 创建 tmux 会话（关掉网页也不会中断）
tmux new -s e5_exp

# 然后按顺序跑实验...

# 随时可以用 Ctrl+B 然后按 D 断开（程序继续跑）
# 重新连接：tmux attach -t e5_exp
```

### 8. 下载结果

实验跑完后，在 JupyterLab 中：
- 右键 `outputs/` 目录 → 下载为 zip
- 或打包后下载：`tar -czf results.tar.gz outputs/`

### 9. 释放机器

> ⚠️ 跑完记得关机/释放！不然会一直扣费！

- AutoDL 实例页 → 点「关机」或「释放」
- 关机：数据保留，下次开机继续用（但会收少量存储费）
- 释放：机器销毁，数据清空（记得先把结果下载下来）

---

## 云环境初始化（一键脚本）

把下面保存为 `cloud_setup.sh`，上传后执行一次：

```bash
#!/bin/bash
set -e

PROJECT_DIR="/root/autodl-tmp/fin-gnn"
cd "$PROJECT_DIR"

echo "=== 1. Install dependencies ==="
pip install -r requirements.txt -q

echo "=== 2. Download MiniLM (~90MB) ==="
mkdir -p cache/models/all-MiniLM-L6-v2
python -c "
from sentence_transformers import SentenceTransformer
m = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
m.save('cache/models/all-MiniLM-L6-v2')
print('MiniLM ✓')
"

echo "=== 3. Download E5-Mistral (~14GB) ==="
mkdir -p cache/models/e5-mistral-7b-instruct
python -c "
from sentence_transformers import SentenceTransformer
m = SentenceTransformer('intfloat/e5-mistral-7b-instruct')
m.save('cache/models/e5-mistral-7b-instruct')
print('E5-Mistral ✓')
"

echo "=== 4. Fix paths ==="
sed -i 's|D:/fin-gnn|/root/autodl-tmp/fin-gnn|g' configs/default.yaml
sed -i 's|D:/fin-gnn|/root/autodl-tmp/fin-gnn|g' configs/e5_mistral.yaml

echo "=== Done! Ready to run experiments ==="
echo "nvidia-smi:"
nvidia-smi
```

---

## 风险与注意事项

1. **4GB 小显存卡**：7B 模型装不下，必须用云 GPU 或 24GB+ 的卡
2. **E5 本地路径不存在**：代码现在会 **fail loud**（不会静默 fallback 到 MiniLM），请确保模型已下载
3. **batch_size=4 太大导致 OOM**：降到 2 或 1
4. **Exp4 训练失败**（训练对不足）：检查日志中的 `train_pairs` 数量，应 > 1000
5. **FAISS 内存**：4096-dim × 31k = ~500MB，GPU 上没问题

---

## 文件汇总

| 文件 | 用途 |
|---|---|
| `configs/e5_mistral.yaml` | E5 专用配置 |
| `feg_rag/retrieval/dense.py` | E5 文本格式化 + fail-loud |
| `feg_rag/graph/features.py` | 动态特征维度 |
| `experiments/exp1_retrieval_baseline.py` | +`--dense_device` `--dense_batch_size` |
| `experiments/exp3_feg_ppr.py` | +`--dense_batch_size` |
| `experiments/exp4_gnn_reranker.py` | 动态维度 + `--dense_batch_size` |
| `e5_mistral_requirement_for_claude.md` | 原始需求文档 |
