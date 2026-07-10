# FinDER / Financial Evidence Graph 实验说明文档

> 用途：把这份 Markdown 交给 Claude / Claude Code / coding agent，让它按照文字说明实现和运行实验。  
> 本文档不是论文正文，而是**实验执行说明**。目标是让 coding agent 不依赖图片，也能理解每个实验要做什么、为什么做、输入输出是什么、如何评价结果。

---

## 0. 项目总目标

本项目研究的是金融文档 RAG 中的 **evidence retrieval / evidence reranking** 问题。

给定一个金融问题 query，系统需要从金融文档语料中找出能支持答案的 evidence passages / chunks。后续可以把这些 evidence 交给 LLM 生成答案，但本文实验的核心首先是：

> 正确证据有没有被找回来？  
> 正确证据有没有排在靠前的位置？  
> 检索错误是不是来自金融结构约束没有被理解？

本项目的核心方法是构建 **Financial Evidence Graph, FEG**，把金融文档中的结构关系显式建模出来，例如：

- company；
- filing；
- filing type，例如 10-K / 10-Q；
- filing year；
- section；
- passage / chunk；
- financial metric；
- year / period；
- query entity。

然后用图算法或图神经网络做 evidence reranking，目标是减少普通 RAG 中常见的错误：

- Wrong Company；
- Wrong Year；
- Wrong Metric；
- Wrong Filing；
- Wrong Passage；
- Missing Evidence；
- Unsupported Answer。

---

## 1. 实验总路线

整个实验分成四个阶段：

| 实验 | 名称 | 核心问题 | 是否用图 | 是否训练 GNN |
|---|---|---|---|---|
| Exp1 | 普通检索 baseline | 不用图结构，普通检索在 FinDER 上能做到什么水平？ | 否 | 否 |
| Exp2 | 错误类型分析 | 普通检索为什么错？这些错误是不是金融结构错误？ | 否 | 否 |
| Exp3 | Financial Evidence Graph + PPR | 只用图算法，不训练 GNN，金融证据图能否改善 evidence ranking？ | 是 | 否 |
| Exp4 | R-GCN / GraphSAGE Evidence Reranker | 训练图重排器，是否比 PPR 和普通 reranker 更强？ | 是 | 是 |

四个实验之间的关系是：

```text
Exp1: 先建立普通检索 baseline
  ↓
Exp2: 分析 Exp1 失败在哪里，证明普通检索没有理解金融结构约束
  ↓
Exp3: 构建 Financial Evidence Graph，用 PPR 做无训练图重排
  ↓
Exp4: 在 Financial Evidence Graph 上训练 GraphSAGE / R-GCN reranker
```

Claude 执行时不要跳过 Exp1 和 Exp2，因为 Exp1/Exp2 是后续图方法的动机和对照。

---

## 2. 推荐项目目录结构

建议使用如下目录结构：

```text
project_root/
├── data/
│   ├── raw/
│   │   └── finder/
│   ├── processed/
│   │   ├── corpus_chunks.jsonl
│   │   ├── queries.jsonl
│   │   ├── qrels.jsonl
│   │   └── metadata.jsonl
│   └── cache/
│       ├── bm25_index/
│       ├── dense_embeddings.npy
│       └── graph_cache/
├── src/
│   ├── data/
│   │   ├── load_finder.py
│   │   ├── build_chunks.py
│   │   └── align_gold_evidence.py
│   ├── retrieval/
│   │   ├── bm25.py
│   │   ├── dense.py
│   │   ├── hybrid.py
│   │   └── cross_encoder.py
│   ├── graph/
│   │   ├── build_feg.py
│   │   ├── ppr.py
│   │   └── graph_utils.py
│   ├── reranker/
│   │   ├── dataset.py
│   │   ├── graphsage.py
│   │   ├── rgcn.py
│   │   └── train.py
│   ├── evaluation/
│   │   ├── retrieval_metrics.py
│   │   ├── error_analysis.py
│   │   └── report_tables.py
│   └── main.py
├── outputs/
│   ├── exp1_baseline/
│   ├── exp2_error_analysis/
│   ├── exp3_feg_ppr/
│   └── exp4_gnn_reranker/
├── configs/
│   ├── exp1.yaml
│   ├── exp2.yaml
│   ├── exp3.yaml
│   └── exp4.yaml
└── README.md
```

如果当前项目已经有自己的结构，可以不完全照搬，但必须保证输出文件清晰可追踪。

---

## 3. 统一数据格式

### 3.1 queries.jsonl

每一行表示一个问题：

```json
{
  "query_id": "q_0001",
  "question": "What was Apple's operating income in 2023?",
  "company": "Apple Inc.",
  "ticker": "AAPL",
  "target_year": "2023",
  "target_metric": "operating income",
  "answer": "..."
}
```

如果 FinDER 原始字段名不同，请在 `load_finder.py` 里做字段映射，并在日志中输出：

```text
FinDER field mapping:
raw question field -> question
raw references field -> gold_evidence
raw answer field -> answer
...
```

### 3.2 corpus_chunks.jsonl

每一行表示一个候选 evidence chunk：

```json
{
  "chunk_id": "c_000001",
  "doc_id": "AAPL_2023_10K",
  "company": "Apple Inc.",
  "ticker": "AAPL",
  "filing_type": "10-K",
  "filing_year": "2023",
  "section": "Management Discussion and Analysis",
  "text": "...",
  "metrics": ["operating income", "net sales"],
  "years": ["2022", "2023"]
}
```

### 3.3 qrels.jsonl

每一行表示一个 query 对应的 gold evidence chunks：

```json
{
  "query_id": "q_0001",
  "gold_chunk_ids": ["c_000123", "c_000124"]
}
```

### 3.4 retrieval result JSONL

每个实验输出的检索结果统一使用以下格式：

```json
{
  "query_id": "q_0001",
  "question": "...",
  "method": "BM25",
  "gold_chunk_ids": ["c_000123"],
  "top_k": [
    {
      "rank": 1,
      "chunk_id": "c_000123",
      "score": 12.34,
      "text": "...",
      "company": "Apple Inc.",
      "filing_year": "2023",
      "filing_type": "10-K",
      "section": "...",
      "metrics": ["operating income"],
      "years": ["2023"],
      "is_gold": true
    }
  ],
  "hit_at_5": true,
  "hit_at_10": true,
  "rr": 1.0,
  "ndcg_at_10": 1.0
}
```

---

# Exp1：普通检索 Baseline

## 4.1 实验目标

Exp1 是第一个正式实验，目标是回答：

> 不使用图结构，只用普通检索方法，在 FinDER 上能做到什么水平？

这个实验的意义是建立 baseline。后续所有图方法都必须和 Exp1 对比，否则无法证明 Financial Evidence Graph 是否真的有用。

---

## 4.2 需要比较的方法

### 方法 1：BM25

BM25 是关键词检索 baseline。

它依赖 query 和 passage 的词面匹配。例如问题里出现 `operating income`、`2023`、`Apple`，如果 passage 里也出现这些词，BM25 会给较高分数。

优点：

- 简单；
- 速度快；
- 对公司名、年份、财务指标这种显式关键词比较敏感。

缺点：

- 同义表达可能匹配不到；
- 不理解语义；
- 不理解金融文档结构；
- 容易找到“词很像但不能回答问题”的 passage。

### 方法 2：Dense Retrieval

Dense Retrieval 把 query 和 chunk 编码成向量，然后用 cosine similarity 或 dot product 检索。

优点：

- 能捕捉语义相似；
- 对措辞变化更鲁棒。

缺点：

- 可能忽略年份、公司、filing type 等硬约束；
- 可能把 `net income` 和 `operating income` 这种语义相近但财务含义不同的指标混淆；
- 可能把 2022 和 2023 的相似 passage 混淆。

推荐模型：

```text
优先：BAAI/bge-m3
备用：intfloat/e5-large-v2
如果算力不足：sentence-transformers/all-MiniLM-L6-v2
```

### 方法 3：Hybrid Retrieval

Hybrid Retrieval 融合 BM25 和 Dense Retrieval。

推荐公式：

```text
hybrid_score = alpha * normalized_bm25_score + (1 - alpha) * normalized_dense_score
```

初始设置：

```text
alpha = 0.5
```

可选调参：

```text
alpha ∈ {0.2, 0.4, 0.5, 0.6, 0.8}
```

### 方法 4：Cross-Encoder Reranker

Cross-Encoder 是强非图 reranker。

流程：

```text
query
  → Hybrid Retrieval top-50 / top-100
  → Cross-Encoder 对 query-chunk pair 打分
  → 重新排序
```

推荐模型：

```text
BAAI/bge-reranker-base
BAAI/bge-reranker-large
cross-encoder/ms-marco-MiniLM-L-6-v2
```

如果时间有限，Cross-Encoder 可以先预留接口，后面补做。

---

## 4.3 Exp1 输入

Exp1 需要读取：

```text
data/processed/queries.jsonl
data/processed/corpus_chunks.jsonl
data/processed/qrels.jsonl
```

其中：

- `queries.jsonl` 提供问题；
- `corpus_chunks.jsonl` 提供候选证据库；
- `qrels.jsonl` 提供 gold evidence；
- 所有方法都在同一个 corpus 上检索，保证公平。

---

## 4.4 Exp1 输出

输出目录：

```text
outputs/exp1_baseline/
├── bm25_results.jsonl
├── dense_results.jsonl
├── hybrid_results.jsonl
├── cross_encoder_results.jsonl          # 可选
├── metrics_summary.csv
├── failed_cases.jsonl
└── README.md
```

`metrics_summary.csv` 示例：

```csv
method,recall@5,recall@10,mrr,ndcg@10,hit@5,hit@10
BM25,,,,,,
Dense,,,,,,
Hybrid,,,,,,
Cross-Encoder,,,,,,
```

`failed_cases.jsonl` 保存至少 Hybrid Retrieval 失败的样本，供 Exp2 使用。

---

## 4.5 Exp1 评价指标

必须计算：

| 指标 | 含义 |
|---|---|
| Recall@5 | top-5 是否找回 gold evidence |
| Recall@10 | top-10 是否找回 gold evidence |
| Hit@K | top-k 中是否至少有一个 gold evidence |
| MRR | 第一个 gold evidence 排得是否靠前 |
| nDCG@10 | top-10 排序质量 |

### Recall@K / Hit@K

如果一个 query 有多个 gold evidence，只要 top-k 中出现至少一个，就算 Hit@K = 1。

```text
Recall@K = 命中 query 数量 / query 总数量
```

### MRR

如果第一个 gold evidence 出现在 rank = r：

```text
RR = 1 / r
```

如果 top-k 中没有 gold evidence：

```text
RR = 0
```

最终：

```text
MRR = 所有 query 的 RR 平均值
```

### nDCG@10

使用二值相关性：

```text
gold evidence: relevance = 1
non-gold evidence: relevance = 0
```

---

## 4.6 Exp1 需要回答的问题

实验完成后，Claude 需要在 `outputs/exp1_baseline/README.md` 中回答：

1. BM25、Dense、Hybrid 哪个最好？
2. Hybrid 是否明显优于单独 BM25 / Dense？
3. gold evidence 经常排在 top-10 内，还是经常排不进去？
4. 失败样本中，错误看起来更像语义匹配失败，还是金融结构约束失败？
5. 是否值得继续做 Financial Evidence Graph？

---

# Exp2：错误类型分析

## 5.1 实验目标

Exp2 的目标是分析 Exp1 为什么错。

核心问题是：

> 普通 RAG 的检索错误，是不是金融结构错误？

也就是说，失败不是简单因为“语义不够强”，而是因为普通检索没有理解金融文档中的结构约束，例如公司、年份、指标、filing、section、passage 是否匹配。

这个实验非常重要，因为它是提出 Financial Evidence Graph 的直接理由。

---

## 5.2 输入

Exp2 主要使用 Exp1 的输出：

```text
outputs/exp1_baseline/bm25_results.jsonl
outputs/exp1_baseline/dense_results.jsonl
outputs/exp1_baseline/hybrid_results.jsonl
outputs/exp1_baseline/failed_cases.jsonl
```

其中最重要的是失败样本：

```text
Hybrid Retrieval top-10 没有命中 gold evidence 的样本
或者
gold evidence 排名很靠后的样本
```

建议至少分析：

```text
100 个失败样本
```

如果数据集小，就分析所有失败样本。

---

## 5.3 错误类型定义

对每个失败样本，给它标注一个或多个错误类型。

### 1. Wrong Company

定义：检索结果找到了相似 passage，但公司不对。

例子：

```text
问题问 Apple 的 revenue，检索结果找到了 Microsoft 的 revenue passage。
```

判断规则：

- query 中能识别 company / ticker；
- top result 的 company / ticker 和 query 不一致；
- passage 内容看起来相关，但属于错误公司。

### 2. Wrong Year

定义：公司或指标可能对，但年份不对。

例子：

```text
问题问 2023，检索到 2022 的 passage。
```

判断规则：

- query 中有目标年份；
- top result 中年份与目标年份不一致；
- 或者 passage 中主要数字对应另一个 fiscal year。

### 3. Wrong Metric

定义：年份、公司可能对，但财务指标不对。

例子：

```text
问题问 operating income，检索到 net income。
```

判断规则：

- query 中有 target metric；
- top result 提到的是不同 metric；
- 该 metric 与目标 metric 语义相近但不能作为答案证据。

### 4. Wrong Filing

定义：检索结果来自错误 filing 或错误 filing type。

例子：

```text
问题需要 10-K 年报，检索到 10-Q 季报。
问题问 2023 filing，检索到 2022 filing。
```

判断规则：

- doc_id / filing_type / filing_year 与 query 需求不一致；
- 即使 passage 内容相似，也不能支持答案。

### 5. Wrong Section

定义：检索到的 section 类型不适合回答问题。

例子：

```text
问题需要 income statement table，检索到 risk factors 段落。
```

判断规则：

- top result 的 section 与 gold evidence 的 section 明显不同；
- top result 是相关背景，但不是答案来源。

### 6. Wrong Passage

定义：语义相似，但 passage 本身不能支持答案。

例子：

```text
问题问具体数值，检索到一段泛泛解释经营表现的文字，没有目标数字。
```

判断规则：

- company/year/metric 可能部分匹配；
- 但 passage 没有足够证据回答问题；
- 或者 passage 缺少关键数字、单位、表格行。

### 7. Missing Evidence

定义：top-k 完全没有找到正确证据。

例子：

```text
top-10 中没有任何 gold evidence，也没有接近正确的 passage。
```

判断规则：

- top-k 全部非 gold；
- 人工检查也没有一个能支持答案。

### 8. Unit Error

定义：检索到的证据单位与问题或答案不一致。

例子：

```text
问题需要 billion，passage 是 million。
问题需要 percentage，passage 是 dollar amount。
```

### 9. Arithmetic / Calculation Evidence Error

定义：问题需要计算，但检索结果只包含部分数字，缺少计算所需的另一个数字。

例子：

```text
问题问增长率，需要 2022 和 2023 两个数，但 top-k 只找到 2023。
```

---

## 5.4 自动标注规则

Claude 可以先实现一个半自动错误分析脚本。

建议函数：

```python
def classify_error(query, gold_chunks, retrieved_chunks):
    """
    Return one or more error labels.
    """
```

可使用以下启发式规则：

```text
if query.company exists and top_chunk.company != query.company:
    add Wrong Company

if query.target_year exists and query.target_year not in top_chunk.years:
    add Wrong Year

if query.target_metric exists and query.target_metric not in top_chunk.metrics:
    add Wrong Metric

if gold_chunk.filing_type != top_chunk.filing_type:
    add Wrong Filing

if gold_chunk.section != top_chunk.section:
    add Wrong Section

if no gold chunk in top_k:
    add Missing Evidence

if top_chunk text does not contain answer-like number:
    add Wrong Passage or Missing Evidence
```

注意：自动规则不可能完全准确，所以需要输出 `manual_check_required` 字段。

---

## 5.5 人工分析输出格式

输出目录：

```text
outputs/exp2_error_analysis/
├── error_cases_labeled.jsonl
├── error_type_summary.csv
├── error_type_by_method.csv
├── examples_wrong_year.md
├── examples_wrong_metric.md
├── examples_wrong_company.md
├── examples_wrong_passage.md
└── README.md
```

`error_cases_labeled.jsonl` 示例：

```json
{
  "query_id": "q_0001",
  "question": "What was Apple's operating income in 2023?",
  "method": "Hybrid",
  "gold_chunk_ids": ["c_gold"],
  "top_1_chunk_id": "c_wrong",
  "top_1_text": "...",
  "gold_text": "...",
  "error_types": ["Wrong Year", "Wrong Metric"],
  "reason": "The retrieved passage mentions Apple and income, but it refers to 2022 net income rather than 2023 operating income.",
  "manual_check_required": true
}
```

`error_type_summary.csv` 示例：

```csv
error_type,count,percentage
Wrong Year,,
Wrong Metric,,
Wrong Company,,
Wrong Filing,,
Wrong Passage,,
Missing Evidence,,
```

---

## 5.6 Exp2 需要回答的问题

在 `outputs/exp2_error_analysis/README.md` 中回答：

1. 最常见的错误类型是什么？
2. Wrong Year 和 Wrong Metric 是否占很大比例？
3. BM25、Dense、Hybrid 的错误类型是否不同？
4. Dense 是否更容易出现语义相似但结构错误？
5. BM25 是否更容易漏掉同义表达？
6. 这些错误能否通过 company / filing / year / metric 图关系缓解？
7. 哪些边类型最值得在 Financial Evidence Graph 中构建？

---

## 5.7 Exp2 的论文意义

Exp2 要证明：

```text
普通检索不是简单“语义不够强”，而是没有理解金融文档中的结构约束。
```

因此，下一步需要构建：

```text
Financial Evidence Graph
```

它显式连接：

```text
company - filing - section - passage - metric - year
```

---

# Exp3：Financial Evidence Graph + PPR

## 6.1 实验目标

Exp3 是第一个图实验，但不训练 GNN。

核心问题：

> 只使用图算法，不训练 GNN，Financial Evidence Graph 能不能改善 evidence ranking？

方法：

```text
Hybrid Retrieval 先召回 top-50 / top-100
然后在 Financial Evidence Graph 上使用 Personalized PageRank, PPR 重新排序
```

这个实验的意义：

1. 证明图结构本身是否有用；
2. 给 Exp4 的 GNN 方法提供图算法 baseline；
3. 提供可解释性，因为 PPR 可以解释证据为什么被推到前面。

---

## 6.2 Financial Evidence Graph 节点设计

构建异构图，最小版本必须包含：

| 节点类型 | 示例 | 是否必须 |
|---|---|---|
| Company | Apple Inc. | 必须 |
| Filing | AAPL_2023_10K | 必须 |
| Section | MD&A | 必须 |
| Chunk / Passage | chunk_0001 | 必须 |
| Metric | operating income | 必须 |
| Year | 2023 | 必须 |
| Query Entity | query_metric / query_year | 推荐 |

最小可行图：

```text
Company — Filing — Section — Chunk — Metric — Year
```

如果原始数据缺少 section 或 filing，可以先用已有字段代替，但要在 README 说明。

---

## 6.3 Financial Evidence Graph 边设计

必须构建以下边：

| 边类型 | 含义 | 作用 |
|---|---|---|
| company-has-filing | 公司拥有某份 filing | 防止 wrong company / wrong filing |
| filing-has-section | filing 包含 section | 保留文档层级 |
| section-has-chunk | section 包含 chunk | 定位 passage |
| chunk-mentions-metric | chunk 提到 metric | 防止 wrong metric |
| chunk-mentions-year | chunk 提到 year | 防止 wrong year |
| chunk-belongs-to-filing | chunk 属于 filing | 强化 filing 约束 |
| same-company | 两个 chunk 属于同公司 | 连接同公司证据 |
| same-year | 两个 chunk 提到同一年 | 连接年份证据 |
| same-metric | 两个 chunk 提到同指标 | 连接指标证据 |
| semantic-similar | 两个 chunk 语义相似 | 保留语义召回能力 |
| query-matches-company | query 匹配 company | PPR 种子边 |
| query-matches-year | query 匹配 year | PPR 种子边 |
| query-matches-metric | query 匹配 metric | PPR 种子边 |

### 重要说明

`semantic-similar` 边不是 Financial Evidence Graph 的唯一核心。真正重要的是：

```text
company / filing / year / metric 等金融结构边
```

因为本文要证明：

```text
金融结构边比普通语义相似边更能减少 wrong-year、wrong-metric、wrong-filing 错误。
```

---

## 6.4 边权重设计

建议先使用以下默认权重：

| 边类型 | 权重 |
|---|---:|
| query-matches-company | 1.0 |
| query-matches-metric | 1.0 |
| query-matches-year | 1.0 |
| chunk-mentions-metric | 0.8 |
| chunk-mentions-year | 0.8 |
| chunk-belongs-to-filing | 0.8 |
| company-has-filing | 0.7 |
| filing-has-section | 0.6 |
| section-has-chunk | 0.6 |
| same-metric | 0.5 |
| same-year | 0.5 |
| same-company | 0.5 |
| semantic-similar | 0.3 |

需要支持配置文件调整，例如：

```yaml
edge_weights:
  query_matches_company: 1.0
  query_matches_metric: 1.0
  query_matches_year: 1.0
  chunk_mentions_metric: 0.8
  chunk_mentions_year: 0.8
  semantic_similar: 0.3
```

---

## 6.5 Query Entity 抽取

PPR 需要 query seed nodes。因此要从 query 中抽取：

- company；
- ticker；
- year；
- filing type；
- financial metric。

第一版可以用规则抽取。

### Year 抽取

正则：

```text
\b(19|20)\d{2}\b
```

### Filing Type 抽取

关键词：

```text
10-K, 10K, 10-Q, 10Q, annual report, quarterly report
```

### Metric 抽取

第一版可以使用财务指标词表，例如：

```text
revenue
net income
operating income
operating expenses
gross profit
total assets
total liabilities
cash and cash equivalents
shareholders' equity
research and development
capital expenditures
free cash flow
earnings per share
EPS
```

需要注意同义词：

```text
sales ≈ revenue
net sales ≈ revenue
income from operations ≈ operating income
```

可以维护一个 `metric_aliases.json`。

---

## 6.6 PPR 重排方法

### Step 1：初始检索

使用 Exp1 中表现最好的 Hybrid Retrieval，召回：

```text
top_N = 50 或 100
```

### Step 2：构建 query-specific subgraph

对每个 query，抽取局部子图：

```text
- 初始检索 top_N chunks
- 这些 chunks 连接的 company / filing / section / metric / year nodes
- query 匹配到的 company / metric / year nodes
- 与候选 chunks 通过 same-metric / same-year / semantic-similar 相连的邻居
```

### Step 3：设置 PPR 种子节点

种子节点包括：

```text
1. query 匹配的 company node
2. query 匹配的 metric node
3. query 匹配的 year node
4. Hybrid Retrieval top chunks
```

种子权重建议：

```text
query company / metric / year seeds: 0.6
retrieval top chunks seeds: 0.4
```

也可以简化为所有 seed 均匀分布。

### Step 4：运行 Personalized PageRank

使用 networkx 或 scipy 实现。

参数建议：

```text
damping factor = 0.85
max_iter = 100
tol = 1e-6
```

### Step 5：得到 chunk 的 graph_score

PPR 会给图上每个节点一个分数，只取候选 chunk 节点的分数作为 `ppr_score`。

### Step 6：融合检索分数和图分数

推荐公式：

```text
score_ppr = alpha * normalized_hybrid_score
          + beta  * normalized_ppr_score
          + gamma * constraint_score
```

初始参数：

```text
alpha = 0.5
beta = 0.4
gamma = 0.1
```

`constraint_score` 可以按如下规则：

```text
+1 if chunk company matches query company
+1 if chunk year matches query year
+1 if chunk metric matches query metric
+1 if chunk filing type matches query filing type
```

然后归一化到 0 到 1。

---

## 6.7 Exp3 对比方法

必须比较：

| 方法 | 说明 |
|---|---|
| Hybrid Retrieval | 非图 baseline |
| Hybrid + Semantic Graph + PPR | 只用语义相似边 |
| Hybrid + Financial Graph + PPR | 只用 company / filing / year / metric 等金融结构边 |
| Hybrid + Full Graph + PPR | 金融结构边 + 语义边 |

这张对比非常重要，因为它回答：

> 是不是随便加一个图都有用？还是金融结构图真的有用？

预期论文结论应该是：

```text
Financial Graph + PPR 优于 Semantic Graph + PPR，说明金融结构边确实有贡献。
Full Graph + PPR 可能进一步提升，说明语义边和金融结构边互补。
```

---

## 6.8 Exp3 指标

和 Exp1 保持一致：

```text
Recall@5
Recall@10
MRR
nDCG@10
Hit@5
Hit@10
```

额外增加错误减少指标：

```text
Wrong-Year Error Rate
Wrong-Metric Error Rate
Wrong-Company Error Rate
Wrong-Filing Error Rate
Missing-Evidence Rate
```

这些错误指标来自 Exp2 的错误分类规则。

---

## 6.9 Exp3 输出

输出目录：

```text
outputs/exp3_feg_ppr/
├── graph_stats.json
├── ppr_results_financial_graph.jsonl
├── ppr_results_semantic_graph.jsonl
├── ppr_results_full_graph.jsonl
├── metrics_summary.csv
├── error_reduction_summary.csv
├── case_studies.md
└── README.md
```

`graph_stats.json` 示例：

```json
{
  "num_nodes": 123456,
  "num_edges": 789012,
  "node_types": {
    "company": 100,
    "filing": 500,
    "section": 2000,
    "chunk": 80000,
    "metric": 300,
    "year": 30
  },
  "edge_types": {
    "company-has-filing": 500,
    "filing-has-section": 2000,
    "section-has-chunk": 80000,
    "chunk-mentions-metric": 150000,
    "chunk-mentions-year": 120000,
    "semantic-similar": 200000
  }
}
```

---

## 6.10 Exp3 Case Study

至少输出 2–3 个案例。

### Case 1：Wrong Year 被修正

说明：

```text
Hybrid 检索把 2022 passage 排第一，但问题问 2023。
FEG + PPR 通过 query-year → year node → chunk-mentions-year 边，把 2023 passage 排到前面。
```

### Case 2：Wrong Metric 被修正

说明：

```text
Dense 检索把 net income passage 排前面，但问题问 operating income。
FEG + PPR 通过 metric node 区分两个财务指标。
```

### Case 3：Wrong Filing 被修正

说明：

```text
Hybrid 找到了同公司但错误 filing 的 passage。
FEG + PPR 通过 company-has-filing 和 chunk-belongs-to-filing 关系提升正确 filing 的 chunk。
```

---

## 6.11 Exp3 需要回答的问题

在 `outputs/exp3_feg_ppr/README.md` 中回答：

1. PPR 是否比 Hybrid Retrieval 提升 Recall@10？
2. PPR 是否提升 MRR，也就是 gold evidence 是否排得更靠前？
3. Financial Graph + PPR 是否优于 Semantic Graph + PPR？
4. Full Graph + PPR 是否最好？
5. PPR 主要减少了哪类错误：Wrong Year、Wrong Metric、Wrong Company，还是 Missing Evidence？
6. 哪些 case 可以证明金融结构边有用？

---

# Exp4：R-GCN / GraphSAGE Evidence Reranker

## 7.1 实验目标

Exp4 是 full paper 版本中最核心的模型实验。

核心问题：

> 训练一个图重排器，是否比 PPR 和普通 reranker 更强？

注意：

```text
这里不是发明新的 GNN 架构。
```

本文使用现有 GNN backbone，例如 GraphSAGE 或 R-GCN。真正的贡献是：

1. Financial Evidence Graph 的构建；
2. financial constraint-aware edge / feature 设计；
3. finance-specific hard negatives；
4. evidence reranking objective；
5. reliability-centered evaluation。

---

## 7.2 需要比较的方法

Exp4 至少比较：

| 方法 | 定位 |
|---|---|
| Hybrid Retrieval | 强检索 baseline |
| Hybrid + Cross-Encoder | 强非图 reranker |
| Hybrid + PPR | 图算法 baseline |
| Hybrid + GraphSAGE | 图学习 baseline |
| Hybrid + R-GCN | 主图重排模型 |
| FEG-Rerank | R-GCN + constraint-aware score 的完整方法 |

如果时间有限，最低限度比较：

```text
Hybrid
Hybrid + Cross-Encoder
Hybrid + PPR
Hybrid + GraphSAGE
Hybrid + R-GCN
```

---

## 7.3 GNN 的任务定义

GNN 不负责生成答案，只负责 evidence reranking。

输入：

```text
query q
candidate chunks C = {c1, c2, ..., cN}
query-specific Financial Evidence Subgraph Gq
```

输出：

```text
每个 candidate chunk 的相关性分数 score(q, c)
```

目标：

```text
gold evidence chunk 的分数 > hard negative chunk 的分数
```

---

## 7.4 训练数据构造

### 正样本

正样本是 FinDER 的 gold evidence chunks：

```text
positive chunks = gold_chunk_ids
```

### 负样本

负样本分三类：

#### 1. Random Negatives

从 corpus 中随机采样的非 gold chunks。

作用：

```text
提供简单负样本，但难度低。
```

#### 2. Top-Retrieved Negatives

从 Exp1 Hybrid Retrieval top-50 中取非 gold chunks。

作用：

```text
这些是普通检索认为相关但实际错误的候选。
```

#### 3. Finance-Specific Hard Negatives

这是本文最重要的负样本设计。

| Hard Negative 类型 | 说明 | 例子 |
|---|---|---|
| same company, wrong year | 公司对，年份错 | Apple 2022 vs Apple 2023 |
| same year, wrong company | 年份对，公司错 | Microsoft 2023 vs Apple 2023 |
| same metric, wrong filing | 指标对，filing 错 | 10-Q vs 10-K |
| same metric, wrong company | 指标对，公司错 | Microsoft operating income vs Apple operating income |
| same section, wrong passage | section 对，passage 不支持答案 | MD&A 中无关段落 |
| semantically similar, unsupported | 语义相似但没有答案证据 | 提到 profitability 但没有目标数字 |

训练时每个 query 推荐采样：

```text
1-3 个 positive chunks
5-20 个 negative chunks
其中至少一半是 finance-specific hard negatives
```

---

## 7.5 GNN 子图构造

对每个 query 构建 query-specific subgraph。

子图包含：

```text
1. Hybrid Retrieval top_N chunks
2. gold evidence chunks，用于训练阶段
3. candidate chunks 的 1-hop / 2-hop 邻居
4. query entity nodes: company / metric / year / filing type
5. candidate chunks 连接的 company / filing / section / metric / year nodes
```

参数建议：

```text
top_N = 50
hop = 2
max_nodes_per_subgraph = 2000
```

如果图太大，优先保留：

```text
query entity nodes
candidate chunk nodes
metric nodes
year nodes
filing nodes
```

---

## 7.6 节点特征设计

### Chunk 节点特征

每个 chunk 节点至少包含：

| 特征 | 类型 | 说明 |
|---|---|---|
| text_embedding | dense vector | chunk 文本向量 |
| bm25_score | scalar | Exp1 BM25 分数 |
| dense_score | scalar | Exp1 dense 分数 |
| hybrid_score | scalar | Exp1 hybrid 分数 |
| ppr_score | scalar | Exp3 PPR 分数，可选 |
| query_company_match | binary | 是否匹配 query company |
| query_year_match | binary | 是否匹配 query year |
| query_metric_match | binary | 是否匹配 query metric |
| query_filing_match | binary | 是否匹配 query filing type |
| node_type_embedding | embedding | 节点类型 embedding |

### 非 chunk 节点特征

Company / Filing / Section / Metric / Year 节点可以使用：

```text
node type embedding + textual name embedding
```

例如 metric 节点 `operating income` 可以用 metric name 的 embedding。

---

## 7.7 GraphSAGE Reranker

GraphSAGE 适合做第一个 GNN baseline，因为实现简单。

输入：

```text
homogeneous 或 simplified heterogeneous graph
```

如果不想处理多关系边，可以先把所有边类型合并，然后使用 GraphSAGE。

输出：

```text
chunk node embedding h_c
query embedding h_q
score(q, c) = MLP([h_q; h_c; h_q * h_c; features])
```

适合定位：

```text
图学习 baseline
```

---

## 7.8 R-GCN Reranker

R-GCN 更适合主模型，因为 Financial Evidence Graph 是多关系异构图。

边类型包括：

```text
company-has-filing
filing-has-section
section-has-chunk
chunk-mentions-metric
chunk-mentions-year
same-company
same-year
same-metric
semantic-similar
query-matches-company
query-matches-year
query-matches-metric
```

R-GCN 的优势是能学习不同边类型的不同作用。

模型输出：

```text
chunk node embedding h_c
```

最终得分：

```text
gnn_score = MLP([h_c, h_q, retrieval_scores, constraint_features])
```

---

## 7.9 训练损失

推荐使用 pairwise ranking loss。

对每个 query，取一个 positive chunk 和一个 negative chunk：

```text
score_pos = score(q, positive_chunk)
score_neg = score(q, negative_chunk)
```

损失：

```text
loss = max(0, margin - score_pos + score_neg)
```

推荐：

```text
margin = 1.0
```

也可以使用 BCE：

```text
gold chunk label = 1
negative chunk label = 0
```

但 pairwise loss 更符合 reranking 任务。

---

## 7.10 Constraint-Aware Fusion Score

完整方法 FEG-Rerank 使用融合分数：

```text
final_score = alpha * retrieval_score
            + beta  * ppr_score
            + gamma * gnn_score
            + delta * constraint_score
```

推荐初始参数：

```text
alpha = 0.3
beta = 0.2
gamma = 0.4
delta = 0.1
```

可在验证集上搜索。

`constraint_score` 计算：

```text
constraint_score =
  w_company * I(company match)
+ w_year    * I(year match)
+ w_metric  * I(metric match)
+ w_filing  * I(filing type match)
```

初始权重：

```text
w_company = 1
w_year = 1
w_metric = 1
w_filing = 1
```

---

## 7.11 Exp4 评价指标

检索指标：

```text
Recall@5
Recall@10
MRR
nDCG@10
Hit@5
Hit@10
```

错误类型指标：

```text
Wrong Company Error Rate
Wrong Filing Error Rate
Wrong Year Error Rate
Wrong Metric Error Rate
Wrong Passage Error Rate
Missing Evidence Rate
```

如果做 LLM 生成，还可以加：

```text
Answer Accuracy
Exact Match
Faithfulness
Numerical Consistency
Unsupported Answer Rate
```

但是 Exp4 首要任务仍然是 evidence reranking。

---

## 7.12 Exp4 输出

输出目录：

```text
outputs/exp4_gnn_reranker/
├── train_config.yaml
├── train_log.txt
├── graph_sage_results.jsonl
├── rgcn_results.jsonl
├── feg_rerank_results.jsonl
├── metrics_summary.csv
├── hard_negative_ablation.csv
├── edge_type_ablation.csv
├── model_checkpoints/
│   ├── graphsage.pt
│   └── rgcn.pt
├── case_studies.md
└── README.md
```

`metrics_summary.csv` 示例：

```csv
method,recall@5,recall@10,mrr,ndcg@10,wrong_year_rate,wrong_metric_rate,missing_evidence_rate
Hybrid,,,,,,,
Hybrid+CrossEncoder,,,,,,,
Hybrid+PPR,,,,,,,
Hybrid+GraphSAGE,,,,,,,
Hybrid+RGCN,,,,,,,
FEG-Rerank,,,,,,,
```

---

## 7.13 Exp4 消融实验

### 消融 1：重排模型对比

比较：

```text
Hybrid
Hybrid + PPR
Hybrid + GraphSAGE
Hybrid + R-GCN
Hybrid + R-GCN + Constraint Score
```

目的：

```text
证明训练式图重排器是否优于无训练 PPR。
```

### 消融 2：图结构消融

比较：

```text
No Graph
Semantic Edges Only
Financial Edges Only
Financial + Semantic Edges
Full Weighted Graph
```

目的：

```text
证明不是任意图都有效，而是金融结构边有效。
```

### 消融 3：边类型消融

比较：

```text
Full Graph
w/o Company Edges
w/o Filing Edges
w/o Section Edges
w/o Metric Edges
w/o Year Edges
w/o Semantic Edges
```

重点观察：

```text
去掉 year edges 后 Wrong-Year Error 是否上升。
去掉 metric edges 后 Wrong-Metric Error 是否上升。
```

### 消融 4：Hard Negative 消融

比较：

```text
Random Negatives
Top-Retrieved Negatives
Finance-Specific Hard Negatives
```

目的：

```text
证明金融 hard negatives 能让 reranker 学会区分“看起来相似但金融约束错误”的证据。
```

---

## 7.14 Exp4 Case Study

至少输出三个案例：

### Case 1：R-GCN 修正 Wrong Year

展示：

```text
Question
Gold evidence
Hybrid top-3
PPR top-3
R-GCN top-3
解释为什么 R-GCN 把正确年份 evidence 排前
```

### Case 2：R-GCN 修正 Wrong Metric

展示：

```text
问题问 operating income
Hybrid 排名前面的是 net income
R-GCN 通过 metric edge 找到 operating income
```

### Case 3：R-GCN 优于 PPR

展示：

```text
PPR 可以利用图结构，但不能学习哪些边在当前任务中更重要。
R-GCN 通过训练学会了 metric/year/company 的组合约束。
```

---

## 7.15 Exp4 需要回答的问题

在 `outputs/exp4_gnn_reranker/README.md` 中回答：

1. GraphSAGE / R-GCN 是否优于 Hybrid Retrieval？
2. R-GCN 是否优于 GraphSAGE？如果是，是否因为 R-GCN 能处理多关系边？
3. GNN 是否优于 PPR？在哪些错误类型上提升最大？
4. Finance-specific hard negatives 是否有效？
5. 去掉 year / metric edges 后性能是否下降？
6. 完整 FEG-Rerank 是否是最优方法？
7. 结果是否支持论文主张：金融结构图能提升 evidence reliability？

---

# 8. 四个实验之间的最终对比表

最终需要形成一个总表：

```text
outputs/final_summary/main_results.csv
```

表格结构：

| Method | Exp | Graph | Trainable | Recall@5 | Recall@10 | MRR | nDCG@10 | Wrong-Year ↓ | Wrong-Metric ↓ |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| BM25 | Exp1 | No | No | | | | | | |
| Dense | Exp1 | No | No | | | | | | |
| Hybrid | Exp1 | No | No | | | | | | |
| Cross-Encoder | Exp1/4 | No | Yes | | | | | | |
| Hybrid + Semantic Graph + PPR | Exp3 | Yes | No | | | | | | |
| Hybrid + Financial Graph + PPR | Exp3 | Yes | No | | | | | | |
| Hybrid + Full Graph + PPR | Exp3 | Yes | No | | | | | | |
| Hybrid + GraphSAGE | Exp4 | Yes | Yes | | | | | | |
| Hybrid + R-GCN | Exp4 | Yes | Yes | | | | | | |
| FEG-Rerank | Exp4 | Yes | Yes | | | | | | |

---

# 9. 推荐命令行接口

建议实现统一入口：

```bash
python -m src.main --exp exp1 --config configs/exp1.yaml
python -m src.main --exp exp2 --config configs/exp2.yaml
python -m src.main --exp exp3 --config configs/exp3.yaml
python -m src.main --exp exp4 --config configs/exp4.yaml
```

也可以拆成：

```bash
python scripts/run_exp1_baseline.py
python scripts/run_exp2_error_analysis.py
python scripts/run_exp3_feg_ppr.py
python scripts/run_exp4_gnn_reranker.py
```

每个脚本都必须：

1. 打印输入文件路径；
2. 打印样本数量；
3. 打印输出目录；
4. 保存 metrics；
5. 保存每个 query 的详细结果；
6. 遇到字段缺失时给出清楚报错。

---

# 10. 实现优先级

如果时间有限，按以下顺序实现：

## 第一优先级

```text
Exp1 BM25 / Dense / Hybrid
Exp1 metrics
Exp1 failed_cases
```

这是所有后续实验的基础。

## 第二优先级

```text
Exp2 错误类型分析
Wrong Year / Wrong Metric / Wrong Company / Missing Evidence 统计
```

这是论文动机的关键。

## 第三优先级

```text
Exp3 Financial Evidence Graph
Exp3 PPR reranking
Hybrid vs Financial Graph + PPR
```

这是证明图结构有用的最小方法实验。

## 第四优先级

```text
Exp4 GraphSAGE / R-GCN reranker
Hard negatives
Edge ablation
```

这是 full paper 版本的核心模型实验。

---

# 11. 最小可交付版本

如果只做最小可运行版本，必须完成：

```text
1. FinDER 数据加载
2. corpus chunk 构建
3. gold evidence mapping
4. BM25 / Dense / Hybrid retrieval
5. Recall@5 / Recall@10 / MRR / nDCG@10
6. 失败样本错误类型分析
7. Financial Evidence Graph 构建
8. PPR reranking
9. Hybrid vs Hybrid + Financial Graph + PPR 对比
```

最小版本可以暂时不训练 GNN，但必须保留 Exp4 的接口。

---

# 12. Full Paper 推荐版本

完整版本应该完成：

```text
1. Exp1: BM25 / Dense / Hybrid / Cross-Encoder
2. Exp2: 100-200 个失败样本错误类型分析
3. Exp3: Semantic Graph + PPR / Financial Graph + PPR / Full Graph + PPR
4. Exp4: GraphSAGE / R-GCN / FEG-Rerank
5. Hard negative ablation
6. Edge type ablation
7. Case studies
8. 如果有生成模块，再加入 LLM answer reliability evaluation
```

---

# 13. 论文中每个实验对应的作用

| 实验 | 论文作用 |
|---|---|
| Exp1 | 建立普通检索 baseline，说明现有方法水平 |
| Exp2 | 证明普通检索错误与金融结构约束有关 |
| Exp3 | 证明不训练 GNN，仅用金融证据图 + PPR 也能改善证据排序 |
| Exp4 | 证明训练式图重排器能进一步提升，并成为完整方法 |

---

# 14. 最终要证明的结论

实验最终应该支撑以下结论：

```text
1. 普通 BM25 / Dense / Hybrid retrieval 在金融 RAG 中会出现大量结构性错误。
2. 这些错误主要包括 wrong company、wrong year、wrong metric、wrong filing 和 wrong passage。
3. Financial Evidence Graph 能显式建模 company、filing、section、passage、metric、year 之间的关系。
4. PPR 可以在不训练模型的情况下利用图结构改善 evidence ranking。
5. GraphSAGE / R-GCN reranker 可以进一步学习不同金融关系的重要性。
6. 金融结构边和 finance-specific hard negatives 对减少 wrong-evidence error 很关键。
```

不要夸大成：

```text
我们解决了金融 RAG 幻觉问题。
我们提出了全新的 GNN 架构。
我们是第一个金融 GraphRAG。
```

更稳妥的论文主张是：

```text
Explicitly modeling financial evidence structure improves evidence reliability in retrieval-augmented financial question answering.
```

中文：

```text
显式建模金融证据结构可以提升金融 RAG 中的证据可靠性。
```

---

# 15. Claude 执行注意事项

Claude / coding agent 执行时必须注意：

1. 不要假设 FinDER 字段名固定，先 inspect 原始数据字段。
2. 不要一开始就训练 GNN，必须先完成 Exp1 baseline。
3. 所有实验必须保存 per-query 结果，不能只输出平均指标。
4. 所有指标计算必须基于同一份 qrels / gold evidence mapping。
5. 图方法必须和相同的 Hybrid Retrieval top-N 候选集比较，保证公平。
6. PPR 和 GNN 都是 reranking，不是重新从全库直接生成答案。
7. Exp2 的错误类型分析必须能追溯到具体 query 和 passage。
8. 如果某些 metadata 缺失，例如 company/year/metric，需要实现规则抽取或在 README 中说明限制。
9. 所有实验输出都要有 README，总结实验是否成功和下一步问题。
10. 代码要模块化，后续可以替换 embedding model、reranker model 和 graph edge weights。

---

# 16. 最后给 Claude 的一句话任务说明

请按照本文档实现 FinDER 上的四阶段实验：先完成 BM25/Dense/Hybrid 普通检索 baseline，并保存 per-query 检索结果和指标；然后对 baseline 的失败样本进行 wrong company、wrong year、wrong metric、wrong filing、wrong passage、missing evidence 等错误类型分析；接着构建 company–filing–section–chunk–metric–year Financial Evidence Graph，并用 Personalized PageRank 对 Hybrid top-N 候选证据进行重排；最后实现 GraphSAGE / R-GCN evidence reranker，使用 gold evidence 和 finance-specific hard negatives 训练模型，比较其与 Hybrid、Cross-Encoder、PPR 的 Recall@K、MRR、nDCG 和错误类型减少效果。
