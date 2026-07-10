# FinDER 实验说明：实验 1 普通检索 Baseline

> 用途：把这份 Markdown 交给 Claude / coding agent，让它根据文字说明完成实验。  
> 当前优先执行的是 **实验 1：普通检索 baseline**。后面的实验 2、3、4 只作为整体研究路线背景，先不要直接实现。

---

## 0. 项目背景

本项目目标是研究金融文档问答 / 证据检索中的 evidence ranking 问题。数据集是 **FinDER**，任务是：

给定一个问题 query，从金融文档的候选文本片段 passages / evidence chunks 中找出能够支持答案的 **gold evidence**，并评估检索系统能否把正确证据排在前面。

后续项目会引入 **Financial Evidence Graph**、PPR、GraphSAGE、R-GCN 等图方法。但实验 1 只做普通检索 baseline，不使用任何图结构。

---

## 1. 实验 1：普通检索 Baseline

### 1.1 实验目标

回答下面这个问题：

> 不用图结构，只用普通检索方法，在 FinDER 上能做到什么水平？

也就是说，先建立一个没有图结构的检索基线，用来衡量后续图方法是否真的带来提升。

---

## 2. 需要比较的方法

实验 1 需要比较以下方法：

| 方法 | 作用 | 是否必须实现 |
|---|---|---|
| BM25 | 关键词检索 baseline | 必须 |
| Dense Retrieval | 向量检索 baseline | 必须 |
| Hybrid Retrieval | BM25 + Dense 融合 | 必须 |
| Cross-Encoder Reranker | 强非图重排 baseline | 可后续做，实验 1 可以先预留接口 |

### 2.1 BM25

BM25 是关键词检索方法，主要依赖 query 和 passage 的词面匹配。

它适合作为最基础的 lexical baseline。金融任务中，如果问题里的公司名、年份、指标名和 passage 完全匹配，BM25 可能表现不错。

### 2.2 Dense Retrieval

Dense Retrieval 是向量检索方法。它需要把 query 和 passage 分别编码成向量，然后计算相似度，例如 cosine similarity 或 dot product。

它适合捕捉语义相似性，但可能会忽略金融任务中的严格约束，例如公司、年份、财报类型、指标名称等。

### 2.3 Hybrid Retrieval

Hybrid Retrieval 是 BM25 和 Dense Retrieval 的融合方法。

推荐做法：

1. 分别计算 BM25 score 和 dense score。
2. 对两个 score 做归一化。
3. 用加权方式融合：

```text
hybrid_score = alpha * bm25_score + (1 - alpha) * dense_score
```

其中 `alpha` 可以先设为 0.5，也可以在验证集上搜索，例如：

```text
alpha ∈ {0.2, 0.4, 0.5, 0.6, 0.8}
```

### 2.4 Cross-Encoder Reranker

Cross-Encoder Reranker 是强非图重排方法。它不是图方法，但通常比单纯 BM25 或 Dense 更强。

基本流程：

1. 先用 Hybrid Retrieval 召回 top-50 或 top-100 候选证据。
2. 用 Cross-Encoder 对每个 query-passage pair 打分。
3. 按 Cross-Encoder score 重新排序。

实验 1 阶段可以先预留这个模块，不一定马上实现。

---

## 3. 实验输入和输出

### 3.1 输入

需要从 FinDER 数据集中读取以下信息：

| 字段 | 含义 |
|---|---|
| query / question | 用户问题 |
| passages / evidence candidates | 候选证据文本片段 |
| gold evidence | 标注的正确证据 |
| company | 公司信息，如果数据集中有 |
| year | 年份信息，如果数据集中有 |
| filing type | 财报类型，例如 10-K / 10-Q，如果数据集中有 |
| metric | 金融指标，例如 operating income / net income，如果数据集中有 |

如果当前数据文件字段名不同，请在代码中做适配，并在 README 或日志中说明字段映射关系。

### 3.2 输出

实验 1 至少需要输出以下结果：

1. 每种方法的整体指标表。
2. 每个 query 的 top-k 检索结果。
3. 每个 query 的 gold evidence 是否被命中。
4. 后续错误分析需要用到的失败样本。

建议输出文件：

```text
outputs/exp1_baseline/
├── bm25_results.jsonl
├── dense_results.jsonl
├── hybrid_results.jsonl
├── metrics_summary.csv
├── failed_cases.jsonl
└── README.md
```

每条检索结果建议保存为 JSONL，格式示例：

```json
{
  "query_id": "xxx",
  "question": "...",
  "method": "BM25",
  "top_k": [
    {
      "rank": 1,
      "passage_id": "p1",
      "score": 12.34,
      "text": "...",
      "is_gold": true
    }
  ],
  "gold_evidence_ids": ["p1"],
  "hit_at_5": true,
  "hit_at_10": true,
  "rr": 1.0
}
```

---

## 4. 主要评价指标

实验 1 需要重点计算以下指标：

| 指标 | 含义 |
|---|---|
| Recall@5 | gold evidence 是否出现在 top-5 中 |
| Recall@10 | gold evidence 是否出现在 top-10 中 |
| MRR | gold evidence 排得靠不靠前 |
| nDCG@10 | top-10 的排序质量 |
| Hit@K | top-k 中有没有命中正确证据 |

### 4.1 Recall@K / Hit@K

如果一个问题有一个或多个 gold evidence，只要 top-k 中出现至少一个 gold evidence，就算命中。

```text
Recall@K = 命中 gold evidence 的 query 数量 / query 总数量
```

在本实验中，Hit@K 和 Recall@K 可以先按相同方式计算。

### 4.2 MRR

MRR 用来衡量第一个正确证据排在多靠前的位置。

如果第一个 gold evidence 出现在 rank = r：

```text
RR = 1 / r
```

如果 top-k 中没有找到 gold evidence：

```text
RR = 0
```

最终：

```text
MRR = 所有 query 的 RR 平均值
```

### 4.3 nDCG@10

nDCG@10 用来衡量排序质量。对于当前实验，可以先用二值相关性：

```text
gold evidence: relevance = 1
non-gold evidence: relevance = 0
```

如果 gold evidence 越靠前，nDCG@10 越高。

---

## 5. 实验 2 背景：错误类型分析

实验 1 完成后，需要做实验 2：分析普通检索为什么失败。

实验 2 的核心问题是：

> 普通 RAG 的检索错误是不是金融结构错误？

需要抽样分析 BM25 / Dense / Hybrid 的失败案例，并将错误分成以下类型：

| 错误类型 | 例子 |
|---|---|
| Wrong Company | 找到相似公司的 passage |
| Wrong Year | 问 2023，找到 2022 |
| Wrong Metric | 问 operating income，找到 net income |
| Wrong Filing | 找错 10-K / 10-Q |
| Wrong Passage | 语义相似但不支持答案 |
| Missing Evidence | 完全没找回正确证据 |

这一步很重要，因为它要证明：

> 普通检索的问题不只是“语义不够强”，而是没有理解金融文档中的结构约束。

这些结构约束包括：

```text
company / year / metric / filing type / passage relation
```

这正是后续构建 Financial Evidence Graph 的理由。

---

## 6. 后续实验路线背景

以下内容是后续实验路线，不属于当前实验 1 的立即实现范围。

### 6.1 实验 3：Financial Evidence Graph + PPR

目标问题：

> 只用图算法，不训练 GNN，金融证据图能不能改善 evidence ranking？

方法：

```text
Hybrid Retrieval 先召回 top-50，然后在 Financial Evidence Graph 上用 PPR 重排。
```

需要比较：

| 方法 | 作用 |
|---|---|
| Hybrid Retrieval | 非图 baseline |
| Hybrid + Semantic Graph + PPR | 只用语义相似边 |
| Hybrid + Financial Graph + PPR | 用 company / filing / year / metric 等结构边 |
| Hybrid + Full Graph + PPR | 金融结构边 + 语义边 |

重点指标：

| 指标 | 目的 |
|---|---|
| Recall@10 | 图传播是否找回更多 gold evidence |
| MRR | gold evidence 是否排得更靠前 |
| Wrong-Year Error | 是否减少错年份 |
| Wrong-Metric Error | 是否减少错指标 |

### 6.2 实验 4：R-GCN / GraphSAGE Evidence Reranker

目标问题：

> 训练一个图重排器，是否比 PPR 和普通 reranker 更强？

这里不是发明新的 GNN，而是使用现有图模型做 evidence reranking。

需要比较：

| 方法 | 定位 |
|---|---|
| Hybrid Retrieval | 强检索 baseline |
| Hybrid + Cross-Encoder | 强非图 reranker |
| Hybrid + PPR | 图算法 baseline |
| Hybrid + GraphSAGE | 图学习 baseline |
| Hybrid + R-GCN | 主图重排模型 |
| FEG-Rerank | R-GCN + constraint-aware score 的完整方法 |

训练目标：

> gold evidence 的得分高于 hard negative evidence。

hard negatives 要专门设计成金融错误，例如：

| Hard Negative | 例子 |
|---|---|
| same company, wrong year | 公司对，年份错 |
| same metric, wrong company | 指标对，公司错 |
| same company, wrong metric | 公司对，指标错 |
| same company, same year, wrong filing | 公司和年份对，但财报类型错 |
| semantic similar but unsupported | 语义相似，但不支持答案 |

---

## 7. 当前 Claude 需要完成的任务

请先只完成 **实验 1：普通检索 baseline**。

具体任务：

1. 检查当前项目目录和 FinDER 数据文件格式。
2. 找到 query、candidate passages、gold evidence 的字段。
3. 实现 BM25 检索。
4. 实现 Dense Retrieval。
5. 实现 Hybrid Retrieval。
6. 计算 Recall@5、Recall@10、MRR、nDCG@10、Hit@K。
7. 保存每种方法的 top-k 检索结果。
8. 保存失败案例，供实验 2 做错误类型分析。
9. 在输出目录生成 metrics_summary.csv。
10. 写一个简短 README，说明如何运行实验和结果文件含义。

---

## 8. 推荐命令接口

可以设计如下命令：

```bash
python -m src.experiments.exp1_retrieval_baseline \
  --data_dir data/FinDER \
  --output_dir outputs/exp1_baseline \
  --top_k 10 \
  --dense_model sentence-transformers/all-MiniLM-L6-v2 \
  --alpha 0.5
```

如果项目结构不同，可以根据实际情况调整，但需要保证 README 中写清楚运行方法。

---

## 9. 推荐代码模块

建议拆成以下模块：

```text
src/
├── experiments/
│   └── exp1_retrieval_baseline.py
├── retrieval/
│   ├── bm25.py
│   ├── dense.py
│   └── hybrid.py
├── evaluation/
│   └── retrieval_metrics.py
└── utils/
    └── finder_loader.py
```

如果当前项目已有类似模块，请优先复用现有结构，不要重复造轮子。

---

## 10. 实验完成标准

实验 1 完成时，至少应该能够得到以下内容：

### 10.1 指标汇总表

示例格式：

| Method | Recall@5 | Recall@10 | MRR | nDCG@10 | Hit@5 | Hit@10 |
|---|---:|---:|---:|---:|---:|---:|
| BM25 | 待实验生成 | 待实验生成 | 待实验生成 | 待实验生成 | 待实验生成 | 待实验生成 |
| Dense Retrieval | 待实验生成 | 待实验生成 | 待实验生成 | 待实验生成 | 待实验生成 | 待实验生成 |
| Hybrid Retrieval | 待实验生成 | 待实验生成 | 待实验生成 | 待实验生成 | 待实验生成 | 待实验生成 |

### 10.2 失败案例文件

`failed_cases.jsonl` 中应该包含：

```json
{
  "query_id": "xxx",
  "question": "...",
  "method": "Hybrid Retrieval",
  "gold_evidence_ids": ["..."],
  "top_k_results": [
    {
      "rank": 1,
      "passage_id": "...",
      "score": 0.87,
      "text": "...",
      "company": "...",
      "year": "...",
      "metric": "...",
      "filing": "..."
    }
  ],
  "possible_error_type": "to_be_annotated"
}
```

### 10.3 README

README 至少说明：

1. 运行命令。
2. 使用了哪些数据文件。
3. 每种方法的含义。
4. 每个输出文件的含义。
5. 当前实验结论，例如哪个 baseline 最强，哪些指标较低。

---

## 11. 注意事项

1. 实验 1 不要使用图结构。
2. 实验 1 的目标不是追求最强效果，而是建立可信 baseline。
3. 必须保存 per-query 检索结果，否则后续无法做错误类型分析。
4. 如果 gold evidence 有多个，只要 top-k 命中任意一个，就算 Hit@K / Recall@K 命中。
5. Dense Retrieval 的 embedding 可以先缓存，避免每次重复计算。
6. Hybrid score 必须做归一化后再融合，避免 BM25 score 和 dense score 数值尺度不一致。
7. 如果数据字段不明确，先写 loader 适配，并在 README 中记录字段解释。

---

## 12. 最终要回答的问题

实验完成后，需要用结果回答：

> 不用图结构，普通检索在 FinDER 上能做到什么水平？

并为后续实验提供对比基础：

1. 图方法是否能提升 Recall@10？
2. 图方法是否能提升 MRR？
3. 图方法是否能减少 Wrong Company / Wrong Year / Wrong Metric / Wrong Filing 等金融结构错误？

