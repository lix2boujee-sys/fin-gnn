# 实验设计：Financial Evidence Graphs for Reliable Financial RAG

## 0. 论文当前定位

本文不再定位为 short paper，而是按照 IEEE BigData full paper 的完整实验规模设计。

当前论文主线：

> 面向金融 RAG 问答任务，构建 Financial Evidence Graph，将公司、filing、section、passage、metric、year 等金融证据关系显式建模为异构图，并通过图传播与图神经网络重排提升证据检索可靠性，减少 wrong-year、wrong-metric、wrong-filing、wrong-passage 和 unsupported answer 等错误。

推荐题目：

**Financial Evidence Graphs for Reliable Retrieval-Augmented Question Answering over Financial Filings**

或者：

**Graph-Structured Evidence Reranking for Reliable Financial RAG**

---

## 1. 研究问题

本文实验围绕四个核心研究问题展开。

### RQ1：金融证据图是否能提升检索质量？

比较标准 BM25、Dense Retrieval、Hybrid Retrieval、Cross-Encoder Reranker 与图结构重排方法，观察 evidence recall、MRR、nDCG 是否提升。

### RQ2：图结构中的哪些关系最有用？

分析不同边类型的贡献，例如：

- company-filing relation；
- filing-section relation；
- section-passage relation；
- passage-metric relation；
- passage-year relation；
- same-metric relation；
- same-year relation；
- semantic-similarity relation。

### RQ3：更好的证据检索是否能提升最终答案可靠性？

在固定 LLM 的情况下，比较不同检索与重排方法对最终 answer correctness、faithfulness、numerical consistency 的影响。

### RQ4：图结构方法主要减少哪些错误？

重点分析：

- wrong company；
- wrong filing；
- wrong year；
- wrong metric；
- wrong document type；
- wrong passage；
- unsupported answer；
- arithmetic error；
- unit error。

---

## 2. 数据集设计

### 2.1 主数据集：FinDER

主数据集建议使用 **FinDER**。

理由：

- 2025 年左右提出，数据集较新；
- 原生面向 financial RAG evaluation；
- 包含 query-evidence-answer triplets；
- 比 TAT-QA / FinQA 更贴近真实金融检索场景；
- 适合做 evidence retrieval 和 answer grounding。

数据字段预期包括：

| 字段 | 用途 |
|---|---|
| `text` | 用户 query / question |
| `references` | gold evidence |
| `answer` | gold answer |
| `category` | 问题类别，用于分组实验 |
| `reasoning` | 推理类型或推理说明 |
| `type` | 问题类型 |

本文中 FinDER 的作用：

> 作为主评测集，用来评估金融证据图重排是否提升 evidence retrieval 与 answer reliability。

---

### 2.2 扩展语料：SEC EDGAR Filings

为了体现 BigData 的 Volume，并增加检索难度，建议加入 SEC EDGAR 10-K / 10-Q filings 作为扩展语料。

用途：

1. 增加大规模 distractor corpus；
2. 构建 filing-level、section-level、passage-level 图结构；
3. 构造 hard negatives；
4. 让任务更接近真实金融检索场景。

扩展语料构造建议：

| 类型 | 示例 |
|---|---|
| same company, wrong year | Apple 2022 filing vs Apple 2023 filing |
| same metric, wrong company | Microsoft revenue vs Apple revenue |
| same section, wrong filing | MD&A in another year |
| same document type, irrelevant passage | 10-K risk factor but unrelated |
| semantically similar but unsupported | 相似描述但没有支持答案的数字 |

---

### 2.3 补充数据集

如果时间允许，可以选择一个补充数据集。

优先级：

| 数据集 | 推荐程度 | 用途 |
|---|---|---|
| FinanceBench | 高 | 作为金融文档问答泛化测试 |
| FinAgentBench | 中高 | 如果公开可用，可测试 agentic retrieval 场景 |
| DocFinQA | 中 | 测试长文档金融推理 |
| TAT-QA / FinQA | 低到中 | 不作为主数据集，只作为传统金融 QA 参考 |

建议 full paper 至少使用：

> FinDER + SEC EDGAR 扩展语料 + 一个补充金融 QA/RAG 数据集。

---

## 3. 数据预处理流程

### 3.1 文档解析

输入金融 filing 文档后，需要解析出：

- company name；
- ticker / CIK；
- filing type；
- filing year；
- filing date；
- section title；
- passage text；
- table text；
- metric mention；
- year / period mention。

输出统一结构：

```json
{
  "doc_id": "AAPL_2023_10K",
  "company": "Apple Inc.",
  "ticker": "AAPL",
  "filing_type": "10-K",
  "filing_year": "2023",
  "section": "Management's Discussion and Analysis",
  "passage_id": "AAPL_2023_10K_MD&A_001",
  "text": "...",
  "metrics": ["net sales", "operating income"],
  "years": ["2022", "2023"]
}
```

---

### 3.2 Chunk 构造

将文档切分为 passage chunks。

建议切分策略：

| 类型 | 策略 |
|---|---|
| 普通文本 | 300–500 tokens，overlap 50 tokens |
| 表格 | 表格整体保留，同时行级展开 |
| 财务指标密集段落 | 尽量不切断 metric-year-value 关系 |
| section 标题 | 作为 metadata 附加到每个 chunk |

每个 chunk 保存：

- chunk_id；
- doc_id；
- company；
- filing year；
- filing type；
- section；
- text；
- extracted metrics；
- extracted years；
- embedding；
- BM25 index text。

---

### 3.3 Gold Evidence 对齐

FinDER 提供 references，需要将 references 映射到 corpus chunks。

对齐方式：

1. exact string matching；
2. fuzzy matching；
3. embedding similarity；
4. manual check for small sample。

输出：

```text
query_id → gold_chunk_ids
```

该映射用于计算 Evidence Recall@K、MRR、nDCG，也用于训练 GNN reranker。

---

## 4. Financial Evidence Graph 构建

### 4.1 图节点

构建异构图：

| 节点类型 | 示例 |
|---|---|
| Company | Apple Inc. |
| Filing | AAPL 2023 10-K |
| Filing Type | 10-K |
| Year | 2023 |
| Section | MD&A |
| Passage / Chunk | chunk_001 |
| Metric | net sales |
| Query Entity | extracted query metric/year/company |

最小可行版本：

> Company — Filing — Section — Chunk — Metric — Year

---

### 4.2 图边

| 边类型 | 含义 |
|---|---|
| company-has-filing | 公司拥有某份 filing |
| filing-has-section | filing 包含 section |
| section-has-chunk | section 包含 chunk |
| chunk-mentions-metric | chunk 提到某财务指标 |
| chunk-mentions-year | chunk 提到某年份 |
| chunk-belongs-to-filing | chunk 属于某 filing |
| same-company | 两个 chunk 属于同一公司 |
| same-filing-year | 两个 chunk 属于同一年份 |
| same-metric | 两个 chunk 提到同一指标 |
| semantic-similar | 两个 chunk 语义相似 |
| query-matches-metric | query 实体匹配 metric |
| query-matches-year | query 实体匹配 year |
| query-matches-company | query 实体匹配 company |

---

### 4.3 边权重设计

为了避免只是普通图建模，建议加入 financial constraint-aware edge weighting。

示例权重：

| 边类型 | 建议权重 |
|---|---:|
| query-matches-company | 1.0 |
| query-matches-metric | 1.0 |
| query-matches-year | 1.0 |
| chunk-mentions-metric | 0.8 |
| chunk-mentions-year | 0.8 |
| company-has-filing | 0.7 |
| filing-has-section | 0.6 |
| section-has-chunk | 0.6 |
| same-metric | 0.5 |
| same-year | 0.5 |
| semantic-similar | 0.3 |

可以在消融中比较：

- unweighted graph；
- semantic-only graph；
- financial-structure graph；
- full weighted graph。

---

## 5. 方法设计

### 5.1 整体 Pipeline

```text
Query
  → Initial Retrieval
  → Candidate Evidence Top-N
  → Financial Evidence Graph Subgraph Extraction
  → Graph-Based / GNN-Based Reranking
  → Top-K Evidence
  → LLM Answer Generation
  → Numerical / Evidence Verification
  → Final Answer
```

---

### 5.2 初始检索

先使用标准检索方法获得候选 evidence。

候选数量：

```text
Top-N = 50 或 100
```

对比方法：

| 方法 | 说明 |
|---|---|
| BM25 | sparse retrieval baseline |
| Dense Retrieval | dense embedding retrieval |
| Hybrid Retrieval | BM25 + Dense score fusion |
| Cross-Encoder Reranker | 强非图重排 baseline |

Dense retriever 建议：

| 模型 | 定位 |
|---|---|
| E5-Mistral | 强 dense retriever |
| BGE-M3 | 更轻量，部署方便 |
| e5-large-v2 | 经典 baseline |

建议主实验使用：

> BM25 + E5-Mistral Hybrid Retrieval

如果算力有限，用：

> BM25 + BGE-M3 Hybrid Retrieval

---

### 5.3 PPR 图传播重排

PPR 作为可解释图算法 baseline。

Seed nodes 包括：

- query 匹配的 company；
- query 匹配的 year；
- query 匹配的 metric；
- 初始检索 top chunks。

PPR 输出每个 chunk 的 graph_score。

最终 PPR 重排分数：

```text
score_ppr = α * retrieval_score + β * ppr_score + γ * constraint_score
```

其中：

- retrieval_score 来自 BM25 / dense / hybrid；
- ppr_score 来自 Financial Evidence Graph；
- constraint_score 来自 company/year/metric 匹配情况。

---

### 5.4 GNN Evidence Reranker

GraphSAGE / R-GCN 是现有 GNN backbone，不是本文原创架构。

本文的创新在于：

- 金融证据图构建；
- 金融约束感知边权重；
- financial hard negative；
- evidence reranking objective；
- reliability-centered evaluation。

#### 模型选择

| 模型 | 用途 |
|---|---|
| GraphSAGE | 简单、可扩展，适合 first GNN reranker |
| R-GCN | 处理多关系边，更适合异构图 |
| GAT | 可作为可选对比，不建议主推 |

建议主模型：

> R-GCN Reranker 或 GraphSAGE Reranker

如果 full paper 追求完整性，可以比较：

```text
PPR vs GraphSAGE vs R-GCN
```

---

### 5.5 GNN 输入特征

每个 chunk 节点的特征：

| 特征 | 来源 |
|---|---|
| text embedding | dense retriever embedding |
| BM25 score | initial retrieval |
| dense score | initial retrieval |
| node type embedding | chunk / metric / year / filing / section |
| query-company match | binary |
| query-year match | binary |
| query-metric match | binary |
| section type | MD&A / risk factor / financial statement |
| PPR score | optional feature |

---

### 5.6 GNN 训练目标

任务：

> evidence reranking

不是 answer generation。

正样本：

- gold evidence chunks。

负样本：

- initial retrieval top chunks 中非 gold evidence；
- hard negatives。

Hard negative 类型：

| 类型 | 说明 |
|---|---|
| same company, wrong year | 公司对，但年份错 |
| same year, wrong company | 年份对，但公司错 |
| same metric, wrong filing | 指标对，但 filing 错 |
| same section, wrong passage | section 对，但 passage 不支持答案 |
| semantically similar, unsupported | 语义相似但没有证据支持 |

训练损失建议：

1. Binary cross-entropy；
2. Pairwise ranking loss；
3. Listwise ranking loss。

推荐使用：

```text
Pairwise ranking loss
```

形式：

```text
score(q, positive) > score(q, negative)
```

---

### 5.7 最终重排分数

建议 full paper 使用 constraint-aware fusion score：

```text
final_score =
  α * retrieval_score
+ β * graph_score
+ γ * gnn_score
+ δ * constraint_score
```

其中：

- retrieval_score：BM25 / dense / hybrid；
- graph_score：PPR；
- gnn_score：GraphSAGE / R-GCN；
- constraint_score：公司、年份、指标、filing 类型是否匹配。

这个设计可以作为你的方法核心。

---

## 6. LLM 生成模块

### 6.1 推荐生成模型

主生成模型：

> Qwen2.5-7B-Instruct

备用生成模型：

> Llama-3.1-8B-Instruct

可选轻量模型：

> Mistral-7B-Instruct-v0.3

建议主实验固定使用：

```text
Qwen2.5-7B-Instruct
```

补充实验使用：

```text
Llama-3.1-8B-Instruct
```

目的：

> 证明检索和重排提升不是某个特定 LLM 的偶然结果。

---

### 6.2 生成 Prompt

Prompt 应强制模型基于证据回答。

示例：

```text
You are a financial question answering assistant.
Answer the question only using the provided evidence.
If the evidence is insufficient, answer "insufficient evidence".
Cite the evidence IDs used.
If numerical calculation is needed, show the calculation.
Keep the unit exactly as in the evidence.

Question:
{query}

Evidence:
{top_k_evidence}

Answer:
```

---

### 6.3 Numerical / Evidence Verifier

Verifier 可以是 rule-based，不需要训练新模型。

检查：

| 检查项 | 说明 |
|---|---|
| evidence support | 答案是否来自给定 evidence |
| year consistency | 答案年份是否与问题一致 |
| metric consistency | 指标是否一致 |
| unit consistency | million / billion / percentage 是否一致 |
| arithmetic consistency | 加减乘除是否正确 |
| unsupported generation | 是否出现证据中没有的信息 |

输出：

```text
verified / unsupported / numerically inconsistent
```

---

## 7. Baseline 设计

### 7.1 Retrieval Baselines

| Baseline | 说明 |
|---|---|
| BM25 | 关键词检索 |
| Dense Retrieval | 向量检索 |
| Hybrid Retrieval | BM25 + Dense |
| Hybrid + Cross-Encoder | 强非图 reranker |

---

### 7.2 Graph Baselines

| Baseline | 说明 |
|---|---|
| Semantic Graph + PPR | 只使用语义相似边 |
| Financial Graph + PPR | 使用金融结构边 |
| Full Graph + PPR | 金融结构边 + 语义边 |
| GraphSAGE Reranker | GNN backbone |
| R-GCN Reranker | 多关系 GNN backbone |

---

### 7.3 Full System

最终系统：

```text
Hybrid Retrieval
+ Financial Evidence Graph
+ R-GCN / GraphSAGE Reranker
+ Constraint-Aware Fusion
+ Qwen2.5-7B-Instruct
+ Numerical Verifier
```

可以命名为：

> FEG-RAG

或：

> FEG-Rerank

---

## 8. 评价指标

### 8.1 检索指标

| 指标 | 含义 |
|---|---|
| Recall@5 / Recall@10 | gold evidence 是否出现在 top-k |
| Precision@K | top-k 中相关证据比例 |
| MRR | gold evidence 排名是否靠前 |
| nDCG@K | 排序质量 |
| Hit@K | 是否命中至少一个 gold evidence |

---

### 8.2 生成指标

| 指标 | 含义 |
|---|---|
| Answer Accuracy | 最终答案是否正确 |
| Exact Match | 数字答案是否完全匹配 |
| F1 | 文本答案部分匹配 |
| Faithfulness | 回答是否被证据支持 |
| Unsupported Answer Rate | 无证据回答比例 |
| Numerical Consistency | 数字、单位、计算是否一致 |

---

### 8.3 错误类型指标

建议人工或半自动标注 100–200 个错误案例。

| 错误类型 | 说明 |
|---|---|
| Wrong Company | 找错公司 |
| Wrong Filing | 找错 filing |
| Wrong Year | 找错年份 |
| Wrong Metric | 找错指标 |
| Wrong Section | 找错章节 |
| Wrong Passage | 找到相似但无关 passage |
| Missing Evidence | 没有检索到支持性证据 |
| Unit Error | 单位错误 |
| Arithmetic Error | 计算错误 |
| Unsupported Generation | LLM 编造答案 |

---

## 9. 主实验表格设计

### Table 1：Retrieval Performance on FinDER

| Method | Recall@5 | Recall@10 | MRR | nDCG@10 |
|---|---:|---:|---:|---:|
| BM25 |  |  |  |  |
| Dense |  |  |  |  |
| Hybrid |  |  |  |  |
| Hybrid + Cross-Encoder |  |  |  |  |
| Hybrid + PPR |  |  |  |  |
| Hybrid + GraphSAGE |  |  |  |  |
| Hybrid + R-GCN |  |  |  |  |
| FEG-Rerank |  |  |  |  |

---

### Table 2：Answer Reliability with Fixed LLM

固定生成模型：

> Qwen2.5-7B-Instruct

| Retrieval Setting | Accuracy | EM | Faithfulness | Numerical Consistency | Unsupported Rate |
|---|---:|---:|---:|---:|---:|
| BM25-RAG |  |  |  |  |  |
| Dense-RAG |  |  |  |  |  |
| Hybrid-RAG |  |  |  |  |  |
| Hybrid + Cross-Encoder |  |  |  |  |  |
| FEG-PPR-RAG |  |  |  |  |  |
| FEG-GNN-RAG |  |  |  |  |  |
| FEG-GNN-RAG + Verifier |  |  |  |  |  |

---

### Table 3：Ablation on Graph Structure

| Graph Setting | Recall@10 | MRR | Accuracy | Wrong-Year Error | Wrong-Metric Error |
|---|---:|---:|---:|---:|---:|
| No Graph |  |  |  |  |  |
| Semantic Edges Only |  |  |  |  |  |
| Financial Edges Only |  |  |  |  |  |
| Financial + Semantic Edges |  |  |  |  |  |
| Full Weighted Graph |  |  |  |  |  |

---

### Table 4：Edge Type Ablation

| Setting | Recall@10 | MRR | nDCG@10 |
|---|---:|---:|---:|
| Full Graph |  |  |  |
| w/o Company Edges |  |  |  |
| w/o Filing Edges |  |  |  |
| w/o Section Edges |  |  |  |
| w/o Metric Edges |  |  |  |
| w/o Year Edges |  |  |  |
| w/o Semantic Edges |  |  |  |

---

### Table 5：Generator Robustness

| Generator | Retrieval Method | Accuracy | Faithfulness | Numerical Consistency |
|---|---|---:|---:|---:|
| Qwen2.5-7B-Instruct | Hybrid |  |  |  |
| Qwen2.5-7B-Instruct | FEG-Rerank |  |  |  |
| Llama-3.1-8B-Instruct | Hybrid |  |  |  |
| Llama-3.1-8B-Instruct | FEG-Rerank |  |  |  |

---

## 10. 消融实验设计

### 10.1 图结构消融

目的：

> 证明不是任意图都有用，而是金融结构关系有用。

比较：

```text
No Graph
Semantic Graph Only
Financial Structure Graph Only
Financial + Semantic Graph
Weighted Financial Evidence Graph
```

---

### 10.2 边类型消融

目的：

> 识别哪些金融关系最关键。

比较：

- remove company edges；
- remove filing edges；
- remove year edges；
- remove metric edges；
- remove section edges；
- remove semantic edges。

预期：

> year 和 metric edges 对 numerical QA 最关键。

---

### 10.3 重排模型消融

目的：

> 比较不同图重排方法。

比较：

```text
Hybrid
Hybrid + PPR
Hybrid + GraphSAGE
Hybrid + R-GCN
Hybrid + R-GCN + Constraint Score
```

---

### 10.4 Verifier 消融

目的：

> 证明 verifier 对减少 unsupported / numerical inconsistency 有作用。

比较：

```text
FEG-RAG
FEG-RAG + Verifier
```

---

### 10.5 Hard Negative 消融

目的：

> 证明金融 hard negatives 对 GNN reranker 训练有用。

比较：

```text
Random Negatives
Top-Retrieved Negatives
Finance-Specific Hard Negatives
```

Hard negatives 包括：

- same company wrong year；
- same year wrong company；
- same metric wrong filing；
- same section wrong passage。

---

## 11. Case Study 设计

至少展示 2–3 个案例。

### Case 1：Wrong Year

问题问 2023，但 Hybrid 检索到 2022。  
FEG-Rerank 通过 query-year 和 chunk-year 关系把 2023 evidence 排到前面。

### Case 2：Wrong Metric

问题问 operating income，但 dense retrieval 检索到 net income。  
FEG-Rerank 通过 metric node 区分相似财务指标。

### Case 3：Unsupported Answer

Hybrid-RAG 检索到相关但不支持答案的 passage，LLM 编造数字。  
FEG-RAG + Verifier 输出 insufficient evidence 或纠正答案。

---

## 12. 预期结论

论文应该保持保守但有说服力的结论。

可以期待的结论：

1. 金融证据图重排能提升 Evidence Recall@K 和 MRR。
2. R-GCN / GraphSAGE 在复杂关系图上通常优于纯 PPR，但 PPR 具备更强可解释性。
3. metric/year/company 等金融结构边对减少 wrong-evidence error 最关键。
4. 固定 LLM 时，检索质量提升能带来更高 answer accuracy 和 faithfulness。
5. Numerical verifier 能降低 unsupported answer 和 unit/year inconsistency。

避免夸大：

- 不说“首次提出金融 GraphRAG”；
- 不说“解决金融幻觉问题”；
- 不说“提出全新 GNN 架构”；
- 不说“全面超过所有金融 RAG 系统”。

---

## 13. 实现优先级

### Phase 1：最小可运行版本

- [ ] 下载 FinDER；
- [ ] 构建 corpus chunks；
- [ ] 建立 gold evidence mapping；
- [ ] 实现 BM25；
- [ ] 实现 dense retrieval；
- [ ] 实现 hybrid retrieval；
- [ ] 计算 Recall@K、MRR、nDCG；
- [ ] 构建 Company–Filing–Chunk–Metric–Year 图；
- [ ] 实现 PPR reranking；
- [ ] 比较 Hybrid vs Hybrid+PPR。

目标：

> 证明图结构有初步效果。

---

### Phase 2：Full Paper 方法版本

- [ ] 加入 SEC EDGAR 扩展语料；
- [ ] 构造 hard negatives；
- [ ] 实现 GraphSAGE 或 R-GCN reranker；
- [ ] 加入 constraint-aware fusion score；
- [ ] 加入 Qwen2.5-7B-Instruct 生成；
- [ ] 加入 numerical verifier；
- [ ] 完成主结果表。

目标：

> 形成完整方法。

---

### Phase 3：分析与补充实验

- [ ] 边类型消融；
- [ ] hard negative 消融；
- [ ] verifier 消融；
- [ ] generator robustness；
- [ ] 错误类型分析；
- [ ] case study；
- [ ] 补充数据集实验。

目标：

> 支撑 full paper 的完整性和说服力。

---

## 14. 论文实验部分写法建议

实验章节可以按如下顺序写：

### 4.1 Datasets and Corpus Construction

介绍 FinDER、SEC EDGAR 扩展语料、chunk 构造、gold evidence 对齐。

### 4.2 Baselines

介绍 BM25、Dense、Hybrid、Cross-Encoder、PPR、GraphSAGE、R-GCN。

### 4.3 Evaluation Metrics

分为 retrieval metrics、generation metrics、reliability metrics。

### 4.4 Main Results

展示主表格：

- retrieval performance；
- answer reliability。

### 4.5 Ablation Studies

展示：

- graph structure ablation；
- edge type ablation；
- reranker ablation；
- verifier ablation。

### 4.6 Error Analysis and Case Study

分析 wrong-year、wrong-metric、wrong-filing、unsupported answer。

---

## 15. 最终推荐实验配置

如果只能选一个最终版本，建议使用：

```text
Dataset:
  FinDER + SEC EDGAR distractor corpus

Retriever:
  BM25 + E5-Mistral Hybrid Retrieval

Graph:
  Company–Filing–Section–Passage–Metric–Year Financial Evidence Graph

Graph Methods:
  PPR baseline
  R-GCN reranker as main graph model

Generator:
  Qwen2.5-7B-Instruct
  Llama-3.1-8B-Instruct as robustness check

Verifier:
  Rule-based numerical and evidence verifier

Metrics:
  Recall@5/10, MRR, nDCG@10,
  Answer Accuracy, Faithfulness,
  Numerical Consistency,
  Unsupported Answer Rate,
  Wrong-Evidence Error Distribution
```

---

## 16. 最终一句话实验目标

> 本实验旨在验证：在近年金融 RAG benchmark 上，显式构建 company–filing–section–passage–metric–year 金融证据图，并通过 PPR / R-GCN 进行结构化证据重排，是否能相比标准 sparse/dense/hybrid retrieval 和非图 reranker 更可靠地找到正确证据，并提升金融问答的可解释性、数值一致性和答案可信度。
