# IEEE BigData Short Paper 中文论文方案

## 暂定题目

**Financial Evidence Graphs for Reliable Retrieval-Augmented Numerical Question Answering**

中文理解：

**面向金融报告数值问答的金融证据图增强 RAG 方法**

可选英文题目：

1. **Graph-Structured Evidence Reranking for Financial Report RAG**
2. **Graph-Enhanced Retrieval-Augmented Generation for Numerical Question Answering over Financial Reports**
3. **Financial Evidence Graphs: Improving RAG Reliability in Financial Report Question Answering**

推荐最终题目：

> **Graph-Structured Evidence Reranking for Financial Report RAG**

这个题目比“GraphRAG for Finance”更具体，重点落在 **证据重排 evidence reranking**，更适合 short paper。

---

## 1. 一句话主线

本文研究：在金融报告数值问答中，是否可以通过构建 **金融证据图 Financial Evidence Graph** 来提升 RAG 的可靠性，减少错误证据导致的错误答案，例如错公司、错年份、错指标、错表格，以及没有证据支撑的数值答案。

---

## 2. 核心研究问题

现有金融 RAG 系统通常依赖 BM25、向量检索或 hybrid retrieval 来召回文本块。但是金融问题往往不是普通语义匹配问题，而是同时包含多个结构化约束：

- 公司；
- 财年；
- 财务指标；
- 表格行 / 列；
- 报告章节；
- 数值运算方式。

核心研究问题是：

> **相比标准 BM25、dense retrieval 和 hybrid RAG，基于图结构的证据重排是否能提升金融数值问答中的证据检索质量和答案可靠性？**

本文不应该声称：

> 我们是第一个金融 RAG 系统。

也不应该声称：

> 我们是第一个 GraphRAG 系统。

更稳妥、清晰的贡献表述是：

> 本文聚焦于金融数值 RAG 中的 **证据重排问题**，通过显式建模 company-report-year-metric-table-chunk 之间的异构关系，提高检索证据的结构正确性。

---

## 3. 研究动机

金融报告问答的难点不是单纯让 LLM 生成答案，而是先要找到精确证据。很多错误答案来自错误证据，而不是语言模型本身。

例如，普通 RAG 可能检索到：

- 错误财年；
- 错误公司；
- 错误财务指标；
- 错误单位，例如 million 和 billion 混淆；
- 相关但并不能回答问题的表格；
- 只有文字解释、没有关键数值的段落；
- 语义相似但不是答案来源的 chunk。

标准 RAG 通常把所有文本块看成独立 chunk，这会丢失金融报告中的结构关系，比如：

- 某个 chunk 属于哪家公司；
- 属于哪一年财报；
- 对应哪个表格；
- 包含哪个财务指标；
- 该指标对应哪个年份；
- 表格数值和文字解释之间是什么关系。

因此，本文提出：

> 不把金融证据看成孤立文本块，而是看成一个由公司、报告、年份、指标、表格和文本块组成的异构证据图。

---

## 4. 为什么适合 IEEE BigData

这个主题符合 IEEE BigData 的 5V 特征：

| BigData 维度 | 本文对应方式 |
|---|---|
| Volume | 大规模金融报告语料、大量报告、大量 chunk、大量表格 |
| Variety | 文本、表格、数值、元数据、图结构 |
| Veracity | 证据可靠性、幻觉控制、数值一致性 |
| Value | 金融分析、财报理解、决策支持 |
| Velocity | 不是本文重点，但可以扩展到持续更新的 SEC filings |

最适合的投稿定位是：

> **Big Data Search and Mining + Foundation Models for Big Data + Finance Application + Data Veracity**

这比单纯“用 LLM 做金融问答”更符合 BigData 会议主题。

---

## 5. 预期数据集

### 5.1 主数据集：TAT-QA

主数据集建议使用 **TAT-QA**。

选择原因：

- 基于真实金融报告；
- 同时包含表格和文本；
- 需要数值推理；
- 提供问题、答案和支持证据；
- 很适合评估证据检索和最终答案正确性。

在本文中的作用：

> TAT-QA 提供核心 QA 样本和 gold evidence，用于训练和评估 retrieval / reranking。

---

### 5.2 辅助数据集：FinQA

第二个数据集可以使用 **FinQA**，作为泛化性或 robustness 检查。

选择原因：

- 包含金融报告 QA；
- 强调金融数值推理；
- 包含 supporting facts 和 reasoning programs；
- 可以验证方法是否不仅适用于 TAT-QA。

在本文中的作用：

> FinQA 用于测试图增强证据重排是否能泛化到另一个金融数值问答数据集。

---

### 5.3 可选长文档 / RAG 数据集：DocFinQA 或 FinDER

如果时间充足，可以加入：

- **DocFinQA**：把 FinQA 扩展到完整 financial report 场景；
- **FinDER**：专门用于金融 RAG 评测。

建议用法：

> 仅作为可选外部验证，不作为核心实验。因为 short paper 只有 6 页，数据集太多会导致论文主线分散。

---

### 5.4 扩展检索库：SEC EDGAR Filings

为了让任务更像真实 RAG，而不是“给定上下文问答”，可以从 **SEC EDGAR** 下载额外 10-K / 10-Q 财报作为干扰文档。

目的：

> 把 TAT-QA / FinQA 这种 context-given QA 改造成更真实的大规模金融文档检索任务。

构造方式：

1. 从 TAT-QA / FinQA 中获得原始问题、答案和 gold evidence；
2. 把原始 context 作为正例证据；
3. 从其他财报中加入大量 distractor chunks；
4. 加入 hard negatives，例如：
   - 同一公司但错误年份；
   - 同一指标但错误公司；
   - 同类型表格但错误报告；
   - 语义相似但没有答案的段落；
   - 同一报告章节但无关财务指标。

这是本文 BigData 属性最关键的一步。

---

## 6. 方法总览

方法名称可以叫：

> **FEG-RAG: Financial Evidence Graph RAG**

整体流程：

```text
Question
  → Initial Retrieval
  → Financial Evidence Graph Construction
  → Graph-based / GNN-based Evidence Reranking
  → Top-k Evidence Selection
  → LLM Answer Generation
  → Numerical and Evidence Consistency Verification
```

中文解释：

> 问题输入后，系统先用普通检索方法召回候选证据；然后基于金融证据图进行图算法或 GNN 重排；接着选择 top-k 证据交给 LLM 回答；最后用数值和证据一致性检查器验证答案是否可靠。

---

## 7. 金融证据图设计

### 7.1 节点类型

构建一个异构图，节点类型包括：

| 节点类型 | 示例 |
|---|---|
| Company | Apple Inc., Microsoft |
| Report | Apple 2023 10-K |
| Section | MD&A, Risk Factors, Financial Statements |
| Table | Consolidated Statements of Operations |
| Row / Metric | Revenue, Net income, Operating income |
| Year / Period | 2021, 2022, 2023 |
| Chunk | 文本 chunk 或表格 chunk |
| Question Entity | 从问题中抽取的 metric / year / company |

对于 short paper，最小可行图可以先只用：

> Report — Section — Chunk — Metric — Year

不要一开始就把图做得太复杂，否则实现成本会过高。

---

### 7.2 边类型

| 边类型 | 含义 |
|---|---|
| report-has-section | 一份报告包含某个章节 |
| section-has-chunk | 一个章节包含某个文本 / 表格 chunk |
| chunk-mentions-metric | 某个 chunk 提到一个财务指标 |
| chunk-mentions-year | 某个 chunk 提到某个年份或财年 |
| table-has-metric | 某个表格包含某个指标行 |
| metric-linked-to-year | 某个指标数值对应某一年份 |
| same-metric | 不同 chunk 提到相同指标 |
| same-year | 不同 chunk 提到相同年份 |
| semantic-similar | 两个 chunk 语义相似 |
| query-matches-metric | 问题实体匹配某个指标节点 |
| query-matches-year | 问题实体匹配某个年份节点 |

最重要的金融结构边是：

1. `chunk-mentions-metric`
2. `chunk-mentions-year`
3. `table-has-metric`
4. `same-metric`
5. `same-year`

这些边最有可能减少“错年份”和“错指标”导致的检索错误。

---

## 8. 检索与重排策略

### 8.1 初始检索

先用标准检索方法召回候选证据：

1. BM25 top-50；
2. Dense retrieval top-50；
3. Hybrid retrieval top-50。

Dense retriever 可以选择：

- sentence-transformers baseline；
- 金融领域 embedding model；
- 如果时间有限，也可以先用通用 embedding model。

---

### 8.2 图算法 baseline：Personalized PageRank

使用 **Personalized PageRank, PPR** 作为图算法 baseline。

种子节点包括：

- 问题中提到的 metric node；
- 问题中提到的 year node；
- 如果能识别，则包括 company / report node；
- 初始检索召回的 chunk nodes。

输出：

> 根据图距离、图传播分数和原始检索分数，对候选 evidence chunks 重新排序。

这个方法重要的原因：

- 实现简单；
- 可解释性强；
- 可以作为 GNN 之前的强图算法 baseline。

---

### 8.3 GNN 证据重排器

GNN 不直接生成答案，只负责 evidence reranking。

推荐模型：

- 第一版用 **GraphSAGE**；
- 如果边类型很重要，可以升级到 **R-GCN**；
- 不建议第一版直接用 HGT，因为实现复杂度较高。

输入：

- 初始检索得到的 candidate chunks；
- 它们在金融证据图中的局部邻域；
- 节点特征。

节点特征可以包括：

| 特征 | 说明 |
|---|---|
| Text embedding | chunk / table text 的向量 |
| BM25 score | 关键词检索得分 |
| Dense score | 向量检索得分 |
| Node type embedding | chunk、metric、year、table、section 等类型 |
| Query-metric match | 是否匹配问题中的指标 |
| Query-year match | 是否匹配问题中的年份 |
| Graph centrality | 可选，如 PageRank / degree |

训练目标：

> 二分类或 pairwise reranking：gold evidence chunks 应该排在 hard negative chunks 前面。

正样本：

- TAT-QA / FinQA 中的 gold supporting facts / gold evidence chunks。

负样本：

- top retrieved 但错误的 chunks；
- 同指标但错误年份；
- 同年份但错误指标；
- 同公司但无关表格；
- 语义相似但没有证据支持的段落。

最终重排分数可以写成：

```text
final_score = α * retrieval_score + β * graph_score + γ * gnn_score
```

论文中可以对这个融合策略做简单调参或消融。

---

## 9. 答案生成与验证

重排后，将 top-k evidence chunks 输入 LLM。

Prompt 需要约束：

1. 只能基于检索到的证据回答；
2. 必须引用 evidence chunk IDs；
3. 如果需要计算，必须展示数值运算过程；
4. 必须明确单位；
5. 如果证据不足，返回 `insufficient evidence`。

---

### 可选数值验证器

可以加入一个轻量 rule-based verifier，检查：

- 答案中的数值是否出现在证据中；
- 答案年份是否和问题匹配；
- 单位是否一致；
- 算术是否正确；
- 引用的 chunk 是否真的支持答案。

这个 verifier 不需要是深度模型，规则方法即可。

---

## 10. 主实验设计

主实验回答一个问题：

> 图结构重排是否能提升金融 RAG 的证据质量和答案可靠性？

### 10.1 对比方法

| 方法 | 说明 |
|---|---|
| No-RAG LLM | 不检索，直接回答 |
| BM25-RAG | 关键词检索 + LLM |
| Dense-RAG | 向量检索 + LLM |
| Hybrid-RAG | BM25 + dense retrieval + LLM |
| Hybrid + Cross-Encoder Reranker | 强非图重排 baseline |
| Graph-PPR-RAG | Hybrid retrieval + Personalized PageRank 图重排 |
| GNN-RAG | Hybrid retrieval + GNN 证据重排 |
| GNN-RAG + Verifier | GNN 重排 + 数值 / 证据一致性验证 |

对于 6 页 short paper，最重要的比较是：

```text
Hybrid-RAG vs Graph-PPR-RAG vs GNN-RAG vs GNN-RAG + Verifier
```

---

### 10.2 评估指标

| 指标 | 含义 |
|---|---|
| Answer Accuracy / Exact Match | 最终答案是否正确 |
| F1 | 对非完全匹配的文本答案有用 |
| Evidence Recall@K | gold evidence 是否进入 top-k |
| Evidence Precision@K | top-k evidence 中有多少是真正相关证据 |
| MRR / nDCG | 证据排序质量 |
| Numerical Consistency | 数字、单位、年份、运算是否一致 |
| Hallucination Rate | 答案是否包含无证据支持的内容 |
| Error Type Distribution | 错年份、错指标、错表格、错公司、算术错误等 |

本文最重要的指标是：

1. Evidence Recall@K；
2. Answer Accuracy；
3. Numerical Consistency；
4. Wrong-evidence error reduction。

---

## 11. 消融实验

消融实验保持简洁，避免把 short paper 写散。

### 消融 1：图结构是否有效

比较：

```text
Hybrid-RAG
Hybrid + semantic-similarity graph
Hybrid + financial structure graph
Hybrid + full evidence graph
```

目的：

> 证明金融结构边比普通语义相似边更有用。

---

### 消融 2：不同边类型的作用

| 设置 | 去掉的部分 |
|---|---|
| Full graph | 不去掉任何边 |
| w/o metric edges | 去掉 metric 相关边 |
| w/o year edges | 去掉 year 相关边 |
| w/o table edges | 去掉 table / chunk 结构边 |
| w/o semantic edges | 只保留金融结构边 |

目的：

> 分析哪些图关系最能提升证据可靠性。

---

### 消融 3：Verifier 是否有效

比较：

```text
GNN-RAG
GNN-RAG + Numerical Verifier
```

目的：

> 证明数值验证器是否能减少无证据答案和数值不一致错误。

---

## 12. 预期结果与可主张结论

结论要写得稳，不要过度夸大。

可以主张：

1. 图结构重排相比标准 hybrid retrieval 能提升 Evidence Recall@K；
2. GNN reranking 能更好地排序结构正确的证据，尤其是涉及年份和指标约束的问题；
3. 数值 verifier 可以减少无证据数值答案、单位错误和年份错误；
4. 对于表格-文本混合推理和多步数值推理问题，提升最明显。

避免主张：

- “第一个金融 GraphRAG 系统”；
- “解决了金融幻觉问题”；
- “全面超过所有金融 RAG 系统”。

更安全的结论是：

> 显式建模金融证据结构可以提升金融数值 RAG 的证据可靠性。

---

## 13. 错误分析计划

建议手动或半自动分析 50–100 个失败案例。

错误类型：

| 错误类型 | 示例 |
|---|---|
| Wrong year | 检索到 2021 而不是 2022 |
| Wrong metric | 检索到 revenue 而不是 operating income |
| Wrong table | 检索到 balance sheet 而不是 income statement |
| Wrong company/report | 检索到相似公司或错误 filing |
| Missing table evidence | 只检索到文字解释，没有检索到表格 |
| Arithmetic error | 数字检索正确，但计算错误 |
| Unit error | million / billion、百分比 / 绝对值混淆 |
| Unsupported generation | LLM 生成了没有证据支撑的答案 |

这个部分很重要，因为它能让论文不只是一个 leaderboard comparison，而是体现你对金融 RAG 失败模式的分析。

---

## 14. 主要创新点

### 创新点 1：把金融 QA 改造成大规模检索任务

很多金融 QA 数据集默认给定相关 context。本文通过加入更大的金融报告语料和 hard distractors，把任务改造成真实 RAG 场景。

贡献表述：

> We transform context-given financial numerical QA into a large-scale evidence retrieval and answering setting.

中文：

> 我们将给定上下文的金融数值问答改造成大规模证据检索与问答任务。

---

### 创新点 2：提出金融证据图

普通 RAG 把 chunk 看成彼此独立的文本块。本文把报告、章节、表格、指标、年份和 chunk 连接成异构图。

贡献表述：

> We introduce a financial evidence graph that captures structural constraints frequently required by financial questions.

中文：

> 我们提出金融证据图，用于建模金融问题中常见的结构约束。

---

### 创新点 3：图算法 / GNN 证据重排

GNN 用于证据重排，而不是答案生成，这让贡献更具体、更容易评估。

贡献表述：

> We propose graph-based and GNN-based evidence reranking methods to select structurally relevant evidence for RAG.

中文：

> 我们提出基于图算法和 GNN 的证据重排方法，用于为 RAG 选择结构上更相关的金融证据。

---

### 创新点 4：以可靠性为中心的评估

本文不只评估最终答案是否正确，还评估证据召回、数值一致性和错误证据类型。

贡献表述：

> We provide a reliability-centered evaluation of financial RAG, including evidence quality and numerical grounding.

中文：

> 我们提供以可靠性为中心的金融 RAG 评估，包括证据质量和数值 grounding。

---

## 15. 相关工作定位

### TAT-QA

TAT-QA 是基于真实金融报告的表格-文本混合问答 benchmark，需要在表格和文本证据上进行数值推理。

本文关系：

> 作为主要 QA 和 evidence benchmark。

---

### FinQA

FinQA 是金融数值推理数据集，包含专家编写的问题、支持事实和推理程序。

本文关系：

> 作为辅助评估或泛化性测试。

---

### DocFinQA

DocFinQA 将 FinQA 扩展到完整金融报告，表明长文档金融问答仍然具有挑战性。

本文关系：

> 支持本文动机，即金融 QA 应该从短 context 走向真实长文档检索。

---

### FinDER

FinDER 是专门用于金融 RAG 评测的数据集，包含 query-evidence-answer 三元组。

本文关系：

> 是重要相关工作，也可以作为可选外部验证。

---

### FinSage

FinSage 是金融 filings 的多方面 RAG 系统，使用多模态预处理、sparse-dense retrieval、query expansion、metadata-aware retrieval 和 DPO reranking。

本文关系：

> FinSage 是强相关金融 RAG 工作，但它的重点不是金融证据图和 GNN 结构化重排。

---

### GANO / 金融 QA 中的 GNN Evidence Extraction

GANO 类工作在给定 context 的 TAT-QA 上使用 GNN 做表格和文本证据抽取。

本文关系：

> 以往 GNN 金融 QA 通常假设相关 context 已经给定；本文研究的是更大 RAG 场景下的图增强证据检索和重排。

---

### Microsoft GraphRAG

Microsoft GraphRAG 构建图索引和 community summaries，用于大规模文档集合上的 query-focused summarization。

本文关系：

> 本文不是做全局总结，而是做金融数值问答中的局部精确证据检索。

---

### G-Retriever

G-Retriever 在带文本属性的图上做 RAG，通过图检索找到相关子图辅助回答。

本文关系：

> 本文借鉴图检索思想，但应用对象是金融证据结构和数值问答。

---

## 16. 6 页 Short Paper 建议结构

### 1. Introduction

内容：

- 金融报告问答需要精确证据；
- 标准 RAG 把 chunks 独立检索；
- 错误证据会导致错误数值答案；
- 提出金融证据图和图 / GNN 重排。

---

### 2. Related Work

简要覆盖：

- 金融数值 QA 数据集；
- 金融 RAG 系统；
- GraphRAG 和图检索；
- 表格 QA 中的 GNN evidence extraction。

---

### 3. Method

内容：

- 任务定义；
- 大规模检索设置构造；
- 金融证据图；
- PPR 和 GNN 重排；
- 答案生成和 verifier。

---

### 4. Experiments

内容：

- 数据集；
- baselines；
- metrics；
- main results；
- ablation。

---

### 5. Analysis

内容：

- 错误类型分析；
- case study：错年份 / 错指标 / 错表格。

---

### 6. Conclusion

内容：

- 图结构可以提升证据可靠性；
- 未来工作：更大 EDGAR corpus、多模态表格/图表、更复杂异构 GNN。

---

## 17. 最小可行版本

如果时间有限，先实现这个版本：

1. 只用 TAT-QA；
2. 从 table + text contexts 构建 chunks；
3. 从其他 TAT-QA examples 中加入 distractor chunks；
4. 图只包含 chunk、metric、year 三类节点；
5. 比较 Hybrid-RAG vs Graph-PPR-RAG；
6. PPR 做通后再加 GNN reranker；
7. 先评估 Evidence Recall@K 和 Answer Accuracy。

这个版本如果实验和分析清楚，已经可以支撑 short paper。

---

## 18. 更强版本

如果时间允许，再扩展：

1. 加入 SEC EDGAR 10-K / 10-Q filings 作为外部干扰文档；
2. 加入 FinQA 作为第二数据集；
3. 实现 GraphSAGE 或 R-GCN reranker；
4. 加入 numerical verifier；
5. 做详细错误分析。

这个版本对 IEEE BigData 更强。

---

## 19. 实现 Checklist

### 数据

- [ ] 下载 TAT-QA；
- [ ] 下载 FinQA；
- [ ] 可选：收集 SEC EDGAR filings；
- [ ] 将表格和文本转换成统一 chunks；
- [ ] 将 gold evidence 映射到 chunks；
- [ ] 生成 hard negatives。

### 检索

- [ ] 实现 BM25；
- [ ] 实现 dense retrieval；
- [ ] 实现 hybrid retrieval；
- [ ] 为每个问题保存 top-50 candidates。

### 图构建

- [ ] 抽取 metric mentions；
- [ ] 抽取 year mentions；
- [ ] 构建 chunk-metric-year graph；
- [ ] 可选：加入 section / table / report nodes；
- [ ] 加入 semantic similarity edges。

### 重排

- [ ] 实现 PPR reranking；
- [ ] 实现 GraphSAGE reranker；
- [ ] 使用 gold evidence 和 hard negatives 训练 reranker；
- [ ] 输出 top-k evidence。

### 答案生成

- [ ] 用 top-k evidence prompt LLM；
- [ ] 要求模型引用 chunk IDs；
- [ ] 实现 numerical verifier。

### 评估

- [ ] Evidence Recall@K；
- [ ] Evidence Precision@K；
- [ ] Answer Accuracy / EM；
- [ ] Numerical consistency；
- [ ] Hallucination / unsupported answer rate；
- [ ] Error type analysis。

---

## 20. 最终推荐表述

最终论文可以这样定位：

> Existing financial RAG methods mainly improve retrieval using sparse-dense search, query expansion, or generic reranking. However, financial numerical QA often fails because retrieved evidence violates structural constraints such as year, metric, table, or report identity. We propose a financial evidence graph and graph-based evidence reranking framework to explicitly model these constraints. Experiments on TAT-QA and FinQA-style retrieval settings show that graph-enhanced reranking improves evidence recall and reduces wrong-evidence errors, leading to more reliable numerical financial answers.

中文版本：

> 现有金融 RAG 方法主要通过稀疏-稠密检索、查询扩展或通用重排来提升检索效果。然而，金融数值问答经常失败的原因是检索到的证据违反了年份、指标、表格或报告身份等结构约束。本文提出金融证据图和基于图结构的证据重排框架，显式建模这些金融结构约束。基于 TAT-QA 和 FinQA 风格的大规模检索实验表明，图增强重排可以提升证据召回率，减少错误证据，从而提高金融数值问答的可靠性。

---

## 21. 参考文献与起点

1. TAT-QA: A Question Answering Benchmark on a Hybrid of Tabular and Textual Content in Finance  
   https://aclanthology.org/2021.acl-long.254/

2. TAT-QA project page  
   https://nextplusplus.github.io/TAT-QA/

3. FinQA: A Dataset of Numerical Reasoning over Financial Data  
   https://aclanthology.org/2021.emnlp-main.300/

4. FinQA project page  
   https://finqasite.github.io/

5. DocFinQA: A Long-Context Financial Reasoning Dataset  
   https://aclanthology.org/2024.acl-short.42/

6. FinDER: Financial Dataset for Question Answering and Evaluating Retrieval-Augmented Generation  
   https://arxiv.org/abs/2504.15800

7. FinSage: A Multi-aspect RAG System for Financial Filings Question Answering  
   https://arxiv.org/abs/2504.14493

8. Enhancing Financial Table and Text Question Answering with Tabular Graph and Numerical Reasoning  
   https://aclanthology.org/2022.aacl-main.72/

9. From Local to Global: A Graph RAG Approach to Query-Focused Summarization  
   https://arxiv.org/abs/2404.16130

10. G-Retriever: Retrieval-Augmented Generation for Textual Graph Understanding and Question Answering  
   https://arxiv.org/abs/2402.07630

11. SEC EDGAR APIs  
   https://www.sec.gov/search-filings/edgar-application-programming-interfaces
