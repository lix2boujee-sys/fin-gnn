# QCE-Graph Lite 工程实现需求

> 目标读者：Claude / 代码实现代理  
> 项目：`fin-gnn`  
> 新方法名：`QCE-Graph Lite`  
> 英文全称：`Query-Conditioned Counterfactual Evidence Graph Reranker`  
> 中文名：查询条件化反事实金融证据图轻量重排器

---

## 0. 实现总原则

请先阅读当前仓库，不要凭旧文件名猜实现。重点阅读：

- `feg_rag/data/chunker.py`
- `feg_rag/graph/entities.py`
- `feg_rag/graph/builder.py`
- `feg_rag/rerank/query_features.py`
- `feg_rag/rerank/scoring.py`
- `feg_rag/rerank/rgcn.py`
- `feg_rag/rerank/train.py`
- `experiments/table1_non_llm_reranking_comparison.py`
- 当前已有的 Table 1 / held-out 评估脚本和输出格式

硬性要求：

1. 不修改 vanilla R-GCN baseline 的算法、checkpoint、结果文件。
2. 新模型可以选择读取已有 R-GCN 分数作为可选输入，但不能把“重新训练/调参 R-GCN”当成新贡献。
3. 不允许把 gold evidence 人工插入候选池。
4. 训练阶段可以用 gold 计算 loss；验证/测试阶段严禁用 gold 决定扩展、候选、路径、权重或最终分数。
5. 复用当前 held-out split；如果仓库已有 split helper 或 manifest，必须直接复用。
6. 新代码必须支持 smoke test、固定随机种子、缓存、断点续跑、进度日志。
7. 不引入大型 Transformer、大型 GNN、LLM API 或在线外部服务。
8. 不增加非必要依赖，优先使用已有 `numpy`、`torch`、`networkx`。
9. 不写“first-ever”等未经文献证明的全球首创表述。
10. 不伪造结果；README 中的结果必须由脚本真实生成。

---

## 1. 研究目标

当前 R-GCN 是强 passage-level reranker。QCE-Graph Lite 的目标不是替换 R-GCN，而是在固定预算内验证：

1. 查询是否能动态选择适合的金融图关系；
2. 图关系是否能在初始 top-N 外补回遗漏证据；
3. support/conflict 双通道是否能稳定提高金融证据排序质量；
4. 模型是否能在没有 R-GCN 分数时独立运行，在有 R-GCN 分数时作为可选增强。

主评估指标：

- MRR
- Recall@5
- Recall@10
- nDCG@10

必须额外输出候选扩展诊断：

- expansion 前 candidate recall；
- expansion 后 candidate recall；
- 新找回 gold 的 query 数；
- 平均扩展候选数；
- 每种关系的使用频率；
- 每种关系的 gold recovery 频率。

注意：不要承诺一定超过 R-GCN。实现目标是形成一个可验证、可消融、可解释的轻量新模型。

---

## 2. 与现有方法的区别

### 2.1 与 R-GCN 的区别

R-GCN：

```text
固定候选集 -> 多关系图消息传递 -> 重排已有候选
```

QCE-Graph Lite：

```text
初始候选
-> query-conditioned relation routing
-> budgeted graph candidate expansion
-> support/conflict 双通道打分
-> 轻量 reranking
```

### 2.2 与 Fast Final Graph 的区别

Fast Final Graph 主要融合已有分数，例如 BGE、R-GCN、MonoT5、PPR。  
QCE-Graph Lite 不只是最后融合分数，而是在重排前改变候选信息流：

1. Query 决定使用哪些关系；
2. 图关系用于补充候选；
3. support 和 conflict 分开建模。

### 2.3 与 FinPath / FinMUSE 原型的区别

旧路径模型偏向对已有候选做路径特征修正。  
QCE-Graph Lite 的重点是：

- 把图作用前移到候选扩展；
- 显式限制扩展预算；
- 显式惩罚 wrong-company、wrong-year、wrong-metric；
- 用轻量可训练 scorer，而不是纯专家打分。

---

## 3. 分阶段实现策略

必须按阶段实现，每阶段先通过测试再进入下一阶段。

### Phase 1：Graph Expansion 诊断，不训练模型

先实现：

- `GraphExpansionIndex`
- `BudgetedGraphExpander`
- 固定关系预算扩展
- expansion before/after candidate recall

先回答：

```text
图扩展是否真的找回了新的 gold evidence？
```

如果 expansion 后 candidate recall 完全没有提升，先停止，不要继续堆神经网络。

### Phase 2：Counterfactual scorer

实现：

- support/conflict features；
- `CounterfactualEvidenceScorer`；
- `qce_counterfactual` ablation；
- 单元测试。

目标：

```text
明确冲突特征能否减少 wrong-year / wrong-company / wrong-metric 排名靠前？
```

### Phase 3：Learnable Router

实现：

- `QueryRelationRouter`；
- relation recovery target；
- router auxiliary loss；
- `qce_router` ablation。

### Phase 4：Full Model

实现：

- `qce_full_no_rgcn`
- `qce_full`
- 3 seeds；
- 完整消融和效率统计。

---

## 4. 必须新增的代码文件

```text
feg_rag/rerank/qce_expansion.py
feg_rag/rerank/qce_features.py
feg_rag/rerank/qce_dataset.py
feg_rag/rerank/qce_graph.py
experiments/qce_graph_ablation.py
tests/test_qce_expansion.py
tests/test_qce_features.py
tests/test_qce_model.py
QCE_GRAPH_README.md
```

需要修改：

```text
feg_rag/rerank/__init__.py
configs/default.yaml
experiments/table1_non_llm_reranking_comparison.py
```

要求：不要改变已有方法默认行为和结果。

---

## 5. 模块一：QueryRelationRouter

文件：`feg_rag/rerank/qce_graph.py`

### 5.1 固定关系集合

第一版只支持以下 7 种关系，不要继续增加关系类型：

```python
RELATION_NAMES = [
    "adjacent_chunk",
    "same_section",
    "same_filing",
    "same_company_year",
    "same_metric",
    "same_year",
    "semantic_similar",
]
```

### 5.2 Query features

建议维度为 10，集中定义常量：

```text
num_years
num_metrics
num_companies
query_length
has_numeric_question
has_comparison_keyword
has_delta_keyword
has_filing_type
has_section_keyword
is_ambiguous_short_query
```

允许根据当前仓库兼容性调整，但必须统一定义 feature order 和 dimension。

### 5.3 Router 网络结构

```text
query_features
-> Linear(10, 32)
-> ReLU
-> Dropout(0.1)
-> Linear(32, 7)
-> relation_logits
-> sigmoid
-> relation_probabilities
```

必须使用 `sigmoid`，不要用 `softmax`，因为同一个 query 可能需要多种关系。

### 5.4 预算分配

默认：

```yaml
expansion_budget: 30
max_budget_per_relation: 10
relation_threshold: 0.10
```

逻辑：

1. 对 `relation_probabilities` 低于 threshold 的关系不扩展；
2. 将剩余概率归一化；
3. 按总预算分配整数 budget；
4. 单关系不超过 `max_budget_per_relation`；
5. 分数相同用 `chunk_id` 稳定 tie-break。

### 5.5 Router 辅助训练目标

训练阶段为每个 query 计算 multi-label relation target：

```text
某关系独立扩展出的候选中包含训练 split gold -> target = 1
否则 -> target = 0
```

注意：

- 扩展候选必须先独立生成，不能为了 target 把 gold 塞进去；
- target 只用于训练 split；
- validation/test 不生成 relation target；
- 对所有关系都找不回 gold 的 query，target 全 0，并在日志中统计数量。

Loss：

```python
router_loss = BCEWithLogitsLoss(relation_logits, relation_targets)
```

---

## 6. 模块二：BudgetedGraphExpander

文件：`feg_rag/rerank/qce_expansion.py`

### 6.1 数据结构

```python
@dataclass
class ExpandedCandidate:
    chunk_id: str
    is_initial: bool
    initial_score: float
    initial_rank: int | None
    source_relations: list[str]
    best_relation: str | None
    best_seed_chunk_id: str | None
    best_seed_rank: int | None
    graph_distance: int
    expansion_priority: float
```

同一 chunk 被多个关系找回时必须去重，并保留所有 `source_relations`。

### 6.2 GraphExpansionIndex

```python
class GraphExpansionIndex:
    chunks_by_doc: dict[str, list[str]]
    chunks_by_section: dict[tuple[str, str], list[str]]
    chunks_by_company_year: dict[tuple[str, str], list[str]]
    chunks_by_metric: dict[str, list[str]]
    chunks_by_year: dict[str, list[str]]
    adjacent_chunks: dict[str, list[str]]
    semantic_neighbors: dict[str, list[tuple[str, float]]]
```

索引来源优先使用现有 `Chunk` 字段和图结构：

- `Chunk.doc_id`
- `Chunk.company`
- `Chunk.filing_year`
- `Chunk.section`
- `Chunk.metadata["word_offset"]`
- `Chunk.metadata["row_offset"]`
- `EntityExtractor`
- graph 中已有 `semantic-similar` edge

索引必须缓存，例如：

```text
outputs/qce_graph/cache/graph_expansion_index.pkl
```

缓存 fingerprint 至少包含：

- chunk/corpus cache 路径、大小、mtime；
- graph cache 路径、大小、mtime；
- index schema version；
- relation config。

### 6.3 扩展关系规则

#### `adjacent_chunk`

- 只在相同 `doc_id` 内；
- 优先相同 section；
- 按 `word_offset` 或 `row_offset` 找前后邻居；
- 每个 seed 最多前后各 1-2 个；
- 不允许跨文档。

#### `same_section`

- 只在相同 `doc_id + section` 内；
- 按与 seed 的 offset 距离排序；
- 不允许全局同名 section 扩展。

#### `same_filing`

- 相同 `doc_id`；
- 优先表格 chunk、相邻 section、包含 query tokens 的 chunk；
- 必须受预算限制。

#### `same_company_year`

- 相同 company 和 filing_year；
- 优先 query metric 匹配；
- 若 company 或 year 为空，不启用该关系。

#### `same_metric`

- candidate 与 query 至少共享一个标准化 metric；
- 优先同 company、同 year、同 filing；
- 不允许仅凭全局 `revenue` 加入大量无关公司候选。

#### `same_year`

- 只在 query 明确包含 year 时启用；
- 优先同 company 或同 filing；
- 不允许只凭年份全局扩展。

#### `semantic_similar`

- 使用图中已有 semantic edges；
- 默认低优先级；
- 默认每 query 最多扩展 5 个；
- 如果没有 semantic edge，降级为空，不报错。

### 6.4 Expansion priority

默认公式：

```text
priority =
    router_probability
  * relation_prior
  * seed_score
  * (1 / seed_rank)
  * (1 / (1 + graph_distance))
  * local_match_bonus
```

默认 `relation_prior`：

```python
relation_prior = {
    "adjacent_chunk": 1.00,
    "same_section": 0.90,
    "same_filing": 0.75,
    "same_company_year": 0.90,
    "same_metric": 0.85,
    "same_year": 0.65,
    "semantic_similar": 0.50,
}
```

这些值必须放入配置或集中常量，不能散落硬编码。

`local_match_bonus` 只允许使用 query 与 candidate 的可观察特征，例如 company/year/metric match，不能使用 gold。

### 6.5 默认候选限制

```yaml
initial_top_n: 50
seed_top_m: 15
expansion_budget: 30
max_total_candidates: 80
```

最终候选池：

```text
initial top-N + expanded candidates
```

候选总量不得超过 `max_total_candidates`。

---

## 7. 模块三：CounterfactualEvidenceScorer

文件：`feg_rag/rerank/qce_features.py` 和 `feg_rag/rerank/qce_graph.py`

### 7.1 核心原则

缺失信息不等于冲突。  
只有 candidate 明确出现了与 query 不一致的实体时，才记为 conflict。

### 7.2 Support features

固定 feature order：

```text
company_match
filing_year_match
year_text_match
metric_match
filing_type_match
section_match
query_text_overlap
same_filing_support
same_section_support
adjacent_support
route_alignment
```

`route_alignment`：

```text
candidate.source_relations 中对应 Router 概率的加权和
```

### 7.3 Conflict features

固定 feature order：

```text
company_conflict
year_conflict
metric_conflict
filing_type_conflict
section_conflict
```

定义：

`company_conflict = 1` 当且仅当：

- query 明确识别出 company；
- candidate 也有明确 company；
- 两者无交集。

`year_conflict = 1` 当且仅当：

- query 明确包含 year；
- candidate metadata 或文本也明确包含 year；
- 两者无交集。

`metric_conflict = 1` 当且仅当：

- query 有明确 metric；
- candidate 也有明确 metric；
- 标准化后无交集。

如果 candidate 没有 metric，只能设 `metric_missing=1` 这类辅助特征，不能直接设 conflict。

### 7.4 Scorer 网络结构

```text
query_features -> query_encoder -> 32-d query embedding

support_features + query_embedding
-> support_head
-> support_score in [0, 1]

conflict_features + query_embedding
-> conflict_head
-> conflict_score in [0, 1]
```

建议：

```python
support_head = Linear -> ReLU -> Dropout -> Linear -> Sigmoid
conflict_head = Linear -> ReLU -> Dropout -> Linear -> Sigmoid
```

hidden dim 默认 32。

---

## 8. 完整模型：QCEGraphLiteReranker

文件：`feg_rag/rerank/qce_graph.py`

```python
class QCEGraphLiteReranker(nn.Module):
    ...
```

输入：

```text
query_features
candidate_base_features
support_features
conflict_features
relation_origin_multi_hot
optional rgcn_score
```

基础分数：

```text
base_score = retrieval_score_norm
```

如果提供 R-GCN：

```text
base_score =
    base_retrieval_weight * retrieval_score_norm
  + base_rgcn_weight * rgcn_score_norm
```

`base_retrieval_weight` 和 `base_rgcn_weight` 可以是两个可学习全局标量，经 softmax 后和为 1。不要做复杂 gating。

最终分数：

```text
final_score =
    base_score
  + support_scale * support_score
  - conflict_scale * conflict_score
  + expansion_scale * expansion_priority_norm
```

要求：

- `support_scale`、`conflict_scale`、`expansion_scale` 为可学习标量；
- 用 sigmoid/softplus 约束为非负；
- 设置上限，避免覆盖基础检索：
  - `support_scale_max = 0.20`
  - `conflict_scale_max = 0.25`
  - `expansion_scale_max = 0.10`
- 初始化时模型应接近基础检索，不能随机破坏 baseline；
- forward 可选返回中间结果。

建议返回：

```python
{
    "score": final_score,
    "base_score": base_score,
    "support_score": support_score,
    "conflict_score": conflict_score,
    "relation_probs": relation_probs,
    "route_alignment": route_alignment,
}
```

---

## 9. 训练数据构造

文件：`feg_rag/rerank/qce_dataset.py`

### 9.1 训练候选池

训练时，为避免未训练 Router 漏掉关系，使用：

```text
initial top-N
+
每种关系最多 train_max_per_relation 个预扩展候选
```

默认：

```yaml
train_max_per_relation: 10
train_pool_cap: 120
```

预扩展必须只依赖 query、seed candidates、chunk metadata、graph，不能看 gold。  
候选池构造完成后，才允许使用训练 gold 构造 positive/negative 和 relation targets。

### 9.2 正负样本

正样本：

```text
candidate_pool 中属于 gold_chunk_ids 的 chunk
```

负样本优先级：

1. 初始 top-10 中非 gold；
2. 扩展后高 priority 但非 gold；
3. wrong-company / wrong-year / wrong-metric 明确冲突候选；
4. 随机普通负样本。

默认：

```yaml
hard_negatives_per_positive: 10
random_negatives_per_positive: 2
```

不要只取第一个 negative。

---

## 10. Loss

总损失：

```text
L_total = L_rank + lambda_router * L_router + lambda_scale * L_scale
```

### 10.1 Pairwise ranking loss

```python
L_rank = softplus(-(positive_score - negative_score))
```

冲突负样本加权：

```text
negative_weight =
    1
  + 0.5 * company_conflict
  + 0.5 * year_conflict
  + 0.5 * metric_conflict
```

### 10.2 Router loss

```python
L_router = binary_cross_entropy_with_logits(
    relation_logits,
    relation_recovery_targets,
)
```

### 10.3 Scale regularization

```text
L_scale = support_scale^2 + conflict_scale^2 + expansion_scale^2
```

默认：

```yaml
lambda_router: 0.20
lambda_scale: 0.001
```

---

## 11. 训练和推理流程

### 11.1 训练

```text
加载 chunks / graph / initial results / optional R-GCN results
-> 构建或加载 GraphExpansionIndex
-> 为训练 query 生成不看 gold 的预扩展候选池
-> 用训练 gold 构造 pair 和 relation targets
-> 物化 features 并缓存
-> 训练 QCEGraphLiteReranker
-> 按 validation nDCG@10 early stopping，secondary metric = MRR
-> 保存 best checkpoint
```

如果每 epoch 全量评估太贵，可以每 2-3 epoch 评估一次。

### 11.2 推理

```text
加载 query
-> 初始 top-N
-> Router 输出 relation probabilities
-> 按预算 graph expansion
-> 合并去重候选
-> 提取 support/conflict/base features
-> 批量打分
-> 输出 top-k
```

---

## 12. 实验脚本

文件：`experiments/qce_graph_ablation.py`

必须支持方法：

```text
initial_retriever
rgcn
qce_fixed
qce_router
qce_counterfactual
qce_full_no_rgcn
qce_full
```

说明：

- `rgcn` 直接读取已有 baseline result，不重新训练；
- `qce_fixed`：固定平均 relation budgets + graph expansion + 普通 lightweight scorer，无 learnable router，无 conflict channel；
- `qce_router`：learnable router + budgeted expansion，无 conflict channel；
- `qce_counterfactual`：不扩展新候选，只在原候选池内用 support/conflict scoring；
- `qce_full_no_rgcn`：Router + expansion + support/conflict + initial retrieval score；
- `qce_full`：在 full_no_rgcn 基础上可选加入 R-GCN score。

---

## 13. 配置

在 `configs/default.yaml` 增加：

```yaml
qce_graph:
  enabled: false

  initial_top_n: 50
  seed_top_m: 15
  expansion_budget: 30
  max_budget_per_relation: 10
  max_total_candidates: 80
  train_max_per_relation: 10
  train_pool_cap: 120

  relation_threshold: 0.10
  semantic_max_per_query: 5

  query_feature_dim: 10
  router_hidden_dim: 32
  scorer_hidden_dim: 32
  dropout: 0.10

  support_scale_max: 0.20
  conflict_scale_max: 0.25
  expansion_scale_max: 0.10

  hard_negatives_per_positive: 10
  random_negatives_per_positive: 2

  epochs: 30
  batch_size: 512
  lr: 0.001
  weight_decay: 0.00001
  early_stopping_patience: 5
  eval_every: 2

  lambda_router: 0.20
  lambda_scale: 0.001

  use_rgcn_score: true
  use_semantic_relation: true
  cache_dir: outputs/qce_graph/cache
```

所有配置都需要 CLI 覆盖。

---

## 14. 输出文件

默认输出目录：

```text
outputs/qce_graph/<timestamp_or_run_name>/
```

必须生成：

```text
config_snapshot.yaml
split_manifest.json
train_history.csv
best_checkpoint.pt
metrics_summary.csv
ablation_summary.csv
per_query_results.jsonl
expansion_stats.json
relation_usage.csv
relation_recovery.csv
error_breakdown.csv
run.log
```

### 14.1 `per_query_results.jsonl`

每行至少包含：

```json
{
  "question_id": "...",
  "question": "...",
  "gold_evidence_ids": ["..."],
  "initial_chunk_ids": ["..."],
  "expanded_chunk_ids": ["..."],
  "retrieved_chunk_ids": ["..."],
  "relation_probabilities": {
    "adjacent_chunk": 0.0,
    "same_section": 0.0,
    "same_filing": 0.0,
    "same_company_year": 0.0,
    "same_metric": 0.0,
    "same_year": 0.0,
    "semantic_similar": 0.0
  },
  "new_gold_recovered": false,
  "method": "qce_full"
}
```

### 14.2 `expansion_stats.json`

至少包含：

```text
num_queries
avg_initial_candidates
avg_expanded_candidates
avg_total_candidates
candidate_recall_before_expansion
candidate_recall_after_expansion
num_queries_with_new_gold
new_gold_recovery_rate
num_queries_no_relation_active
```

---

## 15. 测试要求

### 15.1 Expansion tests

必须测试：

1. adjacent candidate 不跨 `doc_id`；
2. same_section 只限同文档同 section；
3. same_year 不做全局无约束扩展；
4. 候选去重且保留多个 source relations；
5. 候选总数不超过 `max_total_candidates`；
6. 固定 seed 后结果完全可复现；
7. 无 semantic edge 时正常降级；
8. expansion 不使用 gold。

### 15.2 Feature tests

必须测试：

1. candidate 缺少年份不是 year conflict；
2. candidate 明确包含错误年份才是 year conflict；
3. metric alias 正确归一，例如：
   - revenues / net sales / sales -> revenue
   - net earnings / profit -> net income
   - earnings per share / EPS -> eps
4. 空 query entity 不产生 conflict；
5. route alignment 计算正确；
6. feature shape 与常量一致。

### 15.3 Model tests

必须测试：

1. forward 输出 shape；
2. 分数无 NaN/Inf；
3. relation probabilities 在 `[0,1]`；
4. support 增强时分数不应系统性下降；
5. conflict 增强时分数不应系统性上升；
6. 保存和加载 checkpoint 后输出一致；
7. `use_rgcn_score=false` 时仍可运行。

---

## 16. 命令行接口

Smoke test：

```bash
PYTHONPATH=/root/fin-gnn python experiments/qce_graph_ablation.py \
  --sanity \
  --methods initial_retriever,qce_fixed,qce_counterfactual \
  --device cpu \
  --epochs 2 \
  --progress_every 20
```

完整 held-out 实验：

```bash
PYTHONPATH=/root/fin-gnn python experiments/qce_graph_ablation.py \
  --methods initial_retriever,rgcn,qce_fixed,qce_router,qce_counterfactual,qce_full_no_rgcn,qce_full \
  --initial_results_jsonl outputs/v2_table1_bge_m3_correct_corpus_20260715_123130/bge_m3_dense_results.jsonl \
  --rgcn_results_jsonl outputs/v2_table2_graph_bge_pool_rgcn_eval_fast_20260716_032108/rgcn_results.jsonl \
  --graph_cache cache/table2_graph_features_bge_pool_seq4096.pkl \
  --corpus_cache cache/table1_full_corpus_seq4096.pkl \
  --top_n 50 \
  --expansion_budget 30 \
  --max_total_candidates 80 \
  --device cuda \
  --seeds 42,43,44 \
  --output_dir outputs/qce_graph/heldout_main \
  --progress_every 100
```

如果当前仓库输出路径不同，请以现有文件为准，并提供清晰报错，不要静默使用错误文件。

---

## 17. 公平性要求

所有方法必须：

- 使用相同 train/validation/test split；
- 使用相同 gold mapping；
- 使用相同初始检索结果；
- 使用相同 top-k metric 函数；
- 使用相同随机种子集合；
- 不允许 qce_full 使用更好的、但未单独报告的初始检索器。

需要报告 3 个随机种子：

```text
mean ± std
```

成功判定建议：

```text
QCE-Full 在相同 held-out 协议下，相比原始 R-GCN，
至少在 MRR、Recall@10、nDCG@10 三项中的两项取得稳定正提升，
且多个随机种子方向一致。
```

这只是实验目标，未得到结果前不得写成结论。

---

## 18. 效率约束

模型必须保持轻量：

- router hidden dim <= 32；
- scorer hidden dim <= 32；
- 不使用 Transformer；
- 不新增多层大 GNN；
- 不在线运行 MonoT5；
- 不对全图做每 query 多跳最短路；
- relation index 构建一次并缓存；
- query-candidate features 物化并缓存；
- 默认候选总数 <= 80；
- 支持 CPU 推理；
- 参数量应在数万级以内，并输出实际参数量。

日志必须报告：

```text
index build time
feature build time
training time
inference time per query
peak candidate count
model parameter count
```

所有训练/评估循环必须按 `--progress_every` 打印进度、elapsed、ETA。

---

## 19. README 要求

`QCE_GRAPH_README.md` 必须说明：

1. R-GCN 只重排已有候选；QCE 可以通过图关系补充新候选；
2. Router 决定当前问题沿哪些关系找证据；
3. Budget 控制最多扩展多少，避免候选爆炸；
4. Support channel 奖励匹配证据；
5. Conflict channel 惩罚明确错公司、错年份、错指标；
6. R-GCN 是可选基础信号，不是新模型必须依赖的主干；
7. 当前结果必须如实报告，包括失败消融；
8. 不得把常规调参写成主要创新。

建议文件头：

```python
"""
QCE-Graph Lite: Query-Conditioned Counterfactual Evidence Graph Reranker.

Project-level contributions:
1. Query-conditioned multi-label relation routing.
2. Budget-constrained graph candidate expansion.
3. Dual-channel support/conflict evidence scoring.

The model is intentionally lightweight and may optionally consume a
pre-computed R-GCN score. It does not modify the vanilla R-GCN baseline.
"""
```

---

## 20. 最终验收清单

- [ ] 原始 baseline 文件未被修改或覆盖；
- [ ] 新模型能在无 R-GCN 分数时运行；
- [ ] 新模型能在有 R-GCN 分数时运行；
- [ ] graph expansion 不使用 gold；
- [ ] relation router 有独立类和独立 loss；
- [ ] budgeted expander 有独立类；
- [ ] support/conflict scorer 有独立类；
- [ ] 所有创新模块均有 CLI 消融开关；
- [ ] smoke test 通过；
- [ ] pytest 通过；
- [ ] 生成 before/after candidate recall；
- [ ] 生成 held-out MRR、R@5、R@10、nDCG@10；
- [ ] 生成 3 seeds mean ± std；
- [ ] 输出参数量、训练时间和推理时间；
- [ ] README 不包含伪造结果；
- [ ] 完整模型结果不佳时也保留并分析，不隐藏负结果。

---

## 21. 一句话模型定义

QCE-Graph Lite 根据金融问题动态选择图关系，在固定预算内从初始候选周围扩展可能遗漏的证据，然后通过 support/conflict 双通道轻量打分；R-GCN 分数可作为可选基础信号，但 vanilla R-GCN baseline 保持不变。
