# LLM Reranker and Main Generation Experiment Requirements

## Goal

Add the final LLM-related experiment layer on top of the existing FEG-RAG retrieval/reranking pipeline.

There are three large experiment groups:

1. **Non-LLM reranking comparison**: prove that graph-based reranking itself improves evidence ranking.
2. **Graph-assisted LLM reranker**: auxiliary experiment showing graph reranking can reduce LLM reranker input cost.
3. **Main answer generation experiment**: use retrieved/reranked top evidence to generate final answers, without LLM post-processing.

Implementation priority:

1. Implement Experiment 1 first.
2. Implement Experiment 3 second.
3. Implement Experiment 2 later as an auxiliary/cost experiment.

Use **Qwen2.5-7B** through OpenRouter for LLM generation and LLM reranking. The OpenRouter API key must be read from the environment variable `OPENROUTER_API_KEY`; do not hard-code API keys in source code, config files, logs, or result artifacts.

Recommended configurable OpenRouter model id:

```yaml
llm_model: "qwen/qwen-2.5-7b-instruct"
```

Keep the exact model id configurable in YAML/CLI because OpenRouter model slugs may vary.

## Security Requirement

Do not commit or write the OpenRouter key into the repository.

Expected runtime usage:

```bash
export OPENROUTER_API_KEY="..."
```

For Windows PowerShell:

```powershell
$env:OPENROUTER_API_KEY="..."
```

If `OPENROUTER_API_KEY` is missing, fail loudly with a clear error message.

## Existing Context

The project already supports:

- BM25 retrieval
- Dense retrieval with MiniLM and E5-Mistral
- Hybrid retrieval
- PPR graph reranking
- GraphSAGE / R-GCN style GNN reranking
- FinDER evidence-level retrieval evaluation

The new work should reuse existing retrieval, graph, reranking, and metric utilities whenever possible.

Do not rewrite the existing pipeline unless necessary.

## Experiment 1: Non-LLM Reranking Comparison

### Purpose

This is the first key evidence-ranking table.

It proves whether graph-based reranking improves evidence ranking without using an LLM reranker.

Core conclusion expected from this table:

> Graph-based reranking improves evidence ranking compared with non-graph reranking methods.

### Input

Use existing retrieval outputs and/or existing pipeline components:

- Best Retriever
- Cross-Encoder reranker
- PPR reranker
- GraphSAGE reranker
- R-GCN reranker
- R-GCN + constraint score

Use the same FinDER split and same evidence corpus as the existing E5-Mistral experiments.

### Methods To Compare

Output rows:

```text
Best Retriever
+ Cross-Encoder
+ PPR
+ GraphSAGE
+ R-GCN
+ R-GCN + Constraint Score
```

If some methods are not implemented yet, implement missing ones with clear modular files rather than mixing everything into one script.

### Metrics

Report:

```text
Recall@5
Recall@10
MRR
nDCG@10
```

### Output Files

Use clear filenames:

```text
outputs/table1_non_llm_reranking/
  table1_non_llm_reranking_comparison.csv
  table1_non_llm_reranking_comparison.md
  best_retriever_results.jsonl
  cross_encoder_results.jsonl
  ppr_results.jsonl
  graphsage_results.jsonl
  rgcn_results.jsonl
  rgcn_constraint_results.jsonl
  metrics_full.json
  README.md
```

### Suggested Script Name

```text
experiments/table1_non_llm_reranking_comparison.py
```

### Suggested Config Name

```text
configs/table1_non_llm_reranking_e5_mistral.yaml
```

### Important Implementation Notes

- Do not use LLM reranking in this experiment.
- Keep the comparison focused on evidence ranking only.
- Reuse E5-Mistral retrieval outputs when possible.
- If a method uses trained GNN checkpoints, save and load them with clear checkpoint names.
- All methods must evaluate on the same query set and same candidate/evidence pool.

## Experiment 2: Graph-Assisted LLM Reranker

### Purpose

This is an auxiliary experiment.

It tests whether graph-based reranking can serve as a pre-filter before LLM reranking, reducing the number of candidates sent to the LLM while keeping performance close to direct LLM rerank top-50.

Core conclusion expected from this table:

> Graph reranking can act as a pre-filter for the LLM reranker, reducing LLM input candidate count and token cost while preserving or approaching the effectiveness of direct LLM rerank top-50.

This conclusion is better and more defensible than simply saying "GNN beats LLM".

### Methods To Compare

Output rows:

```text
LLM rerank top-50
LLM rerank top-20
Cross-Encoder -> LLM
PPR -> LLM
R-GCN -> LLM
```

Candidate counts:

```text
LLM rerank top-50: 50 input candidates
LLM rerank top-20: 20 input candidates
Cross-Encoder -> LLM: 10 input candidates
PPR -> LLM: 10 input candidates
R-GCN -> LLM: 10 input candidates
```

### LLM Reranker Behavior

Given one query and N candidate evidence passages, the LLM should output a ranked list of candidate IDs.

The prompt must:

- Include the question.
- Include candidate IDs and short evidence text.
- Ask the model to rank candidates by how likely they support the correct answer.
- Require strict JSON output.
- Forbid generating the final answer in this experiment.

Expected JSON format:

```json
{
  "ranked_candidate_ids": ["candidate_id_1", "candidate_id_2"],
  "rationale": "brief explanation"
}
```

The parser should be robust:

- Try JSON parsing first.
- If parsing fails, attempt a conservative recovery.
- If recovery fails, mark the query as failed and keep the original candidate order.

### Metrics

Report:

```text
Recall@5
MRR
nDCG@5
Token Cost
```

Token cost should include at least:

```text
prompt_tokens
completion_tokens
total_tokens
estimated_cost_usd
```

If OpenRouter does not return exact cost, compute an estimated token count and store provider metadata separately.

### Output Files

Use clear filenames:

```text
outputs/table2_graph_assisted_llm_reranker/
  table2_graph_assisted_llm_reranker.csv
  table2_graph_assisted_llm_reranker.md
  llm_rerank_top50_results.jsonl
  llm_rerank_top20_results.jsonl
  cross_encoder_to_llm_results.jsonl
  ppr_to_llm_results.jsonl
  rgcn_to_llm_results.jsonl
  token_cost_summary.csv
  llm_call_failures.jsonl
  metrics_full.json
  README.md
```

### Suggested Script Name

```text
experiments/table2_graph_assisted_llm_reranker.py
```

### Suggested Config Name

```text
configs/table2_graph_assisted_llm_reranker_qwen25_7b.yaml
```

### Important Implementation Notes

- This is not the main result table.
- Run on a configurable subset first to control API cost.
- Add CLI flags:

```text
--num_samples
--llm_model
--max_candidates
--temperature
--output_dir
--overwrite_output_dir
--resume
```

- Default temperature should be `0`.
- Implement resume support so completed LLM calls are not repeated.
- Cache LLM responses by query ID + method + candidate IDs + model name.

## Experiment 3: Main Answer Generation Experiment

### Purpose

This is the most important final experiment.

It measures whether better evidence ranking leads to better generated financial answers.

Important isolation rule:

> Do not use LLM reranking and do not use LLM post-processing in this experiment.

The same Qwen2.5-7B generator should receive the top-5 evidence from each method and generate an answer directly.

Core conclusion expected:

> Better evidence ranking produces more accurate, faithful, and numerically consistent answers.

### Pipeline

For each query:

```text
Query
-> method produces top-5 evidence
-> same Qwen2.5-7B generates answer from only those top-5 evidence passages
-> compare Accuracy / Faithfulness / Numerical Consistency
```

### Methods To Compare

At minimum:

```text
Best Retriever
Cross-Encoder
PPR
GraphSAGE
R-GCN
R-GCN + Constraint Score
```

If the final paper only has room for fewer methods, keep:

```text
Best Retriever
Cross-Encoder
PPR
R-GCN
R-GCN + Constraint Score
```

### LLM Generator Behavior

The generator receives:

- Question
- Top-5 evidence passages
- Optional metadata: company, year, metric, filing/section

The prompt must require:

- Use only the provided evidence.
- Do not use outside knowledge.
- Answer concisely.
- Preserve numbers and units exactly when possible.
- If evidence is insufficient, output `INSUFFICIENT_EVIDENCE`.

Expected JSON output:

```json
{
  "answer": "...",
  "evidence_ids_used": ["..."],
  "confidence": "high|medium|low"
}
```

### No LLM Post-Processing

Do not add a second LLM call to clean, verify, repair, rewrite, or post-process the answer.

The main generation result should reflect the quality of the evidence supplied to the same generator.

### Evaluation Metrics

Report:

```text
Answer Accuracy
Faithfulness
Numerical Consistency
Exact Match or relaxed match if already available
Evidence Hit@5
```

If there is no existing automatic evaluator for answer quality, implement a simple deterministic evaluator first:

- Normalize strings.
- Extract and compare numbers.
- Compare units/percent signs where possible.
- Check whether answer text is supported by retrieved evidence.

Optional later addition:

- LLM-as-judge evaluation, but keep it separate from the main deterministic result.

### Output Files

Use clear filenames:

```text
outputs/table3_main_generation_qwen25_7b/
  table3_main_generation_results.csv
  table3_main_generation_results.md
  best_retriever_generation.jsonl
  cross_encoder_generation.jsonl
  ppr_generation.jsonl
  graphsage_generation.jsonl
  rgcn_generation.jsonl
  rgcn_constraint_generation.jsonl
  answer_eval_full.json
  answer_eval_failures.jsonl
  token_cost_summary.csv
  README.md
```

### Suggested Script Name

```text
experiments/table3_main_generation_qwen25_7b.py
```

### Suggested Config Name

```text
configs/table3_main_generation_qwen25_7b.yaml
```

### Important Implementation Notes

- Use top-5 evidence for every method.
- Use the same LLM model, decoding parameters, and answer prompt for every method.
- Default temperature should be `0`.
- Add resume support to avoid repeated OpenRouter calls.
- Save raw LLM responses for auditability.
- Save parsed answers separately from raw responses.
- Log token usage per method.
- Do not let the answer generator see method names.
- Do not let the answer generator see gold answers.

## Shared OpenRouter Client Requirements

Create a small reusable client instead of duplicating request code.

Suggested file:

```text
feg_rag/generation/openrouter_client.py
```

Responsibilities:

- Read `OPENROUTER_API_KEY`.
- Support configurable model name.
- Support temperature, max tokens, timeout, retry count.
- Return raw response, parsed content, and token usage.
- Handle rate limits and transient network errors with backoff.
- Never print the API key.

Suggested helper files:

```text
feg_rag/generation/llm_prompts.py
feg_rag/generation/llm_response_parser.py
feg_rag/generation/token_cost.py
```

## Shared Resume and Cache Requirements

LLM experiments must support resume.

Suggested cache directory:

```text
cache/llm_calls/
```

Cache key should include:

```text
experiment_name
method_name
query_id
model_name
candidate_ids or evidence_ids
prompt_version
```

If a cached response exists, do not call OpenRouter again.

## CLI Requirements

Each new experiment script should support:

```text
--config
--num_samples
--output_dir
--overwrite_output_dir
--resume
--llm_model
--temperature
--max_tokens
--top_k_evidence
```

LLM reranker script should additionally support:

```text
--llm_candidate_count
--prefilter_method
```

Generation script should additionally support:

```text
--generation_top_k 5
```

## Recommended Development Order

1. Add shared OpenRouter client and prompt/parser utilities.
2. Add Experiment 1 table script if missing methods need orchestration.
3. Add Experiment 3 main generation script.
4. Run small smoke tests:

```bash
--num_samples 5
```

5. Run table-level experiments on a controlled subset:

```bash
--num_samples 100
```

6. Run full experiments only after smoke tests pass.
7. Add Experiment 2 auxiliary LLM reranker after Experiment 1 and 3 are stable.

## Smoke Test Expectations

For LLM calls, first run:

```bash
--num_samples 3
```

Check:

- API key is read from environment.
- Model returns valid JSON most of the time.
- Cache works.
- Resume works.
- Token usage is written.
- No API key appears in logs.

## Paper Table Naming

Use these names consistently:

```text
Table 1: Non-LLM Reranking Comparison
Table 2: Graph-Assisted LLM Reranker
Table 3: Main Generation Results with Qwen2.5-7B
```

Chinese captions:

```text
表 1：不使用 LLM reranker 的重排对比
表 2：图方法辅助 LLM reranker
表 3：主生成实验，不使用 LLM 后处理
```

## Key Experimental Claims

Experiment 1 supports:

> Graph-based reranking improves evidence ranking without relying on an LLM reranker.

Experiment 2 supports:

> Graph reranking can reduce LLM reranker input candidates and token cost while preserving most of the ranking benefit.

Experiment 3 supports:

> Better evidence ranking improves final answer quality when the same Qwen2.5-7B generator is used without LLM post-processing.

