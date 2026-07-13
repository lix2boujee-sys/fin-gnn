# E5-Mistral Dense Encoder Comparison Requirement

## Goal

Add an E5-Mistral dense embedding option to the existing FEG-RAG experiment pipeline so we can compare against the current `all-MiniLM-L6-v2` baseline under the same FinDER setting.

Motivation: the original dataset reported best performance with E5-Mistral, so this project needs an apples-to-apples comparison:

- Current baseline: `all-MiniLM-L6-v2`, 384-dim embeddings.
- Target comparison model: `intfloat/e5-mistral-7b-instruct`, 4096-dim embeddings.
- Keep BM25, Hybrid fusion, graph construction, PPR, GNN training, evaluation split, and metrics unchanged unless a change is strictly required for embedding compatibility.

## Current Code Context

Important files:

- `feg_rag/retrieval/dense.py`
  - Current dense retrieval entry point.
  - Supports a local HuggingFace directory through `_TransformersEncoder`.
  - Falls back to `sentence-transformers` for hub model names.
  - Current local default is `D:/fin-gnn/cache/models/all-MiniLM-L6-v2`.

- `configs/default.yaml`
  - Current dense model path:
    `D:/fin-gnn/cache/models/all-MiniLM-L6-v2`

- `experiments/exp1_retrieval_baseline.py`
  - Runs BM25 / Dense / Hybrid baseline.

- `experiments/exp3_feg_ppr.py`
  - Runs Hybrid + graph/PPR variants.

- `experiments/exp4_gnn_reranker.py`
  - Runs Hybrid / PPR / GraphSAGE or R-GCN reranker.
  - Currently has MiniLM-specific constants:
    - `EXPECTED_DENSE_DIM = 384`
    - `EXPECTED_FEATURE_DIM = 391`
  - These must become dynamic or model-aware. E5-Mistral embeddings are 4096-dim, so node feature dim should become `4096 + 1 + 3 + 2 + 1 = 4103`.

- `feg_rag/graph/features.py`
  - Also warns using MiniLM-specific expected feature dim `391`.
  - This must not warn incorrectly for E5-Mistral.

## Functional Requirements

### 1. Add E5-Mistral as a supported dense encoder

Support the HuggingFace / sentence-transformers model:

```text
intfloat/e5-mistral-7b-instruct
```

Also support a local cached path, preferred for this machine:

```text
D:/fin-gnn/cache/models/e5-mistral-7b-instruct
```

Do not remove or break the existing MiniLM local model.

### 2. Add a dedicated config

Create a separate config file, for example:

```text
configs/e5_mistral.yaml
```

It should inherit the same settings as `configs/default.yaml` where possible, but set:

```yaml
retrieval:
  dense_model: "D:/fin-gnn/cache/models/e5-mistral-7b-instruct"
  top_k: 50
  hybrid_alpha: 0.5
```

If the codebase does not support config inheritance, duplicate only the minimal necessary default settings.

### 3. Use correct E5 text formatting

E5-style models are instruction/query sensitive. Implement model-aware text formatting in the dense encoder:

- For corpus chunks / passages, encode as passage text.
- For search queries, encode as query text.
- Keep current MiniLM behavior unchanged.

Recommended behavior:

- For normal E5 models:
  - passage: `passage: {text}`
  - query: `query: {question}`

- For `intfloat/e5-mistral-7b-instruct`, use an instruction-style query prefix. A practical default:

```text
Instruct: Given a financial question, retrieve relevant financial report evidence passages
Query: {question}
```

For passages, use:

```text
passage: {chunk_text}
```

Make this behavior explicit and testable. Avoid silently applying E5 prefixes to non-E5 models.

### 4. Make embedding dimensions dynamic

Remove MiniLM-only assumptions from the experiment validation logic.

Required changes:

- In `experiments/exp4_gnn_reranker.py`, do not hard-code `EXPECTED_DENSE_DIM = 384` and `EXPECTED_FEATURE_DIM = 391` as universal expectations.
- Compute:

```python
expected_feature_dim = embedding_dim + 1 + num_node_types + 2 + 1
```

- The run should accept both:
  - MiniLM: dense dim `384`, feature dim `391`
  - E5-Mistral: dense dim `4096`, feature dim `4103`

- In `feg_rag/graph/features.py`, remove or generalize the MiniLM-specific warning.

### 5. Control memory usage

E5-Mistral is much larger than MiniLM. The implementation must expose conservative batch settings.

Requirements:

- Add a dense batch size CLI option where needed, especially in:
  - `experiments/exp1_retrieval_baseline.py`
  - `experiments/exp3_feg_ppr.py`
  - `experiments/exp4_gnn_reranker.py`

Suggested flag:

```text
--dense_batch_size
```

Default behavior:

- MiniLM CPU: keep current effective defaults.
- E5-Mistral CPU/GPU: allow small values such as `1`, `2`, `4`, or `8`.

Pass this batch size into `DenseRetriever.index(...)`.

Also make query encoding use a safe small batch implicitly; one query at a time is fine.

### 6. Preserve comparable outputs

When running with E5-Mistral, output directories should be distinct from MiniLM runs.

Suggested output dirs:

```text
outputs/exp1_e5_mistral
outputs/exp3_e5_mistral_ppr
outputs/exp4_e5_mistral_gnn
```

Each output README / status should record:

- dense model path/name
- dense backend
- dense embedding dim
- feature dim, for Exp4
- dense device
- dense batch size
- command used

### 7. Do not change evaluation protocol

For the final comparison, keep these aligned with existing full runs:

- Dataset: FinDER full set, `num_samples=0`.
- Retrieval candidate depth: `top_n=50` for Exp3/Exp4.
- Exp4 evaluation: `--eval_on all`.
- GNN model: default `sage`, unless explicitly comparing `rgcn`.
- Same metrics:
  - MRR
  - Recall@1/3/5/10/20
  - nDCG@1/3/5/10/20

## Suggested Commands

MiniLM baseline already exists. Add E5-Mistral commands like:

```powershell
python experiments/exp1_retrieval_baseline.py `
  --config configs/e5_mistral.yaml `
  --num_samples 0 `
  --top_k 10 `
  --dense_device cpu `
  --dense_batch_size 2 `
  --output_dir outputs/exp1_e5_mistral
```

```powershell
python experiments/exp3_feg_ppr.py `
  --config configs/e5_mistral.yaml `
  --num_samples 0 `
  --top_n 50 `
  --dense_device cpu `
  --dense_batch_size 2 `
  --output_dir outputs/exp3_e5_mistral_ppr
```

```powershell
python experiments/exp4_gnn_reranker.py `
  --config configs/e5_mistral.yaml `
  --num_samples 0 `
  --epochs 50 `
  --no_ablation `
  --eval_on all `
  --dense_device cpu `
  --dense_batch_size 2 `
  --device cuda `
  --output_dir outputs/exp4_e5_mistral_gnn
```

If CPU E5 encoding is too slow, support `--dense_device cuda` with `--dense_batch_size 1`, but do not assume it will fit on a 4GB GPU.

## Smoke Tests

Before full runs, verify with small samples:

```powershell
python experiments/exp1_retrieval_baseline.py `
  --config configs/e5_mistral.yaml `
  --num_samples 20 `
  --top_k 10 `
  --dense_device cpu `
  --dense_batch_size 1 `
  --output_dir outputs/smoke_exp1_e5_mistral `
  --overwrite_output_dir
```

```powershell
python experiments/exp4_gnn_reranker.py `
  --config configs/e5_mistral.yaml `
  --sanity `
  --dense_device cpu `
  --dense_batch_size 1 `
  --device cuda `
  --output_dir outputs/smoke_exp4_e5_mistral `
  --overwrite_output_dir
```

Expected smoke-test checks:

- Dense backend loads E5-Mistral, not MiniLM fallback.
- Dense embedding dim prints as `4096`.
- Exp4 feature dim prints as `4103`.
- No MiniLM-specific feature-dimension warning appears.
- JSONL and `metrics_summary.csv` are written.

## Acceptance Criteria

The implementation is complete when:

1. Existing MiniLM runs still work without changing current commands.
2. `configs/e5_mistral.yaml` can select E5-Mistral without editing source code.
3. E5-Mistral corpus and query encodings use proper E5/instruction prefixes.
4. Exp4 no longer assumes 384-dim dense embeddings.
5. A smoke Exp1 E5 run completes and writes metrics.
6. A smoke Exp4 E5 run completes and writes metrics.
7. Full-run output files clearly distinguish MiniLM vs E5-Mistral results.

## Important Non-Goals

Do not do these unless asked separately:

- Do not fine-tune E5-Mistral.
- Do not change the FinDER train/eval protocol.
- Do not change BM25 tokenization or Hybrid score fusion.
- Do not refactor graph construction unrelated to embedding compatibility.
- Do not replace GraphSAGE/R-GCN architecture just because E5 embeddings are larger.

## Notes for Risk Areas

- The current `DenseRetriever._resolve_model_path()` silently falls back to MiniLM if a requested path does not exist and the default local MiniLM exists. For E5 comparison, this is dangerous. If the user explicitly asks for E5-Mistral and the E5 local path is missing, fail loudly instead of falling back to MiniLM.
- E5-Mistral requires much more RAM/VRAM and time. Keep batch size configurable and log it.
- E5 embeddings are high-dimensional. FAISS `IndexFlatIP` is fine for correctness, but memory use will increase. Full corpus size is about 31k chunks, so embeddings are roughly `31k * 4096 * 4 bytes`, around 500 MB before overhead.
- If `sentence-transformers` cannot load E5-Mistral locally, use `transformers` with mean pooling only as a fallback if results are documented. Prefer the official sentence-transformers path when available because pooling/prompt behavior matters for fair comparison.
