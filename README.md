# FEG-RAG: Financial Evidence Graph Retrieval-Augmented Generation

This repository contains the reproducible code for financial evidence retrieval and graph-assisted reranking experiments on FinDER-style SEC filing evidence retrieval.

The code compares first-stage retrievers such as BM25, dense retrieval, hybrid retrieval, ColBERTv2, and E5-Mistral-7B-Instruct, then evaluates graph-based reranking methods such as PPR, GraphSAGE, and R-GCN.

## What Is Included

- `feg_rag/`: retrieval, graph construction, reranking, evaluation, and generation utilities.
- `experiments/`: scripts for Table I retrieval comparison, non-LLM reranking, graph reranking, and LLM generation experiments.
- `configs/`: YAML configs for local and cloud runs.
- `scripts/`: helper scripts such as E5-Mistral model download.
- `tests/`: smoke tests for critical retrieval paths.

## What Is Not Included

Large files are intentionally excluded from GitHub:

- FinDER data files.
- SEC 10-K HTML files.
- HuggingFace model caches and model weights.
- FAISS indexes, PyTorch checkpoints, and experiment outputs.
- Local paper PDFs and zip archives.

These are ignored by `.gitignore` and should be downloaded or regenerated as needed.

## Environment

Create an environment and install dependencies:

```bash
pip install -r requirements.txt
```

For local Windows setup, you can also use:

```powershell
.\setup_env.ps1
```

For cloud GPU setup, see:

```bash
bash cloud_setup.sh
```

## Data

Download FinDER data:

```bash
python download_finder.py
```

If you use local SEC 10-K files, place them under:

```text
10-k/
```

## Models

MiniLM, E5-Mistral, ColBERTv2, and cross-encoder weights are not committed.

Expected local model paths used by the configs include:

```text
cache/models/all-MiniLM-L6-v2
cache/models/e5-mistral-7b-instruct
cache/models/colbertv2.0
```

Download E5-Mistral helper:

```bash
python scripts/download_e5_mistral.py
```

ColBERTv2 can be downloaded from HuggingFace:

```bash
hf download colbert-ir/colbertv2.0 --local-dir cache/models/colbertv2.0
```

## Key Experiments

### Table I: Initial Retrieval Comparison

Compares BM25, Dense Retriever, Hybrid Retriever, ColBERTv2, and E5-Mistral-7B-Instruct.

```bash
python experiments/table1_initial_retrieval_comparison.py \
  --config configs/table1_initial_retrieval_comparison_cloud.yaml \
  --output_dir outputs/table1_initial_retrieval_comparison \
  --dense_device cuda
```

For quick cloud verification, use a small sample:

```bash
python experiments/table1_initial_retrieval_comparison.py \
  --config configs/table1_initial_retrieval_comparison_cloud.yaml \
  --limit_samples 20 \
  --output_dir outputs/table1_initial_retrieval_smoke \
  --overwrite_output_dir \
  --dense_device cuda
```

### E5-Mistral Standalone Retrieval

This path uses the SentenceTransformer wrapper for `e5-mistral-7b-instruct` and validates top-k outputs.

```bash
python experiments/run_e5_mistral_standalone.py \
  --model_path cache/models/e5-mistral-7b-instruct \
  --output_dir outputs/table1_e5_mistral_fixed \
  --device cuda \
  --batch_size 4 \
  --max_seq_length 512 \
  --max_distractor_files 50
```

### Table I: Non-LLM Reranking

Runs reranking methods over the retriever candidate pool.

```bash
python experiments/table1_non_llm_reranking_comparison.py \
  --config configs/table1_non_llm_reranking_e5_mistral.yaml \
  --output_dir outputs/table1_non_llm_reranking \
  --device cuda \
  --dense_device cuda
```

### Graph-Assisted LLM Reranking

```bash
python experiments/table2_graph_assisted_llm_reranker.py \
  --config configs/table2_graph_assisted_llm_reranker_qwen25_7b.yaml
```

### Main Generation Experiment

```bash
python experiments/table3_main_generation_qwen25_7b.py \
  --config configs/table3_main_generation_qwen25_7b.yaml
```

## E5-Mistral Retrieval Notes

`feg_rag/retrieval/dense.py` includes a dedicated E5-Mistral path:

- `model_name` containing `e5-mistral` routes to `E5MistralEncoder`.
- Queries use the instruction format:

```text
Instruct: Given a financial question, retrieve relevant evidence passages from SEC filings that directly support the answer.
Query: {query}
```

- Passages use raw text.
- Embeddings are normalized before FAISS `IndexFlatIP` search.
- MiniLM and other non-E5 models keep their original behavior.

## Verification Before a Full Run

Run syntax checks:

```bash
python -m py_compile \
  feg_rag/retrieval/dense.py \
  experiments/table1_initial_retrieval_comparison.py \
  experiments/table1_non_llm_reranking_comparison.py
```

Run smoke tests on the cloud GPU, not on a CPU-only local machine.

## Results

Experiment outputs are not committed to this repository. Save paper-ready results separately and include exact configs, command lines, and logs when reporting results.

Do not overwrite final paper tables with suspect or partially validated historical outputs.
