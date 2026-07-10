# FEG-RAG

Financial Evidence Graph RAG — retrieval and graph-based reranking on the [FinDER](https://huggingface.co/datasets) financial QA benchmark.

Builds a **Financial Evidence Graph** (company / filing / section / chunk / metric / year) and compares plain retrieval, PPR reranking, and GNN rerankers (GraphSAGE / R-GCN).

## Experiments

| Exp | Script | Question |
|-----|--------|----------|
| **Exp1** | `experiments/exp1_retrieval_baseline.py` | Plain retrieval baseline (BM25 / Dense / Hybrid) |
| **Exp2** | `experiments/exp2_error_analysis.py` | Are retrieval failures financial-structure errors? |
| **Exp3** | `experiments/exp3_feg_ppr.py` | Does graph + PPR improve ranking without training? |
| **Exp4** | `experiments/exp4_gnn_reranker.py` | Does a trained GNN reranker beat PPR / Hybrid? |

Additional entry points: `experiments/train_gnn.py` (standalone GNN training), `run_pipeline.py`, `experiments/runner.py`.

## Results (FinDER full, 5703 samples)

### Exp1 — Retrieval baseline (test set)

| Method | Recall@10 | MRR |
|--------|-----------|-----|
| BM25 | 0.167 | 0.131 |
| Dense | 0.198 | 0.145 |
| **Hybrid** | **0.244** | **0.184** |

### Exp4 — GNN reranker (validation split, GraphSAGE 50 epoch)

| Method | Recall@10 | MRR |
|--------|-----------|-----|
| Hybrid | 0.256 | 0.184 |
| Hybrid + PPR | 0.256 | 0.097 |
| Hybrid + GraphSAGE | 0.191 | 0.106 |

Training loss: 0.48 → 0.27 (−43%). Artifacts: `outputs/exp4_gnn_reranker/`.

> Full metrics and per-query JSONL live under `outputs/` (gitignored). Re-run experiments to reproduce.

## Quick start

### 1. Environment

```powershell
# Create conda env (one-time)
conda create -p .\conda-env python=3.10
.\conda-env\python.exe -m pip install -r requirements.txt

# Activate environment
.\setup_env.ps1          # (Windows PowerShell)
```

Cache / temp env vars (to avoid filling system drive):

```powershell
$env:HF_HOME='.\cache\huggingface'
$env:PIP_CACHE_DIR='.\cache\pip'
$env:TMP='.\.tmp'
$env:TEMP='.\.tmp'
```

### 2. Data

```powershell
python download_finder.py   # FinDER parquet → FinDER/data/
python extract_10k.py         # optional 10-K distractors → 10-k/
python check_finder.py
```

Download dense model locally (or point `configs/default.yaml` → `retrieval.dense_model`):

```
cache/models/all-MiniLM-L6-v2/
```

### 3. Run experiments

```powershell
# Exp1 — retrieval baseline (full)
python experiments/exp1_retrieval_baseline.py --num_samples 0 --top_k 10

# Exp2 — error analysis (requires Exp1 outputs)
python experiments/exp2_error_analysis.py

# Exp3 — graph + PPR
python experiments/exp3_feg_ppr.py --num_samples 0

# Exp4 — GNN reranker (Dense on CPU, GNN on CUDA; safe for 4 GB GPU)
python experiments/exp4_gnn_reranker.py --num_samples 0 --epochs 50 --no_ablation

# Standalone GNN training
python experiments/train_gnn.py --device cuda --epochs 50 --num_samples 0
```

Smoke / sanity:

```powershell
python experiments/exp4_gnn_reranker.py --sanity --device cuda
```

## Project layout

```
feg_rag/              # core library (retrieval, graph, rerank, eval)
experiments/          # Exp1–4 scripts
configs/default.yaml  # paths and hyperparameters
FinDER/data/          # dataset (not in repo)
10-k/                 # EDGAR distractors (not in repo)
cache/                # models & HF cache (not in repo)
outputs/              # experiment artifacts (not in repo)
conda-env/            # local Python env (not in repo)
```

## Hardware notes

- **RTX 3050 Ti (4 GB)**: use `--dense_device cpu` (Exp4 default) for MiniLM encoding; keep `--device cuda` for GNN training.
- Full Exp4 on 5703 samples ≈ 1–1.5 h (mostly CPU dense encode + PPR eval).

## Requirements

```powershell
pip install -r requirements.txt
```

PyTorch with CUDA is installed separately in `conda-env` (see `setup_env.ps1`).

## Citation & docs

- Experiment specs: `finder_exp1_baseline_instruction.md`, `finder_all_experiments_instruction.md`
- Design notes: `financial_graph_rag_experiment_design_cn.md`, `financial_graph_rag_paper_plan_cn.md`

## License

Research / academic use. Add your license before public release.
