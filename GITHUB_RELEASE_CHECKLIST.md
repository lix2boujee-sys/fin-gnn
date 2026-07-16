# GitHub Release Checklist

Recommended upload asset:

- `feg-rag-github-release-20260716.zip`

Included:

- Source code under `feg_rag/`
- Experiment scripts under `experiments/`
- Config files under `configs/`
- Setup and helper scripts
- README, release notes, requirements, and selected tests

Excluded from the zip:

- `__pycache__/`, `*.pyc`
- `cache/`, `outputs/`
- `FinDER/`, `10-k/`
- model weights, checkpoints, parquet files, and local environment files

Before publishing:

1. Add a license if this will be public.
2. Create the GitHub repository from the contents of this directory, or upload the zip as a Release asset.
3. Keep datasets and model files out of git; use the README download/setup steps instead.
