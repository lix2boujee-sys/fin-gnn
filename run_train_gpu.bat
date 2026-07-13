@echo off
REM 一键 GPU 训练（中等规模：500 样本 / 10 epoch）
call D:\fin-gnn\setup_env.bat >nul 2>&1

set PROJECT_ROOT=D:\fin-gnn
set CONDA_ENV=%PROJECT_ROOT%\conda-env
set PATH=%CONDA_ENV%;%CONDA_ENV%\Scripts;%CONDA_ENV%\Library\bin;%PATH%
set PIP_CACHE_DIR=%PROJECT_ROOT%\cache\pip
set HF_HOME=%PROJECT_ROOT%\cache\huggingface
set TORCH_HOME=%PROJECT_ROOT%\cache\torch
set TMP=%PROJECT_ROOT%\.tmp
set TEMP=%PROJECT_ROOT%\.tmp

cd /d %PROJECT_ROOT%
"%CONDA_ENV%\python.exe" experiments\train_gnn.py --config configs\default.yaml --device cuda --epochs 10 --num_samples 500 --batch_size 16 --retriever bm25
