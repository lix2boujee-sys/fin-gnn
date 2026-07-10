@echo off
REM FEG-RAG 环境 — 项目内 conda-env + cache
REM 用法: setup_env.bat
REM 注意: 首次使用前请先创建 conda 环境:
REM   conda create -p .\conda-env python=3.10
REM   .\conda-env\python.exe -m pip install -r requirements.txt

set PROJECT_ROOT=%CD%
set CONDA_ENV=%PROJECT_ROOT%\conda-env

set PIP_CACHE_DIR=%PROJECT_ROOT%\cache\pip
set HF_HOME=%PROJECT_ROOT%\cache\huggingface
set TORCH_HOME=%PROJECT_ROOT%\cache\torch
set SENTENCE_TRANSFORMERS_HOME=%PROJECT_ROOT%\cache\sentence_transformers
set XDG_CACHE_HOME=%PROJECT_ROOT%\cache
set TMP=%PROJECT_ROOT%\.tmp
set TEMP=%PROJECT_ROOT%\.tmp

mkdir %PROJECT_ROOT%\cache\pip 2>nul
mkdir %PROJECT_ROOT%\cache\huggingface 2>nul
mkdir %PROJECT_ROOT%\cache\torch 2>nul
mkdir %PROJECT_ROOT%\cache\sentence_transformers 2>nul
mkdir %PROJECT_ROOT%\.tmp 2>nul

set PATH=%CONDA_ENV%;%CONDA_ENV%\Scripts;%CONDA_ENV%\Library\bin;%PATH%

echo ========================================
echo  FEG-RAG 环境 (项目内 conda-env)
echo ========================================
"%CONDA_ENV%\python.exe" -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
echo.
echo 训练: python experiments/train_gnn.py --device cuda --epochs 10 --num_samples 500
echo.

cmd /k
