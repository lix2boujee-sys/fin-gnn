@echo off
REM ============================================================
REM E5-Mistral-7B-Instruct Full Pipeline (Windows)
REM
REM Steps:
REM   1. Download the model (~14 GB)
REM   2. Run E5-Mistral retrieval (5703 samples)
REM   3. Merge all results into final Table I CSV
REM ============================================================

setlocal

echo.
echo ================================================================
echo   E5-Mistral-7B-Instruct FULL PIPELINE
echo ================================================================
echo.

REM --- Config ---
set MODEL_DIR=cache\models\e5-mistral-7b-instruct
set E5_OUTPUT_DIR=outputs\table1_e5_mistral_fixed
set MERGE_OUTPUT_DIR=outputs\table1_final_merged
set BDH_DIR=outputs\table1_initial_retrieval_comparison_20260712_no_colbert
set COLBERT_DIR=outputs\table1_initial_retrieval_colbert_20260713_full_rebuild

REM ============================================================
REM Step 1: Download model
REM ============================================================
echo [Step 1/3] Checking model ...

if exist "%MODEL_DIR%\model-00001-of-00002.safetensors" (
    echo   Model already downloaded: %MODEL_DIR%
    goto :run_e5
)

echo   Model not found. Downloading (~14 GB) ...
echo.
echo   Choose download method:
echo     [A] HuggingFace Hub (official)
echo     [B] HF Mirror (for China mainland)
echo     [C] Git clone
echo.
set /p METHOD="  Enter A/B/C: "

if /i "%METHOD%"=="A" (
    python -c "from huggingface_hub import snapshot_download; snapshot_download('intfloat/e5-mistral-7b-instruct', local_dir='%MODEL_DIR%')"
)
if /i "%METHOD%"=="B" (
    set HF_ENDPOINT=https://hf-mirror.com
    python -c "from huggingface_hub import snapshot_download; snapshot_download('intfloat/e5-mistral-7b-instruct', local_dir='%MODEL_DIR%')"
)
if /i "%METHOD%"=="C" (
    git lfs install
    git clone https://huggingface.co/intfloat/e5-mistral-7b-instruct %MODEL_DIR%
)

if not exist "%MODEL_DIR%\model-00001-of-00002.safetensors" (
    echo   [ERROR] Model download failed or incomplete.
    echo   Try manual download: see outputs\E5_MISTRAL_RUN_GUIDE.md
    pause
    exit /b 1
)
echo   Model ready.

REM ============================================================
REM Step 2: Run E5-Mistral retrieval
REM ============================================================
:run_e5
echo.
echo [Step 2/3] Running E5-Mistral retrieval (5703 samples) ...
echo   This will take 6-12 hours on CPU.
echo.

python experiments/run_e5_mistral_standalone.py ^
    --config configs/table1_initial_retrieval_comparison.yaml ^
    --output_dir %E5_OUTPUT_DIR% ^
    --model_path %MODEL_DIR% ^
    --device cpu ^
    --batch_size 1 ^
    --overwrite

if %ERRORLEVEL% NEQ 0 (
    echo   [ERROR] E5 retrieval failed!
    pause
    exit /b 1
)
echo   E5 retrieval complete.

REM ============================================================
REM Step 3: Merge results
REM ============================================================
echo.
echo [Step 3/3] Merging all results into final Table I CSV ...

set MERGE_ARGS=--e5_dir %E5_OUTPUT_DIR% --output_dir %MERGE_OUTPUT_DIR%

if exist "%BDH_DIR%" (
    set MERGE_ARGS=%MERGE_ARGS% --bm25_dense_hybrid_dir %BDH_DIR%
) else (
    echo   [WARN] BM25/Dense/Hybrid dir not found: %BDH_DIR%
    echo   Using fallback: exp1_baseline
    set MERGE_ARGS=%MERGE_ARGS% --fallback_exp1_baseline
)

if exist "%COLBERT_DIR%" (
    set MERGE_ARGS=%MERGE_ARGS% --colbert_dir %COLBERT_DIR%
) else (
    echo   [WARN] ColBERTv2 dir not found: %COLBERT_DIR%
)

python experiments/merge_table1_final.py %MERGE_ARGS%

if %ERRORLEVEL% NEQ 0 (
    echo   [ERROR] Merge failed!
    pause
    exit /b 1
)

echo.
echo ================================================================
echo   PIPELINE COMPLETE
echo ================================================================
echo   E5 results:  %E5_OUTPUT_DIR%
echo   Final Table: %MERGE_OUTPUT_DIR%
echo.

pause
endlocal
