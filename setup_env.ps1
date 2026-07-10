# FEG-RAG GPU 环境配置 — 全部使用 D:\fin-gnn，不占用 C 盘
# 用法:  .\setup_env.ps1

$ProjectRoot = "D:\fin-gnn"
$CondaEnv    = "$ProjectRoot\conda-env"

# 缓存 / 临时目录全部放 D 盘项目内
$env:PIP_CACHE_DIR              = "$ProjectRoot\cache\pip"
$env:HF_HOME                     = "$ProjectRoot\cache\huggingface"
$env:TORCH_HOME                  = "$ProjectRoot\cache\torch"
$env:SENTENCE_TRANSFORMERS_HOME  = "$ProjectRoot\cache\sentence_transformers"
$env:XDG_CACHE_HOME              = "$ProjectRoot\cache"
$env:TMP                         = "$ProjectRoot\.tmp"
$env:TEMP                        = "$ProjectRoot\.tmp"
$env:TMPDIR                      = "$ProjectRoot\.tmp"

foreach ($dir in @(
    "$ProjectRoot\cache\pip",
    "$ProjectRoot\cache\huggingface",
    "$ProjectRoot\cache\torch",
    "$ProjectRoot\cache\sentence_transformers",
    "$ProjectRoot\.tmp"
)) {
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
}

# 优先使用项目内 conda 环境
$env:PATH = "$CondaEnv;$CondaEnv\Scripts;$CondaEnv\Library\bin;" + $env:PATH

Write-Host "========================================" -ForegroundColor Cyan
Write-Host " FEG-RAG 环境 (D 盘项目内)" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host " Python:  $CondaEnv\python.exe"
Write-Host " 缓存:    $ProjectRoot\cache"
Write-Host ""

& "$CondaEnv\python.exe" -c @"
import torch
print(f' PyTorch: {torch.__version__}')
print(f' CUDA:    {torch.cuda.is_available()}', end='')
if torch.cuda.is_available():
    print(f' ({torch.cuda.get_device_name(0)})')
else:
    print()
"@

Write-Host ""
Write-Host "训练示例:" -ForegroundColor Green
Write-Host "  python experiments/train_gnn.py --device cuda --epochs 10 --num_samples 500"
Write-Host ""
