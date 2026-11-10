<#
Paired plaintext vs CKKS PathMNIST experiments (T09/T10 configuration), then
the HE-only cost benchmark, analysis and the HTML report.

Run from anywhere:
  powershell -NoProfile -ExecutionPolicy Bypass -File .\experiments\ckks_vs_plaintext\run_experiments.ps1
Options:
  -Device cuda|cpu|auto   (default auto)
  -Threads 2              (CPU threads for PyTorch)
  -SkipBench              (skip the HE-only benchmark)
  -PythonPath C:\path\python.exe
Finished runs are skipped, so the script can be re-run after an interruption.
#>
param(
    [string]$Device = 'auto',
    [int]$Threads = 2,
    [switch]$SkipBench,
    [string]$PythonPath = ''
)
$ErrorActionPreference = 'Stop'
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
$Repo = (Resolve-Path (Join-Path $Here '..\..')).Path
if (-not $PythonPath) {
    $venv = Join-Path $Repo '.venv\Scripts\python.exe'
    $PythonPath = if (Test-Path $venv) { $venv } else { 'python' }
}
$Out = Join-Path $Here 'results'
$Runs = Join-Path $Out 'runs'
$Bench = Join-Path $Out 'he_cost_benchmark'
New-Item -ItemType Directory -Force -Path $Runs, $Bench | Out-Null
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'

Write-Host "Repo: $Repo`nPython: $PythonPath"
& $PythonPath -c "import tenseal, torch; from tenseal import sealapi; print('tenseal', tenseal.__version__, 'torch', torch.__version__, 'cuda', torch.cuda.is_available())"
if ($LASTEXITCODE) { throw 'Install dependencies first: python -m pip install -r requirements.txt -r requirements-he.txt' }

foreach ($dist in 'iid', 'non_iid') {
    foreach ($mode in 'plaintext', 'ckks') {
        $dir = Join-Path $Runs "${dist}_$mode"
        if (Test-Path (Join-Path $dir 'federated_history.json')) { Write-Host "skip (done): $dir"; continue }
        if (Test-Path $dir) { Remove-Item -Recurse -Force $dir }
        Write-Host "=== $dist / $mode ===" -ForegroundColor Cyan
        $t = Get-Date
        & $PythonPath -I (Join-Path $Here 'run_fl.py') --repo $Repo --data_dir (Join-Path $Repo 'data') --download `
            --out $dir --mode $mode --distribution $dist --device $Device --threads $Threads `
            --num_clients 5 --rounds 20 --local_epochs 5 --batch_size 128 --lr 0.01 --seed 42
        if ($LASTEXITCODE) { throw "Run $dist/$mode failed" }
        Write-Host ("finished in {0:N1} min" -f ((Get-Date) - $t).TotalMinutes)
    }
}

if (-not $SkipBench) {
    $S = Join-Path $Here 'he_cost_bench.py'
    Push-Location $Repo
    & $PythonPath -I $S --repo . --out (Join-Path $Bench 'he_clients.json') --label client_scaling --clients 2 3 5 10 --repeats 3
    & $PythonPath -I $S --repo . --out (Join-Path $Bench 'he_params.json') --label param_ablation --clients 5 --repeats 3 `
        --params '8192:60,40,40,60:40' '8192:60,40,60:40' '16384:60,40,40,60:40' '8192:60,30,60:30'
    Pop-Location
}

& $PythonPath -I (Join-Path $Here 'analyze.py') $Runs $Bench (Join-Path $Out 'summary.json')
& $PythonPath -I (Join-Path $Here 'make_report.py') (Join-Path $Out 'summary.json') (Join-Path $Out 'ckks_vs_plaintext_report.html')
Write-Host "Report: $(Join-Path $Out 'ckks_vs_plaintext_report.html')" -ForegroundColor Green
