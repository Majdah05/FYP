#requires -Version 5.1
<#
.SYNOPSIS
Run six full-dataset PathMNIST federated experiments with CKKS encryption.
.DESCRIPTION
Runs IID and non-IID, each with seeds 42, 43, 44, sequentially. Uses the
existing main.py and CKKS YAML files. All runs use the full training dataset.
main.py also completes its centralized baseline after each federated run.

Seeds control model initialization, client data partitions and training
randomness such as dropout. The three seeds provide repetitions for reporting
mean and standard deviation. Compare each CKKS run against plaintext with the
same seed and hyperparameters. See docs/ckks_experiment_plan.md for the full
experiment list, defaults and explanation. SEAL keys use separate secure randomness.

Uses .venv\Scripts\python.exe when present, otherwise python on PATH.
Device auto uses CUDA when available; CKKS itself runs on CPU.
Every invocation creates a new batch folder, logs and summary.csv. Failed
experiments are recorded and the remaining runs are attempted. Rerunning starts
a new batch; it does not resume interrupted training. Ctrl+C stops the batch.
.EXAMPLE
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_ckks_experiments.ps1
.EXAMPLE
.\run_ckks_experiments.ps1 -DryRun
.EXAMPLE
.\run_ckks_experiments.ps1 -CheckOnly
.EXAMPLE
.\run_ckks_experiments.ps1 -NumClients 5 -LocalEpochs 3 -BatchSize 64 -LearningRate 0.01 -Device cuda
.EXAMPLE
.\run_ckks_experiments.ps1 -PythonPath 'C:\path\to\python.exe'
#>
[CmdletBinding()]
param(
    [string]$ProjectDir = '',
    [string]$PythonPath = '',
    [ValidateRange(1, 2147483647)][int]$NumClients = 3,
    [ValidateRange(1, 2147483647)][int]$Rounds = 20,
    [ValidateRange(1, 2147483647)][int]$LocalEpochs = 1,
    [ValidateRange(1, 2147483647)][int]$BatchSize = 32,
    [double]$LearningRate = 0.01,
    [ValidateSet('sgd', 'adam')][string]$Optimizer = 'sgd',
    [double]$Momentum = 0.9,
    [double]$DirichletAlpha = 0.5,
    [double]$ClientFraction = 1.0,
    [ValidateSet('auto', 'cpu', 'cuda')][string]$Device = 'auto',
    [string]$OutputRoot = 'results\ckks_six_runs',
    [switch]$DryRun,
    [switch]$CheckOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
if (Test-Path variable:PSNativeCommandUseErrorActionPreference) {
    $PSNativeCommandUseErrorActionPreference = $false
}
if ($DryRun -and $CheckOnly) { throw 'Choose either -DryRun or -CheckOnly.' }
if ([string]::IsNullOrWhiteSpace($ProjectDir)) {
    # Resolve in the body: Windows PowerShell -File can bind param defaults
    # before $PSScriptRoot is populated.
    $ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
}
$ProjectDir = (Resolve-Path -LiteralPath $ProjectDir).ProviderPath
foreach ($required in @('main.py', 'configs\pathmnist_iid_ckks.yaml', 'configs\pathmnist_non_iid_ckks.yaml')) {
    if (-not (Test-Path -LiteralPath (Join-Path $ProjectDir $required) -PathType Leaf)) {
        throw "Missing $required. Put this script beside main.py, or supply -ProjectDir."
    }
}
if ([string]::IsNullOrWhiteSpace($PythonPath)) {
    $projectPython = Join-Path $ProjectDir '.venv\Scripts\python.exe'
    if (Test-Path -LiteralPath $projectPython -PathType Leaf) {
        $PythonPath = $projectPython
    } else {
        $PythonPath = (Get-Command python -CommandType Application -ErrorAction Stop | Select-Object -First 1).Source
    }
} else {
    if (-not [System.IO.Path]::IsPathRooted($PythonPath)) {
        $PythonPath = Join-Path $ProjectDir $PythonPath
    }
    $PythonPath = (Resolve-Path -LiteralPath $PythonPath).ProviderPath
}
if (-not (Test-Path -LiteralPath $PythonPath -PathType Leaf)) {
    throw "Python executable not found: $PythonPath"
}
if (-not [System.IO.Path]::IsPathRooted($OutputRoot)) {
    $OutputRoot = Join-Path $ProjectDir $OutputRoot
}
$OutputRoot = [System.IO.Path]::GetFullPath($OutputRoot)
$batchId = (Get-Date -Format 'yyyyMMdd_HHmmss_fff') + '_' + [guid]::NewGuid().ToString('N').Substring(0, 6)
$batchDir = Join-Path $OutputRoot $batchId
$culture = [System.Globalization.CultureInfo]::InvariantCulture

$plan = @(
    foreach ($distribution in @('iid', 'non_iid')) {
        foreach ($seed in @(42, 43, 44)) {
            $name = "ckks_${distribution}_seed${seed}"
            $runOutput = Join-Path $batchDir $name
            $runArgs = @(
                '-u', 'main.py', '--config', "configs/pathmnist_${distribution}_ckks.yaml",
                '--dataset', 'pathmnist', '--model', 'cnn',
                '--aggregation', 'ckks', '--he_backend', 'tenseal_seal',
                '--no-smoke_test', '--save_checkpoints',
                '--distribution', $distribution, '--seed', [string]$seed,
                '--num_clients', [string]$NumClients, '--rounds', [string]$Rounds,
                '--local_epochs', [string]$LocalEpochs, '--batch_size', [string]$BatchSize,
                '--lr', $LearningRate.ToString('R', $culture), '--optimizer', $Optimizer,
                '--momentum', $Momentum.ToString('R', $culture),
                '--dirichlet_alpha', $DirichletAlpha.ToString('R', $culture),
                '--client_fraction', $ClientFraction.ToString('R', $culture),
                '--device', $Device, '--output_dir', $runOutput
            )
            [pscustomobject]@{
                Name = $name; Distribution = $distribution; Seed = $seed
                OutputDirectory = $runOutput; Arguments = [string[]]$runArgs
            }
        }
    }
)

Write-Host "Project: $ProjectDir"
Write-Host "Python: $PythonPath"
Write-Host "Six full-dataset CKKS runs: IID/non-IID x seeds 42, 43, 44."
Write-Host "Clients=$NumClients; rounds=$Rounds; local epochs=$LocalEpochs; batch=$BatchSize; lr=$LearningRate; optimizer=$Optimizer; device=$Device."
Write-Host 'Each main.py invocation also finishes its existing centralized baseline.'
if ($DryRun) {
    foreach ($run in $plan) {
        $quoted = @($run.Arguments | ForEach-Object { "'" + $_.Replace("'", "''") + "'" })
        Write-Host ("`n{0}:`n& '{1}' {2}" -f $run.Name, $PythonPath.Replace("'", "''"), ($quoted -join ' '))
    }
    Write-Host "`nPreview only. No Python process, training, or output folders were started."
    return
}

# Validate all six effective configurations, dependencies and SEAL TC128
# parameters before a long run. No dataset is downloaded or model trained here.
$preflight = @'
import json
import sys
import torch
import tenseal
import main
from federated_learning.config import parse_config
from federated_learning.encryption.ckks_context import load_seal

plan = json.load(sys.stdin)
seal = load_seal()
validated = set()
for run in plan:
    config = parse_config(run['Arguments'][2:])
    assert config.dataset == 'pathmnist' and config.aggregation == 'ckks'
    assert not config.smoke_test and config.save_checkpoints
    if config.device == 'cuda' and not torch.cuda.is_available():
        raise SystemExit('CUDA was requested but is unavailable in this Python. Use -Device auto/cpu, or a CUDA-enabled environment.')
    key = (config.ckks_poly_modulus_degree, tuple(config.ckks_coeff_mod_bit_sizes))
    if key not in validated:
        parms = seal.EncryptionParameters(seal.SCHEME_TYPE.CKKS)
        parms.set_poly_modulus_degree(key[0])
        parms.set_coeff_modulus(seal.CoeffModulus.Create(key[0], list(key[1])))
        context = seal.SEALContext(parms, True, seal.SEC_LEVEL_TYPE.TC128)
        if not context.parameters_set():
            raise SystemExit('Invalid SEAL TC128 parameters: ' + context.parameters_error_message())
        validated.add(key)
print('Python:', sys.executable)
print('PyTorch:', torch.__version__, '| TenSEAL:', tenseal.__version__)
if torch.cuda.is_available() and config.device != 'cpu':
    print('Training GPU:', torch.cuda.get_device_name(0))
    print('GPU operation check:', torch.ones(2, device='cuda').sum().item())
else:
    print('Training device: CPU')
print('CKKS device: CPU; all 6 configurations and SEAL TC128 parameters passed.')
'@

Push-Location -LiteralPath $ProjectDir
try {
    $planJson = ConvertTo-Json -InputObject $plan -Depth 6
    $environmentInfo = @($planJson | & $PythonPath -u -c $preflight)
    if ($LASTEXITCODE -ne 0) {
        throw 'Preflight failed. No training started. Install requirements.txt and requirements-he.txt in the selected Python environment, and check the error above.'
    }
    $environmentInfo | ForEach-Object { Write-Host $_ }
    if ($CheckOnly) {
        Write-Host 'Checks completed. No training or batch output folders were started.'
        return
    }

    New-Item -ItemType Directory -Path $batchDir | Out-Null
    $logDir = Join-Path $batchDir 'logs'
    New-Item -ItemType Directory -Path $logDir | Out-Null
    $environmentInfo | Out-File -LiteralPath (Join-Path $batchDir 'environment.txt') -Encoding utf8
    $planJson | Out-File -LiteralPath (Join-Path $batchDir 'run_plan.json') -Encoding utf8
    $summaryPath = Join-Path $batchDir 'summary.csv'
    $summary = @()
    Write-Host "Batch output: $batchDir"

    for ($index = 0; $index -lt $plan.Count; $index++) {
        $run = $plan[$index]
        $runArgs = $run.Arguments
        $logPath = Join-Path $logDir ($run.Name + '.log')
        Write-Host ("`n[{0}/6] {1}" -f ($index + 1), $run.Name)
        $timer = [System.Diagnostics.Stopwatch]::StartNew()
        $runExitCode = 1
        $failureDetail = ''
        # Capture both native streams on Windows PowerShell 5.1 as well as PS7.
        $global:LASTEXITCODE = 1
        $ErrorActionPreference = 'Continue'
        try {
            & $PythonPath @runArgs 2>&1 | ForEach-Object {
                $line = $_.ToString()
                Add-Content -LiteralPath $logPath -Value $line -Encoding utf8
                Write-Host $line
            }
            $runExitCode = $LASTEXITCODE
        } catch {
            $failureDetail = $_.Exception.Message
            Add-Content -LiteralPath $logPath -Value $failureDetail -Encoding utf8
        } finally {
            $ErrorActionPreference = 'Stop'
            $timer.Stop()
        }
        $experimentDir = Join-Path $run.OutputDirectory 'experiment_001'
        if ($runExitCode -eq 0) {
            try {
                $history = Get-Content -LiteralPath (Join-Path $experimentDir 'federated_history.json') -Raw | ConvertFrom-Json
                $roundsSaved = @($history.history)
                if ($history.config.aggregation -ne 'ckks' -or $history.config.smoke_test -or
                    $roundsSaved.Count -ne $Rounds -or
                    @($roundsSaved | Where-Object { $_.he_ciphertext_count -le 0 }).Count -gt 0) {
                    throw 'Saved results do not contain the expected full-dataset CKKS rounds.'
                }
            } catch {
                $runExitCode = 2
                $failureDetail = 'Result validation failed: ' + $_.Exception.Message
                Add-Content -LiteralPath $logPath -Value $failureDetail -Encoding utf8
            }
        }
        $status = if ($runExitCode -eq 0) { 'Completed' } else { 'Failed' }
        $summary += [pscustomobject]@{
            Experiment = $run.Name; Distribution = $run.Distribution; Seed = $run.Seed
            Aggregation = 'ckks'; Status = $status; ExitCode = $runExitCode
            Clients = $NumClients; Rounds = $Rounds; LocalEpochs = $LocalEpochs
            BatchSize = $BatchSize; LearningRate = $LearningRate; Optimizer = $Optimizer
            ElapsedSeconds = [math]::Round($timer.Elapsed.TotalSeconds, 2)
            OutputDirectory = $experimentDir; LogFile = $logPath; FailureDetail = $failureDetail
        }
        $summary | Export-Csv -LiteralPath $summaryPath -NoTypeInformation -Encoding utf8
        Write-Host "$status (exit $runExitCode). Summary: $summaryPath"
        if ($runExitCode -ne 0) {
            Write-Warning "Run failed. Continuing with the next experiment. Details: $logPath"
        }
    }
    $summary | Format-Table Experiment, Status, ExitCode, ElapsedSeconds -AutoSize | Out-Host
    $failed = @($summary | Where-Object { $_.ExitCode -ne 0 })
    if ($failed.Count -gt 0) {
        throw "$($failed.Count) of 6 runs failed. All 6 were attempted. Inspect $summaryPath"
    }
    Write-Host "All six encrypted PathMNIST experiments completed. Results: $batchDir"
} finally {
    Pop-Location
}
