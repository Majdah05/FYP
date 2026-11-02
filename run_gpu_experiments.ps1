#requires -Version 5.1
<#
.SYNOPSIS
Runs the ten PathMNIST GPU experiments sequentially, with 20 rounds each.
.DESCRIPTION
Copy this file into FederatedLearning, beside main.py, and run:
  powershell -NoProfile -ExecutionPolicy Bypass -File .\run_gpu_experiments.ps1

Uses .venv\Scripts\python.exe unless -PythonPath specifies another interpreter.
The selected Python must have the project dependencies and CUDA-enabled PyTorch.
Each main.py invocation includes its existing centralized baseline (100 epochs
for 5 local epochs, or 60 epochs for 3 local epochs), before the next run starts.
Failed experiments are reported and the remaining experiments still run.
Press Ctrl+C to stop. Re-running starts new experiments; it does not resume them.
.EXAMPLE
.\run_gpu_experiments.ps1 -ProjectDir 'D:\FederatedLearning'
.EXAMPLE
.\run_gpu_experiments.ps1 -DryRun
#>
[CmdletBinding()]
param(
    [string]$ProjectDir = $PSScriptRoot,
    [string]$PythonPath = '',
    [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
# PowerShell 7 can convert native nonzero exit codes to terminating errors.
# Handle those explicitly so that a failed experiment does not end the batch.
if (Test-Path variable:PSNativeCommandUseErrorActionPreference) {
    $PSNativeCommandUseErrorActionPreference = $false
}

$ProjectDir = (Resolve-Path -LiteralPath $ProjectDir).ProviderPath
if (-not (Test-Path -LiteralPath (Join-Path $ProjectDir 'main.py') -PathType Leaf)) {
    throw 'main.py was not found. Copy this file beside main.py, or pass -ProjectDir with your FederatedLearning folder.'
}
if ([string]::IsNullOrWhiteSpace($PythonPath)) {
    $PythonPath = Join-Path $ProjectDir '.venv\Scripts\python.exe'
} elseif (-not [System.IO.Path]::IsPathRooted($PythonPath)) {
    $PythonPath = Join-Path $ProjectDir $PythonPath
}
if (-not $DryRun) {
    if (-not (Test-Path -LiteralPath $PythonPath -PathType Leaf)) {
        throw "Python was not found at '$PythonPath'. Create the project .venv or pass -PythonPath with your Python executable."
    }
    $PythonPath = (Resolve-Path -LiteralPath $PythonPath).ProviderPath
}

$resultsRoot = 'results/university_gpu_tuning_20_rounds'
$experiments = @(
    @{ Name = 'T01_iid_lr0003_e5_b64';      Distribution = 'iid';     LR = '0.003'; Epochs = 5; Batch = 64 }
    @{ Name = 'T02_noniid_lr0003_e5_b64';   Distribution = 'non_iid'; LR = '0.003'; Epochs = 5; Batch = 64 }
    @{ Name = 'T03_iid_lr0005_e5_b64';      Distribution = 'iid';     LR = '0.005'; Epochs = 5; Batch = 64 }
    @{ Name = 'T04_noniid_lr0005_e5_b64';   Distribution = 'non_iid'; LR = '0.005'; Epochs = 5; Batch = 64 }
    @{ Name = 'T05_iid_lr001_e3_b64';       Distribution = 'iid';     LR = '0.01';  Epochs = 3; Batch = 64 }
    @{ Name = 'T06_noniid_lr001_e3_b64';    Distribution = 'non_iid'; LR = '0.01';  Epochs = 3; Batch = 64 }
    @{ Name = 'T07_iid_lr001_e5_b32';       Distribution = 'iid';     LR = '0.01';  Epochs = 5; Batch = 32 }
    @{ Name = 'T08_noniid_lr001_e5_b32';    Distribution = 'non_iid'; LR = '0.01';  Epochs = 5; Batch = 32 }
    @{ Name = 'T09_iid_lr001_e5_b128';      Distribution = 'iid';     LR = '0.01';  Epochs = 5; Batch = 128 }
    @{ Name = 'T10_noniid_lr001_e5_b128';   Distribution = 'non_iid'; LR = '0.01';  Epochs = 5; Batch = 128 }
)

$commonArgs = @(
    '-u', '.\main.py',
    '--dataset', 'pathmnist', '--model', 'cnn',
    '--num_clients', '5', '--rounds', '20',
    '--optimizer', 'sgd', '--momentum', '0.9',
    '--dirichlet_alpha', '0.5', '--client_fraction', '1.0',
    '--seed', '42', '--device', 'cuda', '--save_checkpoints'
)

$summary = @()
$transcriptStarted = $false
Push-Location -LiteralPath $ProjectDir
try {
    if (-not $DryRun) {
        # Verify real GPU computation once before starting any training.
        $gpuCheck = @'
import torch
print('PyTorch:', torch.__version__)
print('CUDA build:', torch.version.cuda)
if not torch.cuda.is_available():
    raise SystemExit('CUDA unavailable. Install CUDA-enabled PyTorch in this Python environment before running the batch.')
print('GPU:', torch.cuda.get_device_name(0))
x = torch.ones(2, device='cuda')
print('GPU tensor check:', (x * x).sum().item())
'@
        $gpuInfo = @(& $PythonPath -c $gpuCheck)
        if ($LASTEXITCODE -ne 0) {
            throw 'The CUDA check failed. No experiments were started.'
        }
        $batchId = (Get-Date -Format 'yyyyMMdd_HHmmss_fff') + '_' + [guid]::NewGuid().ToString('N').Substring(0, 6)
        $logDir = Join-Path $ProjectDir "$resultsRoot/batch_logs/$batchId"
        New-Item -ItemType Directory -Path $logDir -Force | Out-Null
        $summaryPath = Join-Path $logDir 'summary.csv'
        $gpuInfo | Out-File -LiteralPath (Join-Path $logDir 'gpu_environment.txt') -Encoding utf8
        Start-Transcript -Path (Join-Path $logDir 'console.log') | Out-Null
        $transcriptStarted = $true
        $gpuInfo | ForEach-Object { Write-Host $_ }
        Write-Host "Batch logs: $logDir"
    }

    Write-Host "Project: $ProjectDir"
    Write-Host 'Running T01 through T10 in order, with 20 rounds each.'
    Write-Host 'Each run also completes its centralized baseline before the next run starts.'
    for ($index = 0; $index -lt $experiments.Count; $index++) {
        $experiment = $experiments[$index]
        $outputDir = "$resultsRoot/$($experiment.Name)"
        $runArgs = $commonArgs + @(
            '--local_epochs', [string]$experiment.Epochs,
            '--batch_size', [string]$experiment.Batch,
            '--lr', $experiment.LR,
            '--distribution', $experiment.Distribution,
            '--output_dir', $outputDir
        )
        Write-Host "`n[$($index + 1)/$($experiments.Count)] $($experiment.Name)"
        Write-Host ('& "{0}" {1}' -f $PythonPath, ($runArgs -join ' '))
        if ($DryRun) { continue }

        $timer = [System.Diagnostics.Stopwatch]::StartNew()
        # Wait for main.py and capture both streams in the transcript. Windows
        # PowerShell 5.1 otherwise omits native output from transcript files.
        # Its stderr records must not terminate the batch before we read the exit code.
        $global:LASTEXITCODE = 1
        $ErrorActionPreference = 'Continue'
        try {
            & $PythonPath @runArgs 2>&1 | ForEach-Object { Write-Host $_ }
            $runExitCode = $LASTEXITCODE
        } finally {
            $ErrorActionPreference = 'Stop'
        }
        $timer.Stop()
        $status = if ($runExitCode -eq 0) { 'Completed' } else { 'Failed' }
        $summary += [pscustomobject]@{
            Experiment = $experiment.Name
            Status = $status
            ExitCode = $runExitCode
            Rounds = 20
            ElapsedSeconds = [math]::Round($timer.Elapsed.TotalSeconds, 2)
            OutputDirectory = $outputDir
        }
        $summary | Export-Csv -LiteralPath $summaryPath -NoTypeInformation -Encoding utf8
        Write-Host "$status (exit code $runExitCode)."
        if ($runExitCode -ne 0) {
            Write-Warning "$($experiment.Name) failed. Continuing with the next experiment. See console.log for details."
        }
    }

    if ($DryRun) {
        Write-Host "`nPreview finished. No training ran and no results were written."
    } else {
        $summary | Format-Table Experiment, Status, ExitCode, ElapsedSeconds -AutoSize | Out-Host
        Write-Host "Summary saved to: $summaryPath"
        $failed = @($summary | Where-Object { $_.ExitCode -ne 0 })
        if ($failed.Count -gt 0) {
            throw "$($failed.Count) of 10 experiments failed. All 10 were attempted; inspect the summary and console log."
        }
        Write-Host 'All 10 experiments completed successfully.'
    }
} finally {
    if ($transcriptStarted) { Stop-Transcript | Out-Null }
    Pop-Location
}