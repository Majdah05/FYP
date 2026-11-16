#requires -Version 5.1
<#
.SYNOPSIS
Run a six-configuration PathMNIST training sweep with matched plaintext and CKKS runs.
.DESCRIPTION
The paired experiment runner trains both aggregation modes for each combination
of learning rates 0.01, 0.005, and 0.003 and batch sizes 128 and 64. Each of
the 12 training runs uses five clients, five local epochs, and 20 rounds.
.EXAMPLE
.\run_ckks_paired_tuning.ps1
.EXAMPLE
.\run_ckks_paired_tuning.ps1 -DryRun
.EXAMPLE
.\run_ckks_paired_tuning.ps1 -PythonPath .\.venv\Scripts\python.exe
#>
param(
    [string]$ProjectDir = '',
    [string]$PythonPath = 'python',
    [ValidateSet('cuda', 'cpu')][string]$Device = 'cuda',
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
if ([string]::IsNullOrWhiteSpace($ProjectDir)) {
    $ProjectDir = $PSScriptRoot
}
$project = (Resolve-Path -LiteralPath $ProjectDir).ProviderPath
foreach ($required in @('main.py', 'scripts/compare_he.py')) {
    if (!(Test-Path -LiteralPath (Join-Path $project $required) -PathType Leaf)) {
        throw "Select the CKKS project directory that contains $required."
    }
}

$common = @(
    '--dataset', 'pathmnist', '--model', 'cnn', '--aggregation', 'ckks',
    '--distribution', 'non_iid', '--dirichlet_alpha', '0.5', '--num_clients', '5',
    '--client_fraction', '1.0', '--rounds', '20', '--local_epochs', '5',
    '--optimizer', 'sgd', '--momentum', '0.9', '--seed', '42', '--device', $Device,
    '--he_backend', 'tenseal_seal', '--ckks_poly_modulus_degree', '8192',
    '--ckks_coeff_mod_bit_sizes', '60', '40', '40', '60',
    '--ckks_global_scale_power', '40', '--ckks_chunk_size', '4096',
    '--save_checkpoints', '--no-smoke_test'
)

$settings = @(
    @{ Name = '01_reference_lr001_b128'; LearningRate = '0.01'; BatchSize = '128' }
    @{ Name = '02_lr0005_b128'; LearningRate = '0.005'; BatchSize = '128' }
    @{ Name = '03_lr0003_b128'; LearningRate = '0.003'; BatchSize = '128' }
    @{ Name = '04_lr001_b64'; LearningRate = '0.01'; BatchSize = '64' }
    @{ Name = '05_lr0005_b64'; LearningRate = '0.005'; BatchSize = '64' }
    @{ Name = '06_lr0003_b64'; LearningRate = '0.003'; BatchSize = '64' }
)
$batchRoot = Join-Path 'results/ckks_paired_tuning' (Get-Date -Format 'yyyyMMdd_HHmmss_fff')

Push-Location -LiteralPath $project
try {
    foreach ($setting in $settings) {
        $runOutput = Join-Path $batchRoot $setting.Name
        $arguments = @('-m', 'scripts.compare_he', '--output', $runOutput) + $common + @(
            '--lr', $setting.LearningRate, '--batch_size', $setting.BatchSize
        )
        if ($DryRun) {
            [pscustomobject]@{
                Name = $setting.Name
                LearningRate = $setting.LearningRate
                BatchSize = $setting.BatchSize
                Modes = @('plaintext', 'ckks')
                Arguments = $arguments
            }
            continue
        }
        Write-Host "Running plaintext and CKKS for $($setting.Name). Results: $runOutput"
        & $PythonPath -u @arguments
        if ($LASTEXITCODE -ne 0) {
            throw "Pair $($setting.Name) exited with code $LASTEXITCODE. Results so far are in $batchRoot."
        }
    }
} finally {
    Pop-Location
}
