$ErrorActionPreference = "Stop"

param(
    [string]$PythonBin = "python",
    [string]$VenvDir = ".venv",
    [string]$DataPath = "data/Bank_Transaction_Fraud_Detection.xlsx",
    [string]$OutputDir = "outputs",
    [int]$IidRounds = 20,
    [int]$NonIidRounds = 20,
    [int]$SecurityRounds = 20,
    [double]$Lr = 1e-3,
    [int]$MaxTrainSamples = 60000,
    [int]$NumMalicious = 1,
    [double]$ClipThreshold = 1.0,
    [double]$NoiseMultiplier = 0.01,
    [double]$AttackStrength = 5.0,
    [string]$MaliciousClients = "w3",
    [string]$FlModel = "best_from_ml"
)

Write-Host "==> [1/8] Creating/using virtual environment"
if (!(Test-Path $VenvDir)) {
    & $PythonBin -m venv $VenvDir
}

$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
if (!(Test-Path $VenvPython)) {
    throw "Virtual environment python not found at $VenvPython"
}

Write-Host "==> [2/8] Installing dependencies"
& $VenvPython -m pip install -r requirements.txt
& $VenvPython -m pip install --index-url https://download.pytorch.org/whl/cpu torch

Write-Host "==> [3/8] Preprocessing dataset"
& $VenvPython src/preprocess.py --data_path $DataPath --output_dir $OutputDir

Write-Host "==> [4/8] Centralized baseline"
& $VenvPython src/train_central.py --output_dir $OutputDir

Write-Host "==> [5/8] ML single models + hybrids"
& $VenvPython src/train_ml_hybrids.py --output_dir $OutputDir --max_train_samples $MaxTrainSamples
& $VenvPython src/generate_model_report.py --output_dir $OutputDir --top_k 10

Write-Host "==> [6/8] FL FedAvg IID"
& $VenvPython src/flwr_server.py --output_dir $OutputDir --partition_mode iid --rounds $IidRounds --lr $Lr

Write-Host "==> [7/8] FL FedAvg Non-IID"
& $VenvPython src/flwr_server.py --output_dir $OutputDir --partition_mode noniid --rounds $NonIidRounds --lr $Lr

Write-Host "==> [8/8] Security FL attack evaluation (normal vs attack vs defense)"
& $VenvPython src/flwr_server_security.py `
    --output_dir $OutputDir `
    --partition_mode bank_noniid `
    --rounds $SecurityRounds `
    --lr $Lr `
    --fl_model $FlModel `
    --evaluate_attack_scenarios `
    --attack_mode sign_flip `
    --attack_strength $AttackStrength `
    --malicious_clients $MaliciousClients `
    --num_malicious $NumMalicious `
    --clip_threshold $ClipThreshold `
    --noise_multiplier $NoiseMultiplier

Write-Host ""
Write-Host "Pipeline complete."
Write-Host "Key outputs:"
Write-Host "- $OutputDir/metrics/centralized_results.csv"
Write-Host "- $OutputDir/metrics/ml_single_results.csv"
Write-Host "- $OutputDir/metrics/ml_hybrid_results.csv"
Write-Host "- $OutputDir/metrics/fl_results_iid.csv"
Write-Host "- $OutputDir/metrics/fl_results_noniid.csv"
Write-Host "- $OutputDir/metrics/fl_attack_comparison.csv"
Write-Host "- $OutputDir/metrics/fl_attack_robustness.csv"
Write-Host "- $OutputDir/metrics/model_comparison_report.md"
