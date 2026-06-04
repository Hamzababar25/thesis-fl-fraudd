# Federated Fraud Detection Security Evaluation (Flower + PyTorch)

This project implements a leakage-safe fraud detection pipeline and extends it to
security-focused federated learning evaluation.

- Dataset: `data/Bank_Transaction_Fraud_Detection.xlsx`
- Target: `Is_Fraud` (binary, imbalanced)
- FL clients: `w1`, `w2`, `w3`
- Core security focus: poisoning defense, robust aggregation, clipping, and DP noise

## Security-Focused Contributions

- FedAvg baseline retained for reference.
- Multi-Krum robust aggregation added for poisoning attack defense.
- Client-side protections before sending model updates:
  - gradient clipping,
  - differential privacy Gaussian noise.
- Poisoning attack simulation with configurable malicious clients and attack strength.
- Scenario-based evaluation:
  - normal FL,
  - FL under attack,
  - FL with defense.

## Project Structure

- `src/preprocess.py`: preprocessing and feature engineering
- `src/train_central.py`: centralized baseline training and evaluation
- `src/train_ml_hybrids.py`: single-model and hybrid ML comparison
- `src/flwr_client.py`: Flower client with clipping, DP, and optional poisoning
- `src/aggregation.py`: FedAvg and Multi-Krum aggregation module
- `src/attack_simulation.py`: attack utilities, comparison tables, and plots
- `src/flwr_server.py`: baseline FL server (FedAvg)
- `src/flwr_server_security.py`: security evaluation server pipeline
- `src/common.py`: shared model, metrics, and plotting helpers
- `outputs/`: generated artifacts

### Best-Model-to-FL Alignment

The security FL pipeline now supports deploying the selected best centralized model in FL:
- `--fl_model best_from_ml` (default): reads `outputs/metrics/ml_comparison_summary.json`
- currently mapped to `logistic_regression` (single-layer logistic FL model)
- optional override: `--fl_model logistic_regression` or `--fl_model mlp`

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install --index-url https://download.pytorch.org/whl/cpu torch
```

## 1) Preprocess Data

```bash
python3 src/preprocess.py --data_path data/Bank_Transaction_Fraud_Detection.xlsx --output_dir outputs
```

Highlights:
- Drops identity-like columns and unnamed columns.
- Detects amount/balance and datetime columns.
- Adds financial and temporal engineered features.
- Stratified split and train-only preprocessor fitting.

## 2) Centralized Baselines

Centralized logistic regression:

```bash
python3 src/train_central.py --output_dir outputs
```

ML single + hybrid comparison:

```bash
python3 src/train_ml_hybrids.py --output_dir outputs
```

## 3) Baseline FL (FedAvg)

IID:

```bash
python3 src/flwr_server.py --output_dir outputs --partition_mode iid --rounds 20 --lr 1e-3
```

Non-IID:

```bash
python3 src/flwr_server.py --output_dir outputs --partition_mode noniid --rounds 20 --lr 1e-3
```

## 4) Security-Focused FL Run (Single Strategy)

Run FedAvg or Multi-Krum with clipping and DP enabled:

```bash
python3 src/flwr_server_security.py \
  --output_dir outputs \
  --partition_mode bank_noniid \
  --rounds 20 \
  --lr 1e-3 \
  --fl_model best_from_ml \
  --aggregation_method multi_krum \
  --num_malicious 1 \
  --clip_threshold 1.0 \
  --noise_multiplier 0.01
```

## 5) Compare FedAvg vs Multi-Krum

```bash
python3 src/flwr_server_security.py \
  --output_dir outputs \
  --partition_mode bank_noniid \
  --fl_model best_from_ml \
  --compare_strategies \
  --num_malicious 1 \
  --clip_threshold 1.0 \
  --noise_multiplier 0.01
```

Generates:
- `outputs/metrics/fl_strategy_comparison.csv`
- `outputs/metrics/fl_strategy_comparison.json`

## 6) Full Attack Evaluation (Recommended)

This executes:
1) normal FL (no attack, no defense),  
2) FL under attack,  
3) FL with defense (Multi-Krum + clipping + DP).

```bash
python3 src/flwr_server_security.py \
  --output_dir outputs \
  --partition_mode bank_noniid \
  --fl_model best_from_ml \
  --evaluate_attack_scenarios \
  --attack_mode sign_flip \
  --attack_strength 5.0 \
  --malicious_clients w3 \
  --num_malicious 1 \
  --clip_threshold 1.0 \
  --noise_multiplier 0.01
```

## Key Security Configuration Flags

- `--aggregation_method {fedavg,multi_krum}`
- `--fl_model {best_from_ml,logistic_regression,mlp}`
- `--num_malicious <int>`
- `--multi_krum_m <int>` (0 means automatic `n_clients - num_malicious`)
- `--attack_enabled`
- `--attack_mode {sign_flip,scale}`
- `--attack_strength <float>`
- `--malicious_clients <csv>` (example: `w2,w3`)
- `--clip_threshold <float>`
- `--noise_multiplier <float>`

Backward-compatible aliases are still accepted:
- `--clip_norm` (alias of `--clip_threshold`)
- `--dp_noise_std` (alias of `--noise_multiplier`)

## Important Output Files

### Security Runs
- `outputs/metrics/fl_round_metrics_<label>.csv`
- `outputs/metrics/fl_round_client_weights_<label>.csv`
- `outputs/metrics/fl_final_metrics_<label>.json`
- `outputs/metrics/fl_results_<label>.csv`
- `outputs/metrics/fl_aggregation_summary_<label>.json`

### Attack Evaluation
- `outputs/metrics/fl_attack_comparison.csv`
- `outputs/metrics/fl_attack_comparison.json`
- `outputs/metrics/fl_attack_robustness.csv`
- `outputs/plots/fl_attack_evaluation_quality_metrics.png`
- `outputs/plots/fl_attack_evaluation_aggregation_robustness.png`

### Centralized + ML Baselines
- `outputs/metrics/centralized_metrics.json`
- `outputs/metrics/ml_single_results.csv`
- `outputs/metrics/ml_hybrid_results.csv`
- `outputs/metrics/ml_comparison_summary.json`

## Logged Security Signals

Client-side logs include:
- gradient norm before clipping,
- gradient norm after clipping,
- clip threshold,
- noise multiplier,
- noise scale used,
- malicious flag and attack strength (if attacked client).

Server-side logs include:
- robust aggregation scores,
- selected/rejected clients per round,
- round-level rejection summary for robustness analysis.

## Notes on Leakage and Imbalance

- Split is performed before any balancing.
- No SMOTE on validation/test.
- Class imbalance handled with class-weighted losses.
- Preprocessor fit is train-only.

## Python Version

- Linux, Python `3.10+`
