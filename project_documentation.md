# Federated Learning Security Evaluation for Bank Fraud Detection
## A Privacy-Preserving and Attack-Resilient FL Pipeline

---

## 1. Project Overview

This project implements a privacy-preserving fraud detection system using Federated Learning (Flower + PyTorch), then extends it into a security evaluation framework for adversarial settings.

The system keeps the original data and ML workflow intact while shifting FL research emphasis to:
- poisoning attack simulation,
- robust aggregation (Multi-Krum),
- client-side gradient clipping,
- differential privacy noise injection.

It now follows a strict methodology:
**ML comparison -> best model selection -> deploy best model in FL -> attack/defense evaluation**.

### Core Objective
Evaluate how much robustness improves when moving from standard FedAvg to a defended FL pipeline under malicious client behavior.

---

## 2. What Is Preserved vs Refactored

### Preserved
- Preprocessing and feature-engineering pipeline
- Centralized and hybrid ML baselines
- Flower simulation architecture
- `bank_noniid` partitioning
- Metrics and plotting workflow compatibility

### Refactored
- Removed adaptive fraud-aware weighting logic
- Added separate robust aggregation module
- Added explicit attack simulation module
- Added client-side protection before update transmission
- Added scenario-based attack evaluation runbook

---

## 3. Security Threat Model

### Threats Considered
1. **Poisoning / Byzantine Updates**  
   Malicious clients send manipulated model updates to degrade global training.

2. **Gradient Reconstruction / Model Inversion Risk**  
   Adversaries attempt to infer sensitive client data from transmitted updates.

### Security Controls Implemented
1. **Robust Aggregation (Server-side)**
   - FedAvg baseline
   - Multi-Krum defense to reject outlier updates

2. **Protected Client Updates (Client-side)**
   - Global gradient/update clipping
   - Gaussian DP noise after clipping

3. **Attack Simulation**
   - Configurable malicious clients
   - Configurable attack strength and mode
   - Scenario-level comparison outputs

---

## 4. Data and Preprocessing

### Dataset
- Source: `data/Bank_Transaction_Fraud_Detection.xlsx`
- Target: `Is_Fraud` (binary)
- Total samples: 200,000
- Fraud ratio: ~5.04%

### Split
- Train/Val/Test = 70% / 15% / 15% (stratified)

### Feature Pipeline
- Identity-like column removal
- Date/time, amount, and balance detection
- Financial and temporal feature engineering
- Train-only preprocessor fit (anti-leakage)
- Final feature space: 521 (11 numeric + OHE categorical)

---

## 5. Model and FL Architecture

### FraudMLP (PyTorch)
```
Input (521)
 -> Dense(64) + ReLU + Dropout(0.2)
 -> Dense(32) + ReLU + Dropout(0.2)
 -> Output(1)
```

### Baselines
- Logistic Regression
- Random Forest
- XGBoost
- Hybrid ensembles

### FL Setup
- Flower simulation
- 3 clients: `w1`, `w2`, `w3`
- Local epochs per round: 1
- Batch size: 128
- Optimizer: Adam
- Partitioning: `iid`, `noniid`, `bank_noniid` (recommended for realism)
- FL model selection:
  - `best_from_ml` (default, reads best model from ML summary)
  - currently resolved to `logistic_regression`
  - optional manual override to `mlp`

---

## 6. Robust Aggregation Module

New module: `src/aggregation.py`

### Aggregation Strategies
1. **FedAvg (baseline)**
   - weighted by number of local examples

2. **Multi-Krum (defense)**
   - computes pairwise distances between client updates
   - scores each client using nearest-neighbor distances
   - keeps low-score clients, rejects suspected outliers
   - aggregates only selected clients

### Outlier Traceability
Server logs:
- selected clients
- rejected clients
- per-client aggregation score

---

## 7. Client-Side Security Protections

Updated client: `src/flwr_client.py`

### Protection Sequence
1. Train local model
2. Compute update delta (`updated - initial`)
3. Compute gradient norm (before clipping)
4. Clip by configurable threshold
5. Add Gaussian DP noise to clipped update
6. Send protected update

### Configurable Parameters
- `clip_threshold`
- `noise_multiplier`

### Logged Signals
- `gradient_norm_before_clipping`
- `gradient_norm_after_clipping`
- `clip_threshold`
- `noise_multiplier`
- `noise_scale_used`

---

## 8. Attack Simulation Module

New module: `src/attack_simulation.py`

### Attack Capabilities
- Select malicious clients (`w2,w3` etc.)
- Set attack strength
- Attack mode:
  - `sign_flip` (reverse and amplify update)
  - `scale` (magnify update)

### Client Attack Behavior
If client is malicious and attack enabled:
- local update is manipulated before protection/transmission

---

## 9. Evaluation Scenarios

The server supports three key security evaluation scenarios:

1. **Normal FL**
   - no attack
   - no defense

2. **FL Under Attack**
   - attack enabled
   - baseline FedAvg
   - no clipping/DP defense

3. **FL With Defense**
   - attack enabled
   - Multi-Krum + clipping + DP noise

This setup enables direct robustness comparisons under identical data partitions.

---

## 10. Execution Commands

### Step 1: Preprocessing
```bash
python3 src/preprocess.py --data_path data/Bank_Transaction_Fraud_Detection.xlsx --output_dir outputs
```

### Step 2: Centralized Baselines
```bash
python3 src/train_central.py --output_dir outputs
python3 src/train_ml_hybrids.py --output_dir outputs
```

### Step 3: FL Strategy Comparison (FedAvg vs Multi-Krum)
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

### Step 4: Full Security Attack Evaluation
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

---

## 11. Outputs and Reporting

### Strategy Comparison
- `outputs/metrics/fl_strategy_comparison.csv`
- `outputs/metrics/fl_strategy_comparison.json`

### Attack Scenario Comparison
- `outputs/metrics/fl_attack_comparison.csv`
- `outputs/metrics/fl_attack_comparison.json`
- `outputs/metrics/fl_attack_robustness.csv`

### Per-Run FL Files
- `outputs/metrics/fl_round_metrics_<label>.csv`
- `outputs/metrics/fl_round_client_weights_<label>.csv`
- `outputs/metrics/fl_final_metrics_<label>.json`
- `outputs/metrics/fl_aggregation_summary_<label>.json`
- `outputs/plots/fl_confusion_matrix_<label>.png`
- `outputs/plots/fl_roc_curve_<label>.png`
- `outputs/plots/fl_pr_curve_<label>.png`

### Security Evaluation Plots
- `outputs/plots/fl_attack_evaluation_quality_metrics.png` (F1, Precision, Recall)
- `outputs/plots/fl_attack_evaluation_aggregation_robustness.png` (outlier rejection behavior)

---

## 12. Key Metrics for Thesis Tables

Use these as primary indicators:
- F1 score (overall minority-class balance)
- Precision (false alarm control)
- Recall (fraud catch rate)
- Aggregation robustness (rejected malicious/outlier updates per round)

Recommended table columns:
- scenario
- aggregation method
- attack enabled
- clip threshold
- noise multiplier
- F1, precision, recall, ROC-AUC, PR-AUC

---

## 13. Updated Contribution Statement

This thesis contribution is now centered on **security-focused federated learning**:

1. Reproducible poisoning attack simulation in FL.
2. Robust server aggregation via Multi-Krum with outlier audit logs.
3. Client-side leakage mitigation via clipping + DP noise.
4. End-to-end scenario comparison framework (normal vs attacked vs defended).
5. Compatible integration with existing preprocessing, baseline training, and reporting pipeline.

---

## 14. Repository Structure (Updated)

```text
├── data/
├── src/
│   ├── preprocess.py
│   ├── train_central.py
│   ├── train_ml_hybrids.py
│   ├── flwr_client.py
│   ├── flwr_server.py
│   ├── flwr_server_security.py
│   ├── aggregation.py
│   ├── attack_simulation.py
│   └── common.py
├── outputs/
│   ├── processed/
│   ├── metrics/
│   └── plots/
└── requirements.txt
```

---

**Status**: Documentation aligned with current security-focused implementation and thesis evaluation workflow.
