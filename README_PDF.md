# Security-Focused Thesis Runbook (PDF Companion)

This file now serves as a concise companion for generating and presenting the updated
security-focused thesis documentation and results.

---

## What Changed in the Project

The FL architecture has been refocused from adaptive fraud-aware aggregation to
security evaluation under poisoning threats.

### Kept
- Dataset preprocessing and leakage-safe pipeline
- Centralized ML baseline and hybrid comparison
- Flower simulation architecture
- Bank-style non-IID partitioning (`bank_noniid`)
- Existing reporting/evaluation compatibility

### Replaced / Added
- Removed fraud-aware adaptive weighting logic
- Added separate aggregation module:
  - FedAvg baseline
  - Multi-Krum robust aggregation
- Added client-side security protections:
  - gradient clipping
  - differential privacy Gaussian noise
- Added poisoning attack simulation module
- Added scenario-based evaluation:
  - normal FL
  - FL under attack
  - FL with defense

---

## New Security Evaluation Components

### 1) Robust Aggregation
- `src/aggregation.py`
- Methods:
  - `fedavg_aggregate`
  - `multi_krum_aggregate`
- Logs rejected clients as outliers during Multi-Krum rounds.

### 2) Client-Side Protection
- `src/flwr_client.py`
- Before transmission:
  - compute update norm
  - clip by threshold
  - add Gaussian DP noise
- Logs:
  - gradient norm before clipping
  - gradient norm after clipping
  - clipping threshold
  - noise multiplier and effective scale

### 3) Attack Simulation
- `src/attack_simulation.py`
- Supports:
  - malicious client selection
  - configurable attack strength
  - attack mode: `sign_flip` or `scale`
- Produces attack-comparison tables and plots.

---

## Core Commands for Thesis Experiments

### A) Preprocess
```bash
python3 src/preprocess.py --data_path data/Bank_Transaction_Fraud_Detection.xlsx --output_dir outputs
```

### B) Centralized Baselines
```bash
python3 src/train_central.py --output_dir outputs
python3 src/train_ml_hybrids.py --output_dir outputs
```

### C) FL Strategy Comparison (FedAvg vs Multi-Krum)
```bash
python3 src/flwr_server_security.py \
  --output_dir outputs \
  --partition_mode bank_noniid \
  --compare_strategies \
  --num_malicious 1 \
  --clip_threshold 1.0 \
  --noise_multiplier 0.01
```

### D) Full Attack Evaluation (Recommended)
```bash
python3 src/flwr_server_security.py \
  --output_dir outputs \
  --partition_mode bank_noniid \
  --evaluate_attack_scenarios \
  --attack_mode sign_flip \
  --attack_strength 5.0 \
  --malicious_clients w3 \
  --num_malicious 1 \
  --clip_threshold 1.0 \
  --noise_multiplier 0.01
```

---

## Key Metrics Files for Thesis Tables

### Strategy-level comparison
- `outputs/metrics/fl_strategy_comparison.csv`
- `outputs/metrics/fl_strategy_comparison.json`

### Attack scenario comparison
- `outputs/metrics/fl_attack_comparison.csv`
- `outputs/metrics/fl_attack_comparison.json`
- `outputs/metrics/fl_attack_robustness.csv`

### Per-run details
- `outputs/metrics/fl_round_metrics_<label>.csv`
- `outputs/metrics/fl_round_client_weights_<label>.csv`
- `outputs/metrics/fl_final_metrics_<label>.json`
- `outputs/metrics/fl_aggregation_summary_<label>.json`

---

## Plots for Presentation

- `outputs/plots/fl_attack_evaluation_quality_metrics.png`
  - F1
  - Precision
  - Recall
- `outputs/plots/fl_attack_evaluation_aggregation_robustness.png`
  - aggregation robustness via outlier rejection

You also retain per-run confusion matrix, ROC, and PR plots for each scenario label.

---

## Recommended Thesis Narrative (Updated)

1. Start from privacy-preserving fraud detection context.
2. Introduce poisoning and reconstruction risks in FL.
3. Define defense stack:
   - clipping + DP at client side
   - Multi-Krum at server side
4. Show three-scenario comparison:
   - normal FL
   - attacked FL
   - defended FL
5. Conclude with robustness gain and trade-offs.

---

## PDF Regeneration (if needed)

```bash
source .venv/bin/activate
python convert_to_pdf.py
```

---

**Status**: Updated for security-focused FL experimentation and thesis reporting.
