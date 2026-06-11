# Project Handoff for Documentation (Claude Input)

This file is a complete handoff summary of the current codebase, objectives, implementation choices, commands, and latest results.

## 1) Project Identity

- Project: Federated Fraud Detection Security Evaluation
- Stack: Flower (simulation), PyTorch, scikit-learn
- Dataset: `data/Bank_Transaction_Fraud_Detection.xlsx`
- Target column: `Is_Fraud` (binary, imbalanced)
- Primary objective: Security/privacy robustness in FL (not pure accuracy optimization)

## 2) Current Thesis Methodology

The code now follows this methodological story:

1. Preprocess dataset and engineer features
2. Centralized ML comparison (LR, RF, XGBoost)
3. Best model selection
4. Deploy selected best model in FL
5. Run attack simulation (poisoning)
6. Apply defenses (Multi-Krum + clipping + DP)
7. Evaluate normal vs attacked vs defended scenarios

## 3) Best-Model-to-FL Alignment (Important)

- Implemented in `src/flwr_server_security.py`
- New flag: `--fl_model {best_from_ml, logistic_regression, mlp}`
- Default is `best_from_ml`
- `best_from_ml` reads `outputs/metrics/ml_comparison_summary.json`
- Current supported mapping:
  - if best single is `logistic_regression` -> FL model is logistic regression (`FraudLogistic`)
  - otherwise fallback warning -> `logistic_regression`

So the current thesis pipeline is aligned with:
"Best ML model select karo aur usi model ko FL mein deploy karo"

## 4) Security Implementation Summary

### Attack (implemented)
- Type: model/update poisoning
- Modes:
  - `sign_flip` (default)
  - `scale`
- Config:
  - `--attack_mode`
  - `--attack_strength`
  - `--malicious_clients`

### Attacks not yet implemented (important)
- Gradient reconstruction: not implemented in current codebase
- Model inversion: not implemented in current codebase
- Treat both as future extensions unless dedicated implementation/results are added

### Defenses (implemented)
- Client-side gradient clipping (`--clip_threshold`)
- Client-side DP Gaussian noise (`--noise_multiplier`)
- Server-side robust aggregation:
  - `fedavg`
  - `multi_krum`

### Robustness logging
- Per-round selected/rejected clients
- Aggregation scores
- Malicious flags
- Gradient norm before/after clipping
- Noise scale used

## 5) Code Structure (Relevant Files)

- `src/preprocess.py` -> preprocessing + split + transform artifacts
- `src/train_central.py` -> centralized baseline
- `src/train_ml_hybrids.py` -> LR/RF/XGB + hybrid comparison + summary json
- `src/common.py` -> shared model/utilities (includes `FraudMLP` and `FraudLogistic`)
- `src/flwr_client.py` -> FL client, clipping/DP, poisoning behavior
- `src/aggregation.py` -> FedAvg + Multi-Krum implementations
- `src/attack_simulation.py` -> attack helpers, comparison/robustness plotting
- `src/flwr_server_security.py` -> main security FL orchestration
- `src/flwr_server_adaptive_secure.py` -> compatibility wrapper only

## 6) Run Commands (Ubuntu/Linux)

```bash
cd "/home/hamza-babar/Documents/thesis-fl-fraud"
source .venv/bin/activate

python3 src/preprocess.py --data_path data/Bank_Transaction_Fraud_Detection.xlsx --output_dir outputs
python3 src/train_central.py --output_dir outputs
python3 src/train_ml_hybrids.py --output_dir outputs

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
  --noise_multiplier 0.01 \
  --rounds 20
```

## 7) Latest Key Metrics (From Current Files)

Source: `outputs/metrics/fl_attack_comparison.csv`

### Scenario 1: Normal FL
- scenario: `normal_fl`
- fl_model: `logistic_regression`
- aggregation: `fedavg`
- attack_enabled: 0
- clip/noise: 0.0 / 0.0
- loss: 0.6877248563857594
- f1: 0.0906992845603508
- recall: 0.5194976867151355
- precision: 0.049687085150768064
- roc_auc: 0.49887784066158725
- pr_auc: 0.05024755384913769

### Scenario 2: FL Under Attack (No Defense)
- scenario: `fl_under_attack`
- fl_model: `logistic_regression`
- aggregation: `fedavg`
- attack: sign_flip, strength 5.0, malicious client w3
- clip/noise: 0.0 / 0.0
- loss: 0.5624623635574786
- f1: 0.08134141990724224
- recall: 0.15069398545935228
- precision: 0.05570486195944295
- roc_auc: 0.49628073296313013
- pr_auc: 0.050953478032754775

### Scenario 3: FL With Defense
- scenario: `fl_with_defense`
- fl_model: `logistic_regression`
- aggregation: `multi_krum`
- attack: sign_flip, strength 5.0, malicious client w3
- defense: clipping 1.0 + noise 0.01
- loss: 0.6882036605015526
- f1: 0.09027210089921757
- recall: 0.5109054857898215
- precision: 0.04951002369820022
- roc_auc: 0.5013021094651283
- pr_auc: 0.04995511515767746

## 8) Robustness Evidence

Source: `outputs/metrics/fl_attack_robustness.csv`

- `normal_fl`: rounds_with_rejection = 0/20
- `fl_under_attack`: rounds_with_rejection = 0/20
- `fl_with_defense`: rounds_with_rejection = 20/20, avg_rejected_clients = 1.0

Source: `outputs/metrics/fl_aggregation_summary_fl_with_defense.json`
- malicious client `w3` is rejected in all rounds (1..20)

## 9) Centralized/ML Context

### Centralized
Source: `outputs/metrics/centralized_metrics.json`
- f1: 0.09245187436676798
- recall: 0.48248512888301387
- precision: 0.05112402829329785
- roc_auc: 0.5002735098077343
- pr_auc: 0.05063611245584996

### Best-model summary
Source: `outputs/metrics/ml_comparison_summary.json`
- best_single.model: `logistic_regression`
- best_single.f1: 0.09127034995513396
- train_samples_used: 60000

## 10) Output Files to Use in Documentation

Primary:
- `outputs/metrics/fl_attack_comparison.csv`
- `outputs/metrics/fl_attack_robustness.csv`
- `outputs/metrics/fl_aggregation_summary_fl_with_defense.json`

Supporting:
- `outputs/metrics/centralized_metrics.json`
- `outputs/metrics/ml_comparison_summary.json`

Plots:
- `outputs/plots/fl_attack_evaluation_quality_metrics.png`
- `outputs/plots/fl_attack_evaluation_aggregation_robustness.png`

## 11) Important Caveats / Claim Boundaries

- Evaluated attack family: poisoning (update-level sign_flip/scale), not all possible attacks.
- Do not claim universal security.
- Claim should be: robust and modular framework, validated against poisoning under a defined threat model.

## 12) Suggested Documentation Narrative

1. Define privacy/security motivation in FL fraud detection.
2. Describe best-model selection and FL deployment alignment.
3. Describe poisoning threat model and sign_flip attack.
4. Describe layered defenses (clipping + DP + Multi-Krum).
5. Present 3-scenario comparison and robustness evidence.
6. Discuss limitations and extensibility to additional attacks.

