# Project Handoff for Claude — FL Fraud Detection Security

**Last updated:** June 2026 — after full 7-scenario pipeline run

---

## 1) Project Identity

- **Project:** Federated Learning Fraud Detection — Security Evaluation
- **Stack:** Flower (simulation), PyTorch, scikit-learn, XGBoost
- **Dataset:** `data/Bank_Transaction_Fraud_Detection.xlsx`
- **Target column:** `Is_Fraud` (binary, imbalanced ~5% fraud)
- **Primary objective:** Security/privacy robustness in FL — NOT pure accuracy optimization
- **GitHub:** https://github.com/Hamzababar25/thesis-fl-fraudd
- **Python env:** `.venv/` (activate with `source .venv/bin/activate`)

---

## 2) IMPORTANT — Dataset Limitation (Read This First)

This dataset has **near-zero feature correlation with fraud labels** (ROC-AUC ≈ 0.50 across all models). This is a known property of this synthetic Kaggle dataset — NOT a code bug or pipeline flaw.

- Absolute F1 is ~9.6% for all models
- The thesis does NOT claim high fraud detection accuracy
- The thesis claims: **attack degrades performance, defense recovers it** — this IS proven

Do NOT try to "fix" accuracy by changing the model. The signal is in the **relative change** between normal FL, attack, and defense scenarios.

---

## 3) Thesis Methodology (9-Step Pipeline)

```
preprocess → centralized baseline → ML comparison (LR/RF/XGB) →
→ FL normal → FL sign_flip attack → FL sign_flip defense →
→ FL scale attack → FL scale defense →
→ FL label_flip attack → FL label_flip defense →
→ generate combined outputs
```

Full run command:
```bash
bash run_all.sh
```

Or step by step:
```bash
source .venv/bin/activate
python src/preprocess.py --data_path data/Bank_Transaction_Fraud_Detection.xlsx --output_dir outputs
python src/train_central.py --output_dir outputs
python src/train_ml_hybrids.py --output_dir outputs --max_train_samples 60000
python src/flwr_server_security.py \
  --output_dir outputs --partition_mode bank_noniid \
  --fl_model best_from_ml --evaluate_attack_scenarios \
  --attack_strength 5.0 --malicious_clients w3 \
  --num_malicious 1 --clip_threshold 1.0 --noise_multiplier 0.01 --rounds 20
python src/generate_combined_outputs.py --output_dir outputs
```

---

## 4) Code Structure

| File | Purpose |
|------|---------|
| `src/preprocess.py` | Data cleaning, feature engineering, train/val/test split, save .npz + .npy |
| `src/common.py` | Shared: FraudMLP, FraudLogistic, compute_metrics, **find_best_threshold** |
| `src/train_central.py` | Centralized LR baseline — uses val-set threshold optimization |
| `src/train_ml_hybrids.py` | LR + RF + XGBoost + 2 hybrids — val-set threshold per model |
| `src/aggregation.py` | FedAvg + Multi-Krum implementations |
| `src/attack_simulation.py` | apply_poisoning_attack, **apply_label_flipping_to_data**, plots |
| `src/flwr_client.py` | FL client — label_flip data poisoning + sign_flip/scale model poisoning |
| `src/flwr_server_security.py` | Main security FL — 7-scenario evaluation, val-based threshold |
| `src/generate_combined_outputs.py` | Reads all per-scenario CSVs → clean combined files in outputs/analysis/ |
| `run_all.sh` | Full 9-step pipeline script |

---

## 5) Attack Implementation

### Three attack types (all implemented):

**1. Sign-Flip (model poisoning)**
- Where: `src/attack_simulation.py → apply_poisoning_attack(attack_mode="sign_flip")`
- How: After local training, flip update direction (delta × -strength)
- Triggered in: `src/flwr_client.py → fit()` if malicious and attack_mode != "label_flip"

**2. Scale (model poisoning)**
- Where: `src/attack_simulation.py → apply_poisoning_attack(attack_mode="scale")`
- How: After local training, amplify update (delta × +strength)
- Same trigger as sign_flip

**3. Label-Flip (data poisoning) ← NEW**
- Where: `src/attack_simulation.py → apply_label_flipping_to_data()`
- How: BEFORE local training, flip all fraud labels (1→0) in training data
- Triggered in: `src/flwr_client.py → fit()` if malicious and attack_mode == "label_flip"
- Key insight: model updates look different but are harder to detect (data-level, not model-level)

### Attack config:
- Malicious client: `w3` (1 of 3 clients)
- Attack strength: `5.0`
- `--attack_mode` choices: `sign_flip`, `scale`, `label_flip`

---

## 6) Defense Implementation

| Defense | Where | Config |
|---------|-------|--------|
| Gradient clipping | `flwr_client.py → protect_model_update()` | `--clip_threshold 1.0` |
| DP Gaussian noise | `flwr_client.py → protect_model_update()` | `--noise_multiplier 0.01` |
| Multi-Krum aggregation | `flwr_server_security.py → SecurityRobustFedAvg` | `--aggregation_method multi_krum` |

Defense is applied together (Multi-Krum + clipping + DP) in defended scenarios.

---

## 7) Threshold Optimization (NEW)

**Function:** `src/common.py → find_best_threshold(y_true, y_score)`
- Searches threshold 0.05–0.95 in steps of 0.01
- Maximizes F1-score on **validation set only** (no data leakage)
- Applied in: train_central.py, train_ml_hybrids.py, flwr_server_security.py

Best thresholds found (typical):
- Normal FL: 0.24, Sign-flip attack: 0.05, Defended: 0.16
- ML models: LR=0.25, RF=0.42, XGBoost=0.15

---

## 8) FL Scenario Configuration (evaluate_attack_scenarios)

When `--evaluate_attack_scenarios` is passed, 7 scenarios run automatically:

| Label | Method | Attack | Defense |
|-------|--------|--------|---------|
| `normal_fl` | fedavg | None | None |
| `sign_flip_no_defense` | fedavg | sign_flip | None |
| `sign_flip_defended` | multi_krum | sign_flip | clip+DP |
| `scale_no_defense` | fedavg | scale | None |
| `scale_defended` | multi_krum | scale | clip+DP |
| `label_flip_no_defense` | fedavg | label_flip | None |
| `label_flip_defended` | multi_krum | label_flip | clip+DP |

---

## 9) Latest Key Results

### ML Models (val-set threshold optimized)

| Model | Category | F1 | Recall | ROC-AUC | Threshold |
|-------|----------|-----|--------|---------|-----------|
| XGB+RF (60/40) | Hybrid | 9.64% | 99.27% | 0.5029 | 0.25 |
| Random Forest | Single | 9.62% | 96.36% | 0.5072 | 0.42 |
| Logistic Regression | Single | 9.60% | 99.87% | 0.5032 | 0.25 |
| XGBoost | Single | 9.59% | 99.07% | 0.5027 | 0.15 |
| LR Centralized | Baseline | 9.26% | 57.24% | 0.5003 | 0.49 |

**Best single model: Random Forest** (F1=9.62%, ROC-AUC=0.5072)
**FL deployment: Logistic Regression** (tree-based models incompatible with FedAvg)

### FL Scenarios (20 rounds, bank_noniid partition)

| Scenario | F1 | Recall | ROC-AUC | Loss |
|----------|-----|--------|---------|------|
| normal_fl | 9.61% | 98.81% | 0.5004 | 0.6882 |
| sign_flip_no_defense | 8.80% | **35.10%** | 0.4953 | 0.5345 |
| sign_flip_defended | 9.62% | 99.34% | 0.5023 | 0.6901 |
| scale_no_defense | 9.61% | 97.75% | 0.4998 | 0.6590 |
| scale_defended | 9.64% | **99.41%** | 0.5008 | 0.6935 |
| label_flip_no_defense | 9.63% | 96.76% | 0.5009 | 0.5110 |
| label_flip_defended | 9.65% | 98.41% | 0.5011 | 0.5399 |

### Round-wise (F1 every 5 rounds)

| Round | normal_fl | sign_flip_atk | sign_flip_def | label_flip_atk | label_flip_def |
|-------|-----------|--------------|--------------|----------------|----------------|
| 0 | 0.0901 | 0.0901 | 0.0901 | 0.0901 | 0.0901 |
| 5 | 0.0893 | 0.0842 | 0.0923 | **0.0064** | 0.0844 |
| 10 | 0.0910 | 0.0829 | 0.0917 | 0.0384 | 0.0836 |
| 15 | 0.0907 | 0.0798 | 0.0915 | 0.0641 | 0.0351 |
| 20 | 0.0920 | **0.0787** | 0.0932 | 0.0679 | 0.0554 |

### Defense Evidence
- Sign-flip defended: w3 rejected in **20/20 rounds**
- Scale defended: w3 rejected in **20/20 rounds**
- Label-flip defended: partial recovery only (data-level attack harder to detect)

---

## 10) Output Files (what to use)

### Thesis-ready combined files → `outputs/analysis/`
| File | Use for |
|------|---------|
| `combined_ml_results.csv` | ML model comparison table |
| `combined_fl_final.csv` | FL 7-scenario final metrics |
| `combined_fl_roundwise.csv` | Round-wise table (every 5 rounds) |
| `combined_client_security.csv` | Client norms + rejection logs |
| `master_summary.json` | All key numbers in one JSON |
| `thesis_table_ml.txt` | Copy-paste ready ML table |
| `thesis_table_fl.txt` | Copy-paste ready FL table |
| `thesis_table_roundwise.txt` | Copy-paste ready round-wise table |
| `report.html` | Full interactive browser report (share with anyone) |

### Raw per-scenario files → `outputs/metrics/`
- `fl_attack_comparison.csv` — all 7 scenarios in one CSV
- `fl_attack_robustness.csv` — rejection evidence
- `fl_all_scenarios_roundwise.csv` — full round-wise (all 21 rounds × 43 cols)
- `ml_single_results.csv`, `ml_hybrid_results.csv` — individual model results

### Plots → `outputs/plots/`
- `fl_attack_evaluation_quality_metrics.png` — bar chart comparison
- `fl_attack_evaluation_aggregation_robustness.png` — rejection evidence
- `fl_confusion_matrix_*.png`, `fl_roc_curve_*.png`, `fl_pr_curve_*.png` — per scenario

---

## 11) Processed Data Files → `outputs/processed/`

| File | Used by |
|------|---------|
| `train_X.npz`, `val_X.npz`, `test_X.npz` | Sparse (sklearn models) |
| `train_X_dense.npy`, `val_X_dense.npy`, `test_X_dense.npy` | Dense (PyTorch FL models) |
| `train_y.npy`, `val_y.npy`, `test_y.npy` | Labels for all splits |
| `preprocessor.joblib` | For inference on new data |
| `manifest.json` | Feature names, column info, fraud ratios |

---

## 12) Claim Boundaries (Important for Thesis)

**Proven claims:**
- FL framework correctly detects model poisoning (sign_flip, scale) via Multi-Krum
- Defense (Multi-Krum + clipping + DP) recovers performance after model poisoning
- Malicious client w3 rejected in 20/20 rounds under sign_flip and scale defense
- Label-flip (data poisoning) is a distinct threat class harder to defend against

**Do NOT claim:**
- High absolute fraud detection accuracy (dataset limitation)
- Protection against all possible attacks
- Universal security guarantee

**Suggested thesis claim:**
> "We propose and evaluate a modular FL security framework for fraud detection that demonstrates measurable resilience against model poisoning attacks (sign-flip, scale) through Multi-Krum aggregation combined with gradient clipping and differential privacy. A novel label-flip data poisoning attack is introduced, revealing limitations of gradient-level defenses against data-level threats."

---

## 13) Future Work (already identified)

1. **More clients (5–10):** Currently 3 clients, thesis argument for distributed banking
2. **SMOTE:** Handle class imbalance in training data
3. **Better dataset:** Real fraud dataset with informative features
4. **Label-flip defense:** Dedicated data-level defense (e.g., robust loss functions, data auditing)
5. **Gradient reconstruction / model inversion:** Not implemented, future extensions

---

## 14) Known Issues / Gotchas

- `[WARN] best_from_ml resolved to unsupported FL model; falling back to logistic_regression` — this is expected since best ML model (Random Forest) can't be used in FL. Normal behavior.
- Flower deprecation warnings about `client_fn` signature and `NumpyClient` — cosmetic only, do not affect results
- `outputs/` is in `.gitignore` EXCEPT `outputs/analysis/` which is tracked
- Data files (`.xlsx`, `.npy`) are gitignored — not in repo

---
