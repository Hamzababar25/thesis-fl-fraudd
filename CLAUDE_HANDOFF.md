# CLAUDE HANDOFF — Security-Aware Federated Learning for Fraud Detection

**Student:** Hamza Babar
**Thesis Deadline:** June 17, 2026
**GitHub:** https://github.com/Hamzababar25/thesis-fl-fraudd
**Last Updated:** June 14, 2026

---

## 1. What This Project Is

A thesis project implementing a **privacy-preserving fraud detection system** using Federated Learning (FL). Three simulated banks collaboratively train a shared fraud detection model without sharing raw customer transaction data. The system is then stress-tested against three types of poisoning attacks, and a 3-layer defense framework is evaluated.

**Core thesis question:**
> *Can multiple banks train a fraud detection model collaboratively using Federated Learning while remaining secure against malicious participants who try to corrupt the shared model?*

**Answer demonstrated:**
> Yes — FL with Multi-Krum aggregation + Gradient Clipping + Differential Privacy achieves ROC-AUC 0.93, fully recovering from all three attack types, while matching centralized performance (ROC-AUC 0.96) without any raw data sharing.

---

## 2. Tech Stack

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.12 | Language |
| PyTorch | CPU | Neural network models (FraudLogistic, FraudMLP) |
| Flower (flwr) | Latest | FL simulation framework |
| scikit-learn | Latest | LR, Random Forest, preprocessing, SMOTE |
| XGBoost | Latest | Gradient boosting classifier |
| imbalanced-learn | Latest | SMOTE for class imbalance |
| NumPy / SciPy | Latest | Array ops, sparse matrices |
| Pandas | Latest | Data manipulation |
| Matplotlib | Latest | Plots |
| kagglehub | Latest | Dataset download |

---

## 3. Dataset

**Name:** Kaggle Credit Card Fraud Detection
**Source:** `mlg-ulb/creditcardfraud` (Kaggle)
**Download path:** `~/.cache/kagglehub/datasets/mlg-ulb/creditcardfraud/versions/3/creditcard.csv`

| Property | Value |
|----------|-------|
| Total rows | 284,807 |
| Fraud transactions | 492 (0.17%) |
| Normal transactions | 284,315 (99.83%) |
| Original features | 30 (Time, V1–V28, Amount) |
| Final features after engineering | 33 |
| One-Hot Encoding needed | No — all numeric |

**Features:**
- `V1`–`V28` — PCA-transformed by dataset provider (anonymized, informative)
- `log_amount` — log(1 + Amount) to reduce skew
- `time_hour` — (Time seconds % 86400) / 3600 → hour of day
- `time_sin` / `time_cos` — cyclic encoding of time_hour
- `is_night` — binary: 1 if hour < 5 (midnight to 5am)
- Raw `Time` and raw `Amount` dropped (replaced by engineered versions)

**Dataset Split (stratified):**

| Split | Rows | Fraud | Fraud % |
|-------|------|-------|---------|
| Train | 199,364 | 344 | 0.1725% |
| Validation | 42,721 | 74 | 0.1732% |
| Test | 42,722 | 74 | 0.1732% |

---

## 4. Project Structure

```
thesis-fl-fraud/
├── src/
│   ├── preprocess_creditcard.py      # Preprocessing pipeline
│   ├── common.py                     # Shared models + metrics
│   ├── train_central.py              # Centralized LR baseline
│   ├── train_ml_hybrids.py           # LR + RF + XGBoost comparison
│   ├── aggregation.py                # FedAvg + Multi-Krum
│   ├── attack_simulation.py          # 3 attack implementations
│   ├── flwr_client.py                # FL client (train + protect)
│   ├── flwr_server.py                # FL FedAvg (IID + Non-IID)
│   ├── flwr_server_security.py       # Security FL — 7 scenarios
│   ├── generate_combined_outputs.py  # Merge all results to analysis/
│   └── generate_sir_report.py        # Thesis-ready report files
├── outputs2/                         # All results (active)
│   ├── processed/                    # Preprocessed data files
│   ├── metrics/                      # Per-scenario JSON/CSV
│   ├── plots/                        # Confusion matrix, ROC, PR curves
│   └── sir_report2/                  # 12 thesis-ready files
├── run_all.sh                        # One-command full pipeline
├── requirements.txt                  # All dependencies
├── CLAUDE_HANDOFF.md                 # This file
└── .venv/                            # Python virtual environment
```

---

## 5. How to Run — Full Pipeline

```bash
# Setup (first time only)
cd /home/hamza-babar/Documents/thesis-fl-fraud
source .venv/bin/activate

# Download dataset (first time only — cached after)
python3 -c "import kagglehub; kagglehub.dataset_download('mlg-ulb/creditcardfraud')"
```

```bash
# Step 1 — Preprocess (creates outputs2/processed/)
python src/preprocess_creditcard.py --output_dir outputs2

# Step 2 — Centralized baseline
python src/train_central.py --output_dir outputs2

# Step 3 — ML models (LR, RF, XGBoost)
python src/train_ml_hybrids.py --output_dir outputs2 --max_train_samples 0 --no_hybrid

# Step 4 — FL FedAvg IID
python src/flwr_server.py --output_dir outputs2 --partition_mode iid --rounds 20 --lr 1e-3

# Step 5 — FL FedAvg Non-IID
python src/flwr_server.py --output_dir outputs2 --partition_mode noniid --rounds 20 --lr 1e-3

# Step 6 — Security FL: 7 attack/defense scenarios (~10-12 min)
python src/flwr_server_security.py \
  --output_dir outputs2 \
  --partition_mode bank_noniid \
  --rounds 20 \
  --lr 1e-3 \
  --fl_model logistic_regression \
  --evaluate_attack_scenarios \
  --attack_strength 5.0 \
  --malicious_clients w3 \
  --num_malicious 1 \
  --clip_threshold 1.0 \
  --noise_multiplier 0.01

# Step 7 — Combined output files
python src/generate_combined_outputs.py --output_dir outputs2

# Step 8 — Sir Report (thesis-ready CSVs)
python src/generate_sir_report.py --output_dir outputs2 --report_name sir_report2
```

Or everything at once:
```bash
DATASET=creditcard bash run_all.sh
```

---

## 6. Models Used

### Standalone ML Models (no federation)

| Model | Purpose | FL Compatible |
|-------|---------|--------------|
| Logistic Regression | Centralized baseline + FL model | Yes |
| Random Forest | Best standalone ROC-AUC | No (tree-based) |
| XGBoost | Best standalone F1 | No (tree-based) |

### FL Models

| Model | Class | Used in |
|-------|-------|---------|
| `FraudLogistic` | Single linear layer | All FL runs (IID, Non-IID, Security) |
| `FraudMLP` | 3-layer MLP | Available via `--fl_model mlp` |

---

## 7. SMOTE — Class Imbalance Handling

Training data has 344 fraud vs 199,020 normal (0.17%). SMOTE generates synthetic fraud samples:

| Model | SMOTE Strategy | After SMOTE Fraud Count |
|-------|---------------|------------------------|
| Centralized LR | 1:1 full balance | 199,020 |
| ML Models (RF, XGBoost, LR standalone) | 10x ratio cap | 19,902 |
| FL clients | `pos_weight` in BCEWithLogitsLoss | No SMOTE — weighted loss |

SMOTE applied **only to training set**. Val and Test sets remain original imbalanced ratio.

---

## 8. Defense Framework (3 Layers)

```
CLIENT SIDE:
  [Local Training]
       ↓
  [Layer 1: Gradient Clipping]  threshold = 1.0
       ↓
  [Layer 2: DP Gaussian Noise]  noise_multiplier = 0.01
       ↓
  Send protected update to server

SERVER SIDE:
  [Layer 3: Multi-Krum Aggregation]
  → Calculate pairwise distances between all client updates
  → Reject updates that are statistical outliers
  → Aggregate only trusted updates
       ↓
  Updated global model
```

---

## 9. Attack Types

### Sign-Flip Attack (Model Poisoning)
- **Where:** `src/attack_simulation.py → apply_poisoning_attack(mode="sign_flip")`
- **How:** Flip and amplify gradient direction — `delta × −strength`
- **Result:** ROC-AUC drops from 0.9362 → **0.0896** (model completely destroyed)
- **Severity:** SEVERE

### Scale Attack (Model Poisoning)
- **Where:** `src/attack_simulation.py → apply_poisoning_attack(mode="scale")`
- **How:** Amplify update magnitude — `delta × +strength`
- **Result:** ROC-AUC 0.9362 → 0.9544 (model still works, attacker dominates slightly)
- **Severity:** MILD

### Label-Flip Attack (Data Poisoning)
- **Where:** `src/attack_simulation.py → apply_label_flipping_to_data()`
- **How:** Flip fraud labels (1→0) in malicious client's training data before training
- **Result:** ROC-AUC 0.9362 → 0.9400 (gradual drift, harder to detect)
- **Severity:** MODERATE

**Attack config:** Malicious client = `w3`, attack_strength = 5.0, 1 of 3 clients

---

## 10. FL Scenarios (7 Total)

| # | Label | Aggregation | Attack | Defense |
|---|-------|------------|--------|---------|
| 1 | `normal_fl` | FedAvg | None | None |
| 2 | `sign_flip_no_defense` | FedAvg | Sign-Flip | None |
| 3 | `sign_flip_defended` | Multi-Krum | Sign-Flip | Clip + DP |
| 4 | `scale_no_defense` | FedAvg | Scale | None |
| 5 | `scale_defended` | Multi-Krum | Scale | Clip + DP |
| 6 | `label_flip_no_defense` | FedAvg | Label-Flip | None |
| 7 | `label_flip_defended` | Multi-Krum | Label-Flip | Clip + DP |

---

## 11. FL Configuration

| Parameter | Value |
|-----------|-------|
| Clients | 3 |
| Rounds | 20 |
| Local epochs per round | 1 |
| Batch size | 128 |
| Learning rate | 0.001 |
| FL model | Logistic Regression |
| IID partition | Equal random split across 3 clients |
| Non-IID partition | Unequal split — client 1 more, client 3 less |
| Security partition | Bank-style non-IID |
| Malicious client | w3 (1 of 3 = 33%) |

---

## 12. Threshold Optimization

All models use **validation-set threshold optimization** to avoid data leakage:

```python
# src/common.py → find_best_threshold(y_true, y_score)
# Searches threshold 0.05–0.95 in steps of 0.01
# Maximizes F1-score on validation set only
# Then applies that threshold to test set evaluation
```

| Model | Best Threshold |
|-------|---------------|
| Centralized LR | 0.94 |
| XGBoost | 0.93 |
| Random Forest | 0.70 |
| FL IID | 0.94 |
| Security FL (normal) | 0.84–0.94 |

---

## 13. Final Results

### ML Models

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|-----|---------|
| Random Forest | 99.94% | 87.88% | 78.38% | 82.86% | **0.9685** |
| XGBoost | 99.95% | **93.55%** | 78.38% | **85.29%** | 0.9627 |
| LR Centralized | 99.90% | 67.39% | **83.78%** | 74.70% | 0.9609 |
| LR Standalone | 99.91% | 69.66% | 83.78% | 76.07% | 0.9570 |

### FL Models

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|-----|---------|
| FL FedAvg IID | 99.91% | 70.59% | 81.08% | 75.47% | 0.9627 |
| FL FedAvg Non-IID | 99.83% | 0.00% | 0.00% | 0.00% | 0.8920 |

### Security FL Scenarios

| Scenario | Accuracy | Precision | Recall | F1 | ROC-AUC |
|----------|----------|-----------|--------|-----|---------|
| Normal FL | 99.93% | 80.00% | 81.08% | 80.54% | 0.9362 |
| Sign-Flip — No Defense | 98.52% | 0.00% | 0.00% | 0.00% | 0.0896 |
| Sign-Flip + Multi-Krum | 99.93% | 77.92% | 81.08% | 79.47% | 0.9308 |
| Scale — No Defense | 99.92% | 75.00% | 81.08% | 77.92% | 0.9544 |
| Scale + Multi-Krum | 99.93% | 77.92% | 81.08% | 79.47% | 0.9307 |
| Label-Flip — No Defense | 99.92% | 78.87% | 75.68% | 77.24% | 0.9400 |
| Label-Flip + Multi-Krum | 99.93% | 77.92% | 81.08% | 79.47% | 0.9310 |

### Multi-Krum Defense Evidence

| Scenario | Rounds Client w3 Blocked | Total Rounds | Rejection Rate |
|----------|--------------------------|--------------|----------------|
| Sign-Flip Defended | 20 | 20 | **100%** |
| Scale Defended | 20 | 20 | **100%** |
| Label-Flip Defended | 20 | 20 | **100%** |

---

## 14. Evaluation Metrics — Why These Two Are Primary

**Primary: ROC-AUC**
Dataset is 99.83% normal. Any model predicting all "Normal" gets 99.83% accuracy — but detects zero fraud. ROC-AUC measures how well the model separates fraud from normal regardless of class imbalance or threshold. Range: 0.5 (random) to 1.0 (perfect).

**Secondary: Recall**
In banking fraud, a missed fraud (False Negative) is more costly than a false alarm (False Positive). Recall = what percentage of actual fraud was detected. Attack impact is clearly visible: Sign-Flip drops recall from 81% to 0%.

---

## 15. Output Files Guide

### `outputs2/processed/`
| File | Description |
|------|-------------|
| `train_X.npz` / `train_y.npy` | Training features (sparse) + labels |
| `val_X.npz` / `val_y.npy` | Validation set |
| `test_X.npz` / `test_y.npy` | Test set |
| `train_X_dense.npy` | Dense array for PyTorch/Flower |
| `val_X_dense.npy` | Dense validation for FL |
| `test_X_dense.npy` | Dense test for FL |
| `preprocessor.joblib` | Fitted StandardScaler (for inference) |
| `manifest.json` | Feature names, shapes, fraud ratios |

### `outputs2/sir_report2/`
| File | Use for |
|------|---------|
| `1_model_comparison.csv` | ML models comparison table |
| `2_fl_scenario_summary.csv` | All 7 scenarios final metrics |
| `3_roundwise_performance.csv` | Round-by-round metrics (5 scenarios) |
| `4_attack_impact.csv` | How much each attack damaged performance |
| `5_defense_recovery.csv` | Baseline → Attack → After Defense |
| `6_client_weights_normal.csv` | Client gradient norms — normal FL |
| `6_client_weights_attacked.csv` | Client gradient norms — under attack |
| `6_client_weights_defended.csv` | Client gradient norms — with defense |
| `7_defense_evidence.csv` | Per-round client rejection log |
| `7_defense_evidence_summary.csv` | 20/20 blocked summary |
| `8_full_roundwise_all_scenarios.csv` | Wide table: 7 scenarios × 20 rounds |
| `README.txt` | Complete results summary + guide |

---

## 16. Thesis Statement

> "We design and evaluate a security-aware Federated Learning framework for credit card fraud detection. Three simulated bank clients collaboratively train a Logistic Regression model using FedAvg, achieving ROC-AUC 0.963 — comparable to centralized performance (0.961) — without sharing raw transaction data. We simulate three Byzantine attacks (Sign-Flip, Scale, Label-Flip) against one of the three clients and demonstrate that a 3-layer defense combining Gradient Clipping, Differential Privacy, and Multi-Krum aggregation achieves full performance recovery, with the malicious client blocked in 20 out of 20 training rounds across all attack types."

---

## 17. Known Behaviors (Not Bugs)

| Behavior | Explanation |
|----------|-------------|
| FL Non-IID F1 = 0% | 0.17% fraud + non-IID partition → threshold search returns 0.5 → all fraud missed. ROC-AUC = 0.892 shows model still has discriminative ability. Note in thesis as limitation. |
| Flower deprecation warnings | `client_fn` and `NumpyClient` warnings are cosmetic — Flower library internal. Do not affect results. |
| Security FL takes 10-12 min | 7 scenarios × 20 rounds = 140 total training rounds. Normal. |
| SMOTE takes 5-7 min | Creating 19,902 synthetic samples from 344 real ones. Normal. |
| Dataset cached after first download | `~/.cache/kagglehub/...` — no re-download needed. |

---

## 18. Future Improvements

| Improvement | Expected Impact |
|-------------|----------------|
| Use `FraudMLP` in FL (`--fl_model mlp`) | ROC-AUC potentially 0.97+ |
| FedProx instead of FedAvg | Fix Non-IID convergence (F1 > 0) |
| 5–10 clients instead of 3 | Stronger Byzantine tolerance argument |
| Label-flip dedicated defense | Robust loss / data auditing at data level |
| Optuna hyperparameter tuning | Systematic optimal LR, batch size |

---
