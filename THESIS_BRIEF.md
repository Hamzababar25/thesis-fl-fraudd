# Thesis Brief — FL Fraud Detection Security
**Student:** Hamza Babar | **Deadline:** Tuesday, June 17, 2026
**GitHub:** https://github.com/Hamzababar25/thesis-fl-fraudd

---

## Problem Statement

Banks need to collaborate to build better fraud detection models, but they cannot share customer data due to privacy laws (GDPR, banking regulations). Federated Learning (FL) solves the data sharing problem — multiple banks train a shared model by only exchanging model weights, never raw data.

However, FL has a critical vulnerability: **what if one bank is malicious?** A compromised or dishonest bank can send corrupted model updates to the central server, deliberately degrading the global fraud detection model. This is called a **Poisoning Attack** — and standard FL has no protection against it.

**Research Question:**
> How can multiple banks collaboratively train a fraud detection model using FL, while staying secure against participants who deliberately poison the shared model?

---

## Dataset

- **Name:** Bank Transaction Fraud Detection (Kaggle)
- **Size:** 200,000 bank transactions
- **Split:** 140K train / 30K validation / 30K test
- **Target:** Is_Fraud (0 = normal, 1 = fraud) — ~5% fraud rate (imbalanced)
- **Features used:** Age, Transaction Amount, Account Balance, time features (hour, day, weekend, night), location (State), Gender — total 521 features after encoding
- **Note:** This is a synthetic dataset. Features have low correlation with fraud labels (ROC-AUC ≈ 0.50 across all models). The thesis focuses on security framework evaluation, not maximizing accuracy.

---

## Models Applied

### Centralized Comparison (3 models)

| Model | F1 | Recall | ROC-AUC |
|-------|-----|--------|---------|
| Random Forest ← Best | 9.62% | 96.36% | 0.5072 |
| Logistic Regression | 9.60% | 99.87% | 0.5032 |
| XGBoost | 9.59% | 99.07% | 0.5027 |

### Federated Learning Model

**Logistic Regression** (deployed in FL)

Random Forest cannot be used in FL because tree-based models cannot be mathematically averaged with FedAvg. Only models with numeric weight vectors (linear/neural) work in FL.

- 3 bank clients (w1, w2, w3) — Non-IID data distribution
- 20 training rounds
- Malicious client: w3

---

## Attacks Implemented

### 1. Sign-Flip Attack (Model Poisoning)
After local training, client w3 reverses its update direction and amplifies by 5x before sending to server. The global model gets pulled in the wrong direction.
- **Result:** Recall dropped from 98.8% → 35.1% (SEVERE)

### 2. Scale Attack (Model Poisoning)
Client w3 sends its normal update amplified 5x, making its influence disproportionately large in the average.
- **Result:** Recall dropped from 98.8% → 97.8% (MILD)

### 3. Label-Flip Attack (Data Poisoning) — Novel Contribution
Before training, w3 flips all fraud labels in its local data (1→0). The model learns that fraud is normal, then sends those corrupted weights.
- **Result:** Recall collapsed to 0.3% at Round 5, partially recovered to 9.4% by Round 20

---

## Defense Applied

Three mechanisms combined:
1. **Gradient Clipping** — limits update magnitude (clip threshold = 1.0)
2. **Differential Privacy Noise** — adds Gaussian noise to updates (noise multiplier = 0.01)
3. **Multi-Krum Aggregation** — detects and rejects the most outlier client update each round

**Defense Result:** Malicious client w3 was detected and rejected in **20 out of 20 rounds** across all defended scenarios.

---

## Key Results (7 Scenarios)

| Scenario | Recall | Verdict |
|----------|--------|---------|
| Normal FL (no attack) | 98.81% | Baseline |
| Sign-Flip — no defense | 35.10% | SEVERE degradation |
| Sign-Flip + defense | 99.34% | Fully recovered |
| Scale — no defense | 97.75% | Mild degradation |
| Scale + defense | 99.41% | Fully recovered |
| Label-Flip — no defense | 96.76% | Mild at final round |
| Label-Flip + defense | 98.41% | Mostly recovered |

---

*Full results with charts: download `outputs/analysis/report.html` from GitHub and open in any browser.*
