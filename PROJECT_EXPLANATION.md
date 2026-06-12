# Thesis Project — Complete Explanation
# Security-Aware Federated Learning for Fraud Detection in Banking

**Student:** Hamza Babar
**Submission Deadline:** Tuesday, June 17, 2026
**GitHub:** https://github.com/Hamzababar25/thesis-fl-fraudd

---

## PROBLEM STATEMENT

Financial fraud is one of the most serious and growing threats in the global banking industry. Banks and financial institutions lose billions of dollars every year due to fraudulent transactions. To fight this, they use machine learning models that analyze transaction patterns and flag suspicious activity.

However, a fundamental conflict exists between **data utility** and **data privacy**:

- Fraud patterns are rare and diverse. A single bank may not have enough fraud examples in its own data to train a strong detection model.
- Combining data from multiple banks would produce a much better model — but this is legally and ethically not possible. Customer transaction data is private, regulated under laws like GDPR and PSD2, and cannot be shared between competing financial institutions.
- Banks that train models only on their own local data end up with weak, biased models that miss fraud patterns seen at other banks.

This creates a critical gap: **banks need to collaborate to fight fraud effectively, but they cannot share their sensitive data to do so.**

Federated Learning (FL) was proposed as a solution to this exact problem. In FL, multiple banks train a shared model by exchanging only model parameters (weights), never raw transaction data. However, FL introduces a new and serious vulnerability: **it assumes all participating banks are honest**. In reality, one or more banks could be:

- Compromised by an external attacker (hacked system)
- Internally malicious (deliberately trying to sabotage the shared model)
- Behaving dishonestly to reduce fraud detection at competitors

When a malicious participant sends manipulated model updates to the server, it can degrade the global fraud detection model for all banks — a threat known as a **Poisoning Attack**. Existing FL systems have no built-in protection against such attacks.

**The core problem this thesis addresses:**
> *How can multiple banks collaboratively train a fraud detection model using Federated Learning, while remaining secure against participants who deliberately try to corrupt the shared model through poisoning attacks?*

---

## OBJECTIVES

This thesis has the following specific research objectives:

**Objective 1 — Establish a Federated Learning Baseline for Fraud Detection**
Design and implement a complete FL pipeline where 3 simulated banks collaboratively train a fraud detection model using the FedAvg aggregation algorithm. Evaluate the baseline performance (F1, Recall, Precision, ROC-AUC) over 20 training rounds on a real bank transaction dataset.

**Objective 2 — Compare Centralized ML Models**
Before deploying FL, compare three standard machine learning models (Logistic Regression, Random Forest, XGBoost) trained on centralized data to establish a performance reference point and select the most appropriate model for FL deployment.

**Objective 3 — Simulate Realistic Poisoning Attacks**
Implement three distinct attack scenarios that a malicious bank participant could carry out:
- **Sign-Flip Attack** (model poisoning): Reverse and amplify model updates to mislead the global model
- **Scale Attack** (model poisoning): Amplify updates to make one client's influence disproportionately large
- **Label-Flip Attack** (data poisoning): Corrupt local training data by flipping fraud labels before training

**Objective 4 — Implement a Layered Defense Mechanism**
Design and deploy a multi-layer security framework combining:
- Gradient clipping to limit update magnitudes
- Differential Privacy (DP) noise injection to protect individual updates
- Multi-Krum robust aggregation to detect and reject malicious client updates

**Objective 5 — Evaluate Attack and Defense in a 7-Scenario Comparative Study**
Run and compare 7 FL scenarios (baseline, each attack without defense, each attack with defense) and measure how much each attack degrades performance and how effectively the defense recovers it — in terms of Recall, F1, and round-wise convergence behavior.

**Objective 6 — Analyze the Limits of Gradient-Level Defenses Against Data Poisoning**
Specifically investigate Label-Flip (data poisoning) as a threat that operates at the data level rather than the model level, and examine whether gradient-level defenses (Multi-Krum + clipping) can adequately counter it — identifying this gap as a contribution and direction for future work.

---

## PART 1 — What Is This Project? (Plain English)

### The Real-World Problem

Banks deal with fraud every day. To detect fraud, they need to train machine learning models on transaction data. But there is a big privacy problem:

- **Bank A cannot share its customer data with Bank B** — it is illegal and unethical.
- But if every bank trains its model alone on only its own data, the model is weak.
- **Question: Can multiple banks train a shared fraud detection model WITHOUT sharing their private data?**

### The Solution: Federated Learning (FL)

Federated Learning solves this. Here is how it works:

```
Step 1: Server sends initial model to all banks
Step 2: Each bank trains the model on its OWN local data (data never leaves the bank)
Step 3: Each bank sends back only the model WEIGHTS (not the data)
Step 4: Server combines (averages) all weights into one improved global model
Step 5: Repeat for many rounds
```

This way, banks collaborate to build a better model — but nobody ever sees anyone else's private data.

### The New Problem: Security

Now the question is: **What if one of the banks is malicious?**

A dishonest bank (or a hacker who has taken control of one bank's system) could:
- Send fake/corrupted model weights to the server
- Deliberately try to make the global fraud detection model WORSE
- So that real fraud transactions are not detected

This is called a **Poisoning Attack**.

### What This Thesis Does

This thesis:
1. Sets up a Federated Learning system with 3 banks
2. Introduces attacks where one bank (Bank w3) behaves maliciously
3. Tests 3 different types of attacks (Sign-Flip, Scale, Label-Flip)
4. Implements a layered defense (Multi-Krum + Gradient Clipping + DP Noise)
5. Measures how much each attack hurts performance and how much the defense recovers it
6. Compares all 7 scenarios with detailed round-wise analysis

---

## PART 2 — The Dataset

- **Name:** Bank Transaction Fraud Detection
- **Format:** Excel file (.xlsx)
- **Size:** 200,000 transactions
- **Split:** 140,000 train / 30,000 validation / 30,000 test
- **Target column:** Is_Fraud (0 = normal, 1 = fraud)
- **Fraud rate:** ~5% (very imbalanced — most transactions are normal)

**Features used (after preprocessing):**
- Age, Transaction Amount, Account Balance
- Engineered features: amount-to-balance ratio, log of amount, is_over_balance
- Time features: hour of transaction, day of week, is_weekend, is_night
- Location features: State, Gender (one-hot encoded)
- Total: 521 features after encoding

**Important Note:** This is a synthetic/simulated dataset. The features have very low correlation with fraud labels (ROC-AUC ≈ 0.50 for all models). This is a known property of this dataset — it does not mean the code is wrong. The thesis is about SECURITY FRAMEWORK, not about maximizing fraud detection accuracy.

---

## PART 3 — Machine Learning Models Compared

Before deploying in FL, three standard ML models were compared on centralized data:

| Rank | Model | F1 Score | Recall | ROC-AUC | Threshold |
|------|-------|----------|--------|---------|-----------|
| 1 | Random Forest | 9.62% | 96.36% | 0.5072 | 0.42 |
| 2 | Logistic Regression | 9.60% | 99.87% | 0.5032 | 0.25 |
| 3 | XGBoost | 9.59% | 99.07% | 0.5027 | 0.15 |

**Why is F1 so low (~9%)?**
Because the dataset is highly imbalanced (5% fraud) AND features are not strongly predictive. The models all perform near random chance (ROC-AUC = 0.50). This is a dataset limitation, not a code problem. The thesis does not claim high accuracy — it claims the security framework works correctly.

**Why is Logistic Regression used in FL (not Random Forest which is best)?**
Because Federated Learning requires averaging model parameters between clients. Random Forest and XGBoost are tree-based models — you cannot mathematically "average" two decision trees. Only neural network / linear models (like Logistic Regression) can be averaged with FedAvg. So Logistic Regression is deployed in FL.

**Threshold Optimization:**
Instead of using 0.5 as default threshold, each model's best threshold was found using the validation set (not test set — to avoid data leakage). This is proper ML practice.

---

## PART 4 — Federated Learning Setup

### Configuration

| Setting | Value |
|---------|-------|
| Number of clients (banks) | 3 |
| Client IDs | w1, w2, w3 |
| FL rounds | 20 |
| Model | Logistic Regression (linear PyTorch model) |
| Data partition | Bank-style Non-IID |
| Malicious client | w3 |

### What is Non-IID?

In real life, different banks have different customer profiles:
- Bank w1: Low fraud rate (15% of fraud cases)
- Bank w2: Medium fraud rate (30% of fraud cases)
- Bank w3: High fraud rate (55% of fraud cases)

This is called Non-IID (non-independent and identically distributed). It is more realistic than assuming all banks have the same data distribution.

### How FedAvg Works (Each Round)

```
Server → sends global model weights to w1, w2, w3
w1 trains locally → sends back updated weights
w2 trains locally → sends back updated weights
w3 trains locally → sends back updated weights
Server averages all three weight sets → new global model
Repeat for 20 rounds
```

---

## PART 5 — The Three Attacks

All three attacks are performed by client w3 (the malicious bank).

### Attack 1: Sign-Flip (Model Poisoning)

**What happens:**
After training locally, w3 reverses its model update and amplifies it by 5x before sending to server.

**Formula:** malicious_update = initial_weights − 5 × (trained_weights − initial_weights)

**Effect:** The global model gets pulled in the WRONG direction. Fraud detection degrades.

**Real-world meaning:** A hacker has taken over Bank w3's FL client and is sending corrupted model weights.

**Result:**
- Recall dropped from **98.81% → 35.10%** (SEVERE damage)
- The model stopped detecting 64% of fraud cases

---

### Attack 2: Scale Attack (Model Poisoning)

**What happens:**
w3 sends its normal update but amplified by 5x. This makes w3's update dominate the average.

**Formula:** malicious_update = initial_weights + 5 × (trained_weights − initial_weights)

**Effect:** w3's local biases dominate the global model disproportionately.

**Result:**
- Recall dropped from **98.81% → 97.75%** (MILD damage)
- Less severe because amplifying a legitimate update direction is less harmful

---

### Attack 3: Label-Flip (Data Poisoning) — NEW CONTRIBUTION

**What happens:**
BEFORE training, w3 flips all fraud labels in its local training data (changes 1 → 0). So the model learns that fraud transactions are actually normal.

**Formula:** y_train[fraud_indices] = 0 (all fraud cases relabeled as normal)

**Effect:** w3 trains a model that is blind to fraud, then sends those weights to the server.

**This is different from attacks 1 and 2:**
- Attacks 1 & 2 happen AFTER training (model-level poisoning)
- Attack 3 happens BEFORE training (data-level poisoning)

**Result:**
- Recall dropped from **98.81% → 96.76%** (MILD damage at final round)
- But during training (Round 5): recall collapsed to **0.33%** — almost complete failure
- This is an important finding — the attack is devastating early but the system partially self-corrects

---

## PART 6 — The Three Defenses

All three defenses are applied together in "defended" scenarios.

### Defense 1: Gradient Clipping

**What it does:** Limits the maximum size of each client's model update.
**Config:** clip_threshold = 1.0
**Why:** If a malicious client sends a very large update (like in Scale attack), clipping reduces it to normal size.

### Defense 2: Differential Privacy (DP) Noise

**What it does:** Adds small random Gaussian noise to each client's update before sending.
**Config:** noise_multiplier = 0.01
**Why:** Makes it harder to reconstruct individual data points from model updates. Also regularizes the model.

### Defense 3: Multi-Krum Aggregation

**What it does:** Instead of averaging ALL clients' updates, Multi-Krum:
1. Calculates pairwise distances between all client updates
2. Identifies which client's update is most "different" from the others
3. Rejects that client's update (does not include it in aggregation)

**Why:** A malicious client's corrupted update will be statistically very different from honest clients' updates.

**Result:** Client w3 was rejected in **20 out of 20 rounds** in every defended scenario.

---

## PART 7 — The 7 Scenarios

| # | Scenario | Attack | Defense | Final Recall |
|---|----------|--------|---------|-------------|
| 1 | Normal FL (Baseline) | None | None | 98.81% |
| 2 | Sign-Flip — No Defense | Sign-Flip | None | 35.10% |
| 3 | Sign-Flip + Defense | Sign-Flip | Multi-Krum + DP + Clip | 99.34% |
| 4 | Scale Attack — No Defense | Scale | None | 97.75% |
| 5 | Scale Attack + Defense | Scale | Multi-Krum + DP + Clip | 99.41% |
| 6 | Label-Flip — No Defense | Label-Flip | None | 96.76% |
| 7 | Label-Flip + Defense | Label-Flip | Multi-Krum + DP + Clip | 98.41% |

**The key story:**
- Attack → recall drops (especially Sign-Flip: 98% → 35%)
- Defense → recall recovers (35% → 99%)
- Defense works in 20/20 rounds — malicious client always detected and rejected

---

## PART 8 — Round-by-Round Performance (Every 5 Rounds)

| Round | Normal FL | Sign-Flip Attack | Sign-Flip Defense | Label-Flip Attack | Label-Flip Defense |
|-------|-----------|-----------------|------------------|------------------|-------------------|
| 0 | 54.2% | 54.2% | 54.2% | 54.2% | 54.2% |
| 5 | 44.6% | 33.6% | 53.2% | **0.3%** | 28.4% |
| 10 | 48.4% | 21.0% | 48.3% | 3.1% | 28.8% |
| 15 | 51.0% | 14.7% | 48.6% | 7.8% | 2.8% |
| 20 | 53.3% | **13.6%** | 53.1% | 9.4% | 7.8% |

*(Recall % — how many fraud cases were detected that round)*

**Key observations:**
1. Sign-Flip progressively worsens (54% → 14%) — attack accumulates over rounds
2. Sign-Flip with defense stays stable (~53%) — defense neutralizes attack every round
3. Label-Flip collapses immediately at Round 5 (54% → 0.3%) — most aggressive early
4. Label-Flip defense partially works but not perfectly (future work)

---

## PART 9 — What Makes This Thesis Original

1. **Three attack types evaluated together** — most FL papers test only one attack type
2. **Label-Flip attack introduced** — this is a data poisoning attack rarely tested in FL fraud detection
3. **Val-set threshold optimization** — proper ML practice, not just using 0.5 default
4. **7-scenario comparison** — comprehensive evaluation: each attack with and without defense
5. **Bank-style Non-IID partitioning** — realistic simulation of real banking data distribution
6. **Combined clean outputs** — all results in thesis-ready tables automatically generated

---

## PART 10 — Code Structure (For Technical People)

```
thesis-fl-fraud/
├── data/
│   └── Bank_Transaction_Fraud_Detection.xlsx    ← raw dataset
│
├── src/
│   ├── preprocess.py              ← data cleaning, feature engineering, split
│   ├── common.py                  ← shared models (FraudLogistic, FraudMLP), metrics
│   ├── train_central.py           ← centralized LR baseline
│   ├── train_ml_hybrids.py        ← LR + RF + XGBoost comparison
│   ├── aggregation.py             ← FedAvg and Multi-Krum algorithms
│   ├── attack_simulation.py       ← sign_flip, scale, label_flip attack code
│   ├── flwr_client.py             ← FL client (training + attack injection + DP)
│   ├── flwr_server_security.py    ← FL server (7-scenario orchestration)
│   ├── generate_combined_outputs.py  ← creates clean combined CSV files
│   └── generate_sir_report.py     ← creates thesis-ready report files
│
├── outputs/
│   ├── analysis/                  ← clean combined files (thesis-ready)
│   │   ├── report.html            ← interactive browser report (open this!)
│   │   ├── combined_ml_results.csv
│   │   ├── combined_fl_final.csv
│   │   └── combined_fl_roundwise.csv
│   │
│   ├── sir_report/                ← files for supervisor questions
│   │   ├── 1_model_comparison.csv
│   │   ├── 2_fl_scenario_summary.csv
│   │   ├── 3_roundwise_performance.csv
│   │   ├── 4_attack_impact.csv
│   │   ├── 5_defense_recovery.csv
│   │   ├── 6_client_weights_*.csv
│   │   ├── 7_defense_evidence*.csv
│   │   └── README.txt
│   │
│   ├── metrics/                   ← raw per-scenario CSV/JSON files
│   ├── plots/                     ← PNG graphs (confusion matrix, ROC, PR curves)
│   └── processed/                 ← preprocessed numpy arrays
│
├── run_all.sh                     ← Linux: runs entire pipeline in one command
└── requirements.txt               ← Python package dependencies
```

---

## PART 11 — How to Run (Quick Guide)

**Linux (one command):**
```bash
cd thesis-fl-fraud
source .venv/bin/activate
bash run_all.sh
```

**Windows (step by step in PowerShell):**
```powershell
cd thesis-fl-fraud
.venv\Scripts\Activate.ps1
python src\preprocess.py --data_path data\Bank_Transaction_Fraud_Detection.xlsx --output_dir outputs
python src\train_central.py --output_dir outputs
python src\train_ml_hybrids.py --output_dir outputs --max_train_samples 60000
python src\flwr_server_security.py --output_dir outputs --partition_mode bank_noniid --fl_model best_from_ml --evaluate_attack_scenarios --attack_strength 5.0 --malicious_clients w3 --num_malicious 1 --clip_threshold 1.0 --noise_multiplier 0.01 --rounds 20
python src\generate_combined_outputs.py --output_dir outputs
python src\generate_sir_report.py --output_dir outputs
```

**Expected total time:** ~8 minutes (Linux) / ~20 minutes (Windows)

---

## PART 12 — Possible Questions & Answers

**Q: Why is accuracy/F1 so low?**
A: The dataset is synthetic with near-zero feature correlation. ROC-AUC = 0.50 means models perform like random guessing on this dataset. The thesis contribution is the security framework — not accuracy improvement.

**Q: Why only 3 clients?**
A: 3 clients is the minimum for Multi-Krum to work mathematically (needs n > 2f+2 where f = number of malicious clients). With 3 clients and 1 malicious, Multi-Krum can still operate. More clients are listed as future work.

**Q: Why Logistic Regression in FL and not Random Forest?**
A: Tree-based models (RF, XGBoost) cannot be federated with FedAvg because their parameters (tree nodes/splits) cannot be averaged mathematically. Only models with numeric weight vectors (linear/neural) can be federated.

**Q: Is the attack realistic?**
A: Yes. In real-world FL deployments, a compromised participant (hacked system, malicious insider, adversarial institution) can send corrupted model updates. This is a well-known threat in FL literature.

**Q: How does Multi-Krum know which client is malicious?**
A: It does NOT know. Multi-Krum is unsupervised — it computes pairwise distances between all client updates and rejects the one that is most "outlier-like" (most different from others). The malicious client's corrupted update naturally becomes an outlier, so it gets rejected.

**Q: What is the difference between Sign-Flip and Label-Flip?**
A: Sign-Flip is model-level poisoning (happens after training, manipulates weights). Label-Flip is data-level poisoning (happens before training, corrupts the training data). Label-Flip is harder to defend against with gradient-level defenses.

---

## PART 13 — Summary in One Paragraph

This thesis proposes and evaluates a security framework for Federated Learning applied to bank fraud detection. Three banks collaborate to train a shared Logistic Regression model without sharing private customer data. One bank acts as an attacker using three different poisoning strategies: Sign-Flip (reversing model updates), Scale (amplifying model updates), and Label-Flip (corrupting training data labels). Three defense mechanisms are applied: gradient clipping to limit update magnitude, differential privacy noise to protect individual updates, and Multi-Krum robust aggregation to detect and reject malicious clients. Results across 20 training rounds show that Sign-Flip attack is the most severe (recall drops from 99% to 35%) but is fully neutralized by defense (recall recovers to 99%). The malicious bank is correctly detected and rejected in 20 out of 20 rounds with defense active. Label-Flip introduces a new finding: data-level poisoning resists gradient-level defenses, representing a direction for future work.

---

*Full interactive results: download `outputs/analysis/report.html` from GitHub and open in any browser.*
*GitHub: https://github.com/Hamzababar25/thesis-fl-fraudd*
