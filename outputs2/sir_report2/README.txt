FL FRAUD DETECTION — SIR REPORT FILES
======================================
Generated from: outputs2/metrics/ and outputs2/processed/
Report folder : outputs2/sir_report2

DATASET USED:
-------------
  Kaggle Credit Card Fraud Detection (mlg-ulb/creditcardfraud)
  - 284,807 real transactions (2 days, European cardholders)
  - 492 fraud cases (0.17% — severely imbalanced)
  - 30 features: V1-V28 (PCA-anonymized by dataset provider) + Amount + Time
  - Added features: log_amount, time_hour, time_sin, time_cos, is_night (33 total)
  - Split: 70% Train / 15% Validation / 15% Test (stratified)
  - NO One-Hot Encoding needed (all features are numeric)
  - SMOTE used to handle class imbalance during training

FILE GUIDE:
-----------

1_model_comparison.csv
  → ML models comparison: LR (Centralized) vs LR vs Random Forest vs XGBoost
  → Best ROC-AUC: Random Forest = 0.9685
  → Best F1:      XGBoost      = 0.8529
  → FL uses Logistic Regression (tree-based models cannot be federated easily)

2_fl_scenario_summary.csv
  → All 7 FL scenarios final metrics (test set evaluation)
  → Shows: Normal FL → Attack → Defense for each attack type
  → Key finding: Sign-Flip attack drops ROC-AUC from 0.936 to 0.090 (model destroyed)
  → Multi-Krum defense recovers ROC-AUC back to 0.931

3_roundwise_performance.csv
  → Round-by-round (Round 0 to 20) for 5 key scenarios
  → Shows: NormalFL, SignFlip_Attack, SignFlip_Defense, LabelFlip_Attack, LabelFlip_Defense
  → Each row = one FL communication round

4_attack_impact.csv
  → Quantifies damage each attack caused
  → Baseline vs Under-Attack: recall drop, F1 drop, loss increase
  → Verdict: SEVERE / MODERATE / MILD

5_defense_recovery.csv
  → Quantifies how much Multi-Krum defense recovered performance
  → Baseline → Under-Attack → After-Defense for Sign-Flip, Scale, Label-Flip
  → "defense_effective": YES = full recovery, PARTIAL = some recovery

6_client_weights_normal.csv     → Gradient update norms in normal FL (3 clients per round)
6_client_weights_attacked.csv   → Gradient norms under sign-flip attack (client w3 is malicious)
6_client_weights_defended.csv   → Gradient norms when Multi-Krum is active

7_defense_evidence.csv          → Per-round log: which client was rejected by Multi-Krum
7_defense_evidence_summary.csv  → Summary: how many rounds malicious client was blocked

8_full_roundwise_all_scenarios.csv
  → Wide table: all 7 scenarios × all 20 rounds
  → Columns: round, normal_fl_roc_auc, sign_flip_no_defense_roc_auc, ... etc.

KEY RESULTS SUMMARY:
--------------------
  Centralized LR baseline   : ROC-AUC = 0.961,  F1 = 0.747
  XGBoost (standalone)      : ROC-AUC = 0.963,  F1 = 0.853  ← Best F1
  Random Forest (standalone) : ROC-AUC = 0.968,  F1 = 0.829  ← Best ROC-AUC
  FL FedAvg IID             : ROC-AUC = 0.963,  F1 = 0.755  (matches centralized)
  FL Normal (Security setup): ROC-AUC = 0.936,  F1 = 0.805
  Sign-Flip Attack (no def) : ROC-AUC = 0.090,  F1 = 0.000  ← Attack SEVERE
  Sign-Flip + Multi-Krum    : ROC-AUC = 0.931,  F1 = 0.795  ← Defense EFFECTIVE
  Scale Attack (no defense) : ROC-AUC = 0.954,  F1 = 0.779
  Scale + Multi-Krum        : ROC-AUC = 0.931,  F1 = 0.795
  Label-Flip (no defense)   : ROC-AUC = 0.940,  F1 = 0.772
  Label-Flip + Multi-Krum   : ROC-AUC = 0.931,  F1 = 0.795

SECURITY FRAMEWORK (3 Layers):
--------------------------------
  Layer 1 — Gradient Clipping  : Each client clips updates (threshold=1.0) before sending
  Layer 2 — DP Noise           : Gaussian noise added after clipping (noise_mult=0.01)
  Layer 3 — Multi-Krum         : Server rejects statistically-outlier client updates

THESIS CONCLUSION:
------------------
  FL + 3-layer security framework achieves ROC-AUC ~0.93 while:
  (a) Protecting individual data privacy (no raw data leaves clients)
  (b) Defending against Sign-Flip, Scale, and Label-Flip attacks
  (c) Maintaining performance comparable to centralized baseline (0.961)
