FL FRAUD DETECTION — SIR REPORT FILES
======================================
Generated from: outputs/metrics/ and outputs/processed/

FILE GUIDE:
-----------

1_model_comparison.csv
  → ML models comparison: LR vs Random Forest vs XGBoost vs Hybrids
  → Best single model: Random Forest (F1=9.62%)
  → FL uses Logistic Regression (tree-based models cannot be federated)

2_fl_scenario_summary.csv
  → All 7 FL scenarios final metrics
  → Shows: Normal FL → Attack → Defense for each attack type
  → Key story: Recall drops under attack, recovers with defense

3_roundwise_performance.csv
  → Round-by-round (Round 1 to 20) for 5 key scenarios
  → Shows NormalFL, SignFlip_Attack, SignFlip_Defense, LabelFlip_Attack, LabelFlip_Defense
  → Each row = one FL round

4_attack_impact.csv
  → How much each attack degraded performance
  → Baseline vs Under-Attack: recall drop, F1 drop, verdict (SEVERE/MODERATE/MILD)

5_defense_recovery.csv
  → How much defense recovered performance
  → Baseline → Under-Attack → After-Defense for each attack type
  → "defense_effective" column: YES = full recovery, PARTIAL = some recovery

6_client_weights_normal.csv     → Client gradient norms in normal FL (no attack)
6_client_weights_attacked.csv   → Client gradient norms under sign-flip attack
6_client_weights_defended.csv   → Client gradient norms with Multi-Krum defense

7_defense_evidence.csv          → Per-round: which client was rejected by Multi-Krum
7_defense_evidence_summary.csv  → Summary: X/20 rounds malicious client was blocked

8_full_roundwise_all_scenarios.csv
  → Wide table: all 7 scenarios × all 20 rounds
  → Columns: round, normal_fl_f1, sign_flip_no_defense_f1, ... etc.

THESIS CONTEXT:
---------------
- Dataset: Bank Transaction Fraud (synthetic, ~5% fraud rate)
- ROC-AUC ≈ 0.50 for all models (dataset has near-zero feature-fraud correlation)
- This is a SECURITY study, not an accuracy competition
- The thesis proves: attack degrades recall, defense recovers it
- Malicious client w3 rejected in 20/20 rounds under sign-flip and scale defense
