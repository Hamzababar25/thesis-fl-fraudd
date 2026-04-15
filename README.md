# Federated Fraud Detection (Flower + PyTorch)

This project builds a leakage-safe fraud detection pipeline on:

- `data/Bank_Transaction_Fraud_Detection.xlsx`
- Target: `Is_Fraud` (binary 0/1, imbalanced)

It includes:

- Preprocessing + feature engineering for tabular data
- Centralized baseline (logistic regression with class weights)
- Federated training with Flower `FedAvg` and 3 clients (`w1`, `w2`, `w3`)
- Evaluation with imbalance-aware metrics: F1, Recall, Precision, ROC-AUC, PR-AUC

## Project Structure

- `src/preprocess.py`: load excel, detect columns, clean, feature engineering, split, fit preprocessor
- `src/train_central.py`: centralized training/evaluation
- `src/flwr_client.py`: Flower client implementation
- `src/flwr_server.py`: Flower simulation server (`FedAvg`)
- `src/flwr_server_adaptive_secure.py`: adaptive fraud-aware weighted FL with secure-aggregation simulation
- `src/common.py`: shared model, dataloaders, metrics, plots
- `outputs/`: generated metrics, plots, processed artifacts

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install --index-url https://download.pytorch.org/whl/cpu torch
```

## 1) Preprocess

```bash
python src/preprocess.py --data_path data/Bank_Transaction_Fraud_Detection.xlsx --output_dir outputs
```

What it does:

- Drops identity-like columns (`id/name/account/email/contact/description` and unnamed columns)
- Detects amount/balance columns by keyword
- Detects datetime from `date/time/timestamp` keywords
- Adds engineered features:
  - `amount_to_balance_ratio`
  - `amount_minus_balance`
  - `is_over_balance`
  - `log_amount`
  - `tx_hour`, `tx_day_of_week`, `is_weekend`, `is_night`
- Splits train/val/test with stratification
- Fits transformers only on train split
- Saves preprocessed arrays and manifest to `outputs/processed/`

## 2) Train Centralized Baseline

```bash
python src/train_central.py --output_dir outputs
```

## 2.1) Train 4 ML Models + Hybrid (2-model) Ensembles

This script trains 4 single models:

- Logistic Regression
- SGD (log-loss)
- Calibrated Linear SVC
- Bernoulli Naive Bayes

Then it builds all pairwise 2-model hybrids (average probabilities), and reports:

- Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC

```bash
python src/train_ml_hybrids.py --output_dir outputs
```

Optional faster run (for quick smoke checks):

```bash
python src/train_ml_hybrids.py --output_dir outputs --max_train_samples 60000
```

Artifacts:

- `outputs/metrics/ml_single_results.csv`
- `outputs/metrics/ml_hybrid_results.csv`
- `outputs/metrics/ml_comparison_summary.json`
- confusion matrices under `outputs/plots/` with prefixes `cm_single_` and `cm_hybrid_`

Artifacts:

- `outputs/metrics/centralized_metrics.json`
- `outputs/metrics/centralized_classification_report.json`
- `outputs/metrics/centralized_results.csv`
- `outputs/plots/centralized_confusion_matrix.png`
- `outputs/plots/centralized_roc_curve.png`
- `outputs/plots/centralized_pr_curve.png`

## 3) Train Federated (IID)

```bash
python src/flwr_server.py --output_dir outputs --partition_mode iid --rounds 20 --lr 1e-3
```

## 4) Train Federated (Optional Non-IID)

```bash
python src/flwr_server.py --output_dir outputs --partition_mode noniid --rounds 20 --lr 1e-3
```

## 5) Adaptive Secure Aggregation (Thesis Extension)

This run adds:

- non-IID bank-style split (`bank_noniid`: low/medium/high fraud clients)
- fraud-aware client weighting (uses local F1 and fraud ratio)
- adaptive weight update across rounds
- secure aggregation simulation via deterministic masking/unmasking at aggregate level

```bash
python src/flwr_server_adaptive_secure.py --output_dir outputs --partition_mode bank_noniid --rounds 20 --lr 1e-3 --secure_agg
```

Additional artifacts:

- `outputs/metrics/fl_round_client_weights_adaptive_secure.csv`
- `outputs/metrics/fl_round_metrics_adaptive_secure.csv`
- `outputs/metrics/fl_final_metrics_adaptive_secure.json`
- `outputs/metrics/fl_results_adaptive_secure.csv`
- `outputs/plots/fl_confusion_matrix_adaptive_secure.png`
- `outputs/plots/fl_roc_curve_adaptive_secure.png`
- `outputs/plots/fl_pr_curve_adaptive_secure.png`

Artifacts:

- `outputs/metrics/fl_round_metrics_iid.csv`
- `outputs/metrics/fl_final_metrics_iid.json`
- `outputs/metrics/fl_results_iid.csv`
- `outputs/plots/fl_confusion_matrix_iid.png`
- `outputs/plots/fl_roc_curve_iid.png`
- `outputs/plots/fl_pr_curve_iid.png`

And similarly for `noniid`.

## Notes on Leakage and Imbalance

- Split is performed before any balancing
- No SMOTE is applied to validation/test
- Class imbalance handling uses class weighting (`class_weight=balanced` for centralized, `pos_weight` for federated local BCE loss)
- Preprocessor fit is train-only, then reused for val/test

## Python Version

- Linux, Python `3.10+`
