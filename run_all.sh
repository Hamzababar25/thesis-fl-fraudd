#!/usr/bin/env bash
set -euo pipefail

# Run from project root:
#   bash run_all.sh

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
DATA_PATH="${DATA_PATH:-data/Bank_Transaction_Fraud_Detection.xlsx}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs}"

IID_ROUNDS="${IID_ROUNDS:-20}"
NONIID_ROUNDS="${NONIID_ROUNDS:-20}"
ADAPTIVE_ROUNDS="${ADAPTIVE_ROUNDS:-20}"
LR="${LR:-1e-3}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-60000}"

echo "==> [1/8] Creating/using virtual environment"
if [[ ! -d "${VENV_DIR}" ]]; then
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi
source "${VENV_DIR}/bin/activate"

echo "==> [2/8] Installing dependencies"
python -m pip install -r requirements.txt
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch

echo "==> [3/8] Preprocessing dataset"
python src/preprocess.py --data_path "${DATA_PATH}" --output_dir "${OUTPUT_DIR}"

echo "==> [4/8] Centralized baseline"
python src/train_central.py --output_dir "${OUTPUT_DIR}"

echo "==> [5/8] ML single models + hybrids"
python src/train_ml_hybrids.py --output_dir "${OUTPUT_DIR}" --max_train_samples "${MAX_TRAIN_SAMPLES}"
python src/generate_model_report.py --output_dir "${OUTPUT_DIR}" --top_k 10

echo "==> [6/8] FL FedAvg IID"
python src/flwr_server.py --output_dir "${OUTPUT_DIR}" --partition_mode iid --rounds "${IID_ROUNDS}" --lr "${LR}"

echo "==> [7/8] FL FedAvg Non-IID"
python src/flwr_server.py --output_dir "${OUTPUT_DIR}" --partition_mode noniid --rounds "${NONIID_ROUNDS}" --lr "${LR}"

echo "==> [8/8] Adaptive Secure FL (main thesis method)"
python src/flwr_server_adaptive_secure.py --output_dir "${OUTPUT_DIR}" --partition_mode bank_noniid --rounds "${ADAPTIVE_ROUNDS}" --lr "${LR}" --secure_agg

echo
echo "Pipeline complete."
echo "Key outputs:"
echo "- ${OUTPUT_DIR}/metrics/centralized_results.csv"
echo "- ${OUTPUT_DIR}/metrics/ml_single_results.csv"
echo "- ${OUTPUT_DIR}/metrics/ml_hybrid_results.csv"
echo "- ${OUTPUT_DIR}/metrics/fl_results_iid.csv"
echo "- ${OUTPUT_DIR}/metrics/fl_results_noniid.csv"
echo "- ${OUTPUT_DIR}/metrics/fl_results_adaptive_secure.csv"
echo "- ${OUTPUT_DIR}/metrics/model_comparison_report.md"
