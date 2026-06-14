#!/usr/bin/env bash
set -euo pipefail

# Run from project root:
#   bash run_all.sh                         # original Excel dataset
#   DATASET=creditcard bash run_all.sh      # Kaggle Credit Card Fraud dataset

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
DATASET="${DATASET:-bank}"                  # bank | creditcard
DATA_PATH="${DATA_PATH:-data/Bank_Transaction_Fraud_Detection.xlsx}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs}"

IID_ROUNDS="${IID_ROUNDS:-20}"
NONIID_ROUNDS="${NONIID_ROUNDS:-20}"
SECURITY_ROUNDS="${SECURITY_ROUNDS:-20}"
LR="${LR:-1e-3}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-60000}"
NUM_MALICIOUS="${NUM_MALICIOUS:-1}"
CLIP_THRESHOLD="${CLIP_THRESHOLD:-1.0}"
NOISE_MULTIPLIER="${NOISE_MULTIPLIER:-0.01}"
ATTACK_STRENGTH="${ATTACK_STRENGTH:-5.0}"
MALICIOUS_CLIENTS="${MALICIOUS_CLIENTS:-w3}"
FL_MODEL="${FL_MODEL:-best_from_ml}"

echo "==> [1/8] Creating/using virtual environment"
if [[ ! -d "${VENV_DIR}" ]]; then
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi
source "${VENV_DIR}/bin/activate"

echo "==> [2/8] Installing dependencies"
python -m pip install -r requirements.txt
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch

echo "==> [3/8] Preprocessing dataset (DATASET=${DATASET})"
if [[ "${DATASET}" == "creditcard" ]]; then
  # Kaggle Credit Card Fraud dataset – all numeric, no OHE needed
  CREDITCARD_CSV="${CREDITCARD_CSV:-${HOME}/.cache/kagglehub/datasets/mlg-ulb/creditcardfraud/versions/3/creditcard.csv}"
  python src/preprocess_creditcard.py --data_path "${CREDITCARD_CSV}" --output_dir "${OUTPUT_DIR}"
else
  python src/preprocess.py --data_path "${DATA_PATH}" --output_dir "${OUTPUT_DIR}"
fi

echo "==> [4/8] Centralized baseline"
python src/train_central.py --output_dir "${OUTPUT_DIR}"

echo "==> [5/8] ML single models (no hybrids)"
python src/train_ml_hybrids.py --output_dir "${OUTPUT_DIR}" --max_train_samples "${MAX_TRAIN_SAMPLES}" --no_hybrid
python src/generate_model_report.py --output_dir "${OUTPUT_DIR}" --top_k 10

echo "==> [6/8] FL FedAvg IID"
python src/flwr_server.py --output_dir "${OUTPUT_DIR}" --partition_mode iid --rounds "${IID_ROUNDS}" --lr "${LR}"

echo "==> [7/8] FL FedAvg Non-IID"
python src/flwr_server.py --output_dir "${OUTPUT_DIR}" --partition_mode noniid --rounds "${NONIID_ROUNDS}" --lr "${LR}"

echo "==> [8/9] Security FL: 7-scenario attack evaluation (sign_flip + scale + label_flip)"
python src/flwr_server_security.py \
  --output_dir "${OUTPUT_DIR}" \
  --partition_mode bank_noniid \
  --rounds "${SECURITY_ROUNDS}" \
  --lr "${LR}" \
  --fl_model "${FL_MODEL}" \
  --evaluate_attack_scenarios \
  --attack_strength "${ATTACK_STRENGTH}" \
  --malicious_clients "${MALICIOUS_CLIENTS}" \
  --num_malicious "${NUM_MALICIOUS}" \
  --clip_threshold "${CLIP_THRESHOLD}" \
  --noise_multiplier "${NOISE_MULTIPLIER}"

echo "==> [9/9] Generate combined clean output files"
python src/generate_combined_outputs.py --output_dir "${OUTPUT_DIR}"

echo
echo "Pipeline complete."
echo "Key combined outputs (thesis-ready):"
echo "- ${OUTPUT_DIR}/analysis/combined_ml_results.csv"
echo "- ${OUTPUT_DIR}/analysis/combined_fl_final.csv"
echo "- ${OUTPUT_DIR}/analysis/combined_fl_roundwise.csv"
echo "- ${OUTPUT_DIR}/analysis/combined_client_security.csv"
echo "- ${OUTPUT_DIR}/analysis/master_summary.json"
echo "- ${OUTPUT_DIR}/analysis/thesis_table_ml.txt"
echo "- ${OUTPUT_DIR}/analysis/thesis_table_fl.txt"
echo "- ${OUTPUT_DIR}/analysis/thesis_table_roundwise.txt"
