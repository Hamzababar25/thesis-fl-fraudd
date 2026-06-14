"""Preprocessing pipeline for the Kaggle Credit Card Fraud Detection dataset
(mlg-ulb/creditcardfraud).

Dataset structure
-----------------
- 284,807 rows, 31 columns
- Features: Time, V1-V28 (PCA-transformed), Amount
- Target: Class  (0 = normal, 1 = fraud)
- Fraud rate: ~0.17%  (severely imbalanced)
- No null values, all numeric (no One-Hot Encoding needed)

Splits produced
---------------
  Train  70% (~199,365 rows)
  Val    15% (~42,721 rows)
  Test   15% (~42,721 rows)

Outputs (same format as preprocess.py so the rest of the pipeline is unchanged)
----------------------------------------------------------------------
  outputs/processed/train_X.npz  / train_y.npy
  outputs/processed/val_X.npz    / val_y.npy
  outputs/processed/test_X.npz   / test_y.npy
  outputs/processed/train_X_dense.npy  (for PyTorch / Flower)
  outputs/processed/val_X_dense.npy
  outputs/processed/test_X_dense.npy
  outputs/processed/preprocessor.joblib
  outputs/processed/manifest.json
  outputs/metrics/split_fraud_ratio.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
TARGET_COL = "Class"

# Default path – kagglehub cache location (also accept --data_path override)
DEFAULT_CSV = Path.home() / ".cache/kagglehub/datasets/mlg-ulb/creditcardfraud/versions/3/creditcard.csv"
DEFAULT_OUTPUT = Path("outputs")


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add interpretable features on top of the PCA columns."""
    x = df.copy()

    # Log-transform Amount to reduce skew (same idea as original preprocess.py)
    x["log_amount"] = np.log1p(x["Amount"])

    # Time is seconds from first transaction (covers ~48 hours).
    # Convert to hour-of-day so the model can detect night/day patterns.
    seconds_in_day = 86_400
    x["time_hour"] = (x["Time"] % seconds_in_day) / 3_600          # 0-24

    # Cyclic encoding so 23:59 and 00:01 are "close" in feature space
    x["time_sin"] = np.sin(2 * np.pi * x["time_hour"] / 24)
    x["time_cos"] = np.cos(2 * np.pi * x["time_hour"] / 24)

    # Binary flag: transactions between midnight and 05:00
    x["is_night"] = (x["time_hour"] < 5).astype(float)

    return x


# ---------------------------------------------------------------------------
# Split helper
# ---------------------------------------------------------------------------

def split_data(df: pd.DataFrame):
    """Stratified 70 / 15 / 15 split."""
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        stratify=df[TARGET_COL],
        random_state=RANDOM_STATE,
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df[TARGET_COL],
        random_state=RANDOM_STATE,
    )
    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def save_split(x_arr: np.ndarray, y_arr: np.ndarray, split: str, out_dir: Path):
    sp = sparse.csr_matrix(x_arr.astype(np.float32))
    sparse.save_npz(out_dir / f"{split}_X.npz", sp)
    np.save(out_dir / f"{split}_y.npy", y_arr.astype(np.int64))
    np.save(out_dir / f"{split}_X_dense.npy", x_arr.astype(np.float32))


def class_ratio(y: np.ndarray) -> dict:
    return {
        "n": int(len(y)),
        "fraud_count": int(y.sum()),
        "fraud_ratio": float(y.mean()),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Preprocess Kaggle Credit Card Fraud dataset")
    parser.add_argument("--data_path", type=str, default=str(DEFAULT_CSV))
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    data_path = Path(args.data_path)
    output_dir = Path(args.output_dir)
    processed_dir = output_dir / "processed"
    metrics_dir = output_dir / "metrics"
    processed_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"[INFO] Raw shape: {df.shape}  Fraud rate: {df[TARGET_COL].mean()*100:.4f}%")

    # Feature engineering
    df = add_features(df)

    # Drop raw Time and Amount (we keep log_amount, time_hour, time_sin, time_cos, is_night)
    drop_raw = ["Time", "Amount"]
    df = df.drop(columns=drop_raw)

    # Separate features and target
    feature_cols = [c for c in df.columns if c != TARGET_COL]
    X = df[feature_cols]
    y = df[TARGET_COL]

    # Split
    full_df = pd.concat([X, y], axis=1)
    train_df, val_df, test_df = split_data(full_df)

    X_train = train_df[feature_cols].to_numpy(dtype=np.float32)
    y_train = train_df[TARGET_COL].to_numpy(dtype=np.int64)
    X_val   = val_df[feature_cols].to_numpy(dtype=np.float32)
    y_val   = val_df[TARGET_COL].to_numpy(dtype=np.int64)
    X_test  = test_df[feature_cols].to_numpy(dtype=np.float32)
    y_test  = test_df[TARGET_COL].to_numpy(dtype=np.int64)

    # Scale — fit only on train to prevent data leakage
    scaler = Pipeline(steps=[("scaler", StandardScaler())])
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    # Save
    save_split(X_train, y_train, "train", processed_dir)
    save_split(X_val,   y_val,   "val",   processed_dir)
    save_split(X_test,  y_test,  "test",  processed_dir)

    joblib.dump(scaler, processed_dir / "preprocessor.joblib")

    manifest = {
        "dataset": "creditcard_fraud_kaggle",
        "source": "mlg-ulb/creditcardfraud",
        "target_column": TARGET_COL,
        "dropped_columns": drop_raw,
        "numeric_columns": feature_cols,
        "categorical_columns": [],
        "n_features_after_preprocessing": len(feature_cols),
        "feature_names": feature_cols,
        "fraud_ratio": {
            "train": class_ratio(y_train),
            "val":   class_ratio(y_val),
            "test":  class_ratio(y_test),
        },
        "note": (
            "All V1-V28 features are PCA-transformed by the dataset provider. "
            "Added: log_amount, time_hour, time_sin, time_cos, is_night. "
            "No One-Hot Encoding needed (all numeric). "
            "StandardScaler applied (fit on train only)."
        ),
    }
    with (processed_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    split_df = pd.DataFrame([
        {"split": "train", **manifest["fraud_ratio"]["train"]},
        {"split": "val",   **manifest["fraud_ratio"]["val"]},
        {"split": "test",  **manifest["fraud_ratio"]["test"]},
    ])
    split_df.to_csv(metrics_dir / "split_fraud_ratio.csv", index=False)

    print("[OK] Preprocessing complete.")
    print(f"     Train: {len(y_train):,}  Val: {len(y_val):,}  Test: {len(y_test):,}")
    print(f"     Features: {len(feature_cols)}")
    print(f"     Fraud in train: {y_train.sum()} ({y_train.mean()*100:.4f}%)")
    print(f"[OK] Output written to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
