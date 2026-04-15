from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42
DEFAULT_DATA_PATH = Path("data/Bank_Transaction_Fraud_Detection.xlsx")
DEFAULT_OUTPUT = Path("outputs")
TARGET_COL = "Is_Fraud"


def normalize_name(col: str) -> str:
    return col.strip().lower().replace(" ", "_")


def detect_columns_by_keywords(columns: List[str], keywords: List[str]) -> List[str]:
    hits = []
    for c in columns:
        cl = normalize_name(c)
        if any(k in cl for k in keywords):
            hits.append(c)
    return hits


def detect_amount_balance_columns(df: pd.DataFrame) -> Tuple[str, str]:
    amount_candidates = detect_columns_by_keywords(df.columns.tolist(), ["amount"])
    balance_candidates = detect_columns_by_keywords(df.columns.tolist(), ["balance"])

    if not amount_candidates:
        raise ValueError(
            f"Could not detect amount column. Candidates by numeric dtype: "
            f"{df.select_dtypes(include=[np.number]).columns.tolist()}"
        )
    if not balance_candidates:
        raise ValueError(
            f"Could not detect balance column. Candidates by numeric dtype: "
            f"{df.select_dtypes(include=[np.number]).columns.tolist()}"
        )
    return amount_candidates[0], balance_candidates[0]


def detect_datetime_series(df: pd.DataFrame) -> pd.Series:
    date_cols = detect_columns_by_keywords(df.columns.tolist(), ["date"])
    time_cols = detect_columns_by_keywords(df.columns.tolist(), ["time"])
    dt_cols = detect_columns_by_keywords(df.columns.tolist(), ["datetime", "timestamp"])

    if date_cols and time_cols:
        date_col = date_cols[0]
        time_col = time_cols[0]
        date_raw = df[date_col]
        if pd.api.types.is_numeric_dtype(date_raw):
            date_part = pd.to_datetime(date_raw, unit="D", origin="1899-12-30", errors="coerce")
        else:
            date_part = pd.to_datetime(date_raw, errors="coerce")
        time_part = pd.to_timedelta(df[time_col].astype(str), errors="coerce")
        combined = date_part + time_part
        if combined.notna().mean() > 0.5:
            return combined
        return pd.to_datetime(df[date_col].astype(str) + " " + df[time_col].astype(str), errors="coerce")

    all_dt_candidates = dt_cols + date_cols + time_cols
    for c in all_dt_candidates:
        parsed = pd.to_datetime(df[c], errors="coerce")
        if parsed.notna().mean() > 0.5:
            return parsed

    raise ValueError(
        "Could not detect a datetime column. "
        f"Candidates: {all_dt_candidates if all_dt_candidates else df.columns.tolist()}"
    )


def drop_identity_like_columns(
    df: pd.DataFrame,
    target: str,
    preserve_cols: List[str] | None = None,
) -> Tuple[pd.DataFrame, List[str]]:
    preserve = set(preserve_cols or [])
    identity_keywords = [
        "id",
        "name",
        "account",
        "email",
        "contact",
        "phone",
        "description",
    ]
    explicit_drop = {
        "transaction_id",
        "customer_id",
        "merchant_id",
        "account_number",
        "customer_name",
        "customer_email",
        "customer_contact",
    }
    to_drop = []
    for c in df.columns:
        if c == target:
            continue
        if c in preserve:
            continue
        cl = normalize_name(c)
        if cl.startswith("unnamed"):
            to_drop.append(c)
            continue
        if cl in explicit_drop or any(k in cl for k in identity_keywords):
            to_drop.append(c)
    to_drop = sorted(set(to_drop))
    return df.drop(columns=to_drop, errors="ignore"), to_drop


def sanitize_target(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    if target_col not in df.columns:
        raise ValueError(f"Target column `{target_col}` not found. Available: {df.columns.tolist()}")
    y_num = pd.to_numeric(df[target_col], errors="coerce")
    mask = y_num.isin([0, 1])
    dropped = (~mask).sum()
    if dropped > 0:
        print(f"[INFO] Dropping {dropped} rows where target is not in {{0,1}}.")
    df = df.loc[mask].copy()
    df[target_col] = y_num.loc[mask].astype(int)
    return df


def add_feature_engineering(df: pd.DataFrame, amount_col: str, balance_col: str) -> pd.DataFrame:
    x = df.copy()
    x[amount_col] = pd.to_numeric(x[amount_col], errors="coerce")
    x[balance_col] = pd.to_numeric(x[balance_col], errors="coerce")
    x["amount_to_balance_ratio"] = x[amount_col] / (x[balance_col] + 1.0)
    x["amount_minus_balance"] = x[amount_col] - x[balance_col]
    x["is_over_balance"] = (x[amount_col] > x[balance_col]).astype(int)
    x["log_amount"] = np.log1p(np.clip(x[amount_col], a_min=0, a_max=None))
    return x


def add_time_features(df: pd.DataFrame, dt: pd.Series) -> pd.DataFrame:
    x = df.copy()
    x["tx_hour"] = dt.dt.hour
    x["tx_day_of_week"] = dt.dt.dayofweek
    x["is_weekend"] = (dt.dt.dayofweek >= 5).astype(float)
    x["is_night"] = dt.dt.hour.isin([0, 1, 2, 3, 4, 5]).astype(float)
    return x


def split_data(df: pd.DataFrame, target_col: str):
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        stratify=df[target_col],
        random_state=RANDOM_STATE,
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df[target_col],
        random_state=RANDOM_STATE,
    )
    return train_df, val_df, test_df


def make_preprocessor(train_x: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
    numeric_cols = train_x.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = train_x.select_dtypes(include=["object", "category", "string", "str"]).columns.tolist()

    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]
    )
    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_cols),
            ("cat", cat_pipe, categorical_cols),
        ],
        remainder="drop",
    )
    return pre, numeric_cols, categorical_cols


def save_sparse_and_target(x_sp, y: pd.Series, split: str, out_dir: Path):
    sparse.save_npz(out_dir / f"{split}_X.npz", x_sp.tocsr())
    np.save(out_dir / f"{split}_y.npy", y.to_numpy(dtype=np.int64))


def class_ratio(y: pd.Series) -> Dict[str, float]:
    return {
        "n": int(len(y)),
        "fraud_count": int((y == 1).sum()),
        "fraud_ratio": float((y == 1).mean()),
    }


def main():
    parser = argparse.ArgumentParser(description="Preprocess bank fraud Excel dataset")
    parser.add_argument("--data_path", type=str, default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    data_path = Path(args.data_path)
    output_dir = Path(args.output_dir)
    processed_dir = output_dir / "processed"
    metrics_dir = output_dir / "metrics"
    processed_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(data_path)
    df = sanitize_target(df, TARGET_COL)

    dt = detect_datetime_series(df)
    amount_col, balance_col = detect_amount_balance_columns(df)
    dt_like_raw = detect_columns_by_keywords(df.columns.tolist(), ["date", "time", "datetime", "timestamp"])
    preserve_cols = [amount_col, balance_col] + dt_like_raw
    df, dropped_cols = drop_identity_like_columns(df, TARGET_COL, preserve_cols=preserve_cols)
    df = add_feature_engineering(df, amount_col=amount_col, balance_col=balance_col)
    df = add_time_features(df, dt=dt)

    # Drop raw datetime columns from model input after deriving features.
    dt_like = detect_columns_by_keywords(df.columns.tolist(), ["date", "time", "datetime", "timestamp"])
    x_cols = [c for c in df.columns if c != TARGET_COL and c not in dt_like]
    model_df = df[x_cols + [TARGET_COL]].copy()

    train_df, val_df, test_df = split_data(model_df, TARGET_COL)

    train_x, train_y = train_df.drop(columns=[TARGET_COL]), train_df[TARGET_COL]
    val_x, val_y = val_df.drop(columns=[TARGET_COL]), val_df[TARGET_COL]
    test_x, test_y = test_df.drop(columns=[TARGET_COL]), test_df[TARGET_COL]

    preprocessor, numeric_cols, categorical_cols = make_preprocessor(train_x)
    x_train_sp = preprocessor.fit_transform(train_x)
    x_val_sp = preprocessor.transform(val_x)
    x_test_sp = preprocessor.transform(test_x)

    save_sparse_and_target(x_train_sp, train_y, "train", processed_dir)
    save_sparse_and_target(x_val_sp, val_y, "val", processed_dir)
    save_sparse_and_target(x_test_sp, test_y, "test", processed_dir)

    # Save dense arrays for PyTorch/Flower.
    np.save(processed_dir / "train_X_dense.npy", x_train_sp.toarray().astype(np.float32))
    np.save(processed_dir / "val_X_dense.npy", x_val_sp.toarray().astype(np.float32))
    np.save(processed_dir / "test_X_dense.npy", x_test_sp.toarray().astype(np.float32))

    joblib.dump(preprocessor, processed_dir / "preprocessor.joblib")

    feature_names = preprocessor.get_feature_names_out().tolist()
    manifest = {
        "target_column": TARGET_COL,
        "detected_amount_column": amount_col,
        "detected_balance_column": balance_col,
        "dropped_columns": dropped_cols,
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "n_features_after_preprocessing": len(feature_names),
        "feature_names": feature_names,
        "fraud_ratio": {
            "train": class_ratio(train_y),
            "val": class_ratio(val_y),
            "test": class_ratio(test_y),
        },
    }
    with (processed_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    split_df = pd.DataFrame(
        [
            {"split": "train", **manifest["fraud_ratio"]["train"]},
            {"split": "val", **manifest["fraud_ratio"]["val"]},
            {"split": "test", **manifest["fraud_ratio"]["test"]},
        ]
    )
    split_df.to_csv(metrics_dir / "split_fraud_ratio.csv", index=False)
    print("[OK] Preprocessing complete.")
    print(f"[OK] Output written to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
