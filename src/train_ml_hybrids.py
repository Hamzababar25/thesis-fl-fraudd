from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

from common import save_confusion_matrix, save_json


def metrics_from_scores(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_pred = (y_score >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
    }


def get_scores(model, x):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    if hasattr(model, "decision_function"):
        z = model.decision_function(x)
        return 1.0 / (1.0 + np.exp(-z))
    raise ValueError(f"Model {type(model).__name__} does not support score extraction")


def maybe_subsample(x, y, n_max: int, seed: int = 42):
    if n_max <= 0 or len(y) <= n_max:
        return x, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(y), size=n_max, replace=False)
    return x[idx], y[idx]


def class_weight_ratio(y: np.ndarray) -> float:
    neg = int((y == 0).sum())
    pos = int((y == 1).sum())
    return float(neg / max(pos, 1))


def main():
    parser = argparse.ArgumentParser(description="Train 3 selected ML models and hybrids for fraud")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=60000,
        help="Optional cap for faster training (0 = full train set).",
    )
    args = parser.parse_args()

    out = Path(args.output_dir)
    processed = out / "processed"
    metrics_dir = out / "metrics"
    plots_dir = out / "plots"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    x_train = sparse.load_npz(processed / "train_X.npz")
    y_train = np.load(processed / "train_y.npy")
    x_test = sparse.load_npz(processed / "test_X.npz")
    y_test = np.load(processed / "test_y.npy")

    x_train, y_train = maybe_subsample(x_train, y_train, n_max=args.max_train_samples, seed=42)
    scale_pos_weight = class_weight_ratio(y_train)

    models = {
        "logistic_regression": LogisticRegression(
            max_iter=400,
            solver="saga",
            class_weight="balanced",
            random_state=42,
        ),
        "svm_linear_calibrated": CalibratedClassifierCV(
            estimator=LinearSVC(class_weight="balanced", random_state=42),
            method="sigmoid",
            cv=3,
        ),
        "decision_tree": DecisionTreeClassifier(
            class_weight="balanced",
            max_depth=12,
            min_samples_leaf=20,
            random_state=42,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=14,
            min_samples_leaf=8,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=42,
        ),
        "xgboost": XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            reg_lambda=1.0,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1,
        ),
    }

    single_rows = []
    pred_store: Dict[str, np.ndarray] = {}

    print("[INFO] Training single models...")
    for name, model in models.items():
        print(f"[INFO] Fitting {name}...")
        model.fit(x_train, y_train)
        y_score = get_scores(model, x_test)
        pred_store[name] = y_score
        m = metrics_from_scores(y_test, y_score, threshold=args.threshold)
        row = {"model": name}
        row.update(m)
        single_rows.append(row)
        y_pred = (y_score >= args.threshold).astype(int)
        save_confusion_matrix(y_test, y_pred, plots_dir / f"cm_single_{name}.png")

    single_df = pd.DataFrame(single_rows).sort_values("f1", ascending=False)
    single_df.to_csv(metrics_dir / "ml_single_results.csv", index=False)
    save_json(
        metrics_dir / "ml_single_results.json",
        {"threshold": args.threshold, "results": single_df.to_dict(orient="records")},
    )

    print("[INFO] Building hybrid ensembles...")

    hybrid_rows = []
    # Requested weighted hybrids.
    weighted_defs = [
        ("xgboost", "logistic_regression", 0.7, 0.3),
        ("xgboost", "random_forest", 0.6, 0.4),
    ]
    for a, b, wa, wb in weighted_defs:
        if a not in pred_store or b not in pred_store:
            continue
        ens_name = f"{a}+{b}_w{int(wa*100)}_{int(wb*100)}"
        y_score = wa * pred_store[a] + wb * pred_store[b]
        m = metrics_from_scores(y_test, y_score, threshold=args.threshold)
        row = {"hybrid_model": ens_name}
        row.update(m)
        hybrid_rows.append(row)
        y_pred = (y_score >= args.threshold).astype(int)
        save_confusion_matrix(y_test, y_pred, plots_dir / f"cm_hybrid_{ens_name}.png")

    hybrid_df = pd.DataFrame(hybrid_rows).sort_values("f1", ascending=False)
    hybrid_df.to_csv(metrics_dir / "ml_hybrid_results.csv", index=False)
    save_json(
        metrics_dir / "ml_hybrid_results.json",
        {"threshold": args.threshold, "results": hybrid_df.to_dict(orient="records")},
    )

    best_single = single_df.iloc[0].to_dict()
    best_hybrid = hybrid_df.iloc[0].to_dict()
    summary = {
        "best_single": best_single,
        "best_hybrid": best_hybrid,
        "threshold": args.threshold,
        "train_samples_used": int(len(y_train)),
    }
    save_json(metrics_dir / "ml_comparison_summary.json", summary)

    print("[OK] ML single and hybrid comparison complete.")
    print("\nTop single models:")
    print(single_df.head(4).to_string(index=False))
    print("\nTop hybrid models:")
    print(hybrid_df.head(6).to_string(index=False))


if __name__ == "__main__":
    main()
