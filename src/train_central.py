from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from scipy import sparse
from sklearn.linear_model import LogisticRegression

from common import (
    compute_metrics,
    find_best_threshold,
    report_text,
    save_confusion_matrix,
    save_json,
    save_pr_curve,
    save_roc_curve,
    to_dataframe_metrics,
)


def main():
    parser = argparse.ArgumentParser(description="Centralized baseline training")
    parser.add_argument("--output_dir", type=str, default="outputs")
    args = parser.parse_args()

    out = Path(args.output_dir)
    processed = out / "processed"
    metrics_dir = out / "metrics"
    plots_dir = out / "plots"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    x_train = sparse.load_npz(processed / "train_X.npz")
    y_train = np.load(processed / "train_y.npy")
    x_val = sparse.load_npz(processed / "val_X.npz")
    y_val = np.load(processed / "val_y.npy")
    x_test = sparse.load_npz(processed / "test_X.npz")
    y_test = np.load(processed / "test_y.npy")

    # Apply SMOTE to balance training data
    print(f"[INFO] Before SMOTE: {int((y_train==0).sum())} normal, {int((y_train==1).sum())} fraud")
    sm = SMOTE(random_state=42)
    x_train_res, y_train_res = sm.fit_resample(x_train, y_train)
    print(f"[INFO] After  SMOTE: {int((y_train_res==0).sum())} normal, {int((y_train_res==1).sum())} fraud")

    clf = LogisticRegression(
        max_iter=400,
        solver="saga",
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
    clf.fit(x_train_res, y_train_res)

    # Find best threshold on validation set (avoids data leakage)
    y_score_val = clf.predict_proba(x_val)[:, 1]
    best_threshold = find_best_threshold(y_val, y_score_val)
    print(f"[INFO] Best threshold (val F1-optimized): {best_threshold:.4f}")

    y_score = clf.predict_proba(x_test)[:, 1]
    y_pred = (y_score >= best_threshold).astype(int)
    metrics = compute_metrics(y_test, y_score, threshold=best_threshold)
    metrics["threshold"] = best_threshold
    cls_report = report_text(y_test, y_pred)

    save_json(metrics_dir / "centralized_metrics.json", metrics)
    save_json(metrics_dir / "centralized_classification_report.json", cls_report)
    to_dataframe_metrics("logreg_balanced", metrics).to_csv(
        metrics_dir / "centralized_results.csv", index=False
    )
    save_confusion_matrix(y_test, y_pred, plots_dir / "centralized_confusion_matrix.png")
    save_roc_curve(y_test, y_score, plots_dir / "centralized_roc_curve.png")
    save_pr_curve(y_test, y_score, plots_dir / "centralized_pr_curve.png")

    print("[OK] Centralized baseline complete.")
    print(pd.DataFrame([metrics]).to_string(index=False))


if __name__ == "__main__":
    main()
