from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression

from common import (
    compute_metrics,
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
    x_test = sparse.load_npz(processed / "test_X.npz")
    y_test = np.load(processed / "test_y.npy")

    clf = LogisticRegression(
        max_iter=400,
        solver="saga",
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
    clf.fit(x_train, y_train)

    y_score = clf.predict_proba(x_test)[:, 1]
    y_pred = (y_score >= 0.5).astype(int)
    metrics = compute_metrics(y_test, y_score, threshold=0.5)
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
