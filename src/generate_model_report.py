from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def df_to_markdown(df: pd.DataFrame) -> str:
    # Keep columns in stable order if present.
    ordered_cols = [
        c
        for c in ["model", "hybrid_model", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "loss"]
        if c in df.columns
    ]
    if ordered_cols:
        df = df[ordered_cols]
    header = "| " + " | ".join(df.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(df.columns)) + " |"
    rows = []
    for _, row in df.iterrows():
        vals = [str(v) for v in row.tolist()]
        rows.append("| " + " | ".join(vals) + " |")
    return "\n".join([header, sep] + rows)


def main():
    parser = argparse.ArgumentParser(description="Generate markdown report from ML results CSVs")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    metrics_dir = out_dir / "metrics"
    report_path = metrics_dir / "model_comparison_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    single_path = metrics_dir / "ml_single_results.csv"
    hybrid_path = metrics_dir / "ml_hybrid_results.csv"
    summary_path = metrics_dir / "ml_comparison_summary.json"

    lines = []
    lines.append("# Model Comparison Report")
    lines.append("")

    if not single_path.exists() or not hybrid_path.exists():
        lines.append("## Status")
        lines.append("")
        lines.append("Required files are missing. Run:")
        lines.append("")
        lines.append("```bash")
        lines.append("python src/train_ml_hybrids.py --output_dir outputs")
        lines.append("```")
        lines.append("")
        lines.append("Expected files:")
        lines.append("- `outputs/metrics/ml_single_results.csv`")
        lines.append("- `outputs/metrics/ml_hybrid_results.csv`")
        report_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"[OK] Report written to {report_path}")
        return

    single_df = pd.read_csv(single_path).sort_values("f1", ascending=False).head(args.top_k)
    hybrid_df = pd.read_csv(hybrid_path).sort_values("f1", ascending=False).head(args.top_k)

    lines.append("## Top Single Models")
    lines.append("")
    lines.append(df_to_markdown(single_df))
    lines.append("")
    lines.append("## Top Hybrid Models")
    lines.append("")
    lines.append(df_to_markdown(hybrid_df))
    lines.append("")

    if summary_path.exists():
        lines.append("## Summary File")
        lines.append("")
        lines.append("- `outputs/metrics/ml_comparison_summary.json`")
        lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] Report written to {report_path}")


if __name__ == "__main__":
    main()
