"""Generate clean combined output files for thesis from individual scenario CSVs.

Produces in outputs/analysis/:
  combined_ml_results.csv          – all ML models (centralized + single + hybrid)
  combined_fl_final.csv            – final metrics for all FL scenarios
  combined_fl_roundwise.csv        – round-wise metrics (sampled every 5 rounds)
  combined_client_security.csv     – client norms per round across all scenarios
  master_summary.json              – key numbers in one JSON
  thesis_table_ml.txt              – Word-friendly ML comparison table
  thesis_table_fl.txt              – Word-friendly FL scenario comparison table
  thesis_table_roundwise.txt       – Round-wise performance table (every 5 rounds)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


# ─────────────────────────────── helpers ─────────────────────────────────────

def fmt(v, decimals: int = 4) -> str:
    if isinstance(v, float):
        return f"{v:.{decimals}f}"
    return str(v)


def df_to_text_table(df: pd.DataFrame, title: str = "") -> str:
    lines = []
    if title:
        lines.append(title)
        lines.append("=" * len(title))
    lines.append(df.to_string(index=False))
    lines.append("")
    return "\n".join(lines)


def load_if_exists(path: Path) -> pd.DataFrame | None:
    if path.exists():
        return pd.read_csv(path)
    return None


# ──────────────────────────── ML results ─────────────────────────────────────

def build_combined_ml(metrics_dir: Path) -> pd.DataFrame:
    rows: List[Dict] = []

    # Centralized baseline
    cent = load_if_exists(metrics_dir / "centralized_results.csv")
    if cent is not None:
        for _, r in cent.iterrows():
            rows.append({
                "category": "centralized",
                "model": r.get("model", "logreg_balanced"),
                **{k: r[k] for k in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "threshold"] if k in r},
            })

    # Single ML models
    single = load_if_exists(metrics_dir / "ml_single_results.csv")
    if single is not None:
        for _, r in single.iterrows():
            rows.append({
                "category": "single_ml",
                "model": r.get("model", "?"),
                **{k: r[k] for k in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "threshold"] if k in r},
            })

    # Hybrid ML models
    hybrid = load_if_exists(metrics_dir / "ml_hybrid_results.csv")
    if hybrid is not None:
        for _, r in hybrid.iterrows():
            rows.append({
                "category": "hybrid_ml",
                "model": r.get("hybrid_model", r.get("model", "?")),
                **{k: r[k] for k in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "threshold"] if k in r},
            })

    df = pd.DataFrame(rows)
    metric_cols = [c for c in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "threshold"] if c in df.columns]
    for c in metric_cols:
        df[c] = df[c].round(4)
    return df


# ──────────────────────────── FL final results ───────────────────────────────

FL_SCENARIO_ORDER = [
    "normal_fl",
    "sign_flip_no_defense",
    "sign_flip_defended",
    "scale_no_defense",
    "scale_defended",
    "label_flip_no_defense",
    "label_flip_defended",
    # legacy scenario labels for backward compat
    "fl_under_attack",
    "fl_with_defense",
    "fedavg",
    "multi_krum",
]


def build_combined_fl_final(metrics_dir: Path) -> pd.DataFrame:
    # Try the aggregated comparison CSV first
    comp = load_if_exists(metrics_dir / "fl_attack_comparison.csv")
    if comp is not None and not comp.empty:
        metric_cols = [c for c in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "loss", "threshold"] if c in comp.columns]
        for c in metric_cols:
            comp[c] = comp[c].round(4)
        # Reorder rows according to preferred scenario order
        order_map = {s: i for i, s in enumerate(FL_SCENARIO_ORDER)}
        comp["_ord"] = comp["scenario"].map(lambda s: order_map.get(s, 99))
        comp = comp.sort_values("_ord").drop(columns=["_ord"])
        return comp

    # Fall back: read individual fl_results_*.csv files
    rows: List[Dict] = []
    for path in sorted(metrics_dir.glob("fl_results_*.csv")):
        df = pd.read_csv(path)
        if not df.empty:
            r = df.iloc[0].to_dict()
            label = path.stem.replace("fl_results_", "")
            rows.append({"scenario": label, **r})
    if rows:
        df = pd.DataFrame(rows)
        metric_cols = [c for c in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "loss", "threshold"] if c in df.columns]
        for c in metric_cols:
            df[c] = df[c].round(4)
        return df
    return pd.DataFrame()


# ──────────────────────── FL round-wise (sampled) ───────────────────────────

def build_combined_roundwise(metrics_dir: Path, sample_every: int = 5) -> pd.DataFrame:
    # If the pre-built combined file exists, use it
    combined_path = metrics_dir / "fl_all_scenarios_roundwise.csv"
    if combined_path.exists():
        df = pd.read_csv(combined_path)
        # Sample every N rounds for readability
        df = df[df["round"] % sample_every == 0].reset_index(drop=True)
        metric_cols = [c for c in df.columns if c != "round"]
        for c in metric_cols:
            df[c] = df[c].round(4)
        return df

    # Otherwise build from individual files
    round_dfs: List[pd.DataFrame] = []
    for path in sorted(metrics_dir.glob("fl_round_metrics_*.csv")):
        label = path.stem.replace("fl_round_metrics_", "")
        df = pd.read_csv(path)
        if df.empty:
            continue
        cols_keep = [c for c in ["round", "loss", "f1", "recall", "precision", "roc_auc", "pr_auc"] if c in df.columns]
        df = df[cols_keep].rename(columns={c: f"{label}_{c}" for c in cols_keep if c != "round"})
        round_dfs.append(df)

    if not round_dfs:
        return pd.DataFrame()

    combined = round_dfs[0]
    for df in round_dfs[1:]:
        combined = combined.merge(df, on="round", how="outer")
    combined = combined.sort_values("round")
    combined = combined[combined["round"] % sample_every == 0].reset_index(drop=True)
    metric_cols = [c for c in combined.columns if c != "round"]
    for c in metric_cols:
        combined[c] = combined[c].round(4)
    return combined


# ────────────────────── Client security norms ─────────────────────────────

def build_combined_client_security(metrics_dir: Path) -> pd.DataFrame:
    dfs: List[pd.DataFrame] = []
    for path in sorted(metrics_dir.glob("fl_round_client_weights_*.csv")):
        label = path.stem.replace("fl_round_client_weights_", "")
        df = pd.read_csv(path)
        if df.empty:
            continue
        df["scenario"] = label
        dfs.append(df)
    if dfs:
        combined = pd.concat(dfs, ignore_index=True)
        norm_cols = [c for c in combined.columns if "norm" in c.lower() or "rejected" in c.lower() or "round" in c.lower() or "scenario" in c.lower()]
        available = [c for c in norm_cols if c in combined.columns]
        return combined[available]
    return pd.DataFrame()


# ────────────────────── Master summary JSON ──────────────────────────────────

def build_master_summary(ml_df: pd.DataFrame, fl_df: pd.DataFrame) -> Dict:
    summary: Dict = {}

    if not ml_df.empty:
        best_idx = ml_df["f1"].idxmax() if "f1" in ml_df.columns else None
        if best_idx is not None:
            best = ml_df.loc[best_idx].to_dict()
            summary["best_ml_model"] = best

    if not fl_df.empty and "scenario" in fl_df.columns:
        normal = fl_df[fl_df["scenario"] == "normal_fl"]
        if not normal.empty:
            summary["fl_normal_baseline"] = normal.iloc[0].to_dict()

        attacked = fl_df[fl_df["scenario"].isin(["sign_flip_no_defense", "fl_under_attack"])]
        if not attacked.empty:
            summary["fl_sign_flip_attack"] = attacked.iloc[0].to_dict()

        defended = fl_df[fl_df["scenario"].isin(["sign_flip_defended", "fl_with_defense"])]
        if not defended.empty:
            summary["fl_sign_flip_defended"] = defended.iloc[0].to_dict()

        scale_atk = fl_df[fl_df["scenario"] == "scale_no_defense"]
        if not scale_atk.empty:
            summary["fl_scale_attack"] = scale_atk.iloc[0].to_dict()

        scale_def = fl_df[fl_df["scenario"] == "scale_defended"]
        if not scale_def.empty:
            summary["fl_scale_defended"] = scale_def.iloc[0].to_dict()

        lf_atk = fl_df[fl_df["scenario"] == "label_flip_no_defense"]
        if not lf_atk.empty:
            summary["fl_label_flip_attack"] = lf_atk.iloc[0].to_dict()

        lf_def = fl_df[fl_df["scenario"] == "label_flip_defended"]
        if not lf_def.empty:
            summary["fl_label_flip_defended"] = lf_def.iloc[0].to_dict()

    return summary


# ────────────────────── Text tables for Word/thesis ──────────────────────────

def thesis_table_ml(ml_df: pd.DataFrame) -> str:
    if ml_df.empty:
        return "No ML results found.\n"
    display_cols = ["category", "model"] + [c for c in ["f1", "recall", "precision", "roc_auc", "pr_auc", "accuracy", "threshold"] if c in ml_df.columns]
    df = ml_df[[c for c in display_cols if c in ml_df.columns]].copy()
    df = df.sort_values("f1", ascending=False)
    return df_to_text_table(df, "Table: ML Model Comparison (threshold optimized on val set)")


def thesis_table_fl(fl_df: pd.DataFrame) -> str:
    if fl_df.empty:
        return "No FL results found.\n"
    display_cols = ["scenario", "attack_type", "defense"] + [c for c in ["f1", "recall", "precision", "roc_auc", "pr_auc", "loss", "threshold"] if c in fl_df.columns]
    df = fl_df[[c for c in display_cols if c in fl_df.columns]].copy()
    return df_to_text_table(df, "Table: FL Security Scenario Comparison")


def thesis_table_roundwise(rw_df: pd.DataFrame) -> str:
    if rw_df.empty:
        return "No round-wise data found.\n"
    # Show only round and f1 columns for readability
    f1_cols = ["round"] + [c for c in rw_df.columns if c.endswith("_f1")]
    recall_cols = ["round"] + [c for c in rw_df.columns if c.endswith("_recall")]
    text = "Table: Round-wise F1 Score (every 5 rounds)\n"
    text += "=" * 60 + "\n"
    if f1_cols:
        text += rw_df[[c for c in f1_cols if c in rw_df.columns]].to_string(index=False)
    text += "\n\nTable: Round-wise Recall (every 5 rounds)\n"
    text += "=" * 60 + "\n"
    if recall_cols:
        text += rw_df[[c for c in recall_cols if c in rw_df.columns]].to_string(index=False)
    text += "\n"
    return text


# ──────────────────────────── main ────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate combined thesis-ready output files")
    parser.add_argument("--output_dir", type=str, default="outputs")
    args = parser.parse_args()

    out = Path(args.output_dir)
    metrics_dir = out / "metrics"
    analysis_dir = out / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Building combined ML results...")
    ml_df = build_combined_ml(metrics_dir)
    if not ml_df.empty:
        ml_df.to_csv(analysis_dir / "combined_ml_results.csv", index=False)
        print(f"[OK]  combined_ml_results.csv  ({len(ml_df)} rows)")

    print("[INFO] Building combined FL final metrics...")
    fl_df = build_combined_fl_final(metrics_dir)
    if not fl_df.empty:
        fl_df.to_csv(analysis_dir / "combined_fl_final.csv", index=False)
        print(f"[OK]  combined_fl_final.csv  ({len(fl_df)} scenarios)")

    print("[INFO] Building combined round-wise table...")
    rw_df = build_combined_roundwise(metrics_dir, sample_every=5)
    if not rw_df.empty:
        rw_df.to_csv(analysis_dir / "combined_fl_roundwise.csv", index=False)
        print(f"[OK]  combined_fl_roundwise.csv  ({len(rw_df)} rows × {len(rw_df.columns)} cols)")

    print("[INFO] Building client security norms...")
    cs_df = build_combined_client_security(metrics_dir)
    if not cs_df.empty:
        cs_df.to_csv(analysis_dir / "combined_client_security.csv", index=False)
        print(f"[OK]  combined_client_security.csv  ({len(cs_df)} rows)")

    print("[INFO] Building master summary JSON...")
    summary = build_master_summary(ml_df, fl_df)
    with (analysis_dir / "master_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print("[OK]  master_summary.json")

    print("[INFO] Generating thesis text tables...")
    with (analysis_dir / "thesis_table_ml.txt").open("w", encoding="utf-8") as f:
        f.write(thesis_table_ml(ml_df))
    with (analysis_dir / "thesis_table_fl.txt").open("w", encoding="utf-8") as f:
        f.write(thesis_table_fl(fl_df))
    with (analysis_dir / "thesis_table_roundwise.txt").open("w", encoding="utf-8") as f:
        f.write(thesis_table_roundwise(rw_df))
    print("[OK]  thesis text tables written")

    print("\n[DONE] All combined outputs in:", analysis_dir)
    if not ml_df.empty:
        print("\n--- ML Top Models ---")
        show_cols = [c for c in ["model", "f1", "recall", "precision", "roc_auc", "threshold"] if c in ml_df.columns]
        print(ml_df[show_cols].sort_values("f1", ascending=False).head(6).to_string(index=False))
    if not fl_df.empty:
        print("\n--- FL Scenario Summary ---")
        show_cols = [c for c in ["scenario", "f1", "recall", "precision", "roc_auc"] if c in fl_df.columns]
        print(fl_df[show_cols].to_string(index=False))


if __name__ == "__main__":
    main()
