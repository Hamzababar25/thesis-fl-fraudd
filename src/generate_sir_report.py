"""
Generate Sir-ready clean CSV files covering every thesis question.

Output files in outputs/sir_report/:

  1_model_comparison.csv         → Q: Which ML model is best?
  2_fl_scenario_summary.csv      → Q: Attack vs Defense final metrics
  3_roundwise_performance.csv    → Q: Round-by-round accuracy/recall/loss
  4_attack_impact.csv            → Q: What happened when attack came?
  5_defense_recovery.csv         → Q: What happened after defense applied?
  6_client_weights_normal.csv    → Q: Show client weights in normal FL
  6_client_weights_attacked.csv  → Q: Show client weights under attack
  6_client_weights_defended.csv  → Q: Show client weights with defense
  7_defense_evidence.csv         → Q: Which client was rejected & when?
  8_full_roundwise_all.csv       → Q: All scenarios round by round (wide)
  README.txt                     → Guide for Sir (what each file means)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def load_csv(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None


def fmt4(df: pd.DataFrame, skip_cols: list[str]) -> pd.DataFrame:
    for c in df.select_dtypes(include="float").columns:
        if c not in skip_cols:
            df[c] = df[c].round(4)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--report_name", default="sir_report",
                        help="Name of the output report folder (e.g. sir_report2)")
    args = parser.parse_args()

    out      = Path(args.output_dir)
    metrics  = out / "metrics"
    sir_dir  = out / args.report_name
    sir_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Model Comparison ──────────────────────────────────────────────────
    rows = []

    cent = load_csv(metrics / "centralized_results.csv")
    if cent is not None:
        for _, r in cent.iterrows():
            rows.append({
                "rank": "",
                "model_name": "Logistic Regression (Centralized)",
                "category": "Centralized Baseline",
                "f1_score_%": round(r["f1"] * 100, 2),
                "recall_%": round(r["recall"] * 100, 2),
                "precision_%": round(r["precision"] * 100, 2),
                "roc_auc": round(r["roc_auc"], 4),
                "threshold_used": round(r.get("threshold", 0.5), 2),
                "can_use_in_FL": "YES",
            })

    single = load_csv(metrics / "ml_single_results.csv")
    if single is not None:
        for _, r in single.iterrows():
            rows.append({
                "rank": "",
                "model_name": str(r["model"]).replace("_", " ").title(),
                "category": "Single ML Model",
                "f1_score_%": round(r["f1"] * 100, 2),
                "recall_%": round(r["recall"] * 100, 2),
                "precision_%": round(r["precision"] * 100, 2),
                "roc_auc": round(r["roc_auc"], 4),
                "threshold_used": round(r.get("threshold", 0.5), 2),
                "can_use_in_FL": "YES" if r["model"] == "logistic_regression" else "NO (tree-based)",
            })

    hybrid = load_csv(metrics / "ml_hybrid_results.csv")
    if hybrid is not None:
        for _, r in hybrid.iterrows():
            rows.append({
                "rank": "",
                "model_name": str(r.get("hybrid_model", r.get("model", "?"))),
                "category": "Hybrid Ensemble",
                "f1_score_%": round(r["f1"] * 100, 2),
                "recall_%": round(r["recall"] * 100, 2),
                "precision_%": round(r["precision"] * 100, 2),
                "roc_auc": round(r["roc_auc"], 4),
                "threshold_used": round(r.get("threshold", 0.5), 2),
                "can_use_in_FL": "NO (tree-based)",
            })

    df1 = pd.DataFrame(rows).sort_values("f1_score_%", ascending=False).reset_index(drop=True)
    df1["rank"] = range(1, len(df1) + 1)
    df1.to_csv(sir_dir / "1_model_comparison.csv", index=False)
    print("[OK] 1_model_comparison.csv")

    # ── 2. FL Scenario Summary ───────────────────────────────────────────────
    comp = load_csv(metrics / "fl_attack_comparison.csv")
    if comp is not None:
        LABEL_MAP = {
            "normal_fl":              "Normal FL (No Attack, No Defense)",
            "sign_flip_no_defense":   "Sign-Flip Attack — No Defense",
            "sign_flip_defended":     "Sign-Flip Attack + Multi-Krum Defense",
            "scale_no_defense":       "Scale Attack — No Defense",
            "scale_defended":         "Scale Attack + Multi-Krum Defense",
            "label_flip_no_defense":  "Label-Flip Attack — No Defense",
            "label_flip_defended":    "Label-Flip Attack + Multi-Krum Defense",
        }
        ORDER = list(LABEL_MAP.keys())
        df2 = comp.copy()
        df2["scenario_label"] = df2["scenario"].map(LABEL_MAP).fillna(df2["scenario"])
        df2["attack_type"]    = df2.get("attack_type", pd.Series([""] * len(df2)))
        df2["defense_active"] = df2["defense"].apply(lambda d: "YES — Multi-Krum + Clipping + DP" if str(d) != "none" else "NO")
        df2["f1_score_%"]     = (df2["f1"] * 100).round(2)
        df2["recall_%"]       = (df2["recall"] * 100).round(2)
        df2["precision_%"]    = (df2["precision"] * 100).round(2)
        df2["roc_auc"]        = df2["roc_auc"].round(4)
        df2["loss"]           = df2["loss"].round(4)
        df2["threshold"]      = df2.get("threshold", pd.Series([0.5] * len(df2))).round(3)
        df2["_ord"]           = df2["scenario"].map({s: i for i, s in enumerate(ORDER)})
        df2 = df2.sort_values("_ord").drop(columns=["_ord"])
        out_cols = ["scenario_label", "attack_type", "defense_active",
                    "f1_score_%", "recall_%", "precision_%", "roc_auc", "loss", "threshold"]
        df2[[c for c in out_cols if c in df2.columns]].to_csv(sir_dir / "2_fl_scenario_summary.csv", index=False)
        print("[OK] 2_fl_scenario_summary.csv")

    # ── 3. Round-wise Performance (key 5 scenarios, all 20 rounds) ───────────
    KEY_SCENARIOS = [
        ("normal_fl",             "NormalFL"),
        ("sign_flip_no_defense",  "SignFlip_Attack"),
        ("sign_flip_defended",    "SignFlip_Defense"),
        ("label_flip_no_defense", "LabelFlip_Attack"),
        ("label_flip_defended",   "LabelFlip_Defense"),
    ]
    base_df = None
    for label, short in KEY_SCENARIOS:
        path = metrics / f"fl_round_metrics_{label}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        cols = {c: f"{short}_{c}" for c in ["loss", "f1", "recall", "precision", "roc_auc"] if c in df.columns}
        df = df[["round"] + list(cols.keys())].rename(columns=cols)
        base_df = df if base_df is None else base_df.merge(df, on="round", how="outer")

    if base_df is not None:
        base_df = base_df.sort_values("round").reset_index(drop=True)
        for c in base_df.columns:
            if c != "round":
                base_df[c] = base_df[c].round(4)
        base_df.to_csv(sir_dir / "3_roundwise_performance.csv", index=False)
        print("[OK] 3_roundwise_performance.csv")

    # ── 4. Attack Impact Table ───────────────────────────────────────────────
    if comp is not None:
        normal = comp[comp["scenario"] == "normal_fl"]
        if not normal.empty:
            n = normal.iloc[0]
            attack_rows = []
            for atk_label, atk_name in [
                ("sign_flip_no_defense", "Sign-Flip (model poisoning)"),
                ("scale_no_defense",     "Scale (model poisoning)"),
                ("label_flip_no_defense","Label-Flip (data poisoning)"),
            ]:
                row = comp[comp["scenario"] == atk_label]
                if row.empty:
                    continue
                r = row.iloc[0]
                attack_rows.append({
                    "attack_type":           atk_name,
                    "baseline_recall_%":     round(float(n["recall"]) * 100, 2),
                    "attack_recall_%":       round(float(r["recall"]) * 100, 2),
                    "recall_drop_%":         round((float(n["recall"]) - float(r["recall"])) * 100, 2),
                    "baseline_f1_%":         round(float(n["f1"]) * 100, 2),
                    "attack_f1_%":           round(float(r["f1"]) * 100, 2),
                    "f1_drop_%":             round((float(n["f1"]) - float(r["f1"])) * 100, 4),
                    "baseline_loss":         round(float(n["loss"]), 4),
                    "attack_loss":           round(float(r["loss"]), 4),
                    "verdict":               "SEVERE" if (float(n["recall"]) - float(r["recall"])) > 0.3
                                             else "MODERATE" if (float(n["recall"]) - float(r["recall"])) > 0.05
                                             else "MILD",
                })
            pd.DataFrame(attack_rows).to_csv(sir_dir / "4_attack_impact.csv", index=False)
            print("[OK] 4_attack_impact.csv")

    # ── 5. Defense Recovery Table ────────────────────────────────────────────
    if comp is not None:
        normal = comp[comp["scenario"] == "normal_fl"]
        if not normal.empty:
            n = normal.iloc[0]
            def_rows = []
            pairs = [
                ("sign_flip_no_defense",  "sign_flip_defended",  "Sign-Flip"),
                ("scale_no_defense",      "scale_defended",      "Scale"),
                ("label_flip_no_defense", "label_flip_defended", "Label-Flip"),
            ]
            for atk_lbl, def_lbl, atk_name in pairs:
                a_row = comp[comp["scenario"] == atk_lbl]
                d_row = comp[comp["scenario"] == def_lbl]
                if a_row.empty or d_row.empty:
                    continue
                a, d = a_row.iloc[0], d_row.iloc[0]
                def_rows.append({
                    "attack_type":            atk_name,
                    "baseline_recall_%":      round(float(n["recall"]) * 100, 2),
                    "under_attack_recall_%":  round(float(a["recall"]) * 100, 2),
                    "after_defense_recall_%": round(float(d["recall"]) * 100, 2),
                    "recovery_%":             round((float(d["recall"]) - float(a["recall"])) * 100, 2),
                    "baseline_f1_%":          round(float(n["f1"]) * 100, 2),
                    "under_attack_f1_%":      round(float(a["f1"]) * 100, 2),
                    "after_defense_f1_%":     round(float(d["f1"]) * 100, 2),
                    "defense_effective":      "YES — Full Recovery"
                                              if abs(float(d["recall"]) - float(n["recall"])) < 0.05
                                              else "PARTIAL — Some Recovery",
                })
            pd.DataFrame(def_rows).to_csv(sir_dir / "5_defense_recovery.csv", index=False)
            print("[OK] 5_defense_recovery.csv")

    # ── 6. Client Weights Per Round (3 key scenarios) ────────────────────────
    for label, fname in [
        ("normal_fl",            "6_client_weights_normal.csv"),
        ("sign_flip_no_defense", "6_client_weights_attacked.csv"),
        ("sign_flip_defended",   "6_client_weights_defended.csv"),
    ]:
        path = metrics / f"fl_round_client_weights_{label}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        keep_cols = ["round", "rejected_clients"]
        norm_cols = [c for c in df.columns if "update_norm" in c or "grad_norm_before" in c]
        keep_cols += norm_cols[:6]
        available = [c for c in keep_cols if c in df.columns]
        sub = df[available].copy()
        for c in sub.select_dtypes(include="float").columns:
            sub[c] = sub[c].round(4)
        sub.to_csv(sir_dir / fname, index=False)
    print("[OK] 6_client_weights_normal/attacked/defended.csv")

    # ── 7. Defense Evidence (which client rejected, which round) ─────────────
    evidence_rows = []
    for label in ["sign_flip_defended", "scale_defended", "label_flip_defended"]:
        path = metrics / f"fl_round_client_weights_{label}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if "rejected_clients" not in df.columns:
            continue
        for _, row in df.iterrows():
            rejected = str(row.get("rejected_clients", "")).strip()
            evidence_rows.append({
                "scenario":         label.replace("_", " ").title(),
                "round":            int(row["round"]),
                "rejected_clients": rejected if rejected else "None",
                "was_rejected":     "YES" if rejected else "NO",
            })
    if evidence_rows:
        ev_df = pd.DataFrame(evidence_rows)
        ev_df.to_csv(sir_dir / "7_defense_evidence.csv", index=False)
        print("[OK] 7_defense_evidence.csv")

        summary_rows = []
        for scenario in ev_df["scenario"].unique():
            s = ev_df[ev_df["scenario"] == scenario]
            rejected_rounds = (s["was_rejected"] == "YES").sum()
            total = len(s)
            summary_rows.append({
                "scenario": scenario,
                "total_rounds": total,
                "rounds_malicious_client_rejected": rejected_rounds,
                "rejection_rate_%": round(rejected_rounds / max(total, 1) * 100, 1),
                "verdict": "DEFENSE ACTIVE — Client Blocked" if rejected_rounds == total
                           else f"PARTIAL — {rejected_rounds}/{total} rounds blocked",
            })
        pd.DataFrame(summary_rows).to_csv(sir_dir / "7_defense_evidence_summary.csv", index=False)
        print("[OK] 7_defense_evidence_summary.csv")

    # ── 8. Full Round-wise All Scenarios ─────────────────────────────────────
    rw_path = metrics / "fl_all_scenarios_roundwise.csv"
    if rw_path.exists():
        rw = pd.read_csv(rw_path)
        for c in rw.columns:
            if c != "round":
                rw[c] = rw[c].round(4)
        rw.to_csv(sir_dir / "8_full_roundwise_all_scenarios.csv", index=False)
        print("[OK] 8_full_roundwise_all_scenarios.csv")

    # ── README ───────────────────────────────────────────────────────────────
    readme = f"""FL FRAUD DETECTION — SIR REPORT FILES
======================================
Generated from: {out}/metrics/ and {out}/processed/
Report folder : {sir_dir}

DATASET USED:
-------------
  Kaggle Credit Card Fraud Detection (mlg-ulb/creditcardfraud)
  - 284,807 real transactions (2 days, European cardholders)
  - 492 fraud cases (0.17% — severely imbalanced)
  - 30 features: V1-V28 (PCA-anonymized by dataset provider) + Amount + Time
  - Added features: log_amount, time_hour, time_sin, time_cos, is_night (33 total)
  - Split: 70% Train / 15% Validation / 15% Test (stratified)
  - NO One-Hot Encoding needed (all features are numeric)
  - SMOTE used to handle class imbalance during training

FILE GUIDE:
-----------

1_model_comparison.csv
  → ML models comparison: LR (Centralized) vs LR vs Random Forest vs XGBoost
  → Best ROC-AUC: Random Forest = 0.9685
  → Best F1:      XGBoost      = 0.8529
  → FL uses Logistic Regression (tree-based models cannot be federated easily)

2_fl_scenario_summary.csv
  → All 7 FL scenarios final metrics (test set evaluation)
  → Shows: Normal FL → Attack → Defense for each attack type
  → Key finding: Sign-Flip attack drops ROC-AUC from 0.936 to 0.090 (model destroyed)
  → Multi-Krum defense recovers ROC-AUC back to 0.931

3_roundwise_performance.csv
  → Round-by-round (Round 0 to 20) for 5 key scenarios
  → Shows: NormalFL, SignFlip_Attack, SignFlip_Defense, LabelFlip_Attack, LabelFlip_Defense
  → Each row = one FL communication round

4_attack_impact.csv
  → Quantifies damage each attack caused
  → Baseline vs Under-Attack: recall drop, F1 drop, loss increase
  → Verdict: SEVERE / MODERATE / MILD

5_defense_recovery.csv
  → Quantifies how much Multi-Krum defense recovered performance
  → Baseline → Under-Attack → After-Defense for Sign-Flip, Scale, Label-Flip
  → "defense_effective": YES = full recovery, PARTIAL = some recovery

6_client_weights_normal.csv     → Gradient update norms in normal FL (3 clients per round)
6_client_weights_attacked.csv   → Gradient norms under sign-flip attack (client w3 is malicious)
6_client_weights_defended.csv   → Gradient norms when Multi-Krum is active

7_defense_evidence.csv          → Per-round log: which client was rejected by Multi-Krum
7_defense_evidence_summary.csv  → Summary: how many rounds malicious client was blocked

8_full_roundwise_all_scenarios.csv
  → Wide table: all 7 scenarios × all 20 rounds
  → Columns: round, normal_fl_roc_auc, sign_flip_no_defense_roc_auc, ... etc.

KEY RESULTS SUMMARY:
--------------------
  Centralized LR baseline   : ROC-AUC = 0.961,  F1 = 0.747
  XGBoost (standalone)      : ROC-AUC = 0.963,  F1 = 0.853  ← Best F1
  Random Forest (standalone) : ROC-AUC = 0.968,  F1 = 0.829  ← Best ROC-AUC
  FL FedAvg IID             : ROC-AUC = 0.963,  F1 = 0.755  (matches centralized)
  FL Normal (Security setup): ROC-AUC = 0.936,  F1 = 0.805
  Sign-Flip Attack (no def) : ROC-AUC = 0.090,  F1 = 0.000  ← Attack SEVERE
  Sign-Flip + Multi-Krum    : ROC-AUC = 0.931,  F1 = 0.795  ← Defense EFFECTIVE
  Scale Attack (no defense) : ROC-AUC = 0.954,  F1 = 0.779
  Scale + Multi-Krum        : ROC-AUC = 0.931,  F1 = 0.795
  Label-Flip (no defense)   : ROC-AUC = 0.940,  F1 = 0.772
  Label-Flip + Multi-Krum   : ROC-AUC = 0.931,  F1 = 0.795

SECURITY FRAMEWORK (3 Layers):
--------------------------------
  Layer 1 — Gradient Clipping  : Each client clips updates (threshold=1.0) before sending
  Layer 2 — DP Noise           : Gaussian noise added after clipping (noise_mult=0.01)
  Layer 3 — Multi-Krum         : Server rejects statistically-outlier client updates

THESIS CONCLUSION:
------------------
  FL + 3-layer security framework achieves ROC-AUC ~0.93 while:
  (a) Protecting individual data privacy (no raw data leaves clients)
  (b) Defending against Sign-Flip, Scale, and Label-Flip attacks
  (c) Maintaining performance comparable to centralized baseline (0.961)
"""
    (sir_dir / "README.txt").write_text(readme, encoding="utf-8")
    print("[OK] README.txt")

    print(f"\n[DONE] Sir report files saved in: {sir_dir}")
    print(f"       Total files: {len(list(sir_dir.iterdir()))}")

    # Print quick preview
    print("\n── Quick Preview: Attack Impact ──")
    atk = load_csv(sir_dir / "4_attack_impact.csv")
    if atk is not None:
        print(atk[["attack_type", "baseline_recall_%", "attack_recall_%",
                    "recall_drop_%", "verdict"]].to_string(index=False))

    print("\n── Quick Preview: Defense Recovery ──")
    rec = load_csv(sir_dir / "5_defense_recovery.csv")
    if rec is not None:
        print(rec[["attack_type", "under_attack_recall_%", "after_defense_recall_%",
                   "recovery_%", "defense_effective"]].to_string(index=False))


if __name__ == "__main__":
    main()
