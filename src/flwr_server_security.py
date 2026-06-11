from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import flwr as fl
import numpy as np
import pandas as pd
import torch
from flwr.common import NDArrays, Scalar, ndarrays_to_parameters, parameters_to_ndarrays
from sklearn.metrics import log_loss

from aggregation import fedavg_aggregate, multi_krum_aggregate
from attack_simulation import build_attack_comparison_rows, build_robustness_rows, save_attack_comparison_plots
from common import (
    FraudLogistic,
    FraudMLP,
    build_dataloader,
    compute_metrics,
    find_best_threshold,
    save_confusion_matrix,
    save_json,
    save_pr_curve,
    save_roc_curve,
    to_dataframe_metrics,
)
from flwr_client import DEVICE, FraudClient, make_client_datasets, set_weights


def resolve_fl_model(fl_model: str, output_dir: Path) -> str:
    if fl_model != "best_from_ml":
        return fl_model
    summary_path = output_dir / "metrics" / "ml_comparison_summary.json"
    if not summary_path.exists():
        print("[WARN] best_from_ml requested but summary not found; using logistic_regression.")
        return "logistic_regression"
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        best_single = str(payload.get("best_single", {}).get("model", "logistic_regression"))
        if best_single == "logistic_regression":
            return "logistic_regression"
    except Exception:
        pass
    print("[WARN] best_from_ml resolved to unsupported FL model; falling back to logistic_regression.")
    return "logistic_regression"


def evaluate_global(
    model,
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int = 128,
) -> Tuple[float, Dict[str, float], np.ndarray]:
    loader = build_dataloader(x, y, batch_size=batch_size, shuffle=False)
    model.eval()
    probs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            p = torch.sigmoid(model(xb)).cpu().numpy()
            probs.append(p)
            ys.append(yb.numpy())
    y_true = np.concatenate(ys)
    y_score = np.concatenate(probs)
    metrics = compute_metrics(y_true, y_score, threshold=0.5)
    loss = float(log_loss(y_true, y_score, labels=[0, 1]))
    return loss, metrics, y_score


class SecurityRobustFedAvg(fl.server.strategy.FedAvg):
    """Security-focused strategy with robust aggregation on protected updates."""

    def __init__(
        self,
        aggregation_method: str = "fedavg",
        num_malicious: int = 1,
        multi_krum_m: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.aggregation_method = aggregation_method
        self.num_malicious = int(max(0, num_malicious))
        self.multi_krum_m = int(max(0, multi_krum_m))
        self.round_security_rows: List[Dict[str, float]] = []

    def aggregate_fit(self, server_round: int, results, failures):
        if not results:
            return None, {}

        cids: List[str] = []
        arrays_by_client: Dict[str, NDArrays] = {}
        num_examples_by_client: Dict[str, int] = {}
        row: Dict[str, float] = {"round": float(server_round), "num_clients": float(len(results))}

        for client_proxy, fit_res in results:
            raw_cid = str(client_proxy.cid)
            if "client_idx" in fit_res.metrics:
                cid = f"w{int(float(fit_res.metrics['client_idx']))}"
            else:
                cid = raw_cid
            cids.append(cid)
            arrays_by_client[cid] = parameters_to_ndarrays(fit_res.parameters)
            num_examples_by_client[cid] = int(fit_res.num_examples)
            row[f"{cid}_update_norm"] = float(fit_res.metrics.get("update_norm", 0.0))
            row[f"{cid}_clipped_norm"] = float(fit_res.metrics.get("clipped_update_norm", 0.0))
            row[f"{cid}_grad_norm_before"] = float(fit_res.metrics.get("gradient_norm_before_clipping", 0.0))
            row[f"{cid}_grad_norm_after"] = float(fit_res.metrics.get("gradient_norm_after_clipping", 0.0))
            row[f"{cid}_clip_threshold"] = float(fit_res.metrics.get("clip_threshold", 0.0))
            row[f"{cid}_noise_multiplier"] = float(fit_res.metrics.get("noise_multiplier", 0.0))
            row[f"{cid}_noise_scale"] = float(fit_res.metrics.get("noise_scale_used", 0.0))
            row[f"{cid}_is_malicious"] = float(fit_res.metrics.get("is_malicious", 0.0))
            row[f"{cid}_attack_strength"] = float(fit_res.metrics.get("attack_strength", 0.0))

        arrays = [arrays_by_client[cid] for cid in cids]
        num_examples = [num_examples_by_client[cid] for cid in cids]

        if self.aggregation_method == "multi_krum":
            maybe_m = self.multi_krum_m if self.multi_krum_m > 0 else None
            agg_res = multi_krum_aggregate(
                cids=cids,
                arrays=arrays,
                num_malicious=self.num_malicious,
                num_selected=maybe_m,
            )
        else:
            agg_res = fedavg_aggregate(cids=cids, arrays=arrays, num_examples=num_examples)

        for cid in cids:
            row[f"{cid}_agg_score"] = float(agg_res.client_scores.get(cid, 0.0))
            row[f"{cid}_selected"] = float(1.0 if cid in agg_res.selected_clients else 0.0)
        row["rejected_clients"] = ",".join(agg_res.rejected_clients)

        self.round_security_rows.append(row)
        return ndarrays_to_parameters(agg_res.aggregated), {"aggregation_method": self.aggregation_method}


def run_security_fl(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    output_dir: Path,
    fl_model: str = "logistic_regression",
    rounds: int = 20,
    lr: float = 1e-3,
    partition_mode: str = "bank_noniid",
    clip_threshold: float = 1.0,
    noise_multiplier: float = 0.01,
    aggregation_method: str = "fedavg",
    num_malicious: int = 1,
    multi_krum_m: int = 0,
    run_label: str = "security_run",
    attack_enabled: bool = False,
    attack_strength: float = 0.0,
    attack_mode: str = "sign_flip",
    malicious_clients: str = "",
    x_val: np.ndarray = None,
    y_val: np.ndarray = None,
):
    input_dim = x_train.shape[1]
    client_local = make_client_datasets(
        x_train=x_train,
        y_train=y_train,
        n_clients=3,
        mode=partition_mode,
        seed=42,
    )
    cids = list(client_local.keys())

    def client_fn(cid: str):
        mapped = cids[int(cid)] if cid.isdigit() else cid
        return FraudClient(
            cid=mapped,
            data=client_local[mapped],
            input_dim=input_dim,
            model_name=fl_model,
            lr=lr,
        )

    round_logs: List[Dict[str, float]] = []
    if fl_model == "logistic_regression":
        global_model = FraudLogistic(input_dim=input_dim).to(DEVICE)
    else:
        global_model = FraudMLP(input_dim=input_dim).to(DEVICE)

    def fit_config(server_round: int):
        return {
            "local_epochs": 1,
            "batch_size": 128,
            "server_round": server_round,
            "clip_threshold": clip_threshold,
            "noise_multiplier": noise_multiplier,
            "dp_seed": 2026,
            "attack_enabled": attack_enabled,
            "attack_strength": attack_strength,
            "attack_mode": attack_mode,
            "malicious_clients": malicious_clients,
        }

    def server_eval(server_round: int, parameters: NDArrays, config: Dict[str, Scalar]):
        set_weights(global_model, parameters)
        loss, metrics, _ = evaluate_global(global_model, x_test, y_test, batch_size=128)
        row = {"round": server_round, "loss": loss}
        row.update(metrics)
        round_logs.append(row)
        return loss, metrics

    strategy = SecurityRobustFedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=3,
        min_evaluate_clients=3,
        min_available_clients=3,
        on_fit_config_fn=fit_config,
        evaluate_fn=server_eval,
        aggregation_method=aggregation_method,
        num_malicious=num_malicious,
        multi_krum_m=multi_krum_m,
    )

    run_config = fl.server.ServerConfig(num_rounds=rounds)
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=3,
        config=run_config,
        strategy=strategy,
        client_resources={"num_cpus": 1},
        ray_init_args={"include_dashboard": False},
    )

    # Find best threshold on validation set, then evaluate test set with it
    if x_val is not None and y_val is not None:
        _, _, y_score_val = evaluate_global(global_model, x_val, y_val, batch_size=128)
        best_threshold = find_best_threshold(y_val, y_score_val)
    else:
        best_threshold = 0.5
    print(f"[INFO] {run_label}: best threshold (val F1) = {best_threshold:.4f}")

    final_loss, _, y_score = evaluate_global(global_model, x_test, y_test, batch_size=128)
    final_metrics_no_loss = compute_metrics(y_test, y_score, threshold=best_threshold)
    final_metrics_no_loss["threshold"] = best_threshold
    y_pred = (y_score >= best_threshold).astype(int)
    final_metrics = {"loss": final_loss, **final_metrics_no_loss}

    metrics_dir = output_dir / "metrics"
    plots_dir = output_dir / "plots"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    tag = run_label
    pd.DataFrame(round_logs).to_csv(metrics_dir / f"fl_round_metrics_{tag}.csv", index=False)
    pd.DataFrame(strategy.round_security_rows).to_csv(metrics_dir / f"fl_round_client_weights_{tag}.csv", index=False)
    save_json(metrics_dir / f"fl_final_metrics_{tag}.json", final_metrics)
    to_dataframe_metrics(f"fl_{tag}", final_metrics_no_loss).assign(loss=final_loss).to_csv(
        metrics_dir / f"fl_results_{tag}.csv", index=False
    )

    save_confusion_matrix(y_test, y_pred, plots_dir / f"fl_confusion_matrix_{tag}.png")
    save_roc_curve(y_test, y_score, plots_dir / f"fl_roc_curve_{tag}.png")
    save_pr_curve(y_test, y_score, plots_dir / f"fl_pr_curve_{tag}.png")

    outlier_events = []
    for rec in strategy.round_security_rows:
        rejected = str(rec.get("rejected_clients", "")).strip()
        if rejected:
            outlier_events.append({"round": int(rec["round"]), "rejected_clients": rejected.split(",")})

    strategy_config = {
        "fl_model": fl_model,
        "aggregation_method": aggregation_method,
        "num_malicious": num_malicious,
        "multi_krum_m": multi_krum_m,
        "clip_threshold": clip_threshold,
        "noise_multiplier": noise_multiplier,
        "attack_enabled": attack_enabled,
        "attack_strength": attack_strength,
        "attack_mode": attack_mode,
        "malicious_clients": malicious_clients,
    }
    save_json(
        metrics_dir / f"fl_aggregation_summary_{tag}.json",
        {
            "run_label": tag,
            "strategy_config": strategy_config,
            "selected_clients_last_round": strategy.round_security_rows[-1] if strategy.round_security_rows else {},
            "outlier_events": outlier_events,
        },
    )

    print(f"[OK] Security FL run complete ({aggregation_method}).")
    print(pd.DataFrame([final_metrics]).to_string(index=False))
    return final_metrics


def main():
    parser = argparse.ArgumentParser(description="Run Flower Security-Focused FL")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--fl_model",
        type=str,
        default="best_from_ml",
        choices=["best_from_ml", "logistic_regression", "mlp"],
        help="Federated model. best_from_ml currently maps to logistic_regression.",
    )
    parser.add_argument("--partition_mode", type=str, default="bank_noniid", choices=["iid", "noniid", "bank_noniid"])
    parser.add_argument("--clip_threshold", type=float, default=1.0)
    parser.add_argument("--noise_multiplier", type=float, default=0.01)
    parser.add_argument(
        "--aggregation_method",
        type=str,
        default="fedavg",
        choices=["fedavg", "multi_krum"],
    )
    parser.add_argument("--num_malicious", type=int, default=1)
    parser.add_argument(
        "--multi_krum_m",
        type=int,
        default=0,
        help="Number of selected updates in Multi-Krum (0 => n_clients - num_malicious).",
    )
    parser.add_argument(
        "--compare_strategies",
        action="store_true",
        help="Run both FedAvg and Multi-Krum and generate comparison metrics.",
    )
    parser.add_argument("--attack_enabled", action="store_true")
    parser.add_argument("--attack_strength", type=float, default=5.0)
    parser.add_argument("--attack_mode", type=str, default="sign_flip", choices=["sign_flip", "scale", "label_flip"])
    parser.add_argument(
        "--malicious_clients",
        type=str,
        default="w3",
        help="Comma-separated malicious client IDs, e.g., w2,w3",
    )
    parser.add_argument(
        "--evaluate_attack_scenarios",
        action="store_true",
        help="Run normal FL, FL under attack, and FL with defense; then generate comparison metrics and plots.",
    )
    parser.add_argument(
        "--evaluate_security_techniques",
        action="store_true",
        help="Run baseline, DP-only, clipping-only, and Multi-Krum-only technique comparison.",
    )
    args = parser.parse_args()

    out = Path(args.output_dir)
    resolved_fl_model = resolve_fl_model(args.fl_model, out)
    processed = out / "processed"
    x_train = np.load(processed / "train_X_dense.npy")
    y_train = np.load(processed / "train_y.npy")
    x_test = np.load(processed / "test_X_dense.npy")
    y_test = np.load(processed / "test_y.npy")

    metrics_dir = out / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    x_val = np.load(processed / "val_X_dense.npy")
    y_val = np.load(processed / "val_y.npy")

    run_records = []
    if args.evaluate_attack_scenarios:
        run_plan = [
            {
                "method": "fedavg",
                "label": "normal_fl",
                "attack_enabled": False,
                "attack_mode": "sign_flip",
                "clip_threshold": 0.0,
                "noise_multiplier": 0.0,
            },
            {
                "method": "fedavg",
                "label": "sign_flip_no_defense",
                "attack_enabled": True,
                "attack_mode": "sign_flip",
                "clip_threshold": 0.0,
                "noise_multiplier": 0.0,
            },
            {
                "method": "multi_krum",
                "label": "sign_flip_defended",
                "attack_enabled": True,
                "attack_mode": "sign_flip",
                "clip_threshold": args.clip_threshold,
                "noise_multiplier": args.noise_multiplier,
            },
            {
                "method": "fedavg",
                "label": "scale_no_defense",
                "attack_enabled": True,
                "attack_mode": "scale",
                "clip_threshold": 0.0,
                "noise_multiplier": 0.0,
            },
            {
                "method": "multi_krum",
                "label": "scale_defended",
                "attack_enabled": True,
                "attack_mode": "scale",
                "clip_threshold": args.clip_threshold,
                "noise_multiplier": args.noise_multiplier,
            },
            {
                "method": "fedavg",
                "label": "label_flip_no_defense",
                "attack_enabled": True,
                "attack_mode": "label_flip",
                "clip_threshold": 0.0,
                "noise_multiplier": 0.0,
            },
            {
                "method": "multi_krum",
                "label": "label_flip_defended",
                "attack_enabled": True,
                "attack_mode": "label_flip",
                "clip_threshold": args.clip_threshold,
                "noise_multiplier": args.noise_multiplier,
            },
        ]
    elif args.evaluate_security_techniques:
        run_plan = [
            {
                "method": "fedavg",
                "label": "baseline_fl",
                "clip_threshold": 0.0,
                "noise_multiplier": 0.0,
            },
            {
                "method": "fedavg",
                "label": "dp_only",
                "clip_threshold": 0.0,
                "noise_multiplier": args.noise_multiplier,
            },
            {
                "method": "fedavg",
                "label": "clipping_only",
                "clip_threshold": args.clip_threshold,
                "noise_multiplier": 0.0,
            },
            {
                "method": "multi_krum",
                "label": "multi_krum_only",
                "clip_threshold": 0.0,
                "noise_multiplier": 0.0,
            },
        ]
    elif args.compare_strategies:
        run_plan = [
            {
                "method": "fedavg",
                "label": "fedavg",
                "clip_threshold": args.clip_threshold,
                "noise_multiplier": args.noise_multiplier,
            },
            {
                "method": "multi_krum",
                "label": "multi_krum",
                "clip_threshold": args.clip_threshold,
                "noise_multiplier": args.noise_multiplier,
            },
        ]
    else:
        run_plan = [
            {
                "method": args.aggregation_method,
                "label": args.aggregation_method,
                "clip_threshold": args.clip_threshold,
                "noise_multiplier": args.noise_multiplier,
            }
        ]

    for item in run_plan:
        final_metrics = run_security_fl(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            output_dir=out,
            fl_model=resolved_fl_model,
            rounds=args.rounds,
            lr=args.lr,
            partition_mode=args.partition_mode,
            clip_threshold=float(item["clip_threshold"]),
            noise_multiplier=float(item["noise_multiplier"]),
            aggregation_method=str(item["method"]),
            num_malicious=args.num_malicious,
            multi_krum_m=args.multi_krum_m,
            run_label=str(item["label"]),
            attack_enabled=bool(item.get("attack_enabled", args.attack_enabled)),
            attack_strength=args.attack_strength,
            attack_mode=str(item.get("attack_mode", args.attack_mode)),
            malicious_clients=args.malicious_clients,
            x_val=x_val,
            y_val=y_val,
        )
        row = {
            "scenario": str(item["label"]),
            "fl_model": resolved_fl_model,
            "aggregation_method": str(item["method"]),
            "attack_type": str(item.get("attack_mode", "none")),
            "attack_enabled": float(1.0 if bool(item.get("attack_enabled", args.attack_enabled)) else 0.0),
            "defense": "multi_krum+clip+dp" if str(item["method"]) == "multi_krum" else "none",
            "clip_threshold": float(item["clip_threshold"]),
            "noise_multiplier": float(item["noise_multiplier"]),
        }
        row.update(final_metrics)
        run_records.append(row)

    if args.evaluate_attack_scenarios:
        comparison_df = build_attack_comparison_rows(run_records)
        comparison_df.to_csv(metrics_dir / "fl_attack_comparison.csv", index=False)
        save_json(metrics_dir / "fl_attack_comparison.json", {"results": comparison_df.to_dict(orient="records")})

        robustness_df = build_robustness_rows(out, comparison_df["scenario"].tolist())
        robustness_df.to_csv(metrics_dir / "fl_attack_robustness.csv", index=False)
        save_attack_comparison_plots(
            metrics_df=comparison_df,
            robustness_df=robustness_df,
            output_dir=out,
            prefix="fl_attack_evaluation",
        )

        # Combined round-wise table: all scenarios side by side
        scenario_labels = [str(item["label"]) for item in run_plan]
        round_dfs: List[pd.DataFrame] = []
        for label in scenario_labels:
            path = metrics_dir / f"fl_round_metrics_{label}.csv"
            if path.exists():
                df = pd.read_csv(path)
                cols_keep = [c for c in ["round", "loss", "f1", "recall", "precision", "roc_auc", "pr_auc"] if c in df.columns]
                df = df[cols_keep].rename(columns={c: f"{label}_{c}" for c in cols_keep if c != "round"})
                round_dfs.append(df)
        if round_dfs:
            combined_rounds = round_dfs[0]
            for df in round_dfs[1:]:
                combined_rounds = combined_rounds.merge(df, on="round", how="outer")
            combined_rounds.sort_values("round").to_csv(metrics_dir / "fl_all_scenarios_roundwise.csv", index=False)
            print(f"[OK] Combined roundwise CSV saved ({len(combined_rounds)} rounds × {len(combined_rounds.columns)} cols)")
    elif args.evaluate_security_techniques:
        comparison_df = build_attack_comparison_rows(run_records)
        comparison_df.to_csv(metrics_dir / "fl_security_technique_comparison.csv", index=False)
        save_json(metrics_dir / "fl_security_technique_comparison.json", {"results": comparison_df.to_dict(orient="records")})

        robustness_df = build_robustness_rows(out, comparison_df["scenario"].tolist())
        robustness_df.to_csv(metrics_dir / "fl_security_technique_robustness.csv", index=False)
        save_attack_comparison_plots(
            metrics_df=comparison_df,
            robustness_df=robustness_df,
            output_dir=out,
            prefix="fl_security_technique_evaluation",
        )
    elif len(run_records) > 1:
        pd.DataFrame(run_records).to_csv(metrics_dir / "fl_strategy_comparison.csv", index=False)
        save_json(metrics_dir / "fl_strategy_comparison.json", {"results": run_records})


if __name__ == "__main__":
    main()
