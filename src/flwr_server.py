from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import flwr as fl
import numpy as np
import pandas as pd
import torch
from flwr.common import NDArrays, Scalar
from sklearn.metrics import log_loss

from common import (
    FraudMLP,
    build_dataloader,
    compute_metrics,
    save_confusion_matrix,
    save_json,
    save_pr_curve,
    save_roc_curve,
    to_dataframe_metrics,
)
from flwr_client import DEVICE, FraudClient, make_client_datasets, set_weights


def evaluate_global(
    model: FraudMLP,
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


def fit_config(server_round: int):
    return {"local_epochs": 1, "batch_size": 128}


def run_federated(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    output_dir: Path,
    partition_mode: str = "iid",
    rounds: int = 20,
    lr: float = 1e-3,
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
        return FraudClient(cid=mapped, data=client_local[mapped], input_dim=input_dim, lr=lr)

    round_logs: List[Dict[str, float]] = []
    global_model = FraudMLP(input_dim=input_dim).to(DEVICE)

    def server_eval(server_round: int, parameters: NDArrays, config: Dict[str, Scalar]):
        set_weights(global_model, parameters)
        loss, metrics, _ = evaluate_global(global_model, x_test, y_test, batch_size=128)
        row = {"round": server_round, "loss": loss}
        row.update(metrics)
        round_logs.append(row)
        return loss, metrics

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=3,
        min_evaluate_clients=3,
        min_available_clients=3,
        on_fit_config_fn=fit_config,
        evaluate_fn=server_eval,
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

    # Final global test evaluation after training.
    if round_logs:
        final_metrics = {k: v for k, v in round_logs[-1].items() if k != "round"}
    else:
        loss, metrics, _ = evaluate_global(global_model, x_test, y_test, batch_size=128)
        final_metrics = {"loss": loss, **metrics}

    # Re-evaluate once to collect prediction scores for plots.
    final_loss, final_metrics_no_loss, y_score = evaluate_global(global_model, x_test, y_test, batch_size=128)
    y_pred = (y_score >= 0.5).astype(int)
    final_metrics = {"loss": final_loss, **final_metrics_no_loss}

    metrics_dir = output_dir / "metrics"
    plots_dir = output_dir / "plots"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(round_logs).to_csv(metrics_dir / f"fl_round_metrics_{partition_mode}.csv", index=False)
    save_json(metrics_dir / f"fl_final_metrics_{partition_mode}.json", final_metrics)
    to_dataframe_metrics(f"fl_fedavg_{partition_mode}", final_metrics_no_loss).assign(
        loss=final_loss
    ).to_csv(metrics_dir / f"fl_results_{partition_mode}.csv", index=False)

    save_confusion_matrix(y_test, y_pred, plots_dir / f"fl_confusion_matrix_{partition_mode}.png")
    save_roc_curve(y_test, y_score, plots_dir / f"fl_roc_curve_{partition_mode}.png")
    save_pr_curve(y_test, y_score, plots_dir / f"fl_pr_curve_{partition_mode}.png")

    print(f"[OK] Flower run complete ({partition_mode}).")
    print(pd.DataFrame([final_metrics]).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="Run Flower FedAvg fraud detection")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--partition_mode", type=str, default="iid", choices=["iid", "noniid"])
    args = parser.parse_args()

    out = Path(args.output_dir)
    processed = out / "processed"
    x_train = np.load(processed / "train_X_dense.npy")
    y_train = np.load(processed / "train_y.npy")
    x_test = np.load(processed / "test_X_dense.npy")
    y_test = np.load(processed / "test_y.npy")

    run_federated(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        output_dir=out,
        partition_mode=args.partition_mode,
        rounds=args.rounds,
        lr=args.lr,
    )


if __name__ == "__main__":
    main()
