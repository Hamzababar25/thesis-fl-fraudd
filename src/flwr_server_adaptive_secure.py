from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import flwr as fl
import numpy as np
import pandas as pd
import torch
from flwr.common import FitRes, NDArrays, Scalar, ndarrays_to_parameters, parameters_to_ndarrays
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
from flwr_client import DEVICE, FraudClient, apply_deterministic_mask, make_client_datasets, set_weights


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


class AdaptiveSecureFedAvg(fl.server.strategy.FedAvg):
    def __init__(
        self,
        alpha_fraud: float = 0.7,
        adapt_lr: float = 0.4,
        min_client_weight: float = 0.05,
        secure_agg: bool = True,
        mask_seed: int = 2026,
        mask_scale: float = 1e-4,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.alpha_fraud = alpha_fraud
        self.adapt_lr = adapt_lr
        self.min_client_weight = min_client_weight
        self.secure_agg = secure_agg
        self.mask_seed = mask_seed
        self.mask_scale = mask_scale
        self.prev_score: Dict[str, float] = {}
        self.prev_weight: Dict[str, float] = {}
        self.round_weight_rows: List[Dict[str, float]] = []

    def _normalize(self, raw: Dict[str, float]) -> Dict[str, float]:
        vals = {k: max(v, 0.0) for k, v in raw.items()}
        s = sum(vals.values())
        if s <= 0:
            uniform = 1.0 / max(len(vals), 1)
            return {k: uniform for k in vals}
        return {k: v / s for k, v in vals.items()}

    def aggregate_fit(self, server_round: int, results, failures):
        if not results:
            return None, {}

        cids = []
        arrays_by_client: Dict[str, NDArrays] = {}
        perf_score: Dict[str, float] = {}

        for client_proxy, fit_res in results:
            raw_cid = str(client_proxy.cid)
            if "client_idx" in fit_res.metrics:
                cid = f"w{int(float(fit_res.metrics['client_idx']))}"
            else:
                cid = raw_cid
            cids.append(cid)
            arrays_by_client[cid] = parameters_to_ndarrays(fit_res.parameters)
            f1 = float(fit_res.metrics.get("f1", 0.0))
            fraud_ratio = float(fit_res.metrics.get("fraud_ratio", 0.0))
            perf_score[cid] = max(f1, 1e-8) * (1.0 + self.alpha_fraud * fraud_ratio)

        base = self._normalize(perf_score)
        adaptive_raw: Dict[str, float] = {}
        for cid in cids:
            prev_w = self.prev_weight.get(cid, 1.0 / len(cids))
            prev_s = self.prev_score.get(cid, perf_score[cid])
            delta = perf_score[cid] - prev_s
            adaptive_raw[cid] = prev_w + self.adapt_lr * delta
        adaptive = self._normalize(adaptive_raw)

        mixed = {cid: 0.5 * base[cid] + 0.5 * adaptive[cid] for cid in cids}
        mixed = {cid: max(v, self.min_client_weight) for cid, v in mixed.items()}
        client_weight = self._normalize(mixed)

        row = {"round": float(server_round)}
        for cid in cids:
            row[f"{cid}_weight"] = float(client_weight[cid])
            row[f"{cid}_score"] = float(perf_score[cid])
        self.round_weight_rows.append(row)

        template = arrays_by_client[cids[0]]
        aggregated = [np.zeros_like(x) for x in template]
        for cid in cids:
            for i, arr in enumerate(arrays_by_client[cid]):
                aggregated[i] += client_weight[cid] * arr

        if self.secure_agg:
            for cid in cids:
                mask = apply_deterministic_mask(
                    arrays=[np.zeros_like(x) for x in template],
                    cid=cid,
                    round_number=server_round,
                    base_seed=self.mask_seed,
                    scale=self.mask_scale,
                    sign=1.0,
                )
                for i, m in enumerate(mask):
                    aggregated[i] -= client_weight[cid] * m

        for cid in cids:
            self.prev_score[cid] = perf_score[cid]
            self.prev_weight[cid] = client_weight[cid]

        return ndarrays_to_parameters(aggregated), {"secure_agg": float(self.secure_agg)}


def run_adaptive_secure_fl(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    output_dir: Path,
    rounds: int = 20,
    lr: float = 1e-3,
    partition_mode: str = "bank_noniid",
    secure_agg: bool = True,
    alpha_fraud: float = 0.7,
    adapt_lr: float = 0.4,
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

    def fit_config(server_round: int):
        return {
            "local_epochs": 1,
            "batch_size": 128,
            "server_round": server_round,
            "secure_agg": secure_agg,
            "mask_seed": 2026,
            "mask_scale": 1e-4,
        }

    def server_eval(server_round: int, parameters: NDArrays, config: Dict[str, Scalar]):
        set_weights(global_model, parameters)
        loss, metrics, _ = evaluate_global(global_model, x_test, y_test, batch_size=128)
        row = {"round": server_round, "loss": loss}
        row.update(metrics)
        round_logs.append(row)
        return loss, metrics

    strategy = AdaptiveSecureFedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=3,
        min_evaluate_clients=3,
        min_available_clients=3,
        on_fit_config_fn=fit_config,
        evaluate_fn=server_eval,
        alpha_fraud=alpha_fraud,
        adapt_lr=adapt_lr,
        secure_agg=secure_agg,
        mask_seed=2026,
        mask_scale=1e-4,
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

    final_loss, final_metrics_no_loss, y_score = evaluate_global(global_model, x_test, y_test, batch_size=128)
    y_pred = (y_score >= 0.5).astype(int)
    final_metrics = {"loss": final_loss, **final_metrics_no_loss}

    metrics_dir = output_dir / "metrics"
    plots_dir = output_dir / "plots"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    tag = "adaptive_secure"
    pd.DataFrame(round_logs).to_csv(metrics_dir / f"fl_round_metrics_{tag}.csv", index=False)
    pd.DataFrame(strategy.round_weight_rows).to_csv(metrics_dir / f"fl_round_client_weights_{tag}.csv", index=False)
    save_json(metrics_dir / f"fl_final_metrics_{tag}.json", final_metrics)
    to_dataframe_metrics(f"fl_{tag}", final_metrics_no_loss).assign(loss=final_loss).to_csv(
        metrics_dir / f"fl_results_{tag}.csv", index=False
    )

    save_confusion_matrix(y_test, y_pred, plots_dir / f"fl_confusion_matrix_{tag}.png")
    save_roc_curve(y_test, y_score, plots_dir / f"fl_roc_curve_{tag}.png")
    save_pr_curve(y_test, y_score, plots_dir / f"fl_pr_curve_{tag}.png")

    print("[OK] Adaptive Secure FL run complete.")
    print(pd.DataFrame([final_metrics]).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="Run Flower Adaptive Secure FL")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--partition_mode", type=str, default="bank_noniid", choices=["iid", "noniid", "bank_noniid"])
    parser.add_argument("--secure_agg", action="store_true")
    parser.add_argument("--alpha_fraud", type=float, default=0.7)
    parser.add_argument("--adapt_lr", type=float, default=0.4)
    args = parser.parse_args()

    out = Path(args.output_dir)
    processed = out / "processed"
    x_train = np.load(processed / "train_X_dense.npy")
    y_train = np.load(processed / "train_y.npy")
    x_test = np.load(processed / "test_X_dense.npy")
    y_test = np.load(processed / "test_y.npy")

    run_adaptive_secure_fl(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        output_dir=out,
        rounds=args.rounds,
        lr=args.lr,
        partition_mode=args.partition_mode,
        secure_agg=args.secure_agg,
        alpha_fraud=args.alpha_fraud,
        adapt_lr=args.adapt_lr,
    )


if __name__ == "__main__":
    main()
