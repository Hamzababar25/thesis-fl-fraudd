from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple

import flwr as fl
import numpy as np
import torch
from flwr.common import NDArrays, Scalar
from torch import nn

from attack_simulation import apply_poisoning_attack, is_malicious_client, parse_malicious_clients
from common import FraudMLP, build_dataloader, compute_metrics, train_one_epoch


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_weights(model: nn.Module) -> NDArrays:
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_weights(model: nn.Module, parameters: NDArrays) -> None:
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    model.load_state_dict(state_dict, strict=True)


def _stable_seed(cid: str, round_number: int, base_seed: int) -> int:
    key = f"{cid}:{round_number}:{base_seed}".encode("utf-8")
    digest = hashlib.sha256(key).hexdigest()[:16]
    return int(digest, 16) % (2**32)


def protect_model_update(
    initial: NDArrays,
    updated: NDArrays,
    cid: str,
    round_number: int,
    clip_norm: float,
    noise_multiplier: float,
    dp_seed: int,
) -> Tuple[NDArrays, float, float, float]:
    """Clip and noise model updates before sending them to server."""
    deltas = [u - i for i, u in zip(initial, updated)]
    global_norm = float(np.sqrt(sum(float(np.sum(np.square(d))) for d in deltas)))

    if clip_norm > 0.0:
        clip_coef = min(1.0, clip_norm / (global_norm + 1e-12))
    else:
        clip_coef = 1.0
    clipped = [d * clip_coef for d in deltas]
    clipped_norm = float(np.sqrt(sum(float(np.sum(np.square(d))) for d in clipped)))

    noise_scale = 0.0
    if noise_multiplier > 0.0:
        rng = np.random.default_rng(_stable_seed(cid, round_number, dp_seed))
        noise_scale = noise_multiplier * max(clip_norm, 1e-12)
        noisy = [
            d + rng.normal(loc=0.0, scale=noise_scale, size=d.shape).astype(d.dtype)
            for d in clipped
        ]
    else:
        noisy = clipped

    protected = [i + d for i, d in zip(initial, noisy)]
    return protected, global_norm, clipped_norm, float(noise_scale)


@dataclass
class LocalData:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray


class FraudClient(fl.client.NumPyClient):
    def __init__(
        self,
        cid: str,
        data: LocalData,
        input_dim: int,
        lr: float = 1e-3,
    ) -> None:
        self.cid = cid
        self.data = data
        self.model = FraudMLP(input_dim=input_dim).to(DEVICE)
        self.lr = lr

    def get_parameters(self, config: Dict[str, Scalar]):
        return get_weights(self.model)

    def fit(self, parameters: NDArrays, config: Dict[str, Scalar]):
        initial_parameters = [np.copy(p) for p in parameters]
        set_weights(self.model, initial_parameters)
        batch_size = int(config.get("batch_size", 128))
        local_epochs = int(config.get("local_epochs", 1))
        round_number = int(config.get("server_round", 0))
        # Support both new and old config keys for backward compatibility.
        clip_norm = float(config.get("clip_threshold", config.get("clip_norm", 1.0)))
        noise_multiplier = float(config.get("noise_multiplier", config.get("dp_noise_std", 0.0)))
        dp_seed = int(config.get("dp_seed", 2026))
        attack_enabled = bool(config.get("attack_enabled", False))
        attack_strength = float(config.get("attack_strength", 0.0))
        attack_mode = str(config.get("attack_mode", "sign_flip"))
        malicious_clients = parse_malicious_clients(str(config.get("malicious_clients", "")))

        loader = build_dataloader(self.data.x_train, self.data.y_train, batch_size=batch_size, shuffle=True)
        neg = (self.data.y_train == 0).sum()
        pos = (self.data.y_train == 1).sum()
        pos_weight = float(neg / max(pos, 1))
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=DEVICE))
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        train_losses = []
        for _ in range(local_epochs):
            train_losses.append(train_one_epoch(self.model, loader, optimizer, criterion, DEVICE))

        val_loader = build_dataloader(self.data.x_val, self.data.y_val, batch_size=batch_size, shuffle=False)
        self.model.eval()
        ys = []
        ps = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(DEVICE)
                logits = self.model(xb)
                p = torch.sigmoid(logits).cpu().numpy()
                ys.append(yb.numpy())
                ps.append(p)
        y_true = np.concatenate(ys)
        y_score = np.concatenate(ps)
        val_metrics = compute_metrics(y_true, y_score, threshold=0.5)

        updated = get_weights(self.model)
        is_malicious = attack_enabled and is_malicious_client(self.cid, malicious_clients)
        if is_malicious:
            updated = apply_poisoning_attack(
                initial=initial_parameters,
                updated=updated,
                attack_strength=attack_strength,
                attack_mode=attack_mode,
            )
        protected, update_norm, clipped_update_norm, noise_scale_used = protect_model_update(
            initial=initial_parameters,
            updated=updated,
            cid=self.cid,
            round_number=round_number,
            clip_norm=clip_norm,
            noise_multiplier=noise_multiplier,
            dp_seed=dp_seed,
        )

        metrics = {
            "f1": float(val_metrics["f1"]),
            "recall": float(val_metrics["recall"]),
            "precision": float(val_metrics["precision"]),
            "pos_weight": float(pos_weight),
            "train_loss": float(np.mean(train_losses)),
            # Logged explicitly for security evaluation/traceability.
            "gradient_norm_before_clipping": float(update_norm),
            "gradient_norm_after_clipping": float(clipped_update_norm),
            "clip_threshold": float(clip_norm),
            "noise_multiplier": float(noise_multiplier),
            "noise_scale_used": float(noise_scale_used),
            "dp_seed": float(dp_seed),
            "is_malicious": float(1.0 if is_malicious else 0.0),
            "attack_strength": float(attack_strength if is_malicious else 0.0),
            # Backward-compatible aliases used by existing server logs.
            "update_norm": float(update_norm),
            "clipped_update_norm": float(clipped_update_norm),
            "client_idx": float(int(self.cid.replace("w", ""))),
        }
        return protected, len(self.data.y_train), metrics

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        set_weights(self.model, parameters)
        batch_size = int(config.get("batch_size", 128))
        loader = build_dataloader(self.data.x_val, self.data.y_val, batch_size=batch_size, shuffle=False)
        self.model.eval()
        ys = []
        ps = []
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(DEVICE)
                logits = self.model(xb)
                p = torch.sigmoid(logits).cpu().numpy()
                ys.append(yb.numpy())
                ps.append(p)
        y_true = np.concatenate(ys)
        y_score = np.concatenate(ps)
        metrics = compute_metrics(y_true, y_score, threshold=0.5)
        loss = float(nn.BCELoss()(torch.tensor(y_score), torch.tensor(y_true)).item())
        return loss, len(self.data.y_val), metrics


def create_iid_partitions(n_samples: int, n_clients: int, seed: int = 42) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n_samples)
    rng.shuffle(idx)
    return np.array_split(idx, n_clients)


def create_fraud_skew_partitions(y: np.ndarray, n_clients: int) -> List[np.ndarray]:
    """Create simple non-IID partitions by sorting on label."""
    idx = np.argsort(y)
    return np.array_split(idx, n_clients)


def create_bank_style_noniid_partitions(
    y: np.ndarray,
    n_clients: int,
    seed: int = 42,
) -> List[np.ndarray]:
    """
    Build 3 disjoint client partitions with low/medium/high fraud ratios.
    This avoids zero-fraud clients while keeping non-IID behavior.
    """
    if n_clients != 3:
        raise ValueError("bank-style non-IID currently expects exactly 3 clients")

    rng = np.random.default_rng(seed)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)

    # Low / medium / high fraud bank composition
    pos_splits = [0.15, 0.30, 0.55]
    neg_splits = [0.45, 0.35, 0.20]

    pos_counts = [int(len(pos_idx) * s) for s in pos_splits]
    pos_counts[-1] = len(pos_idx) - sum(pos_counts[:-1])
    neg_counts = [int(len(neg_idx) * s) for s in neg_splits]
    neg_counts[-1] = len(neg_idx) - sum(neg_counts[:-1])

    parts = []
    pos_start = 0
    neg_start = 0
    for i in range(3):
        p_take = pos_idx[pos_start : pos_start + pos_counts[i]]
        n_take = neg_idx[neg_start : neg_start + neg_counts[i]]
        pos_start += pos_counts[i]
        neg_start += neg_counts[i]
        idx = np.concatenate([p_take, n_take])
        rng.shuffle(idx)
        parts.append(idx)
    return parts


def make_client_datasets(
    x_train: np.ndarray,
    y_train: np.ndarray,
    n_clients: int = 3,
    mode: str = "iid",
    seed: int = 42,
) -> Dict[str, LocalData]:
    if mode == "iid":
        partitions = create_iid_partitions(len(y_train), n_clients=n_clients, seed=seed)
    elif mode == "noniid":
        partitions = create_fraud_skew_partitions(y_train, n_clients=n_clients)
    elif mode == "bank_noniid":
        partitions = create_bank_style_noniid_partitions(y_train, n_clients=n_clients, seed=seed)
    else:
        raise ValueError("mode must be one of: iid, noniid, bank_noniid")

    client_data: Dict[str, LocalData] = {}
    for i, pidx in enumerate(partitions):
        cid = f"w{i+1}"
        c_x = x_train[pidx]
        c_y = y_train[pidx]
        n = len(c_y)
        split = int(0.85 * n)
        client_data[cid] = LocalData(
            x_train=c_x[:split],
            y_train=c_y[:split],
            x_val=c_x[split:],
            y_val=c_y[split:],
        )
    return client_data


def main():
    parser = argparse.ArgumentParser(description="Run a Flower client process")
    parser.add_argument("--server_address", type=str, default="127.0.0.1:8080")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--cid", type=str, default="w1", choices=["w1", "w2", "w3"])
    parser.add_argument("--partition_mode", type=str, default="iid", choices=["iid", "noniid", "bank_noniid"])
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    processed = Path(args.output_dir) / "processed"
    x_train = np.load(processed / "train_X_dense.npy")
    y_train = np.load(processed / "train_y.npy")
    clients = make_client_datasets(x_train=x_train, y_train=y_train, n_clients=3, mode=args.partition_mode)
    input_dim = x_train.shape[1]
    client = FraudClient(cid=args.cid, data=clients[args.cid], input_dim=input_dim, lr=args.lr)
    fl.client.start_numpy_client(server_address=args.server_address, client=client)


if __name__ == "__main__":
    main()
