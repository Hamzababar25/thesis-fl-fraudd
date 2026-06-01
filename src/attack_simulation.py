from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Set

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from flwr.common import NDArrays


def parse_malicious_clients(raw: str) -> Set[str]:
    if not raw.strip():
        return set()
    return {part.strip() for part in raw.split(",") if part.strip()}


def is_malicious_client(cid: str, malicious_clients: Set[str]) -> bool:
    return cid in malicious_clients


def apply_poisoning_attack(
    initial: NDArrays,
    updated: NDArrays,
    attack_strength: float,
    attack_mode: str = "sign_flip",
) -> NDArrays:
    """Manipulate a client update before transmission."""
    strength = float(max(0.0, attack_strength))
    deltas = [u - i for i, u in zip(initial, updated)]

    if attack_mode == "scale":
        manipulated = [i + strength * d for i, d in zip(initial, deltas)]
    else:
        # sign_flip: reverse update direction and amplify by attack strength
        manipulated = [i - strength * d for i, d in zip(initial, deltas)]
    return manipulated


def build_attack_comparison_rows(
    run_records: List[Dict[str, float]],
) -> pd.DataFrame:
    return pd.DataFrame(run_records)


def build_robustness_rows(output_dir: Path, run_labels: Iterable[str]) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    metrics_dir = output_dir / "metrics"
    for label in run_labels:
        path = metrics_dir / f"fl_round_client_weights_{label}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        rejected = df["rejected_clients"].fillna("").astype(str)
        rows.append(
            {
                "scenario": label,
                "rounds": float(len(df)),
                "rounds_with_rejection": float((rejected.str.len() > 0).sum()),
                "avg_rejected_clients": float(
                    rejected.apply(lambda s: 0 if not s else len([x for x in s.split(",") if x])).mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def save_attack_comparison_plots(
    metrics_df: pd.DataFrame,
    robustness_df: pd.DataFrame,
    output_dir: Path,
    prefix: str = "attack_eval",
) -> None:
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: model quality metrics
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    metric_names = ["f1", "precision", "recall"]
    labels = metrics_df["scenario"].tolist()
    x = np.arange(len(labels))
    for ax, metric in zip(axes, metric_names):
        vals = metrics_df[metric].to_numpy(dtype=float)
        ax.bar(x, vals)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_title(metric.upper())
        ax.set_ylim(0.0, max(1.0, vals.max() * 1.15 if len(vals) else 1.0))
    fig.tight_layout()
    fig.savefig(plots_dir / f"{prefix}_quality_metrics.png", dpi=140)
    plt.close(fig)

    # Plot 2: aggregation robustness
    if not robustness_df.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        labels = robustness_df["scenario"].tolist()
        x = np.arange(len(labels))
        vals = robustness_df["avg_rejected_clients"].to_numpy(dtype=float)
        ax.bar(x, vals)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_title("Aggregation Robustness (Avg Rejected Clients)")
        ax.set_ylabel("Clients rejected per round")
        fig.tight_layout()
        fig.savefig(plots_dir / f"{prefix}_aggregation_robustness.png", dpi=140)
        plt.close(fig)

