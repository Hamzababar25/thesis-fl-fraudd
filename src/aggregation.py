from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
from flwr.common import NDArrays


@dataclass
class AggregationResult:
    aggregated: NDArrays
    selected_clients: List[str]
    rejected_clients: List[str]
    client_scores: Dict[str, float]


def _weighted_average(arrays: Sequence[NDArrays], weights: np.ndarray) -> NDArrays:
    template = arrays[0]
    out: NDArrays = [np.zeros_like(x) for x in template]
    for i, client_arr in enumerate(arrays):
        for j, arr in enumerate(client_arr):
            out[j] += weights[i] * arr
    return out


def fedavg_aggregate(cids: List[str], arrays: List[NDArrays], num_examples: List[int]) -> AggregationResult:
    weights = np.asarray(num_examples, dtype=np.float64)
    weights = weights / max(weights.sum(), 1.0)
    aggregated = _weighted_average(arrays, weights)
    return AggregationResult(
        aggregated=aggregated,
        selected_clients=list(cids),
        rejected_clients=[],
        client_scores={cid: float(weights[i]) for i, cid in enumerate(cids)},
    )


def _flatten_update(client_arr: NDArrays) -> np.ndarray:
    return np.concatenate([layer.ravel() for layer in client_arr], axis=0)


def _pairwise_squared_distances(vectors: np.ndarray) -> np.ndarray:
    # vectors shape: [n_clients, n_params]
    diffs = vectors[:, None, :] - vectors[None, :, :]
    return np.sum(diffs * diffs, axis=2)


def multi_krum_aggregate(
    cids: List[str],
    arrays: List[NDArrays],
    num_malicious: int = 1,
    num_selected: int | None = None,
) -> AggregationResult:
    n_clients = len(cids)
    if n_clients == 0:
        raise ValueError("No client updates received for aggregation")
    if n_clients <= 2:
        # Multi-Krum not meaningful; fall back to simple mean.
        aggregated = _weighted_average(arrays, np.ones(n_clients, dtype=np.float64) / n_clients)
        return AggregationResult(
            aggregated=aggregated,
            selected_clients=list(cids),
            rejected_clients=[],
            client_scores={cid: 0.0 for cid in cids},
        )

    vectors = np.stack([_flatten_update(a) for a in arrays], axis=0)
    d2 = _pairwise_squared_distances(vectors)
    np.fill_diagonal(d2, np.inf)

    byzantine = int(max(0, num_malicious))
    neighbors = n_clients - byzantine - 2
    if neighbors <= 0:
        neighbors = max(1, n_clients - 2)

    scores: Dict[str, float] = {}
    for i, cid in enumerate(cids):
        nearest = np.partition(d2[i], neighbors - 1)[:neighbors]
        scores[cid] = float(nearest.sum())

    ranked = sorted(cids, key=lambda c: scores[c])
    m = num_selected if num_selected is not None else (n_clients - byzantine)
    m = int(np.clip(m, 1, n_clients))
    selected = ranked[:m]
    rejected = ranked[m:]

    selected_arrays = [arrays[cids.index(cid)] for cid in selected]
    aggregated = _weighted_average(
        selected_arrays,
        np.ones(len(selected), dtype=np.float64) / len(selected),
    )

    return AggregationResult(
        aggregated=aggregated,
        selected_clients=selected,
        rejected_clients=rejected,
        client_scores=scores,
    )

