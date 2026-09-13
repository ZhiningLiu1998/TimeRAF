from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from ts_rag.benchmark_rag import (
    _feature_configs,
    _validation_partition,
    apply_selected_rag,
    run_validation_selected_rag,
)
from ts_rag.evaluation import compute_protocol_metrics
from ts_rag.historical_retrieval import HistoricalResidualIndex


SYSTEM_IDS = (
    "base",
    "analog_future",
    "raft_adapted",
    "saraf_adapted",
    "residual_retrieval",
    "timeraf",
)


def load_protocol(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as source:
        protocol = json.load(source)
    systems = tuple(system["id"] for system in protocol["systems"])
    if systems != SYSTEM_IDS:
        raise ValueError(f"Unexpected retrieval systems: {systems}")
    return protocol


def load_replay_bundle(
    path: str | Path,
    task_family: str,
    splits: tuple[str, ...] = ("validation", "test"),
) -> tuple[dict, ...]:
    """Load the frozen replay bundle.

    ``splits`` exists so a validation-only pass does not materialize the test
    arrays, which dominate resident memory for the widest long-horizon cells.
    """

    with np.load(path, allow_pickle=False) as archive:
        available = set(archive.files)
        required = {
            f"{split}_{name}"
            for split in ("validation", "test")
            for name in ("x", "y", "y_base", "x_mark", "y_mark")
        }
        missing = required - available
        if missing:
            raise ValueError(f"Replay bundle is missing arrays: {sorted(missing)}")
        payload = {
            f"{split}_{name}": np.asarray(archive[f"{split}_{name}"])
            for split in splits
            for name in ("x", "y", "y_base", "x_mark", "y_mark")
        }

    bundles = []
    for split in splits:
        bundle = {
            name: np.asarray(payload[f"{split}_{name}"], dtype=np.float32)
            for name in ("x", "y", "y_base", "x_mark", "y_mark")
        }
        sample_count = len(bundle["y"])
        if any(len(value) != sample_count for value in bundle.values()):
            raise ValueError(f"{split} arrays have inconsistent sample counts")
        if bundle["y"].shape != bundle["y_base"].shape:
            raise ValueError(f"{split} target and base forecast shapes differ")
        if bundle["x"].shape[2] != bundle["y"].shape[2]:
            raise ValueError(f"{split} input and output channel counts differ")
        if not all(np.isfinite(value).all() for value in bundle.values()):
            raise ValueError(f"{split} bundle contains non-finite values")
        if task_family == "pems":
            bundle["metric_space"] = "inverse_scaled"
        bundles.append(bundle)
    return tuple(bundles)


def validation_partition(sample_count: int, pred_len: int) -> tuple[np.ndarray, np.ndarray]:
    memory, query = _validation_partition(sample_count, pred_len)
    if not len(memory) or not len(query):
        raise ValueError("Validation partition must have memory and query rows")
    if int(memory[-1]) + pred_len >= int(query[0]):
        raise ValueError("Validation memory does not preserve the frozen horizon gap")
    return memory, query


def subset_bundle(bundle: dict, indices: np.ndarray) -> dict:
    sample_count = len(bundle["y"])
    return {
        key: (
            value[indices]
            if isinstance(value, np.ndarray)
            and value.ndim >= 1
            and len(value) == sample_count
            else value
        )
        for key, value in bundle.items()
    }


def query_view(bundle: dict) -> dict:
    return {
        key: bundle[key]
        for key in ("x", "x_mark", "y_base", "y_mark", "metric_space")
        if key in bundle
    }


def _pool_channels(values: np.ndarray, max_channels: int = 16) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.shape[2] <= max_channels:
        return values
    groups = np.array_split(np.arange(values.shape[2]), max_channels)
    return np.stack(
        [values[:, :, group].mean(axis=2) for group in groups],
        axis=2,
    )


def _pool_steps(values: np.ndarray, steps: int = 12) -> np.ndarray:
    steps = min(steps, values.shape[1])
    groups = np.array_split(np.arange(values.shape[1]), steps)
    return np.stack(
        [values[:, group, :].mean(axis=1) for group in groups],
        axis=1,
    )


def _normalize_over_time(values: np.ndarray) -> np.ndarray:
    mean = values.mean(axis=1, keepdims=True)
    scale = values.std(axis=1, keepdims=True)
    return (values - mean) / (scale + 1e-6)


def _l2_normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return values / (np.linalg.norm(values, axis=1, keepdims=True) + 1e-8)


def _analog_descriptor(values: np.ndarray) -> np.ndarray:
    pooled = _pool_channels(values)
    normalized = _normalize_over_time(pooled)
    return _pool_steps(normalized).reshape(len(values), -1).astype(np.float32)


def _period_smooth(values: np.ndarray, period: int) -> np.ndarray:
    if period == 1:
        return np.asarray(values, dtype=np.float32)
    if values.shape[1] % period:
        raise ValueError(
            f"Period {period} does not divide sequence length {values.shape[1]}"
        )
    shape = (
        values.shape[0],
        values.shape[1] // period,
        period,
        values.shape[2],
    )
    means = np.asarray(values, dtype=np.float32).reshape(shape).mean(axis=2)
    return np.repeat(means, period, axis=1)


def _compatible_raft_periods(
    periods: list[int],
    context_length: int,
    prediction_length: int,
) -> list[int]:
    compatible = [
        int(period)
        for period in periods
        if context_length % int(period) == 0
        and prediction_length % int(period) == 0
    ]
    if not compatible:
        raise ValueError(
            "RAFT period set has no period compatible with context "
            f"{context_length} and prediction length {prediction_length}"
        )
    return compatible


def _raft_descriptor(values: np.ndarray, period: int) -> np.ndarray:
    pooled = _pool_channels(values)
    smoothed = _period_smooth(pooled, period)
    offset = smoothed - smoothed[:, -1:, :]
    descriptor = _pool_steps(offset).reshape(len(values), -1)
    descriptor -= descriptor.mean(axis=1, keepdims=True)
    return descriptor.astype(np.float32)


def _mark_descriptor(bundle: dict) -> np.ndarray:
    future_positions = np.unique(
        np.linspace(0, bundle["y_mark"].shape[1] - 1, 4, dtype=int)
    )
    return np.concatenate(
        [
            np.asarray(bundle["x_mark"][:, -1, :], dtype=np.float32),
            np.asarray(
                bundle["y_mark"][:, future_positions, :],
                dtype=np.float32,
            ).reshape(len(bundle["y_mark"]), -1),
        ],
        axis=1,
    )


def _topk_cosine(
    memory: np.ndarray,
    query: np.ndarray,
    max_k: int,
    device: str,
    chunk_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    memory = _l2_normalize(memory)
    query = _l2_normalize(query)
    k = min(max_k, len(memory))
    if k < 1:
        raise ValueError("Retrieval memory is empty")

    torch_device = torch.device(device)
    memory_tensor = torch.from_numpy(memory).to(torch_device)
    indices = np.empty((len(query), k), dtype=np.int64)
    scores = np.empty((len(query), k), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(query), chunk_size):
            end = min(start + chunk_size, len(query))
            query_tensor = torch.from_numpy(query[start:end]).to(torch_device)
            values, local_indices = torch.topk(
                query_tensor @ memory_tensor.T,
                k=k,
                dim=1,
                largest=True,
                sorted=True,
            )
            indices[start:end] = local_indices.cpu().numpy()
            scores[start:end] = values.cpu().numpy()
    return indices, scores


def _cosine_matrix(
    memory: np.ndarray,
    query: np.ndarray,
    device: str,
    chunk_size: int = 256,
) -> np.ndarray:
    memory = _l2_normalize(memory)
    query = _l2_normalize(query)
    torch_device = torch.device(device)
    memory_tensor = torch.from_numpy(memory).to(torch_device)
    similarity = np.empty((len(query), len(memory)), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(query), chunk_size):
            end = min(start + chunk_size, len(query))
            query_tensor = torch.from_numpy(query[start:end]).to(torch_device)
            similarity[start:end] = (
                query_tensor @ memory_tensor.T
            ).cpu().numpy()
    return similarity


def _softmax_weights(scores: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 0:
        return np.full_like(scores, 1.0 / scores.shape[1], dtype=np.float32)
    logits = np.asarray(scores, dtype=np.float64) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    weights = np.exp(logits)
    weights /= weights.sum(axis=1, keepdims=True)
    return weights.astype(np.float32)


def _aligned_future_prediction(
    memory: dict,
    query: dict,
    neighbors: np.ndarray,
    weights: np.ndarray,
    *,
    period: int = 0,
    future_endpoint: bool = False,
    chunk_size: int = 32,
) -> np.ndarray:
    memory_y = np.asarray(memory["y"], dtype=np.float32)
    if period:
        memory_y = _period_smooth(memory_y, period)
    memory_endpoint = (
        memory_y[:, -1:, :]
        if future_endpoint
        else np.asarray(memory["x"][:, -1:, :], dtype=np.float32)
    )
    query_endpoint = np.asarray(query["x"][:, -1:, :], dtype=np.float32)
    output = np.empty(
        (len(neighbors),) + memory_y.shape[1:],
        dtype=np.float32,
    )
    for start in range(0, len(neighbors), chunk_size):
        end = min(start + chunk_size, len(neighbors))
        selected = neighbors[start:end]
        aligned = (
            memory_y[selected]
            - memory_endpoint[selected]
            + query_endpoint[start:end, None, :, :]
        )
        output[start:end] = np.einsum(
            "nk,nkpc->npc",
            weights[start:end],
            aligned,
            optimize=True,
        )
    return output


def _metrics(bundle: dict, prediction: np.ndarray, task_family: str) -> dict:
    prediction = np.asarray(prediction, dtype=np.float32)
    if prediction.shape != bundle["y"].shape:
        raise ValueError(
            f"Prediction shape {prediction.shape} != target shape {bundle['y'].shape}"
        )
    if not np.isfinite(prediction).all():
        raise ValueError("Prediction contains non-finite values")
    metrics = compute_protocol_metrics(prediction, bundle["y"], task_family)
    if not all(math.isfinite(float(value)) for value in metrics.values()):
        raise ValueError("Metrics contain non-finite values")
    return metrics


def _selection_score(baseline: dict, candidate: dict, metric_names: tuple[str, ...]) -> float:
    return max(float(candidate[name]) / float(baseline[name]) for name in metric_names)


def _candidate(
    params: dict,
    metrics: dict,
    baseline: dict,
    metric_names: tuple[str, ...],
) -> dict:
    return {
        "params": params,
        "metrics": metrics,
        "selection_score": _selection_score(baseline, metrics, metric_names),
    }


def _choose(candidates: list[dict]) -> dict:
    return min(
        candidates,
        key=lambda row: (
            row["selection_score"],
            float(row["params"].get("alpha", 0.0)),
            int(row["params"].get("k", 0)),
            json.dumps(
                row["params"],
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ),
        ),
    )


def _blend(base: np.ndarray, retrieval: np.ndarray, alpha: float) -> np.ndarray:
    return (
        (1.0 - float(alpha)) * np.asarray(base, dtype=np.float32)
        + float(alpha) * np.asarray(retrieval, dtype=np.float32)
    ).astype(np.float32)


def _base_candidate(
    query_bundle: dict,
    task_family: str,
    metric_names: tuple[str, ...],
) -> tuple[dict, list[dict]]:
    baseline = _metrics(query_bundle, query_bundle["y_base"], task_family)
    identity = _candidate(
        {"alpha": 0.0},
        baseline,
        baseline,
        metric_names,
    )
    return baseline, [identity]


def _analog_search(
    memory: dict,
    query_bundle: dict,
    protocol: dict,
    task_family: str,
    metric_names: tuple[str, ...],
    device: str,
) -> dict:
    baseline, candidates = _base_candidate(
        query_bundle, task_family, metric_names
    )
    config = next(
        system for system in protocol["systems"] if system["id"] == "analog_future"
    )
    neighbors, scores = _topk_cosine(
        _analog_descriptor(memory["x"]),
        _analog_descriptor(query_bundle["x"]),
        max(config["k_grid"]),
        device,
    )
    for weighting in config["weighting"]:
        for k in config["k_grid"]:
            local_neighbors = neighbors[:, :k]
            local_scores = scores[:, :k]
            if weighting == "uniform":
                weights = _softmax_weights(local_scores, 0.0)
            elif weighting == "softmax_cosine_temperature_0.1":
                weights = _softmax_weights(local_scores, 0.1)
            else:
                raise ValueError(f"Unknown analog weighting: {weighting}")
            retrieval = _aligned_future_prediction(
                memory,
                query_view(query_bundle),
                local_neighbors,
                weights,
            )
            for alpha in protocol["common_processing"]["alpha_grid"]:
                prediction = _blend(query_bundle["y_base"], retrieval, alpha)
                candidates.append(
                    _candidate(
                        {"alpha": alpha, "k": k, "weighting": weighting},
                        _metrics(query_bundle, prediction, task_family),
                        baseline,
                        metric_names,
                    )
                )
    return {
        "baseline": baseline,
        "selected": _choose(candidates),
        "candidate_count": len(candidates),
    }


def _analog_apply(
    memory: dict,
    query: dict,
    params: dict,
    device: str,
) -> np.ndarray:
    if params["alpha"] == 0:
        return np.asarray(query["y_base"], dtype=np.float32).copy()
    neighbors, scores = _topk_cosine(
        _analog_descriptor(memory["x"]),
        _analog_descriptor(query["x"]),
        params["k"],
        device,
    )
    temperature = (
        0.0
        if params["weighting"] == "uniform"
        else 0.1
    )
    retrieval = _aligned_future_prediction(
        memory,
        query,
        neighbors,
        _softmax_weights(scores, temperature),
    )
    return _blend(query["y_base"], retrieval, params["alpha"])


def _raft_retrieval(
    memory: dict,
    query: dict,
    periods: list[int],
    k: int,
    temperature: float,
    device: str,
) -> np.ndarray:
    predictions = []
    for period in periods:
        neighbors, scores = _topk_cosine(
            _raft_descriptor(memory["x"], period),
            _raft_descriptor(query["x"], period),
            k,
            device,
        )
        predictions.append(
            _aligned_future_prediction(
                memory,
                query,
                neighbors,
                _softmax_weights(scores, temperature),
                period=period,
                future_endpoint=True,
            )
        )
    return np.mean(predictions, axis=0, dtype=np.float32)


def _raft_search(
    memory: dict,
    query_bundle: dict,
    protocol: dict,
    task_family: str,
    metric_names: tuple[str, ...],
    device: str,
) -> dict:
    baseline, candidates = _base_candidate(
        query_bundle, task_family, metric_names
    )
    config = next(
        system for system in protocol["systems"] if system["id"] == "raft_adapted"
    )
    for configured_periods in config["period_sets"]:
        periods = _compatible_raft_periods(
            configured_periods,
            memory["x"].shape[1],
            query_bundle["y"].shape[1],
        )
        for k in config["k_grid"]:
            for temperature in config["temperature_grid"]:
                retrieval = _raft_retrieval(
                    memory,
                    query_view(query_bundle),
                    periods,
                    k,
                    temperature,
                    device,
                )
                for alpha in protocol["common_processing"]["alpha_grid"]:
                    prediction = _blend(
                        query_bundle["y_base"], retrieval, alpha
                    )
                    candidates.append(
                        _candidate(
                            {
                                "alpha": alpha,
                                "k": k,
                                "periods": periods,
                                "temperature": temperature,
                            },
                            _metrics(query_bundle, prediction, task_family),
                            baseline,
                            metric_names,
                        )
                    )
    return {
        "baseline": baseline,
        "selected": _choose(candidates),
        "candidate_count": len(candidates),
    }


def _raft_apply(
    memory: dict,
    query: dict,
    params: dict,
    device: str,
) -> np.ndarray:
    if params["alpha"] == 0:
        return np.asarray(query["y_base"], dtype=np.float32).copy()
    retrieval = _raft_retrieval(
        memory,
        query,
        params["periods"],
        params["k"],
        params["temperature"],
        device,
    )
    return _blend(query["y_base"], retrieval, params["alpha"])


def _stationarity(memory_x: np.ndarray) -> float:
    values = _pool_channels(memory_x)
    windows = np.array_split(np.arange(values.shape[1]), 6)
    means = np.stack(
        [values[:, window, :].mean(axis=1) for window in windows],
        axis=1,
    )
    standard_deviations = np.stack(
        [values[:, window, :].std(axis=1) for window in windows],
        axis=1,
    )
    mean_variation = means.std(axis=1).mean(axis=1)
    std_variation = standard_deviations.std(axis=1).mean(axis=1)
    global_std = values.std(axis=1).mean(axis=1)
    mean_score = 1.0 - np.clip(mean_variation / (global_std + 1e-8), 0, 1)
    std_score = 1.0 - np.clip(std_variation / (global_std + 1e-8), 0, 1)
    return float(np.clip(0.5 * (mean_score + std_score), 0, 1).mean())


def _time_bonus(memory: dict, query: dict, device: str) -> np.ndarray:
    memory_marks = _mark_descriptor(memory)
    query_marks = _mark_descriptor(query)
    memory_norms = np.linalg.norm(memory_marks, axis=1)
    query_norms = np.linalg.norm(query_marks, axis=1)
    similarity = _cosine_matrix(
        memory_marks,
        query_marks,
        device,
    )
    bonus = 0.5 * (similarity + 1.0)
    neutral = (query_norms[:, None] == 0) | (memory_norms[None, :] == 0)
    bonus[neutral] = 0.5
    return bonus.astype(np.float32)


def _deterministic_mmr(
    candidate_indices: np.ndarray,
    candidate_scores: np.ndarray,
    k: int,
    lambda_value: float,
) -> tuple[np.ndarray, np.ndarray]:
    selected_indices = np.empty((len(candidate_indices), k), dtype=np.int64)
    selected_scores = np.empty((len(candidate_indices), k), dtype=np.float32)
    for row in range(len(candidate_indices)):
        order = np.lexsort(
            (candidate_indices[row], -candidate_scores[row])
        )
        indices = candidate_indices[row, order]
        relevance = candidate_scores[row, order]
        chosen = []
        available = np.ones(len(indices), dtype=bool)
        for step in range(k):
            if step == 0:
                position = 0
            else:
                selected_relevance = relevance[np.asarray(chosen)]
                redundancy = np.max(
                    1.0
                    - np.abs(
                        relevance[:, None]
                        - selected_relevance[None, :]
                    ),
                    axis=1,
                )
                score = (
                    lambda_value * relevance
                    - (1.0 - lambda_value) * redundancy
                )
                score[~available] = -np.inf
                best = np.flatnonzero(score == np.max(score))
                position = int(
                    best[np.argmin(indices[best])]
                )
            chosen.append(position)
            available[position] = False
            selected_indices[row, step] = indices[position]
            selected_scores[row, step] = relevance[position]
    return selected_indices, selected_scores


def _saraf_select(
    similarity: np.ndarray,
    candidate_pool: int,
    max_k: int,
    lambda_value: float,
) -> tuple[np.ndarray, np.ndarray]:
    pool_indices = np.argpartition(
        similarity,
        -min(candidate_pool, similarity.shape[1]),
        axis=1,
    )[:, -min(candidate_pool, similarity.shape[1]):]
    pool_scores = np.take_along_axis(similarity, pool_indices, axis=1)
    return _deterministic_mmr(
        pool_indices,
        pool_scores,
        min(max_k, pool_indices.shape[1]),
        lambda_value,
    )


def _gaussian_weights(scores: np.ndarray, sigma: float) -> np.ndarray:
    weights = np.exp(
        -np.square(1.0 - np.asarray(scores, dtype=np.float64))
        / (2.0 * sigma * sigma)
    )
    totals = weights.sum(axis=1, keepdims=True)
    zero = totals[:, 0] == 0
    if np.any(zero):
        weights[zero] = 1.0
        totals = weights.sum(axis=1, keepdims=True)
    return (weights / totals).astype(np.float32)


def _saraf_search(
    memory: dict,
    query_bundle: dict,
    protocol: dict,
    task_family: str,
    metric_names: tuple[str, ...],
    device: str,
) -> dict:
    baseline, candidates = _base_candidate(
        query_bundle, task_family, metric_names
    )
    config = next(
        system for system in protocol["systems"] if system["id"] == "saraf_adapted"
    )
    stationarity = _stationarity(memory["x"])
    sigma = config["sigma_range"][0] + (1.0 - stationarity) * (
        config["sigma_range"][1] - config["sigma_range"][0]
    )
    lambda_value = config["mmr_lambda_range"][0] + stationarity * (
        config["mmr_lambda_range"][1] - config["mmr_lambda_range"][0]
    )
    temporal = _cosine_matrix(
        _raft_descriptor(memory["x"], 1),
        _raft_descriptor(query_bundle["x"], 1),
        device,
    )
    time_bonus = None
    for time_weight in config["time_alignment_weight_grid"]:
        if time_weight:
            if time_bonus is None:
                time_bonus = _time_bonus(
                    memory,
                    query_view(query_bundle),
                    device,
                )
            similarity = (
                (1.0 - time_weight) * temporal
                + time_weight * time_bonus
            )
        else:
            similarity = temporal
        neighbors, scores = _saraf_select(
            similarity,
            config["candidate_pool"],
            max(config["k_grid"]),
            lambda_value,
        )
        for k in config["k_grid"]:
            retrieval = _aligned_future_prediction(
                memory,
                query_view(query_bundle),
                neighbors[:, :k],
                _gaussian_weights(scores[:, :k], sigma),
                period=1,
                future_endpoint=True,
            )
            for alpha in protocol["common_processing"]["alpha_grid"]:
                prediction = _blend(query_bundle["y_base"], retrieval, alpha)
                candidates.append(
                    _candidate(
                        {
                            "alpha": alpha,
                            "k": k,
                            "lambda": lambda_value,
                            "sigma": sigma,
                            "stationarity": stationarity,
                            "time_alignment_weight": time_weight,
                        },
                        _metrics(query_bundle, prediction, task_family),
                        baseline,
                        metric_names,
                    )
                )
    return {
        "baseline": baseline,
        "selected": _choose(candidates),
        "candidate_count": len(candidates),
    }


def _saraf_apply(
    memory: dict,
    query: dict,
    params: dict,
    protocol: dict,
    device: str,
) -> np.ndarray:
    if params["alpha"] == 0:
        return np.asarray(query["y_base"], dtype=np.float32).copy()
    config = next(
        system for system in protocol["systems"] if system["id"] == "saraf_adapted"
    )
    temporal = _cosine_matrix(
        _raft_descriptor(memory["x"], 1),
        _raft_descriptor(query["x"], 1),
        device,
    )
    similarity = temporal
    if params["time_alignment_weight"]:
        similarity = (
            (1.0 - params["time_alignment_weight"]) * temporal
            + params["time_alignment_weight"]
            * _time_bonus(memory, query, device)
        )
    neighbors, scores = _saraf_select(
        similarity,
        config["candidate_pool"],
        params["k"],
        params["lambda"],
    )
    retrieval = _aligned_future_prediction(
        memory,
        query,
        neighbors,
        _gaussian_weights(scores, params["sigma"]),
        period=1,
        future_endpoint=True,
    )
    return _blend(query["y_base"], retrieval, params["alpha"])


def _residual_search(
    memory: dict,
    query_bundle: dict,
    protocol: dict,
    task_family: str,
    metric_names: tuple[str, ...],
) -> dict:
    baseline, candidates = _base_candidate(
        query_bundle, task_family, metric_names
    )
    config = next(
        system
        for system in protocol["systems"]
        if system["id"] == "residual_retrieval"
    )
    for feature_config in _feature_configs(memory, task_family):
        index = HistoricalResidualIndex(feature_config).fit(memory)
        neighbors, scores = index.query_neighbors(
            query_bundle,
            max_k=16,
        )
        for k, temperature, shrinkage in (
            (8, 0.1, 1.0),
            (16, 0.3, 4.0),
        ):
            correction = index.aggregate(
                neighbors,
                scores,
                k=k,
                temperature=temperature,
                shrinkage=shrinkage,
            )
            for alpha in config["alpha_grid"]:
                prediction = (
                    np.asarray(query_bundle["y_base"], dtype=np.float32)
                    + float(alpha) * correction
                )
                candidates.append(
                    _candidate(
                        {
                            "alpha": alpha,
                            "feature_config": feature_config.to_dict(),
                            "k": k,
                            "temperature": temperature,
                            "shrinkage": shrinkage,
                        },
                        _metrics(query_bundle, prediction, task_family),
                        baseline,
                        metric_names,
                    )
                )
    return {
        "baseline": baseline,
        "selected": _choose(candidates),
        "candidate_count": len(candidates),
    }


def _residual_apply(
    memory: dict,
    query: dict,
    params: dict,
) -> np.ndarray:
    if params["alpha"] == 0:
        return np.asarray(query["y_base"], dtype=np.float32).copy()
    from ts_rag.historical_retrieval import RetrievalFeatureConfig

    index = HistoricalResidualIndex(
        RetrievalFeatureConfig(**params["feature_config"])
    ).fit(memory)
    neighbors, scores = index.query_neighbors(query, max_k=params["k"])
    correction = index.aggregate(
        neighbors,
        scores,
        k=params["k"],
        temperature=params["temperature"],
        shrinkage=params["shrinkage"],
    )
    return (
        np.asarray(query["y_base"], dtype=np.float32)
        + float(params["alpha"]) * correction
    )


def _prediction_sha256(prediction: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(prediction, dtype=np.float32)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def _system_result(
    system_id: str,
    validation_search: dict,
    test_bundle: dict,
    prediction: np.ndarray,
    task_family: str,
    metric_names: tuple[str, ...],
) -> dict:
    baseline = _metrics(test_bundle, test_bundle["y_base"], task_family)
    metrics = _metrics(test_bundle, prediction, task_family)
    return {
        "system": system_id,
        "validation": validation_search,
        "test_metrics": metrics,
        "metric_gain_percent": {
            name: 100.0
            * (float(baseline[name]) - float(metrics[name]))
            / float(baseline[name])
            for name in metric_names
        },
        "strict_win": all(
            float(metrics[name]) < float(baseline[name])
            for name in metric_names
        ),
        "prediction_sha256": _prediction_sha256(prediction),
    }


def _time_raf_no_lookahead(
    validation_bundle: dict,
    test_bundle: dict,
    selected: dict,
    task_family: str,
    pred_len: int,
    reference: np.ndarray,
) -> dict:
    if selected["method"] != "causal_bias":
        modified = dict(test_bundle)
        modified["y"] = np.asarray(test_bundle["y"]).copy() + 12345.0
        repeated = apply_selected_rag(
            validation_bundle,
            modified,
            selected,
            task_family,
            pred_len,
        )
        np.testing.assert_array_equal(reference, repeated)
        return {"passed": True, "mode": "all_test_labels_perturbed"}

    from ts_rag.benchmark_rag import _origin_layout

    _, test_origins = _origin_layout(
        validation_bundle,
        test_bundle,
        task_family,
        pred_len,
    )
    checked = sorted({0, len(test_origins) // 2, len(test_origins) - 1})
    target_times = test_origins[:, None] + np.arange(pred_len)[None, :]
    for query_index in checked:
        modified = dict(test_bundle)
        changed = np.asarray(test_bundle["y"]).copy()
        changed[target_times >= test_origins[query_index]] += 12345.0
        modified["y"] = changed
        repeated = apply_selected_rag(
            validation_bundle,
            modified,
            selected,
            task_family,
            pred_len,
        )
        np.testing.assert_array_equal(
            reference[query_index],
            repeated[query_index],
        )
    return {
        "passed": True,
        "mode": "unrealized_targets_perturbed",
        "query_indices": checked,
    }


def run_retrieval_baseline_cell(
    validation_bundle: dict,
    test_bundle: dict,
    task_family: str,
    metric_names: tuple[str, ...],
    pred_len: int,
    protocol: dict,
    device: str,
) -> dict:
    memory_indices, query_indices = validation_partition(
        len(validation_bundle["y"]),
        pred_len,
    )
    memory = subset_bundle(validation_bundle, memory_indices)
    validation_query = subset_bundle(validation_bundle, query_indices)
    test_query = query_view(test_bundle)

    base_metrics = _metrics(test_bundle, test_bundle["y_base"], task_family)
    systems = {
        "base": {
            "system": "base",
            "validation": None,
            "test_metrics": base_metrics,
            "metric_gain_percent": {name: 0.0 for name in metric_names},
            "strict_win": False,
            "prediction_sha256": _prediction_sha256(test_bundle["y_base"]),
        }
    }

    search = _analog_search(
        memory,
        validation_query,
        protocol,
        task_family,
        metric_names,
        device,
    )
    prediction = _analog_apply(
        validation_bundle,
        test_query,
        search["selected"]["params"],
        device,
    )
    systems["analog_future"] = _system_result(
        "analog_future",
        search,
        test_bundle,
        prediction,
        task_family,
        metric_names,
    )

    search = _raft_search(
        memory,
        validation_query,
        protocol,
        task_family,
        metric_names,
        device,
    )
    prediction = _raft_apply(
        validation_bundle,
        test_query,
        search["selected"]["params"],
        device,
    )
    systems["raft_adapted"] = _system_result(
        "raft_adapted",
        search,
        test_bundle,
        prediction,
        task_family,
        metric_names,
    )

    search = _saraf_search(
        memory,
        validation_query,
        protocol,
        task_family,
        metric_names,
        device,
    )
    prediction = _saraf_apply(
        validation_bundle,
        test_query,
        search["selected"]["params"],
        protocol,
        device,
    )
    systems["saraf_adapted"] = _system_result(
        "saraf_adapted",
        search,
        test_bundle,
        prediction,
        task_family,
        metric_names,
    )

    search = _residual_search(
        memory,
        validation_query,
        protocol,
        task_family,
        metric_names,
    )
    prediction = _residual_apply(
        validation_bundle,
        test_query,
        search["selected"]["params"],
    )
    systems["residual_retrieval"] = _system_result(
        "residual_retrieval",
        search,
        test_bundle,
        prediction,
        task_family,
        metric_names,
    )

    timeraf_result, timeraf_prediction = run_validation_selected_rag(
        validation_bundle,
        test_bundle,
        task_family,
        metric_names,
        pred_len,
    )
    systems["timeraf"] = {
        "system": "timeraf",
        "validation": timeraf_result["validation"],
        "test_metrics": timeraf_result["test_corrected"],
        "metric_gain_percent": timeraf_result["metric_improvement_percent"],
        "strict_win": timeraf_result["all_test_metrics_improve"],
        "prediction_sha256": _prediction_sha256(timeraf_prediction),
    }

    no_lookahead = {
        "retrieval_systems": {
            "passed": True,
            "mode": "query_labels_absent_from_prediction_api",
        },
        "timeraf": _time_raf_no_lookahead(
            validation_bundle,
            test_bundle,
            timeraf_result["validation"]["selected"],
            task_family,
            pred_len,
            timeraf_prediction,
        ),
    }
    if tuple(systems) != SYSTEM_IDS:
        raise AssertionError("Retrieval system output order drifted")
    return {
        "validation_memory_count": int(len(memory_indices)),
        "validation_query_count": int(len(query_indices)),
        "test_count": int(len(test_bundle["y"])),
        "systems": systems,
        "no_lookahead": no_lookahead,
    }
