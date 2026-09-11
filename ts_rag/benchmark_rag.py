import itertools
import json

import numpy as np

from ts_rag.benchmark import all_metrics_improve
from ts_rag.evaluation import compute_protocol_metrics
from ts_rag.historical_retrieval import (
    HistoricalResidualIndex,
    RetrievalFeatureConfig,
)
from ts_rag.online_ensemble import overlapping_forecast_ensemble


def prediction_in_metric_space(bundle, prediction, task_family):
    prediction = np.asarray(prediction)
    if task_family != "pems":
        return prediction, np.asarray(bundle["y"])
    if bundle.get("metric_space") == "inverse_scaled":
        return prediction, np.asarray(bundle["y"])
    scale = np.asarray(bundle["output_scale"]).reshape(1, 1, -1)
    mean = np.asarray(bundle["output_mean"]).reshape(1, 1, -1)
    return prediction * scale + mean, np.asarray(bundle["y_inv"])


def protocol_metrics(bundle, prediction, task_family):
    pred, true = prediction_in_metric_space(bundle, prediction, task_family)
    return compute_protocol_metrics(pred, true, task_family)


def _selection_score(baseline, candidate, metric_names):
    return max(float(candidate[name]) / float(baseline[name]) for name in metric_names)


def _validation_partition(sample_count, pred_len):
    split = max(sample_count // 2, pred_len + 8)
    if split >= sample_count:
        split = max(2, sample_count * 2 // 3)
    memory_end = max(1, split - pred_len)
    return np.arange(memory_end), np.arange(split, sample_count)


def _feature_configs(bundle, task_family):
    tail_lengths = sorted({min(48, bundle["x"].shape[1]), bundle["x"].shape[1]})
    for tail_len, include_forecast in itertools.product(tail_lengths, (False, True)):
        yield RetrievalFeatureConfig(
            tail_len=tail_len,
            pooled_steps=min(12, tail_len),
            include_differences=True,
            include_forecast=include_forecast,
            include_calendar=task_family != "pems",
            max_channels=16,
        )


def _seasonal_periods(bundle, task_family):
    seq_len = bundle["x"].shape[1]
    if task_family == "pems":
        candidates = (12, 24, 48, 96)
    elif task_family == "epf":
        candidates = (24, 48, 168)
    else:
        candidates = (4, 8, 12, 24, 48, 96, 168)
    return sorted({min(period, seq_len) for period in candidates})


def _seasonal_prediction(bundle, period):
    pred_len = bundle["y_base"].shape[1]
    repeats = int(np.ceil(pred_len / period))
    return np.tile(bundle["x"][:, -period:, :], (1, repeats, 1))[:, :pred_len]


def _candidate_row(name, params, metrics, baseline, metric_names):
    return {
        "method": name,
        "params": params,
        "metrics": metrics,
        "selection_score": _selection_score(baseline, metrics, metric_names),
        "all_validation_metrics_improve": all_metrics_improve(
            baseline, metrics, metric_names
        ),
    }


def _subset_bundle(bundle, indices):
    sample_count = len(bundle["y"])
    return {
        key: (
            value[indices]
            if isinstance(value, np.ndarray)
            and value.ndim >= 2
            and len(value) == sample_count
            else value
        )
        for key, value in bundle.items()
    }


def _origin_layout(validation_bundle, test_bundle, task_family, pred_len):
    validation_origins = np.arange(len(validation_bundle["y"]), dtype=np.int64)
    test_stride = 12 if task_family == "pems" else 1
    test_start = len(validation_origins) + pred_len - 1
    if task_family == "pems":
        # PEMS splits raw arrays before windowing, so the test split consumes
        # a fresh context window rather than reusing validation-tail history.
        test_start += validation_bundle["x"].shape[1]
    test_origins = test_start + test_stride * np.arange(
        len(test_bundle["y"]), dtype=np.int64
    )
    return validation_origins, test_origins


def build_causal_bias_statistics(bundles, origins, horizon_block=0):
    pred_len = bundles[0]["y"].shape[1]
    channel_count = bundles[0]["y"].shape[2]
    group_count = (
        int(np.ceil(pred_len / horizon_block)) if horizon_block else 1
    )
    timeline_length = max(
        int(split_origins[-1]) + pred_len
        for split_origins in origins
    )
    sums = np.zeros(
        (timeline_length, group_count, channel_count), dtype=np.float64
    )
    counts = np.zeros((timeline_length, group_count, 1), dtype=np.float64)

    for bundle, split_origins in zip(bundles, origins):
        residual = (
            np.asarray(bundle["y"], dtype=np.float64)
            - np.asarray(bundle["y_base"], dtype=np.float64)
        )
        for horizon in range(pred_len):
            target_times = split_origins + horizon
            group = horizon // horizon_block if horizon_block else 0
            np.add.at(sums[:, group], target_times, residual[:, horizon])
            np.add.at(counts[:, group, 0], target_times, 1.0)

    return (
        np.concatenate([np.zeros_like(sums[:1]), np.cumsum(sums, axis=0)]),
        np.concatenate([np.zeros_like(counts[:1]), np.cumsum(counts, axis=0)]),
    )


def causal_residual_bias(
    statistics,
    query_origins,
    pred_len,
    window,
    horizon_block=0,
):
    sums, counts = statistics
    query_origins = np.asarray(query_origins, dtype=np.int64)
    starts = np.maximum(query_origins - window, 0)
    local_sums = sums[query_origins] - sums[starts]
    local_counts = counts[query_origins] - counts[starts]
    grouped_bias = np.divide(
        local_sums,
        local_counts,
        out=np.zeros_like(local_sums),
        where=local_counts > 0,
    )
    if not horizon_block:
        return np.repeat(grouped_bias, pred_len, axis=1)
    return np.repeat(grouped_bias, horizon_block, axis=1)[:, :pred_len]


def _online_validation_candidates(
    bundle,
    query_indices,
    baseline_metrics,
    task_family,
    metric_names,
    pred_len,
):
    rows = []
    query_bundle = _subset_bundle(bundle, query_indices)
    baseline_prediction = query_bundle["y_base"]

    for max_age, decay in ((8, 0.75), (24, 0.9)):
        revised = overlapping_forecast_ensemble(
            bundle["y_base"], max_age=max_age, decay=decay
        )[query_indices]
        for alpha in (0.1, 0.25, 0.5):
            prediction = (
                (1.0 - alpha) * baseline_prediction + alpha * revised
            )
            rows.append(
                _candidate_row(
                    "overlap_blend",
                    {
                        "max_age": max_age,
                        "decay": decay,
                        "alpha": alpha,
                    },
                    protocol_metrics(query_bundle, prediction, task_family),
                    baseline_metrics,
                    metric_names,
                )
            )

    origins = np.arange(len(bundle["y"]), dtype=np.int64)
    horizon_blocks = sorted({0, min(24, pred_len)})
    windows = sorted({2 * pred_len, 8 * pred_len, 32 * pred_len})
    for horizon_block in horizon_blocks:
        statistics = build_causal_bias_statistics(
            [bundle], [origins], horizon_block=horizon_block
        )
        for window in windows:
            bias = causal_residual_bias(
                statistics,
                origins[query_indices],
                pred_len,
                window,
                horizon_block=horizon_block,
            )
            for alpha in (0.1, 0.25, 0.5, 1.0):
                prediction = baseline_prediction + alpha * bias
                rows.append(
                    _candidate_row(
                        "causal_bias",
                        {
                            "window": window,
                            "horizon_block": horizon_block,
                            "alpha": alpha,
                        },
                        protocol_metrics(
                            query_bundle, prediction, task_family
                        ),
                        baseline_metrics,
                        metric_names,
                    )
                )
    return rows


def _choose_best(rows):
    return min(
        rows,
        key=lambda row: (
            row["selection_score"],
            0 if row["method"] == "identity" else 1,
            row["method"],
            json.dumps(
                row["params"],
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ),
        ),
    )


def search_validation_rag(bundle, task_family, metric_names, pred_len):
    memory_indices, query_indices = _validation_partition(len(bundle["y"]), pred_len)
    query_bundle = _subset_bundle(bundle, query_indices)
    baseline_prediction = query_bundle["y_base"]
    baseline_metrics = protocol_metrics(query_bundle, baseline_prediction, task_family)
    rows = [
        _candidate_row(
            "identity",
            {},
            baseline_metrics,
            baseline_metrics,
            metric_names,
        )
    ]

    for config in _feature_configs(bundle, task_family):
        index = HistoricalResidualIndex(config).fit(bundle, memory_indices)
        neighbors, scores = index.query_neighbors(bundle, query_indices, max_k=16)
        for k, temperature, shrinkage in ((8, 0.1, 1.0), (16, 0.3, 4.0)):
            correction = index.aggregate(
                neighbors,
                scores,
                k=k,
                temperature=temperature,
                shrinkage=shrinkage,
            )
            for alpha in (0.05, 0.1, 0.2, 0.3, 0.5):
                prediction = baseline_prediction + alpha * correction
                metrics = protocol_metrics(query_bundle, prediction, task_family)
                rows.append(
                    _candidate_row(
                        "historical_residual",
                        {
                            "feature_config": config.to_dict(),
                            "k": k,
                            "temperature": temperature,
                            "shrinkage": shrinkage,
                            "alpha": alpha,
                        },
                        metrics,
                        baseline_metrics,
                        metric_names,
                    )
                )

    for period in _seasonal_periods(query_bundle, task_family):
        seasonal = _seasonal_prediction(query_bundle, period)
        for alpha in (0.02, 0.05, 0.1, 0.2, 0.3):
            prediction = (1.0 - alpha) * baseline_prediction + alpha * seasonal
            metrics = protocol_metrics(query_bundle, prediction, task_family)
            rows.append(
                _candidate_row(
                    "seasonal_blend",
                    {"period": period, "alpha": alpha},
                    metrics,
                    baseline_metrics,
                    metric_names,
                )
            )

    residual_bias = np.mean(
        bundle["y"][memory_indices] - bundle["y_base"][memory_indices],
        axis=(0, 1),
        keepdims=True,
    )
    for alpha in (0.1, 0.25, 0.5, 1.0):
        prediction = baseline_prediction + alpha * residual_bias
        metrics = protocol_metrics(query_bundle, prediction, task_family)
        rows.append(
            _candidate_row(
                "residual_bias",
                {"alpha": alpha},
                metrics,
                baseline_metrics,
                metric_names,
            )
        )

    rows.extend(
        _online_validation_candidates(
            bundle,
            query_indices,
            baseline_metrics,
            task_family,
            metric_names,
            pred_len,
        )
    )

    return {
        "memory_count": int(len(memory_indices)),
        "query_count": int(len(query_indices)),
        "baseline": baseline_metrics,
        "selection_policy": "validation_argmin_v3",
        "selected": _choose_best(rows),
        "candidates": rows,
    }


def apply_selected_rag(
    validation_bundle,
    test_bundle,
    selected,
    task_family,
    pred_len,
):
    method = selected["method"]
    params = selected["params"]
    if method == "identity":
        return np.asarray(test_bundle["y_base"]).copy()
    if method == "seasonal_blend":
        seasonal = _seasonal_prediction(test_bundle, params["period"])
        return (
            (1.0 - params["alpha"]) * test_bundle["y_base"]
            + params["alpha"] * seasonal
        )
    if method == "residual_bias":
        bias = np.mean(
            validation_bundle["y"] - validation_bundle["y_base"],
            axis=(0, 1),
            keepdims=True,
        )
        return test_bundle["y_base"] + params["alpha"] * bias
    if method == "overlap_blend":
        revised = overlapping_forecast_ensemble(
            test_bundle["y_base"],
            max_age=params["max_age"],
            decay=params["decay"],
        )
        return (
            (1.0 - params["alpha"]) * test_bundle["y_base"]
            + params["alpha"] * revised
        )
    if method == "causal_bias":
        validation_origins, test_origins = _origin_layout(
            validation_bundle,
            test_bundle,
            task_family,
            pred_len,
        )
        statistics = build_causal_bias_statistics(
            [validation_bundle, test_bundle],
            [validation_origins, test_origins],
            horizon_block=params["horizon_block"],
        )
        bias = causal_residual_bias(
            statistics,
            test_origins,
            pred_len,
            params["window"],
            horizon_block=params["horizon_block"],
        )
        return test_bundle["y_base"] + params["alpha"] * bias
    if method == "historical_residual":
        config = RetrievalFeatureConfig(**params["feature_config"])
        index = HistoricalResidualIndex(config).fit(validation_bundle)
        neighbors, scores = index.query_neighbors(
            test_bundle, max_k=params["k"]
        )
        correction = index.aggregate(
            neighbors,
            scores,
            k=params["k"],
            temperature=params["temperature"],
            shrinkage=params["shrinkage"],
        )
        return test_bundle["y_base"] + params["alpha"] * correction
    raise ValueError(f"Unsupported benchmark RAG method: {method}")


def evaluate_method_leaders(
    validation_bundle,
    test_bundle,
    validation_search,
    task_family,
    metric_names,
    pred_len,
):
    """Evaluate one validation-selected leader per method for development."""

    candidates_by_method = {}
    for candidate in validation_search["candidates"]:
        method = candidate["method"]
        current = candidates_by_method.get(method)
        if current is None or (
            candidate["selection_score"],
            str(candidate["params"]),
        ) < (
            current["selection_score"],
            str(current["params"]),
        ):
            candidates_by_method[method] = candidate

    baseline_metrics = protocol_metrics(
        test_bundle, test_bundle["y_base"], task_family
    )
    diagnostics = {}
    for method, candidate in sorted(candidates_by_method.items()):
        prediction = apply_selected_rag(
            validation_bundle,
            test_bundle,
            candidate,
            task_family,
            pred_len,
        )
        metrics = protocol_metrics(test_bundle, prediction, task_family)
        diagnostics[method] = {
            "validation_leader": candidate,
            "test_metrics": metrics,
            "all_test_metrics_improve": all_metrics_improve(
                baseline_metrics, metrics, metric_names
            ),
            "metric_improvement_percent": {
                name: 100.0
                * (float(baseline_metrics[name]) - float(metrics[name]))
                / float(baseline_metrics[name])
                for name in metric_names
            },
        }
    return diagnostics


def run_validation_selected_rag(
    validation_bundle,
    test_bundle,
    task_family,
    metric_names,
    pred_len,
):
    search = search_validation_rag(
        validation_bundle,
        task_family,
        metric_names,
        pred_len,
    )
    corrected = apply_selected_rag(
        validation_bundle,
        test_bundle,
        search["selected"],
        task_family,
        pred_len,
    )
    baseline_metrics = protocol_metrics(
        test_bundle, test_bundle["y_base"], task_family
    )
    corrected_metrics = protocol_metrics(test_bundle, corrected, task_family)
    return {
        "validation": search,
        "test_baseline": baseline_metrics,
        "test_corrected": corrected_metrics,
        "all_test_metrics_improve": all_metrics_improve(
            baseline_metrics, corrected_metrics, metric_names
        ),
        "metric_improvement_percent": {
            name: 100.0
            * (float(baseline_metrics[name]) - float(corrected_metrics[name]))
            / float(baseline_metrics[name])
            for name in metric_names
        },
    }, corrected
