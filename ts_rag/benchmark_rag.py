import itertools
import json
from dataclasses import asdict, dataclass, field

import numpy as np

from ts_rag.benchmark import all_metrics_improve
from ts_rag.evaluation import compute_protocol_metrics
from ts_rag.historical_retrieval import (
    HistoricalResidualIndex,
    RetrievalFeatureConfig,
)
from ts_rag.online_ensemble import overlapping_forecast_ensemble


@dataclass(frozen=True)
class PortfolioConfig:
    """Portfolio grid selector.

    The default reproduces the frozen ``d9be338`` candidate set exactly. The v2
    options add the unified retrieval family, in which one dial ``beta`` moves
    the retrieved correction from the memory residual alone towards the
    endpoint-aligned analogue future.
    """

    enable_drift: bool = False
    drift_betas: tuple = (0.25, 0.5, 1.0)
    drift_alphas: tuple = (0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0)
    drift_ramps: tuple = ("flat", "linear")
    drift_aggregations: tuple = ((8, 0.1, 0.0), (16, 0.3, 4.0))
    shape_descriptors: bool = False
    conservative_tolerance: float = 0.0
    fold_selection: str = "none"

    def to_dict(self):
        return asdict(self)


V1_PORTFOLIO = PortfolioConfig()

PURE_ANALOGUE_PARAMS = {
    "alpha": 1.0,
    "beta": 1.0,
    "ramp": "flat",
    "shrinkage": 0.0,
}


@dataclass(frozen=True)
class SelectionPolicy:
    """How one validation candidate is chosen from a scored pool.

    ``fold_selection`` decides which validation view scores a candidate.
    ``family_prior_margin`` and ``horizon_conditional_prior`` express a prior in
    favour of pure error retrieval: the drift family has to beat the best static
    candidate by a margin before the base trajectory is overwritten.
    """

    fold_selection: str = "none"
    conservative_tolerance: float = 0.0
    family_prior_margin: float = 0.0
    family_prior_relative_margin: float = 0.0
    horizon_conditional_prior: tuple = ()
    analogue_gate: bool = False
    rank_stability_fraction: float = 0.0

    def to_dict(self):
        return asdict(self)

    def margin_for(self, pred_len):
        if not self.horizon_conditional_prior:
            return self.family_prior_margin
        threshold, at_or_below, above = self.horizon_conditional_prior
        if pred_len is None:
            raise ValueError("A horizon conditional prior needs the cell horizon")
        return float(at_or_below if pred_len <= threshold else above)

    def select(self, rows, pred_len=None):
        pool = list(rows)
        if self.analogue_gate:
            admitted = {
                json.dumps(row["params"]["feature_config"], sort_keys=True)
                for row in pool
                if row["method"] == "residual_drift"
                and all(
                    row["params"].get(key) == value
                    for key, value in PURE_ANALOGUE_PARAMS.items()
                )
                and _score_of(row, self.fold_selection) < 1.0
            }
            pool = [
                row
                for row in pool
                if row["method"] != "residual_drift"
                or json.dumps(row["params"]["feature_config"], sort_keys=True)
                in admitted
            ]

        margin = self.margin_for(pred_len)
        if margin > 0.0 or self.family_prior_relative_margin > 0.0:
            static = [row for row in pool if row["method"] != "residual_drift"]
            if static:
                best_static = min(
                    _score_of(row, self.fold_selection) for row in static
                )
                budget = best_static
                if margin > 0.0:
                    budget = min(budget, best_static - margin)
                if self.family_prior_relative_margin > 0.0:
                    budget = min(
                        budget,
                        best_static * (1.0 - self.family_prior_relative_margin),
                    )
                pool = [
                    row
                    for row in pool
                    if row["method"] != "residual_drift"
                    or _score_of(row, self.fold_selection) <= budget
                ]

        if self.rank_stability_fraction > 0.0:
            keep = None
            for fold_index in (0, 1):
                ordered = sorted(
                    pool,
                    key=lambda row: float(row["fold_selection_scores"][fold_index]),
                )
                cutoff = max(1, int(round(self.rank_stability_fraction * len(ordered))))
                names = {
                    json.dumps([row["method"], row["params"]], sort_keys=True)
                    for row in ordered[:cutoff]
                }
                keep = names if keep is None else (keep & names)
            stable = [
                row
                for row in pool
                if json.dumps([row["method"], row["params"]], sort_keys=True) in keep
            ]
            identity = [row for row in pool if row["method"] == "identity"]
            pool = stable or identity or pool

        if not pool:
            raise ValueError("Selection policy admitted no candidate")
        return _choose_best(pool, self.conservative_tolerance, self.fold_selection)


def selection_policy_from_dict(payload):
    payload = dict(payload)
    prior = payload.pop("horizon_conditional_prior", None)
    if prior:
        payload["horizon_conditional_prior"] = (
            int(prior["threshold"]),
            float(prior["margin_at_or_below"]),
            float(prior["margin_above"]),
        )
    payload.pop("id", None)
    payload.pop("portfolio", None)
    return SelectionPolicy(**payload)


def _drift_ramp(name, pred_len):
    if name == "flat":
        return np.ones((1, pred_len, 1), dtype=np.float32)
    if name == "linear":
        steps = np.arange(1, pred_len + 1, dtype=np.float32) / float(pred_len)
        return steps.reshape(1, pred_len, 1)
    raise ValueError(f"Unsupported drift ramp: {name}")


def _drift_correction(index, neighbors, scores, k, temperature, shrinkage, query_bundle):
    """Residual plus forecast-drift correction fields for one neighbourhood.

    ``residual`` is the current TimeRAF correction. ``drift`` completes it into
    the endpoint-aligned analogue transfer:

        aligned_future_i - base_forecast_q
            = (y_i - base_i) + (base_i - x_i_last + x_q_last - base_q)
            = residual_i + drift_i
    """

    residual = index.aggregate(
        neighbors,
        scores,
        k=k,
        temperature=temperature,
        shrinkage=shrinkage,
    )
    drift = index.aggregate_offset_forecast(
        neighbors,
        scores,
        k=k,
        temperature=temperature,
    )
    query_last = np.asarray(query_bundle["x"][:, -1:, :], dtype=np.float32)
    drift = drift + query_last - np.asarray(query_bundle["y_base"], dtype=np.float32)
    return residual, drift


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


def _feature_configs(bundle, task_family, shape_descriptors=False):
    tail_lengths = sorted({min(48, bundle["x"].shape[1]), bundle["x"].shape[1]})
    level_options = (True, False) if shape_descriptors else (True,)
    for tail_len, include_forecast, include_level in itertools.product(
        tail_lengths, (False, True), level_options
    ):
        yield RetrievalFeatureConfig(
            tail_len=tail_len,
            pooled_steps=min(12, tail_len),
            include_differences=True,
            include_forecast=include_forecast,
            include_calendar=task_family != "pems",
            max_channels=16,
            include_level_features=include_level,
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


class _CandidateScorer:
    """Score validation candidates on the whole query window, and on two folds.

    The whole-window score reproduces the frozen ``validation_argmin_v3``
    selection exactly. The optional fold score splits the validation query
    window into an early and a late half and reports the worse half, so a
    candidate that only helps in part of the validation window cannot be
    selected. That controls selection variance as the candidate pool grows.
    """

    def __init__(self, query_bundle, task_family, metric_names, compute_folds=False):
        self.query_bundle = query_bundle
        self.task_family = task_family
        self.metric_names = metric_names
        self.baseline = protocol_metrics(
            query_bundle, query_bundle["y_base"], task_family
        )
        self.folds = None
        if not compute_folds:
            return
        count = len(query_bundle["y"])
        if count < 4:
            return
        split = count // 2
        self.folds = [
            (
                indices,
                _subset_bundle(query_bundle, indices),
            )
            for indices in (np.arange(split), np.arange(split, count))
        ]
        self.fold_baselines = [
            protocol_metrics(fold_bundle, fold_bundle["y_base"], task_family)
            for _, fold_bundle in self.folds
        ]

    def row(self, name, params, prediction=None):
        metrics = (
            self.baseline
            if prediction is None
            else protocol_metrics(self.query_bundle, prediction, self.task_family)
        )
        row = _candidate_row(
            name, params, metrics, self.baseline, self.metric_names
        )
        if self.folds is None:
            row["fold_selection_score"] = row["selection_score"]
            row["fold_selection_scores"] = None
            return row

        fold_scores = []
        improved = True
        for (indices, fold_bundle), fold_baseline in zip(
            self.folds, self.fold_baselines
        ):
            observed = (
                fold_baseline
                if prediction is None
                else protocol_metrics(
                    fold_bundle, prediction[indices], self.task_family
                )
            )
            fold_scores.append(
                _selection_score(fold_baseline, observed, self.metric_names)
            )
            improved = improved and all_metrics_improve(
                fold_baseline, observed, self.metric_names
            )
        row["fold_selection_scores"] = [float(value) for value in fold_scores]
        row["fold_selection_score"] = float(max(fold_scores))
        row["all_fold_metrics_improve"] = bool(improved)
        return row


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
    scorer,
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
                scorer.row(
                    "overlap_blend",
                    {
                        "max_age": max_age,
                        "decay": decay,
                        "alpha": alpha,
                    },
                    prediction=prediction,
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
                    scorer.row(
                        "causal_bias",
                        {
                            "window": window,
                            "horizon_block": horizon_block,
                            "alpha": alpha,
                        },
                        prediction=prediction,
                    )
                )
    return rows


def _score_of(row, fold_selection):
    if fold_selection == "none":
        return row["selection_score"]
    if fold_selection != "minimax2":
        raise ValueError(f"Unsupported fold selection: {fold_selection}")
    if "fold_selection_score" not in row:
        raise ValueError("Candidate rows were scored without validation folds")
    return row["fold_selection_score"]


def _selection_key(row, fold_selection="none"):
    return (
        _score_of(row, fold_selection),
        0 if row["method"] == "identity" else 1,
        row["method"],
        json.dumps(
            row["params"],
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ),
    )


def _correction_magnitude(row):
    params = row["params"]
    alpha = float(params.get("alpha", 0.0))
    beta = float(params.get("beta", 0.0))
    return alpha * (1.0 + beta)


def _choose_best(rows, conservative_tolerance=0.0, fold_selection="none"):
    def key(row):
        return _selection_key(row, fold_selection)

    leader = min(rows, key=key)
    if conservative_tolerance <= 0.0:
        return leader
    leader_score = _score_of(leader, fold_selection)
    achieved = 1.0 - leader_score
    if achieved <= 0.0:
        return leader
    # One-standard-error style rule: give up at most a fixed fraction of the
    # achieved validation improvement in exchange for the least aggressive
    # correction that still reaches it.
    budget = leader_score + conservative_tolerance * achieved
    admissible = [
        row for row in rows if _score_of(row, fold_selection) <= budget
    ]
    return min(
        admissible,
        key=lambda row: (_correction_magnitude(row),) + key(row),
    )


def search_validation_rag(
    bundle,
    task_family,
    metric_names,
    pred_len,
    portfolio=V1_PORTFOLIO,
):
    memory_indices, query_indices = _validation_partition(len(bundle["y"]), pred_len)
    query_bundle = _subset_bundle(bundle, query_indices)
    baseline_prediction = query_bundle["y_base"]
    scorer = _CandidateScorer(
        query_bundle,
        task_family,
        metric_names,
        compute_folds=portfolio.fold_selection != "none",
    )
    baseline_metrics = scorer.baseline
    rows = [scorer.row("identity", {})]

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
                rows.append(
                    scorer.row(
                        "historical_residual",
                        {
                            "feature_config": config.to_dict(),
                            "k": k,
                            "temperature": temperature,
                            "shrinkage": shrinkage,
                            "alpha": alpha,
                        },
                        prediction=prediction,
                    )
                )

    if portfolio.enable_drift:
        ramps = {name: _drift_ramp(name, pred_len) for name in portfolio.drift_ramps}
        for config in _feature_configs(
            bundle, task_family, shape_descriptors=portfolio.shape_descriptors
        ):
            index = HistoricalResidualIndex(config).fit(bundle, memory_indices)
            neighbors, scores = index.query_neighbors(
                bundle, query_indices, max_k=16
            )
            for k, temperature, shrinkage in portfolio.drift_aggregations:
                residual, drift = _drift_correction(
                    index,
                    neighbors,
                    scores,
                    k,
                    temperature,
                    shrinkage,
                    query_bundle,
                )
                for beta, ramp_name in itertools.product(
                    portfolio.drift_betas, portfolio.drift_ramps
                ):
                    correction = residual + beta * ramps[ramp_name] * drift
                    for alpha in portfolio.drift_alphas:
                        prediction = baseline_prediction + alpha * correction
                        rows.append(
                            scorer.row(
                                "residual_drift",
                                {
                                    "feature_config": config.to_dict(),
                                    "k": k,
                                    "temperature": temperature,
                                    "shrinkage": shrinkage,
                                    "alpha": alpha,
                                    "beta": beta,
                                    "ramp": ramp_name,
                                },
                                prediction=prediction,
                            )
                        )

    for period in _seasonal_periods(query_bundle, task_family):
        seasonal = _seasonal_prediction(query_bundle, period)
        for alpha in (0.02, 0.05, 0.1, 0.2, 0.3):
            prediction = (1.0 - alpha) * baseline_prediction + alpha * seasonal
            rows.append(
                scorer.row(
                    "seasonal_blend",
                    {"period": period, "alpha": alpha},
                    prediction=prediction,
                )
            )

    residual_bias = np.mean(
        bundle["y"][memory_indices] - bundle["y_base"][memory_indices],
        axis=(0, 1),
        keepdims=True,
    )
    for alpha in (0.1, 0.25, 0.5, 1.0):
        prediction = baseline_prediction + alpha * residual_bias
        rows.append(
            scorer.row(
                "residual_bias",
                {"alpha": alpha},
                prediction=prediction,
            )
        )

    rows.extend(
        _online_validation_candidates(
            bundle,
            query_indices,
            scorer,
            task_family,
            metric_names,
            pred_len,
        )
    )

    selection_policy = "validation_argmin_v3"
    if portfolio.fold_selection != "none":
        selection_policy += f"_{portfolio.fold_selection}"
    if portfolio.conservative_tolerance > 0.0:
        selection_policy += f"_conservative_{portfolio.conservative_tolerance:g}"
    return {
        "memory_count": int(len(memory_indices)),
        "query_count": int(len(query_indices)),
        "baseline": baseline_metrics,
        "selection_policy": selection_policy,
        "portfolio": portfolio.to_dict(),
        "selected": _choose_best(
            rows,
            portfolio.conservative_tolerance,
            portfolio.fold_selection,
        ),
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
    if method == "residual_drift":
        config = RetrievalFeatureConfig(**params["feature_config"])
        index = HistoricalResidualIndex(config).fit(validation_bundle)
        neighbors, scores = index.query_neighbors(
            test_bundle, max_k=params["k"]
        )
        residual, drift = _drift_correction(
            index,
            neighbors,
            scores,
            params["k"],
            params["temperature"],
            params["shrinkage"],
            test_bundle,
        )
        ramp = _drift_ramp(params["ramp"], int(test_bundle["y_base"].shape[1]))
        correction = residual + params["beta"] * ramp * drift
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
    portfolio=V1_PORTFOLIO,
):
    search = search_validation_rag(
        validation_bundle,
        task_family,
        metric_names,
        pred_len,
        portfolio=portfolio,
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
