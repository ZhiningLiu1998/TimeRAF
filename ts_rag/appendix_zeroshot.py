import gc
from pathlib import Path

import numpy as np
from scipy.stats import kurtosis, skew

from ts_rag.appendix_ensembles import (
    _validation_loss,
    apply_static_weights,
    load_base_prediction_library,
)
from ts_rag.appendix_rag import ARRAY_KEYS, save_prediction_bundles
from ts_rag.benchmark import DATASETS


META_FEATURE_NAMES = (
    "level_mean",
    "level_std",
    "level_min",
    "level_max",
    "level_skew",
    "level_kurtosis",
    "lag1_autocorrelation",
    "lag1_below_0_95_fraction",
    "relative_change_mean",
    "relative_change_std",
    "absolute_change_mean",
    "autoreg_coef_mean",
    "autoreg_residual_std",
    "spectral_power_mean",
    "dominant_frequency",
    "spectral_entropy",
    "spectral_variation",
    "spectral_skew",
    "spectral_kurtosis",
    "covariance_mean",
    "covariance_max",
    "covariance_min",
    "covariance_std",
)


def _sample_indices(length, maximum):
    if length <= maximum:
        return np.arange(length)
    return np.linspace(0, length - 1, maximum, dtype=np.int64)


def _mean_nonconstant_moment(values, function):
    varying = np.std(values, axis=1) > 1e-12
    if not np.any(varying):
        return 0.0
    return float(np.mean(function(values[varying], axis=1)))


def dataset_meta_feature_vector(x, max_samples=256, max_channels=64):
    x = np.asarray(x)
    if x.ndim != 3 or x.shape[1] < 2:
        raise ValueError("Meta-feature input must have sample/time/channel axes")
    sample_indices = _sample_indices(x.shape[0], max_samples)
    channel_indices = _sample_indices(x.shape[2], max_channels)
    sampled = np.asarray(
        x[np.ix_(sample_indices, np.arange(x.shape[1]), channel_indices)],
        dtype=np.float64,
    )
    series = sampled.transpose(0, 2, 1).reshape(-1, sampled.shape[1])
    finite = np.isfinite(series)
    if not finite.all():
        raise ValueError("Meta-feature input contains non-finite values")

    centered = series - np.mean(series, axis=1, keepdims=True)
    denominator = np.sum(np.square(centered[:, :-1]), axis=1)
    lag1 = np.divide(
        np.sum(centered[:, :-1] * centered[:, 1:], axis=1),
        denominator,
        out=np.zeros(len(series), dtype=np.float64),
        where=denominator > 0,
    )
    difference = np.diff(series, axis=1)
    relative_scale = np.maximum(
        np.mean(np.abs(series[:, :-1]), axis=1, keepdims=True),
        1e-6,
    )
    relative_change = difference / relative_scale
    autoreg_residual = centered[:, 1:] - lag1[:, None] * centered[:, :-1]

    spectrum = np.abs(np.fft.rfft(centered, axis=1)) ** 2
    spectrum_sum = np.sum(spectrum, axis=1, keepdims=True)
    spectrum_probability = np.divide(
        spectrum,
        spectrum_sum,
        out=np.zeros_like(spectrum),
        where=spectrum_sum > 0,
    )
    log_probability = np.zeros_like(spectrum_probability)
    np.log(
        spectrum_probability,
        out=log_probability,
        where=spectrum_probability > 0,
    )
    spectral_entropy = -np.sum(
        spectrum_probability * log_probability,
        axis=1,
    )
    frequency = np.fft.rfftfreq(series.shape[1])
    dominant_frequency = frequency[np.argmax(spectrum, axis=1)]
    spectral_grid = spectrum.reshape(
        sampled.shape[0],
        sampled.shape[2],
        spectrum.shape[1],
    )
    if sampled.shape[2] > 1:
        spectral_variation = np.mean(
            np.linalg.norm(np.diff(spectral_grid, axis=1), axis=2)
        )
    else:
        spectral_variation = 0.0

    flattened = sampled.reshape(-1, sampled.shape[2])
    covariance = np.atleast_2d(np.cov(flattened, rowvar=False))
    values = np.asarray(
        [
            np.mean(np.mean(series, axis=1)),
            np.mean(np.std(series, axis=1)),
            np.mean(np.min(series, axis=1)),
            np.mean(np.max(series, axis=1)),
            _mean_nonconstant_moment(series, skew),
            _mean_nonconstant_moment(series, kurtosis),
            np.mean(lag1),
            np.mean(np.abs(lag1) < 0.95),
            np.mean(relative_change),
            np.std(relative_change),
            np.mean(np.abs(difference)),
            np.mean(lag1),
            np.mean(np.std(autoreg_residual, axis=1)),
            np.mean(spectrum),
            np.mean(dominant_frequency),
            np.mean(spectral_entropy),
            spectral_variation,
            _mean_nonconstant_moment(spectrum, skew),
            _mean_nonconstant_moment(spectrum, kurtosis),
            np.mean(covariance),
            np.max(covariance),
            np.min(covariance),
            np.std(covariance),
        ],
        dtype=np.float64,
    )
    values = np.nan_to_num(values, nan=0.0, posinf=1e12, neginf=-1e12)
    return {
        "feature_names": list(META_FEATURE_NAMES),
        "values": values,
        "sampling": {
            "available_samples": int(x.shape[0]),
            "available_channels": int(x.shape[2]),
            "selected_samples": int(len(sample_indices)),
            "selected_channels": int(len(channel_indices)),
            "max_samples": max_samples,
            "max_channels": max_channels,
            "sample_index_policy": "deterministic_even_spacing",
            "channel_index_policy": "deterministic_even_spacing",
        },
    }


def _reciprocal_rank_scores(predictions, truth, task_family):
    losses = [
        _validation_loss(prediction, truth, task_family)
        for prediction in predictions
    ]
    ranking = sorted(range(len(losses)), key=lambda index: (losses[index], index))
    ranks = np.empty(len(losses), dtype=np.int64)
    for rank, model_index in enumerate(ranking, start=1):
        ranks[model_index] = rank
    scores = 1.0 / ranks.astype(np.float64)
    scores /= np.sum(scores)
    return {
        "validation_losses": losses,
        "ranking": ranking,
        "ranks": ranks.tolist(),
        "reciprocal_rank_scores": scores.tolist(),
    }


def build_task_profile(dataset, library, task_family):
    meta = dataset_meta_feature_vector(library["validation"]["x"])
    performance = _reciprocal_rank_scores(
        library["validation_predictions"],
        library["validation"]["y"],
        task_family,
    )
    return {
        "dataset": dataset,
        "task_family": task_family,
        "model_order": library["models"],
        "meta_feature_names": meta["feature_names"],
        "meta_feature_values": meta["values"].tolist(),
        "meta_feature_sampling": meta["sampling"],
        "source_validation_performance": performance,
        "base_prediction_bundles": library["base_prediction_bundles"],
    }


def zeroshot_weights(target_dataset, profiles):
    if target_dataset not in profiles:
        raise ValueError(f"Target profile is missing: {target_dataset}")
    target = profiles[target_dataset]
    source_names = sorted(name for name in profiles if name != target_dataset)
    if len(source_names) < 2:
        raise ValueError("ZeroShot weighting requires at least two source tasks")
    sources = [profiles[name] for name in source_names]
    if any(
        source["model_order"] != target["model_order"]
        or source["meta_feature_names"] != target["meta_feature_names"]
        for source in sources
    ):
        raise ValueError("ZeroShot task profiles do not share one feature/model schema")

    source_features = np.asarray(
        [source["meta_feature_values"] for source in sources],
        dtype=np.float64,
    )
    target_features = np.asarray(target["meta_feature_values"], dtype=np.float64)
    center = np.median(source_features, axis=0)
    scale = np.median(np.abs(source_features - center), axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    standardized_sources = (source_features - center) / scale
    standardized_target = (target_features - center) / scale
    distances = np.linalg.norm(
        standardized_sources - standardized_target,
        axis=1,
    )

    pairwise = []
    for left in range(len(standardized_sources)):
        for right in range(left + 1, len(standardized_sources)):
            distance = np.linalg.norm(
                standardized_sources[left] - standardized_sources[right]
            )
            if distance > 0:
                pairwise.append(float(distance))
    bandwidth = float(np.median(pairwise)) if pairwise else 1.0
    bandwidth = max(bandwidth, 1e-12)
    similarities = np.exp(-0.5 * np.square(distances / bandwidth))
    if not np.any(similarities > 0):
        similarities = np.ones_like(similarities)
    similarities /= np.sum(similarities)

    source_model_scores = np.asarray(
        [
            source["source_validation_performance"][
                "reciprocal_rank_scores"
            ]
            for source in sources
        ],
        dtype=np.float64,
    )
    weights = similarities @ source_model_scores
    weights /= np.sum(weights)
    return {
        "method": "leave_one_dataset_out_meta_similarity",
        "selection_uses": (
            "target_validation_inputs_and_source_validation_labels_only"
        ),
        "target_dataset": target_dataset,
        "source_datasets": source_names,
        "model_order": target["model_order"],
        "robust_center": center.tolist(),
        "robust_scale": scale.tolist(),
        "rbf_bandwidth": bandwidth,
        "source_distances": distances.tolist(),
        "source_similarities": similarities.tolist(),
        "weights": weights.tolist(),
    }


def _expected_family_datasets(task_family):
    return sorted(dataset.name for dataset in DATASETS if dataset.family == task_family)


def build_zeroshot_ensemble_bundles(
    task_bundles,
    task_family,
    output_root,
):
    expected = _expected_family_datasets(task_family)
    if sorted(task_bundles) != expected:
        raise ValueError(
            f"ZeroShot {task_family} tasks must be {expected}, "
            f"found {sorted(task_bundles)}"
        )

    profiles = {}
    for dataset in expected:
        library = load_base_prediction_library(
            task_bundles[dataset],
            task_family,
        )
        profiles[dataset] = build_task_profile(
            dataset,
            library,
            task_family,
        )
        del library
        gc.collect()

    outputs = {}
    for dataset in expected:
        selection = zeroshot_weights(dataset, profiles)
        library = load_base_prediction_library(
            task_bundles[dataset],
            task_family,
        )
        validation = {
            key: library["validation"][key]
            for key in ARRAY_KEYS
            if key != "y_base"
        }
        test = {
            key: library["test"][key]
            for key in ARRAY_KEYS
            if key != "y_base"
        }
        validation["y_base"] = apply_static_weights(
            library["validation_predictions"],
            selection["weights"],
        )
        test["y_base"] = apply_static_weights(
            library["test_predictions"],
            selection["weights"],
        )
        output_path = Path(output_root) / dataset / "prediction_bundle.npz"
        digest = save_prediction_bundles(
            output_path,
            validation,
            test,
            task_family,
            already_reported=True,
        )
        outputs[dataset] = {
            "prediction_bundle": str(output_path),
            "prediction_bundle_sha256": digest,
            "target_base_prediction_bundles": library[
                "base_prediction_bundles"
            ],
            "selection": selection,
        }
        del library
        gc.collect()

    return {
        "task_family": task_family,
        "protocol": {
            "task_holdout": "leave_one_dataset_out",
            "target_label_use": "none",
            "target_input_split": "validation",
            "source_label_split": "validation",
            "task_similarity": "robust_scaled_rbf",
            "source_model_score": "normalized_reciprocal_validation_rank",
        },
        "profiles": profiles,
        "outputs": outputs,
    }
