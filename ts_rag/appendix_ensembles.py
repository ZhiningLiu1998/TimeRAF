from pathlib import Path

import numpy as np

from ts_rag.appendix_rag import (
    ARRAY_KEYS,
    load_prediction_bundles,
    save_prediction_bundles,
    sha256_file,
)
from ts_rag.benchmark import MODELS


def _validation_loss(prediction, truth, task_family):
    error = np.asarray(prediction, dtype=np.float64) - np.asarray(
        truth,
        dtype=np.float64,
    )
    if task_family == "pems":
        return float(np.mean(np.abs(error)))
    return float(np.mean(np.square(error)))


def _normalized_weights(counts):
    counts = np.asarray(counts, dtype=np.float64)
    total = float(np.sum(counts))
    if total <= 0:
        raise ValueError("Ensemble must assign positive weight to a model")
    return counts / total


def forward_selection_weights(
    validation_predictions,
    validation_truth,
    task_family,
    ensemble_size=50,
):
    predictions = np.asarray(validation_predictions)
    if predictions.ndim != 4:
        raise ValueError("Base predictions must have model/sample/time/channel axes")
    if ensemble_size < 1:
        raise ValueError("ensemble_size must be positive")

    counts = np.zeros(len(predictions), dtype=np.int64)
    running_sum = np.zeros_like(predictions[0], dtype=np.float64)
    selection_trace = []
    for step in range(ensemble_size):
        losses = [
            _validation_loss(
                (running_sum + candidate) / (step + 1),
                validation_truth,
                task_family,
            )
            for candidate in predictions
        ]
        selected = min(range(len(losses)), key=lambda index: (losses[index], index))
        counts[selected] += 1
        running_sum += predictions[selected]
        selection_trace.append(
            {
                "step": step + 1,
                "model_index": selected,
                "validation_loss": losses[selected],
            }
        )
    weights = _normalized_weights(counts)
    return {
        "method": "caruana_forward_selection",
        "selection_uses": "validation_labels_only",
        "selection_loss": "mae" if task_family == "pems" else "mse",
        "ensemble_size": ensemble_size,
        "counts": counts.tolist(),
        "weights": weights.tolist(),
        "validation_loss": selection_trace[-1]["validation_loss"],
        "selection_trace": selection_trace,
    }


def portfolio_weights(
    validation_predictions,
    validation_truth,
    task_family,
):
    predictions = np.asarray(validation_predictions)
    if predictions.ndim != 4:
        raise ValueError("Base predictions must have model/sample/time/channel axes")

    individual_losses = [
        _validation_loss(prediction, validation_truth, task_family)
        for prediction in predictions
    ]
    ranking = sorted(
        range(len(predictions)),
        key=lambda index: (individual_losses[index], index),
    )
    candidates = []
    running_sum = np.zeros_like(predictions[0], dtype=np.float64)
    for subset_size, model_index in enumerate(ranking, start=1):
        running_sum += predictions[model_index]
        loss = _validation_loss(
            running_sum / subset_size,
            validation_truth,
            task_family,
        )
        candidates.append(
            {
                "subset_size": subset_size,
                "model_indices": ranking[:subset_size],
                "validation_loss": loss,
            }
        )
    selected = min(
        candidates,
        key=lambda row: (row["validation_loss"], row["subset_size"]),
    )
    counts = np.zeros(len(predictions), dtype=np.int64)
    counts[selected["model_indices"]] = 1
    weights = _normalized_weights(counts)
    return {
        "method": "validation_ranked_subset_mean",
        "selection_uses": "validation_labels_only",
        "selection_loss": "mae" if task_family == "pems" else "mse",
        "individual_validation_losses": individual_losses,
        "ranking": ranking,
        "selected_subset_size": selected["subset_size"],
        "selected_model_indices": selected["model_indices"],
        "weights": weights.tolist(),
        "validation_loss": selected["validation_loss"],
        "candidates": candidates,
    }


def apply_static_weights(predictions, weights):
    predictions = np.asarray(predictions)
    weights = np.asarray(weights, dtype=np.float64)
    if predictions.ndim != 4:
        raise ValueError("Base predictions must have model/sample/time/channel axes")
    if weights.shape != (len(predictions),):
        raise ValueError(
            f"Expected {len(predictions)} weights, found {weights.shape}"
        )
    if np.any(weights < 0) or not np.isclose(np.sum(weights), 1.0):
        raise ValueError("Static ensemble weights must be nonnegative and sum to one")

    output = np.zeros_like(predictions[0], dtype=np.float64)
    for weight, prediction in zip(weights, predictions):
        if weight:
            output += weight * prediction
    return output.astype(predictions.dtype, copy=False)


def _assert_aligned(reference, candidate, split, model):
    for key in ("x", "y", "x_mark", "y_mark"):
        if reference[key].shape != candidate[key].shape:
            raise ValueError(
                f"{split} {model} {key} shape {candidate[key].shape} "
                f"does not match {reference[key].shape}"
            )
        if not np.array_equal(reference[key], candidate[key]):
            raise ValueError(
                f"{split} {model} {key} is not aligned with the model library"
            )


def load_base_prediction_library(base_bundles, task_family):
    missing = [model for model in MODELS if model not in base_bundles]
    extra = sorted(set(base_bundles) - set(MODELS))
    if missing or extra:
        raise ValueError(
            f"Base prediction library mismatch: missing={missing}, extra={extra}"
        )

    splits = {"validation": None, "test": None}
    predictions = {"validation": [], "test": []}
    provenance = []
    for model in MODELS:
        path = Path(base_bundles[model])
        validation, test = load_prediction_bundles(path, task_family)
        for split, bundle in (("validation", validation), ("test", test)):
            reference = splits[split]
            if reference is None:
                splits[split] = bundle
            else:
                _assert_aligned(reference, bundle, split, model)
            predictions[split].append(bundle["y_base"])
        provenance.append(
            {
                "model": model,
                "path": str(path),
                "sha256": sha256_file(path),
            }
        )

    return {
        "models": list(MODELS),
        "validation": splits["validation"],
        "test": splits["test"],
        "validation_predictions": np.stack(predictions["validation"]),
        "test_predictions": np.stack(predictions["test"]),
        "base_prediction_bundles": provenance,
    }


def build_advanced_ensemble_bundle(
    base_bundles,
    task_family,
    method,
    output_path,
    forward_ensemble_size=50,
):
    library = load_base_prediction_library(base_bundles, task_family)
    if method == "forward_selection":
        selection = forward_selection_weights(
            library["validation_predictions"],
            library["validation"]["y"],
            task_family,
            ensemble_size=forward_ensemble_size,
        )
    elif method == "portfolio_ensemble":
        selection = portfolio_weights(
            library["validation_predictions"],
            library["validation"]["y"],
            task_family,
        )
    else:
        raise ValueError(f"Unsupported advanced ensemble method: {method}")

    selection["model_order"] = library["models"]
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
    bundle_sha256 = save_prediction_bundles(
        output_path,
        validation,
        test,
        task_family,
        already_reported=True,
    )
    return {
        "method": method,
        "task_family": task_family,
        "prediction_bundle": str(output_path),
        "prediction_bundle_sha256": bundle_sha256,
        "base_prediction_bundles": library["base_prediction_bundles"],
        "selection": selection,
    }
