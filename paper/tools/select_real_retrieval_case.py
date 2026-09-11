#!/usr/bin/env python3
"""Select a reproducible real-data retrieval case for the paper figures."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import pickle
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT = (
    PROJECT_ROOT
    / "ts_rag_outputs"
    / "baselines"
    / (
        "long_term_forecast_ts_rag_TimeMixer_ETTh1_ftM_sl96_ll0_pl96_"
        "dm16_nh8_el2_dl1_df32_expand2_dc4_fc1_ebtimeF_dtTrue_ts_rag_0"
    )
)
DEFAULT_OUTPUT = PROJECT_ROOT / "paper" / "data" / "real_retrieval_case.json"
CHANNEL_NAMES = ("HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_retrieval_module() -> ModuleType:
    path = PROJECT_ROOT / "ts_rag" / "historical_retrieval.py"
    spec = importlib.util.spec_from_file_location(
        "timeraf_historical_retrieval",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load retrieval implementation from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_pickle_bundle(path: Path) -> dict[str, Any]:
    with path.open("rb") as source:
        bundle = pickle.load(source)
    required = ("x", "y", "y_base", "x_mark", "y_mark")
    missing = [key for key in required if key not in bundle]
    if missing:
        raise KeyError(f"{path} is missing bundle keys: {', '.join(missing)}")
    arrays = {key: np.asarray(bundle[key]) for key in required}
    sample_count = len(arrays["y"])
    if arrays["y"].shape != arrays["y_base"].shape:
        raise ValueError("Truth and base prediction shapes differ")
    if arrays["x"].ndim != 3 or arrays["y"].ndim != 3:
        raise ValueError("Expected rank-three context and forecast arrays")
    if any(len(value) != sample_count for value in arrays.values()):
        raise ValueError("Prediction bundle arrays have inconsistent sample counts")
    if any(not np.isfinite(value).all() for value in arrays.values()):
        raise ValueError(f"{path} contains non-finite arrays")
    arrays["future_residual"] = arrays["y"] - arrays["y_base"]
    return arrays


def percentile_rank(values: np.ndarray) -> np.ndarray:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    order = np.argsort(flat, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.linspace(0.0, 1.0, len(flat), endpoint=True)
    return ranks.reshape(values.shape)


def normalize_shape(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    mean = values.mean(axis=-1, keepdims=True)
    scale = values.std(axis=-1, keepdims=True)
    return (values - mean) / (scale + 1e-6)


def json_series(values: np.ndarray) -> list[float]:
    return [round(float(value), 8) for value in np.asarray(values)]


def candidate_record(
    sample_index: int,
    channel_index: int,
    score: np.ndarray,
    metrics: dict[str, np.ndarray],
) -> dict[str, Any]:
    return {
        "sample_index": int(sample_index),
        "channel_index": int(channel_index),
        "channel_name": CHANNEL_NAMES[channel_index],
        "score": round(float(score[sample_index, channel_index]), 8),
        "base_mse": round(
            float(metrics["base_mse"][sample_index, channel_index]), 8
        ),
        "corrected_mse": round(
            float(metrics["corrected_mse"][sample_index, channel_index]), 8
        ),
        "relative_improvement": round(
            float(metrics["relative_improvement"][sample_index, channel_index]),
            8,
        ),
        "context_similarity": round(
            float(metrics["context_similarity"][sample_index, channel_index]), 8
        ),
        "future_spread": round(
            float(metrics["future_spread"][sample_index, channel_index]), 8
        ),
        "residual_agreement": round(
            float(metrics["residual_agreement"][sample_index, channel_index]), 8
        ),
        "correction_alignment": round(
            float(metrics["correction_alignment"][sample_index, channel_index]),
            8,
        ),
    }


def select_case(
    artifact_dir: Path,
    output_path: Path,
    top_neighbors: int,
) -> dict[str, Any]:
    validation_path = artifact_dir / "val_predictions.pkl"
    test_path = artifact_dir / "test_predictions.pkl"
    result_path = artifact_dir / "historical_rag_predictions.npz"
    selection_path = artifact_dir / "historical_rag.json"
    for path in (validation_path, test_path, result_path, selection_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    validation = load_pickle_bundle(validation_path)
    test = load_pickle_bundle(test_path)
    with selection_path.open("r", encoding="utf-8") as source:
        selection_report = json.load(source)
    selected = selection_report["selected"]
    with np.load(result_path, allow_pickle=False) as archive:
        saved_corrected = np.asarray(archive["corrected"])
        saved_true = np.asarray(archive["true"])
        saved_baseline = np.asarray(archive["baseline"])

    if not np.array_equal(saved_true, test["y"]):
        raise ValueError("Saved truth does not match the test prediction bundle")
    if not np.array_equal(saved_baseline, test["y_base"]):
        raise ValueError("Saved baseline does not match the test prediction bundle")

    retrieval = load_retrieval_module()
    feature_config = retrieval.RetrievalFeatureConfig(**selected["feature_config"])
    index = retrieval.HistoricalResidualIndex(feature_config).fit(validation)
    neighbor_positions, neighbor_scores = index.query_neighbors(
        test,
        max_k=int(selected["k"]),
    )
    correction = index.aggregate(
        neighbor_positions,
        neighbor_scores,
        k=int(selected["k"]),
        temperature=float(selected["temperature"]),
        shrinkage=float(selected["shrinkage"]),
    )
    reconstructed = test["y_base"] + float(selected["alpha"]) * correction
    max_reconstruction_error = float(
        np.max(np.abs(reconstructed - saved_corrected))
    )
    if max_reconstruction_error > 1e-5:
        raise ValueError(
            "Recomputed retrieval predictions differ from the saved artifact: "
            f"max_abs_error={max_reconstruction_error}"
        )

    neighbor_indices = index.memory_indices[neighbor_positions]
    diagnostic_k = min(8, neighbor_indices.shape[1])
    diagnostic_neighbors = neighbor_indices[:, :diagnostic_k]
    tail_len = min(feature_config.tail_len, test["x"].shape[1])

    query_context = np.moveaxis(test["x"][:, -tail_len:, :], 1, 2)
    memory_context = np.moveaxis(
        validation["x"][diagnostic_neighbors, -tail_len:, :],
        2,
        3,
    )
    normalized_query = normalize_shape(query_context)
    normalized_memory = normalize_shape(memory_context)
    context_rmse = np.sqrt(
        np.mean(
            np.square(normalized_memory - normalized_query[:, None, :, :]),
            axis=(1, 3),
        )
    )
    context_similarity = np.exp(-context_rmse)

    neighbor_truth = np.moveaxis(
        validation["y"][diagnostic_neighbors],
        2,
        3,
    )
    query_scale = np.std(test["y"], axis=1) + 1e-6
    future_spread = (
        np.sqrt(np.mean(np.var(neighbor_truth, axis=1), axis=2)) / query_scale
    )

    neighbor_residual = np.moveaxis(
        validation["future_residual"][diagnostic_neighbors],
        2,
        3,
    )
    residual_mean = neighbor_residual.mean(axis=1)
    residual_mean_abs = np.abs(neighbor_residual).mean(axis=1)
    residual_agreement = np.mean(
        np.abs(residual_mean) / (residual_mean_abs + 1e-6),
        axis=2,
    )

    base_error = test["y"] - test["y_base"]
    applied_correction = saved_corrected - test["y_base"]
    base_mse = np.mean(np.square(base_error), axis=1)
    corrected_mse = np.mean(
        np.square(test["y"] - saved_corrected),
        axis=1,
    )
    relative_improvement = (base_mse - corrected_mse) / (base_mse + 1e-8)
    correction_alignment = np.sum(base_error * applied_correction, axis=1) / (
        np.sqrt(np.sum(np.square(base_error), axis=1))
        * np.sqrt(np.sum(np.square(applied_correction), axis=1))
        + 1e-8
    )
    correction_energy = np.mean(np.square(applied_correction), axis=1)
    absolute_gain = base_mse - corrected_mse

    metrics = {
        "base_mse": base_mse,
        "corrected_mse": corrected_mse,
        "relative_improvement": relative_improvement,
        "context_similarity": context_similarity,
        "future_spread": future_spread,
        "residual_agreement": residual_agreement,
        "correction_alignment": correction_alignment,
    }
    score = (
        0.30 * percentile_rank(relative_improvement)
        + 0.18 * percentile_rank(absolute_gain)
        + 0.17 * percentile_rank(context_similarity)
        + 0.13 * percentile_rank(residual_agreement)
        + 0.12 * percentile_rank(future_spread)
        + 0.10 * percentile_rank(correction_energy)
    )
    eligible = (
        (relative_improvement >= 0.15)
        & (correction_alignment >= 0.45)
        & (context_similarity >= np.quantile(context_similarity, 0.50))
        & (residual_agreement >= np.quantile(residual_agreement, 0.50))
    )
    score = np.where(eligible, score, -np.inf)
    if not np.isfinite(score).any():
        raise ValueError("No real-data case satisfies the visualization criteria")

    flat_order = np.argsort(score.reshape(-1), kind="mergesort")[::-1]
    finite_order = [
        int(index_value)
        for index_value in flat_order
        if np.isfinite(score.reshape(-1)[index_value])
    ]
    sample_count, channel_count = score.shape
    top_records = []
    for flat_index in finite_order[:10]:
        sample_index, channel_index = np.unravel_index(
            flat_index,
            (sample_count, channel_count),
        )
        top_records.append(
            candidate_record(sample_index, channel_index, score, metrics)
        )

    selected_sample = top_records[0]["sample_index"]
    selected_channel = top_records[0]["channel_index"]
    plotted_k = min(top_neighbors, neighbor_indices.shape[1])
    plotted_neighbors = neighbor_indices[selected_sample, :plotted_k]
    plotted_scores = neighbor_scores[selected_sample, :plotted_k]

    horizon = test["y"].shape[1]
    display_length = min(36, horizon)
    segment_starts = np.arange(horizon - display_length + 1)
    segment_metrics = {
        "relative_improvement": [],
        "correction_alignment": [],
        "future_spread": [],
        "residual_agreement": [],
    }
    query_truth_full = test["y"][selected_sample, :, selected_channel]
    query_base_full = test["y_base"][selected_sample, :, selected_channel]
    query_corrected_full = saved_corrected[
        selected_sample, :, selected_channel
    ]
    neighbor_futures_full = validation["y"][
        plotted_neighbors, :, selected_channel
    ]
    neighbor_residuals_full = validation["future_residual"][
        plotted_neighbors, :, selected_channel
    ]
    for start in segment_starts:
        end = start + display_length
        truth_segment = query_truth_full[start:end]
        base_segment = query_base_full[start:end]
        corrected_segment = query_corrected_full[start:end]
        error_segment = truth_segment - base_segment
        correction_segment = corrected_segment - base_segment
        base_segment_mse = np.mean(np.square(error_segment))
        corrected_segment_mse = np.mean(
            np.square(truth_segment - corrected_segment)
        )
        segment_metrics["relative_improvement"].append(
            (base_segment_mse - corrected_segment_mse)
            / (base_segment_mse + 1e-8)
        )
        segment_metrics["correction_alignment"].append(
            np.dot(error_segment, correction_segment)
            / (
                np.linalg.norm(error_segment)
                * np.linalg.norm(correction_segment)
                + 1e-8
            )
        )
        local_futures = neighbor_futures_full[:, start:end]
        segment_metrics["future_spread"].append(
            np.sqrt(np.mean(np.var(local_futures, axis=0)))
            / (np.std(truth_segment) + 1e-6)
        )
        local_residuals = neighbor_residuals_full[:, start:end]
        segment_metrics["residual_agreement"].append(
            np.mean(
                np.abs(local_residuals.mean(axis=0))
                / (np.abs(local_residuals).mean(axis=0) + 1e-6)
            )
        )
    segment_metrics = {
        key: np.asarray(values, dtype=np.float64)
        for key, values in segment_metrics.items()
    }
    segment_score = (
        0.40 * percentile_rank(segment_metrics["relative_improvement"])
        + 0.25 * percentile_rank(segment_metrics["correction_alignment"])
        + 0.20 * percentile_rank(segment_metrics["residual_agreement"])
        + 0.15 * percentile_rank(segment_metrics["future_spread"])
    )
    segment_eligible = (
        (segment_metrics["relative_improvement"] > 0.05)
        & (segment_metrics["correction_alignment"] > 0.40)
    )
    segment_score = np.where(segment_eligible, segment_score, -np.inf)
    if not np.isfinite(segment_score).any():
        raise ValueError("Selected case has no eligible display segment")
    display_start = int(np.argmax(segment_score))
    display_end = display_start + display_length

    query_context_values = test["x"][
        selected_sample, -tail_len:, selected_channel
    ]
    memory_context_values = validation["x"][
        plotted_neighbors, -tail_len:, selected_channel
    ]
    query_context_normalized = normalize_shape(query_context_values)
    memory_context_normalized = normalize_shape(memory_context_values)

    payload = {
        "schema_version": 1,
        "description": (
            "Real ETTh1 test case selected deterministically for the TimeRaf "
            "motivation and residual-retrieval figures."
        ),
        "source": {
            "dataset": "ETTh1",
            "model": "TimeMixer",
            "split": "test",
            "artifact_dir": str(artifact_dir.relative_to(PROJECT_ROOT)),
            "validation_bundle": str(validation_path.relative_to(PROJECT_ROOT)),
            "validation_bundle_sha256": sha256_file(validation_path),
            "test_bundle": str(test_path.relative_to(PROJECT_ROOT)),
            "test_bundle_sha256": sha256_file(test_path),
            "prediction_artifact": str(result_path.relative_to(PROJECT_ROOT)),
            "prediction_artifact_sha256": sha256_file(result_path),
            "selection_report": str(selection_path.relative_to(PROJECT_ROOT)),
            "selection_report_sha256": sha256_file(selection_path),
            "sample_index": selected_sample,
            "channel_index": selected_channel,
            "channel_name": CHANNEL_NAMES[selected_channel],
            "context_length": tail_len,
            "forecast_horizon": int(horizon),
            "display_horizon_start": display_start,
            "display_horizon_end": display_end,
        },
        "retrieval": {
            "selected_validation_config": selected,
            "neighbor_indices": [int(value) for value in plotted_neighbors],
            "neighbor_scores": [
                round(float(value), 8) for value in plotted_scores
            ],
            "recomputed_prediction_max_abs_error": max_reconstruction_error,
        },
        "screening": {
            "candidate_count": int(score.size),
            "eligible_count": int(np.isfinite(score).sum()),
            "criteria": {
                "relative_improvement_min": 0.15,
                "correction_alignment_min": 0.45,
                "context_similarity_quantile_min": 0.50,
                "residual_agreement_quantile_min": 0.50,
            },
            "score_weights": {
                "relative_improvement_rank": 0.30,
                "absolute_gain_rank": 0.18,
                "context_similarity_rank": 0.17,
                "residual_agreement_rank": 0.13,
                "future_spread_rank": 0.12,
                "correction_energy_rank": 0.10,
            },
            "selected": top_records[0],
            "top_candidates": top_records,
            "display_segment": {
                "start": display_start,
                "end": display_end,
                "length": display_length,
                "score": round(float(segment_score[display_start]), 8),
                "relative_improvement": round(
                    float(
                        segment_metrics["relative_improvement"][display_start]
                    ),
                    8,
                ),
                "correction_alignment": round(
                    float(
                        segment_metrics["correction_alignment"][display_start]
                    ),
                    8,
                ),
                "future_spread": round(
                    float(segment_metrics["future_spread"][display_start]),
                    8,
                ),
                "residual_agreement": round(
                    float(
                        segment_metrics["residual_agreement"][display_start]
                    ),
                    8,
                ),
            },
        },
        "series": {
            "query_context": json_series(query_context_values),
            "query_context_normalized": json_series(query_context_normalized),
            "neighbor_contexts": [
                json_series(values) for values in memory_context_values
            ],
            "neighbor_contexts_normalized": [
                json_series(values) for values in memory_context_normalized
            ],
            "neighbor_futures": [
                json_series(
                    validation["y"][
                        neighbor,
                        display_start:display_end,
                        selected_channel,
                    ]
                )
                for neighbor in plotted_neighbors
            ],
            "neighbor_base_forecasts": [
                json_series(
                    validation["y_base"][
                        neighbor,
                        display_start:display_end,
                        selected_channel,
                    ]
                )
                for neighbor in plotted_neighbors
            ],
            "neighbor_residuals": [
                json_series(
                    validation["future_residual"][
                        neighbor,
                        display_start:display_end,
                        selected_channel,
                    ]
                )
                for neighbor in plotted_neighbors
            ],
            "retrieved_correction": json_series(
                applied_correction[
                    selected_sample,
                    display_start:display_end,
                    selected_channel,
                ]
            ),
            "query_truth": json_series(
                query_truth_full[display_start:display_end]
            ),
            "query_base_forecast": json_series(
                query_base_full[display_start:display_end]
            ),
            "query_corrected_forecast": json_series(
                query_corrected_full[display_start:display_end]
            ),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        json.dump(payload, output, indent=2, sort_keys=True)
        output.write("\n")
    temporary.replace(output_path)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=DEFAULT_ARTIFACT,
        help=f"Experiment artifact directory (default: {DEFAULT_ARTIFACT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Selected case JSON (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--top-neighbors",
        type=int,
        default=3,
        help="Number of highest-scoring real neighbors to export",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.top_neighbors < 2:
        raise ValueError("--top-neighbors must be at least 2")
    payload = select_case(
        args.artifact_dir.resolve(),
        args.output.resolve(),
        args.top_neighbors,
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "source": payload["source"],
                "selected": payload["screening"]["selected"],
                "recomputed_prediction_max_abs_error": payload["retrieval"][
                    "recomputed_prediction_max_abs_error"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
