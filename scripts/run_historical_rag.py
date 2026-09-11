import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from ts_rag.config import build_parser, build_setting, prepare_args
from ts_rag.evaluation import compute_metrics, save_json
from ts_rag.historical_retrieval import HistoricalResidualIndex, RetrievalFeatureConfig
from ts_rag.pipeline import load_prediction_bundle


def _candidate_feature_configs(search_profile):
    if search_profile == "broad":
        combinations = itertools.product((24, 48, 96), (False, True), (False, True))
    else:
        combinations = itertools.product((48, 96), (True,), (False,))
    for tail_len, include_forecast, include_calendar in combinations:
        yield RetrievalFeatureConfig(
            tail_len=tail_len,
            pooled_steps=12,
            include_differences=True,
            include_forecast=include_forecast,
            include_calendar=include_calendar,
        )


def _load_bundles(args, setting):
    artifact_dir = os.path.join(args.output_dir, setting)
    return {
        split: load_prediction_bundle(os.path.join(artifact_dir, f"{split}_predictions.pkl"))
        for split in ("val", "test")
    }


def _search_validation(val_bundle, pred_len, search_profile):
    sample_count = len(val_bundle["y"])
    boundaries = (sample_count // 3, 2 * sample_count // 3, sample_count)
    folds = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        memory_indices = np.arange(0, max(1, start - pred_len))
        query_indices = np.arange(start, end)
        folds.append((memory_indices, query_indices))

    baseline = np.concatenate(
        [val_bundle["y_base"][query_indices] for _, query_indices in folds], axis=0
    )
    true = np.concatenate(
        [val_bundle["y"][query_indices] for _, query_indices in folds], axis=0
    )
    baseline_metrics = compute_metrics(baseline, true)

    search_rows = []
    best = None

    def selection_score(metrics):
        mse_ratio = metrics["mse"] / baseline_metrics["mse"]
        mae_ratio = metrics["mae"] / baseline_metrics["mae"]
        return max(mse_ratio, mae_ratio)

    for feature_config in _candidate_feature_configs(search_profile):
        fold_retrieval = []
        for memory_indices, query_indices in folds:
            index = HistoricalResidualIndex(feature_config).fit(val_bundle, memory_indices)
            neighbor_indices, neighbor_scores = index.query_neighbors(
                val_bundle, query_indices, max_k=64
            )
            fold_retrieval.append((index, neighbor_indices, neighbor_scores))
        if search_profile == "broad":
            combinations = itertools.product(
                (4, 8, 16, 32, 64), (0.03, 0.1, 0.3), (0.0, 0.25, 1.0, 4.0)
            )
            alphas = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
        else:
            combinations = itertools.product((16, 32, 64), (0.1, 0.3), (1.0, 4.0))
            alphas = (0.0, 0.25, 0.5, 0.75)
        for k, temperature, shrinkage in combinations:
            correction = np.concatenate(
                [
                    index.aggregate(
                        neighbor_indices,
                        neighbor_scores,
                        k,
                        temperature,
                        shrinkage=shrinkage,
                    )
                    for index, neighbor_indices, neighbor_scores in fold_retrieval
                ],
                axis=0,
            )
            for alpha in alphas:
                metrics = compute_metrics(baseline + alpha * correction, true)
                row = {
                    "feature_config": feature_config.to_dict(),
                    "k": k,
                    "temperature": temperature,
                    "shrinkage": shrinkage,
                    "alpha": alpha,
                    "selection_score": selection_score(metrics),
                    "metrics": metrics,
                }
                search_rows.append(row)
                if best is None or row["selection_score"] < best["selection_score"]:
                    best = row
    return baseline_metrics, best, search_rows


def _evaluate_test(val_bundle, test_bundle, selected):
    feature_config = RetrievalFeatureConfig(**selected["feature_config"])
    index = HistoricalResidualIndex(feature_config).fit(val_bundle)
    neighbors, scores = index.query_neighbors(
        test_bundle, max_k=selected["k"]
    )
    correction = index.aggregate(
        neighbors,
        scores,
        k=selected["k"],
        temperature=selected["temperature"],
        shrinkage=selected["shrinkage"],
    )
    corrected = test_bundle["y_base"] + selected["alpha"] * correction
    return corrected, compute_metrics(corrected, test_bundle["y"])


def _search_seasonal(val_bundle):
    start = len(val_bundle["y"]) // 3
    baseline = val_bundle["y_base"][start:]
    true = val_bundle["y"][start:]
    x = val_bundle["x"][start:]
    baseline_metrics = compute_metrics(baseline, true)
    rows = []
    best = None
    for period in (4, 8, 12, 24, 32, 48, 96):
        repeats = int(np.ceil(baseline.shape[1] / period))
        seasonal = np.tile(x[:, -period:, :], (1, repeats, 1))[:, : baseline.shape[1], :]
        for alpha in (0.0, 0.05, 0.1, 0.2, 0.3, 0.5):
            metrics = compute_metrics((1.0 - alpha) * baseline + alpha * seasonal, true)
            row = {
                "method": "seasonal_self_retrieval",
                "period": period,
                "alpha": alpha,
                "selection_score": max(
                    metrics["mse"] / baseline_metrics["mse"],
                    metrics["mae"] / baseline_metrics["mae"],
                ),
                "metrics": metrics,
            }
            rows.append(row)
            if best is None or row["selection_score"] < best["selection_score"]:
                best = row
    return best, rows


def _evaluate_seasonal(test_bundle, selected):
    period = selected["period"]
    repeats = int(np.ceil(test_bundle["y_base"].shape[1] / period))
    seasonal = np.tile(
        test_bundle["x"][:, -period:, :], (1, repeats, 1)
    )[:, : test_bundle["y_base"].shape[1], :]
    corrected = (
        (1.0 - selected["alpha"]) * test_bundle["y_base"]
        + selected["alpha"] * seasonal
    )
    return corrected, compute_metrics(corrected, test_bundle["y"])


def main():
    parser = build_parser("Validation-selected historical residual retrieval")
    parser.add_argument("--search_profile", choices=("narrow", "broad"), default="narrow")
    args = prepare_args(parser.parse_args())
    setting = build_setting(args)
    bundles = _load_bundles(args, setting)

    val_baseline, historical_selected, historical_rows = _search_validation(
        bundles["val"], args.pred_len, args.search_profile
    )
    historical_selected = {"method": "historical_residual_retrieval", **historical_selected}
    seasonal_selected, seasonal_rows = _search_seasonal(bundles["val"])
    if seasonal_selected["selection_score"] < historical_selected["selection_score"]:
        selected = seasonal_selected
        corrected, test_metrics = _evaluate_seasonal(bundles["test"], selected)
    else:
        selected = historical_selected
        corrected, test_metrics = _evaluate_test(bundles["val"], bundles["test"], selected)
    test_baseline = compute_metrics(bundles["test"]["y_base"], bundles["test"]["y"])

    artifact_dir = os.path.join(args.output_dir, setting)
    payload = {
        "validation_baseline": val_baseline,
        "selected": selected,
        "test_baseline": test_baseline,
        "test_corrected": test_metrics,
        "test_mse_improvement_percent": 100.0
        * (test_baseline["mse"] - test_metrics["mse"])
        / test_baseline["mse"],
        "historical_selected": historical_selected,
        "seasonal_selected": seasonal_selected,
        "validation_search": historical_rows + seasonal_rows,
    }
    save_json(payload, os.path.join(artifact_dir, "adaptive_timeraf.json"))
    np.savez_compressed(
        os.path.join(artifact_dir, "adaptive_timeraf_predictions.npz"),
        corrected=corrected,
        true=bundles["test"]["y"],
        baseline=bundles["test"]["y_base"],
    )
    print(f"setting={setting}")
    print(f"selected={selected}")
    print(f"test_baseline={test_baseline}")
    print(f"test_corrected={test_metrics}")
    print(f"mse_improvement_percent={payload['test_mse_improvement_percent']:.4f}")


if __name__ == "__main__":
    main()
