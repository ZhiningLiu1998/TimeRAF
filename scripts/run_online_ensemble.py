import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from ts_rag.config import build_parser, build_setting, prepare_args
from ts_rag.evaluation import compute_metrics, save_json
from ts_rag.online_ensemble import (
    OnlineEnsembleConfig,
    SharedNLinearRidge,
    apply_online_ensemble,
    build_online_statistics,
    causal_online_weights,
    make_auxiliary_forecasts,
)
from ts_rag.pipeline import load_prediction_bundle


def _load_bundles(args, setting):
    artifact_dir = os.path.join(args.output_dir, setting)
    return {
        split: load_prediction_bundle(
            os.path.join(artifact_dir, f"{split}_predictions.pkl")
        )
        for split in ("train", "val", "test")
    }


def _candidate_configs():
    priors = (
        (0.0, 0.0, 0.0),
        (0.5, 0.25, 0.1),
        (0.75, 0.2, 0.05),
    )
    for window, ridge_strength, prior, horizon_block in itertools.product(
        (96, 192, 384, 768, 1536),
        (0.01, 0.1, 1.0),
        priors,
        (0, 24),
    ):
        yield OnlineEnsembleConfig(window, ridge_strength, prior, horizon_block)


def _selection_score(candidate, baseline):
    return max(
        candidate["mse"] / baseline["mse"],
        candidate["mae"] / baseline["mae"],
    )


def _search_validation(bundle, auxiliary):
    start = len(bundle["y"]) // 3
    query_origins = np.arange(start, len(bundle["y"]))
    statistics = {
        horizon_block: build_online_statistics(
            [bundle],
            [auxiliary],
            [np.arange(len(bundle["y"]))],
            horizon_block=horizon_block,
        )
        for horizon_block in (0, 24)
    }
    baseline = compute_metrics(bundle["y_base"][start:], bundle["y"][start:])
    rows = []
    best = None
    for config in _candidate_configs():
        weights = causal_online_weights(
            statistics[config.horizon_block], query_origins, config
        )
        candidate = apply_online_ensemble(
            {key: value[start:] if isinstance(value, np.ndarray) else value
             for key, value in bundle.items()},
            tuple(member[start:] for member in auxiliary),
            weights,
        )
        metrics = compute_metrics(candidate, bundle["y"][start:])
        row = {
            "config": config.to_dict(),
            "selection_score": _selection_score(metrics, baseline),
            "metrics": metrics,
        }
        rows.append(row)
        if best is None or row["selection_score"] < best["selection_score"]:
            best = row
    return baseline, best, rows


def _fixed_robust_selection(rows):
    fixed = {
        "window": 1536,
        "ridge_strength": 0.01,
        "prior": [0.5, 0.25, 0.1],
        "horizon_block": 24,
    }
    for row in rows:
        if row["config"] == fixed:
            return row
    raise RuntimeError(f"Fixed robust configuration was not evaluated: {fixed}")


def _evaluate_test(val_bundle, test_bundle, val_auxiliary, test_auxiliary, config):
    val_count = len(val_bundle["y"])
    test_origins = val_count + test_bundle["y"].shape[1] - 1 + np.arange(
        len(test_bundle["y"])
    )
    statistics = build_online_statistics(
        [val_bundle, test_bundle],
        [val_auxiliary, test_auxiliary],
        [np.arange(val_count), test_origins],
        horizon_block=config.horizon_block,
    )
    weights = causal_online_weights(statistics, test_origins, config)
    corrected = apply_online_ensemble(test_bundle, test_auxiliary, weights)
    return corrected, weights


def main():
    parser = build_parser("Causal online ensemble for TimeMixer")
    parser.add_argument("--linear_ridge_alpha", type=float, default=100.0)
    parser.add_argument(
        "--online_policy",
        choices=("validation_best", "fixed_robust"),
        default="validation_best",
    )
    args = prepare_args(parser.parse_args())
    setting = build_setting(args)
    bundles = _load_bundles(args, setting)

    linear_model = SharedNLinearRidge(args.linear_ridge_alpha).fit(bundles["train"])
    auxiliary = {
        split: make_auxiliary_forecasts(bundle, linear_model)
        for split, bundle in bundles.items()
        if split != "train"
    }
    validation_baseline, selected, rows = _search_validation(
        bundles["val"], auxiliary["val"]
    )
    if args.online_policy == "fixed_robust":
        selected = _fixed_robust_selection(rows)
    config = OnlineEnsembleConfig(
        window=selected["config"]["window"],
        ridge_strength=selected["config"]["ridge_strength"],
        prior=tuple(selected["config"]["prior"]),
        horizon_block=selected["config"]["horizon_block"],
    )
    corrected, weights = _evaluate_test(
        bundles["val"],
        bundles["test"],
        auxiliary["val"],
        auxiliary["test"],
        config,
    )
    test_baseline = compute_metrics(bundles["test"]["y_base"], bundles["test"]["y"])
    test_corrected = compute_metrics(corrected, bundles["test"]["y"])

    artifact_dir = os.path.join(args.output_dir, setting)
    payload = {
        "online_policy": args.online_policy,
        "linear_ridge_alpha": args.linear_ridge_alpha,
        "validation_baseline": validation_baseline,
        "selected": selected,
        "test_baseline": test_baseline,
        "test_corrected": test_corrected,
        "test_mse_improvement_percent": 100.0
        * (test_baseline["mse"] - test_corrected["mse"])
        / test_baseline["mse"],
        "validation_search": rows,
        "mean_test_weights": weights.mean(
            axis=tuple(range(weights.ndim - 1))
        ).tolist(),
    }
    result_stem = (
        "robust_online_ensemble"
        if args.online_policy == "fixed_robust"
        else "online_ensemble"
    )
    save_json(payload, os.path.join(artifact_dir, f"{result_stem}.json"))
    np.savez_compressed(
        os.path.join(artifact_dir, f"{result_stem}_predictions.npz"),
        corrected=corrected,
        true=bundles["test"]["y"],
        baseline=bundles["test"]["y_base"],
        weights=weights,
    )
    print(f"setting={setting}")
    print(f"selected={selected}")
    print(f"mean_test_weights={payload['mean_test_weights']}")
    print(f"test_baseline={test_baseline}")
    print(f"test_corrected={test_corrected}")
    print(f"mse_improvement_percent={payload['test_mse_improvement_percent']:.4f}")


if __name__ == "__main__":
    main()
