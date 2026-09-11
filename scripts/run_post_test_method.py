import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from ts_rag.config import build_parser, build_setting, prepare_args
from ts_rag.evaluation import compute_metrics, save_json
from ts_rag.historical_retrieval import HistoricalResidualIndex, RetrievalFeatureConfig
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
        for split in ("train", "val", "test", "post_test")
    }


def _concatenate_bundles(*bundles):
    keys = ("x", "x_mark", "y", "y_mark", "y_base")
    return {
        key: np.concatenate([bundle[key] for bundle in bundles], axis=0)
        for key in keys
    }


def _run_online_ensemble(bundles):
    config = OnlineEnsembleConfig(
        window=1536,
        ridge_strength=0.01,
        prior=(0.5, 0.25, 0.1),
        horizon_block=24,
    )
    linear_model = SharedNLinearRidge(alpha=100.0).fit(bundles["train"])
    auxiliary = {
        split: make_auxiliary_forecasts(bundles[split], linear_model)
        for split in ("val", "test", "post_test")
    }
    pred_len = bundles["post_test"]["y"].shape[1]
    val_origins = np.arange(len(bundles["val"]["y"]))
    test_origins = len(val_origins) + pred_len - 1 + np.arange(
        len(bundles["test"]["y"])
    )
    post_origins = test_origins[-1] + pred_len + np.arange(
        len(bundles["post_test"]["y"])
    )
    statistics = build_online_statistics(
        [bundles["val"], bundles["test"], bundles["post_test"]],
        [auxiliary["val"], auxiliary["test"], auxiliary["post_test"]],
        [val_origins, test_origins, post_origins],
        horizon_block=config.horizon_block,
    )
    weights = causal_online_weights(statistics, post_origins, config)
    corrected = apply_online_ensemble(
        bundles["post_test"],
        auxiliary["post_test"],
        weights,
    )
    return corrected, {
        "method": "causal_online_ensemble",
        "config": config.to_dict(),
        "linear_ridge_alpha": 100.0,
        "mean_weights": weights.mean(
            axis=tuple(range(weights.ndim - 1))
        ).tolist(),
    }


def _run_historical_retrieval(bundles, artifact_dir):
    selection_path = os.path.join(artifact_dir, "adaptive_timeraf.json")
    with open(selection_path, encoding="utf-8") as file:
        selection = json.load(file)["selected"]
    if selection["method"] != "historical_residual_retrieval":
        raise RuntimeError(
            "Post-test minute evaluation expected historical residual retrieval, "
            f"received {selection['method']}"
        )

    history = _concatenate_bundles(bundles["val"], bundles["test"])
    feature_config = RetrievalFeatureConfig(**selection["feature_config"])
    index = HistoricalResidualIndex(feature_config).fit(history)
    neighbors, scores = index.query_neighbors(
        bundles["post_test"],
        max_k=selection["k"],
    )
    correction = index.aggregate(
        neighbors,
        scores,
        k=selection["k"],
        temperature=selection["temperature"],
        shrinkage=selection["shrinkage"],
    )
    corrected = bundles["post_test"]["y_base"] + selection["alpha"] * correction
    return corrected, {
        "method": "historical_residual_retrieval",
        "selection_source": selection_path,
        "config": selection,
        "history_window_count": len(history["y"]),
    }


def main():
    parser = build_parser("Evaluate the frozen TimeRAF method on ETT post-test data")
    args = prepare_args(parser.parse_args())
    setting = build_setting(args)
    bundles = _load_bundles(args, setting)
    artifact_dir = os.path.join(args.output_dir, setting)

    if args.dataset.startswith("ETTh"):
        corrected, method = _run_online_ensemble(bundles)
    else:
        corrected, method = _run_historical_retrieval(bundles, artifact_dir)

    post_test = bundles["post_test"]
    baseline = compute_metrics(post_test["y_base"], post_test["y"])
    candidate = compute_metrics(corrected, post_test["y"])
    payload = {
        "setting": setting,
        "method": method,
        "baseline": baseline,
        "candidate": candidate,
        "mse_improvement_percent": 100.0
        * (baseline["mse"] - candidate["mse"])
        / baseline["mse"],
        "mae_improvement_percent": 100.0
        * (baseline["mae"] - candidate["mae"])
        / baseline["mae"],
    }
    save_json(payload, os.path.join(artifact_dir, "post_test_method.json"))
    np.savez_compressed(
        os.path.join(artifact_dir, "post_test_method_predictions.npz"),
        baseline=post_test["y_base"],
        corrected=corrected,
        true=post_test["y"],
    )
    print(f"setting={setting}")
    print(f"method={method}")
    print(f"baseline={baseline}")
    print(f"candidate={candidate}")
    print(f"mse_improvement_percent={payload['mse_improvement_percent']:.4f}")
    print(f"mae_improvement_percent={payload['mae_improvement_percent']:.4f}")


if __name__ == "__main__":
    main()
