import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from ts_rag.config import build_parser, build_setting, prepare_args
from ts_rag.evaluation import compute_metrics, save_json
from ts_rag.historical_retrieval import (
    CausalHistoricalResidualIndex,
    RetrievalFeatureConfig,
)
from ts_rag.pipeline import load_prediction_bundle


def _feature_configs():
    for tail_len in (48, 96):
        yield RetrievalFeatureConfig(
            tail_len=tail_len,
            pooled_steps=12,
            include_differences=True,
            include_forecast=True,
            include_calendar=False,
        )


def _load_bundles(args, setting):
    artifact_dir = os.path.join(args.output_dir, setting)
    return {
        split: load_prediction_bundle(os.path.join(artifact_dir, f"{split}_predictions.pkl"))
        for split in ("val", "test")
    }


def _concatenate_bundles(first, second):
    keys = ("x", "x_mark", "y", "y_mark", "y_base")
    combined = {key: np.concatenate([first[key], second[key]], axis=0) for key in keys}
    combined["split"] = "causal_timeline"
    return combined


def _selection_score(metrics, baseline):
    return max(metrics["mse"] / baseline["mse"], metrics["mae"] / baseline["mae"])


def _search_validation(val_bundle, pred_len):
    count = len(val_bundle["y"])
    history_end = count // 3
    normalization_indices = np.arange(0, history_end - pred_len)
    query_indices = np.arange(history_end, count)
    origins = np.arange(count)
    baseline_pred = val_bundle["y_base"][query_indices]
    true = val_bundle["y"][query_indices]
    baseline_metrics = compute_metrics(baseline_pred, true)

    rows = []
    best = None
    for feature_config in _feature_configs():
        index = CausalHistoricalResidualIndex(feature_config).fit_timeline(
            val_bundle, normalization_indices
        )
        neighbors, scores = index.query_causal(
            query_indices, origins, min_lag=pred_len, max_k=64
        )
        for k, temperature, shrinkage in itertools.product(
            (16, 32, 64), (0.1, 0.3), (1.0, 4.0)
        ):
            correction = index.aggregate(
                neighbors,
                scores,
                k=k,
                temperature=temperature,
                shrinkage=shrinkage,
            )
            for alpha in (0.0, 0.25, 0.5, 0.75):
                metrics = compute_metrics(baseline_pred + alpha * correction, true)
                row = {
                    "feature_config": feature_config.to_dict(),
                    "k": k,
                    "temperature": temperature,
                    "shrinkage": shrinkage,
                    "alpha": alpha,
                    "selection_score": _selection_score(metrics, baseline_metrics),
                    "metrics": metrics,
                }
                rows.append(row)
                if best is None or row["selection_score"] < best["selection_score"]:
                    best = row
    return baseline_metrics, best, rows


def _evaluate_test(val_bundle, test_bundle, selected, pred_len):
    combined = _concatenate_bundles(val_bundle, test_bundle)
    val_count = len(val_bundle["y"])
    test_count = len(test_bundle["y"])
    split_span = val_count + pred_len - 1
    origins = np.concatenate(
        [np.arange(val_count), split_span + np.arange(test_count)]
    )
    query_indices = np.arange(val_count, val_count + test_count)

    feature_config = RetrievalFeatureConfig(**selected["feature_config"])
    index = CausalHistoricalResidualIndex(feature_config).fit_timeline(
        combined, np.arange(val_count)
    )
    neighbors, scores = index.query_causal(
        query_indices, origins, min_lag=pred_len, max_k=selected["k"]
    )
    correction = index.aggregate(
        neighbors,
        scores,
        k=selected["k"],
        temperature=selected["temperature"],
        shrinkage=selected["shrinkage"],
    )
    corrected = test_bundle["y_base"] + selected["alpha"] * correction
    return corrected, compute_metrics(corrected, test_bundle["y"]), neighbors, origins


def main():
    parser = build_parser("Causal online historical residual retrieval")
    args = prepare_args(parser.parse_args())
    setting = build_setting(args)
    bundles = _load_bundles(args, setting)
    val_baseline, selected, rows = _search_validation(bundles["val"], args.pred_len)
    corrected, test_metrics, neighbors, origins = _evaluate_test(
        bundles["val"], bundles["test"], selected, args.pred_len
    )
    test_baseline = compute_metrics(bundles["test"]["y_base"], bundles["test"]["y"])

    val_count = len(bundles["val"]["y"])
    query_origins = origins[val_count:]
    retrieved_origins = origins[neighbors]
    max_causality_gap = int(np.max(retrieved_origins - query_origins[:, None]))
    if max_causality_gap > -args.pred_len:
        raise AssertionError("A retrieved residual was not fully observed at query time")

    artifact_dir = os.path.join(args.output_dir, setting)
    payload = {
        "validation_baseline": val_baseline,
        "selected": selected,
        "test_baseline": test_baseline,
        "test_corrected": test_metrics,
        "test_mse_improvement_percent": 100.0
        * (test_baseline["mse"] - test_metrics["mse"])
        / test_baseline["mse"],
        "max_retrieved_origin_minus_query_origin": max_causality_gap,
        "minimum_required_lag": args.pred_len,
        "validation_search": rows,
    }
    save_json(payload, os.path.join(artifact_dir, "causal_historical_rag.json"))
    np.savez_compressed(
        os.path.join(artifact_dir, "causal_historical_rag_predictions.npz"),
        corrected=corrected,
        true=bundles["test"]["y"],
        baseline=bundles["test"]["y_base"],
    )
    print(f"setting={setting}")
    print(f"selected={selected}")
    print(f"test_baseline={test_baseline}")
    print(f"test_corrected={test_metrics}")
    print(f"mse_improvement_percent={payload['test_mse_improvement_percent']:.4f}")
    print(f"max_retrieved_origin_minus_query_origin={max_causality_gap}")


if __name__ == "__main__":
    main()
