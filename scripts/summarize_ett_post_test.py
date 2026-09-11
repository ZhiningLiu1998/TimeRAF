import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from scripts.summarize_ett_research import (
    DATASETS,
    EXPECTED_SEEDS,
    _discover_runs,
    _improvement_percent,
    _load_json,
)
from ts_rag.evaluation import compute_metrics, save_json
from ts_rag.significance import paired_multi_seed_moving_block_bootstrap


def _summarize_dataset(seed_dirs):
    per_seed = {}
    baselines, candidates, truths = [], [], []
    method = None
    for seed in EXPECTED_SEEDS:
        artifact_dir = seed_dirs[seed]
        metadata = _load_json(os.path.join(artifact_dir, "post_test_method.json"))
        with np.load(
            os.path.join(artifact_dir, "post_test_method_predictions.npz")
        ) as archive:
            baseline_prediction = np.asarray(archive["baseline"])
            candidate_prediction = np.asarray(archive["corrected"])
            true = np.asarray(archive["true"])
        baseline = compute_metrics(baseline_prediction, true)
        candidate = compute_metrics(candidate_prediction, true)
        per_seed[str(seed)] = {
            "baseline": baseline,
            "candidate": candidate,
            "mse_improvement_percent": _improvement_percent(
                baseline, candidate, "mse"
            ),
            "mae_improvement_percent": _improvement_percent(
                baseline, candidate, "mae"
            ),
        }
        method = metadata["method"]["method"]
        baselines.append(baseline_prediction)
        candidates.append(candidate_prediction)
        truths.append(true)

    pooled_baseline = compute_metrics(
        np.concatenate(baselines), np.concatenate(truths)
    )
    pooled_candidate = compute_metrics(
        np.concatenate(candidates), np.concatenate(truths)
    )
    significance = {
        loss: paired_multi_seed_moving_block_bootstrap(
            baselines,
            candidates,
            truths,
            loss=loss,
            block_length=96,
        )
        for loss in ("mse", "mae")
    }
    all_seed_mse_at_least_one_percent = all(
        row["mse_improvement_percent"] >= 1.0 for row in per_seed.values()
    )
    all_seed_mae_improved = all(
        row["mae_improvement_percent"] > 0.0 for row in per_seed.values()
    )
    mse_ci_below_zero = significance["mse"]["confidence_interval_95"][1] < 0.0
    return {
        "method": method,
        "per_seed": per_seed,
        "pooled": {
            "baseline": pooled_baseline,
            "candidate": pooled_candidate,
            "mse_improvement_percent": _improvement_percent(
                pooled_baseline, pooled_candidate, "mse"
            ),
            "mae_improvement_percent": _improvement_percent(
                pooled_baseline, pooled_candidate, "mae"
            ),
        },
        "significance": significance,
        "gates": {
            "all_seed_mse_at_least_one_percent": all_seed_mse_at_least_one_percent,
            "all_seed_mae_improved": all_seed_mae_improved,
            "pooled_mse_ci_below_zero": mse_ci_below_zero,
            "passed": (
                all_seed_mse_at_least_one_percent
                and all_seed_mae_improved
                and mse_ci_below_zero
            ),
        },
    }


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "./ts_rag_outputs/baselines"
    runs = _discover_runs(root)
    summary = {
        dataset: _summarize_dataset(runs[dataset])
        for dataset in DATASETS
    }
    summary["all_datasets_passed"] = all(
        summary[dataset]["gates"]["passed"] for dataset in DATASETS
    )
    save_json(summary, os.path.join(root, "ett_post_test_summary.json"))

    print(
        "dataset pooled_mse_gain pooled_mae_gain seed_mse_gains "
        "mse_ci all_gates_passed"
    )
    for dataset in DATASETS:
        result = summary[dataset]
        seed_gains = ",".join(
            f"{result['per_seed'][str(seed)]['mse_improvement_percent']:.2f}%"
            for seed in EXPECTED_SEEDS
        )
        mse_ci = result["significance"]["mse"]["confidence_interval_95"]
        print(
            dataset,
            f"{result['pooled']['mse_improvement_percent']:.2f}%",
            f"{result['pooled']['mae_improvement_percent']:.2f}%",
            seed_gains,
            f"[{mse_ci[0]:.6f}, {mse_ci[1]:.6f}]",
            result["gates"]["passed"],
        )
    print(f"all_datasets_passed={summary['all_datasets_passed']}")


if __name__ == "__main__":
    main()
