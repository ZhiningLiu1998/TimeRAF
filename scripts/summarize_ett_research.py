import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from ts_rag.evaluation import compute_metrics, save_json
from ts_rag.significance import paired_multi_seed_moving_block_bootstrap


DATASETS = ("ETTh1", "ETTh2", "ETTm1", "ETTm2")
EXPECTED_SEEDS = (2021, 2022, 2023)
METHOD_ARTIFACTS = {
    "ETTh1": ("robust_online_ensemble", "Causal online ensemble"),
    "ETTh2": ("robust_online_ensemble", "Causal online ensemble"),
    "ETTm1": ("adaptive_timeraf", "Adaptive TimeRAF"),
    "ETTm2": ("adaptive_timeraf", "Adaptive TimeRAF"),
}


def _load_json(path):
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def _discover_runs(root):
    runs = {dataset: {} for dataset in DATASETS}
    for metadata_path in glob.glob(os.path.join(root, "*", "baseline_run.json")):
        metadata = _load_json(metadata_path)
        dataset = metadata["args"]["dataset"]
        seed = int(metadata["args"]["seed"])
        if dataset not in runs or seed not in EXPECTED_SEEDS:
            continue
        if seed in runs[dataset]:
            raise RuntimeError(f"Duplicate {dataset} seed {seed} run")
        runs[dataset][seed] = os.path.dirname(metadata_path)

    missing = {
        dataset: sorted(set(EXPECTED_SEEDS) - set(dataset_runs))
        for dataset, dataset_runs in runs.items()
        if set(dataset_runs) != set(EXPECTED_SEEDS)
    }
    if missing:
        raise RuntimeError(f"Missing required baseline runs: {missing}")
    return runs


def _load_candidate(artifact_dir, stem):
    metadata = _load_json(os.path.join(artifact_dir, f"{stem}.json"))
    with np.load(
        os.path.join(artifact_dir, f"{stem}_predictions.npz")
    ) as archive:
        arrays = {
            key: np.asarray(archive[key])
            for key in ("baseline", "corrected", "true")
        }
    return metadata, arrays


def _improvement_percent(baseline, candidate, metric):
    return 100.0 * (baseline[metric] - candidate[metric]) / baseline[metric]


def _summarize_dataset(dataset, seed_dirs):
    stem, method_name = METHOD_ARTIFACTS[dataset]
    per_seed = {}
    baseline_predictions = []
    candidate_predictions = []
    truths = []

    for seed in EXPECTED_SEEDS:
        metadata, arrays = _load_candidate(seed_dirs[seed], stem)
        baseline = compute_metrics(arrays["baseline"], arrays["true"])
        candidate = compute_metrics(arrays["corrected"], arrays["true"])
        per_seed[str(seed)] = {
            "baseline": baseline,
            "candidate": candidate,
            "mse_improvement_percent": _improvement_percent(
                baseline, candidate, "mse"
            ),
            "mae_improvement_percent": _improvement_percent(
                baseline, candidate, "mae"
            ),
            "selected": metadata["selected"],
        }
        baseline_predictions.append(arrays["baseline"])
        candidate_predictions.append(arrays["corrected"])
        truths.append(arrays["true"])

    pooled_baseline = compute_metrics(
        np.concatenate(baseline_predictions, axis=0),
        np.concatenate(truths, axis=0),
    )
    pooled_candidate = compute_metrics(
        np.concatenate(candidate_predictions, axis=0),
        np.concatenate(truths, axis=0),
    )
    significance = {
        loss: paired_multi_seed_moving_block_bootstrap(
            baseline_predictions,
            candidate_predictions,
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
        "method": method_name,
        "artifact_stem": stem,
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
        dataset: _summarize_dataset(dataset, runs[dataset])
        for dataset in DATASETS
    }
    summary["all_datasets_passed"] = all(
        summary[dataset]["gates"]["passed"] for dataset in DATASETS
    )
    save_json(summary, os.path.join(root, "ett_multiseed_summary.json"))

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
