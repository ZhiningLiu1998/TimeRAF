import copy
import hashlib

import pytest

from ts_rag.matrix import build_confirmation_summary


def _cell(index, family):
    metrics = ["mae", "rmse"] if family == "pems" else ["mse", "mae"]
    return {
        "id": f"{family}/D/M/{index}",
        "task_family": family,
        "dataset": "D",
        "model": "M",
        "pred_len": 96,
        "metrics": metrics,
    }


def _ids_sha256(cell_ids):
    payload = "".join(f"{cell_id}\n" for cell_id in sorted(cell_ids))
    return hashlib.sha256(payload.encode()).hexdigest()


def _fixture():
    manifest = [
        *[_cell(index, "long_term") for index in range(10)],
        *[_cell(index, "pems") for index in range(10, 20)],
    ]
    states = []
    for index, cell in enumerate(manifest):
        improved = index not in {8, 18}
        metrics = cell["metrics"]
        states.append(
            {
                "cell_id": cell["id"],
                "task_family": cell["task_family"],
                "dataset": cell["dataset"],
                "model": cell["model"],
                "pred_len": cell["pred_len"],
                "state": "completed",
                "started_unix": float(index + 1),
                "all_test_metrics_improve": improved,
                "metric_gain_percent": {
                    metric: 10.0 if improved else -1.0
                    for metric in metrics
                },
            }
        )
    development_ids = [row["cell_id"] for row in states[:2]]
    protocol = {
        "full_matrix_cell_count": 20,
        "development_cell_count": 2,
        "confirmatory_cell_count": 18,
        "method_source_revision": "revision",
        "development_selection": {
            "rule": "first_completed_cells_by_started_unix",
            "tie_breaker": "cell_id",
            "cell_ids_sha256": _ids_sha256(development_ids),
            "last_started_unix": 2.0,
        },
        "confirmatory_selection": {
            "rule": "full matrix complement of frozen development cells",
            "first_started_unix": 3.0,
            "minimum_boundary_gap_seconds": 1.0,
        },
        "thresholds": {
            "overall_strict_improvement_rate": 0.8,
            "task_family_strict_improvement_rate": 0.7,
            "positive_task_family_metric_means": True,
            "positive_overall_metric_means_and_medians": True,
            "complete_without_runtime_failures": True,
        },
        "independence": {
            "selection_uses_outcomes": False,
            "method_changes_after_boundary_allowed": False,
            "development_cells_excluded_from_confirmatory_denominators": True,
        },
    }
    matrix_summary = {
        "source_revision": "revision-full",
        "cell_states": states,
    }
    return manifest, matrix_summary, protocol


def test_confirmation_gate_uses_only_frozen_complement():
    manifest, matrix_summary, protocol = _fixture()

    summary = build_confirmation_summary(
        manifest,
        matrix_summary,
        protocol,
    )

    assert summary["scope"]["development"] == 2
    assert summary["scope"]["confirmatory"] == 18
    assert summary["selection"]["boundary_gap_seconds"] == 1.0
    gate = summary["confirmatory_gate"]
    assert gate["overall"]["improved"] == 16
    assert gate["overall"]["improvement_rate"] == pytest.approx(16 / 18)
    assert gate["confirmatory_gate_passed"]


def test_confirmation_rejects_development_set_drift():
    manifest, matrix_summary, protocol = _fixture()
    drifted = copy.deepcopy(matrix_summary)
    drifted["cell_states"][0]["started_unix"] = 100.0

    with pytest.raises(ValueError, match="cell ID hash"):
        build_confirmation_summary(manifest, drifted, protocol)


def test_confirmation_remains_incomplete_until_all_heldout_cells_finish():
    manifest, matrix_summary, protocol = _fixture()
    matrix_summary["cell_states"][-1]["state"] = "pending"
    matrix_summary["cell_states"][-1].pop("metric_gain_percent")
    matrix_summary["cell_states"][-1].pop("all_test_metrics_improve")

    summary = build_confirmation_summary(
        manifest,
        matrix_summary,
        protocol,
    )

    assert not summary["confirmatory_gate"]["confirmatory_gate_passed"]
    assert not summary["confirmatory_gate"]["criteria"][
        "confirmatory_complete_without_runtime_failures"
    ]
