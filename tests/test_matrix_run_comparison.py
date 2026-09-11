import copy
import json

import pytest

from scripts import compare_timefuse_matrix_runs
from scripts.greenland_common import METHOD_REVISION
from ts_rag.matrix import load_manifest


def _records(manifest, *, elapsed, mode="trained"):
    records = {}
    for index, cell in enumerate(manifest):
        baseline = {
            metric: 1.0 + index / 10000 for metric in cell["metrics"]
        }
        corrected = {
            metric: value * 0.9 for metric, value in baseline.items()
        }
        records[cell["id"]] = {
            "status": {
                "cell_id": cell["id"],
                "seed": 2021,
                "source_revision": METHOD_REVISION,
                "smoke": False,
                "status": "completed",
                "started_unix": 1000.0 + index * elapsed / 8,
                "elapsed_seconds": elapsed,
            },
            "result": {
                "cell": {"id": cell["id"]},
                "checkpoint": {"mode": mode},
                "validation": {
                    "selected": {
                        "method": "historical_residual",
                        "params": {"alpha": 0.1},
                    }
                },
                "test_baseline": baseline,
                "test_corrected": corrected,
                "all_test_metrics_improve": True,
            },
        }
    return records


def test_full_matrix_comparison_accepts_consistent_results_and_times_speedup():
    manifest = load_manifest("docs/timefuse_experiment_manifest.jsonl")
    a10g = _records(manifest, elapsed=20.0)
    a100 = _records(manifest, elapsed=10.0)

    report = compare_timefuse_matrix_runs.build_comparison(
        manifest,
        a10g,
        a100,
        manifest_sha256="a" * 64,
        method_revision=METHOD_REVISION,
        seed=2021,
        absolute_tolerance=1e-5,
        relative_tolerance=1e-4,
        a100_matrix_seconds=800.0,
    )

    assert report["consistency_gates"]["all_passed"]
    expected_metrics = sum(len(cell["metrics"]) for cell in manifest)
    assert report["metric_consistency"]["baseline"]["count"] == expected_metrics
    assert (
        report["metric_consistency"]["corrected"]["within_tolerance"]
        == expected_metrics
    )
    assert report["selection_and_outcome_consistency"][
        "selected_parameter_agreements"
    ] == 585
    trained = report["runtime"]["both_trained_comparable_subset"]
    assert trained["cell_count"] == 585
    assert trained["ratio_of_summed_cell_seconds"] == pytest.approx(2.0)
    assert trained["per_cell_a100_speedup"]["median"] == pytest.approx(2.0)


def test_full_matrix_comparison_reports_numerical_and_selection_drift():
    manifest = load_manifest("docs/timefuse_experiment_manifest.jsonl")
    a10g = _records(manifest, elapsed=20.0)
    a100 = _records(manifest, elapsed=10.0)
    cell = manifest[0]
    changed = copy.deepcopy(a100[cell["id"]])
    changed["result"]["test_corrected"][cell["metrics"][0]] += 0.1
    changed["result"]["validation"]["selected"]["params"]["alpha"] = 0.2
    changed["result"]["all_test_metrics_improve"] = False
    a100[cell["id"]] = changed

    report = compare_timefuse_matrix_runs.build_comparison(
        manifest,
        a10g,
        a100,
        manifest_sha256="a" * 64,
        method_revision=METHOD_REVISION,
        seed=2021,
        absolute_tolerance=1e-5,
        relative_tolerance=1e-4,
    )

    assert not report["consistency_gates"]["all_passed"]
    assert not report["consistency_gates"][
        "all_corrected_metrics_within_tolerance"
    ]
    assert not report["consistency_gates"]["all_selected_parameters_agree"]
    assert not report["consistency_gates"][
        "all_strict_improvement_classifications_agree"
    ]
    expected_metrics = sum(len(cell["metrics"]) for cell in manifest)
    assert (
        report["metric_consistency"]["corrected"]["within_tolerance"]
        == expected_metrics - 1
    )


def test_runtime_uses_separate_first_pass_records_after_recovery_composition():
    manifest = load_manifest("docs/timefuse_experiment_manifest.jsonl")
    composed_a10g = _records(manifest, elapsed=200.0)
    composed_a100 = _records(manifest, elapsed=100.0)
    primary_a10g = _records(manifest, elapsed=20.0)
    primary_a100 = _records(manifest, elapsed=10.0)

    report = compare_timefuse_matrix_runs.build_comparison(
        manifest,
        composed_a10g,
        composed_a100,
        manifest_sha256="a" * 64,
        method_revision=METHOD_REVISION,
        seed=2021,
        absolute_tolerance=1e-5,
        relative_tolerance=1e-4,
        a100_matrix_seconds=800.0,
        a10g_runtime_records=primary_a10g,
        a100_runtime_records=primary_a100,
        runtime_source="first_pass_summaries",
    )

    runtime = report["runtime"]
    assert runtime["gates"]["all_passed"]
    assert runtime["comparison_scope"] == "first_pass_summaries"
    assert runtime["a10g_4gpu"]["sum_cell_worker_seconds"] == pytest.approx(
        585 * 20.0
    )
    assert runtime["a100_8gpu"]["sum_cell_worker_seconds"] == pytest.approx(
        585 * 10.0
    )
    assert report["cells"][0]["elapsed_seconds"] == {
        "a10g": 20.0,
        "a100": 10.0,
    }
    assert runtime["both_trained_comparable_subset"][
        "ratio_of_summed_cell_seconds"
    ] == pytest.approx(2.0)


def test_comparison_rejects_tolerance_or_required_field_drift():
    manifest = load_manifest("docs/timefuse_experiment_manifest.jsonl")
    a10g = _records(manifest, elapsed=20.0)
    a100 = _records(manifest, elapsed=10.0)
    arguments = {
        "manifest_sha256": "a" * 64,
        "method_revision": METHOD_REVISION,
        "seed": 2021,
        "absolute_tolerance": 1e-5,
        "relative_tolerance": 1e-4,
        "a100_matrix_seconds": 800.0,
    }

    with pytest.raises(ValueError, match="tolerances are frozen"):
        compare_timefuse_matrix_runs.build_comparison(
            manifest,
            a10g,
            a100,
            **{**arguments, "relative_tolerance": 1e-3},
        )

    cell_id = manifest[0]["id"]
    broken = copy.deepcopy(a100)
    del broken[cell_id]["result"]["all_test_metrics_improve"]
    with pytest.raises(ValueError, match="selection is incomplete"):
        compare_timefuse_matrix_runs.build_comparison(
            manifest,
            a10g,
            broken,
            **arguments,
        )

    broken = copy.deepcopy(a100)
    del broken[cell_id]["result"]["checkpoint"]["mode"]
    with pytest.raises(ValueError, match="checkpoint mode is missing"):
        compare_timefuse_matrix_runs.build_comparison(
            manifest,
            a10g,
            broken,
            **arguments,
        )


def test_comparison_reads_explicit_composed_artifact_selection(tmp_path):
    manifest = load_manifest("docs/timefuse_experiment_manifest.jsonl")[:2]
    rows = []
    for index, cell in enumerate(manifest):
        artifact = tmp_path / f"cell-{index}"
        artifact.mkdir()
        record = _records([cell], elapsed=10.0)[cell["id"]]
        (artifact / "status.json").write_text(
            json.dumps(record["status"]),
            encoding="utf-8",
        )
        (artifact / "result.json").write_text(
            json.dumps(record["result"]),
            encoding="utf-8",
        )
        rows.append(
            {
                "cell_id": cell["id"],
                "state": "completed",
                "artifact_dir": str(artifact),
                "numerical_recovery": {
                    "profile": "exact",
                    "selection_uses_metric_quality": False,
                },
            }
        )
    summary = tmp_path / "composed.json"
    summary.write_text(
        json.dumps(
            {
                "source_revision": METHOD_REVISION,
                "cell_states": rows,
            }
        ),
        encoding="utf-8",
    )

    records = compare_timefuse_matrix_runs.records_from_composed_summary(
        summary,
        manifest,
        METHOD_REVISION,
        2021,
    )

    assert set(records) == {cell["id"] for cell in manifest}
    assert all(
        record["numerical_recovery"]["profile"] == "exact"
        for record in records.values()
    )


def test_comparison_reads_terminal_first_pass_runtime_summary(tmp_path):
    manifest = load_manifest("docs/timefuse_experiment_manifest.jsonl")[:2]
    rows = []
    for index, cell in enumerate(manifest):
        artifact = tmp_path / f"runtime-cell-{index}"
        artifact.mkdir()
        record = _records([cell], elapsed=10.0)[cell["id"]]
        (artifact / "status.json").write_text(
            json.dumps(record["status"]),
            encoding="utf-8",
        )
        (artifact / "result.json").write_text(
            json.dumps(record["result"]),
            encoding="utf-8",
        )
        rows.append(
            {
                "cell_id": cell["id"],
                "state": "incomplete" if index == 0 else "completed",
                "artifact_dir": str(artifact),
                "numerical_integrity_error": (
                    "test_corrected.mse is non-finite"
                    if index == 0
                    else None
                ),
            }
        )
    summary = tmp_path / "runtime-summary.json"
    summary.write_text(
        json.dumps(
            {
                "source_revision": METHOD_REVISION,
                "cell_states": rows,
            }
        ),
        encoding="utf-8",
    )

    records = compare_timefuse_matrix_runs.records_from_runtime_summary(
        summary,
        manifest,
        METHOD_REVISION,
        2021,
    )

    assert records[manifest[0]["id"]]["summary_state"] == "incomplete"
    assert records[manifest[1]["id"]]["summary_state"] == "completed"
