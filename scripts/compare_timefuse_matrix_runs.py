"""Compare full TimeFuse matrix results and runtime across two GPU systems."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from scripts.greenland_common import (
    FULL_MATRIX_CELLS,
    METHOD_REVISION,
    validate_full_matrix_manifest,
)
from ts_rag.matrix import load_manifest, save_json_atomic


FROZEN_ABSOLUTE_TOLERANCE = 1e-5
FROZEN_RELATIVE_TOLERANCE = 1e-4
_GIT_REVISION = re.compile(r"[0-9a-f]{7,40}")


def _load_json(path):
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid JSON artifact: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return payload


def _revision_matches(actual, expected):
    actual = str(actual or "")
    expected = str(expected or "")
    return bool(
        _GIT_REVISION.fullmatch(actual)
        and _GIT_REVISION.fullmatch(expected)
        and (actual.startswith(expected) or expected.startswith(actual))
    )


def _finite_number(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _validate_record_identity(record, cell_id, method_revision, seed):
    if not isinstance(record, dict):
        raise ValueError(f"Comparison record must be an object for {cell_id}")
    status = record.get("status")
    result = record.get("result")
    if not isinstance(status, dict) or not isinstance(result, dict):
        raise ValueError(f"Comparison record artifacts are invalid for {cell_id}")
    result_cell = result.get("cell")
    checkpoint = result.get("checkpoint")
    if (
        status.get("cell_id") != cell_id
        or status.get("status") != "completed"
        or status.get("seed") != seed
        or status.get("smoke") is not False
        or status.get("export_only", False) is not False
        or result.get("export_only", False) is not False
        or not _revision_matches(status.get("source_revision"), method_revision)
        or not isinstance(result_cell, dict)
        or result_cell.get("id") != cell_id
    ):
        raise ValueError(f"Comparison record identity drifted for {cell_id}")
    if (
        not _finite_number(status.get("started_unix"))
        or status["started_unix"] <= 0
        or not _finite_number(status.get("elapsed_seconds"))
        or status["elapsed_seconds"] <= 0
    ):
        raise ValueError(f"Comparison timing is invalid for {cell_id}")
    checkpoint_mode = (
        checkpoint.get("mode") if isinstance(checkpoint, dict) else None
    )
    if not isinstance(checkpoint_mode, str) or not checkpoint_mode:
        raise ValueError(f"Comparison checkpoint mode is missing for {cell_id}")
    return status, result


def _validate_result_record(record, cell, method_revision, seed):
    cell_id = cell["id"]
    _, result = _validate_record_identity(
        record,
        cell_id,
        method_revision,
        seed,
    )
    validation = result.get("validation")
    selected = (
        validation.get("selected") if isinstance(validation, dict) else None
    )
    if (
        not isinstance(selected, dict)
        or not isinstance(selected.get("method"), str)
        or not selected["method"]
        or not isinstance(selected.get("params"), dict)
        or not isinstance(result.get("all_test_metrics_improve"), bool)
    ):
        raise ValueError(f"Comparison selection is incomplete for {cell_id}")
    for field in ("test_baseline", "test_corrected"):
        metrics = result.get(field)
        if not isinstance(metrics, dict):
            raise ValueError(f"Comparison metrics are missing for {cell_id}")
        for metric in cell["metrics"]:
            if not _finite_number(metrics.get(metric)):
                raise ValueError(
                    f"Comparison metric {field}.{metric} is invalid for "
                    f"{cell_id}"
                )


def _validate_frozen_tolerances(absolute_tolerance, relative_tolerance):
    if (
        absolute_tolerance != FROZEN_ABSOLUTE_TOLERANCE
        or relative_tolerance != FROZEN_RELATIVE_TOLERANCE
    ):
        raise ValueError(
            "Cross-hardware tolerances are frozen at "
            f"atol={FROZEN_ABSOLUTE_TOLERANCE}, "
            f"rtol={FROZEN_RELATIVE_TOLERANCE}"
        )


def discover_completed_results(root, manifest, method_revision, seed):
    root = Path(root).resolve()
    expected_ids = {cell["id"] for cell in manifest}
    candidates = defaultdict(list)
    for status_path in root.glob("**/status.json"):
        status = _load_json(status_path)
        cell_id = status.get("cell_id")
        if (
            cell_id not in expected_ids
            or status.get("status") != "completed"
            or int(status.get("seed", -1)) != seed
            or status.get("smoke") is not False
            or bool(status.get("export_only", False))
            or not _revision_matches(
                status.get("source_revision"),
                method_revision,
            )
        ):
            continue
        result_path = status_path.with_name("result.json")
        if not result_path.is_file():
            continue
        result = _load_json(result_path)
        result_cell_id = result.get("cell", {}).get("id")
        if result_cell_id is not None and result_cell_id != cell_id:
            raise ValueError(
                f"Result/status cell ID mismatch below {status_path.parent}"
            )
        candidates[cell_id].append(
            {
                "artifact_dir": str(status_path.parent),
                "status_path": str(status_path),
                "result_path": str(result_path),
                "status": status,
                "result": result,
            }
        )
    selected = {
        cell_id: max(
            records,
            key=lambda record: float(
                record["status"].get("started_unix", 0.0)
            ),
        )
        for cell_id, records in candidates.items()
    }
    return selected


def records_from_composed_summary(path, manifest, method_revision, seed):
    summary = _load_json(path)
    if not _revision_matches(
        summary.get("source_revision"),
        method_revision,
    ):
        raise ValueError("Composed summary method revision drifted")
    manifest = list(manifest)
    expected_ids = [cell["id"] for cell in manifest]
    states = summary.get("cell_states")
    if not isinstance(states, list):
        raise ValueError("Composed summary has no cell_states")
    indexed = {row.get("cell_id"): row for row in states}
    if (
        len(indexed) != len(states)
        or set(indexed) != set(expected_ids)
        or any(row.get("state") != "completed" for row in states)
    ):
        raise ValueError(
            "Composed summary does not contain a complete manifest scope"
        )
    records = {}
    for cell_id in expected_ids:
        artifact_dir = Path(indexed[cell_id]["artifact_dir"])
        status_path = artifact_dir / "status.json"
        result_path = artifact_dir / "result.json"
        status = _load_json(status_path)
        result = _load_json(result_path)
        if (
            status.get("cell_id") != cell_id
            or status.get("status") != "completed"
            or status.get("seed") != seed
            or status.get("smoke") is not False
            or status.get("export_only", False) is not False
            or not _revision_matches(
                status.get("source_revision"),
                method_revision,
            )
            or (
                result.get("cell", {}).get("id") is not None
                and result["cell"]["id"] != cell_id
            )
        ):
            raise ValueError(
                f"Composed artifact identity drifted for {cell_id}"
            )
        records[cell_id] = {
            "artifact_dir": str(artifact_dir),
            "status_path": str(status_path),
            "result_path": str(result_path),
            "status": status,
            "result": result,
            "numerical_recovery": indexed[cell_id].get(
                "numerical_recovery"
            ),
        }
    return records


def records_from_runtime_summary(path, manifest, method_revision, seed):
    summary = _load_json(path)
    if not _revision_matches(
        summary.get("source_revision"),
        method_revision,
    ):
        raise ValueError("Runtime summary method revision drifted")
    manifest = list(manifest)
    expected_ids = [cell["id"] for cell in manifest]
    states = summary.get("cell_states")
    if not isinstance(states, list):
        raise ValueError("Runtime summary has no cell_states")
    indexed = {row.get("cell_id"): row for row in states}
    if (
        len(indexed) != len(states)
        or set(indexed) != set(expected_ids)
        or any(
            row.get("state") not in {"completed", "incomplete"}
            for row in states
        )
    ):
        raise ValueError(
            "Runtime summary does not contain a complete terminal manifest scope"
        )
    records = {}
    for cell_id in expected_ids:
        artifact_dir = Path(indexed[cell_id]["artifact_dir"])
        status_path = artifact_dir / "status.json"
        result_path = artifact_dir / "result.json"
        record = {
            "artifact_dir": str(artifact_dir),
            "status_path": str(status_path),
            "result_path": str(result_path),
            "status": _load_json(status_path),
            "result": _load_json(result_path),
            "summary_state": indexed[cell_id]["state"],
            "numerical_integrity_error": indexed[cell_id].get(
                "numerical_integrity_error"
            ),
        }
        _validate_record_identity(
            record,
            cell_id,
            method_revision,
            seed,
        )
        records[cell_id] = record
    return records


def _relative_error(left, right):
    return abs(left - right) / max(abs(left), abs(right), 1e-12)


def _metric_comparison(left, right, absolute_tolerance, relative_tolerance):
    if not all(_finite_number(value) for value in (left, right)):
        raise ValueError("Compared matrix metrics must be finite numbers")
    absolute_error = abs(float(left) - float(right))
    relative_error = _relative_error(float(left), float(right))
    threshold = absolute_tolerance + relative_tolerance * max(
        abs(float(left)),
        abs(float(right)),
    )
    return {
        "a10g": float(left),
        "a100": float(right),
        "absolute_error": absolute_error,
        "relative_error": relative_error,
        "tolerance_threshold": threshold,
        "within_tolerance": absolute_error <= threshold,
    }


def _percentile(values, quantile):
    ordered = sorted(values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _distribution(values, *, geometric=False):
    values = [float(value) for value in values]
    if not values:
        empty = {
            key: None
            for key in (
                "minimum",
                "p10",
                "median",
                "mean",
                "geometric_mean",
                "p90",
                "p95",
                "maximum",
            )
        }
        return {"count": 0, **empty}
    return {
        "count": len(values),
        "minimum": min(values),
        "p10": _percentile(values, 0.10),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "geometric_mean": (
            math.exp(statistics.fmean(math.log(value) for value in values))
            if geometric and all(value > 0 for value in values)
            else None
        ),
        "p90": _percentile(values, 0.90),
        "p95": _percentile(values, 0.95),
        "maximum": max(values),
    }


def _comparison_summary(entries):
    return {
        "count": len(entries),
        "within_tolerance": sum(
            entry["within_tolerance"] for entry in entries
        ),
        "within_tolerance_rate": (
            sum(entry["within_tolerance"] for entry in entries) / len(entries)
            if entries
            else 0.0
        ),
        "absolute_error": _distribution(
            [entry["absolute_error"] for entry in entries]
        ),
        "relative_error": _distribution(
            [entry["relative_error"] for entry in entries]
        ),
    }


def _runtime_summary(records, gpu_count, reported_matrix_seconds=None):
    elapsed = [
        float(record["status"]["elapsed_seconds"])
        for record in records.values()
    ]
    intervals = [
        (
            float(record["status"]["started_unix"]),
            float(record["status"]["started_unix"])
            + float(record["status"]["elapsed_seconds"]),
        )
        for record in records.values()
    ]
    envelope = (
        max(finished for _, finished in intervals)
        - min(started for started, _ in intervals)
        if intervals
        else None
    )
    if reported_matrix_seconds is not None and (
        not _finite_number(reported_matrix_seconds)
        or reported_matrix_seconds <= 0
    ):
        raise ValueError("Reported matrix elapsed time must be finite and positive")
    reported_matrix_seconds = (
        float(reported_matrix_seconds)
        if reported_matrix_seconds is not None
        else None
    )
    matrix_seconds = reported_matrix_seconds or envelope
    worker_seconds = sum(elapsed)
    mode_counts = Counter(
        str(record["result"].get("checkpoint", {}).get("mode", "unknown"))
        for record in records.values()
    )
    state_counts = Counter(
        str(record.get("summary_state", "completed"))
        for record in records.values()
    )
    return {
        "completed_cells": len(records),
        "terminal_cells": len(records),
        "gpu_count": gpu_count,
        "summary_state_counts": dict(sorted(state_counts.items())),
        "checkpoint_mode_counts": dict(sorted(mode_counts.items())),
        "sum_cell_worker_seconds": worker_seconds,
        "sum_cell_worker_hours": worker_seconds / 3600.0,
        "cell_elapsed_seconds": _distribution(elapsed),
        "observed_cell_envelope_seconds": envelope,
        "reported_matrix_seconds": reported_matrix_seconds,
        "throughput_basis": (
            "reported_matrix_seconds"
            if reported_matrix_seconds is not None
            else "observed_cell_envelope_seconds"
        ),
        "throughput_basis_seconds": matrix_seconds,
        "cells_per_hour": (
            len(records) * 3600.0 / matrix_seconds
            if matrix_seconds
            else None
        ),
        "cells_per_gpu_hour": (
            len(records) * 3600.0 / matrix_seconds / gpu_count
            if matrix_seconds
            else None
        ),
        "worker_occupancy_fraction": (
            worker_seconds / (matrix_seconds * gpu_count)
            if matrix_seconds
            else None
        ),
    }


def _speedup_summary(a10g, a100, cell_ids):
    rows = []
    for cell_id in sorted(cell_ids):
        left = float(a10g[cell_id]["status"]["elapsed_seconds"])
        right = float(a100[cell_id]["status"]["elapsed_seconds"])
        if left <= 0 or right <= 0:
            continue
        rows.append(
            {
                "cell_id": cell_id,
                "a10g_seconds": left,
                "a100_seconds": right,
                "a100_speedup_over_a10g": left / right,
            }
        )
    return {
        "cell_count": len(rows),
        "sum_a10g_seconds": sum(row["a10g_seconds"] for row in rows),
        "sum_a100_seconds": sum(row["a100_seconds"] for row in rows),
        "ratio_of_summed_cell_seconds": (
            sum(row["a10g_seconds"] for row in rows)
            / sum(row["a100_seconds"] for row in rows)
            if rows
            else None
        ),
        "per_cell_a100_speedup": _distribution(
            [row["a100_speedup_over_a10g"] for row in rows],
            geometric=True,
        ),
        "slowest_a100_relative_cells": sorted(
            rows,
            key=lambda row: row["a100_speedup_over_a10g"],
        )[:20],
        "fastest_a100_relative_cells": sorted(
            rows,
            key=lambda row: row["a100_speedup_over_a10g"],
            reverse=True,
        )[:20],
    }


def build_comparison(
    manifest,
    a10g,
    a100,
    *,
    manifest_sha256,
    method_revision,
    seed,
    absolute_tolerance,
    relative_tolerance,
    a100_matrix_seconds=None,
    a10g_runtime_records=None,
    a100_runtime_records=None,
    runtime_source="comparison_artifacts",
):
    _validate_frozen_tolerances(absolute_tolerance, relative_tolerance)
    if runtime_source not in {
        "comparison_artifacts",
        "first_pass_summaries",
    }:
        raise ValueError(f"Unsupported runtime source: {runtime_source}")
    manifest = list(manifest)
    expected_ids = {cell["id"] for cell in manifest}
    if len(expected_ids) != len(manifest):
        raise ValueError("Comparison manifest contains duplicate cell IDs")
    for cell in manifest:
        cell_id = cell["id"]
        if cell_id in a10g:
            _validate_result_record(
                a10g[cell_id],
                cell,
                method_revision,
                seed,
            )
        if cell_id in a100:
            _validate_result_record(
                a100[cell_id],
                cell,
                method_revision,
                seed,
            )
    a10g_runtime_records = (
        a10g if a10g_runtime_records is None else a10g_runtime_records
    )
    a100_runtime_records = (
        a100 if a100_runtime_records is None else a100_runtime_records
    )
    for cell_id in expected_ids & set(a10g_runtime_records):
        _validate_record_identity(
            a10g_runtime_records[cell_id],
            cell_id,
            method_revision,
            seed,
        )
    for cell_id in expected_ids & set(a100_runtime_records):
        _validate_record_identity(
            a100_runtime_records[cell_id],
            cell_id,
            method_revision,
            seed,
        )
    paired_ids = expected_ids & set(a10g) & set(a100)
    runtime_paired_ids = (
        expected_ids
        & set(a10g_runtime_records)
        & set(a100_runtime_records)
    )
    baseline_entries = []
    corrected_entries = []
    cells = []
    method_agreements = 0
    parameter_agreements = 0
    classification_agreements = 0
    for cell in manifest:
        cell_id = cell["id"]
        if cell_id not in paired_ids:
            continue
        left = a10g[cell_id]["result"]
        right = a100[cell_id]["result"]
        selected_left = left.get("validation", {}).get("selected", {})
        selected_right = right.get("validation", {}).get("selected", {})
        runtime_left = a10g_runtime_records.get(cell_id)
        runtime_right = a100_runtime_records.get(cell_id)
        method_agrees = selected_left.get("method") == selected_right.get(
            "method"
        )
        parameters_agree = selected_left.get("params", {}) == selected_right.get(
            "params",
            {},
        )
        classification_agrees = left.get(
            "all_test_metrics_improve"
        ) == right.get("all_test_metrics_improve")
        method_agreements += method_agrees
        parameter_agreements += parameters_agree
        classification_agreements += classification_agrees
        metrics = {}
        for metric in cell["metrics"]:
            baseline = _metric_comparison(
                left.get("test_baseline", {}).get(metric),
                right.get("test_baseline", {}).get(metric),
                absolute_tolerance,
                relative_tolerance,
            )
            corrected = _metric_comparison(
                left.get("test_corrected", {}).get(metric),
                right.get("test_corrected", {}).get(metric),
                absolute_tolerance,
                relative_tolerance,
            )
            baseline_entries.append(
                {"cell_id": cell_id, "metric": metric, **baseline}
            )
            corrected_entries.append(
                {"cell_id": cell_id, "metric": metric, **corrected}
            )
            metrics[metric] = {
                "baseline": baseline,
                "corrected": corrected,
            }
        cells.append(
            {
                "cell_id": cell_id,
                "task_family": cell["task_family"],
                "dataset": cell["dataset"],
                "model": cell["model"],
                "pred_len": cell["pred_len"],
                "selected_method": {
                    "a10g": selected_left.get("method"),
                    "a100": selected_right.get("method"),
                    "agrees": method_agrees,
                },
                "selected_params": {
                    "a10g": selected_left.get("params", {}),
                    "a100": selected_right.get("params", {}),
                    "agrees": parameters_agree,
                },
                "strict_improvement": {
                    "a10g": left["all_test_metrics_improve"],
                    "a100": right["all_test_metrics_improve"],
                    "agrees": classification_agrees,
                },
                "checkpoint_mode": {
                    "a10g": (
                        runtime_left["result"]["checkpoint"]["mode"]
                        if runtime_left
                        else None
                    ),
                    "a100": (
                        runtime_right["result"]["checkpoint"]["mode"]
                        if runtime_right
                        else None
                    ),
                },
                "elapsed_seconds": {
                    "a10g": (
                        runtime_left["status"]["elapsed_seconds"]
                        if runtime_left
                        else None
                    ),
                    "a100": (
                        runtime_right["status"]["elapsed_seconds"]
                        if runtime_right
                        else None
                    ),
                },
                "runtime_source": runtime_source,
                "metrics": metrics,
            }
        )
    expected_metrics = sum(len(cell["metrics"]) for cell in manifest)
    baseline_summary = _comparison_summary(baseline_entries)
    corrected_summary = _comparison_summary(corrected_entries)
    scope_complete = (
        len(manifest) == FULL_MATRIX_CELLS
        and set(a10g) == expected_ids
        and set(a100) == expected_ids
        and len(baseline_entries) == expected_metrics
        and len(corrected_entries) == expected_metrics
    )
    selection = {
        "paired_cells": len(paired_ids),
        "selected_method_agreements": method_agreements,
        "selected_method_agreement_rate": (
            method_agreements / len(paired_ids) if paired_ids else 0.0
        ),
        "selected_parameter_agreements": parameter_agreements,
        "selected_parameter_agreement_rate": (
            parameter_agreements / len(paired_ids) if paired_ids else 0.0
        ),
        "strict_improvement_classification_agreements": (
            classification_agreements
        ),
        "strict_improvement_classification_agreement_rate": (
            classification_agreements / len(paired_ids)
            if paired_ids
            else 0.0
        ),
    }
    gates = {
        "complete_585_cell_scope": scope_complete,
        "all_baseline_metrics_within_tolerance": (
            baseline_summary["count"] == expected_metrics
            and baseline_summary["within_tolerance"] == expected_metrics
        ),
        "all_corrected_metrics_within_tolerance": (
            corrected_summary["count"] == expected_metrics
            and corrected_summary["within_tolerance"] == expected_metrics
        ),
        "all_selected_methods_agree": (
            len(paired_ids) == len(manifest)
            and method_agreements == len(manifest)
        ),
        "all_selected_parameters_agree": (
            len(paired_ids) == len(manifest)
            and parameter_agreements == len(manifest)
        ),
        "all_strict_improvement_classifications_agree": (
            len(paired_ids) == len(manifest)
            and classification_agreements == len(manifest)
        ),
    }
    runtime_scope_complete = (
        len(manifest) == FULL_MATRIX_CELLS
        and set(a10g_runtime_records) == expected_ids
        and set(a100_runtime_records) == expected_ids
    )
    a10g_runtime = _runtime_summary(a10g_runtime_records, 4)
    a100_runtime = _runtime_summary(
        a100_runtime_records,
        8,
        a100_matrix_seconds,
    )
    common_trained = set()
    for cell_id in runtime_paired_ids:
        a10g_mode = a10g_runtime_records[cell_id]["result"]["checkpoint"][
            "mode"
        ]
        a100_mode = a100_runtime_records[cell_id]["result"]["checkpoint"][
            "mode"
        ]
        if a10g_mode == "trained" and a100_mode == "trained":
            common_trained.add(cell_id)
    runtime_gates = {
        "complete_585_cell_scope": runtime_scope_complete,
        "all_cells_have_finite_positive_timing": (
            runtime_scope_complete
            and a10g_runtime["cell_elapsed_seconds"]["count"]
            == FULL_MATRIX_CELLS
            and a100_runtime["cell_elapsed_seconds"]["count"]
            == FULL_MATRIX_CELLS
        ),
        "a100_reported_matrix_seconds_present": (
            a100_runtime["reported_matrix_seconds"] is not None
        ),
    }
    if runtime_source == "first_pass_summaries":
        interpretation = (
            "Runtime uses the first-pass 585-cell artifacts on both systems; "
            "recovery-composed artifacts are used only for numerical "
            "consistency. The A10G cell envelope includes any Studio "
            "interruption. Use the both-trained subset for hardware guidance "
            "because loaded release checkpoints do not perform the same work "
            "as fresh training."
        )
    else:
        interpretation = (
            "The A10G cell envelope includes any Studio interruption. "
            "Use the both-trained subset for hardware guidance because loaded "
            "release checkpoints do not perform the same work as fresh "
            "training."
        )
    runtime = {
        "comparison_scope": runtime_source,
        "gates": {
            **runtime_gates,
            "all_passed": all(runtime_gates.values()),
        },
        "a10g_4gpu": a10g_runtime,
        "a100_8gpu": a100_runtime,
        "observed_matrix_makespan_speedup": (
            a10g_runtime["throughput_basis_seconds"]
            / a100_runtime["throughput_basis_seconds"]
            if a10g_runtime["throughput_basis_seconds"]
            and a100_runtime["throughput_basis_seconds"]
            else None
        ),
        "all_paired_cells": _speedup_summary(
            a10g_runtime_records,
            a100_runtime_records,
            runtime_paired_ids,
        ),
        "both_trained_comparable_subset": _speedup_summary(
            a10g_runtime_records,
            a100_runtime_records,
            common_trained,
        ),
        "interpretation": interpretation,
    }
    return {
        "schema_version": 1,
        "generated_unix": time.time(),
        "method_revision": method_revision,
        "seed": seed,
        "manifest": {
            "sha256": manifest_sha256,
            "expected_cells": len(manifest),
            "expected_metric_values_per_run": expected_metrics,
        },
        "tolerances": {
            "formula": (
                "abs(a10g-a100) <= absolute + relative * "
                "max(abs(a10g), abs(a100))"
            ),
            "absolute": absolute_tolerance,
            "relative": relative_tolerance,
        },
        "scope": {
            "a10g_completed_cells": len(a10g),
            "a100_completed_cells": len(a100),
            "paired_cells": len(paired_ids),
            "a10g_runtime_cells": len(a10g_runtime_records),
            "a100_runtime_cells": len(a100_runtime_records),
            "runtime_paired_cells": len(runtime_paired_ids),
            "missing_from_a10g": sorted(expected_ids - set(a10g)),
            "missing_from_a100": sorted(expected_ids - set(a100)),
            "missing_runtime_from_a10g": sorted(
                expected_ids - set(a10g_runtime_records)
            ),
            "missing_runtime_from_a100": sorted(
                expected_ids - set(a100_runtime_records)
            ),
        },
        "consistency_gates": {
            **gates,
            "all_passed": all(gates.values()),
        },
        "metric_consistency": {
            "baseline": baseline_summary,
            "corrected": corrected_summary,
        },
        "selection_and_outcome_consistency": selection,
        "runtime": runtime,
        "largest_baseline_discrepancies": sorted(
            baseline_entries,
            key=lambda entry: entry["relative_error"],
            reverse=True,
        )[:20],
        "largest_corrected_discrepancies": sorted(
            corrected_entries,
            key=lambda entry: entry["relative_error"],
            reverse=True,
        )[:20],
        "cells": cells,
    }


def render_markdown(report):
    gates = report["consistency_gates"]
    metrics = report["metric_consistency"]
    selection = report["selection_and_outcome_consistency"]
    runtime = report["runtime"]
    a10g = runtime["a10g_4gpu"]
    a100 = runtime["a100_8gpu"]
    trained = runtime["both_trained_comparable_subset"]

    def value(number, digits=3):
        return "n/a" if number is None else f"{number:.{digits}f}"

    lines = [
        "# TimeFuse A10G vs A100 Full-Matrix Comparison",
        "",
        f"- Method revision: `{report['method_revision']}`",
        f"- Manifest SHA-256: `{report['manifest']['sha256']}`",
        (
            "- Numerical tolerance: "
            f"`atol={report['tolerances']['absolute']}`, "
            f"`rtol={report['tolerances']['relative']}`"
        ),
        f"- Overall consistency gate: **{'PASS' if gates['all_passed'] else 'FAIL'}**",
        (
            "- Runtime evidence gate: "
            f"**{'PASS' if runtime['gates']['all_passed'] else 'FAIL'}**"
        ),
        f"- Runtime source: `{runtime['comparison_scope']}`",
        "",
        "## Consistency",
        "",
        "| Check | Result |",
        "|---|---:|",
        f"| Complete 585-cell scope | {gates['complete_585_cell_scope']} |",
        (
            "| Baseline metric values within tolerance | "
            f"{metrics['baseline']['within_tolerance']}/"
            f"{metrics['baseline']['count']} |"
        ),
        (
            "| Corrected metric values within tolerance | "
            f"{metrics['corrected']['within_tolerance']}/"
            f"{metrics['corrected']['count']} |"
        ),
        (
            "| Selected method agreement | "
            f"{selection['selected_method_agreements']}/"
            f"{selection['paired_cells']} |"
        ),
        (
            "| Selected parameter agreement | "
            f"{selection['selected_parameter_agreements']}/"
            f"{selection['paired_cells']} |"
        ),
        (
            "| Strict-improvement classification agreement | "
            f"{selection['strict_improvement_classification_agreements']}/"
            f"{selection['paired_cells']} |"
        ),
        "",
        "## Runtime",
        "",
        "| Measure | 4x A10G | 8x A100 |",
        "|---|---:|---:|",
        (
            "| Runtime cell attempts | "
            f"{a10g['terminal_cells']} | {a100['terminal_cells']} |"
        ),
        (
            "| Observed cell envelope (hours) | "
            f"{value(a10g['observed_cell_envelope_seconds'] / 3600 if a10g['observed_cell_envelope_seconds'] else None)} | "
            f"{value(a100['observed_cell_envelope_seconds'] / 3600 if a100['observed_cell_envelope_seconds'] else None)} |"
        ),
        (
            "| Reported matrix time (hours) | "
            f"{value(a10g['reported_matrix_seconds'] / 3600 if a10g['reported_matrix_seconds'] else None)} | "
            f"{value(a100['reported_matrix_seconds'] / 3600 if a100['reported_matrix_seconds'] else None)} |"
        ),
        (
            "| Throughput basis (hours) | "
            f"{value(a10g['throughput_basis_seconds'] / 3600 if a10g['throughput_basis_seconds'] else None)} | "
            f"{value(a100['throughput_basis_seconds'] / 3600 if a100['throughput_basis_seconds'] else None)} |"
        ),
        (
            "| Cells/hour | "
            f"{value(a10g['cells_per_hour'])} | "
            f"{value(a100['cells_per_hour'])} |"
        ),
        (
            "| Sum cell-worker hours | "
            f"{value(a10g['sum_cell_worker_hours'])} | "
            f"{value(a100['sum_cell_worker_hours'])} |"
        ),
        "",
        (
            f"Observed matrix makespan speedup: "
            f"**{value(runtime['observed_matrix_makespan_speedup'])}x**."
        ),
        (
            f"Both-trained subset: {trained['cell_count']} cells; "
            f"ratio of summed cell time "
            f"**{value(trained['ratio_of_summed_cell_seconds'])}x**; "
            f"median per-cell A100 speedup "
            f"**{value(trained['per_cell_a100_speedup']['median'])}x**."
        ),
        "",
        runtime["interpretation"],
        "",
    ]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Compare 4x A10G and 8x A100 TimeFuse full matrices"
    )
    parser.add_argument(
        "--manifest",
        default="docs/timefuse_experiment_manifest.jsonl",
    )
    parser.add_argument("--a10g-output-root")
    parser.add_argument("--a100-output-root")
    parser.add_argument("--a10g-summary")
    parser.add_argument("--a100-summary")
    parser.add_argument("--a10g-runtime-summary")
    parser.add_argument("--a100-runtime-summary")
    parser.add_argument("--a100-final-status")
    parser.add_argument("--method-revision", default=METHOD_REVISION)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument(
        "--absolute-tolerance",
        type=float,
        default=FROZEN_ABSOLUTE_TOLERANCE,
    )
    parser.add_argument(
        "--relative-tolerance",
        type=float,
        default=FROZEN_RELATIVE_TOLERANCE,
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--markdown")
    args = parser.parse_args(argv)
    try:
        _validate_frozen_tolerances(
            args.absolute_tolerance,
            args.relative_tolerance,
        )
    except ValueError as error:
        parser.error(str(error))
    if bool(args.a10g_output_root) == bool(args.a10g_summary):
        parser.error(
            "Use exactly one of --a10g-output-root or --a10g-summary"
        )
    if bool(args.a100_output_root) == bool(args.a100_summary):
        parser.error(
            "Use exactly one of --a100-output-root or --a100-summary"
        )
    if bool(args.a10g_runtime_summary) != bool(args.a100_runtime_summary):
        parser.error(
            "Use --a10g-runtime-summary and --a100-runtime-summary together"
        )

    evidence = validate_full_matrix_manifest(args.manifest)
    manifest = load_manifest(args.manifest)
    a10g = (
        discover_completed_results(
            args.a10g_output_root,
            manifest,
            args.method_revision,
            args.seed,
        )
        if args.a10g_output_root
        else records_from_composed_summary(
            args.a10g_summary,
            manifest,
            args.method_revision,
            args.seed,
        )
    )
    a100 = (
        discover_completed_results(
            args.a100_output_root,
            manifest,
            args.method_revision,
            args.seed,
        )
        if args.a100_output_root
        else records_from_composed_summary(
            args.a100_summary,
            manifest,
            args.method_revision,
            args.seed,
        )
    )
    a10g_runtime_records = (
        records_from_runtime_summary(
            args.a10g_runtime_summary,
            manifest,
            args.method_revision,
            args.seed,
        )
        if args.a10g_runtime_summary
        else a10g
    )
    a100_runtime_records = (
        records_from_runtime_summary(
            args.a100_runtime_summary,
            manifest,
            args.method_revision,
            args.seed,
        )
        if args.a100_runtime_summary
        else a100
    )
    a100_matrix_seconds = None
    if args.a100_final_status:
        final_status = _load_json(args.a100_final_status)
        if (
            final_status.get("method_revision") != args.method_revision
            or final_status.get("status") != "succeeded"
        ):
            raise ValueError("A100 final status identity or outcome is invalid")
        a100_matrix_seconds = final_status.get("matrix_elapsed_seconds")
    report = build_comparison(
        manifest,
        a10g,
        a100,
        manifest_sha256=evidence["sha256"],
        method_revision=args.method_revision,
        seed=args.seed,
        absolute_tolerance=args.absolute_tolerance,
        relative_tolerance=args.relative_tolerance,
        a100_matrix_seconds=a100_matrix_seconds,
        a10g_runtime_records=a10g_runtime_records,
        a100_runtime_records=a100_runtime_records,
        runtime_source=(
            "first_pass_summaries"
            if args.a10g_runtime_summary
            else "comparison_artifacts"
        ),
    )
    save_json_atomic(report, args.output)
    if args.markdown:
        markdown_path = Path(args.markdown)
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = markdown_path.with_suffix(markdown_path.suffix + ".tmp")
        temporary.write_text(render_markdown(report), encoding="utf-8")
        temporary.replace(markdown_path)
    print(
        json.dumps(
            {
                "consistency_gates": report["consistency_gates"],
                "scope": report["scope"],
                "runtime": {
                    "comparison_scope": report["runtime"][
                        "comparison_scope"
                    ],
                    "gates": report["runtime"]["gates"],
                    "observed_matrix_makespan_speedup": report["runtime"][
                        "observed_matrix_makespan_speedup"
                    ],
                    "both_trained_comparable_subset": report["runtime"][
                        "both_trained_comparable_subset"
                    ],
                },
                "output": str(Path(args.output)),
                "markdown": args.markdown,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
