import json
import hashlib
import math
import os
import statistics
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path


PUBLICATION_EXPECTED_CELLS = 585


def resolve_source_revision():
    revision = os.environ.get("TIMERAF_SOURCE_REVISION")
    if revision:
        return revision
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def load_manifest(path):
    with open(path, "r", encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def save_json_atomic(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        json.dump(payload, output, indent=2, sort_keys=True)
        output.write("\n")
    os.replace(temporary, path)


def _load_json(path):
    try:
        with Path(path).open("r", encoding="utf-8") as source:
            return json.load(source)
    except (OSError, json.JSONDecodeError):
        return None


def _attempt_matches(
    attempt,
    smoke,
    seed,
    source_revision=None,
    export_only=False,
):
    status = attempt["status"]
    result = attempt["result"]
    attempt_smoke = result.get("smoke") if result else status.get("smoke")
    if attempt_smoke is not smoke or int(status.get("seed", -1)) != seed:
        return False
    attempt_export_only = bool(
        result.get("export_only") if result else status.get("export_only")
    )
    if attempt_export_only is not export_only:
        return False
    if source_revision is None:
        return True
    return status.get("source_revision") == source_revision


def discover_attempts(output_root):
    attempts = defaultdict(list)
    for status_path in Path(output_root).glob("**/status.json"):
        status = _load_json(status_path)
        if not status or not status.get("cell_id"):
            continue
        result = _load_json(status_path.with_name("result.json"))
        attempts[status["cell_id"]].append(
            {
                "artifact_dir": str(status_path.parent),
                "status": status,
                "result": result,
            }
        )
    return attempts


def _select_attempt(
    attempts,
    smoke,
    seed,
    source_revision=None,
    export_only=False,
):
    matches = [
        attempt
        for attempt in attempts
        if _attempt_matches(
            attempt,
            smoke=smoke,
            seed=seed,
            source_revision=source_revision,
            export_only=export_only,
        )
    ]
    if not matches:
        return None
    completed = [
        attempt
        for attempt in matches
        if attempt["status"].get("status") == "completed"
        and attempt["result"] is not None
    ]
    candidates = completed or matches
    return max(
        candidates,
        key=lambda attempt: float(attempt["status"].get("started_unix", 0.0)),
    )


def _metric_gains(result, metrics):
    gains = {}
    baseline = result.get("test_baseline", {})
    corrected = result.get("test_corrected", {})
    for metric in metrics:
        before = baseline.get(metric)
        after = corrected.get(metric)
        if before is None or after is None or before == 0:
            gains[metric] = None
        else:
            gains[metric] = 100.0 * (before - after) / abs(before)
    return gains


def _result_metrics_are_finite(result, metrics):
    if result.get("export_only"):
        return True
    sections = {}
    for section in ("test_baseline", "test_corrected"):
        values = result.get(section)
        if not isinstance(values, dict):
            return False
        sections[section] = values
        for metric in metrics:
            value = values.get(metric)
            if isinstance(value, bool):
                return False
            try:
                if not math.isfinite(float(value)):
                    return False
            except (TypeError, ValueError):
                return False
    for metric in metrics:
        before = float(sections["test_baseline"][metric])
        after = float(sections["test_corrected"][metric])
        if before == 0:
            continue
        gain = 100.0 * (before - after) / abs(before)
        if not math.isfinite(gain):
            return False
    return True


def _cell_state(cell, attempt):
    if attempt is None:
        return {
            "cell_id": cell["id"],
            "task_family": cell["task_family"],
            "dataset": cell["dataset"],
            "model": cell["model"],
            "pred_len": cell["pred_len"],
            "state": "pending",
        }

    status = attempt["status"]
    result = attempt["result"]
    raw_state = status.get("status", "unknown")
    numerical_integrity_error = None
    if raw_state == "completed" and result is None:
        state = "incomplete"
    elif (
        raw_state == "completed"
        and not _result_metrics_are_finite(result, cell["metrics"])
    ):
        state = "incomplete"
        numerical_integrity_error = (
            "completed result has missing or non-finite paper metrics"
        )
    elif raw_state == "completed":
        state = "completed"
    elif raw_state in {"failed", "running"}:
        state = raw_state
    else:
        state = "incomplete"

    row = {
        "cell_id": cell["id"],
        "task_family": cell["task_family"],
        "dataset": cell["dataset"],
        "model": cell["model"],
        "pred_len": cell["pred_len"],
        "state": state,
        "artifact_dir": attempt["artifact_dir"],
        "started_unix": status.get("started_unix"),
        "elapsed_seconds": status.get("elapsed_seconds"),
    }
    if numerical_integrity_error:
        row["numerical_integrity_error"] = numerical_integrity_error
    if state == "completed":
        row["all_test_metrics_improve"] = bool(
            result.get("all_test_metrics_improve", False)
        )
        row["method"] = (
            result.get("validation", {}).get("selected", {}).get("method")
        )
        row["test_baseline"] = result.get("test_baseline")
        row["test_corrected"] = result.get("test_corrected")
        row["metric_gain_percent"] = _metric_gains(result, cell["metrics"])
        valid_gains = [
            value
            for value in row["metric_gain_percent"].values()
            if value is not None
        ]
        row["minimum_metric_gain_percent"] = (
            min(valid_gains) if valid_gains else None
        )
    elif state == "failed":
        row["error_type"] = status.get("error_type")
        row["error"] = status.get("error")
    return row


def _group_summary(cell_states, key):
    groups = {}
    for value in sorted({row[key] for row in cell_states}):
        rows = [row for row in cell_states if row[key] == value]
        completed = [row for row in rows if row["state"] == "completed"]
        groups[value] = {
            "expected": len(rows),
            "completed": len(completed),
            "improved": sum(
                bool(row.get("all_test_metrics_improve")) for row in completed
            ),
            "not_improved": sum(
                not bool(row.get("all_test_metrics_improve"))
                for row in completed
            ),
            "failed": sum(row["state"] == "failed" for row in rows),
            "running": sum(row["state"] == "running" for row in rows),
            "pending": sum(row["state"] == "pending" for row in rows),
            "incomplete": sum(row["state"] == "incomplete" for row in rows),
        }
    return groups


def _gain_summary(cell_states):
    by_metric = defaultdict(list)
    for row in cell_states:
        for metric, value in row.get("metric_gain_percent", {}).items():
            if value is not None:
                by_metric[metric].append(value)
    return {
        metric: {
            "count": len(values),
            "minimum": min(values),
            "median": statistics.median(values),
            "mean": statistics.fmean(values),
            "maximum": max(values),
        }
        for metric, values in sorted(by_metric.items())
    }


def _expected_metric_counts(manifest):
    counts = Counter()
    for cell in manifest:
        counts.update(cell["metrics"])
    return dict(sorted(counts.items()))


def _metric_criterion(summary, expected_counts, require_median):
    if not expected_counts:
        return False
    for metric, expected in expected_counts.items():
        aggregate = summary.get(metric)
        if aggregate is None or aggregate["count"] != expected:
            return False
        if aggregate["mean"] <= 0:
            return False
        if require_median and aggregate["median"] <= 0:
            return False
    return True


def _publication_gate(manifest, cell_states, required_cells):
    overall_expected = len(manifest)
    overall_improved = sum(
        bool(row.get("all_test_metrics_improve")) for row in cell_states
    )
    overall_rate = (
        overall_improved / overall_expected if overall_expected else 0.0
    )
    overall_gains = _gain_summary(cell_states)

    families = {}
    for family in sorted({cell["task_family"] for cell in manifest}):
        family_manifest = [
            cell for cell in manifest if cell["task_family"] == family
        ]
        family_states = [
            row for row in cell_states if row["task_family"] == family
        ]
        expected = len(family_manifest)
        improved = sum(
            bool(row.get("all_test_metrics_improve"))
            for row in family_states
        )
        gains = _gain_summary(family_states)
        families[family] = {
            "expected": expected,
            "improved": improved,
            "improvement_rate": improved / expected if expected else 0.0,
            "metric_gain_percent": gains,
            "positive_metric_means": _metric_criterion(
                gains,
                _expected_metric_counts(family_manifest),
                require_median=False,
            ),
        }

    counts = Counter(row["state"] for row in cell_states)
    criteria = {
        "full_matrix_scope": overall_expected == required_cells,
        "matrix_complete_without_runtime_failures": (
            overall_expected > 0
            and counts["completed"] == overall_expected
            and counts["failed"] == 0
        ),
        "overall_improvement_rate": overall_rate >= 0.80,
        "task_family_improvement_rates": all(
            family["improvement_rate"] >= 0.70
            for family in families.values()
        ),
        "positive_task_family_metric_means": all(
            family["positive_metric_means"] for family in families.values()
        ),
        "positive_overall_metric_means_and_medians": _metric_criterion(
            overall_gains,
            _expected_metric_counts(manifest),
            require_median=True,
        ),
    }
    return {
        "scope": {
            "selected_cells": overall_expected,
            "required_cells": required_cells,
            "is_full_matrix": overall_expected == required_cells,
        },
        "thresholds": {
            "overall_improvement_rate": 0.80,
            "task_family_improvement_rate": 0.70,
        },
        "overall": {
            "expected": overall_expected,
            "improved": overall_improved,
            "improvement_rate": overall_rate,
            "metric_gain_percent": overall_gains,
        },
        "task_families": families,
        "criteria": criteria,
        "development_gate_passed": all(criteria.values()),
        "confirmatory_evidence_required": True,
    }


def _cell_id_sha256(cell_ids):
    payload = "".join(f"{cell_id}\n" for cell_id in sorted(cell_ids))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _revision_matches(actual, expected):
    return bool(
        actual
        and expected
        and (actual.startswith(expected) or expected.startswith(actual))
    )


def build_confirmation_summary(manifest, matrix_summary, protocol):
    """Evaluate the outcome-independent post-freeze complement of a matrix."""
    manifest = list(manifest)
    thresholds = protocol.get("thresholds", {})
    expected_thresholds = {
        "overall_strict_improvement_rate": 0.80,
        "task_family_strict_improvement_rate": 0.70,
        "positive_task_family_metric_means": True,
        "positive_overall_metric_means_and_medians": True,
        "complete_without_runtime_failures": True,
    }
    if thresholds != expected_thresholds:
        raise ValueError("Confirmation thresholds do not match the frozen gate")
    independence = protocol.get("independence", {})
    if independence != {
        "selection_uses_outcomes": False,
        "method_changes_after_boundary_allowed": False,
        "development_cells_excluded_from_confirmatory_denominators": True,
    }:
        raise ValueError("Confirmation independence contract has drifted")
    expected_full = int(protocol["full_matrix_cell_count"])
    development_count = int(protocol["development_cell_count"])
    confirmatory_count = int(protocol["confirmatory_cell_count"])
    if len(manifest) != expected_full:
        raise ValueError(
            f"Confirmation protocol requires {expected_full} manifest cells"
        )
    manifest_by_id = {cell["id"]: cell for cell in manifest}
    if len(manifest_by_id) != len(manifest):
        raise ValueError("Confirmation manifest contains duplicate cell IDs")
    states = matrix_summary.get("cell_states")
    if not isinstance(states, list):
        raise ValueError("Matrix summary requires a cell_states array")
    states_by_id = {row.get("cell_id"): row for row in states}
    if (
        len(states_by_id) != len(states)
        or set(states_by_id) != set(manifest_by_id)
    ):
        raise ValueError(
            "Matrix summary cell IDs do not exactly match the manifest"
        )
    if not _revision_matches(
        matrix_summary.get("source_revision"),
        protocol.get("method_source_revision"),
    ):
        raise ValueError("Matrix summary method revision does not match protocol")
    if development_count + confirmatory_count != expected_full:
        raise ValueError(
            "Development and confirmatory counts do not partition the matrix"
        )

    started = [
        row
        for row in states
        if row.get("started_unix") is not None
    ]
    started.sort(
        key=lambda row: (float(row["started_unix"]), row["cell_id"])
    )
    if len(started) < development_count:
        raise ValueError(
            "Matrix summary does not contain the frozen development boundary"
        )
    development_states = started[:development_count]
    development_ids = {row["cell_id"] for row in development_states}
    selection = protocol["development_selection"]
    if (
        selection.get("rule") != "first_completed_cells_by_started_unix"
        or selection.get("tie_breaker") != "cell_id"
    ):
        raise ValueError("Unsupported frozen development selection rule")
    actual_hash = _cell_id_sha256(development_ids)
    if actual_hash != selection["cell_ids_sha256"]:
        raise ValueError("Frozen development cell ID hash does not match")
    last_started = float(development_states[-1]["started_unix"])
    if not math.isclose(
        last_started,
        float(selection["last_started_unix"]),
        abs_tol=1e-6,
    ):
        raise ValueError("Frozen development boundary timestamp does not match")

    confirmatory_selection = protocol["confirmatory_selection"]
    if (
        confirmatory_selection.get("rule")
        != "full matrix complement of frozen development cells"
    ):
        raise ValueError("Unsupported frozen confirmatory selection rule")
    first_confirmatory_started = None
    boundary_gap_seconds = None
    if len(started) > development_count:
        first_confirmatory_started = float(
            started[development_count]["started_unix"]
        )
        expected_first = float(
            confirmatory_selection["first_started_unix"]
        )
        if not math.isclose(
            first_confirmatory_started,
            expected_first,
            abs_tol=1e-6,
        ):
            raise ValueError(
                "First confirmatory boundary timestamp does not match"
            )
        boundary_gap_seconds = first_confirmatory_started - last_started
        if boundary_gap_seconds < float(
            confirmatory_selection["minimum_boundary_gap_seconds"]
        ):
            raise ValueError("Development/confirmation boundary gap is too small")

    confirmatory_manifest = [
        cell for cell in manifest if cell["id"] not in development_ids
    ]
    if len(confirmatory_manifest) != confirmatory_count:
        raise ValueError("Confirmatory complement has the wrong cell count")
    confirmatory_states = [
        states_by_id[cell["id"]] for cell in confirmatory_manifest
    ]
    gate = _publication_gate(
        confirmatory_manifest,
        confirmatory_states,
        required_cells=confirmatory_count,
    )
    criteria = dict(gate["criteria"])
    criteria["full_confirmatory_scope"] = criteria.pop(
        "full_matrix_scope"
    )
    criteria["confirmatory_complete_without_runtime_failures"] = (
        criteria.pop("matrix_complete_without_runtime_failures")
    )
    gate["criteria"] = criteria
    gate["confirmatory_gate_passed"] = all(criteria.values())
    gate.pop("development_gate_passed", None)
    gate.pop("confirmatory_evidence_required", None)

    return {
        "generated_unix": time.time(),
        "schema_version": 1,
        "method_source_revision": matrix_summary.get("source_revision"),
        "selection": {
            "rule": selection["rule"],
            "development_cell_count": development_count,
            "development_cell_ids_sha256": actual_hash,
            "last_development_started_unix": last_started,
            "first_confirmatory_started_unix": first_confirmatory_started,
            "boundary_gap_seconds": boundary_gap_seconds,
            "selection_uses_outcomes": False,
        },
        "scope": {
            "full_matrix": expected_full,
            "development": development_count,
            "confirmatory": confirmatory_count,
            "confirmatory_task_families": sorted(
                {cell["task_family"] for cell in confirmatory_manifest}
            ),
            "confirmatory_datasets": sorted(
                {cell["dataset"] for cell in confirmatory_manifest}
            ),
            "confirmatory_models": sorted(
                {cell["model"] for cell in confirmatory_manifest}
            ),
        },
        "development_counts": dict(
            Counter(row["state"] for row in development_states)
        ),
        "confirmatory_groups": {
            "task_family": _group_summary(
                confirmatory_states,
                "task_family",
            ),
            "dataset": _group_summary(confirmatory_states, "dataset"),
            "model": _group_summary(confirmatory_states, "model"),
        },
        "confirmatory_gate": gate,
        "confirmatory_cell_ids": sorted(
            cell["id"] for cell in confirmatory_manifest
        ),
    }


def build_matrix_summary(
    manifest,
    output_root,
    smoke=False,
    seed=2021,
    source_revision=None,
    export_only=False,
    publication_expected_cells=PUBLICATION_EXPECTED_CELLS,
):
    manifest = list(manifest)
    attempts = discover_attempts(output_root)
    cell_states = [
        _cell_state(
            cell,
            _select_attempt(
                attempts.get(cell["id"], []),
                smoke,
                seed,
                source_revision=source_revision,
                export_only=export_only,
            ),
        )
        for cell in manifest
    ]
    return summarize_cell_states(
        manifest,
        cell_states,
        output_root=output_root,
        smoke=smoke,
        seed=seed,
        source_revision=source_revision,
        export_only=export_only,
        publication_expected_cells=publication_expected_cells,
    )


def summarize_cell_states(
    manifest,
    cell_states,
    *,
    output_root,
    smoke=False,
    seed=2021,
    source_revision=None,
    export_only=False,
    publication_expected_cells=PUBLICATION_EXPECTED_CELLS,
):
    manifest = list(manifest)
    cell_states = list(cell_states)
    manifest_ids = [cell["id"] for cell in manifest]
    state_ids = [row.get("cell_id") for row in cell_states]
    if (
        len(set(manifest_ids)) != len(manifest_ids)
        or len(set(state_ids)) != len(state_ids)
        or state_ids != manifest_ids
    ):
        raise ValueError(
            "Cell states must exactly match manifest order and IDs"
        )
    completed = [row for row in cell_states if row["state"] == "completed"]
    improved = [
        row for row in completed if row.get("all_test_metrics_improve")
    ]
    not_improved = [
        row for row in completed if not row.get("all_test_metrics_improve")
    ]
    ranked = [
        row
        for row in completed
        if row.get("minimum_metric_gain_percent") is not None
    ]
    ranked.sort(
        key=lambda row: row["minimum_metric_gain_percent"], reverse=True
    )
    counts = Counter(row["state"] for row in cell_states)
    counts["expected"] = len(cell_states)
    counts["completed"] = len(completed)
    counts["improved"] = len(improved)
    counts["not_improved"] = len(not_improved)
    for state in ("failed", "running", "pending", "incomplete"):
        counts.setdefault(state, 0)
    methods = Counter(
        row["method"] for row in completed if row.get("method") is not None
    )
    return {
        "generated_unix": time.time(),
        "output_root": str(output_root),
        "smoke": smoke,
        "seed": seed,
        "source_revision": source_revision,
        "export_only": export_only,
        "counts": dict(counts),
        "all_completed": len(completed) == len(cell_states),
        "all_improved": (
            len(improved) == len(cell_states) and len(cell_states) > 0
        ),
        "groups": {
            "task_family": _group_summary(cell_states, "task_family"),
            "dataset": _group_summary(cell_states, "dataset"),
            "model": _group_summary(cell_states, "model"),
        },
        "method_counts": dict(sorted(methods.items())),
        "metric_gain_percent": _gain_summary(cell_states),
        "publication_gate": _publication_gate(
            manifest,
            cell_states,
            required_cells=publication_expected_cells,
        ),
        "best_cells": ranked[:20],
        "weakest_cells": list(reversed(ranked[-20:])),
        "not_improved_cells": not_improved,
        "failed_cells": [
            row for row in cell_states if row["state"] == "failed"
        ],
        "cell_states": cell_states,
    }
