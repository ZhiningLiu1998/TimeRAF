import hashlib
import json
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

from ts_rag.appendix_benchmark import (
    AUTOGLOUON_TIMESERIES_VERSION,
    autogluon_checkpoint_compatibility,
)


APPENDIX_EXPECTED_TOTAL = 96
APPENDIX_EXPECTED_EVALUABLE = 94
APPENDIX_METHOD_REVISION = (
    "052fcb407f155fe6bf29bc4d214f7ac5a56936ee"
)
APPENDIX_SELECTOR_POLICY = "validation_argmin_v3"
APPENDIX_SELECTOR_PROTOCOL_SHA256 = (
    "6c7de88b2d10e19b04ad56259088ec51"
    "d0b14dc3cd5f092f5cc14d66878f53f1"
)
APPENDIX_RAG_GPU_INACTIVITY_REASON = (
    "Appendix RAG replays numerical corrections over existing prediction "
    "bundles and performs no model inference."
)
APPENDIX_EXECUTION_TOPOLOGY = {
    "instance_type": "ml.g5.12xlarge",
    "instance_count": 1,
    "reserved_gpus_per_host": 4,
    "active_gpu_workers": 0,
    "inactive_reserved_gpus": 4,
    "cpu_workers": 4,
    "gpu_inactivity_reason": APPENDIX_RAG_GPU_INACTIVITY_REASON,
}
APPENDIX_CATALOG_SCHEMA_VERSION = 2
APPENDIX_CATALOG_COUNTS = {
    "reported": APPENDIX_EXPECTED_TOTAL,
    "evaluable": APPENDIX_EXPECTED_EVALUABLE,
    "cataloged": APPENDIX_EXPECTED_EVALUABLE,
    "missing_evaluable": 0,
}
_CATALOG_IDENTITY_FIELDS = (
    "baseline",
    "dataset",
    "task_family",
    "pred_len",
)
_STATIC_BASELINES = {
    "forward_selection",
    "portfolio_ensemble",
    "zeroshot_ensemble",
}
_EXTERNAL_BASELINES = {
    "autogluon_high_quality",
    "chronos_bolt_finetuned",
    "chronos_bolt_zeroshot",
}


def _load_json(path):
    try:
        with Path(path).open("r", encoding="utf-8") as source:
            return json.load(source)
    except (OSError, json.JSONDecodeError):
        return None


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_safe_relative_path(value, field, root):
    if not isinstance(value, str) or "\\" in value:
        raise ValueError(f"{field} must be a safe relative path")
    relative = PurePosixPath(value)
    if (
        relative.is_absolute()
        or not relative.parts
        or ".." in relative.parts
    ):
        raise ValueError(f"{field} must be a safe relative path")
    path = Path(root).joinpath(*relative.parts)
    resolved_root = Path(root).resolve()
    resolved = path.resolve()
    if resolved == resolved_root or resolved_root not in resolved.parents:
        raise ValueError(f"{field} escapes the catalog project root")
    return path


def _validate_catalog_header(catalog):
    if not isinstance(catalog, dict):
        raise ValueError("Appendix bundle catalog must be an object")
    if catalog.get("schema_version") != APPENDIX_CATALOG_SCHEMA_VERSION:
        raise ValueError(
            "Appendix bundle catalog requires schema_version=2"
        )
    if catalog.get("hashes_verified") is not True:
        raise ValueError("Appendix bundle catalog hashes_verified must be true")
    project_root = catalog.get("project_root")
    if (
        not isinstance(project_root, str)
        or not project_root
        or not Path(project_root).is_absolute()
    ):
        raise ValueError(
            "Appendix bundle catalog project_root must be absolute"
        )
    if catalog.get("counts") != APPENDIX_CATALOG_COUNTS:
        raise ValueError(
            "Appendix bundle catalog counts must be "
            "reported=96, evaluable=94, cataloged=94, missing_evaluable=0"
        )
    if not isinstance(catalog.get("bundles"), dict):
        raise ValueError("Appendix bundle catalog requires a bundles object")
    if len(catalog["bundles"]) != APPENDIX_EXPECTED_EVALUABLE:
        raise ValueError("Appendix bundle catalog must contain 94 bundles")
    for cell_id, record in catalog["bundles"].items():
        if not isinstance(cell_id, str) or not cell_id:
            raise ValueError("Appendix bundle catalog cell IDs must be strings")
        if not isinstance(record, dict):
            raise ValueError(
                f"Bundle catalog entry {cell_id!r} must be an object"
            )


def load_bundle_catalog(path, project_root=None):
    with Path(path).open("r", encoding="utf-8") as source:
        catalog = json.load(source)
    _validate_catalog_header(catalog)
    catalog_root = Path(catalog["project_root"]).resolve()
    requested_root = (
        catalog_root if project_root is None else Path(project_root).resolve()
    )
    if catalog_root != requested_root:
        raise ValueError("Appendix bundle catalog project_root drifted")
    return {
        **catalog,
        "_catalog_project_root": str(requested_root),
    }


def validate_catalog_scope(
    manifest,
    catalog,
    require_files=True,
    allow_extra=False,
):
    del allow_extra
    _validate_catalog_header(catalog)
    project_root = Path(
        catalog.get("_catalog_project_root", Path.cwd())
    ).resolve()

    manifest_path = _require_safe_relative_path(
        catalog.get("manifest"),
        "catalog manifest",
        project_root,
    )
    expected_manifest_sha256 = catalog.get("manifest_sha256")
    if not _is_sha256(expected_manifest_sha256):
        raise ValueError("Appendix catalog manifest SHA-256 is invalid")
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise FileNotFoundError(
            f"Appendix catalog manifest is missing: {manifest_path}"
        )
    actual_manifest_sha256 = _sha256_file(manifest_path)
    if actual_manifest_sha256 != expected_manifest_sha256:
        raise ValueError("Appendix catalog manifest SHA-256 mismatch")

    replay_catalog_path = _require_safe_relative_path(
        catalog.get("replay_catalog"),
        "replay catalog",
        project_root,
    )
    expected_replay_catalog_sha256 = catalog.get(
        "replay_catalog_sha256"
    )
    if not _is_sha256(expected_replay_catalog_sha256):
        raise ValueError("Replay catalog SHA-256 is invalid")
    if (
        not replay_catalog_path.is_file()
        or replay_catalog_path.is_symlink()
    ):
        raise FileNotFoundError(
            f"Replay catalog is missing: {replay_catalog_path}"
        )
    if (
        _sha256_file(replay_catalog_path)
        != expected_replay_catalog_sha256
    ):
        raise ValueError("Replay catalog SHA-256 mismatch")

    source_revision = catalog.get("source_revision")
    if not isinstance(source_revision, str) or not source_revision:
        raise ValueError("Appendix bundle catalog source_revision is invalid")

    with manifest_path.open("r", encoding="utf-8") as source:
        full_rows = [json.loads(line) for line in source if line.strip()]
    full_by_id = {row.get("id"): row for row in full_rows}
    if (
        len(full_rows) != APPENDIX_EXPECTED_TOTAL
        or len(full_by_id) != APPENDIX_EXPECTED_TOTAL
        or None in full_by_id
    ):
        raise ValueError(
            "Appendix catalog manifest must contain 96 unique cells"
        )
    evaluable_ids = {
        row["id"]
        for row in full_rows
        if row.get("paper_status", {}).get("evaluable") is True
    }
    if len(evaluable_ids) != APPENDIX_EXPECTED_EVALUABLE:
        raise ValueError(
            "Appendix catalog manifest must contain 94 evaluable cells"
        )
    if set(catalog["bundles"]) != evaluable_ids:
        raise ValueError(
            "Appendix bundle catalog must exactly cover all 94 evaluable cells"
        )

    rows = list(manifest)
    selected_by_id = {row.get("id"): row for row in rows}
    if len(selected_by_id) != len(rows) or None in selected_by_id:
        raise ValueError("Selected Appendix manifest cell IDs must be unique")
    unknown = sorted(set(selected_by_id) - set(full_by_id))
    if unknown:
        raise ValueError(
            f"Selected Appendix manifest contains unknown cells: {unknown}"
        )
    for cell_id, row in selected_by_id.items():
        if row != full_by_id[cell_id]:
            raise ValueError(
                f"Selected Appendix manifest cell drifted: {cell_id}"
            )

    for cell_id in sorted(evaluable_ids):
        cell = full_by_id[cell_id]
        record = catalog["bundles"][cell_id]
        for field in _CATALOG_IDENTITY_FIELDS:
            if record.get(field) != cell.get(field):
                raise ValueError(
                    f"Appendix bundle identity mismatch for {cell_id}: {field}"
                )

        bundle_path = _require_safe_relative_path(
            record.get("path"),
            f"bundle path for {cell_id}",
            project_root,
        )
        size_bytes = record.get("size_bytes")
        if (
            not isinstance(size_bytes, int)
            or isinstance(size_bytes, bool)
            or size_bytes <= 0
        ):
            raise ValueError(
                f"Appendix bundle size is invalid for {cell_id}"
            )
        expected_bundle_sha256 = record.get("sha256")
        if not _is_sha256(expected_bundle_sha256):
            raise ValueError(
                f"Appendix bundle SHA-256 is invalid for {cell_id}"
            )

        metadata_path = _require_safe_relative_path(
            record.get("metadata_path"),
            f"metadata path for {cell_id}",
            project_root,
        )
        expected_metadata_sha256 = record.get("metadata_sha256")
        if not _is_sha256(expected_metadata_sha256):
            raise ValueError(
                f"Appendix metadata SHA-256 is invalid for {cell_id}"
            )
        if bundle_path.resolve() == metadata_path.resolve():
            raise ValueError(
                f"Appendix bundle and metadata paths collide for {cell_id}"
            )
        if (
            cell["baseline"] in _EXTERNAL_BASELINES
            and record.get("autogluon_timeseries_version")
            != AUTOGLOUON_TIMESERIES_VERSION
        ):
            raise ValueError(
                "Appendix catalog AutoGluon version mismatch "
                f"for {cell_id}"
            )
        expected_compatibility = (
            autogluon_checkpoint_compatibility(cell["baseline"])
            if cell["baseline"] == "autogluon_high_quality"
            else None
        )
        if (
            expected_compatibility is not None
            and record.get("checkpoint_compatibility")
            != expected_compatibility
        ):
            raise ValueError(
                "Appendix catalog checkpoint compatibility mismatch "
                f"for {cell_id}"
            )

        if not require_files:
            continue
        if not bundle_path.is_file() or bundle_path.is_symlink():
            raise FileNotFoundError(
                f"Appendix bundle is missing for {cell_id}: {bundle_path}"
            )
        if bundle_path.stat().st_size != size_bytes:
            raise ValueError(
                f"Appendix bundle size mismatch for {cell_id}"
            )
        if _sha256_file(bundle_path) != expected_bundle_sha256:
            raise ValueError(
                f"Appendix bundle SHA-256 mismatch for {cell_id}"
            )
        if not metadata_path.is_file() or metadata_path.is_symlink():
            raise FileNotFoundError(
                f"Appendix metadata is missing for {cell_id}: "
                f"{metadata_path}"
            )
        if _sha256_file(metadata_path) != expected_metadata_sha256:
            raise ValueError(
                f"Appendix metadata SHA-256 mismatch for {cell_id}"
            )
        metadata = _load_json(metadata_path)
        if not isinstance(metadata, dict):
            raise ValueError(
                f"Appendix metadata is invalid for {cell_id}"
            )
        if metadata.get("cell_id") != cell_id:
            raise ValueError(
                f"Appendix metadata cell_id mismatch for {cell_id}"
            )
        if (
            metadata.get("prediction_bundle_sha256")
            != expected_bundle_sha256
        ):
            raise ValueError(
                "Appendix metadata prediction bundle SHA-256 mismatch "
                f"for {cell_id}"
            )
        run_identity = metadata.get("run_identity")
        if not isinstance(run_identity, dict):
            raise ValueError(
                f"Appendix metadata run_identity is invalid for {cell_id}"
            )
        if run_identity.get("source_revision") != source_revision:
            raise ValueError(
                "Appendix metadata source revision mismatch "
                f"for {cell_id}"
            )
        if cell["baseline"] in _EXTERNAL_BASELINES:
            if (
                run_identity.get("autogluon_timeseries_version")
                != AUTOGLOUON_TIMESERIES_VERSION
            ):
                raise ValueError(
                    "Appendix metadata run identity AutoGluon version "
                    f"mismatch for {cell_id}"
                )
            predictor = metadata.get("predictor")
            if (
                not isinstance(predictor, dict)
                or predictor.get("autogluon_timeseries_version")
                != AUTOGLOUON_TIMESERIES_VERSION
            ):
                raise ValueError(
                    "Appendix predictor AutoGluon version mismatch "
                    f"for {cell_id}"
                )
            if (
                expected_compatibility is not None
                and (
                    run_identity.get("checkpoint_compatibility")
                    != expected_compatibility
                    or predictor.get("checkpoint_compatibility")
                    != expected_compatibility
                    or metadata.get("checkpoint_compatibility")
                    != expected_compatibility
                )
            ):
                raise ValueError(
                    "Appendix metadata checkpoint compatibility mismatch "
                    f"for {cell_id}"
                )
        if (
            cell["baseline"] in _STATIC_BASELINES
            and run_identity.get("replay_catalog_sha256")
            != expected_replay_catalog_sha256
        ):
            raise ValueError(
                "Appendix metadata replay catalog SHA-256 mismatch "
                f"for {cell_id}"
            )


def discover_appendix_attempts(output_root):
    attempts = defaultdict(list)
    for status_path in Path(output_root).glob("**/status.json"):
        status = _load_json(status_path)
        if not status or not status.get("cell_id"):
            continue
        attempts[status["cell_id"]].append(
            {
                "artifact_dir": str(status_path.parent),
                "status": status,
                "result": _load_json(status_path.with_name("result.json")),
            }
        )
    return attempts


def _select_attempt(attempts, source_revision, method_revision=None):
    matches = [
        attempt
        for attempt in attempts
        if source_revision is None
        or attempt["status"].get("source_revision") == source_revision
    ]
    if method_revision is not None:
        matches = [
            attempt
            for attempt in matches
            if attempt["status"].get("method_revision") == method_revision
        ]
    if not matches:
        return None
    complete = [
        attempt
        for attempt in matches
        if attempt["status"].get("status") in {"completed", "paper_oot"}
        and attempt["result"] is not None
    ]
    return max(
        complete or matches,
        key=lambda attempt: float(
            attempt["status"].get("started_unix", 0.0)
        ),
    )


def _metric_gains(result, metrics):
    gains = {}
    baseline = result.get("test_baseline", {})
    corrected = result.get("test_corrected", {})
    for metric in metrics:
        before = baseline.get(metric)
        after = corrected.get(metric)
        gains[metric] = (
            None
            if before is None or after is None or before == 0
            else 100.0 * (before - after) / abs(before)
        )
    return gains


def _cell_state(cell, attempt):
    base = {
        "cell_id": cell["id"],
        "task_family": cell["task_family"],
        "dataset": cell["dataset"],
        "baseline": cell["baseline"],
        "pred_len": cell["pred_len"],
        "paper_evaluable": bool(cell["paper_status"]["evaluable"]),
    }
    if attempt is None:
        return {**base, "state": "pending"}

    status = attempt["status"]
    result = attempt["result"]
    raw_state = status.get("status", "unknown")
    if raw_state == "completed" and result is not None:
        state = "completed"
    elif raw_state == "paper_oot" and result is not None:
        state = "paper_oot"
    elif raw_state in {"running", "failed"}:
        state = raw_state
    else:
        state = "incomplete"
    row = {
        **base,
        "state": state,
        "artifact_dir": attempt["artifact_dir"],
        "started_unix": status.get("started_unix"),
        "elapsed_seconds": status.get("elapsed_seconds"),
        "method_revision": status.get("method_revision"),
    }
    if state == "completed":
        row["all_test_metrics_improve"] = bool(
            result.get("all_test_metrics_improve")
        )
        row["method"] = (
            result.get("validation", {}).get("selected", {}).get("method")
        )
        row["test_baseline"] = result.get("test_baseline")
        row["test_corrected"] = result.get("test_corrected")
        row["metric_gain_percent"] = _metric_gains(
            result,
            cell["metrics"],
        )
    elif state == "failed":
        row["error_type"] = status.get("error_type")
        row["error"] = status.get("error")
    return row


def _group_summary(states, key):
    groups = {}
    for value in sorted({row[key] for row in states}):
        rows = [row for row in states if row[key] == value]
        evaluable = [row for row in rows if row["paper_evaluable"]]
        completed = [row for row in evaluable if row["state"] == "completed"]
        improved = sum(
            bool(row.get("all_test_metrics_improve")) for row in completed
        )
        groups[value] = {
            "reported": len(rows),
            "evaluable": len(evaluable),
            "completed": len(completed),
            "improved": improved,
            "improvement_rate": (
                improved / len(evaluable) if evaluable else None
            ),
            "not_improved": sum(
                not bool(row.get("all_test_metrics_improve"))
                for row in completed
            ),
            "paper_oot": sum(row["state"] == "paper_oot" for row in rows),
            "failed": sum(row["state"] == "failed" for row in rows),
            "running": sum(row["state"] == "running" for row in rows),
            "pending": sum(row["state"] == "pending" for row in rows),
            "incomplete": sum(row["state"] == "incomplete" for row in rows),
        }
    return groups


def _gain_summary(states):
    values_by_metric = defaultdict(list)
    for row in states:
        for metric, value in row.get("metric_gain_percent", {}).items():
            if value is not None:
                values_by_metric[metric].append(value)
    return {
        metric: {
            "count": len(values),
            "minimum": min(values),
            "median": statistics.median(values),
            "mean": statistics.fmean(values),
            "maximum": max(values),
        }
        for metric, values in sorted(values_by_metric.items())
    }


def _expected_metric_counts(manifest):
    counts = Counter()
    for cell in manifest:
        if cell["paper_status"]["evaluable"]:
            counts.update(cell["metrics"])
    return dict(sorted(counts.items()))


def _positive_metrics(summary, expected_counts, require_median):
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


def _publication_gate(
    manifest,
    states,
    expected_total,
    expected_evaluable,
):
    evaluable_manifest = [
        cell for cell in manifest if cell["paper_status"]["evaluable"]
    ]
    evaluable_states = [row for row in states if row["paper_evaluable"]]
    completed = [
        row for row in evaluable_states if row["state"] == "completed"
    ]
    improved = sum(
        bool(row.get("all_test_metrics_improve")) for row in completed
    )
    overall_rate = (
        improved / len(evaluable_manifest) if evaluable_manifest else 0.0
    )
    overall_gains = _gain_summary(evaluable_states)
    family_groups = _group_summary(states, "task_family")
    baseline_groups = _group_summary(states, "baseline")

    family_details = {}
    for family, group in family_groups.items():
        family_manifest = [
            cell
            for cell in evaluable_manifest
            if cell["task_family"] == family
        ]
        family_states = [
            row
            for row in evaluable_states
            if row["task_family"] == family
        ]
        gains = _gain_summary(family_states)
        family_details[family] = {
            **group,
            "metric_gain_percent": gains,
            "positive_metric_means": _positive_metrics(
                gains,
                _expected_metric_counts(family_manifest),
                require_median=False,
            ),
        }

    counts = Counter(row["state"] for row in states)
    criteria = {
        "full_appendix_scope": (
            len(manifest) == expected_total
            and len(evaluable_manifest) == expected_evaluable
        ),
        "complete_without_runtime_failures": (
            len(completed) == len(evaluable_manifest)
            and counts["failed"] == 0
            and counts["paper_oot"] == len(manifest) - len(evaluable_manifest)
        ),
        "overall_improvement_rate": overall_rate >= 0.80,
        "task_family_improvement_rates": all(
            group["improvement_rate"] is not None
            and group["improvement_rate"] >= 0.70
            for group in family_groups.values()
            if group["evaluable"]
        ),
        "baseline_improvement_rates": all(
            group["improvement_rate"] is not None
            and group["improvement_rate"] >= 0.70
            for group in baseline_groups.values()
            if group["evaluable"]
        ),
        "positive_task_family_metric_means": all(
            detail["positive_metric_means"]
            for detail in family_details.values()
            if detail["evaluable"]
        ),
        "positive_overall_metric_means_and_medians": _positive_metrics(
            overall_gains,
            _expected_metric_counts(evaluable_manifest),
            require_median=True,
        ),
    }
    return {
        "scope": {
            "reported": len(manifest),
            "evaluable": len(evaluable_manifest),
            "required_reported": expected_total,
            "required_evaluable": expected_evaluable,
        },
        "thresholds": {
            "overall_improvement_rate": 0.80,
            "task_family_improvement_rate": 0.70,
            "baseline_improvement_rate": 0.70,
        },
        "overall": {
            "evaluable": len(evaluable_manifest),
            "improved": improved,
            "improvement_rate": overall_rate,
            "metric_gain_percent": overall_gains,
        },
        "task_families": family_details,
        "baselines": baseline_groups,
        "criteria": criteria,
        "development_gate_passed": all(criteria.values()),
        "confirmatory_evidence_required": True,
    }


def build_appendix_summary(
    manifest,
    output_root,
    source_revision=None,
    method_revision=None,
    expected_total=APPENDIX_EXPECTED_TOTAL,
    expected_evaluable=APPENDIX_EXPECTED_EVALUABLE,
):
    manifest = list(manifest)
    attempts = discover_appendix_attempts(output_root)
    states = [
        _cell_state(
            cell,
            _select_attempt(
                attempts.get(cell["id"], []),
                source_revision,
                method_revision,
            ),
        )
        for cell in manifest
    ]
    evaluable = [row for row in states if row["paper_evaluable"]]
    completed = [row for row in evaluable if row["state"] == "completed"]
    improved = [
        row for row in completed if row.get("all_test_metrics_improve")
    ]
    not_improved = [
        row for row in completed if not row.get("all_test_metrics_improve")
    ]
    counts = Counter(row["state"] for row in states)
    for state in (
        "completed",
        "paper_oot",
        "failed",
        "running",
        "pending",
        "incomplete",
    ):
        counts.setdefault(state, 0)
    counts.update(
        {
            "reported": len(states),
            "evaluable": len(evaluable),
            "improved": len(improved),
            "not_improved": len(not_improved),
        }
    )
    methods = Counter(
        row["method"] for row in completed if row.get("method")
    )
    return {
        "generated_unix": time.time(),
        "output_root": str(output_root),
        "source_revision": source_revision,
        "method_revision": method_revision,
        "counts": dict(counts),
        "all_evaluable_completed": len(completed) == len(evaluable),
        "groups": {
            "task_family": _group_summary(states, "task_family"),
            "dataset": _group_summary(states, "dataset"),
            "baseline": _group_summary(states, "baseline"),
        },
        "method_counts": dict(sorted(methods.items())),
        "metric_gain_percent": _gain_summary(evaluable),
        "publication_gate": _publication_gate(
            manifest,
            states,
            expected_total=expected_total,
            expected_evaluable=expected_evaluable,
        ),
        "not_improved_cells": not_improved,
        "failed_cells": [
            row for row in states if row["state"] == "failed"
        ],
        "paper_oot_cells": [
            row for row in states if not row["paper_evaluable"]
        ],
        "cell_states": states,
    }
