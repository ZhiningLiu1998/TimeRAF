import hashlib
import json
from pathlib import Path

import pytest

from scripts import fetch_greenland_full_matrix
from scripts import fetch_greenland_numerical_recovery
from scripts.greenland_common import (
    FULL_MATRIX_MANIFEST_SHA256,
    METHOD_REVISION,
    NUMERICAL_RECOVERY_CELL_IDS,
    NUMERICAL_RECOVERY_CELL_IDS_SHA256,
    NUMERICAL_RECOVERY_COHORT_ID,
    NUMERICAL_RECOVERY_JOB,
    NUMERICAL_RECOVERY_PROFILES,
    NUMERICAL_RECOVERY_PROTOCOL,
    NUMERICAL_RECOVERY_PROTOCOL_SHA256,
)
from ts_rag.matrix import load_manifest


RUN_ID = "full-matrix-fixture"
SOURCE_REVISION = "a" * 40
LAUNCHER_REVISION = "b" * 40
IMAGE_URI = (
    "<ECR_REGISTRY>/"
    "timeraf-greenland@sha256:" + "c" * 64
)


def _sha256(payload):
    return hashlib.sha256(payload).hexdigest()


class _Paginator:
    def __init__(self, objects):
        self.objects = objects

    def paginate(self, Bucket, Prefix):
        del Bucket
        yield {
            "Contents": [
                {"Key": key, "Size": len(payload)}
                for key, payload in sorted(self.objects.items())
                if key.startswith(Prefix)
            ]
        }


class _S3:
    def __init__(self, objects):
        self.objects = objects

    def get_paginator(self, name):
        assert name == "list_objects_v2"
        return _Paginator(self.objects)

    def head_object(self, Bucket, Key):
        del Bucket
        payload = self.objects[Key]
        return {
            "ContentLength": len(payload),
            "Metadata": {"sha256": _sha256(payload)},
            "ETag": _sha256(payload),
        }

    def download_file(self, bucket, key, filename):
        del bucket
        Path(filename).write_bytes(self.objects[key])


def _spec():
    digests = {"source": "1" * 64, "dataset": "2" * 64}
    return {
        "schema_version": 1,
        "run_id": RUN_ID,
        "job_kind": "full_matrix_replication",
        "method_revision": METHOD_REVISION,
        "image_uri": IMAGE_URI,
        "source_revision": SOURCE_REVISION,
        "launcher_revision": LAUNCHER_REVISION,
        "inputs": [
            {
                "name": name,
                "s3_uri": (
                    "s3://<DEV_BUCKET>/timeraf/greenland/"
                    f"inputs/{name}/{digest}.tar.gz"
                ),
                "sha256": digest,
                "extract_to": "project",
            }
            for name, digest in digests.items()
        ],
        "output_s3_uri": (
            "s3://<DEV_BUCKET>/timeraf/greenland/"
            f"runs/{RUN_ID}/"
        ),
        "matrix": {
            "manifest": "docs/timefuse_experiment_manifest.jsonl",
            "output_root": "outputs/full_matrix_replication",
            "checkpoint_root": (
                "outputs/full_matrix_replication/checkpoints"
            ),
            "summary": (
                "outputs/full_matrix_replication/matrix_summary.json"
            ),
            "log_root": "outputs/full_matrix_replication/logs",
            "queued_cells": 585,
            "seed": 2021,
            "extra_args": ["--max-failures=0", "--num-workers=4"],
        },
    }


def _add_json(objects, prefix, relative, payload):
    objects[prefix + relative] = (
        json.dumps(payload, sort_keys=True) + "\n"
    ).encode()


def _run_objects(rows, *, nonfinite_ids=(), missing_result_ids=()):
    prefix = f"timeraf/greenland/runs/{RUN_ID}/"
    objects = {}
    _add_json(
        objects,
        prefix,
        "evidence/accepted_job_spec.json",
        _spec(),
    )
    _add_json(
        objects,
        prefix,
        "evidence/topology_summary.json",
        {
            "topology_gate_passed": True,
            "eight_distinct_bindings_observed": True,
            "all_eight_gpus_utilized": True,
            "expected_gpu_ids": list(range(8)),
            "max_utilization_gpu_percent": {
                index: 20 for index in range(8)
            },
            "binding_evidence": {
                "gpu_ids": list(range(8)),
                "pids": list(range(100, 108)),
            },
        },
    )
    _add_json(
        objects,
        prefix,
        "evidence/full_matrix_manifest.json",
        {
            "cells": 585,
            "sha256": FULL_MATRIX_MANIFEST_SHA256,
            "family_counts": {
                "long_term": 364,
                "pems": 156,
                "epf": 65,
            },
        },
    )
    _add_json(
        objects,
        prefix,
        "final_status.json",
        {
            "run_id": RUN_ID,
            "job_kind": "full_matrix_replication",
            "method_revision": METHOD_REVISION,
            "source_revision": SOURCE_REVISION,
            "launcher_revision": LAUNCHER_REVISION,
            "exit_code": 0,
            "status": "succeeded",
            "topology_gate_passed": True,
            "matrix_elapsed_seconds": 1234.0,
        },
    )
    counts = {
        "expected": 585,
        "completed": 585,
        "improved": 585,
        "not_improved": 0,
        "failed": 0,
        "pending": 0,
        "running": 0,
        "incomplete": 0,
    }
    _add_json(
        objects,
        prefix,
        "full_matrix_replication/matrix_summary.json",
        {
            "source_revision": METHOD_REVISION,
            "export_only": False,
            "all_completed": True,
            "counts": counts,
        },
    )
    nonfinite_ids = set(nonfinite_ids)
    missing_result_ids = set(missing_result_ids)
    for index, row in enumerate(rows):
        artifact = f"full_matrix_replication/cells/{index:03d}"
        baseline = {metric: 1.0 for metric in row["metrics"]}
        corrected = {metric: 0.9 for metric in row["metrics"]}
        if row["id"] in nonfinite_ids:
            corrected[row["metrics"][0]] = float("nan")
        _add_json(
            objects,
            prefix,
            f"{artifact}/status.json",
            {
                "cell_id": row["id"],
                "source_revision": METHOD_REVISION,
                "seed": 2021,
                "smoke": False,
                "status": "completed",
                "started_unix": index,
                "elapsed_seconds": 1.0,
            },
        )
        if row["id"] not in missing_result_ids:
            _add_json(
                objects,
                prefix,
                f"{artifact}/result.json",
                {
                    "cell": {"id": row["id"]},
                    "smoke": False,
                    "checkpoint": {"mode": "trained"},
                    "validation": {"selected": {"method": "identity"}},
                    "test_baseline": baseline,
                    "test_corrected": corrected,
                    "all_test_metrics_improve": True,
                },
            )
    objects[
        prefix
        + "full_matrix_replication/checkpoints/cell/checkpoint.pth"
    ] = b"checkpoint"
    runtime_records = []
    for key, payload in sorted(objects.items()):
        relative = key[len(prefix) :]
        runtime_records.append(
            {
                "relative_path": relative,
                "s3_uri": f"s3://<DEV_BUCKET>/{key}",
                "sha256": _sha256(payload),
                "size_bytes": len(payload),
            }
        )
    _add_json(
        objects,
        prefix,
        "upload_complete.json",
        {
            "schema_version": 1,
            "run_id": RUN_ID,
            "job_kind": "full_matrix_replication",
            "method_revision": METHOD_REVISION,
            "source_revision": SOURCE_REVISION,
            "launcher_revision": LAUNCHER_REVISION,
            "uploaded_file_count": len(runtime_records),
            "uploaded_size_bytes": sum(
                record["size_bytes"] for record in runtime_records
            ),
            "objects": runtime_records,
        },
    )
    return objects


def test_full_matrix_import_verifies_results_and_leaves_checkpoints_in_s3(
    tmp_path,
):
    project = tmp_path / "project"
    manifest = project / "docs" / "timefuse_experiment_manifest.jsonl"
    manifest.parent.mkdir(parents=True)
    manifest.write_bytes(
        Path("docs/timefuse_experiment_manifest.jsonl").read_bytes()
    )
    rows = load_manifest(manifest)
    destination = project / "outputs" / "greenland_runs" / RUN_ID

    receipt = fetch_greenland_full_matrix.fetch_full_matrix_run(
        _S3(_run_objects(rows)),
        RUN_ID,
        project,
        destination,
        manifest,
    )

    assert receipt["matrix_counts"]["completed"] == 585
    assert receipt["topology_gate_passed"]
    assert receipt["checkpoint_objects_retained_in_s3"] == 1
    recomputed = receipt["recomputed_matrix_summary"]
    recomputed_path = Path(recomputed["path"])
    assert recomputed_path == destination / "recomputed_matrix_summary.json"
    assert recomputed["sha256"] == _sha256(recomputed_path.read_bytes())
    assert recomputed["size_bytes"] == recomputed_path.stat().st_size
    recomputed_payload = json.loads(recomputed_path.read_text())
    assert recomputed_payload["counts"]["completed"] == 585
    assert all(
        destination in Path(row["artifact_dir"]).parents
        and Path(row["artifact_dir"], "status.json").is_file()
        and Path(row["artifact_dir"], "result.json").is_file()
        for row in recomputed_payload["cell_states"]
    )
    assert not (
        destination
        / "full_matrix_replication"
        / "checkpoints"
        / "cell"
        / "checkpoint.pth"
    ).exists()
    assert receipt["imported_object_count"] == receipt["object_count"] - 1


def test_upload_marker_rejects_missing_runtime_object():
    records = {
        "final_status.json": {
            "key": f"timeraf/greenland/runs/{RUN_ID}/final_status.json",
            "sha256": "a" * 64,
            "size_bytes": 1,
        }
    }
    marker = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "job_kind": "full_matrix_replication",
        "uploaded_file_count": 0,
        "uploaded_size_bytes": 0,
        "objects": [],
    }

    with pytest.raises(ValueError, match="does not cover"):
        fetch_greenland_full_matrix._validate_upload_marker(
            marker,
            records,
            RUN_ID,
        )


def test_full_matrix_terminal_first_pass_import_preserves_nonfinite_evidence(
    tmp_path,
):
    project = tmp_path / "project"
    manifest = project / "docs" / "timefuse_experiment_manifest.jsonl"
    manifest.parent.mkdir(parents=True)
    manifest.write_bytes(
        Path("docs/timefuse_experiment_manifest.jsonl").read_bytes()
    )
    rows = load_manifest(manifest)
    nonfinite_ids = list(NUMERICAL_RECOVERY_CELL_IDS[:2])
    objects = _run_objects(rows, nonfinite_ids=nonfinite_ids)
    destination = project / "outputs" / RUN_ID

    with pytest.raises(ValueError, match="complete full matrix"):
        fetch_greenland_full_matrix.fetch_full_matrix_run(
            _S3(objects),
            RUN_ID,
            project,
            destination,
            manifest,
        )

    receipt = fetch_greenland_full_matrix.fetch_full_matrix_run(
        _S3(objects),
        RUN_ID,
        project,
        destination,
        manifest,
        accept_terminal_nonfinite_first_pass=True,
    )

    assert receipt["matrix_counts"]["completed"] == 583
    assert receipt["matrix_counts"]["incomplete"] == 2
    acceptance = receipt["import_acceptance"]
    assert acceptance["mode"] == "terminal-nonfinite-first-pass"
    assert acceptance["terminal_attempt_count"] == 585
    assert acceptance["nonfinite_cell_ids"] == sorted(nonfinite_ids)
    assert len(acceptance["nonfinite_artifacts"]) == 2
    for artifact in acceptance["nonfinite_artifacts"]:
        assert artifact["status"]["sha256"] == _sha256(
            Path(artifact["status"]["path"]).read_bytes()
        )
        assert artifact["result"]["sha256"] == _sha256(
            Path(artifact["result"]["path"]).read_bytes()
        )
        assert artifact["metric_evidence"]
    topology = receipt["topology_evidence"]
    assert topology["sha256"] == _sha256(
        Path(topology["path"]).read_bytes()
    )


def test_full_matrix_terminal_first_pass_rejects_uncovered_or_missing_result(
    tmp_path,
):
    project = tmp_path / "project"
    manifest = project / "docs" / "timefuse_experiment_manifest.jsonl"
    manifest.parent.mkdir(parents=True)
    manifest.write_bytes(
        Path("docs/timefuse_experiment_manifest.jsonl").read_bytes()
    )
    rows = load_manifest(manifest)
    outside = next(
        row["id"]
        for row in rows
        if row["id"] not in set(NUMERICAL_RECOVERY_CELL_IDS)
    )
    with pytest.raises(ValueError, match="outside the frozen recovery cohort"):
        fetch_greenland_full_matrix.fetch_full_matrix_run(
            _S3(_run_objects(rows, nonfinite_ids={outside})),
            RUN_ID,
            project,
            project / "outputs" / "outside",
            manifest,
            accept_terminal_nonfinite_first_pass=True,
        )

    missing = NUMERICAL_RECOVERY_CELL_IDS[0]
    with pytest.raises(ValueError, match="not a numerical integrity outcome"):
        fetch_greenland_full_matrix.fetch_full_matrix_run(
            _S3(_run_objects(rows, missing_result_ids={missing})),
            RUN_ID,
            project,
            project / "outputs" / "missing",
            manifest,
            accept_terminal_nonfinite_first_pass=True,
        )


def _recovery_spec(run_id, profile):
    root = f"outputs/numerical_recovery/{profile}"
    return {
        "schema_version": 2,
        "run_id": run_id,
        "job_kind": NUMERICAL_RECOVERY_JOB,
        "method_revision": METHOD_REVISION,
        "image_uri": IMAGE_URI,
        "source_revision": SOURCE_REVISION,
        "launcher_revision": LAUNCHER_REVISION,
        "recovery_revision": SOURCE_REVISION,
        "inputs": [
            {
                "name": name,
                "s3_uri": (
                    "s3://<DEV_BUCKET>/timeraf/greenland/"
                    f"inputs/{name}/{digest}.tar.gz"
                ),
                "sha256": digest,
                "extract_to": "project",
            }
            for name, digest in {
                "source": "1" * 64,
                "dataset": "2" * 64,
            }.items()
        ],
        "output_s3_uri": (
            "s3://<DEV_BUCKET>/timeraf/greenland/"
            f"runs/{run_id}/"
        ),
        "matrix": {
            "manifest": "docs/timefuse_experiment_manifest.jsonl",
            "output_root": root,
            "checkpoint_root": f"{root}/checkpoints",
            "summary": f"{root}/matrix_summary.json",
            "log_root": f"{root}/logs",
            "queued_cells": 9,
            "seed": 2021,
            "extra_args": ["--max-failures=0", "--num-workers=4"],
        },
        "recovery": {
            "protocol": NUMERICAL_RECOVERY_PROTOCOL,
            "protocol_sha256": NUMERICAL_RECOVERY_PROTOCOL_SHA256,
            "cohort_id": NUMERICAL_RECOVERY_COHORT_ID,
            "cell_ids": list(NUMERICAL_RECOVERY_CELL_IDS),
            "cell_ids_sha256": NUMERICAL_RECOVERY_CELL_IDS_SHA256,
            "profile": profile,
            "overrides": NUMERICAL_RECOVERY_PROFILES[profile],
            "parent_run_id": "full-matrix-a100-9974eac-20260731",
        },
    }


def _recovery_objects(
    rows,
    run_id,
    profile,
    *,
    failed_ids=(),
    failed_error_type="NumericalIntegrityError",
    nonfinite_id=None,
):
    prefix = f"timeraf/greenland/runs/{run_id}/"
    objects = {}
    spec = _recovery_spec(run_id, profile)
    root = f"numerical_recovery/{profile}"
    _add_json(
        objects,
        prefix,
        "evidence/accepted_job_spec.json",
        spec,
    )
    topology = {
        "topology_gate_passed": True,
        "eight_distinct_bindings_observed": True,
        "all_eight_gpus_utilized": True,
        "expected_gpu_ids": list(range(8)),
        "max_utilization_gpu_percent": {
            index: 20 for index in range(8)
        },
        "binding_evidence": {
            "gpu_ids": list(range(8)),
            "pids": list(range(100, 108)),
        },
    }
    _add_json(
        objects,
        prefix,
        "evidence/topology_summary.json",
        topology,
    )
    _add_json(
        objects,
        prefix,
        "evidence/full_matrix_manifest.json",
        {
            "cells": 585,
            "sha256": FULL_MATRIX_MANIFEST_SHA256,
            "family_counts": {
                "long_term": 364,
                "pems": 156,
                "epf": 65,
            },
        },
    )
    _add_json(
        objects,
        prefix,
        "evidence/numerical_recovery_protocol.json",
        {
            "path": NUMERICAL_RECOVERY_PROTOCOL,
            "sha256": NUMERICAL_RECOVERY_PROTOCOL_SHA256,
            "cohort_id": NUMERICAL_RECOVERY_COHORT_ID,
            "cell_ids": list(NUMERICAL_RECOVERY_CELL_IDS),
            "cell_ids_sha256": NUMERICAL_RECOVERY_CELL_IDS_SHA256,
            "profiles": NUMERICAL_RECOVERY_PROFILES,
        },
    )
    failed_ids = set(failed_ids)
    completed = len(rows) - len(failed_ids)
    counts = {
        "expected": len(rows),
        "completed": completed,
        "improved": completed,
        "not_improved": 0,
        "failed": len(failed_ids),
        "pending": 0,
        "running": 0,
        "incomplete": 0,
    }
    _add_json(
        objects,
        prefix,
        f"{root}/matrix_summary.json",
        {
            "source_revision": METHOD_REVISION,
            "export_only": False,
            "all_completed": not failed_ids,
            "counts": counts,
            "failed_cells": [
                {
                    "cell_id": row["id"],
                    "state": "failed",
                    "error_type": failed_error_type,
                }
                for row in rows
                if row["id"] in failed_ids
            ],
        },
    )
    for index, row in enumerate(rows):
        artifact = f"{root}/cells/{index:03d}"
        status = {
            "cell_id": row["id"],
            "source_revision": METHOD_REVISION,
            "seed": 2021,
            "smoke": False,
            "started_unix": index,
            "elapsed_seconds": 1.0,
            "status": "failed" if row["id"] in failed_ids else "completed",
        }
        if row["id"] in failed_ids:
            status.update(
                {
                    "error_type": failed_error_type,
                    "error": "non-finite result",
                }
            )
        _add_json(
            objects,
            prefix,
            f"{artifact}/status.json",
            status,
        )
        if row["id"] in failed_ids:
            continue
        baseline = {metric: 1.0 for metric in row["metrics"]}
        corrected = {metric: 0.9 for metric in row["metrics"]}
        if row["id"] == nonfinite_id:
            corrected[row["metrics"][0]] = float("nan")
        _add_json(
            objects,
            prefix,
            f"{artifact}/result.json",
            {
                "smoke": False,
                "checkpoint": {"mode": "trained"},
                "validation": {"selected": {"method": "identity"}},
                "test_baseline": baseline,
                "test_corrected": corrected,
                "all_test_metrics_improve": True,
            },
        )
    succeeded = not failed_ids
    _add_json(
        objects,
        prefix,
        "final_status.json",
        {
            "run_id": run_id,
            "job_kind": NUMERICAL_RECOVERY_JOB,
            "method_revision": METHOD_REVISION,
            "source_revision": SOURCE_REVISION,
            "launcher_revision": LAUNCHER_REVISION,
            "recovery_revision": SOURCE_REVISION,
            "recovery_profile": profile,
            "exit_code": 0 if succeeded else 1,
            "status": "succeeded" if succeeded else "failed",
            "topology_gate_passed": True,
            "matrix_elapsed_seconds": 123.0,
        },
    )
    objects[
        prefix + f"{root}/checkpoints/cell/checkpoint.pth"
    ] = b"checkpoint"
    runtime_records = []
    for key, payload in sorted(objects.items()):
        relative = key[len(prefix) :]
        runtime_records.append(
            {
                "relative_path": relative,
                "s3_uri": f"s3://<DEV_BUCKET>/{key}",
                "sha256": _sha256(payload),
                "size_bytes": len(payload),
            }
        )
    _add_json(
        objects,
        prefix,
        "upload_complete.json",
        {
            "schema_version": 1,
            "run_id": run_id,
            "job_kind": NUMERICAL_RECOVERY_JOB,
            "method_revision": METHOD_REVISION,
            "source_revision": SOURCE_REVISION,
            "launcher_revision": LAUNCHER_REVISION,
            "recovery_revision": SOURCE_REVISION,
            "recovery_profile": profile,
            "uploaded_file_count": len(runtime_records),
            "uploaded_size_bytes": sum(
                record["size_bytes"] for record in runtime_records
            ),
            "objects": runtime_records,
        },
    )
    return objects


def _recovery_project(tmp_path):
    project = tmp_path / "project"
    docs = project / "docs"
    docs.mkdir(parents=True)
    manifest = docs / "timefuse_experiment_manifest.jsonl"
    manifest.write_bytes(
        Path("docs/timefuse_experiment_manifest.jsonl").read_bytes()
    )
    (docs / "timefuse_numerical_recovery_protocol.json").write_bytes(
        Path(NUMERICAL_RECOVERY_PROTOCOL).read_bytes()
    )
    rows_by_id = {row["id"]: row for row in load_manifest(manifest)}
    rows = [rows_by_id[cell_id] for cell_id in NUMERICAL_RECOVERY_CELL_IDS]
    return project, manifest, rows


def test_numerical_recovery_imports_fallback_and_exact_diagnostics(tmp_path):
    project, manifest, rows = _recovery_project(tmp_path)

    fallback_run = "numerical-recovery-fallback-fixture"
    fallback = fetch_greenland_numerical_recovery.fetch_numerical_recovery_run(
        _S3(
            _recovery_objects(
                rows,
                fallback_run,
                "fallback-v1",
            )
        ),
        fallback_run,
        project,
        project / "outputs" / fallback_run,
        manifest,
    )
    assert fallback["matrix_counts"]["completed"] == 9
    assert fallback["failed_cell_ids"] == []
    assert fallback["checkpoint_objects_retained_in_s3"] == 1
    fallback_summary = fallback["recomputed_matrix_summary"]
    fallback_summary_path = Path(fallback_summary["path"])
    assert fallback_summary_path.is_file()
    assert fallback_summary["sha256"] == _sha256(
        fallback_summary_path.read_bytes()
    )
    fallback_payload = json.loads(fallback_summary_path.read_text())
    assert fallback_payload["counts"]["completed"] == 9
    assert all(
        project in Path(row["artifact_dir"]).parents
        and Path(row["artifact_dir"], "status.json").is_file()
        and Path(row["artifact_dir"], "result.json").is_file()
        for row in fallback_payload["cell_states"]
    )

    exact_run = "numerical-recovery-exact-fixture"
    failed_id = rows[0]["id"]
    exact = fetch_greenland_numerical_recovery.fetch_numerical_recovery_run(
        _S3(
            _recovery_objects(
                rows,
                exact_run,
                "exact",
                failed_ids={failed_id},
            )
        ),
        exact_run,
        project,
        project / "outputs" / exact_run,
        manifest,
    )
    assert exact["matrix_counts"]["completed"] == 8
    assert exact["matrix_counts"]["failed"] == 1
    assert exact["failed_cell_ids"] == [failed_id]
    exact_summary_path = Path(exact["recomputed_matrix_summary"]["path"])
    exact_payload = json.loads(exact_summary_path.read_text())
    assert exact_payload["counts"]["completed"] == 8
    assert exact_payload["counts"]["failed"] == 1


def test_numerical_recovery_import_accepts_pinned_protocol_below_project(
    tmp_path,
):
    project, manifest, rows = _recovery_project(tmp_path)
    protocol = (
        project
        / "operations"
        / "controller"
        / "docs"
        / "timefuse_numerical_recovery_protocol.json"
    )
    protocol.parent.mkdir(parents=True)
    protocol.write_bytes(Path(NUMERICAL_RECOVERY_PROTOCOL).read_bytes())
    (
        project / "docs" / "timefuse_numerical_recovery_protocol.json"
    ).unlink()
    run_id = "numerical-recovery-pinned-protocol-fixture"

    receipt = (
        fetch_greenland_numerical_recovery.fetch_numerical_recovery_run(
            _S3(_recovery_objects(rows, run_id, "fallback-v1")),
            run_id,
            project,
            project / "outputs" / run_id,
            manifest,
            protocol=protocol,
        )
    )

    assert receipt["protocol_sha256"] == NUMERICAL_RECOVERY_PROTOCOL_SHA256


def test_numerical_recovery_import_rejects_nonfinite_completed_result(
    tmp_path,
):
    project, manifest, rows = _recovery_project(tmp_path)
    run_id = "numerical-recovery-nonfinite-fixture"

    with pytest.raises(ValueError, match="reconstruct"):
        fetch_greenland_numerical_recovery.fetch_numerical_recovery_run(
            _S3(
                _recovery_objects(
                    rows,
                    run_id,
                    "fallback-v1",
                    nonfinite_id=rows[0]["id"],
                )
            ),
            run_id,
            project,
            project / "outputs" / run_id,
            manifest,
        )


def test_numerical_recovery_import_rejects_non_integrity_exact_failure(
    tmp_path,
):
    project, manifest, rows = _recovery_project(tmp_path)
    run_id = "numerical-recovery-runtime-failure-fixture"

    with pytest.raises(ValueError, match="explicit NumericalIntegrityError"):
        fetch_greenland_numerical_recovery.fetch_numerical_recovery_run(
            _S3(
                _recovery_objects(
                    rows,
                    run_id,
                    "exact",
                    failed_ids={rows[0]["id"]},
                    failed_error_type="FileNotFoundError",
                )
            ),
            run_id,
            project,
            project / "outputs" / run_id,
            manifest,
        )
