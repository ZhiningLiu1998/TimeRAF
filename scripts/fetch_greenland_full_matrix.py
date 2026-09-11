"""Import and verify one Greenland full-matrix replication below Studio EFS."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from scripts import fetch_greenland_outputs as output_io
from scripts.greenland_common import (
    FULL_MATRIX_CELLS,
    FULL_MATRIX_MANIFEST_SHA256,
    FULL_MATRIX_REPLICATION_JOB,
    METHOD_REVISION,
    NUMERICAL_RECOVERY_CELL_IDS,
    NUMERICAL_RECOVERY_CELL_IDS_SHA256,
    NUMERICAL_RECOVERY_COHORT_ID,
    S3_ROOT,
    parse_s3_uri,
    sha256_file,
    validate_full_matrix_manifest,
    validate_gpu_topology_evidence,
    validate_job_spec,
)
from ts_rag.matrix import build_matrix_summary, load_manifest, save_json_atomic


def _validate_upload_marker(
    marker,
    records,
    run_id,
    job_kind=FULL_MATRIX_REPLICATION_JOB,
):
    if (
        marker.get("schema_version") != 1
        or marker.get("run_id") != run_id
        or marker.get("job_kind") != job_kind
    ):
        raise ValueError("Greenland upload completion marker is inconsistent")
    objects = marker.get("objects")
    if not isinstance(objects, list):
        raise ValueError("Greenland upload completion marker has no object list")
    indexed = {}
    for record in objects:
        relative = record.get("relative_path")
        if (
            not isinstance(relative, str)
            or not relative
            or relative in indexed
        ):
            raise ValueError(
                "Greenland upload completion marker has invalid object paths"
            )
        indexed[relative] = record
    runtime_records = {
        relative: record
        for relative, record in records.items()
        if relative != "upload_complete.json"
        and not relative.startswith("control/")
    }
    if set(indexed) != set(runtime_records):
        raise ValueError(
            "Greenland upload completion marker does not cover runtime outputs"
        )
    for relative, expected in indexed.items():
        actual = runtime_records[relative]
        if (
            expected.get("sha256") != actual["sha256"]
            or expected.get("size_bytes") != actual["size_bytes"]
            or expected.get("s3_uri")
            != f"s3://<DEV_BUCKET>/{actual['key']}"
        ):
            raise ValueError(
                f"Greenland uploaded object drifted after completion: {relative}"
            )
    if (
        marker.get("uploaded_file_count") != len(runtime_records)
        or marker.get("uploaded_size_bytes")
        != sum(record["size_bytes"] for record in runtime_records.values())
    ):
        raise ValueError("Greenland upload completion totals are inconsistent")
    return marker


def _validate_downloaded_run(
    destination,
    run_id,
    records,
    *,
    accept_terminal_nonfinite_first_pass=False,
):
    destination = Path(destination)
    spec = validate_job_spec(
        output_io._load_json(
            destination / "evidence" / "accepted_job_spec.json",
            "Accepted Greenland job spec",
        )
    )
    if (
        spec["run_id"] != run_id
        or spec["job_kind"] != FULL_MATRIX_REPLICATION_JOB
    ):
        raise ValueError("Accepted Greenland full-matrix spec is inconsistent")
    final_status = output_io._load_json(
        destination / "final_status.json",
        "Greenland final status",
    )
    topology = output_io._load_json(
        destination / "evidence" / "topology_summary.json",
        "Greenland topology evidence",
    )
    manifest_evidence = output_io._load_json(
        destination / "evidence" / "full_matrix_manifest.json",
        "Greenland full-matrix manifest evidence",
    )
    marker = output_io._load_json(
        destination / "upload_complete.json",
        "Greenland upload completion marker",
    )
    _validate_upload_marker(marker, records, run_id)
    summary_path = destination / output_io._output_relative(
        spec["matrix"]["summary"],
        "matrix.summary",
    )
    summary = output_io._load_json(
        summary_path,
        "Greenland full-matrix summary",
    )
    if (
        final_status.get("run_id") != run_id
        or final_status.get("job_kind") != FULL_MATRIX_REPLICATION_JOB
        or final_status.get("status") != "succeeded"
        or final_status.get("exit_code") != 0
        or final_status.get("topology_gate_passed") is not True
        or final_status.get("method_revision") != METHOD_REVISION
        or final_status.get("source_revision") != spec["source_revision"]
        or final_status.get("launcher_revision") != spec["launcher_revision"]
        or not isinstance(final_status.get("matrix_elapsed_seconds"), (int, float))
        or final_status["matrix_elapsed_seconds"] <= 0
    ):
        raise ValueError(
            "Greenland final status did not prove a successful full matrix"
        )
    validate_gpu_topology_evidence(topology)
    if (
        manifest_evidence.get("cells") != FULL_MATRIX_CELLS
        or manifest_evidence.get("sha256") != FULL_MATRIX_MANIFEST_SHA256
        or manifest_evidence.get("family_counts")
        != {"long_term": 364, "pems": 156, "epf": 65}
    ):
        raise ValueError("Greenland full-matrix manifest evidence drifted")
    counts = summary.get("counts", {})
    terminal_scope = (
        counts.get("expected") == FULL_MATRIX_CELLS
        and counts.get("completed", 0) + counts.get("incomplete", 0)
        == FULL_MATRIX_CELLS
        and counts.get("failed") == 0
        and counts.get("pending") == 0
        and counts.get("running") == 0
    )
    strict_scope = (
        summary.get("all_completed") is True
        and counts.get("completed") == FULL_MATRIX_CELLS
        and counts.get("incomplete") == 0
    )
    if (
        summary.get("source_revision") != METHOD_REVISION
        or summary.get("export_only") is not False
        or not terminal_scope
        or (
            not accept_terminal_nonfinite_first_pass
            and not strict_scope
        )
    ):
        raise ValueError(
            "Greenland full-matrix summary is incomplete or inconsistent"
        )
    if (
        marker.get("method_revision") != spec["method_revision"]
        or marker.get("source_revision") != spec["source_revision"]
        or marker.get("launcher_revision") != spec["launcher_revision"]
    ):
        raise ValueError("Greenland upload marker revision identity drifted")
    return spec, final_status, topology, summary, marker


def _file_evidence(path):
    path = Path(path).resolve()
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _nonfinite_metric_evidence(result, metrics):
    sections = {}
    evidence = []
    for section in ("test_baseline", "test_corrected"):
        values = result.get(section)
        if not isinstance(values, dict):
            raise ValueError(
                "Terminal first-pass result is missing a paper metric section"
            )
        sections[section] = values
        for metric in metrics:
            value = values.get(metric)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(
                    "Terminal first-pass result has a missing or non-numeric "
                    "paper metric"
                )
            if not math.isfinite(float(value)):
                evidence.append(
                    {
                        "section": section,
                        "metric": metric,
                        "kind": (
                            "nan"
                            if math.isnan(float(value))
                            else "infinite"
                        ),
                    }
                )
    for metric in metrics:
        before = float(sections["test_baseline"][metric])
        after = float(sections["test_corrected"][metric])
        if not math.isfinite(before) or not math.isfinite(after) or before == 0:
            continue
        gain = 100.0 * (before - after) / abs(before)
        if not math.isfinite(gain):
            evidence.append(
                {
                    "section": "metric_gain_percent",
                    "metric": metric,
                    "kind": "nonfinite-derived-gain",
                }
            )
    if not evidence:
        raise ValueError(
            "Incomplete terminal first-pass cell is not proven non-finite"
        )
    return evidence


def _validate_import_acceptance(
    recomputed_summary,
    manifest,
    matrix_output,
    *,
    accept_terminal_nonfinite_first_pass=False,
):
    counts = recomputed_summary["counts"]
    if (
        counts.get("expected") != FULL_MATRIX_CELLS
        or counts.get("failed") != 0
        or counts.get("pending") != 0
        or counts.get("running") != 0
    ):
        raise ValueError(
            "Imported Greenland artifacts do not reconstruct a terminal "
            "first-pass matrix"
        )
    incomplete = [
        row
        for row in recomputed_summary["cell_states"]
        if row["state"] == "incomplete"
    ]
    if not accept_terminal_nonfinite_first_pass:
        if (
            recomputed_summary["all_completed"] is not True
            or counts.get("completed") != FULL_MATRIX_CELLS
            or counts.get("incomplete") != 0
        ):
            raise ValueError(
                "Imported Greenland result/status artifacts do not "
                "reconstruct a complete full matrix"
            )
        return {
            "mode": "strict-complete",
            "terminal_attempt_count": FULL_MATRIX_CELLS,
            "finite_completed_count": FULL_MATRIX_CELLS,
            "nonfinite_completed_count": 0,
            "nonfinite_cell_ids": [],
            "nonfinite_artifacts": [],
        }

    if (
        counts.get("completed", 0) + counts.get("incomplete", 0)
        != FULL_MATRIX_CELLS
    ):
        raise ValueError(
            "Terminal first-pass import does not cover all matrix cells"
        )
    manifest_by_id = {row["id"]: row for row in manifest}
    if len(manifest_by_id) != FULL_MATRIX_CELLS:
        raise ValueError("Full matrix manifest contains duplicate cell IDs")
    nonfinite_ids = sorted(row["cell_id"] for row in incomplete)
    outside_cohort = set(nonfinite_ids) - set(NUMERICAL_RECOVERY_CELL_IDS)
    if outside_cohort:
        raise ValueError(
            "Terminal first-pass non-finite cells fall outside the frozen "
            f"recovery cohort: {sorted(outside_cohort)}"
        )

    matrix_output = Path(matrix_output).resolve()
    nonfinite_artifacts = []
    for row in incomplete:
        if row.get("numerical_integrity_error") != (
            "completed result has missing or non-finite paper metrics"
        ):
            raise ValueError(
                "Incomplete terminal first-pass cell is not a numerical "
                "integrity outcome"
            )
        artifact_dir = Path(row.get("artifact_dir", "")).resolve()
        if (
            artifact_dir == matrix_output
            or matrix_output not in artifact_dir.parents
        ):
            raise ValueError(
                "Terminal first-pass artifact is outside the matrix output"
            )
        status_path = artifact_dir / "status.json"
        result_path = artifact_dir / "result.json"
        status = output_io._load_json(
            status_path,
            "Terminal first-pass status",
        )
        result = output_io._load_json(
            result_path,
            "Terminal first-pass result",
        )
        if (
            status.get("status") != "completed"
            or status.get("cell_id") != row["cell_id"]
            or status.get("source_revision") != METHOD_REVISION
            or status.get("seed") != 2021
            or status.get("smoke") is not False
        ):
            raise ValueError(
                "Terminal first-pass non-finite status identity drifted"
            )
        metric_evidence = _nonfinite_metric_evidence(
            result,
            manifest_by_id[row["cell_id"]]["metrics"],
        )
        nonfinite_artifacts.append(
            {
                "cell_id": row["cell_id"],
                "artifact_dir": str(artifact_dir),
                "status": _file_evidence(status_path),
                "result": _file_evidence(result_path),
                "metric_evidence": metric_evidence,
            }
        )
    nonfinite_artifacts.sort(key=lambda row: row["cell_id"])
    return {
        "mode": "terminal-nonfinite-first-pass",
        "terminal_attempt_count": FULL_MATRIX_CELLS,
        "finite_completed_count": counts["completed"],
        "nonfinite_completed_count": counts["incomplete"],
        "nonfinite_cell_ids": nonfinite_ids,
        "nonfinite_cell_ids_sha256": hashlib.sha256(
            "".join(f"{cell_id}\n" for cell_id in nonfinite_ids).encode(
                "utf-8"
            )
        ).hexdigest(),
        "recovery_cohort_id": NUMERICAL_RECOVERY_COHORT_ID,
        "recovery_cohort_cell_ids_sha256": (
            NUMERICAL_RECOVERY_CELL_IDS_SHA256
        ),
        "nonfinite_artifacts": nonfinite_artifacts,
    }


def fetch_full_matrix_run(
    s3,
    run_id,
    project_root,
    destination,
    manifest,
    *,
    receipt_path=None,
    required_filesystem=None,
    download_checkpoints=False,
    accept_terminal_nonfinite_first_pass=False,
):
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{5,62}", run_id):
        raise ValueError("run_id must be a lowercase K8s-safe identifier")
    project_root = Path(project_root).resolve()
    if not project_root.is_dir():
        raise ValueError("project_root must be an existing directory")
    if (
        required_filesystem
        and output_io._filesystem_type(project_root) != required_filesystem
    ):
        raise ValueError(
            f"project_root must use {required_filesystem} durable storage"
        )
    destination = output_io._below_project(
        project_root,
        destination,
        "destination",
    )
    manifest = Path(manifest).resolve()
    if manifest != project_root and project_root not in manifest.parents:
        raise ValueError("manifest must live below the project root")
    validate_full_matrix_manifest(manifest)

    prefix_uri = f"{S3_ROOT}runs/{run_id}/"
    bucket, prefix = parse_s3_uri(prefix_uri)
    records = output_io._list_objects(s3, bucket, prefix)
    if "upload_complete.json" not in records:
        raise ValueError("Greenland run upload is not complete")
    selected = {
        relative: record
        for relative, record in records.items()
        if download_checkpoints
        or not relative.startswith(
            "full_matrix_replication/checkpoints/"
        )
    }
    imported = output_io._download_objects(
        s3,
        bucket,
        selected,
        destination,
    )
    spec, final_status, topology, summary, marker = _validate_downloaded_run(
        destination,
        run_id,
        records,
        accept_terminal_nonfinite_first_pass=(
            accept_terminal_nonfinite_first_pass
        ),
    )
    matrix_output = (
        destination
        / output_io._output_relative(
            spec["matrix"]["output_root"],
            "matrix.output_root",
        )
    )
    manifest_rows = load_manifest(manifest)
    recomputed_summary = build_matrix_summary(
        manifest_rows,
        matrix_output,
        seed=int(spec["matrix"]["seed"]),
        source_revision=METHOD_REVISION,
    )
    recomputed_counts = recomputed_summary["counts"]
    import_acceptance = _validate_import_acceptance(
        recomputed_summary,
        manifest_rows,
        matrix_output,
        accept_terminal_nonfinite_first_pass=(
            accept_terminal_nonfinite_first_pass
        ),
    )
    recomputed_summary["generated_unix"] = summary.get(
        "generated_unix",
        final_status.get(
            "finished_unix",
            marker.get("completed_unix", 0.0),
        ),
    )
    recomputed_summary_path = destination / "recomputed_matrix_summary.json"
    save_json_atomic(recomputed_summary, recomputed_summary_path)
    recomputed_summary_evidence = {
        "path": str(recomputed_summary_path),
        "sha256": sha256_file(recomputed_summary_path),
        "size_bytes": recomputed_summary_path.stat().st_size,
    }
    receipt_path = output_io._below_project(
        project_root,
        receipt_path or destination / "fetch_receipt.json",
        "receipt_path",
    )
    skipped = set(records) - set(selected)
    receipt = {
        "schema_version": 1,
        "run_id": run_id,
        "job_kind": FULL_MATRIX_REPLICATION_JOB,
        "source_s3_prefix": prefix_uri,
        "project_root": str(project_root),
        "destination": str(destination),
        "method_revision": spec["method_revision"],
        "source_revision": spec["source_revision"],
        "launcher_revision": spec["launcher_revision"],
        "object_count": len(records),
        "total_size_bytes": sum(
            record["size_bytes"] for record in records.values()
        ),
        "imported_object_count": len(imported),
        "imported_size_bytes": sum(
            record["size_bytes"] for record in imported.values()
        ),
        "downloaded_count": sum(
            record["mode"] == "downloaded" for record in imported.values()
        ),
        "reused_count": sum(
            record["mode"] == "reused" for record in imported.values()
        ),
        "checkpoint_objects_retained_in_s3": len(skipped),
        "checkpoint_bytes_retained_in_s3": sum(
            records[relative]["size_bytes"] for relative in skipped
        ),
        "topology_gate_passed": topology["topology_gate_passed"],
        "topology_evidence": _file_evidence(
            destination / "evidence" / "topology_summary.json"
        ),
        "matrix_counts": recomputed_counts,
        "stored_matrix_counts": summary["counts"],
        "import_acceptance": import_acceptance,
        "matrix_elapsed_seconds": final_status["matrix_elapsed_seconds"],
        "matrix_output": str(matrix_output),
        "recomputed_matrix_summary": recomputed_summary_evidence,
        "upload_marker_sha256": hashlib.sha256(
            (destination / "upload_complete.json").read_bytes()
        ).hexdigest(),
        "objects": imported,
        "final_status": final_status,
        "upload_marker": marker,
    }
    save_json_atomic(receipt, receipt_path)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Import a hash-verified Greenland full matrix into EFS"
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--destination")
    parser.add_argument(
        "--manifest",
        default="docs/timefuse_experiment_manifest.jsonl",
    )
    parser.add_argument("--receipt")
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--download-checkpoints", action="store_true")
    parser.add_argument(
        "--accept-terminal-nonfinite-first-pass",
        action="store_true",
        help=(
            "accept all-terminal first-pass artifacts only when every "
            "incomplete row is a hash-bound non-finite result inside the "
            "frozen recovery cohort"
        ),
    )
    args = parser.parse_args(argv)

    project_root = Path(args.project_root).resolve()
    destination = args.destination or (
        project_root / "outputs" / "greenland_runs" / args.run_id
    )
    manifest = (
        project_root / args.manifest
        if not Path(args.manifest).is_absolute()
        else Path(args.manifest)
    )
    receipt = fetch_full_matrix_run(
        output_io._assumed_s3_client(args.region),
        args.run_id,
        project_root,
        destination,
        manifest,
        receipt_path=args.receipt,
        required_filesystem="nfs4",
        download_checkpoints=args.download_checkpoints,
        accept_terminal_nonfinite_first_pass=(
            args.accept_terminal_nonfinite_first_pass
        ),
    )
    print(
        json.dumps(
            {
                "run_id": receipt["run_id"],
                "object_count": receipt["object_count"],
                "imported_object_count": receipt["imported_object_count"],
                "checkpoint_objects_retained_in_s3": receipt[
                    "checkpoint_objects_retained_in_s3"
                ],
                "topology_gate_passed": receipt["topology_gate_passed"],
                "matrix_counts": receipt["matrix_counts"],
                "matrix_elapsed_seconds": receipt[
                    "matrix_elapsed_seconds"
                ],
                "matrix_output": receipt["matrix_output"],
                "recomputed_matrix_summary": receipt[
                    "recomputed_matrix_summary"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
