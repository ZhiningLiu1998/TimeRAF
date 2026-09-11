"""Import and verify one Greenland numerical-recovery profile below EFS."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from scripts import fetch_greenland_outputs as output_io
from scripts.fetch_greenland_full_matrix import _validate_upload_marker
from scripts.greenland_common import (
    FULL_MATRIX_MANIFEST_SHA256,
    METHOD_REVISION,
    NUMERICAL_RECOVERY_CELL_IDS,
    NUMERICAL_RECOVERY_CELLS,
    NUMERICAL_RECOVERY_JOB,
    NUMERICAL_RECOVERY_PROTOCOL,
    S3_ROOT,
    parse_s3_uri,
    sha256_file,
    validate_full_matrix_manifest,
    validate_gpu_topology_evidence,
    validate_job_spec,
    validate_numerical_recovery_protocol,
)
from ts_rag.matrix import build_matrix_summary, load_manifest, save_json_atomic


def _validate_exact_failure_types(summary, label):
    failed = summary.get("failed_cells")
    if not isinstance(failed, list):
        raise ValueError(f"{label} has no failed-cell evidence")
    if any(
        row.get("state") != "failed"
        or row.get("error_type") != "NumericalIntegrityError"
        for row in failed
    ):
        raise ValueError(
            f"{label} exact failures must be explicit NumericalIntegrityError"
        )
    return failed


def _validate_downloaded_run(destination, run_id, records):
    destination = Path(destination)
    spec = validate_job_spec(
        output_io._load_json(
            destination / "evidence" / "accepted_job_spec.json",
            "Accepted Greenland job spec",
        )
    )
    if (
        spec["run_id"] != run_id
        or spec["job_kind"] != NUMERICAL_RECOVERY_JOB
    ):
        raise ValueError("Accepted numerical recovery spec is inconsistent")
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
    recovery_evidence = output_io._load_json(
        destination / "evidence" / "numerical_recovery_protocol.json",
        "Greenland numerical recovery evidence",
    )
    marker = output_io._load_json(
        destination / "upload_complete.json",
        "Greenland upload completion marker",
    )
    _validate_upload_marker(
        marker,
        records,
        run_id,
        job_kind=NUMERICAL_RECOVERY_JOB,
    )
    summary_path = destination / output_io._output_relative(
        spec["matrix"]["summary"],
        "matrix.summary",
    )
    summary = output_io._load_json(
        summary_path,
        "Greenland numerical recovery summary",
    )
    profile = spec["recovery"]["profile"]
    allowed_terminal = (
        {(0, "succeeded")}
        if profile == "fallback-v1"
        else {(0, "succeeded"), (1, "failed")}
    )
    if (
        final_status.get("run_id") != run_id
        or final_status.get("job_kind") != NUMERICAL_RECOVERY_JOB
        or (
            final_status.get("exit_code"),
            final_status.get("status"),
        )
        not in allowed_terminal
        or final_status.get("topology_gate_passed") is not True
        or final_status.get("method_revision") != METHOD_REVISION
        or final_status.get("source_revision") != spec["source_revision"]
        or final_status.get("launcher_revision")
        != spec["launcher_revision"]
        or final_status.get("recovery_revision")
        != spec["recovery_revision"]
        or final_status.get("recovery_profile") != profile
        or not isinstance(
            final_status.get("matrix_elapsed_seconds"),
            (int, float),
        )
        or final_status["matrix_elapsed_seconds"] <= 0
    ):
        raise ValueError(
            "Greenland final status is invalid for numerical recovery"
        )
    validate_gpu_topology_evidence(topology)
    if (
        manifest_evidence.get("cells") != 585
        or manifest_evidence.get("sha256")
        != FULL_MATRIX_MANIFEST_SHA256
        or recovery_evidence.get("sha256")
        != spec["recovery"]["protocol_sha256"]
        or recovery_evidence.get("cohort_id")
        != spec["recovery"]["cohort_id"]
        or recovery_evidence.get("cell_ids")
        != spec["recovery"]["cell_ids"]
        or recovery_evidence.get("cell_ids_sha256")
        != spec["recovery"]["cell_ids_sha256"]
    ):
        raise ValueError("Numerical recovery evidence drifted")
    counts = summary.get("counts", {})
    if (
        summary.get("source_revision") != METHOD_REVISION
        or summary.get("export_only") is not False
        or counts.get("expected") != NUMERICAL_RECOVERY_CELLS
        or counts.get("pending") != 0
        or counts.get("running") != 0
        or counts.get("incomplete") != 0
        or counts.get("completed", 0) + counts.get("failed", 0)
        != NUMERICAL_RECOVERY_CELLS
        or (
            profile == "fallback-v1"
            and (
                summary.get("all_completed") is not True
                or counts.get("completed") != NUMERICAL_RECOVERY_CELLS
                or counts.get("failed") != 0
            )
        )
    ):
        raise ValueError(
            "Stored numerical recovery summary has invalid profile scope"
        )
    if profile == "exact":
        _validate_exact_failure_types(
            summary,
            "Stored numerical recovery summary",
        )
    if (
        marker.get("method_revision") != spec["method_revision"]
        or marker.get("source_revision") != spec["source_revision"]
        or marker.get("launcher_revision") != spec["launcher_revision"]
        or marker.get("recovery_revision") != spec["recovery_revision"]
        or marker.get("recovery_profile") != profile
    ):
        raise ValueError("Numerical recovery upload marker identity drifted")
    return spec, final_status, topology, summary, marker


def fetch_numerical_recovery_run(
    s3,
    run_id,
    project_root,
    destination,
    manifest,
    *,
    protocol=None,
    receipt_path=None,
    required_filesystem=None,
    download_checkpoints=False,
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
    protocol = Path(
        protocol or project_root / NUMERICAL_RECOVERY_PROTOCOL
    ).resolve()
    if protocol != project_root and project_root not in protocol.parents:
        raise ValueError("protocol must live below the project root")
    protocol_evidence = validate_numerical_recovery_protocol(protocol)

    prefix_uri = f"{S3_ROOT}runs/{run_id}/"
    bucket, prefix = parse_s3_uri(prefix_uri)
    records = output_io._list_objects(s3, bucket, prefix)
    if "upload_complete.json" not in records:
        raise ValueError("Greenland numerical recovery upload is incomplete")
    selected = dict(records)
    if not download_checkpoints:
        selected = {
            relative: record
            for relative, record in selected.items()
            if "/checkpoints/" not in relative
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
    )
    if (
        protocol_evidence["sha256"]
        != spec["recovery"]["protocol_sha256"]
    ):
        raise ValueError(
            "Importer numerical recovery protocol differs from the run"
        )
    manifest_rows = load_manifest(manifest)
    rows_by_id = {row["id"]: row for row in manifest_rows}
    if len(rows_by_id) != len(manifest_rows):
        raise ValueError("Full matrix manifest contains duplicate cell IDs")
    recovery_rows = [
        rows_by_id[cell_id] for cell_id in NUMERICAL_RECOVERY_CELL_IDS
    ]
    matrix_output = destination / output_io._output_relative(
        spec["matrix"]["output_root"],
        "matrix.output_root",
    )
    recomputed_summary = build_matrix_summary(
        recovery_rows,
        matrix_output,
        seed=int(spec["matrix"]["seed"]),
        source_revision=METHOD_REVISION,
    )
    counts = recomputed_summary["counts"]
    profile = spec["recovery"]["profile"]
    if (
        counts.get("expected") != NUMERICAL_RECOVERY_CELLS
        or counts.get("pending") != 0
        or counts.get("running") != 0
        or counts.get("incomplete") != 0
        or counts.get("completed", 0) + counts.get("failed", 0)
        != NUMERICAL_RECOVERY_CELLS
        or (
            profile == "fallback-v1"
            and (
                recomputed_summary["all_completed"] is not True
                or counts.get("completed") != NUMERICAL_RECOVERY_CELLS
                or counts.get("failed") != 0
            )
        )
    ):
        raise ValueError(
            "Imported artifacts do not reconstruct the recovery profile"
        )
    if counts != summary["counts"]:
        raise ValueError(
            "Stored and independently recomputed recovery counts differ"
        )
    if profile == "exact":
        _validate_exact_failure_types(
            recomputed_summary,
            "Recomputed numerical recovery summary",
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
        "job_kind": NUMERICAL_RECOVERY_JOB,
        "profile": profile,
        "source_s3_prefix": prefix_uri,
        "project_root": str(project_root),
        "destination": str(destination),
        "method_revision": spec["method_revision"],
        "source_revision": spec["source_revision"],
        "launcher_revision": spec["launcher_revision"],
        "recovery_revision": spec["recovery_revision"],
        "protocol_sha256": protocol_evidence["sha256"],
        "cell_ids_sha256": protocol_evidence["cell_ids_sha256"],
        "object_count": len(records),
        "imported_object_count": len(imported),
        "imported_size_bytes": sum(
            record["size_bytes"] for record in imported.values()
        ),
        "checkpoint_objects_retained_in_s3": len(skipped),
        "checkpoint_bytes_retained_in_s3": sum(
            records[relative]["size_bytes"] for relative in skipped
        ),
        "topology_gate_passed": topology["topology_gate_passed"],
        "matrix_counts": counts,
        "failed_cell_ids": [
            row["cell_id"] for row in recomputed_summary["failed_cells"]
        ],
        "failed_cell_errors": [
            {
                "cell_id": row["cell_id"],
                "error_type": row.get("error_type"),
            }
            for row in recomputed_summary["failed_cells"]
        ],
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
        description="Import a hash-verified Greenland numerical recovery"
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--destination")
    parser.add_argument(
        "--manifest",
        default="docs/timefuse_experiment_manifest.jsonl",
    )
    parser.add_argument(
        "--protocol",
        default=NUMERICAL_RECOVERY_PROTOCOL,
    )
    parser.add_argument("--receipt")
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--download-checkpoints", action="store_true")
    args = parser.parse_args(argv)

    project_root = Path(args.project_root).resolve()
    destination = args.destination or (
        project_root / "outputs" / "greenland_recovery_runs" / args.run_id
    )
    manifest = (
        project_root / args.manifest
        if not Path(args.manifest).is_absolute()
        else Path(args.manifest)
    )
    protocol = (
        project_root / args.protocol
        if not Path(args.protocol).is_absolute()
        else Path(args.protocol)
    )
    receipt = fetch_numerical_recovery_run(
        output_io._assumed_s3_client(args.region),
        args.run_id,
        project_root,
        destination,
        manifest,
        protocol=protocol,
        receipt_path=args.receipt,
        required_filesystem="nfs4",
        download_checkpoints=args.download_checkpoints,
    )
    print(
        json.dumps(
            {
                "run_id": receipt["run_id"],
                "profile": receipt["profile"],
                "matrix_counts": receipt["matrix_counts"],
                "topology_gate_passed": receipt["topology_gate_passed"],
                "checkpoint_objects_retained_in_s3": receipt[
                    "checkpoint_objects_retained_in_s3"
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
