"""Shared validation and evidence helpers for TimeRAF Greenland jobs."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import tarfile
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse


INSTANCE_TYPE = "p4de.24xlarge"
SAGEMAKER_INSTANCE_TYPE = f"ml.{INSTANCE_TYPE}"
GPUS_PER_HOST = 8
PROCESSES_PER_HOST = 8
REPLAY_CELLS = 208
FULL_MATRIX_CELLS = 585
NUMERICAL_RECOVERY_CELLS = 9
METHOD_REVISION = "d9be338"
CHECKPOINT_REPLAY_JOB = "checkpoint_replay"
FULL_MATRIX_REPLICATION_JOB = "full_matrix_replication"
NUMERICAL_RECOVERY_JOB = "supplemental_numerical_recovery"
REPLAY_MANIFEST = "docs/timefuse_checkpoint_replay_manifest.jsonl"
REPLAY_MANIFEST_SHA256 = (
    "447391d99c46bc1dd4170e71a8388bad"
    "5edb0a4c48a3048e230e404b96a0a77a"
)
REPLAY_HORIZONS = {
    "long_term": 96,
    "pems": 24,
    "epf": 24,
}
FULL_MATRIX_MANIFEST = "docs/timefuse_experiment_manifest.jsonl"
FULL_MATRIX_MANIFEST_SHA256 = (
    "45830d23f3b017c15d3680c88f837f84"
    "a9441eef9a6ceaaa5370c14872660c2a"
)
NUMERICAL_RECOVERY_PROTOCOL = (
    "docs/timefuse_numerical_recovery_protocol.json"
)
NUMERICAL_RECOVERY_PROTOCOL_SHA256 = (
    "1320f3eca815a93a6500caadbf7e52661"
    "0379b2af5f6976f1db94f91ed259d14"
)
NUMERICAL_RECOVERY_COHORT_ID = (
    "nonstationary-transformer-numerical-recovery-v1"
)
NUMERICAL_RECOVERY_CELL_IDS = (
    "epf/BE/Nonstationary_Transformer/24",
    "epf/DE/Nonstationary_Transformer/24",
    "epf/FR/Nonstationary_Transformer/24",
    "epf/NP/Nonstationary_Transformer/24",
    "epf/PJM/Nonstationary_Transformer/24",
    "long_term/electricity/Nonstationary_Transformer/192",
    "long_term/electricity/Nonstationary_Transformer/336",
    "long_term/electricity/Nonstationary_Transformer/720",
    "long_term/electricity/Nonstationary_Transformer/96",
)
NUMERICAL_RECOVERY_CELL_IDS_SHA256 = (
    "1bc266570503ab1de56c9819129b72333"
    "2ae1b597071aeea473fdd2d90451f3a"
)
NUMERICAL_RECOVERY_PROFILES = {
    "exact": {},
    "fallback-v1": {"learning_rate": 0.0001},
}
INITIATIVE_ID = "feedml-sp-os"
REGION = "us-east-1"
SERVICE_ROLE_ARN = (
    "arn:aws:iam::<AWS_ACCOUNT_ID>:role/"
    "<GREENLAND_SERVICE_ROLE>"
)
S3_ROOT = "s3://<DEV_BUCKET>/timeraf/greenland/"
APPROVED_IMAGE_RECORD = "docs/greenland_approved_image.json"
IMAGE_PATTERN = re.compile(
    r"^<AWS_ACCOUNT_ID>\.dkr\.ecr\.us-east-1\.amazonaws\.com/"
    r"[a-z0-9][a-z0-9._/-]*@sha256:[0-9a-f]{64}$"
)
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")

_RESERVED_MATRIX_FLAGS = {
    "--checkpoint-catalog",
    "--checkpoint-root",
    "--cell-id",
    "--dataset",
    "--dry-run",
    "--family",
    "--force-train",
    "--gpu",
    "--instance-type",
    "--launcher-revision",
    "--log-root",
    "--manifest",
    "--max-cells",
    "--model",
    "--output-root",
    "--override",
    "--pred-len",
    "--processes-per-host",
    "--release-checkpoint-root",
    "--release-only",
    "--reserved-gpus-per-host",
    "--rerun-completed",
    "--smoke",
    "--smoke-eval-samples",
    "--smoke-train-batches",
    "--source-revision",
    "--summary",
}


def canonical_json_bytes(payload):
    return (
        json.dumps(payload, indent=2, sort_keys=True, separators=(",", ": "))
        + "\n"
    ).encode("utf-8")


def sha256_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        while True:
            chunk = source.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def parse_s3_uri(uri):
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.strip("/"):
        raise ValueError(f"Expected an object S3 URI, got {uri!r}")
    return parsed.netloc, parsed.path.lstrip("/")


def validate_image_uri(image_uri):
    if not IMAGE_PATTERN.fullmatch(str(image_uri)):
        raise ValueError(
            "Greenland image must be an immutable modeldev ECR URI using "
            "@sha256:<64 lowercase hex characters>"
        )
    return image_uri


def validate_approved_image_uri(
    image_uri,
    approval_record,
    required_job_kind=CHECKPOINT_REPLAY_JOB,
):
    validate_image_uri(image_uri)
    with Path(approval_record).open("r", encoding="utf-8") as source:
        approval = json.load(source)
    if approval.get("schema_version") != 1:
        raise ValueError("Unsupported Greenland image approval schema")
    if approval.get("image_uri") != image_uri:
        raise ValueError("Greenland image is not the approved immutable digest")
    digest = str(image_uri).rsplit("@", 1)[-1]
    if approval.get("image_digest") != digest:
        raise ValueError("Approved Greenland image digest is inconsistent")
    source_revision = approval.get("image_source_revision")
    if (
        not isinstance(source_revision, str)
        or not re.fullmatch(r"[0-9a-f]{40}", source_revision)
    ):
        raise ValueError("Approved image source revision is invalid")
    if required_job_kind not in approval.get("approved_for", []):
        raise ValueError(
            f"Greenland image is not approved for {required_job_kind}"
        )
    verification = approval.get("black_box_verification", {})
    required_checks = {
        "pip_check_passed",
        "png_jpeg_codecs_passed",
        "matplotlib_render_passed",
        "checkpoint_preflight_present",
        "removed_development_packages_absent",
    }
    if required_job_kind == NUMERICAL_RECOVERY_JOB:
        required_checks.add("numerical_recovery_preflight_present")
    if verification.get("paper_model_imports_passed") != 13 or any(
        verification.get(check) is not True for check in required_checks
    ):
        raise ValueError("Greenland image black-box verification is incomplete")
    scan = approval.get("ecr_scan", {})
    if (
        scan.get("status") != "COMPLETE"
        or scan.get("finding_severity_counts") != {}
    ):
        raise ValueError("Greenland image ECR scan is not zero-finding complete")
    return approval


def _relative_path(value, field, required_prefix=None):
    path = PurePosixPath(str(value))
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError(f"{field} must be a safe relative path")
    if required_prefix and path.parts[0] != required_prefix:
        raise ValueError(f"{field} must be below {required_prefix}/")
    return str(path)


def _validate_content_addressed_input(record):
    required = {"name", "s3_uri", "sha256", "extract_to"}
    missing = required - set(record)
    if missing:
        raise ValueError(
            f"Input record is missing: {', '.join(sorted(missing))}"
        )
    digest = str(record["sha256"])
    if not SHA256_PATTERN.fullmatch(digest):
        raise ValueError(f"Invalid SHA-256 for input {record['name']!r}")
    bucket, key = parse_s3_uri(record["s3_uri"])
    if bucket != "<DEV_BUCKET>":
        raise ValueError("Greenland inputs must use <DEV_BUCKET>")
    expected_prefix = f"timeraf/greenland/inputs/{record['name']}/"
    if not key.startswith(expected_prefix):
        raise ValueError(
            f"Input {record['name']!r} must be below {expected_prefix}"
        )
    if not PurePosixPath(key).name.startswith(digest):
        raise ValueError(
            f"Input {record['name']!r} key must begin with its SHA-256"
        )
    if record["extract_to"] != "project":
        raise ValueError("All TimeRAF archives must extract below project/")


def validate_job_spec(spec, expected_image_uri=None):
    run_id = str(spec.get("run_id", ""))
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{5,62}", run_id):
        raise ValueError("run_id must be a lowercase K8s-safe identifier")
    job_kind = spec.get("job_kind")
    if job_kind not in {
        CHECKPOINT_REPLAY_JOB,
        FULL_MATRIX_REPLICATION_JOB,
        NUMERICAL_RECOVERY_JOB,
    }:
        raise ValueError("Unsupported Greenland job kind")
    expected_schema = 2 if job_kind == NUMERICAL_RECOVERY_JOB else 1
    if spec.get("schema_version") != expected_schema:
        raise ValueError(
            f"{job_kind} requires Greenland job spec schema "
            f"{expected_schema}"
        )
    if spec.get("method_revision") != METHOD_REVISION:
        raise ValueError(
            f"Greenland method revision must be {METHOD_REVISION}"
        )

    image_uri = validate_image_uri(spec.get("image_uri", ""))
    if expected_image_uri and image_uri != expected_image_uri:
        raise ValueError("Job spec image URI does not match submitted image")

    revisions = (spec.get("source_revision"), spec.get("launcher_revision"))
    if any(
        not re.fullmatch(r"[0-9a-f]{7,40}", str(revision or ""))
        for revision in revisions
    ):
        raise ValueError("Source and launcher revisions must be Git object IDs")
    if job_kind == NUMERICAL_RECOVERY_JOB:
        recovery_revision = str(spec.get("recovery_revision", ""))
        if (
            not re.fullmatch(r"[0-9a-f]{40}", recovery_revision)
            or recovery_revision != spec.get("source_revision")
        ):
            raise ValueError(
                "Numerical recovery revision must equal the full source "
                "revision"
            )

    inputs = spec.get("inputs")
    if not isinstance(inputs, list):
        raise ValueError("inputs must be a list")
    names = [record.get("name") for record in inputs]
    expected_inputs = (
        ["checkpoints", "dataset", "source"]
        if job_kind == CHECKPOINT_REPLAY_JOB
        else ["dataset", "source"]
    )
    if sorted(names) != expected_inputs:
        raise ValueError(
            f"{job_kind} requires inputs {expected_inputs}"
        )
    for record in inputs:
        _validate_content_addressed_input(record)

    output_uri = str(spec.get("output_s3_uri", ""))
    expected_output = f"{S3_ROOT}runs/{run_id}/"
    if output_uri != expected_output:
        raise ValueError(f"output_s3_uri must equal {expected_output}")

    matrix = spec.get("matrix")
    if not isinstance(matrix, dict):
        raise ValueError("matrix must be an object")
    common_fields = (
        "manifest",
        "output_root",
        "checkpoint_root",
        "summary",
        "log_root",
    )
    replay_fields = (
        "checkpoint_catalog",
        "release_checkpoint_root",
    )
    for field in common_fields + (
        replay_fields if job_kind == CHECKPOINT_REPLAY_JOB else ()
    ):
        _relative_path(
            matrix.get(field, ""),
            f"matrix.{field}",
            required_prefix="outputs"
            if field in {
                "output_root",
                "summary",
                "log_root",
                *(
                    {"checkpoint_root"}
                    if job_kind
                    in {
                        FULL_MATRIX_REPLICATION_JOB,
                        NUMERICAL_RECOVERY_JOB,
                    }
                    else set()
                ),
            }
            else None,
        )
    expected_manifest = (
        REPLAY_MANIFEST
        if job_kind == CHECKPOINT_REPLAY_JOB
        else FULL_MATRIX_MANIFEST
    )
    expected_cells = {
        CHECKPOINT_REPLAY_JOB: REPLAY_CELLS,
        FULL_MATRIX_REPLICATION_JOB: FULL_MATRIX_CELLS,
        NUMERICAL_RECOVERY_JOB: NUMERICAL_RECOVERY_CELLS,
    }[job_kind]
    if matrix.get("manifest") != expected_manifest:
        job_label = job_kind.replace("_", " ")
        raise ValueError(
            f"{job_label} manifest must be {expected_manifest}"
        )
    if int(matrix.get("queued_cells", 0)) != expected_cells:
        raise ValueError(
            f"{job_kind} must declare exactly {expected_cells} cells"
        )
    if int(matrix.get("seed", 0)) < 0:
        raise ValueError("matrix.seed must be non-negative")

    matrix_args = matrix.get("extra_args", [])
    if not isinstance(matrix_args, list) or not all(
        isinstance(value, str) for value in matrix_args
    ):
        raise ValueError("matrix.extra_args must be a list of strings")
    for value in matrix_args:
        flag = value.split("=", 1)[0]
        if flag in _RESERVED_MATRIX_FLAGS:
            raise ValueError(f"matrix.extra_args cannot override {flag}")
        if flag == "--cpu":
            raise ValueError("A p4de job cannot run in CPU mode")
        if flag == "--max-failures" and value != "--max-failures=0":
            raise ValueError(
                "Greenland jobs permit only --max-failures=0"
            )
    if matrix_args.count("--max-failures=0") != 1:
        raise ValueError(
            "Greenland jobs require exactly one --max-failures=0"
        )
    if job_kind == CHECKPOINT_REPLAY_JOB:
        if "--no-train" not in matrix_args or "--export-only" not in matrix_args:
            raise ValueError(
                "checkpoint replay requires --no-train and --export-only"
            )
    else:
        forbidden = {
            "--export-only",
            "--force-train",
            "--no-train",
            "--save-arrays",
        }
        if any(value.split("=", 1)[0] in forbidden for value in matrix_args):
            raise ValueError(
                "full matrix replication must train and evaluate without "
                "array exports or training overrides"
            )
        worker_args = [
            value
            for value in matrix_args
            if value.split("=", 1)[0] == "--num-workers"
        ]
        if worker_args != ["--num-workers=4"]:
            raise ValueError(
                f"{job_kind} requires exactly one --num-workers=4"
            )
    if job_kind == NUMERICAL_RECOVERY_JOB:
        recovery = spec.get("recovery")
        if not isinstance(recovery, dict):
            raise ValueError("Numerical recovery spec requires recovery data")
        profile = recovery.get("profile")
        expected_overrides = NUMERICAL_RECOVERY_PROFILES.get(profile)
        if expected_overrides is None:
            raise ValueError("Unsupported numerical recovery profile")
        expected_recovery = {
            "protocol": NUMERICAL_RECOVERY_PROTOCOL,
            "protocol_sha256": NUMERICAL_RECOVERY_PROTOCOL_SHA256,
            "cohort_id": NUMERICAL_RECOVERY_COHORT_ID,
            "cell_ids": list(NUMERICAL_RECOVERY_CELL_IDS),
            "cell_ids_sha256": NUMERICAL_RECOVERY_CELL_IDS_SHA256,
            "profile": profile,
            "overrides": expected_overrides,
            "parent_run_id": recovery.get("parent_run_id"),
        }
        if recovery != expected_recovery:
            raise ValueError(
                "Numerical recovery spec does not match the frozen protocol"
            )
        if not re.fullmatch(
            r"[a-z0-9][a-z0-9-]{5,62}",
            str(recovery.get("parent_run_id", "")),
        ):
            raise ValueError("Numerical recovery parent run ID is invalid")
    return spec


def validate_replay_manifest(path):
    rows = []
    with open(path, "r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid replay manifest JSON on line {line_number}"
                ) from error
            rows.append(row)
    if len(rows) != REPLAY_CELLS:
        raise ValueError(
            f"Replay manifest must contain {REPLAY_CELLS} cells, "
            f"found {len(rows)}"
        )
    cell_ids = [row.get("id") for row in rows]
    if len(set(cell_ids)) != REPLAY_CELLS or any(not value for value in cell_ids):
        raise ValueError("Replay manifest cell IDs must be present and unique")
    horizons = {
        (row.get("task_family"), row.get("pred_len")) for row in rows
    }
    expected_horizons = set(REPLAY_HORIZONS.items())
    if horizons != expected_horizons:
        raise ValueError(
            "Replay manifest must contain only long_term/96, pems/24, "
            "and epf/24 cells"
        )
    digest = sha256_file(path)
    if digest != REPLAY_MANIFEST_SHA256:
        raise ValueError(
            "Replay manifest SHA-256 mismatch: "
            f"expected {REPLAY_MANIFEST_SHA256}, found {digest}"
        )
    return {
        "path": str(path),
        "cells": len(rows),
        "sha256": digest,
        "family_counts": {
            family: sum(
                row.get("task_family") == family for row in rows
            )
            for family in REPLAY_HORIZONS
        },
    }


def validate_full_matrix_manifest(path):
    rows = []
    with open(path, "r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid full matrix JSON on line {line_number}"
                ) from error
    cell_ids = [row.get("id") for row in rows]
    if (
        len(rows) != FULL_MATRIX_CELLS
        or len(set(cell_ids)) != FULL_MATRIX_CELLS
        or any(not cell_id for cell_id in cell_ids)
    ):
        raise ValueError(
            f"Full matrix manifest must contain {FULL_MATRIX_CELLS} "
            "unique cells"
        )
    family_counts = {
        family: sum(row.get("task_family") == family for row in rows)
        for family in ("long_term", "pems", "epf")
    }
    if family_counts != {"long_term": 364, "pems": 156, "epf": 65}:
        raise ValueError("Full matrix task-family scope drifted")
    digest = sha256_file(path)
    if digest != FULL_MATRIX_MANIFEST_SHA256:
        raise ValueError(
            "Full matrix manifest SHA-256 mismatch: "
            f"expected {FULL_MATRIX_MANIFEST_SHA256}, found {digest}"
        )
    return {
        "path": str(path),
        "cells": len(rows),
        "sha256": digest,
        "family_counts": family_counts,
    }


def validate_numerical_recovery_protocol(path):
    digest = sha256_file(path)
    if digest != NUMERICAL_RECOVERY_PROTOCOL_SHA256:
        raise ValueError(
            "Numerical recovery protocol SHA-256 mismatch: "
            f"expected {NUMERICAL_RECOVERY_PROTOCOL_SHA256}, found {digest}"
        )
    with Path(path).open("r", encoding="utf-8") as source:
        protocol = json.load(source)
    if (
        protocol.get("schema_version") != 1
        or protocol.get("cohort_id") != NUMERICAL_RECOVERY_COHORT_ID
        or protocol.get("method_revision") != METHOD_REVISION
        or protocol.get("base_manifest") != FULL_MATRIX_MANIFEST
        or protocol.get("base_manifest_sha256")
        != FULL_MATRIX_MANIFEST_SHA256
        or protocol.get("seed") != 2021
        or protocol.get("cell_ids")
        != list(NUMERICAL_RECOVERY_CELL_IDS)
        or protocol.get("cell_ids_sha256")
        != NUMERICAL_RECOVERY_CELL_IDS_SHA256
    ):
        raise ValueError("Numerical recovery protocol identity drifted")
    profiles = protocol.get("profiles", {})
    if set(profiles) != set(NUMERICAL_RECOVERY_PROFILES):
        raise ValueError("Numerical recovery profiles drifted")
    for profile, expected_overrides in NUMERICAL_RECOVERY_PROFILES.items():
        if profiles[profile].get("overrides") != expected_overrides:
            raise ValueError(
                f"Numerical recovery overrides drifted for {profile}"
            )
    execution = protocol.get("execution", {})
    if (
        execution.get("run_every_profile_for_all_cells_on_both_hardware")
        is not True
        or execution.get("a10g_processes_per_host") != 4
        or execution.get("a100_processes_per_host") != 8
        or execution.get("max_failures") != 0
        or execution.get("num_workers") != 4
        or execution.get("fresh_output_and_checkpoint_roots") is not True
    ):
        raise ValueError("Numerical recovery execution contract drifted")
    selection = protocol.get("selection", {})
    if (
        selection.get("metric_quality_used_for_selection") is not False
        or not selection.get("originally_affected_rule")
        or not selection.get("affected_exact_rule")
        or not selection.get("affected_fallback_rule")
        or not selection.get("failure_rule")
    ):
        raise ValueError("Numerical recovery selection contract drifted")
    return {
        "path": str(path),
        "sha256": digest,
        "cohort_id": protocol["cohort_id"],
        "cell_ids": protocol["cell_ids"],
        "cell_ids_sha256": protocol["cell_ids_sha256"],
        "profiles": {
            name: record["overrides"]
            for name, record in profiles.items()
        },
    }


def _resolve_relative(root, value):
    root = Path(root).resolve()
    path = (root / _relative_path(value, "path")).resolve()
    if path != root and root not in path.parents:
        raise ValueError(f"Path escapes work root: {value}")
    return path


def _revision_matches(actual, expected):
    return bool(
        actual
        and expected
        and (
            str(actual).startswith(str(expected))
            or str(expected).startswith(str(actual))
        )
    )


def validate_replay_checkpoints(
    project_root,
    manifest,
    checkpoint_catalog,
    release_checkpoint_root,
    checkpoint_storage_root=None,
    expected_method_revision=None,
):
    project_root = Path(project_root).resolve()
    checkpoint_storage_root = (
        project_root
        if checkpoint_storage_root is None
        else Path(checkpoint_storage_root).resolve()
    )
    manifest_path = _resolve_relative(project_root, manifest)
    catalog_path = _resolve_relative(project_root, checkpoint_catalog)
    checkpoint_root = _resolve_relative(
        checkpoint_storage_root, release_checkpoint_root
    )
    manifest_evidence = validate_replay_manifest(manifest_path)
    with manifest_path.open("r", encoding="utf-8") as source:
        rows = [json.loads(line) for line in source if line.strip()]
    try:
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"Checkpoint catalog is missing or invalid: {catalog_path}"
        ) from error
    checkpoints = catalog.get("checkpoints")
    if not isinstance(checkpoints, dict):
        raise ValueError("Checkpoint catalog has no checkpoints object")
    expected_ids = {row["id"] for row in rows}
    if catalog.get("schema_version") != 1:
        raise ValueError("Unsupported replay checkpoint catalog schema")
    if set(checkpoints) != expected_ids:
        raise ValueError(
            "Replay checkpoint catalog does not exactly match the manifest"
        )
    if (
        catalog.get("selected_count") != REPLAY_CELLS
        or catalog.get("checkpoint_count") != REPLAY_CELLS
        or catalog.get("usable_count") != REPLAY_CELLS
        or catalog.get("missing_count") != 0
        or catalog.get("missing_selected_count") != 0
    ):
        raise ValueError(
            "Replay checkpoint catalog counts are incomplete or inconsistent"
        )
    if catalog.get("hashes_included") is not True:
        raise ValueError("Replay checkpoint catalog must include hashes")
    if expected_method_revision and not _revision_matches(
        catalog.get("source_revision"),
        expected_method_revision,
    ):
        raise ValueError("Replay checkpoint catalog method revision drifted")

    verified_paths = set()
    total_size = 0
    for row in rows:
        cell_id = row["id"]
        record = checkpoints.get(cell_id)
        if not isinstance(record, dict) or record.get("usable") is not True:
            raise ValueError(
                f"Replay checkpoint is not usable for {cell_id}"
            )
        if expected_method_revision and not _revision_matches(
            record.get("source_revision"),
            expected_method_revision,
        ):
            raise ValueError(
                f"Replay checkpoint method revision drifted for {cell_id}"
            )
        relative_path = _relative_path(
            record.get("relative_path", ""),
            f"checkpoint {cell_id}",
        )
        if relative_path in verified_paths:
            raise ValueError(
                f"Replay checkpoint path is reused: {relative_path}"
            )
        checkpoint_path = _resolve_relative(
            checkpoint_root, relative_path
        )
        if not checkpoint_path.is_file():
            raise ValueError(
                f"Replay checkpoint is missing for {cell_id}: "
                f"{checkpoint_path}"
            )
        if checkpoint_path.is_symlink():
            raise ValueError(
                f"Replay checkpoint cannot be a symlink: {checkpoint_path}"
            )
        expected_size = record.get("file_size")
        actual_size = checkpoint_path.stat().st_size
        if expected_size != actual_size:
            raise ValueError(
                f"Replay checkpoint size mismatch for {cell_id}: "
                f"expected {expected_size}, found {actual_size}"
            )
        expected_digest = str(record.get("sha256", ""))
        if not SHA256_PATTERN.fullmatch(expected_digest):
            raise ValueError(
                f"Replay checkpoint has no valid SHA-256 for {cell_id}"
            )
        actual_digest = sha256_file(checkpoint_path)
        if actual_digest != expected_digest:
            raise ValueError(
                f"Replay checkpoint hash mismatch for {cell_id}: "
                f"expected {expected_digest}, found {actual_digest}"
            )
        verified_paths.add(relative_path)
        total_size += actual_size

    return {
        "manifest_sha256": manifest_evidence["sha256"],
        "catalog_sha256": sha256_file(catalog_path),
        "catalog_source_revision": catalog.get("source_revision"),
        "method_revision": expected_method_revision,
        "checkpoint_count": len(rows),
        "unique_checkpoint_paths": len(verified_paths),
        "total_size_bytes": total_size,
        "hashes_verified": True,
    }


def build_matrix_command(spec, work_root, python_executable):
    validate_job_spec(spec)
    project_root = _resolve_relative(work_root, "project")
    matrix = spec["matrix"]

    def project_path(field):
        return str(_resolve_relative(project_root, matrix[field]))

    def work_path(field):
        return str(_resolve_relative(work_root, matrix[field]))

    matrix_source_revision = (
        spec["method_revision"]
        if spec["job_kind"]
        in {FULL_MATRIX_REPLICATION_JOB, NUMERICAL_RECOVERY_JOB}
        else spec["source_revision"]
    )
    checkpoint_root = (
        work_path("checkpoint_root")
        if spec["job_kind"]
        in {FULL_MATRIX_REPLICATION_JOB, NUMERICAL_RECOVERY_JOB}
        else project_path("checkpoint_root")
    )
    command = [
        str(python_executable),
        "scripts/run_benchmark_matrix.py",
        "--manifest",
        project_path("manifest"),
        "--output-root",
        work_path("output_root"),
        "--checkpoint-root",
        checkpoint_root,
        "--summary",
        work_path("summary"),
        "--log-root",
        work_path("log_root"),
        "--seed",
        str(matrix["seed"]),
        "--source-revision",
        str(matrix_source_revision),
        "--launcher-revision",
        str(spec["launcher_revision"]),
        "--instance-type",
        SAGEMAKER_INSTANCE_TYPE,
        "--reserved-gpus-per-host",
        str(GPUS_PER_HOST),
        "--processes-per-host",
        str(PROCESSES_PER_HOST),
        "--gpu",
        "0",
    ]
    if spec["job_kind"] == CHECKPOINT_REPLAY_JOB:
        command.extend(
            [
                "--checkpoint-catalog",
                project_path("checkpoint_catalog"),
                "--release-checkpoint-root",
                project_path("release_checkpoint_root"),
            ]
        )
    elif spec["job_kind"] == NUMERICAL_RECOVERY_JOB:
        for cell_id in spec["recovery"]["cell_ids"]:
            command.extend(["--cell-id", cell_id])
        for key, value in sorted(spec["recovery"]["overrides"].items()):
            encoded = json.dumps(value, separators=(",", ":"))
            command.extend(["--override", f"{key}={encoded}"])
    command.extend(matrix["extra_args"])
    return command


def safe_extract_tar(archive_path, destination):
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:*") as archive:
        members = archive.getmembers()
        for member in members:
            member_path = PurePosixPath(member.name)
            if (
                member_path.is_absolute()
                or ".." in member_path.parts
                or member.issym()
                or member.islnk()
                or member.isdev()
            ):
                raise ValueError(f"Unsafe archive member: {member.name}")
            resolved = (destination / member.name).resolve()
            if resolved != destination and destination not in resolved.parents:
                raise ValueError(f"Archive member escapes destination: {member.name}")
        archive.extractall(destination, members=members, filter="data")


def parse_gpu_rows(payload):
    rows = []
    for row in csv.reader(io.StringIO(payload.strip())):
        if not row:
            continue
        if len(row) != 5:
            raise ValueError(f"Expected five GPU columns, found {len(row)}")
        rows.append(
            {
                "index": int(row[0].strip()),
                "uuid": row[1].strip(),
                "name": row[2].strip(),
                "utilization_gpu_percent": int(row[3].strip()),
                "memory_used_mib": int(row[4].strip()),
                "processes": [],
            }
        )
    return rows


def parse_compute_rows(payload):
    rows = []
    for row in csv.reader(io.StringIO(payload.strip())):
        if not row:
            continue
        if len(row) != 4:
            raise ValueError(f"Expected four process columns, found {len(row)}")
        rows.append(
            {
                "gpu_uuid": row[0].strip(),
                "pid": int(row[1].strip()),
                "process_name": row[2].strip(),
                "memory_used_mib": int(row[3].strip()),
            }
        )
    return rows


def summarize_gpu_samples(samples):
    expected = set(range(GPUS_PER_HOST))
    max_utilization = {index: 0 for index in expected}
    bound_sample = None
    sample_count = 0
    for sample in samples:
        sample_count += 1
        gpus = {gpu["index"]: gpu for gpu in sample.get("gpus", [])}
        for index in expected & set(gpus):
            max_utilization[index] = max(
                max_utilization[index],
                int(gpus[index].get("utilization_gpu_percent", 0)),
            )
        if set(gpus) != expected:
            continue
        pids = []
        bindings_ok = True
        for index in sorted(expected):
            processes = gpus[index].get("processes", [])
            matching = [
                process
                for process in processes
                if process.get("cuda_visible_devices") == str(index)
            ]
            if not matching:
                bindings_ok = False
                break
            pids.append(matching[0]["pid"])
        if bindings_ok and len(set(pids)) == GPUS_PER_HOST:
            bound_sample = {
                "recorded_unix": sample.get("recorded_unix"),
                "pids": pids,
                "gpu_ids": sorted(expected),
            }
    utilization_verified = all(value > 0 for value in max_utilization.values())
    return {
        "sample_count": sample_count,
        "expected_gpu_ids": sorted(expected),
        "eight_distinct_bindings_observed": bound_sample is not None,
        "binding_evidence": bound_sample,
        "max_utilization_gpu_percent": max_utilization,
        "all_eight_gpus_utilized": utilization_verified,
        "topology_gate_passed": (
            bound_sample is not None and utilization_verified
        ),
    }


def validate_gpu_topology_evidence(topology):
    expected_ids = list(range(GPUS_PER_HOST))
    if not isinstance(topology, dict):
        raise ValueError("Greenland topology evidence must be an object")
    binding = topology.get("binding_evidence")
    utilization = topology.get("max_utilization_gpu_percent")
    try:
        normalized_utilization = {
            int(key): int(value) for key, value in utilization.items()
        }
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError("Greenland utilization evidence is invalid") from error
    pids = binding.get("pids", []) if isinstance(binding, dict) else []
    if (
        topology.get("topology_gate_passed") is not True
        or topology.get("eight_distinct_bindings_observed") is not True
        or topology.get("all_eight_gpus_utilized") is not True
        or topology.get("expected_gpu_ids") != expected_ids
        or not isinstance(binding, dict)
        or binding.get("gpu_ids") != expected_ids
        or len(pids) != GPUS_PER_HOST
        or not all(
            isinstance(pid, int)
            and not isinstance(pid, bool)
            and pid > 0
            for pid in pids
        )
        or len(set(pids)) != GPUS_PER_HOST
        or set(normalized_utilization) != set(expected_ids)
        or any(normalized_utilization[index] <= 0 for index in expected_ids)
    ):
        raise ValueError(
            "Greenland topology did not prove eight active distinct A100 workers"
        )
    return topology
