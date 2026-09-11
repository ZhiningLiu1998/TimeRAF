"""Build and upload content-addressed inputs for a TimeRAF Greenland job."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

from greenland_common import (
    APPROVED_IMAGE_RECORD,
    CHECKPOINT_REPLAY_JOB,
    FULL_MATRIX_CELLS,
    FULL_MATRIX_MANIFEST,
    FULL_MATRIX_REPLICATION_JOB,
    METHOD_REVISION,
    NUMERICAL_RECOVERY_CELLS,
    NUMERICAL_RECOVERY_CELL_IDS,
    NUMERICAL_RECOVERY_CELL_IDS_SHA256,
    NUMERICAL_RECOVERY_COHORT_ID,
    NUMERICAL_RECOVERY_JOB,
    NUMERICAL_RECOVERY_PROFILES,
    NUMERICAL_RECOVERY_PROTOCOL,
    NUMERICAL_RECOVERY_PROTOCOL_SHA256,
    REPLAY_CELLS,
    REPLAY_MANIFEST,
    S3_ROOT,
    SERVICE_ROLE_ARN,
    canonical_json_bytes,
    parse_s3_uri,
    sha256_bytes,
    sha256_file,
    validate_approved_image_uri,
    validate_full_matrix_manifest,
    validate_job_spec,
    validate_numerical_recovery_protocol,
    validate_replay_checkpoints,
)


def _git_revision(project_root, revision):
    return subprocess.run(
        ["git", "rev-parse", "--verify", f"{revision}^{{commit}}"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _assert_revision_file_matches(project_root, revision, path):
    project_root = Path(project_root).resolve()
    path = Path(path).resolve()
    try:
        relative = path.relative_to(project_root).as_posix()
    except ValueError as error:
        raise ValueError(
            f"Revision input is outside project: {path}"
        ) from error
    committed = subprocess.run(
        ["git", "show", f"{revision}:{relative}"],
        cwd=project_root,
        check=True,
        capture_output=True,
    ).stdout
    working = path.read_bytes()
    if working != committed:
        raise ValueError(
            f"{relative} does not match source revision {revision}"
        )


def _build_source_archive(project_root, revision, output_path):
    temporary_tar = output_path.with_suffix("")
    subprocess.run(
        [
            "git",
            "archive",
            "--format=tar",
            "--output",
            str(temporary_tar),
            revision,
        ],
        cwd=project_root,
        check=True,
    )
    with temporary_tar.open("rb") as source, output_path.open("wb") as raw:
        with gzip.GzipFile(
            filename="", fileobj=raw, mode="wb", mtime=0
        ) as compressed:
            shutil.copyfileobj(source, compressed, length=8 * 1024 * 1024)
    temporary_tar.unlink()


def _validate_numerical_integrity_source(project_root, source_revision):
    project_root = Path(project_root).resolve()
    entrypoint = project_root / "scripts" / "run_benchmark_cell.py"
    program = f"""
from pathlib import Path
from types import SimpleNamespace

from scripts.run_benchmark_cell import NumericalIntegrityError, _train_or_load

root = Path({str(project_root)!r})
checkpoint_root = root / "__timeraf_missing_checkpoint_preflight__"
setting = "cell"
checkpoint = checkpoint_root / setting / "checkpoint.pth"
runtime_args = SimpleNamespace(checkpoints=str(checkpoint_root))
cli = SimpleNamespace(
    checkpoint=None,
    force_train=False,
    no_train=False,
    smoke=False,
)

class Experiment:
    @staticmethod
    def train(_setting):
        raise FileNotFoundError(
            2,
            "No such file or directory",
            str(checkpoint),
        )

try:
    _train_or_load(Experiment(), runtime_args, setting, cli)
except NumericalIntegrityError:
    pass
else:
    raise SystemExit(
        "missing finite checkpoint was not converted to "
        "NumericalIntegrityError"
    )
"""
    environment = dict(os.environ)
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        str(project_root)
        if not existing_pythonpath
        else f"{project_root}{os.pathsep}{existing_pythonpath}"
    )
    subprocess.run(
        [sys.executable, "-c", program],
        cwd=project_root,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    return {
        "contract": "explicit-numerical-integrity-error-v1",
        "source_revision": source_revision,
        "entrypoint": str(entrypoint.relative_to(project_root)),
        "entrypoint_sha256": sha256_file(entrypoint),
        "passed": True,
    }


def _normalized_tar_info(archive, path, arcname):
    info = archive.gettarinfo(str(path), arcname=arcname)
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = 0
    return info


def _build_directory_archive(project_root, source_path, output_path):
    project_root = Path(project_root).resolve()
    source_path = Path(source_path).resolve()
    if source_path != project_root and project_root not in source_path.parents:
        raise ValueError(f"Input path is outside project root: {source_path}")
    if not source_path.is_dir():
        raise FileNotFoundError(source_path)
    with output_path.open("wb") as raw:
        with gzip.GzipFile(
            filename="", fileobj=raw, mode="wb", mtime=0
        ) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as archive:
                paths = [source_path, *sorted(source_path.rglob("*"))]
                for path in paths:
                    if path.is_symlink():
                        raise ValueError(f"Input archives cannot contain links: {path}")
                    arcname = path.relative_to(project_root).as_posix()
                    info = _normalized_tar_info(archive, path, arcname)
                    if info.isfile():
                        with path.open("rb") as source:
                            archive.addfile(info, source)
                    else:
                        archive.addfile(info)


def _build_checkpoint_archive(
    project_root,
    manifest_path,
    catalog_path,
    checkpoint_root,
    output_path,
    artifact_root=None,
    method_revision=METHOD_REVISION,
):
    project_root = Path(project_root).resolve()
    artifact_root = (
        project_root
        if artifact_root is None
        else Path(artifact_root).resolve()
    )
    manifest_path = Path(manifest_path).resolve()
    catalog_path = Path(catalog_path).resolve()
    checkpoint_root = Path(checkpoint_root).resolve()
    try:
        manifest_relative = manifest_path.relative_to(project_root).as_posix()
        catalog_relative = catalog_path.relative_to(project_root).as_posix()
        checkpoint_relative = checkpoint_root.relative_to(
            artifact_root
        ).as_posix()
    except ValueError as error:
        raise ValueError(
            "Replay source inputs must live below project-root and "
            "checkpoints below artifact-root"
        ) from error
    evidence = validate_replay_checkpoints(
        project_root,
        manifest_relative,
        catalog_relative,
        checkpoint_relative,
        checkpoint_storage_root=artifact_root,
        expected_method_revision=method_revision,
    )
    with manifest_path.open("r", encoding="utf-8") as source:
        rows = [json.loads(line) for line in source if line.strip()]
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    checkpoint_paths = sorted(
        {
            checkpoint_root
            / catalog["checkpoints"][row["id"]]["relative_path"]
            for row in rows
        }
    )
    with output_path.open("wb") as raw:
        with gzip.GzipFile(
            filename="", fileobj=raw, mode="wb", mtime=0
        ) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as archive:
                for path in checkpoint_paths:
                    arcname = path.relative_to(artifact_root).as_posix()
                    info = _normalized_tar_info(archive, path, arcname)
                    with path.open("rb") as source:
                        archive.addfile(info, source)
    return evidence


def _assumed_s3_client():
    import boto3

    sts = boto3.client("sts", region_name="us-east-1")
    response = sts.assume_role(
        RoleArn=SERVICE_ROLE_ARN,
        RoleSessionName="timeraf-greenland-inputs",
        DurationSeconds=3600,
    )
    credentials = response["Credentials"]
    return boto3.client(
        "s3",
        region_name="us-east-1",
        aws_access_key_id=credentials["AccessKeyId"],
        aws_secret_access_key=credentials["SecretAccessKey"],
        aws_session_token=credentials["SessionToken"],
    )


def _upload_content_addressed(s3, path, name):
    from botocore.exceptions import ClientError

    digest = sha256_file(path)
    uri = f"{S3_ROOT}inputs/{name}/{digest}.tar.gz"
    bucket, key = parse_s3_uri(uri)
    try:
        head = s3.head_object(Bucket=bucket, Key=key)
    except ClientError as error:
        if error.response.get("Error", {}).get("Code") not in {
            "404",
            "NoSuchKey",
            "NotFound",
        }:
            raise
    else:
        if (
            head.get("Metadata", {}).get("sha256") != digest
            or head["ContentLength"] != path.stat().st_size
        ):
            raise ValueError(f"Existing content-addressed object drifted: {uri}")
        return uri, digest
    s3.upload_file(
        str(path),
        bucket,
        key,
        ExtraArgs={
            "ContentType": "application/gzip",
            "Metadata": {"sha256": digest, "input-name": name},
        },
    )
    return uri, digest


def _put_spec(s3, spec):
    payload = canonical_json_bytes(spec)
    digest = sha256_bytes(payload)
    uri = f"{S3_ROOT}specs/{digest}.json"
    bucket, key = parse_s3_uri(uri)
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=payload,
        ContentType="application/json",
        Metadata={"sha256": digest},
    )
    return uri, digest, payload


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Prepare immutable TimeRAF Greenland matrix inputs"
    )
    parser.add_argument("--project-root", required=True)
    parser.add_argument(
        "--artifact-root",
        help=(
            "EFS root containing dataset, checkpoints, and staging; "
            "defaults to project-root"
        ),
    )
    parser.add_argument("--staging-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--job-kind",
        choices=(
            CHECKPOINT_REPLAY_JOB,
            FULL_MATRIX_REPLICATION_JOB,
            NUMERICAL_RECOVERY_JOB,
        ),
        default=CHECKPOINT_REPLAY_JOB,
    )
    parser.add_argument(
        "--recovery-profile",
        choices=tuple(NUMERICAL_RECOVERY_PROFILES),
    )
    parser.add_argument("--parent-run-id")
    parser.add_argument("--source-revision", default="HEAD")
    parser.add_argument("--launcher-revision", default="HEAD")
    parser.add_argument("--method-revision", default=METHOD_REVISION)
    parser.add_argument("--image-uri", required=True)
    parser.add_argument("--dataset-dir", default="dataset")
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--manifest")
    parser.add_argument("--checkpoint-catalog")
    parser.add_argument("--checkpoint-root")
    parser.add_argument("--release-checkpoint-root")
    parser.add_argument("--output-root")
    parser.add_argument("--summary")
    parser.add_argument("--log-root")
    parser.add_argument("--queued-cells", type=int)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument(
        "--extra-arg",
        action="append",
        default=None,
    )
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args(argv)
    if args.job_kind == CHECKPOINT_REPLAY_JOB:
        args.manifest = args.manifest or REPLAY_MANIFEST
        args.output_root = args.output_root or "outputs/checkpoint_replay"
        args.summary = (
            args.summary
            or "outputs/checkpoint_replay/matrix_summary.json"
        )
        args.log_root = args.log_root or "outputs/checkpoint_replay/logs"
        if args.queued_cells is None:
            args.queued_cells = REPLAY_CELLS
        args.extra_arg = args.extra_arg or [
            "--no-train",
            "--export-only",
            "--max-failures=0",
        ]
        missing = [
            name
            for name in (
                "checkpoint_dir",
                "checkpoint_catalog",
                "checkpoint_root",
                "release_checkpoint_root",
            )
            if not getattr(args, name)
        ]
        if missing:
            parser.error(
                "checkpoint replay requires " + ", ".join(missing)
            )
    elif args.job_kind == FULL_MATRIX_REPLICATION_JOB:
        args.manifest = args.manifest or FULL_MATRIX_MANIFEST
        args.output_root = (
            args.output_root or "outputs/full_matrix_replication"
        )
        args.checkpoint_root = (
            args.checkpoint_root
            or "outputs/full_matrix_replication/checkpoints"
        )
        args.summary = (
            args.summary
            or "outputs/full_matrix_replication/matrix_summary.json"
        )
        args.log_root = (
            args.log_root or "outputs/full_matrix_replication/logs"
        )
        if args.queued_cells is None:
            args.queued_cells = FULL_MATRIX_CELLS
        args.extra_arg = args.extra_arg or [
            "--max-failures=0",
            "--num-workers=4",
        ]
        if any(
            getattr(args, name)
            for name in (
                "checkpoint_dir",
                "checkpoint_catalog",
                "release_checkpoint_root",
            )
        ):
            parser.error(
                "full matrix replication does not accept replay checkpoints"
            )
    else:
        if not args.recovery_profile or not args.parent_run_id:
            parser.error(
                "numerical recovery requires --recovery-profile and "
                "--parent-run-id"
            )
        args.manifest = args.manifest or FULL_MATRIX_MANIFEST
        profile_root = (
            f"outputs/numerical_recovery/{args.recovery_profile}"
        )
        args.output_root = args.output_root or profile_root
        args.checkpoint_root = (
            args.checkpoint_root or f"{profile_root}/checkpoints"
        )
        args.summary = args.summary or f"{profile_root}/matrix_summary.json"
        args.log_root = args.log_root or f"{profile_root}/logs"
        if args.queued_cells is None:
            args.queued_cells = NUMERICAL_RECOVERY_CELLS
        args.extra_arg = args.extra_arg or [
            "--max-failures=0",
            "--num-workers=4",
        ]
        if any(
            getattr(args, name)
            for name in (
                "checkpoint_dir",
                "checkpoint_catalog",
                "release_checkpoint_root",
            )
        ):
            parser.error(
                "numerical recovery does not accept replay checkpoints"
            )

    project_root = Path(args.project_root).resolve()
    artifact_root = Path(args.artifact_root or project_root).resolve()
    staging_root = Path(args.staging_root).resolve()
    if (
        project_root != artifact_root
        and artifact_root not in project_root.parents
    ):
        raise ValueError("project-root must live at or below artifact-root")
    if artifact_root not in staging_root.parents:
        raise ValueError("staging-root must live below artifact-root")
    staging_root.mkdir(parents=True, exist_ok=True)
    source_revision = _git_revision(project_root, args.source_revision)
    launcher_revision = _git_revision(project_root, args.launcher_revision)
    if args.method_revision != METHOD_REVISION:
        raise ValueError(
            f"method-revision must remain frozen at {METHOD_REVISION}"
        )
    approval_path = (project_root / APPROVED_IMAGE_RECORD).resolve()
    _assert_revision_file_matches(
        project_root,
        source_revision,
        approval_path,
    )
    validate_approved_image_uri(
        args.image_uri,
        approval_path,
        required_job_kind=args.job_kind,
    )
    manifest_path = (project_root / args.manifest).resolve()
    _assert_revision_file_matches(
        project_root, source_revision, manifest_path
    )
    if args.job_kind in {
        FULL_MATRIX_REPLICATION_JOB,
        NUMERICAL_RECOVERY_JOB,
    }:
        manifest_evidence = validate_full_matrix_manifest(manifest_path)
    else:
        manifest_evidence = None
    recovery_evidence = None
    numerical_integrity_source_evidence = None
    if args.job_kind == NUMERICAL_RECOVERY_JOB:
        protocol_path = (project_root / NUMERICAL_RECOVERY_PROTOCOL).resolve()
        _assert_revision_file_matches(
            project_root,
            source_revision,
            protocol_path,
        )
        recovery_evidence = validate_numerical_recovery_protocol(
            protocol_path
        )
        numerical_integrity_source_evidence = (
            _validate_numerical_integrity_source(
                project_root,
                source_revision,
            )
        )

    archives = {
        "source": staging_root / f"source-{source_revision}.tar.gz",
        "dataset": staging_root / "dataset.tar.gz",
    }
    _build_source_archive(project_root, source_revision, archives["source"])
    _build_directory_archive(
        artifact_root,
        artifact_root / args.dataset_dir,
        archives["dataset"],
    )
    checkpoint_evidence = None
    if args.job_kind == CHECKPOINT_REPLAY_JOB:
        catalog_path = (project_root / args.checkpoint_catalog).resolve()
        checkpoint_root = (artifact_root / args.checkpoint_dir).resolve()
        release_checkpoint_root = (
            artifact_root / args.release_checkpoint_root
        ).resolve()
        if checkpoint_root != release_checkpoint_root:
            raise ValueError(
                "checkpoint-dir and release-checkpoint-root must resolve "
                "to the same archived directory"
            )
        _assert_revision_file_matches(
            project_root, source_revision, catalog_path
        )
        archives["checkpoints"] = (
            staging_root / "checkpoints.tar.gz"
        )
        checkpoint_evidence = _build_checkpoint_archive(
            project_root,
            manifest_path,
            catalog_path,
            checkpoint_root,
            archives["checkpoints"],
            artifact_root=artifact_root,
            method_revision=args.method_revision,
        )

    if not args.upload:
        print(
            json.dumps(
                {
                    "archives": {
                        name: {
                            "path": str(path),
                            "sha256": sha256_file(path),
                            "size_bytes": path.stat().st_size,
                        }
                        for name, path in archives.items()
                    },
                    "checkpoint_evidence": checkpoint_evidence,
                    "manifest_evidence": manifest_evidence,
                    "recovery_evidence": recovery_evidence,
                    "numerical_integrity_source_evidence": (
                        numerical_integrity_source_evidence
                    ),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    s3 = _assumed_s3_client()
    inputs = []
    for name, path in archives.items():
        uri, digest = _upload_content_addressed(s3, path, name)
        inputs.append(
            {
                "name": name,
                "s3_uri": uri,
                "sha256": digest,
                "extract_to": "project",
            }
        )
    spec = {
        "schema_version": (
            2 if args.job_kind == NUMERICAL_RECOVERY_JOB else 1
        ),
        "run_id": args.run_id,
        "job_kind": args.job_kind,
        "method_revision": args.method_revision,
        "image_uri": args.image_uri,
        "source_revision": source_revision,
        "launcher_revision": launcher_revision,
        "inputs": inputs,
        "output_s3_uri": f"{S3_ROOT}runs/{args.run_id}/",
        "matrix": {
            "manifest": args.manifest,
            "output_root": args.output_root,
            "checkpoint_root": args.checkpoint_root,
            "summary": args.summary,
            "log_root": args.log_root,
            "queued_cells": args.queued_cells,
            "seed": args.seed,
            "extra_args": args.extra_arg,
        },
    }
    if args.job_kind == CHECKPOINT_REPLAY_JOB:
        spec["matrix"].update(
            {
                "checkpoint_catalog": args.checkpoint_catalog,
                "release_checkpoint_root": args.release_checkpoint_root,
            }
        )
    elif args.job_kind == NUMERICAL_RECOVERY_JOB:
        spec["recovery_revision"] = source_revision
        spec["recovery"] = {
            "protocol": NUMERICAL_RECOVERY_PROTOCOL,
            "protocol_sha256": NUMERICAL_RECOVERY_PROTOCOL_SHA256,
            "cohort_id": NUMERICAL_RECOVERY_COHORT_ID,
            "cell_ids": list(NUMERICAL_RECOVERY_CELL_IDS),
            "cell_ids_sha256": NUMERICAL_RECOVERY_CELL_IDS_SHA256,
            "profile": args.recovery_profile,
            "overrides": NUMERICAL_RECOVERY_PROFILES[
                args.recovery_profile
            ],
            "parent_run_id": args.parent_run_id,
        }
    validate_job_spec(spec)
    spec_uri, spec_digest, payload = _put_spec(s3, spec)
    local_spec = staging_root / f"job-spec-{spec_digest}.json"
    local_spec.write_bytes(payload)
    print(
        json.dumps(
            {
                "job_spec": str(local_spec),
                "job_spec_s3_uri": spec_uri,
                "job_spec_sha256": spec_digest,
                "inputs": inputs,
                "checkpoint_evidence": checkpoint_evidence,
                "manifest_evidence": manifest_evidence,
                "recovery_evidence": recovery_evidence,
                "numerical_integrity_source_evidence": (
                    numerical_integrity_source_evidence
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
