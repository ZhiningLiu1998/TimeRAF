"""Import and verify one Greenland replay run below the Studio EFS root."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path, PurePosixPath

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from scripts.greenland_common import (
    REPLAY_CELLS,
    S3_ROOT,
    SERVICE_ROLE_ARN,
    SHA256_PATTERN,
    parse_s3_uri,
    sha256_file,
    validate_gpu_topology_evidence,
    validate_job_spec,
    validate_replay_manifest,
)
from ts_rag.matrix import load_manifest, save_json_atomic
from ts_rag.prediction_bundles import build_prediction_bundle_catalog


def _assumed_s3_client(region):
    import boto3

    sts = boto3.client("sts", region_name=region)
    identity = sts.get_caller_identity()
    if identity["Account"] != "<AWS_ACCOUNT_ID>":
        raise RuntimeError(
            "Greenland output import must start in modeldev account "
            f"<AWS_ACCOUNT_ID>, found {identity['Account']}"
        )
    response = sts.assume_role(
        RoleArn=SERVICE_ROLE_ARN,
        RoleSessionName="timeraf-greenland-output-import",
        DurationSeconds=3600,
    )
    credentials = response["Credentials"]
    return boto3.client(
        "s3",
        region_name=region,
        aws_access_key_id=credentials["AccessKeyId"],
        aws_secret_access_key=credentials["SecretAccessKey"],
        aws_session_token=credentials["SessionToken"],
    )


def _below_project(project_root, value, field):
    project_root = Path(project_root).resolve()
    path = Path(value)
    path = path.resolve() if path.is_absolute() else (project_root / path).resolve()
    if path == project_root or project_root not in path.parents:
        raise ValueError(f"{field} must live below the project root")
    return path


def _filesystem_type(path):
    return subprocess.run(
        ["findmnt", "-T", str(path), "-o", "FSTYPE", "-n"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _safe_object_relative(key, prefix):
    if not key.startswith(prefix):
        raise ValueError(f"S3 object escaped run prefix: {key}")
    relative = PurePosixPath(key[len(prefix) :])
    if (
        relative.is_absolute()
        or not relative.parts
        or ".." in relative.parts
    ):
        raise ValueError(f"Unsafe Greenland output key: {key}")
    return relative


def _list_objects(s3, bucket, prefix):
    records = {}
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for item in page.get("Contents", []):
            key = item["Key"]
            if key == prefix or key.endswith("/"):
                continue
            relative = _safe_object_relative(key, prefix).as_posix()
            if relative in records:
                raise ValueError(f"Duplicate Greenland output key: {relative}")
            head = s3.head_object(Bucket=bucket, Key=key)
            digest = head.get("Metadata", {}).get("sha256")
            if not isinstance(digest, str) or not SHA256_PATTERN.fullmatch(
                digest
            ):
                raise ValueError(f"Output object has no valid SHA-256: {key}")
            size = int(head["ContentLength"])
            if size != int(item["Size"]):
                raise ValueError(f"Output object size drifted while listing: {key}")
            records[relative] = {
                "key": key,
                "sha256": digest,
                "size_bytes": size,
                "etag": head.get("ETag"),
                "version_id": head.get("VersionId"),
            }
    if not records:
        raise ValueError("Greenland run prefix contains no output objects")
    return records


def _validate_existing_files(destination, expected):
    allowed_generated = {
        "fetch_receipt.json",
        "replay_bundle_catalog.json",
    }
    if not destination.exists():
        return
    for path in destination.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Output destination contains a symlink: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(destination).as_posix()
        if relative not in expected and relative not in allowed_generated:
            raise ValueError(f"Unexpected file in output destination: {relative}")


def _download_objects(s3, bucket, records, destination):
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    _validate_existing_files(destination, set(records))
    imported = {}
    for relative, record in sorted(records.items()):
        target = (destination / relative).resolve()
        if destination not in target.parents:
            raise ValueError(f"Output object escapes destination: {relative}")
        target.parent.mkdir(parents=True, exist_ok=True)
        mode = "downloaded"
        if target.exists():
            if (
                target.is_symlink()
                or not target.is_file()
                or target.stat().st_size != record["size_bytes"]
                or sha256_file(target) != record["sha256"]
            ):
                raise ValueError(f"Existing imported output drifted: {target}")
            mode = "reused"
        else:
            temporary = target.with_name(
                f".{target.name}.part-{os.getpid()}"
            )
            try:
                s3.download_file(bucket, record["key"], str(temporary))
                if (
                    temporary.stat().st_size != record["size_bytes"]
                    or sha256_file(temporary) != record["sha256"]
                ):
                    raise ValueError(
                        f"Downloaded output hash mismatch: {record['key']}"
                    )
                os.replace(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)
        imported[relative] = {**record, "path": str(target), "mode": mode}
    return imported


def _load_json(path, description):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{description} is missing or invalid: {path}") from error


def _output_relative(value, field):
    path = PurePosixPath(str(value))
    if path.is_absolute() or len(path.parts) < 2 or path.parts[0] != "outputs":
        raise ValueError(f"{field} must be below outputs/")
    if ".." in path.parts:
        raise ValueError(f"{field} is unsafe")
    return Path(*path.parts[1:])


def _validate_downloaded_run(destination, run_id):
    destination = Path(destination)
    spec = validate_job_spec(
        _load_json(
            destination / "evidence" / "accepted_job_spec.json",
            "Accepted Greenland job spec",
        )
    )
    if spec["run_id"] != run_id:
        raise ValueError("Accepted Greenland job spec has the wrong run ID")
    final_status = _load_json(
        destination / "final_status.json",
        "Greenland final status",
    )
    topology = _load_json(
        destination / "evidence" / "topology_summary.json",
        "Greenland topology evidence",
    )
    summary_path = destination / _output_relative(
        spec["matrix"]["summary"],
        "matrix.summary",
    )
    summary = _load_json(summary_path, "Greenland replay matrix summary")
    if (
        final_status.get("run_id") != run_id
        or final_status.get("status") != "succeeded"
        or final_status.get("exit_code") != 0
        or final_status.get("topology_gate_passed") is not True
        or final_status.get("method_revision") != spec["method_revision"]
        or final_status.get("source_revision") != spec["source_revision"]
        or final_status.get("launcher_revision") != spec["launcher_revision"]
    ):
        raise ValueError("Greenland final status did not prove a successful run")
    validate_gpu_topology_evidence(topology)
    counts = summary.get("counts", {})
    if (
        summary.get("source_revision") != spec["source_revision"]
        or summary.get("export_only") is not True
        or summary.get("all_completed") is not True
        or counts.get("expected") != REPLAY_CELLS
        or counts.get("completed") != REPLAY_CELLS
        or counts.get("failed") != 0
        or counts.get("pending") != 0
        or counts.get("running") != 0
        or counts.get("incomplete") != 0
    ):
        raise ValueError("Greenland replay summary is incomplete or inconsistent")
    return spec, final_status, topology, summary


def fetch_greenland_run(
    s3,
    run_id,
    project_root,
    destination,
    replay_manifest,
    *,
    catalog_path=None,
    receipt_path=None,
    required_filesystem=None,
):
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{5,62}", run_id):
        raise ValueError("run_id must be a lowercase K8s-safe identifier")
    project_root = Path(project_root).resolve()
    if not project_root.is_dir():
        raise ValueError("project_root must be an existing directory")
    if required_filesystem and _filesystem_type(project_root) != required_filesystem:
        raise ValueError(
            f"project_root must use {required_filesystem} durable storage"
        )
    destination = _below_project(project_root, destination, "destination")
    replay_manifest = Path(replay_manifest).resolve()
    if replay_manifest != project_root and project_root not in replay_manifest.parents:
        raise ValueError("replay_manifest must live below the project root")
    validate_replay_manifest(replay_manifest)

    prefix_uri = f"{S3_ROOT}runs/{run_id}/"
    bucket, prefix = parse_s3_uri(prefix_uri)
    records = _list_objects(s3, bucket, prefix)
    imported = _download_objects(s3, bucket, records, destination)
    spec, final_status, topology, summary = _validate_downloaded_run(
        destination,
        run_id,
    )
    matrix_output = destination / _output_relative(
        spec["matrix"]["output_root"],
        "matrix.output_root",
    )
    replay_rows = load_manifest(replay_manifest)
    catalog = build_prediction_bundle_catalog(
        replay_rows,
        matrix_output,
        project_root,
        source_revision=spec["source_revision"],
        compute_hashes=True,
    )
    expected_ids = {row["id"] for row in replay_rows}
    if (
        set(catalog["bundles"]) != expected_ids
        or catalog["expected_cells"] != REPLAY_CELLS
        or catalog["cataloged_cells"] != REPLAY_CELLS
        or catalog["usable_count"] != REPLAY_CELLS
        or catalog["missing_count"] != 0
    ):
        raise ValueError("Imported replay bundle catalog is not exactly complete")

    catalog_path = _below_project(
        project_root,
        catalog_path or destination / "replay_bundle_catalog.json",
        "catalog_path",
    )
    receipt_path = _below_project(
        project_root,
        receipt_path or destination / "fetch_receipt.json",
        "receipt_path",
    )
    save_json_atomic(catalog, catalog_path)
    receipt = {
        "schema_version": 1,
        "run_id": run_id,
        "source_s3_prefix": prefix_uri,
        "project_root": str(project_root),
        "destination": str(destination),
        "method_revision": spec["method_revision"],
        "source_revision": spec["source_revision"],
        "launcher_revision": spec["launcher_revision"],
        "object_count": len(imported),
        "total_size_bytes": sum(
            record["size_bytes"] for record in imported.values()
        ),
        "downloaded_count": sum(
            record["mode"] == "downloaded" for record in imported.values()
        ),
        "reused_count": sum(
            record["mode"] == "reused" for record in imported.values()
        ),
        "topology_gate_passed": topology["topology_gate_passed"],
        "replay_counts": summary["counts"],
        "replay_bundle_catalog": str(catalog_path),
        "replay_bundle_catalog_sha256": hashlib.sha256(
            catalog_path.read_bytes()
        ).hexdigest(),
        "objects": imported,
        "final_status": final_status,
    }
    save_json_atomic(receipt, receipt_path)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Import a hash-verified Greenland replay run into EFS"
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--destination")
    parser.add_argument(
        "--replay-manifest",
        default="docs/timefuse_checkpoint_replay_manifest.jsonl",
    )
    parser.add_argument("--catalog")
    parser.add_argument("--receipt")
    parser.add_argument("--region", default="us-east-1")
    args = parser.parse_args(argv)

    project_root = Path(args.project_root).resolve()
    destination = args.destination or (
        project_root / "outputs" / "greenland_runs" / args.run_id
    )
    replay_manifest = (
        project_root / args.replay_manifest
        if not Path(args.replay_manifest).is_absolute()
        else Path(args.replay_manifest)
    )
    receipt = fetch_greenland_run(
        _assumed_s3_client(args.region),
        args.run_id,
        project_root,
        destination,
        replay_manifest,
        catalog_path=args.catalog,
        receipt_path=args.receipt,
        required_filesystem="nfs4",
    )
    print(
        json.dumps(
            {
                "run_id": receipt["run_id"],
                "object_count": receipt["object_count"],
                "total_size_bytes": receipt["total_size_bytes"],
                "topology_gate_passed": receipt["topology_gate_passed"],
                "replay_counts": receipt["replay_counts"],
                "replay_bundle_catalog": receipt["replay_bundle_catalog"],
                "replay_bundle_catalog_sha256": receipt[
                    "replay_bundle_catalog_sha256"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
