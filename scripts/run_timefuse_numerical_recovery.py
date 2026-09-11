"""Run one frozen numerical-recovery profile on the Studio A10G pool."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from scripts.greenland_common import (
    FULL_MATRIX_MANIFEST,
    METHOD_REVISION,
    NUMERICAL_RECOVERY_CELL_IDS,
    NUMERICAL_RECOVERY_CELL_IDS_SHA256,
    NUMERICAL_RECOVERY_COHORT_ID,
    NUMERICAL_RECOVERY_PROFILES,
    NUMERICAL_RECOVERY_PROTOCOL,
    NUMERICAL_RECOVERY_PROTOCOL_SHA256,
    validate_full_matrix_manifest,
    validate_numerical_recovery_protocol,
)
from ts_rag.matrix import save_json_atomic


A10G_INSTANCE_TYPE = "ml.g5.12xlarge"
A10G_GPUS = 4


def _below(root, value, label):
    root = Path(root).resolve()
    path = Path(value)
    path = path.resolve() if path.is_absolute() else (root / path).resolve()
    if path != root and root not in path.parents:
        raise ValueError(f"{label} must live below the artifact root")
    return path


def build_recovery_command(
    *,
    python_executable,
    source_root,
    artifact_root,
    manifest,
    output_root,
    checkpoint_root,
    profile,
    launcher_revision,
    dry_run=False,
):
    if profile not in NUMERICAL_RECOVERY_PROFILES:
        raise ValueError(f"Unsupported recovery profile: {profile}")
    command = [
        str(python_executable),
        str(source_root / "scripts" / "run_benchmark_matrix.py"),
        "--manifest",
        str(manifest),
        "--output-root",
        str(output_root),
        "--checkpoint-root",
        str(checkpoint_root),
        "--summary",
        str(output_root / "matrix_summary.json"),
        "--log-root",
        str(output_root / "logs"),
        "--seed",
        "2021",
        "--source-revision",
        METHOD_REVISION,
        "--launcher-revision",
        launcher_revision,
        "--instance-type",
        A10G_INSTANCE_TYPE,
        "--reserved-gpus-per-host",
        str(A10G_GPUS),
        "--processes-per-host",
        str(A10G_GPUS),
        "--gpu",
        "0",
        "--num-workers",
        "4",
        "--max-failures=0",
        "--working-root",
        str(artifact_root),
    ]
    for cell_id in NUMERICAL_RECOVERY_CELL_IDS:
        command.extend(["--cell-id", cell_id])
    for key, value in sorted(NUMERICAL_RECOVERY_PROFILES[profile].items()):
        encoded = json.dumps(value, separators=(",", ":"))
        command.extend(["--override", f"{key}={encoded}"])
    if dry_run:
        command.append("--dry-run")
    return command


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run one frozen TimeRAF numerical recovery profile"
    )
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument(
        "--profile",
        choices=tuple(NUMERICAL_RECOVERY_PROFILES),
        required=True,
    )
    parser.add_argument("--parent-run-id", required=True)
    parser.add_argument("--recovery-revision", required=True)
    parser.add_argument("--materialized-revision")
    parser.add_argument("--output-root")
    parser.add_argument("--checkpoint-root")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    source_root = Path(args.source_root).resolve()
    artifact_root = Path(args.artifact_root).resolve()
    if source_root != artifact_root and artifact_root not in source_root.parents:
        raise ValueError("source-root must live below artifact-root")
    if not re.fullmatch(r"[0-9a-f]{40}", args.recovery_revision):
        raise ValueError("recovery-revision must be a full Git commit")
    actual_revision = args.materialized_revision
    if actual_revision is None:
        actual_revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=source_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    if actual_revision != args.recovery_revision:
        raise ValueError("source-root does not match recovery-revision")
    if not re.fullmatch(
        r"[a-z0-9][a-z0-9-]{5,62}",
        args.parent_run_id,
    ):
        raise ValueError("parent-run-id must be a lowercase stable ID")

    manifest = source_root / FULL_MATRIX_MANIFEST
    protocol = source_root / NUMERICAL_RECOVERY_PROTOCOL
    validate_full_matrix_manifest(manifest)
    validate_numerical_recovery_protocol(protocol)
    default_root = (
        artifact_root
        / "outputs"
        / "numerical_recovery"
        / "a10g"
        / args.recovery_revision
        / args.profile
    )
    output_root = _below(
        artifact_root,
        args.output_root or default_root,
        "output-root",
    )
    checkpoint_root = _below(
        artifact_root,
        args.checkpoint_root or output_root / "checkpoints",
        "checkpoint-root",
    )
    metadata = {
        "schema_version": 1,
        "cohort_id": NUMERICAL_RECOVERY_COHORT_ID,
        "profile": args.profile,
        "overrides": NUMERICAL_RECOVERY_PROFILES[args.profile],
        "method_revision": METHOD_REVISION,
        "recovery_revision": args.recovery_revision,
        "parent_run_id": args.parent_run_id,
        "protocol": NUMERICAL_RECOVERY_PROTOCOL,
        "protocol_sha256": NUMERICAL_RECOVERY_PROTOCOL_SHA256,
        "cell_ids": list(NUMERICAL_RECOVERY_CELL_IDS),
        "cell_ids_sha256": NUMERICAL_RECOVERY_CELL_IDS_SHA256,
        "seed": 2021,
        "instance_type": A10G_INSTANCE_TYPE,
        "instance_count": 1,
        "reserved_gpus_per_host": A10G_GPUS,
        "processes_per_host": A10G_GPUS,
        "total_gpus": A10G_GPUS,
        "world_size": A10G_GPUS,
        "inactive_reserved_gpus": 0,
        "output_root": str(output_root),
        "checkpoint_root": str(checkpoint_root),
        "working_root": str(artifact_root),
    }
    metadata_path = output_root / "recovery_metadata.json"
    if metadata_path.exists():
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        if existing != metadata:
            raise ValueError("Existing recovery metadata has drifted")
    else:
        save_json_atomic(metadata, metadata_path)

    command = build_recovery_command(
        python_executable=sys.executable,
        source_root=source_root,
        artifact_root=artifact_root,
        manifest=manifest,
        output_root=output_root,
        checkpoint_root=checkpoint_root,
        profile=args.profile,
        launcher_revision=args.recovery_revision,
        dry_run=args.dry_run,
    )
    environment = dict(os.environ)
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = os.pathsep.join(
        value
        for value in (str(source_root), existing_pythonpath)
        if value
    )
    return subprocess.run(
        command,
        cwd=artifact_root,
        env=environment,
        check=False,
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
