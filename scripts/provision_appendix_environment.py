"""Provision the dedicated AutoGluon 1.4 Appendix environment on project EFS."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from scripts.appendix_environment import (
    DEFAULT_ENVIRONMENT,
    DEFAULT_RECEIPT,
    DEFAULT_REQUIREMENTS,
    appendix_environment_dir,
    appendix_python_entry,
    inspect_appendix_environment,
    managed_path,
    sha256_file,
)


MARKER = ".timeraf-appendix-environment.json"


def _write_json_atomic(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _filesystem_type(path, *, runner=subprocess.run):
    completed = runner(
        ["findmnt", "-T", str(path), "-o", "FSTYPE", "-n"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _run(command, *, environment, runner=subprocess.run):
    runner(command, check=True, env=environment)


def provision_environment(
    project_root,
    environment_path,
    bootstrap_python,
    requirements_path,
    receipt_path,
    source_revision,
    *,
    expected_cuda_devices=4,
    required_filesystem="nfs4",
    runner=subprocess.run,
    inspector=inspect_appendix_environment,
    filesystem_probe=_filesystem_type,
):
    if not re.fullmatch(r"[0-9a-f]{40}", source_revision):
        raise ValueError("source-revision must be a full lowercase commit SHA")
    project_root = Path(project_root).resolve()
    if not project_root.is_dir():
        raise ValueError(f"project-root does not exist: {project_root}")
    filesystem_type = filesystem_probe(project_root)
    if required_filesystem and filesystem_type != required_filesystem:
        raise ValueError(
            f"project-root filesystem is {filesystem_type}, "
            f"expected {required_filesystem}"
        )
    environment_path = appendix_environment_dir(
        project_root,
        environment_path,
    )
    requirements_path = managed_path(
        project_root,
        requirements_path,
        "Appendix requirements",
        require_exists=True,
    )
    if requirements_path.is_symlink() or not requirements_path.is_file():
        raise ValueError("Appendix requirements must be a regular file")
    receipt_path = managed_path(
        project_root,
        receipt_path,
        "Appendix environment receipt",
    )
    bootstrap_python = Path(bootstrap_python).resolve()
    if not bootstrap_python.is_file() or not os.access(
        bootstrap_python,
        os.X_OK,
    ):
        raise ValueError(
            f"bootstrap Python is not executable: {bootstrap_python}"
        )

    operations = project_root / "operations" / "appendix-environment"
    operations.mkdir(parents=True, exist_ok=True)
    lock_path = operations / "provision.lock"
    lock = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError(
            "Another Appendix environment provisioner holds the lock"
        ) from error

    marker_path = environment_path / MARKER
    if environment_path.exists():
        if not marker_path.is_file():
            raise ValueError(
                "Refusing to modify an unmarked existing Appendix environment"
            )
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("Appendix environment marker is invalid") from error
        if (
            marker.get("schema_version") != 1
            or marker.get("purpose") != "timeraf-appendix-autogluon-1.4"
        ):
            raise ValueError("Appendix environment marker identity drifted")
    else:
        environment_path.parent.mkdir(parents=True, exist_ok=True)
        _run(
            [
                str(bootstrap_python),
                "-m",
                "venv",
                str(environment_path),
            ],
            environment=dict(os.environ),
            runner=runner,
        )
        _write_json_atomic(
            {
                "schema_version": 1,
                "purpose": "timeraf-appendix-autogluon-1.4",
                "created_unix": time.time(),
            },
            marker_path,
        )

    python = appendix_python_entry(
        project_root,
        environment_path / "bin" / "python",
    )
    cache_root = project_root / ".cache" / "appendix-environment"
    cache_root.mkdir(parents=True, exist_ok=True)
    install_environment = dict(os.environ)
    install_environment.update(
        {
            "PIP_CACHE_DIR": str(cache_root / "pip"),
            "TMPDIR": str(cache_root / "tmp"),
            "TEMP": str(cache_root / "tmp"),
            "TMP": str(cache_root / "tmp"),
        }
    )
    (cache_root / "tmp").mkdir(parents=True, exist_ok=True)
    _run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--requirement",
            str(requirements_path),
        ],
        environment=install_environment,
        runner=runner,
    )
    _run(
        [str(python), "-m", "pip", "check"],
        environment=install_environment,
        runner=runner,
    )
    evidence = inspector(
        python,
        expected_cuda_devices=expected_cuda_devices,
        require_cuda=expected_cuda_devices > 0,
        runner=runner,
    )
    receipt = {
        "schema_version": 1,
        "recorded_unix": time.time(),
        "source_revision": source_revision,
        "project_root": str(project_root),
        "filesystem_type": filesystem_type,
        "environment_path": str(environment_path),
        "bootstrap_python": str(bootstrap_python),
        "requirements_path": str(requirements_path),
        "requirements_sha256": sha256_file(requirements_path),
        "cache_root": str(cache_root),
        "environment": evidence,
    }
    _write_json_atomic(receipt, receipt_path)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Create and verify the isolated AutoGluon 1.4 Appendix venv"
        )
    )
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--environment", default=DEFAULT_ENVIRONMENT)
    parser.add_argument("--bootstrap-python", default=sys.executable)
    parser.add_argument(
        "--requirements",
        default=DEFAULT_REQUIREMENTS,
    )
    parser.add_argument("--receipt", default=DEFAULT_RECEIPT)
    parser.add_argument("--expected-cuda-devices", type=int, default=4)
    parser.add_argument("--required-filesystem", default="nfs4")
    args = parser.parse_args(argv)
    if args.expected_cuda_devices < 0:
        parser.error("--expected-cuda-devices cannot be negative")
    receipt = provision_environment(
        args.project_root,
        args.environment,
        args.bootstrap_python,
        args.requirements,
        args.receipt,
        args.source_revision,
        expected_cuda_devices=args.expected_cuda_devices,
        required_filesystem=args.required_filesystem,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
