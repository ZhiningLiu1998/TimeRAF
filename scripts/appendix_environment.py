"""Validate the dedicated TimeFuse Appendix Python environment."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path


AUTOGLOUON_TIMESERIES_VERSION = "1.4.0"
TORCH_VERSION = "2.7.1"
DEFAULT_ENVIRONMENT = ".venv-appendix140"
DEFAULT_RECEIPT = "operations/appendix-environment/receipt.json"
DEFAULT_REQUIREMENTS = "requirements-appendix.lock"


def _lexical_path(value):
    return Path(os.path.abspath(os.path.expanduser(os.fspath(value))))


def managed_path(
    project_root,
    value,
    description,
    *,
    require_exists=False,
    allow_root=False,
):
    root = Path(project_root).resolve()
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    path = _lexical_path(path)
    if (path == root and not allow_root) or (
        path != root and root not in path.parents
    ):
        raise ValueError(f"{description} must live below project-root")
    resolved_parent = path.parent.resolve()
    if resolved_parent != root and root not in resolved_parent.parents:
        raise ValueError(
            f"{description} parent resolves outside project-root"
        )
    if require_exists and not path.exists():
        raise ValueError(f"{description} does not exist: {path}")
    return path


def appendix_environment_dir(project_root, value=None, *, require_exists=False):
    root = Path(project_root).resolve()
    environment = managed_path(
        root,
        value or root / DEFAULT_ENVIRONMENT,
        "Appendix environment",
        require_exists=require_exists,
    )
    if environment.exists():
        if environment.is_symlink() or not environment.is_dir():
            raise ValueError(
                "Appendix environment must be a real directory below "
                "project-root"
            )
        resolved = environment.resolve()
        if resolved == root or root not in resolved.parents:
            raise ValueError(
                "Appendix environment resolves outside project-root"
            )
    return environment


def appendix_python_entry(project_root, value=None):
    root = Path(project_root).resolve()
    if value is None:
        environment = appendix_environment_dir(
            root,
            require_exists=True,
        )
        python = environment / "bin" / "python"
    else:
        python = managed_path(
            root,
            value,
            "Appendix Python",
            require_exists=True,
        )
        if python.name != "python" or python.parent.name != "bin":
            raise ValueError(
                "Appendix Python must be a bin/python virtualenv entry"
            )
        environment = appendix_environment_dir(
            root,
            python.parent.parent,
            require_exists=True,
        )
    if not (environment / "pyvenv.cfg").is_file():
        raise ValueError("Appendix environment has no pyvenv.cfg")
    configuration = (environment / "pyvenv.cfg").read_text(
        encoding="utf-8"
    ).lower()
    if "include-system-site-packages = true" in configuration:
        raise ValueError(
            "Appendix environment cannot include mutable system packages"
        )
    if not python.is_file() or not os.access(python, os.X_OK):
        raise ValueError(f"Appendix Python is not executable: {python}")
    return python


def _probe_code():
    return """
import hashlib
import importlib.metadata as metadata
import json
import sys

import torch
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

inventory = sorted(
    (
        (distribution.metadata.get("Name") or "").lower(),
        distribution.version,
    )
    for distribution in metadata.distributions()
    if distribution.metadata.get("Name")
)
inventory_bytes = json.dumps(
    inventory,
    separators=(",", ":"),
).encode("utf-8")
print(json.dumps({
    "python_version": ".".join(map(str, sys.version_info[:3])),
    "autogluon_timeseries": metadata.version("autogluon.timeseries"),
    "torch": metadata.version("torch"),
    "torch_cuda": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "cuda_device_count": torch.cuda.device_count(),
    "cuda_device_names": [
        torch.cuda.get_device_name(index)
        for index in range(torch.cuda.device_count())
    ],
    "package_count": len(inventory),
    "package_inventory_sha256": hashlib.sha256(inventory_bytes).hexdigest(),
    "backend_symbols": sorted([
        TimeSeriesDataFrame.__name__,
        TimeSeriesPredictor.__name__,
    ]),
}, sort_keys=True))
""".strip()


def inspect_appendix_environment(
    python,
    *,
    expected_cuda_devices=None,
    require_cuda=True,
    runner=subprocess.run,
):
    python = _lexical_path(python)
    try:
        completed = runner(
            [str(python), "-c", _probe_code()],
            check=True,
            capture_output=True,
            text=True,
        )
        evidence = json.loads(completed.stdout)
    except (
        OSError,
        subprocess.CalledProcessError,
        json.JSONDecodeError,
    ) as error:
        raise ValueError(
            "Appendix Python failed its AutoGluon/CUDA import probe"
        ) from error

    try:
        python_parts = tuple(
            int(part) for part in evidence["python_version"].split(".")
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "Appendix Python reported an invalid version"
        ) from error
    valid = (
        (3, 9) <= python_parts[:2] < (3, 13)
        and evidence.get("autogluon_timeseries")
        == AUTOGLOUON_TIMESERIES_VERSION
        and evidence.get("torch") == TORCH_VERSION
        and isinstance(evidence.get("package_count"), int)
        and evidence["package_count"] > 0
        and isinstance(evidence.get("package_inventory_sha256"), str)
        and len(evidence["package_inventory_sha256"]) == 64
        and evidence.get("backend_symbols")
        == ["TimeSeriesDataFrame", "TimeSeriesPredictor"]
    )
    if not valid:
        raise ValueError(
            "Appendix environment version or import identity drifted"
        )
    if require_cuda and evidence.get("cuda_available") is not True:
        raise ValueError("Appendix environment cannot access CUDA")
    if (
        expected_cuda_devices is not None
        and evidence.get("cuda_device_count") != expected_cuda_devices
    ):
        raise ValueError(
            "Appendix environment sees the wrong number of CUDA devices"
        )
    return {
        "python_entry": str(python),
        **evidence,
    }


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_appendix_environment_receipt(
    receipt_path,
    project_root,
    source_revision,
    requirements_path,
    environment,
):
    receipt_path = managed_path(
        project_root,
        receipt_path,
        "Appendix environment receipt",
        require_exists=True,
    )
    if receipt_path.is_symlink() or not receipt_path.is_file():
        raise ValueError("Appendix environment receipt must be a regular file")
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("Appendix environment receipt is invalid") from error
    requirements_path = managed_path(
        project_root,
        requirements_path,
        "Appendix requirements",
        require_exists=True,
    )
    if requirements_path.is_symlink() or not requirements_path.is_file():
        raise ValueError("Appendix requirements must be a regular file")
    try:
        python_entry = managed_path(
            project_root,
            environment["python_entry"],
            "Appendix Python receipt entry",
            require_exists=True,
        )
    except (KeyError, TypeError) as error:
        raise ValueError(
            "Appendix environment receipt has no Python entry"
        ) from error
    expected_environment_path = python_entry.parent.parent
    if (
        receipt.get("schema_version") != 1
        or receipt.get("source_revision") != source_revision
        or receipt.get("project_root") != str(Path(project_root).resolve())
        or receipt.get("filesystem_type") != "nfs4"
        or receipt.get("environment_path")
        != str(expected_environment_path)
        or receipt.get("requirements_sha256")
        != sha256_file(requirements_path)
        or receipt.get("environment") != environment
    ):
        raise ValueError("Appendix environment receipt does not match this run")
    return receipt
