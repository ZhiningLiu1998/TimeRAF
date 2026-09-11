import json
from types import SimpleNamespace

import pytest

from scripts import appendix_environment
from scripts import provision_appendix_environment as provision


REVISION = "a" * 40


def _evidence(python, *, cuda_devices=4, autogluon="1.4.0", torch="2.7.1"):
    return {
        "python_entry": str(python),
        "python_version": "3.12.11",
        "autogluon_timeseries": autogluon,
        "torch": torch,
        "torch_cuda": "12.6",
        "cuda_available": cuda_devices > 0,
        "cuda_device_count": cuda_devices,
        "cuda_device_names": ["NVIDIA A10G"] * cuda_devices,
        "package_count": 100,
        "package_inventory_sha256": "b" * 64,
        "backend_symbols": ["TimeSeriesDataFrame", "TimeSeriesPredictor"],
    }


def _make_venv(root, *, target=None):
    environment = root / ".venv-appendix140"
    (environment / "bin").mkdir(parents=True)
    (environment / "pyvenv.cfg").write_text("home = /opt/conda\n")
    target = target or root / "base-python"
    target.write_text("#!/bin/sh\n")
    target.chmod(0o755)
    (environment / "bin" / "python").symlink_to(target)
    return environment, environment / "bin" / "python"


def test_python_entry_preserves_efs_venv_symlink(tmp_path):
    outside = tmp_path.parent / f"{tmp_path.name}-python"
    outside.write_text("#!/bin/sh\n")
    outside.chmod(0o755)
    environment, python = _make_venv(tmp_path, target=outside)

    entry = appendix_environment.appendix_python_entry(tmp_path, python)

    assert entry == python
    assert entry.resolve() == outside
    assert environment.is_dir()


def test_environment_probe_requires_exact_versions_and_cuda(tmp_path):
    _environment, python = _make_venv(tmp_path)

    def runner_for(evidence):
        return lambda *_args, **_kwargs: SimpleNamespace(
            stdout=json.dumps(evidence)
        )

    valid = _evidence(python)
    inspected = appendix_environment.inspect_appendix_environment(
        python,
        expected_cuda_devices=4,
        runner=runner_for(valid),
    )
    assert inspected["autogluon_timeseries"] == "1.4.0"
    assert inspected["package_inventory_sha256"] == "b" * 64

    drifted = _evidence(python, autogluon="1.5.0")
    with pytest.raises(ValueError, match="identity drifted"):
        appendix_environment.inspect_appendix_environment(
            python,
            expected_cuda_devices=4,
            runner=runner_for(drifted),
        )

    no_cuda = _evidence(python, cuda_devices=0)
    with pytest.raises(ValueError, match="cannot access CUDA"):
        appendix_environment.inspect_appendix_environment(
            python,
            expected_cuda_devices=4,
            runner=runner_for(no_cuda),
        )


def test_provisioner_creates_marked_environment_and_receipt(tmp_path):
    requirements = tmp_path / "requirements-appendix.txt"
    requirements.write_text(
        "torch==2.7.1\nautogluon.timeseries==1.4.0\n"
    )
    bootstrap = tmp_path / "bootstrap-python"
    bootstrap.write_text("#!/bin/sh\n")
    bootstrap.chmod(0o755)
    commands = []

    def runner(command, **_kwargs):
        commands.append(command)
        if command[1:3] == ["-m", "venv"]:
            environment = tmp_path / ".venv-appendix140"
            (environment / "bin").mkdir(parents=True)
            (environment / "pyvenv.cfg").write_text("home = /opt/conda\n")
            python = environment / "bin" / "python"
            python.write_text("#!/bin/sh\n")
            python.chmod(0o755)
        return SimpleNamespace(stdout="")

    def inspector(python, **_kwargs):
        return _evidence(python)

    receipt = provision.provision_environment(
        tmp_path,
        ".venv-appendix140",
        bootstrap,
        requirements,
        "operations/appendix-environment/receipt.json",
        REVISION,
        runner=runner,
        inspector=inspector,
        filesystem_probe=lambda _path: "nfs4",
    )

    assert commands[0][1:3] == ["-m", "venv"]
    assert commands[1][1:4] == ["-m", "pip", "install"]
    assert commands[2][1:4] == ["-m", "pip", "check"]
    assert receipt["environment"]["torch"] == "2.7.1"
    receipt_path = (
        tmp_path / "operations" / "appendix-environment" / "receipt.json"
    )
    assert json.loads(receipt_path.read_text()) == receipt


def test_provisioner_rejects_unmarked_existing_environment(tmp_path):
    requirements = tmp_path / "requirements-appendix.txt"
    requirements.write_text("autogluon.timeseries==1.4.0\n")
    bootstrap = tmp_path / "python"
    bootstrap.write_text("#!/bin/sh\n")
    bootstrap.chmod(0o755)
    (tmp_path / ".venv-appendix140").mkdir()

    with pytest.raises(ValueError, match="unmarked"):
        provision.provision_environment(
            tmp_path,
            ".venv-appendix140",
            bootstrap,
            requirements,
            "operations/appendix-environment/receipt.json",
            REVISION,
            runner=lambda *_args, **_kwargs: SimpleNamespace(stdout=""),
            filesystem_probe=lambda _path: "nfs4",
        )


def test_receipt_binds_source_requirements_and_inventory(tmp_path):
    requirements = tmp_path / "requirements-appendix.txt"
    requirements.write_text(
        "torch==2.7.1\nautogluon.timeseries==1.4.0\n"
    )
    environment = _evidence(
        tmp_path / ".venv-appendix140" / "bin" / "python"
    )
    receipt = {
        "schema_version": 1,
        "source_revision": REVISION,
        "project_root": str(tmp_path.resolve()),
        "filesystem_type": "nfs4",
        "environment_path": str(tmp_path / ".venv-appendix140"),
        "requirements_sha256": appendix_environment.sha256_file(requirements),
        "environment": environment,
    }
    _make_venv(tmp_path)
    receipt_path = (
        tmp_path / "operations" / "appendix-environment" / "receipt.json"
    )
    receipt_path.parent.mkdir(parents=True)
    receipt_path.write_text(json.dumps(receipt))

    verified = appendix_environment.verify_appendix_environment_receipt(
        receipt_path,
        tmp_path,
        REVISION,
        requirements,
        environment,
    )
    assert verified == receipt

    environment["package_inventory_sha256"] = "c" * 64
    with pytest.raises(ValueError, match="does not match"):
        appendix_environment.verify_appendix_environment_receipt(
            receipt_path,
            tmp_path,
            REVISION,
            requirements,
            environment,
        )


def test_managed_path_rejects_parent_symlink_escape(tmp_path):
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    (tmp_path / "linked").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="parent resolves outside"):
        appendix_environment.managed_path(
            tmp_path,
            tmp_path / "linked" / "receipt.json",
            "receipt",
        )
