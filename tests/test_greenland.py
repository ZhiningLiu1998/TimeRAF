import hashlib
import io
import json
import subprocess
import tarfile
from pathlib import Path

import pytest

from scripts import greenland_common
from scripts import build_checkpoint_replay_manifest
from scripts import greenland_credential_proxy
from scripts import prepare_greenland_inputs
from scripts import submit_greenland_job
from scripts import greenland_run_status


DIGESTS = {
    "source": "1" * 64,
    "dataset": "2" * 64,
    "checkpoints": "3" * 64,
}
IMAGE_URI = (
    "<ECR_REGISTRY>/"
    "timeraf-greenland@sha256:" + "a" * 64
)


def _image_approval(image_uri=IMAGE_URI):
    return {
        "schema_version": 1,
        "image_uri": image_uri,
        "image_digest": image_uri.rsplit("@", 1)[-1],
        "image_source_revision": "a" * 40,
        "approved_for": [
            "checkpoint_replay",
            "full_matrix_replication",
            "supplemental_numerical_recovery",
        ],
        "black_box_verification": {
            "paper_model_imports_passed": 13,
            "pip_check_passed": True,
            "png_jpeg_codecs_passed": True,
            "matplotlib_render_passed": True,
            "checkpoint_preflight_present": True,
            "numerical_recovery_preflight_present": True,
            "removed_development_packages_absent": True,
        },
        "ecr_scan": {
            "status": "COMPLETE",
            "finding_severity_counts": {},
        },
    }


def _spec():
    run_id = "replay-20260730"
    return {
        "schema_version": 1,
        "run_id": run_id,
        "job_kind": "checkpoint_replay",
        "method_revision": greenland_common.METHOD_REVISION,
        "image_uri": IMAGE_URI,
        "source_revision": "a1b2c3d",
        "launcher_revision": "d4e5f6a",
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
            for name, digest in DIGESTS.items()
        ],
        "output_s3_uri": (
            "s3://<DEV_BUCKET>/timeraf/greenland/"
            f"runs/{run_id}/"
        ),
        "matrix": {
            "manifest": "docs/timefuse_checkpoint_replay_manifest.jsonl",
            "output_root": "outputs/checkpoint_replay",
            "checkpoint_root": "checkpoints/replay",
            "checkpoint_catalog": "docs/checkpoint_catalog.json",
            "release_checkpoint_root": "checkpoints/replay",
            "summary": "outputs/checkpoint_replay/summary.json",
            "log_root": "outputs/checkpoint_replay/logs",
            "queued_cells": 208,
            "seed": 2021,
            "extra_args": [
                "--no-train",
                "--export-only",
                "--max-failures=0",
            ],
        },
    }


def _full_matrix_spec():
    spec = _spec()
    spec["run_id"] = "full-matrix-20260731"
    spec["job_kind"] = "full_matrix_replication"
    spec["inputs"] = [
        record
        for record in spec["inputs"]
        if record["name"] != "checkpoints"
    ]
    spec["output_s3_uri"] = (
        "s3://<DEV_BUCKET>/timeraf/greenland/"
        "runs/full-matrix-20260731/"
    )
    spec["matrix"] = {
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
        "extra_args": [
            "--max-failures=0",
            "--num-workers=4",
        ],
    }
    return spec


def _numerical_recovery_spec(profile="exact"):
    spec = _full_matrix_spec()
    source_revision = "a" * 40
    run_id = f"numerical-recovery-{profile.replace('_', '-')}"
    spec.update(
        {
            "schema_version": 2,
            "run_id": run_id,
            "job_kind": greenland_common.NUMERICAL_RECOVERY_JOB,
            "source_revision": source_revision,
            "launcher_revision": "b" * 40,
            "recovery_revision": source_revision,
            "output_s3_uri": (
                "s3://<DEV_BUCKET>/timeraf/greenland/"
                f"runs/{run_id}/"
            ),
        }
    )
    root = f"outputs/numerical_recovery/{profile}"
    spec["matrix"] = {
        "manifest": greenland_common.FULL_MATRIX_MANIFEST,
        "output_root": root,
        "checkpoint_root": f"{root}/checkpoints",
        "summary": f"{root}/matrix_summary.json",
        "log_root": f"{root}/logs",
        "queued_cells": greenland_common.NUMERICAL_RECOVERY_CELLS,
        "seed": 2021,
        "extra_args": [
            "--max-failures=0",
            "--num-workers=4",
        ],
    }
    spec["recovery"] = {
        "protocol": greenland_common.NUMERICAL_RECOVERY_PROTOCOL,
        "protocol_sha256": (
            greenland_common.NUMERICAL_RECOVERY_PROTOCOL_SHA256
        ),
        "cohort_id": greenland_common.NUMERICAL_RECOVERY_COHORT_ID,
        "cell_ids": list(greenland_common.NUMERICAL_RECOVERY_CELL_IDS),
        "cell_ids_sha256": (
            greenland_common.NUMERICAL_RECOVERY_CELL_IDS_SHA256
        ),
        "profile": profile,
        "overrides": greenland_common.NUMERICAL_RECOVERY_PROFILES[profile],
        "parent_run_id": "full-matrix-a100-9974eac-20260731",
    }
    return spec


def test_job_spec_requires_content_addressed_inputs_and_eight_cells():
    spec = _spec()

    assert greenland_common.validate_job_spec(spec) is spec

    spec["inputs"][0]["s3_uri"] = (
        "s3://<DEV_BUCKET>/timeraf/greenland/inputs/source/latest.tar.gz"
    )
    with pytest.raises(ValueError, match="key must begin"):
        greenland_common.validate_job_spec(spec)

    spec = _spec()
    spec["matrix"]["queued_cells"] = 207
    with pytest.raises(ValueError, match="exactly 208"):
        greenland_common.validate_job_spec(spec)

    spec = _spec()
    spec["matrix"]["manifest"] = "docs/timefuse_experiment_manifest.jsonl"
    with pytest.raises(ValueError, match="checkpoint replay manifest"):
        greenland_common.validate_job_spec(spec)

    spec = _spec()
    spec["method_revision"] = "other"
    with pytest.raises(ValueError, match="method revision"):
        greenland_common.validate_job_spec(spec)


def test_full_matrix_job_requires_exact_training_scope():
    spec = _full_matrix_spec()

    assert greenland_common.validate_job_spec(spec) is spec

    spec["matrix"]["queued_cells"] = 584
    with pytest.raises(ValueError, match="exactly 585"):
        greenland_common.validate_job_spec(spec)

    spec = _full_matrix_spec()
    spec["matrix"]["extra_args"].append("--no-train")
    with pytest.raises(ValueError, match="must train and evaluate"):
        greenland_common.validate_job_spec(spec)

    spec = _full_matrix_spec()
    spec["matrix"]["extra_args"].remove("--num-workers=4")
    with pytest.raises(ValueError, match="num-workers=4"):
        greenland_common.validate_job_spec(spec)

    spec = _full_matrix_spec()
    spec["matrix"]["extra_args"].append("--num-workers=8")
    with pytest.raises(ValueError, match="num-workers=4"):
        greenland_common.validate_job_spec(spec)

    spec = _full_matrix_spec()
    spec["inputs"].append(_spec()["inputs"][2])
    with pytest.raises(ValueError, match="requires inputs"):
        greenland_common.validate_job_spec(spec)


def test_numerical_recovery_spec_is_exactly_protocol_bound():
    spec = _numerical_recovery_spec()

    assert greenland_common.validate_job_spec(spec) is spec

    spec["schema_version"] = 1
    with pytest.raises(ValueError, match="schema 2"):
        greenland_common.validate_job_spec(spec)

    spec = _numerical_recovery_spec()
    spec["matrix"]["queued_cells"] = 8
    with pytest.raises(ValueError, match="exactly 9"):
        greenland_common.validate_job_spec(spec)

    spec = _numerical_recovery_spec()
    spec["recovery"]["cell_ids"].pop()
    with pytest.raises(ValueError, match="frozen protocol"):
        greenland_common.validate_job_spec(spec)

    spec = _numerical_recovery_spec()
    spec["recovery"]["overrides"] = {"learning_rate": 0.0001}
    with pytest.raises(ValueError, match="frozen protocol"):
        greenland_common.validate_job_spec(spec)

    fallback = _numerical_recovery_spec("fallback-v1")
    assert greenland_common.validate_job_spec(fallback) is fallback


def test_numerical_recovery_protocol_and_command_are_pinned(tmp_path):
    protocol = (
        Path(__file__).resolve().parents[1]
        / greenland_common.NUMERICAL_RECOVERY_PROTOCOL
    )
    evidence = greenland_common.validate_numerical_recovery_protocol(
        protocol
    )

    assert evidence["cell_ids"] == list(
        greenland_common.NUMERICAL_RECOVERY_CELL_IDS
    )
    assert evidence["profiles"]["fallback-v1"] == {
        "learning_rate": 0.0001
    }

    tampered = tmp_path / "protocol.json"
    tampered.write_bytes(protocol.read_bytes() + b" ")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        greenland_common.validate_numerical_recovery_protocol(tampered)

    command = greenland_common.build_matrix_command(
        _numerical_recovery_spec("fallback-v1"),
        tmp_path,
        "/usr/bin/python3",
    )
    assert command.count("--cell-id") == 9
    assert command.count("--override") == 1
    override_index = command.index("--override")
    assert command[override_index + 1] == "learning_rate=0.0001"
    assert "--processes-per-host" in command
    assert command[command.index("--processes-per-host") + 1] == "8"


def test_numerical_recovery_source_preflight_exercises_checkpoint_failure():
    project = Path(__file__).resolve().parents[1]
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    evidence = (
        prepare_greenland_inputs._validate_numerical_integrity_source(
            project,
            revision,
        )
    )

    assert evidence == {
        "contract": "explicit-numerical-integrity-error-v1",
        "source_revision": revision,
        "entrypoint": "scripts/run_benchmark_cell.py",
        "entrypoint_sha256": prepare_greenland_inputs.sha256_file(
            project / "scripts" / "run_benchmark_cell.py"
        ),
        "passed": True,
    }


def test_approved_image_record_rejects_unapproved_or_vulnerable_image(
    tmp_path,
):
    approval = _image_approval()
    path = tmp_path / "approval.json"
    path.write_text(json.dumps(approval))

    assert (
        greenland_common.validate_approved_image_uri(IMAGE_URI, path)
        == approval
    )

    with pytest.raises(ValueError, match="not the approved"):
        greenland_common.validate_approved_image_uri(
            IMAGE_URI.replace("a" * 64, "b" * 64),
            path,
        )
    approval["black_box_verification"]["checkpoint_preflight_present"] = False
    path.write_text(json.dumps(approval))
    with pytest.raises(ValueError, match="black-box"):
        greenland_common.validate_approved_image_uri(IMAGE_URI, path)

    approval["black_box_verification"]["checkpoint_preflight_present"] = True
    approval["black_box_verification"][
        "numerical_recovery_preflight_present"
    ] = False
    path.write_text(json.dumps(approval))
    with pytest.raises(ValueError, match="black-box"):
        greenland_common.validate_approved_image_uri(
            IMAGE_URI,
            path,
            required_job_kind=greenland_common.NUMERICAL_RECOVERY_JOB,
        )

    approval["black_box_verification"][
        "numerical_recovery_preflight_present"
    ] = True
    approval["ecr_scan"]["finding_severity_counts"] = {"HIGH": 1}
    path.write_text(json.dumps(approval))
    with pytest.raises(ValueError, match="zero-finding"):
        greenland_common.validate_approved_image_uri(IMAGE_URI, path)


def test_repository_approved_image_record_is_valid():
    path = Path(greenland_common.APPROVED_IMAGE_RECORD)
    image_uri = (
        "<ECR_REGISTRY>/"
        "timeraf-greenland@sha256:"
        "2417e94636d74e455f8bfdc97d7f7e4ff9022107e6ea7b2ebc4a4295e994d25d"
    )
    approval = greenland_common.validate_approved_image_uri(
        approval_record=path,
        image_uri=image_uri,
    )
    assert greenland_common.validate_approved_image_uri(
        approval_record=path,
        image_uri=image_uri,
        required_job_kind=greenland_common.FULL_MATRIX_REPLICATION_JOB,
    ) == approval

    assert approval["image_source_revision"] == (
        "b41a2ad3b6856c441c244ee082c139304a8ad13e"
    )
    assert greenland_common.validate_approved_image_uri(
        approval_record=path,
        image_uri=image_uri,
        required_job_kind=greenland_common.NUMERICAL_RECOVERY_JOB,
    ) == approval
    assert approval["ecr_scan"]["finding_severity_counts"] == {}


def test_greenland_terminal_status_requires_exact_job_identity():
    pending = {
        "run_id": "full-matrix-fixture",
        "upload_complete": False,
        "upload_failed": False,
    }
    assert greenland_run_status.validate_terminal_identity(
        pending,
        job_kind="full_matrix_replication",
        method_revision=greenland_common.METHOD_REVISION,
        source_revision="a" * 40,
        launcher_revision="a" * 40,
    ) is pending

    completed = {
        **pending,
        "upload_complete": True,
        "job_kind": "full_matrix_replication",
        "method_revision": greenland_common.METHOD_REVISION,
        "source_revision": "a" * 40,
        "launcher_revision": "a" * 40,
    }
    assert greenland_run_status.validate_terminal_identity(
        completed,
        job_kind="full_matrix_replication",
        method_revision=greenland_common.METHOD_REVISION,
        source_revision="a" * 40,
        launcher_revision="a" * 40,
    ) is completed

    completed["source_revision"] = "b" * 40
    with pytest.raises(ValueError, match="identity drifted"):
        greenland_run_status.validate_terminal_identity(
            completed,
            job_kind="full_matrix_replication",
            method_revision=greenland_common.METHOD_REVISION,
            source_revision="a" * 40,
            launcher_revision="a" * 40,
        )


def test_replication_worker_uses_fail_closed_remote_gates():
    worker = Path(
        "scripts/timefuse_greenland_replication_worker.sh"
    ).read_text(encoding="utf-8")

    assert ".venv-gpu312/bin/python" in worker
    assert "waiting for valid Greenland status JSON" in worker
    assert "REMOTE_MARKER=" in worker
    assert "export AWS_PROFILE='$AWS_PROFILE'" not in worker
    assert "TIMERAF_GREENLAND_POST_IMPORT_ACTION" in worker
    assert "TIMERAF_GREENLAND_CONTROLLER_REVISION" in worker
    assert "'$REMOTE_SOURCE/scripts/greenland_run_status.py'" in worker
    assert (
        "'$REMOTE_CONTROLLER_SOURCE/scripts/fetch_greenland_full_matrix.py'"
        in worker
    )
    assert "recomputed_matrix_summary.sha256" in worker
    assert "recomputed_matrix_summary.size_bytes" in worker
    assert "--accept-terminal-nonfinite-first-pass" in worker
    assert (
        'import_acceptance.mode == \\"terminal-nonfinite-first-pass\\"'
        in worker
    )
    assert "recomputed_summary_sha256=" in worker
    assert (
        'event "STOPPED run_id=$RUN_ID revision=$REVISION rc=$rc"'
        in worker
    )
    assert '== "import-only"' in worker
    assert "COMPLETE_IMPORT_ONLY" in worker
    assert worker.index("COMPLETE_IMPORT_ONLY") < worker.index(
        "waiting for A10G comparison readiness"
    )


def test_matrix_command_hard_codes_full_p4de_topology(tmp_path):
    command = greenland_common.build_matrix_command(
        _spec(), tmp_path, "/usr/bin/python"
    )

    assert command[command.index("--instance-type") + 1] == "ml.p4de.24xlarge"
    assert command[command.index("--reserved-gpus-per-host") + 1] == "8"
    assert command[command.index("--processes-per-host") + 1] == "8"
    assert command[command.index("--gpu") + 1] == "0"
    assert "--cpu" not in command


def test_full_matrix_command_trains_into_uploaded_output_tree(tmp_path):
    command = greenland_common.build_matrix_command(
        _full_matrix_spec(),
        tmp_path,
        "/usr/bin/python",
    )

    assert command[command.index("--source-revision") + 1] == (
        greenland_common.METHOD_REVISION
    )
    assert command[command.index("--launcher-revision") + 1] == "d4e5f6a"
    checkpoint_root = command[command.index("--checkpoint-root") + 1]
    assert checkpoint_root == str(
        tmp_path
        / "outputs"
        / "full_matrix_replication"
        / "checkpoints"
    )
    assert "--checkpoint-catalog" not in command
    assert "--no-train" not in command
    assert "--export-only" not in command
    assert "--num-workers=4" in command


def test_matrix_args_cannot_override_topology():
    spec = _spec()
    spec["matrix"]["extra_args"].append("--processes-per-host=4")

    with pytest.raises(ValueError, match="cannot override"):
        greenland_common.validate_job_spec(spec)

    spec = _full_matrix_spec()
    spec["matrix"]["extra_args"].append(
        "--override=learning_rate=0.0001"
    )
    with pytest.raises(ValueError, match="cannot override"):
        greenland_common.validate_job_spec(spec)


def test_matrix_args_cannot_filter_or_stop_replay_early():
    spec = _spec()
    spec["matrix"]["extra_args"].append("--max-cells=8")
    with pytest.raises(ValueError, match="cannot override"):
        greenland_common.validate_job_spec(spec)

    spec = _spec()
    spec["matrix"]["extra_args"].remove("--max-failures=0")
    with pytest.raises(ValueError, match="max-failures=0"):
        greenland_common.validate_job_spec(spec)

    spec = _spec()
    spec["matrix"]["extra_args"].append("--max-failures=1")
    with pytest.raises(ValueError, match="permit only"):
        greenland_common.validate_job_spec(spec)


def test_credential_watcher_waits_for_main_process(monkeypatch):
    snapshots = iter(
        [
            ["1", "99"],
            ["1", "99", "100"],
            ["1", "99"],
        ]
    )
    exit_codes = []

    monkeypatch.setattr(
        greenland_credential_proxy.os, "getpid", lambda: 99
    )
    monkeypatch.setattr(
        greenland_credential_proxy.os,
        "listdir",
        lambda _path: next(snapshots),
    )
    monkeypatch.setattr(
        greenland_credential_proxy.time, "sleep", lambda _seconds: None
    )

    def exit_process(code):
        exit_codes.append(code)
        raise SystemExit(code)

    monkeypatch.setattr(
        greenland_credential_proxy.os, "_exit", exit_process
    )

    with pytest.raises(SystemExit, match="0"):
        greenland_credential_proxy._main_container_watcher()

    assert exit_codes == [0]


def test_safe_extract_rejects_path_traversal(tmp_path):
    archive_path = tmp_path / "unsafe.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        info = tarfile.TarInfo("../../outside.txt")
        payload = b"unsafe"
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))

    with pytest.raises(ValueError, match="Unsafe archive member"):
        greenland_common.safe_extract_tar(
            archive_path, tmp_path / "destination"
        )

    assert not (tmp_path / "outside.txt").exists()


def _sample(utilization=20, correct_bindings=True):
    gpus = []
    for index in range(8):
        visible = str(index if correct_bindings else (index + 1) % 8)
        gpus.append(
            {
                "index": index,
                "utilization_gpu_percent": utilization,
                "processes": [
                    {
                        "pid": 1000 + index,
                        "cuda_visible_devices": visible,
                    }
                ],
            }
        )
    return {"recorded_unix": 1.0, "gpus": gpus}


def test_gpu_evidence_requires_distinct_bindings_and_all_utilized():
    passed = greenland_common.summarize_gpu_samples([_sample()])
    wrong_binding = greenland_common.summarize_gpu_samples(
        [_sample(correct_bindings=False)]
    )
    idle = greenland_common.summarize_gpu_samples([_sample(utilization=0)])

    assert passed["topology_gate_passed"] is True
    assert passed["binding_evidence"]["gpu_ids"] == list(range(8))
    assert len(set(passed["binding_evidence"]["pids"])) == 8
    assert wrong_binding["eight_distinct_bindings_observed"] is False
    assert idle["all_eight_gpus_utilized"] is False
    assert idle["topology_gate_passed"] is False
    assert greenland_common.validate_gpu_topology_evidence(passed) is passed

    forged = dict(passed)
    forged["binding_evidence"] = {
        **passed["binding_evidence"],
        "pids": [1000] * 8,
    }
    with pytest.raises(ValueError, match="eight active distinct A100"):
        greenland_common.validate_gpu_topology_evidence(forged)


class _Resource:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _Role:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.metadata = {}


class _AppDef:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _Specs:
    Resource = _Resource
    Role = _Role
    AppDef = _AppDef


def test_app_definition_requests_one_pod_with_all_eight_gpus():
    overlays = []

    def set_overlay(role, backend, kind, overlay):
        overlays.append((role, backend, kind, overlay))

    app = submit_greenland_job.build_app_definition(
        _spec(),
        "s3://<DEV_BUCKET>/timeraf/greenland/specs/" + "f" * 64 + ".json",
        "f" * 64,
        torchx_specs=_Specs,
        set_overlay=set_overlay,
        auth_token="test-token",
    )

    assert len(app.roles) == 1
    role = app.roles[0]
    assert role.num_replicas == 1
    assert role.min_replicas == 1
    assert role.resource.gpu == 8
    assert role.resource.capabilities == {
        "node.kubernetes.io/instance-type": "p4de.24xlarge"
    }
    assert role.env["TIMERAF_IMAGE_URI"] == IMAGE_URI
    assert overlays[0][1:3] == ("kubernetes", "V1Pod")
    pod = overlays[0][3]["spec"]
    assert pod["shareProcessNamespace"] is True
    assert pod["containers"][0]["name"] == "credential-proxy"
    assert (
        pod["containers"][0]["volumeMounts"][0]["name"]
        == "greenland-aws-config"
    )


def test_manifest_validator_accepts_torchx_replica_container_name(monkeypatch):
    manifest = {
        "metadata": {"name": "timeraf-test-unique"},
        "spec": {
            "minAvailable": 1,
            "tasks": [
                {
                    "replicas": 1,
                    "template": {
                        "spec": {
                            "nodeSelector": {
                                "node.kubernetes.io/instance-type":
                                "p4de.24xlarge"
                            },
                            "containers": [
                                {
                                    "name": "timeraf-0",
                                    "image": IMAGE_URI,
                                    "resources": {
                                        "limits": {"nvidia.com/gpu": "8"},
                                        "requests": {"nvidia.com/gpu": "8"},
                                    },
                                },
                                {"name": "credential-proxy"},
                            ],
                        }
                    },
                }
            ],
        },
    }

    class _Yaml:
        @staticmethod
        def safe_load(_payload):
            return manifest

    monkeypatch.setitem(__import__("sys").modules, "yaml", _Yaml)
    dryrun = type(
        "DryRun",
        (),
        {
            "request": type(
                "Request",
                (),
                {
                    "instance_type": "p4de.24xlarge",
                    "instance_count": 1,
                    "eks_manifest_yaml": "rendered",
                },
            )()
        },
    )()

    assert submit_greenland_job.validate_scheduler_manifest(
        dryrun, IMAGE_URI
    ) is manifest


def test_directory_archives_are_deterministic_and_project_relative(tmp_path):
    project = tmp_path / "project"
    dataset = project / "dataset"
    dataset.mkdir(parents=True)
    (dataset / "values.csv").write_text("time,value\n1,2\n", encoding="utf-8")
    first = project / "first.tar.gz"
    second = project / "second.tar.gz"

    prepare_greenland_inputs._build_directory_archive(
        project, dataset, first
    )
    prepare_greenland_inputs._build_directory_archive(
        project, dataset, second
    )

    assert greenland_common.sha256_file(first) == greenland_common.sha256_file(
        second
    )
    with tarfile.open(first, "r:gz") as archive:
        assert archive.getnames() == ["dataset", "dataset/values.csv"]


def test_canonical_job_spec_digest_is_stable():
    first = greenland_common.canonical_json_bytes(_spec())
    second = greenland_common.canonical_json_bytes(
        json.loads(first.decode("utf-8"))
    )

    assert first == second
    assert greenland_common.sha256_bytes(first) == (
        greenland_common.sha256_bytes(second)
    )


def test_checkpoint_replay_manifest_has_exact_family_horizons():
    with open(
        "docs/timefuse_experiment_manifest.jsonl", "r", encoding="utf-8"
    ) as source:
        rows = [json.loads(line) for line in source if line.strip()]

    selected = (
        build_checkpoint_replay_manifest.select_checkpoint_replay_cells(rows)
    )

    assert len(selected) == 208
    assert {
        (row["task_family"], row["pred_len"]) for row in selected
    } == {("long_term", 96), ("pems", 24), ("epf", 24)}

    replay_path = Path("docs/timefuse_checkpoint_replay_manifest.jsonl")
    replay_rows = [
        json.loads(line)
        for line in replay_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert replay_rows == selected
    evidence = greenland_common.validate_replay_manifest(replay_path)
    assert evidence["cells"] == 208
    assert evidence["sha256"] == (
        "447391d99c46bc1dd4170e71a8388bad"
        "5edb0a4c48a3048e230e404b96a0a77a"
    )
    assert evidence["family_counts"] == {
        "long_term": 91,
        "pems": 52,
        "epf": 65,
    }


def test_checkpoint_replay_manifest_rejects_content_drift(tmp_path):
    manifest = tmp_path / "replay.jsonl"
    manifest.write_bytes(
        Path("docs/timefuse_checkpoint_replay_manifest.jsonl").read_bytes()
        + b"\n"
    )

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        greenland_common.validate_replay_manifest(manifest)


def test_full_matrix_manifest_is_exact_and_frozen():
    evidence = greenland_common.validate_full_matrix_manifest(
        "docs/timefuse_experiment_manifest.jsonl"
    )

    assert evidence["cells"] == 585
    assert evidence["family_counts"] == {
        "long_term": 364,
        "pems": 156,
        "epf": 65,
    }
    assert evidence["sha256"] == (
        "45830d23f3b017c15d3680c88f837f84"
        "a9441eef9a6ceaaa5370c14872660c2a"
    )


def _replay_checkpoint_fixture(tmp_path):
    project = tmp_path / "project"
    docs = project / "docs"
    docs.mkdir(parents=True)
    manifest = docs / "timefuse_checkpoint_replay_manifest.jsonl"
    manifest.write_bytes(
        Path("docs/timefuse_checkpoint_replay_manifest.jsonl").read_bytes()
    )
    rows = [
        json.loads(line)
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line
    ]
    checkpoint_root = project / "checkpoints" / "replay"
    records = {}
    for index, row in enumerate(rows):
        checkpoint = checkpoint_root / f"{index:03d}" / "checkpoint.pth"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(row["id"].encode("utf-8"))
        records[row["id"]] = {
            "usable": True,
            "relative_path": str(checkpoint.relative_to(checkpoint_root)),
            "file_size": checkpoint.stat().st_size,
            "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "source_revision": greenland_common.METHOD_REVISION,
        }
    catalog = docs / "replay-checkpoints.json"
    catalog.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_revision": greenland_common.METHOD_REVISION,
                "selected_count": 208,
                "checkpoint_count": 208,
                "usable_count": 208,
                "missing_count": 0,
                "missing_selected_count": 0,
                "hashes_included": True,
                "checkpoints": records,
            }
        ),
        encoding="utf-8",
    )
    return project, manifest, catalog, checkpoint_root, rows


def test_checkpoint_archive_contains_exact_verified_replay_scope(tmp_path):
    project, manifest, catalog, checkpoint_root, rows = (
        _replay_checkpoint_fixture(tmp_path)
    )
    archive_path = project / "checkpoints.tar.gz"

    evidence = prepare_greenland_inputs._build_checkpoint_archive(
        project,
        manifest,
        catalog,
        checkpoint_root,
        archive_path,
    )

    assert evidence["checkpoint_count"] == 208
    assert evidence["unique_checkpoint_paths"] == 208
    assert evidence["hashes_verified"] is True
    with tarfile.open(archive_path, "r:gz") as archive:
        names = archive.getnames()
    assert len(names) == len(rows)
    assert all(name.startswith("checkpoints/replay/") for name in names)


def test_replay_checkpoint_validator_rejects_hash_drift(tmp_path):
    project, manifest, catalog, checkpoint_root, _rows = (
        _replay_checkpoint_fixture(tmp_path)
    )
    first = next(checkpoint_root.rglob("checkpoint.pth"))
    first.write_bytes(b"tampered")

    with pytest.raises(ValueError, match="size mismatch|hash mismatch"):
        greenland_common.validate_replay_checkpoints(
            project,
            str(manifest.relative_to(project)),
            str(catalog.relative_to(project)),
            str(checkpoint_root.relative_to(project)),
            expected_method_revision=greenland_common.METHOD_REVISION,
        )


def test_replay_checkpoint_validator_rejects_method_revision_drift(tmp_path):
    project, manifest, catalog, checkpoint_root, _rows = (
        _replay_checkpoint_fixture(tmp_path)
    )
    payload = json.loads(catalog.read_text(encoding="utf-8"))
    payload["checkpoints"][next(iter(payload["checkpoints"]))][
        "source_revision"
    ] = "other"
    catalog.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="method revision drifted"):
        greenland_common.validate_replay_checkpoints(
            project,
            str(manifest.relative_to(project)),
            str(catalog.relative_to(project)),
            str(checkpoint_root.relative_to(project)),
            expected_method_revision=greenland_common.METHOD_REVISION,
        )


def test_prepare_greenland_inputs_end_to_end(tmp_path, capsys):
    artifact_root = tmp_path / "efs-root"
    project, manifest, catalog, checkpoint_root, _rows = (
        _replay_checkpoint_fixture(artifact_root / "operations")
    )
    staged_checkpoint_root = artifact_root / "checkpoints" / "replay"
    staged_checkpoint_root.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_root.rename(staged_checkpoint_root)
    checkpoint_root = staged_checkpoint_root
    dataset = artifact_root / "dataset"
    dataset.mkdir()
    (dataset / "values.csv").write_text("time,value\n1,2\n", encoding="utf-8")
    (project / ".gitignore").write_text(
        "staging/\n", encoding="utf-8"
    )
    (project / "docs" / "greenland_approved_image.json").write_text(
        json.dumps(_image_approval()),
        encoding="utf-8",
    )
    subprocess.run(["git", "init", "-q"], cwd=project, check=True)
    subprocess.run(
        ["git", "config", "user.email", "timeraf-test@example.com"],
        cwd=project,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "TimeRAF Test"],
        cwd=project,
        check=True,
    )
    subprocess.run(
        ["git", "add", ".gitignore", "docs"],
        cwd=project,
        check=True,
    )
    subprocess.run(
        ["git", "commit", "-q", "-m", "fixture"],
        cwd=project,
        check=True,
    )
    assert prepare_greenland_inputs.main(
        [
            "--project-root",
            str(project),
            "--artifact-root",
            str(artifact_root),
            "--staging-root",
            str(artifact_root / "staging"),
            "--run-id",
            "replay-fixture",
            "--image-uri",
            IMAGE_URI,
            "--checkpoint-dir",
            str(checkpoint_root.relative_to(artifact_root)),
            "--checkpoint-catalog",
            str(catalog.relative_to(project)),
            "--checkpoint-root",
            str(checkpoint_root.relative_to(artifact_root)),
            "--release-checkpoint-root",
            str(checkpoint_root.relative_to(artifact_root)),
        ]
    ) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["checkpoint_evidence"]["checkpoint_count"] == 208
    for record in payload["archives"].values():
        assert Path(record["path"]).is_file()
        assert record["size_bytes"] > 0


def test_prepare_full_matrix_inputs_excludes_replay_checkpoints(
    tmp_path,
    capsys,
):
    project = tmp_path / "project"
    docs = project / "docs"
    dataset = project / "dataset"
    docs.mkdir(parents=True)
    dataset.mkdir()
    (docs / "timefuse_experiment_manifest.jsonl").write_bytes(
        Path("docs/timefuse_experiment_manifest.jsonl").read_bytes()
    )
    (docs / "greenland_approved_image.json").write_text(
        json.dumps(_image_approval()),
        encoding="utf-8",
    )
    (dataset / "values.csv").write_text(
        "time,value\n1,2\n",
        encoding="utf-8",
    )
    (project / ".gitignore").write_text("staging/\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=project, check=True)
    subprocess.run(
        ["git", "config", "user.email", "timeraf-test@example.com"],
        cwd=project,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "TimeRAF Test"],
        cwd=project,
        check=True,
    )
    subprocess.run(["git", "add", "."], cwd=project, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "fixture"],
        cwd=project,
        check=True,
    )

    assert prepare_greenland_inputs.main(
        [
            "--project-root",
            str(project),
            "--staging-root",
            str(project / "staging"),
            "--run-id",
            "full-matrix-fixture",
            "--job-kind",
            "full_matrix_replication",
            "--image-uri",
            IMAGE_URI,
        ]
    ) == 0

    payload = json.loads(capsys.readouterr().out)
    assert set(payload["archives"]) == {"source", "dataset"}
    assert payload["checkpoint_evidence"] is None
    assert payload["manifest_evidence"]["cells"] == 585
