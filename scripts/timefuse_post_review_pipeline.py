"""Run the review-gated TimeFuse post-primary publication stages."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from scripts.greenland_common import METHOD_REVISION
from scripts.materialize_greenland_source import materialize_source
from scripts.appendix_environment import (
    DEFAULT_RECEIPT as DEFAULT_APPENDIX_ENVIRONMENT_RECEIPT,
    DEFAULT_REQUIREMENTS as DEFAULT_APPENDIX_REQUIREMENTS,
    appendix_python_entry,
    inspect_appendix_environment,
    managed_path,
    verify_appendix_environment_receipt,
)
from ts_rag.appendix_matrix import (
    APPENDIX_EXECUTION_TOPOLOGY,
    APPENDIX_EXPECTED_EVALUABLE,
    APPENDIX_EXPECTED_TOTAL,
    APPENDIX_METHOD_REVISION,
    APPENDIX_RAG_GPU_INACTIVITY_REASON,
    APPENDIX_SELECTOR_POLICY,
    APPENDIX_SELECTOR_PROTOCOL_SHA256,
)
from ts_rag.matrix import load_manifest, save_json_atomic
from ts_rag.matrix_checkpoints import validate_selected_checkpoint_catalog


STAGES = ("prepare", "dry-run", "submit", "import", "appendix", "audit")
DEFAULT_CHECKPOINT_CATALOG = (
    "docs/timefuse_matrix_checkpoint_catalog.json"
)
DEFAULT_REPLAY_MANIFEST = "docs/timefuse_checkpoint_replay_manifest.jsonl"
DEFAULT_RECOVERY_PROTOCOL = "docs/timefuse_numerical_recovery_protocol.json"
EXPECTED_REPLAY_CHECKPOINTS = 208
EXPECTED_RECOVERY_COHORT_REPLAY_CHECKPOINTS = 6
GREENLAND_LAUNCHER_VERSION = "1.0.47"
TORCHX_VERSION = "2026.7.30"
APPENDIX_MAX_BATCH_WINDOWS = 32768
APPENDIX_MAX_PREDICTION_ITEMS = 32768
APPENDIX_INFERENCE_BATCH_SIZE = 256
APPENDIX_CONTROLLER_FILES = (
    "scripts/run_appendix_bundle_pipeline.py",
    "scripts/run_appendix_matrix.py",
    "scripts/run_appendix_rag_cell.py",
    "scripts/audit_timefuse_publication.py",
    "scripts/timefuse_post_review_pipeline.py",
    "scripts/timefuse_remote_stage_probe.py",
    "ts_rag/appendix_matrix.py",
    "ts_rag/benchmark_rag.py",
    "ts_rag/publication.py",
)


def _sha256_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _under(root, value, description, require_exists=False):
    root = Path(root).resolve()
    path = Path(value)
    path = path.resolve() if path.is_absolute() else (root / path).resolve()
    if path == root or root not in path.parents:
        raise ValueError(f"{description} must live below artifact-root")
    if require_exists and not path.exists():
        raise ValueError(f"{description} does not exist: {path}")
    return path


def _safe_repo_path(value):
    if not isinstance(value, str) or "\\" in value:
        raise ValueError("checkpoint-catalog must be a repository-relative path")
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError("checkpoint-catalog must be a repository-relative path")
    return path.as_posix()


def research_python_entry(artifact_root, value):
    artifact_root = Path(artifact_root).resolve()
    expected_environment = artifact_root / ".venv-gpu312"
    python = Path(
        os.path.abspath(
            os.path.expanduser(
                os.fspath(value or expected_environment / "bin" / "python")
            )
        )
    )
    if (
        python.name != "python"
        or python.parent.name != "bin"
        or python.parent.parent != expected_environment
        or not (expected_environment / "pyvenv.cfg").is_file()
        or not python.is_file()
        or not os.access(python, os.X_OK)
    ):
        raise ValueError(
            "research Python must be "
            "<artifact-root>/.venv-gpu312/bin/python"
        )
    return python


def control_python_entry(artifact_root, value):
    artifact_root = Path(artifact_root).resolve()
    expected_environment = artifact_root / ".venv-greenland-control"
    python = Path(
        os.path.abspath(
            os.path.expanduser(
                os.fspath(value or expected_environment / "bin" / "python")
            )
        )
    )
    if (
        python.name != "python"
        or python.parent.name != "bin"
        or python.parent.parent != expected_environment
        or not (expected_environment / "pyvenv.cfg").is_file()
        or not python.is_file()
        or not os.access(python, os.X_OK)
    ):
        raise ValueError(
            "control Python must be "
            "<artifact-root>/.venv-greenland-control/bin/python"
        )
    return python


def _state_research_python(artifact_root, args, state):
    python = research_python_entry(artifact_root, args.python)
    if state.get("research_python") != str(python):
        raise ValueError("Research Python path drifted after prepare")
    return python


def _git_file_bytes(source_root, revision, relative_path):
    return subprocess.run(
        ["git", "show", f"{revision}:{relative_path}"],
        cwd=source_root,
        check=True,
        capture_output=True,
    ).stdout


def verify_reviewed_catalog(
    source_root,
    revision,
    relative_path,
    reviewed_sha256,
    *,
    git_file_reader=_git_file_bytes,
):
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("source-revision must be a full lowercase commit SHA")
    if not re.fullmatch(r"[0-9a-f]{64}", reviewed_sha256):
        raise ValueError("reviewed-catalog-sha256 must be lowercase SHA-256")
    relative_path = _safe_repo_path(relative_path)
    source_root = Path(source_root).resolve()
    catalog_path = (source_root / relative_path).resolve()
    if (
        source_root not in catalog_path.parents
        or not catalog_path.is_file()
        or catalog_path.is_symlink()
    ):
        raise ValueError("Reviewed checkpoint catalog is missing")
    working_bytes = catalog_path.read_bytes()
    committed_bytes = git_file_reader(
        source_root,
        revision,
        relative_path,
    )
    if working_bytes != committed_bytes:
        raise ValueError(
            "Checkpoint catalog bytes do not match the declared commit"
        )
    actual_sha256 = _sha256_bytes(committed_bytes)
    if actual_sha256 != reviewed_sha256:
        raise ValueError(
            "Checkpoint catalog does not match the reviewed SHA-256"
        )
    try:
        catalog = json.loads(committed_bytes)
    except json.JSONDecodeError as error:
        raise ValueError("Reviewed checkpoint catalog is invalid JSON") from error
    if (
        catalog.get("schema_version") != 1
        or catalog.get("source_revision") != METHOD_REVISION
        or catalog.get("selected_count") != EXPECTED_REPLAY_CHECKPOINTS
        or catalog.get("checkpoint_count") != EXPECTED_REPLAY_CHECKPOINTS
        or catalog.get("usable_count") != EXPECTED_REPLAY_CHECKPOINTS
        or catalog.get("missing_count") != 0
        or catalog.get("missing_selected_count") != 0
        or catalog.get("hashes_included") is not True
        or not isinstance(catalog.get("checkpoints"), dict)
        or len(catalog["checkpoints"]) != EXPECTED_REPLAY_CHECKPOINTS
    ):
        raise ValueError(
            "Reviewed checkpoint catalog is not a complete frozen replay catalog"
        )
    replay_manifest = source_root / DEFAULT_REPLAY_MANIFEST
    recovery_protocol = source_root / DEFAULT_RECOVERY_PROTOCOL
    if (
        replay_manifest.is_symlink()
        or not replay_manifest.is_file()
        or recovery_protocol.is_symlink()
        or not recovery_protocol.is_file()
    ):
        raise ValueError("Frozen replay manifest or recovery protocol is missing")
    try:
        replay_ids = [row["id"] for row in load_manifest(replay_manifest)]
        protocol = json.loads(recovery_protocol.read_text(encoding="utf-8"))
        recovery_ids = protocol["cell_ids"]
        recovery_marker = catalog.get("numerical_recovery")
        recovery_cohort = catalog.get("numerical_recovery_cohort")
        cohort_protocol = (
            recovery_cohort.get("protocol")
            if isinstance(recovery_cohort, dict)
            else None
        )
        if (
            len(replay_ids) != EXPECTED_REPLAY_CHECKPOINTS
            or len(set(replay_ids)) != EXPECTED_REPLAY_CHECKPOINTS
            or protocol.get("method_revision") != METHOD_REVISION
            or not isinstance(recovery_ids, list)
            or len(recovery_ids) != len(set(recovery_ids))
            or not isinstance(recovery_marker, dict)
            or recovery_marker.get("cohort_id")
            != protocol.get("cohort_id")
            or not isinstance(recovery_marker.get("affected_cell_ids"), list)
            or not isinstance(recovery_cohort, dict)
            or recovery_cohort.get("cohort_id")
            != protocol.get("cohort_id")
            or recovery_cohort.get("cell_ids") != recovery_ids
            or not isinstance(cohort_protocol, dict)
            or cohort_protocol.get("sha256")
            != _sha256_file(recovery_protocol)
            or cohort_protocol.get("size_bytes")
            != recovery_protocol.stat().st_size
        ):
            raise ValueError
        checkpoint_evidence = validate_selected_checkpoint_catalog(
            catalog,
            replay_ids,
            expected_source_revision=METHOD_REVISION,
            require_hashes=True,
        )
        expected_cohort = len(set(replay_ids) & set(recovery_ids))
        expected_recovered = len(
            set(replay_ids)
            & set(recovery_marker["affected_cell_ids"])
        )
        if (
            expected_cohort != EXPECTED_RECOVERY_COHORT_REPLAY_CHECKPOINTS
            or checkpoint_evidence["recovery_cohort_checkpoint_count"]
            != expected_cohort
            or checkpoint_evidence["recovered_checkpoint_count"]
            != expected_recovered
        ):
            raise ValueError
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(
            "Reviewed checkpoint catalog does not preserve the frozen replay "
            "scope and numerical-recovery provenance"
        ) from error
    return {
        "path": str(catalog_path),
        "relative_path": relative_path,
        "sha256": actual_sha256,
        "source_revision": catalog["source_revision"],
        "checkpoint_count": len(catalog["checkpoints"]),
        "recovered_checkpoint_count": checkpoint_evidence[
            "recovered_checkpoint_count"
        ],
        "recovery_cohort_checkpoint_count": checkpoint_evidence[
            "recovery_cohort_checkpoint_count"
        ],
        "first_pass_recovery_cohort_checkpoint_count": checkpoint_evidence[
            "first_pass_recovery_cohort_checkpoint_count"
        ],
        "replay_manifest_sha256": _sha256_file(replay_manifest),
        "recovery_protocol_sha256": _sha256_file(recovery_protocol),
        "numerical_recovery_receipt": dict(
            catalog["numerical_recovery_receipt"]
        ),
        "committed_in_revision": revision,
    }


def _run_json(command, cwd):
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Command did not return JSON: {' '.join(command)}"
        ) from error


def _run_streaming(command, cwd):
    subprocess.run(command, cwd=cwd, check=True)


def _run_publication_audit(command, cwd):
    return subprocess.run(command, cwd=cwd, check=False).returncode


def _validate_appendix_stage_outputs(
    bundle_catalog_path,
    appendix_summary_path,
    source_revision,
    method_revision=APPENDIX_METHOD_REVISION,
    launcher_revision=None,
):
    paths = {
        "bundle_catalog": Path(bundle_catalog_path),
        "appendix_summary": Path(appendix_summary_path),
    }
    payloads = {}
    for name, path in paths.items():
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Appendix {name} is missing or not a regular file")
        try:
            payloads[name] = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(f"Appendix {name} is invalid JSON") from error
        if not isinstance(payloads[name], dict):
            raise ValueError(f"Appendix {name} must be a JSON object")

    catalog = payloads["bundle_catalog"]
    catalog_counts = catalog.get("counts")
    if (
        catalog.get("schema_version") != 2
        or catalog.get("source_revision") != source_revision
        or catalog.get("hashes_verified") is not True
        or catalog_counts
        != {
            "reported": APPENDIX_EXPECTED_TOTAL,
            "evaluable": APPENDIX_EXPECTED_EVALUABLE,
            "cataloged": APPENDIX_EXPECTED_EVALUABLE,
            "missing_evaluable": 0,
        }
        or not isinstance(catalog.get("bundles"), dict)
        or len(catalog["bundles"]) != APPENDIX_EXPECTED_EVALUABLE
    ):
        raise ValueError("Appendix bundle catalog is structurally incomplete")

    summary = payloads["appendix_summary"]
    counts = summary.get("counts")
    states = summary.get("cell_states")
    if (
        summary.get("source_revision") != source_revision
        or summary.get("method_revision") != method_revision
        or summary.get("launcher_revision") != launcher_revision
        or summary.get("selector_policy") != APPENDIX_SELECTOR_POLICY
        or summary.get("selector_protocol_sha256")
        != APPENDIX_SELECTOR_PROTOCOL_SHA256
        or summary.get("execution_topology") != APPENDIX_EXECUTION_TOPOLOGY
        or not isinstance(counts, dict)
        or counts.get("reported") != APPENDIX_EXPECTED_TOTAL
        or counts.get("evaluable") != APPENDIX_EXPECTED_EVALUABLE
        or counts.get("completed") != APPENDIX_EXPECTED_EVALUABLE
        or counts.get("paper_oot")
        != APPENDIX_EXPECTED_TOTAL - APPENDIX_EXPECTED_EVALUABLE
        or any(
            counts.get(name) != 0
            for name in ("failed", "running", "pending", "incomplete")
        )
        or summary.get("all_evaluable_completed") is not True
        or not isinstance(states, list)
        or len(states) != APPENDIX_EXPECTED_TOTAL
        or len({state.get("cell_id") for state in states})
        != APPENDIX_EXPECTED_TOTAL
        or not isinstance(summary.get("publication_gate"), dict)
    ):
        raise ValueError("Appendix RAG summary is structurally incomplete")
    return {
        "bundle_catalog_sha256": _sha256_file(paths["bundle_catalog"]),
        "appendix_summary_sha256": _sha256_file(paths["appendix_summary"]),
        "reported_cells": counts["reported"],
        "evaluable_cells": counts["evaluable"],
        "completed_evaluable_cells": counts["completed"],
        "paper_oot_cells": counts["paper_oot"],
        "publication_gate_passed": summary["publication_gate"].get(
            "development_gate_passed"
        )
        is True,
    }


def inspect_control_environment(python, *, runner=subprocess.run):
    python = Path(
        os.path.abspath(os.path.expanduser(os.fspath(python)))
    )
    if not python.is_file() or not os.access(python, os.X_OK):
        raise ValueError(f"Greenland control Python is not executable: {python}")
    probe = (
        "import importlib.metadata as m,json;"
        "from torchx.runner import get_runner;"
        "print(json.dumps({"
        "'launcher':m.version('amzn-greenland-torchx-launcher'),"
        "'torchx':m.version('torchx-nightly'),"
        "'schedulers':get_runner().scheduler_backends()"
        "},sort_keys=True))"
    )
    try:
        completed = runner(
            [str(python), "-c", probe],
            check=True,
            capture_output=True,
            text=True,
        )
        evidence = json.loads(completed.stdout)
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError) as error:
        raise ValueError(
            "Greenland control Python failed its import probe"
        ) from error
    if (
        evidence.get("launcher") != GREENLAND_LAUNCHER_VERSION
        or evidence.get("torchx") != TORCHX_VERSION
        or "greenland" not in evidence.get("schedulers", [])
    ):
        raise ValueError("Greenland control environment version drifted")
    return {
        "python": str(python),
        **evidence,
    }


def _load_state(path):
    try:
        state = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            "Post-review state is missing; run the prepare stage first"
        ) from error
    if state.get("schema_version") != 1:
        raise ValueError("Unsupported post-review state schema")
    return state


def _validate_state(state, args):
    if (
        state.get("run_id") != args.run_id
        or state.get("source_revision") != args.source_revision
        or state.get("reviewed_catalog_sha256")
        != args.reviewed_catalog_sha256
        or state.get("method_revision") != METHOD_REVISION
    ):
        raise ValueError("Post-review state identity drifted")


def _record_stage(state, state_path, stage, command, details=None):
    stages = dict(state.get("stages", {}))
    stages[stage] = {
        "completed_unix": time.time(),
        "command": command,
        "details": details or {},
    }
    state["stages"] = stages
    state["updated_unix"] = time.time()
    save_json_atomic(state, state_path)


def _require_stage(state, stage):
    if stage not in state.get("stages", {}):
        raise ValueError(f"The {stage} stage has not completed")


def _source_layout(args):
    artifact_root = Path(args.artifact_root).resolve()
    operations_root = _under(
        artifact_root,
        args.operations_root
        or artifact_root / "operations" / f"post-review-{args.run_id}",
        "operations-root",
    )
    workspace = operations_root / "materialized"
    source_root = workspace / "source"
    return artifact_root, operations_root, workspace, source_root


def _appendix_controller(args, artifact_root, source_root):
    value = args.appendix_controller_root
    if value is None:
        if args.appendix_launcher_revision is not None:
            raise ValueError(
                "--appendix-launcher-revision requires "
                "--appendix-controller-root"
            )
        return {
            "root": source_root,
            "revision": args.source_revision,
            "files": {},
        }
    controller_root = _under(
        artifact_root,
        value,
        "appendix-controller-root",
        require_exists=True,
    )
    revision = args.appendix_launcher_revision
    if not isinstance(revision, str) or not re.fullmatch(
        r"[0-9a-f]{40}",
        revision,
    ):
        raise ValueError(
            "appendix-launcher-revision must be a full lowercase commit SHA"
        )
    files = {}
    for relative_path in APPENDIX_CONTROLLER_FILES:
        path = controller_root / relative_path
        if (
            not path.is_file()
            or path.is_symlink()
            or path.read_bytes()
            != _git_file_bytes(controller_root, revision, relative_path)
        ):
            raise ValueError(
                "Appendix controller files do not match the declared commit"
            )
        files[relative_path] = {
            "path": str(path),
            "sha256": _sha256_file(path),
        }
    return {
        "root": controller_root,
        "revision": revision,
        "files": files,
    }


def _approved_image(source_root):
    path = source_root / "docs" / "greenland_approved_image.json"
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("Committed Greenland image approval is invalid") from error
    image_uri = record.get("image_uri")
    if not isinstance(image_uri, str) or "@sha256:" not in image_uri:
        raise ValueError("Committed Greenland image approval has no digest")
    return image_uri


def _prepare(args):
    artifact_root, operations_root, workspace, source_root = _source_layout(
        args
    )
    operations_root.mkdir(parents=True, exist_ok=True)
    state_path = operations_root / "pipeline_state.json"
    if state_path.exists():
        state = _load_state(state_path)
        _validate_state(state, args)
        raise ValueError(
            "Post-review pipeline is already prepared; resume from its state"
        )
    bundle = _under(
        artifact_root,
        args.git_bundle,
        "git-bundle",
        require_exists=True,
    )
    materialization = materialize_source(
        bundle,
        args.bundle_ref,
        args.source_revision,
        artifact_root,
        workspace,
        required_filesystem=args.required_filesystem,
    )
    catalog = verify_reviewed_catalog(
        source_root,
        args.source_revision,
        args.checkpoint_catalog,
        args.reviewed_catalog_sha256,
    )
    image_uri = _approved_image(source_root)
    research_python = research_python_entry(artifact_root, args.python)
    control_python = control_python_entry(
        artifact_root,
        args.control_python,
    )
    control_environment = inspect_control_environment(control_python)
    staging_root = operations_root / "staging"
    command = [
        str(research_python),
        "scripts/prepare_greenland_inputs.py",
        "--project-root",
        str(source_root),
        "--artifact-root",
        str(artifact_root),
        "--staging-root",
        str(staging_root),
        "--run-id",
        args.run_id,
        "--source-revision",
        args.source_revision,
        "--launcher-revision",
        appendix_launcher_revision,
        "--launcher-revision",
        args.source_revision,
        "--method-revision",
        METHOD_REVISION,
        "--image-uri",
        image_uri,
        "--checkpoint-dir",
        "checkpoints/replay",
        "--checkpoint-catalog",
        args.checkpoint_catalog,
        "--checkpoint-root",
        "checkpoints/replay",
        "--release-checkpoint-root",
        "checkpoints/replay",
        "--upload",
    ]
    prepared = _run_json(command, source_root)
    state = {
        "schema_version": 1,
        "created_unix": time.time(),
        "updated_unix": time.time(),
        "artifact_root": str(artifact_root),
        "operations_root": str(operations_root),
        "source_root": str(source_root),
        "run_id": args.run_id,
        "method_revision": METHOD_REVISION,
        "source_revision": args.source_revision,
        "launcher_revision": args.source_revision,
        "reviewed_catalog_sha256": args.reviewed_catalog_sha256,
        "reviewed_catalog": catalog,
        "image_uri": image_uri,
        "research_python": str(research_python),
        "control_environment": control_environment,
        "materialization": materialization,
        "job_spec": prepared["job_spec"],
        "job_spec_s3_uri": prepared["job_spec_s3_uri"],
        "job_spec_sha256": prepared["job_spec_sha256"],
        "stages": {},
    }
    _record_stage(
        state,
        state_path,
        "prepare",
        command,
        {
            "checkpoint_evidence": prepared["checkpoint_evidence"],
            "inputs": prepared["inputs"],
        },
    )
    return state_path


def _submission(args, submit):
    artifact_root, operations_root, _workspace, source_root = _source_layout(
        args
    )
    state_path = operations_root / "pipeline_state.json"
    state = _load_state(state_path)
    _validate_state(state, args)
    _require_stage(state, "prepare")
    stage = "submit" if submit else "dry-run"
    if submit:
        _require_stage(state, "dry-run")
    if submit and stage in state.get("stages", {}):
        raise ValueError("Greenland job was already submitted")
    control_python = control_python_entry(
        artifact_root,
        args.control_python,
    )
    control_environment = inspect_control_environment(control_python)
    if control_environment != state.get("control_environment"):
        raise ValueError("Greenland control environment drifted after prepare")
    receipt = operations_root / (
        "submission_receipt.json" if submit else "scheduler_dry_run.json"
    )
    command = [
        str(control_python),
        "scripts/submit_greenland_job.py",
        "--job-spec",
        state["job_spec"],
        "--job-spec-s3-uri",
        state["job_spec_s3_uri"],
        "--image-uri",
        state["image_uri"],
        "--receipt",
        str(receipt),
    ]
    if submit:
        if args.confirm != "SUBMIT":
            raise ValueError("A real Greenland launch requires --confirm SUBMIT")
        command.extend(["--submit", "--confirm", "SUBMIT"])
    result = _run_json(command, source_root)
    if bool(result.get("dry_run")) is submit:
        raise ValueError("Greenland submission receipt has the wrong mode")
    _record_stage(state, state_path, stage, command, result)
    return state_path


def _import(args):
    artifact_root, operations_root, _workspace, source_root = _source_layout(
        args
    )
    state_path = operations_root / "pipeline_state.json"
    state = _load_state(state_path)
    _validate_state(state, args)
    _require_stage(state, "submit")
    research_python = _state_research_python(artifact_root, args, state)
    destination = (
        artifact_root / "outputs" / "greenland_runs" / args.run_id
    )
    receipt = destination / "fetch_receipt.json"
    catalog = destination / "replay_bundle_catalog.json"
    command = [
        str(research_python),
        "scripts/fetch_greenland_outputs.py",
        "--run-id",
        args.run_id,
        "--project-root",
        str(artifact_root),
        "--destination",
        str(destination),
        "--replay-manifest",
        str(
            source_root
            / "docs"
            / "timefuse_checkpoint_replay_manifest.jsonl"
        ),
        "--catalog",
        str(catalog),
        "--receipt",
        str(receipt),
    ]
    result = _run_json(command, source_root)
    _record_stage(
        state,
        state_path,
        "import",
        command,
        {
            **result,
            "receipt": str(receipt),
            "replay_catalog": str(catalog),
        },
    )
    return state_path


def _appendix(args):
    artifact_root, operations_root, _workspace, source_root = _source_layout(
        args
    )
    state_path = operations_root / "pipeline_state.json"
    state = _load_state(state_path)
    _validate_state(state, args)
    _require_stage(state, "import")
    controller = _appendix_controller(args, artifact_root, source_root)
    controller_root = Path(controller["root"])
    appendix_launcher_revision = controller["revision"]
    research_python = _state_research_python(artifact_root, args, state)
    imported = state["stages"]["import"]["details"]
    appendix_python = appendix_python_entry(
        artifact_root,
        args.appendix_python
        or artifact_root / ".venv-appendix140" / "bin" / "python",
    )
    appendix_environment = inspect_appendix_environment(
        appendix_python,
        expected_cuda_devices=4,
        require_cuda=True,
    )
    environment_receipt_path = managed_path(
        artifact_root,
        args.appendix_environment_receipt
        or artifact_root / DEFAULT_APPENDIX_ENVIRONMENT_RECEIPT,
        "Appendix environment receipt",
        require_exists=True,
    )
    environment_receipt = verify_appendix_environment_receipt(
        environment_receipt_path,
        artifact_root,
        args.source_revision,
        source_root / DEFAULT_APPENDIX_REQUIREMENTS,
        appendix_environment,
    )
    bundle_root = (
        artifact_root
        / "outputs"
        / "timefuse_appendix_bundles"
        / args.source_revision[:12]
    )
    cache_root = (
        artifact_root
        / ".cache"
        / "appendix-runtime"
        / args.source_revision[:12]
    )
    bundle_catalog = bundle_root / "appendix_bundle_catalog.json"
    bundle_command = [
        str(appendix_python),
        str(
            controller_root
            / "scripts"
            / "run_appendix_bundle_pipeline.py"
        ),
        "--appendix-manifest",
        str(
            source_root
            / "docs"
            / "timefuse_appendix_experiment_manifest.jsonl"
        ),
        "--replay-manifest",
        str(
            source_root
            / "docs"
            / "timefuse_checkpoint_replay_manifest.jsonl"
        ),
        "--replay-catalog",
        imported["replay_catalog"],
        "--output-root",
        str(bundle_root),
        "--bundle-catalog",
        str(bundle_catalog),
        "--project-root",
        str(artifact_root),
        "--cache-root",
        str(cache_root),
        "--source-revision",
        args.source_revision,
        "--launcher-revision",
        appendix_launcher_revision,
        "--data-root",
        "long_term="
        + str(
            artifact_root
            / "dataset"
            / "timefuse"
            / "long_term_forecast"
        ),
        "--data-root",
        "pems="
        + str(
            artifact_root
            / "dataset"
            / "timefuse"
            / "short_term_forecast"
            / "PEMS"
        ),
        "--data-root",
        "epf="
        + str(
            artifact_root
            / "dataset"
            / "timefuse"
            / "short_term_forecast"
            / "EPF"
        ),
        "--instance-type",
        "ml.g5.12xlarge",
        "--reserved-gpus-per-host",
        "4",
        "--processes-per-host",
        "4",
        "--gpu-ids",
        "0,1,2,3",
        "--max-failures",
        "0",
        "--batch-windows",
        str(APPENDIX_MAX_BATCH_WINDOWS),
        "--max-prediction-items",
        str(APPENDIX_MAX_PREDICTION_ITEMS),
        "--inference-batch-size",
        str(APPENDIX_INFERENCE_BATCH_SIZE),
    ]
    _run_streaming(bundle_command, controller_root)

    rag_root = (
        artifact_root
        / "outputs"
        / "timefuse_appendix_rag"
        / APPENDIX_METHOD_REVISION[:12]
    )
    appendix_summary = rag_root / "appendix_summary.json"
    rag_command = [
        str(research_python),
        str(controller_root / "scripts" / "run_appendix_matrix.py"),
        "--manifest",
        str(
            source_root
            / "docs"
            / "timefuse_appendix_experiment_manifest.jsonl"
        ),
        "--bundle-catalog",
        str(bundle_catalog),
        "--project-root",
        str(artifact_root),
        "--output-root",
        str(rag_root),
        "--summary",
        str(appendix_summary),
        "--source-revision",
        args.source_revision,
        "--method-revision",
        APPENDIX_METHOD_REVISION,
        "--workers",
        "4",
        "--instance-type",
        "ml.g5.12xlarge",
        "--reserved-gpus-per-host",
        "4",
        "--inactive-reserved-gpus",
        "4",
        "--gpu-inactivity-reason",
        APPENDIX_RAG_GPU_INACTIVITY_REASON,
        "--max-failures",
        "0",
    ]
    _run_streaming(rag_command, controller_root)
    output_evidence = _validate_appendix_stage_outputs(
        bundle_catalog,
        appendix_summary,
        args.source_revision,
        APPENDIX_METHOD_REVISION,
        appendix_launcher_revision,
    )
    _record_stage(
        state,
        state_path,
        "appendix",
        [bundle_command, rag_command],
        {
            "bundle_catalog": str(bundle_catalog),
            "appendix_summary": str(appendix_summary),
            "appendix_output_evidence": output_evidence,
            "appendix_environment": appendix_environment,
            "appendix_environment_receipt": str(
                environment_receipt_path
            ),
            "appendix_environment_receipt_sha256": _sha256_file(
                environment_receipt_path
            ),
            "appendix_environment_source_revision": environment_receipt[
                "source_revision"
            ],
            "appendix_controller": {
                **controller,
                "root": str(controller_root),
            },
            "appendix_launcher_revision": appendix_launcher_revision,
            "cache_root": str(cache_root),
        },
    )
    return state_path


def _audit(args):
    artifact_root, operations_root, _workspace, source_root = _source_layout(
        args
    )
    state_path = operations_root / "pipeline_state.json"
    state = _load_state(state_path)
    _validate_state(state, args)
    _require_stage(state, "appendix")
    research_python = _state_research_python(artifact_root, args, state)
    if not args.primary_summary or not args.confirmation_summary:
        raise ValueError(
            "audit requires --primary-summary and --confirmation-summary"
        )
    imported = state["stages"]["import"]["details"]
    appendix = state["stages"]["appendix"]["details"]
    controller = _appendix_controller(args, artifact_root, source_root)
    controller_root = Path(controller["root"])
    recovery_record = state.get("reviewed_catalog", {}).get(
        "numerical_recovery_receipt"
    )
    if not isinstance(recovery_record, dict):
        raise ValueError(
            "Reviewed checkpoint catalog has no numerical-recovery receipt"
        )
    recovery_receipt = _under(
        artifact_root,
        recovery_record.get("path", ""),
        "recovery-receipt",
        require_exists=True,
    )
    if (
        not recovery_receipt.is_file()
        or recovery_record.get("size_bytes") != recovery_receipt.stat().st_size
        or recovery_record.get("sha256") != _sha256_file(recovery_receipt)
    ):
        raise ValueError(
            "Numerical-recovery receipt drifted after catalog review"
        )
    output = operations_root / (
        f"publication_audit-{APPENDIX_METHOD_REVISION[:12]}.json"
    )
    command = [
        str(research_python),
        str(controller_root / "scripts" / "audit_timefuse_publication.py"),
        "--project-root",
        str(artifact_root),
        "--primary-manifest",
        str(source_root / "docs" / "timefuse_experiment_manifest.jsonl"),
        "--primary-summary",
        str(_under(artifact_root, args.primary_summary, "primary-summary")),
        "--recovery-receipt",
        str(recovery_receipt),
        "--recovery-protocol",
        str(source_root / DEFAULT_RECOVERY_PROTOCOL),
        "--confirmation-protocol",
        str(source_root / "docs" / "timefuse_confirmation_protocol.json"),
        "--confirmation-summary",
        str(
            _under(
                artifact_root,
                args.confirmation_summary,
                "confirmation-summary",
            )
        ),
        "--replay-manifest",
        str(
            source_root
            / "docs"
            / "timefuse_checkpoint_replay_manifest.jsonl"
        ),
        "--replay-receipt",
        imported["receipt"],
        "--appendix-manifest",
        str(
            source_root
            / "docs"
            / "timefuse_appendix_experiment_manifest.jsonl"
        ),
        "--appendix-catalog",
        appendix["bundle_catalog"],
        "--appendix-summary",
        appendix["appendix_summary"],
        "--method-revision",
        METHOD_REVISION,
        "--appendix-method-revision",
        APPENDIX_METHOD_REVISION,
        "--appendix-launcher-revision",
        appendix["appendix_launcher_revision"],
        "--output",
        str(output),
    ]
    audit_return_code = _run_publication_audit(command, controller_root)
    if audit_return_code not in {0, 1}:
        raise RuntimeError(
            "Final publication audit failed structurally or could not run"
        )
    if not output.is_file() or output.is_symlink():
        raise ValueError("Final publication audit report is missing")
    try:
        report = json.loads(output.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError("Final publication audit report is invalid JSON") from error
    if not isinstance(report, dict):
        raise ValueError("Final publication audit report must be a JSON object")
    publication_ready = report.get("publication_ready")
    if (
        report.get("schema_version") != 1
        or not isinstance(publication_ready, bool)
        or "error" in report
        or audit_return_code != (0 if publication_ready else 1)
    ):
        raise ValueError(
            "Final publication audit report contradicts its exit status"
        )
    _record_stage(
        state,
        state_path,
        "audit",
        command,
        {
            "publication_audit": str(output),
            "publication_audit_sha256": _sha256_file(output),
            "publication_ready": publication_ready,
            "audit_return_code": audit_return_code,
        },
    )
    return state_path


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Resume the reviewed TimeFuse publication pipeline by stage"
        )
    )
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--operations-root")
    parser.add_argument("--git-bundle", required=True)
    parser.add_argument("--bundle-ref", default="HEAD")
    parser.add_argument("--source-revision", required=True)
    parser.add_argument(
        "--checkpoint-catalog",
        default=DEFAULT_CHECKPOINT_CATALOG,
    )
    parser.add_argument("--reviewed-catalog-sha256", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--appendix-python",
        help=(
            "Python containing autogluon.timeseries==1.4.0 and torch==2.7.1; "
            "defaults to <artifact-root>/.venv-appendix140/bin/python"
        ),
    )
    parser.add_argument(
        "--appendix-environment-receipt",
        help=(
            "Receipt from provision_appendix_environment.py; defaults to "
            "<artifact-root>/operations/appendix-environment/receipt.json"
        ),
    )
    parser.add_argument(
        "--appendix-controller-root",
        help=(
            "Optional EFS materialization of a committed infrastructure-only "
            "Appendix controller"
        ),
    )
    parser.add_argument(
        "--appendix-launcher-revision",
        help=(
            "Full commit SHA for --appendix-controller-root; recorded "
            "separately from the frozen source revision"
        ),
    )
    parser.add_argument(
        "--control-python",
        help=(
            "Python containing the pinned Greenland launcher; defaults to "
            "<artifact-root>/.venv-greenland-control/bin/python"
        ),
    )
    parser.add_argument("--required-filesystem", default="nfs4")
    parser.add_argument("--confirm", default="")
    parser.add_argument("--primary-summary")
    parser.add_argument("--confirmation-summary")
    args = parser.parse_args(argv)

    handlers = {
        "prepare": _prepare,
        "dry-run": lambda parsed: _submission(parsed, False),
        "submit": lambda parsed: _submission(parsed, True),
        "import": _import,
        "appendix": _appendix,
        "audit": _audit,
    }
    state_path = handlers[args.stage](args)
    print(
        json.dumps(
            {
                "stage": args.stage,
                "state": str(state_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
