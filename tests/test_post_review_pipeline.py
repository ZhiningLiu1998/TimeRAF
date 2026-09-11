import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import timefuse_post_review_pipeline as pipeline


REVISION = "a" * 40
REPLAY_IDS = [f"cell-{index}" for index in range(208)]
RECOVERY_COHORT_IDS = [
    *REPLAY_IDS[:6],
    "recovery-only-0",
    "recovery-only-1",
    "recovery-only-2",
]
AFFECTED_IDS = [
    *REPLAY_IDS[:4],
    "recovery-only-0",
    "recovery-only-1",
    "recovery-only-2",
]


def _catalog(protocol_sha256="d" * 64, protocol_size=123):
    selected_profiles = {cell_id: "exact" for cell_id in AFFECTED_IDS}
    cohort_profiles = {
        cell_id: selected_profiles.get(cell_id, "first-pass")
        for cell_id in RECOVERY_COHORT_IDS
    }
    return {
        "schema_version": 1,
        "source_revision": "d9be338",
        "selected_count": 208,
        "checkpoint_count": 208,
        "usable_count": 208,
        "missing_count": 0,
        "missing_selected_count": 0,
        "hashes_included": True,
        "numerical_recovery": {
            "cohort_id": "test-recovery",
            "affected_cell_ids": AFFECTED_IDS,
            "selected_profiles": selected_profiles,
            "selection_uses_metric_quality": False,
        },
        "numerical_recovery_cohort": {
            "cohort_id": "test-recovery",
            "cell_ids": RECOVERY_COHORT_IDS,
            "profiles": cohort_profiles,
            "selection_uses_metric_quality": False,
            "protocol": {
                "path": "docs/timefuse_numerical_recovery_protocol.json",
                "size_bytes": protocol_size,
                "sha256": protocol_sha256,
            },
            "composition_receipt_sha256": "c" * 64,
        },
        "numerical_recovery_receipt": {
            "path": "outputs/recovery/receipt.json",
            "size_bytes": 123,
            "sha256": "c" * 64,
        },
        "checkpoints": {
            cell_id: {
                "usable": True,
                "relative_path": f"{index}/checkpoint.pth",
                "file_size": 100 + index,
                "sha256": f"{index:064x}",
                "source_revision": "d9be338",
                "numerical_recovery": (
                    {
                        "profile": "exact",
                        "selection_uses_metric_quality": False,
                    }
                    if cell_id in AFFECTED_IDS
                    else None
                ),
                "numerical_recovery_cohort": (
                    {
                        "cohort_id": "test-recovery",
                        "profile": cohort_profiles[cell_id],
                        "affected": cell_id in AFFECTED_IDS,
                        "selection_uses_metric_quality": False,
                        "composition_receipt_sha256": "c" * 64,
                    }
                    if cell_id in RECOVERY_COHORT_IDS
                    else None
                ),
            }
            for index, cell_id in enumerate(REPLAY_IDS)
        },
    }


def _write_frozen_inputs(source):
    manifest = source / pipeline.DEFAULT_REPLAY_MANIFEST
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        "".join(json.dumps({"id": cell_id}) + "\n" for cell_id in REPLAY_IDS),
        encoding="utf-8",
    )
    protocol = source / pipeline.DEFAULT_RECOVERY_PROTOCOL
    protocol.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "method_revision": pipeline.METHOD_REVISION,
                "cohort_id": "test-recovery",
                "cell_ids": RECOVERY_COHORT_IDS,
            }
        ),
        encoding="utf-8",
    )
    return hashlib.sha256(protocol.read_bytes()).hexdigest()


def test_frozen_replay_recovery_intersection_is_six():
    replay_ids = {
        row["id"]
        for row in pipeline.load_manifest(pipeline.DEFAULT_REPLAY_MANIFEST)
    }
    protocol = json.loads(
        Path(pipeline.DEFAULT_RECOVERY_PROTOCOL).read_text(
            encoding="utf-8",
        )
    )

    assert len(replay_ids) == pipeline.EXPECTED_REPLAY_CHECKPOINTS
    assert (
        len(replay_ids & set(protocol["cell_ids"]))
        == pipeline.EXPECTED_RECOVERY_COHORT_REPLAY_CHECKPOINTS
    )


def test_review_gate_requires_committed_bytes_and_explicit_hash(tmp_path):
    source = tmp_path / "source"
    protocol_sha256 = _write_frozen_inputs(source)
    catalog_path = (
        source / "docs" / "timefuse_matrix_checkpoint_catalog.json"
    )
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            _catalog(
                protocol_sha256,
                (
                    source / pipeline.DEFAULT_RECOVERY_PROTOCOL
                ).stat().st_size,
            ),
            sort_keys=True,
        )
        + "\n"
    ).encode()
    catalog_path.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()

    evidence = pipeline.verify_reviewed_catalog(
        source,
        REVISION,
        "docs/timefuse_matrix_checkpoint_catalog.json",
        digest,
        git_file_reader=lambda *_args: payload,
    )

    assert evidence["sha256"] == digest
    assert evidence["checkpoint_count"] == 208
    assert evidence["recovered_checkpoint_count"] == 4
    assert evidence["recovery_cohort_checkpoint_count"] == 6
    assert evidence["first_pass_recovery_cohort_checkpoint_count"] == 2
    assert evidence["numerical_recovery_receipt"] == {
        "path": "outputs/recovery/receipt.json",
        "size_bytes": 123,
        "sha256": "c" * 64,
    }
    assert evidence["committed_in_revision"] == REVISION

    with pytest.raises(ValueError, match="declared commit"):
        pipeline.verify_reviewed_catalog(
            source,
            REVISION,
            "docs/timefuse_matrix_checkpoint_catalog.json",
            digest,
            git_file_reader=lambda *_args: b"drifted",
        )
    with pytest.raises(ValueError, match="reviewed SHA-256"):
        pipeline.verify_reviewed_catalog(
            source,
            REVISION,
            "docs/timefuse_matrix_checkpoint_catalog.json",
            "0" * 64,
            git_file_reader=lambda *_args: payload,
        )


def test_review_gate_rejects_partial_or_unfrozen_catalog(tmp_path):
    source = tmp_path / "source"
    _write_frozen_inputs(source)
    path = source / "docs" / "catalog.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    catalog = _catalog()
    catalog["usable_count"] = 207
    payload = json.dumps(catalog).encode()
    path.write_bytes(payload)

    with pytest.raises(ValueError, match="complete frozen replay catalog"):
        pipeline.verify_reviewed_catalog(
            source,
            REVISION,
            "docs/catalog.json",
            hashlib.sha256(payload).hexdigest(),
            git_file_reader=lambda *_args: payload,
        )


def test_review_gate_rejects_stripped_recovery_provenance(tmp_path):
    source = tmp_path / "source"
    _write_frozen_inputs(source)
    path = source / "docs" / "catalog.json"
    catalog = _catalog()
    catalog["numerical_recovery"] = None
    catalog["numerical_recovery_receipt"] = None
    for record in catalog["checkpoints"].values():
        record["numerical_recovery"] = None
    payload = json.dumps(catalog).encode()
    path.write_bytes(payload)

    with pytest.raises(ValueError, match="numerical-recovery provenance"):
        pipeline.verify_reviewed_catalog(
            source,
            REVISION,
            "docs/catalog.json",
            hashlib.sha256(payload).hexdigest(),
            git_file_reader=lambda *_args: payload,
        )


def test_review_gate_rejects_replay_manifest_scope_drift(tmp_path):
    source = tmp_path / "source"
    _write_frozen_inputs(source)
    manifest = source / pipeline.DEFAULT_REPLAY_MANIFEST
    manifest.write_text(
        manifest.read_text(encoding="utf-8").replace(
            '"cell-207"',
            '"unexpected-cell"',
        ),
        encoding="utf-8",
    )
    path = source / "docs" / "catalog.json"
    payload = json.dumps(_catalog()).encode()
    path.write_bytes(payload)

    with pytest.raises(ValueError, match="frozen replay scope"):
        pipeline.verify_reviewed_catalog(
            source,
            REVISION,
            "docs/catalog.json",
            hashlib.sha256(payload).hexdigest(),
            git_file_reader=lambda *_args: payload,
        )


def test_control_environment_requires_pinned_launcher_and_greenland(tmp_path):
    python = tmp_path / "python"
    python.write_text("#!/bin/sh\n")
    python.chmod(0o755)

    def run_ok(*_args, **_kwargs):
        return SimpleNamespace(
            stdout=json.dumps(
                {
                    "launcher": "1.0.47",
                    "torchx": "2026.7.30",
                    "schedulers": ["greenland", "local_cwd"],
                }
            )
        )

    evidence = pipeline.inspect_control_environment(python, runner=run_ok)
    assert evidence["python"] == str(python)
    assert evidence["launcher"] == "1.0.47"

    def run_drifted(*_args, **_kwargs):
        return SimpleNamespace(
            stdout=json.dumps(
                {
                    "launcher": "1.0.46",
                    "torchx": "2026.7.30",
                    "schedulers": ["greenland"],
                }
            )
        )

    with pytest.raises(ValueError, match="version drifted"):
        pipeline.inspect_control_environment(
            python,
            runner=run_drifted,
        )


def test_control_python_preserves_efs_venv_entry_symlink(tmp_path):
    environment = tmp_path / ".venv-greenland-control"
    python = environment / "bin" / "python"
    python.parent.mkdir(parents=True)
    (environment / "pyvenv.cfg").write_text(
        "home = /opt/conda\n",
        encoding="utf-8",
    )
    python.symlink_to(sys.executable)

    invocation = pipeline.control_python_entry(tmp_path, python)

    assert invocation == python
    assert invocation.resolve() == Path(sys.executable).resolve()
    with pytest.raises(ValueError, match="greenland-control"):
        pipeline.control_python_entry(tmp_path, sys.executable)


def _args(tmp_path):
    research_environment = tmp_path / ".venv-gpu312"
    research_python = research_environment / "bin" / "python"
    research_python.parent.mkdir(parents=True, exist_ok=True)
    (research_environment / "pyvenv.cfg").write_text(
        "include-system-site-packages = false\n",
        encoding="utf-8",
    )
    research_python.write_text("#!/bin/sh\n", encoding="utf-8")
    research_python.chmod(0o755)
    return SimpleNamespace(
        artifact_root=str(tmp_path),
        operations_root=None,
        git_bundle=str(tmp_path / "source.bundle"),
        bundle_ref="HEAD",
        source_revision=REVISION,
        checkpoint_catalog="docs/catalog.json",
        reviewed_catalog_sha256="b" * 64,
        run_id="reviewed-run",
        python=str(research_python),
        appendix_python=str(
            tmp_path / ".venv-appendix140" / "bin" / "python"
        ),
        appendix_environment_receipt=str(
            tmp_path / "operations" / "appendix-environment" / "receipt.json"
        ),
        appendix_controller_root=None,
        appendix_launcher_revision=None,
        control_python=None,
        required_filesystem="nfs4",
        confirm="",
        primary_summary=None,
        confirmation_summary=None,
    )


def _state(args, stages):
    return {
        "schema_version": 1,
        "run_id": args.run_id,
        "source_revision": args.source_revision,
        "reviewed_catalog_sha256": args.reviewed_catalog_sha256,
        "method_revision": pipeline.METHOD_REVISION,
        "research_python": args.python,
        "stages": {stage: {} for stage in stages},
    }


def test_research_python_must_use_efs_artifact_environment(tmp_path):
    args = _args(tmp_path)

    assert pipeline.research_python_entry(
        tmp_path,
        args.python,
    ) == Path(args.python)

    outside = tmp_path / "other" / "bin" / "python"
    outside.parent.mkdir(parents=True)
    outside.write_text("#!/bin/sh\n", encoding="utf-8")
    outside.chmod(0o755)
    with pytest.raises(ValueError, match="venv-gpu312"):
        pipeline.research_python_entry(tmp_path, outside)


def test_appendix_controller_is_bound_to_separate_commit(
    tmp_path,
    monkeypatch,
):
    args = _args(tmp_path)
    controller_root = tmp_path / "operations" / "appendix-controller"
    controller_files = pipeline.APPENDIX_CONTROLLER_FILES
    for relative_path in controller_files:
        path = controller_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative_path, encoding="utf-8")
    args.appendix_controller_root = str(controller_root)
    args.appendix_launcher_revision = "e" * 40
    monkeypatch.setattr(
        pipeline,
        "_git_file_bytes",
        lambda root, _revision, relative_path: (
            Path(root) / relative_path
        ).read_bytes(),
    )

    controller = pipeline._appendix_controller(
        args,
        tmp_path,
        tmp_path / "frozen-source",
    )

    assert controller["root"] == controller_root
    assert controller["revision"] == "e" * 40
    assert set(controller["files"]) == set(controller_files)

    monkeypatch.setattr(
        pipeline,
        "_git_file_bytes",
        lambda _root, _revision, relative_path: relative_path.encode(),
    )
    (controller_root / controller_files[0]).write_text(
        "drifted",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="declared commit"):
        pipeline._appendix_controller(
            args,
            tmp_path,
            tmp_path / "frozen-source",
        )


def test_stage_rejects_research_python_path_drift(tmp_path):
    args = _args(tmp_path)
    state = _state(args, ["prepare"])
    state["research_python"] = str(tmp_path / "drifted" / "python")

    with pytest.raises(ValueError, match="path drifted"):
        pipeline._state_research_python(tmp_path, args, state)


def test_prepare_refuses_to_overwrite_existing_pipeline_state(
    tmp_path,
    monkeypatch,
):
    args = _args(tmp_path)
    _artifact, operations, _workspace, _source = pipeline._source_layout(args)
    operations.mkdir(parents=True)
    state = _state(args, ["prepare", "dry-run", "submit"])
    (operations / "pipeline_state.json").write_text(json.dumps(state))
    monkeypatch.setattr(
        pipeline,
        "materialize_source",
        lambda *_args, **_kwargs: pytest.fail(
            "existing state must be checked before materialization"
        ),
    )

    with pytest.raises(ValueError, match="already prepared"):
        pipeline._prepare(args)


def test_submit_requires_completed_scheduler_dry_run(tmp_path, monkeypatch):
    args = _args(tmp_path)
    monkeypatch.setattr(
        pipeline,
        "_load_state",
        lambda _path: _state(args, ["prepare"]),
    )

    with pytest.raises(ValueError, match="dry-run stage has not completed"):
        pipeline._submission(args, submit=True)


def test_appendix_commands_use_canonical_artifact_project_root(
    tmp_path,
    monkeypatch,
):
    args = _args(tmp_path)
    state = _state(args, ["prepare", "dry-run", "submit", "import"])
    replay_catalog = tmp_path / "outputs" / "replay_catalog.json"
    state["stages"]["import"]["details"] = {
        "replay_catalog": str(replay_catalog),
    }
    monkeypatch.setattr(pipeline, "_load_state", lambda _path: state)
    appendix_environment = {
        "python_entry": args.appendix_python,
        "autogluon_timeseries": "1.4.0",
        "torch": "2.7.1",
        "cuda_available": True,
        "cuda_device_count": 4,
    }
    monkeypatch.setattr(
        pipeline,
        "appendix_python_entry",
        lambda *_args: args.appendix_python,
    )
    monkeypatch.setattr(
        pipeline,
        "inspect_appendix_environment",
        lambda *_args, **_kwargs: appendix_environment,
    )
    monkeypatch.setattr(
        pipeline,
        "managed_path",
        lambda _root, value, *_args, **_kwargs: value,
    )
    monkeypatch.setattr(
        pipeline,
        "verify_appendix_environment_receipt",
        lambda *_args, **_kwargs: {
            "source_revision": args.source_revision,
            "environment": appendix_environment,
        },
    )
    monkeypatch.setattr(
        pipeline,
        "_sha256_file",
        lambda _path: "receipt-sha256",
    )
    monkeypatch.setattr(
        pipeline,
        "_validate_appendix_stage_outputs",
        lambda *_args, **_kwargs: {
            "reported_cells": 96,
            "evaluable_cells": 94,
        },
    )
    commands = []
    monkeypatch.setattr(
        pipeline,
        "_run_streaming",
        lambda command, _cwd: commands.append(command),
    )
    monkeypatch.setattr(
        pipeline,
        "_record_stage",
        lambda *_args, **_kwargs: None,
    )

    pipeline._appendix(args)

    assert len(commands) == 2
    assert commands[0][0] == args.appendix_python
    assert commands[1][0] == args.python
    artifact_root = str(tmp_path.resolve())
    for command in commands:
        assert command.count("--project-root") == 1
        index = command.index("--project-root")
        assert command[index + 1] == artifact_root
    expected_data_roots = {
        "long_term="
        + str(
            tmp_path
            / "dataset"
            / "timefuse"
            / "long_term_forecast"
        ),
        "pems="
        + str(
            tmp_path
            / "dataset"
            / "timefuse"
            / "short_term_forecast"
            / "PEMS"
        ),
        "epf="
        + str(
            tmp_path
            / "dataset"
            / "timefuse"
            / "short_term_forecast"
            / "EPF"
        ),
    }
    data_root_values = {
        commands[0][index + 1]
        for index, value in enumerate(commands[0][:-1])
        if value == "--data-root"
    }
    assert data_root_values == expected_data_roots
    batch_windows_index = commands[0].index("--batch-windows")
    assert (
        commands[0][batch_windows_index + 1]
        == str(pipeline.APPENDIX_MAX_BATCH_WINDOWS)
    )
    max_items_index = commands[0].index("--max-prediction-items")
    assert (
        commands[0][max_items_index + 1]
        == str(pipeline.APPENDIX_MAX_PREDICTION_ITEMS)
    )
    inference_batch_index = commands[0].index("--inference-batch-size")
    assert (
        commands[0][inference_batch_index + 1]
        == str(pipeline.APPENDIX_INFERENCE_BATCH_SIZE)
    )


def test_appendix_refuses_environment_drift_before_launch(
    tmp_path,
    monkeypatch,
):
    args = _args(tmp_path)
    state = _state(args, ["prepare", "dry-run", "submit", "import"])
    state["stages"]["import"]["details"] = {
        "replay_catalog": str(tmp_path / "replay.json"),
    }
    monkeypatch.setattr(pipeline, "_load_state", lambda _path: state)
    monkeypatch.setattr(
        pipeline,
        "appendix_python_entry",
        lambda *_args: args.appendix_python,
    )
    monkeypatch.setattr(
        pipeline,
        "inspect_appendix_environment",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ValueError("Appendix environment version or import identity drifted")
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "_run_streaming",
        lambda *_args, **_kwargs: pytest.fail("no command may launch"),
    )

    with pytest.raises(ValueError, match="identity drifted"):
        pipeline._appendix(args)


def _write_appendix_stage_outputs(tmp_path, source_revision):
    catalog = tmp_path / "appendix_bundle_catalog.json"
    summary = tmp_path / "appendix_summary.json"
    catalog.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "source_revision": source_revision,
                "hashes_verified": True,
                "counts": {
                    "reported": 96,
                    "evaluable": 94,
                    "cataloged": 94,
                    "missing_evaluable": 0,
                },
                "bundles": {
                    f"cell-{index}": {} for index in range(94)
                },
            }
        ),
        encoding="utf-8",
    )
    states = [
        {
            "cell_id": f"cell-{index}",
            "state": "completed" if index < 94 else "paper_oot",
        }
        for index in range(96)
    ]
    summary.write_text(
        json.dumps(
            {
                "source_revision": source_revision,
                "method_revision": pipeline.APPENDIX_METHOD_REVISION,
                "launcher_revision": REVISION,
                "selector_policy": pipeline.APPENDIX_SELECTOR_POLICY,
                "selector_protocol_sha256": (
                    pipeline.APPENDIX_SELECTOR_PROTOCOL_SHA256
                ),
                "execution_topology": pipeline.APPENDIX_EXECUTION_TOPOLOGY,
                "counts": {
                    "reported": 96,
                    "evaluable": 94,
                    "completed": 94,
                    "paper_oot": 2,
                    "failed": 0,
                    "running": 0,
                    "pending": 0,
                    "incomplete": 0,
                },
                "all_evaluable_completed": True,
                "publication_gate": {
                    "development_gate_passed": False,
                },
                "cell_states": states,
            }
        ),
        encoding="utf-8",
    )
    return catalog, summary


def test_appendix_stage_accepts_complete_below_threshold_outcome(tmp_path):
    catalog, summary = _write_appendix_stage_outputs(tmp_path, REVISION)

    evidence = pipeline._validate_appendix_stage_outputs(
        catalog,
        summary,
        REVISION,
        pipeline.APPENDIX_METHOD_REVISION,
        REVISION,
    )

    assert evidence["reported_cells"] == 96
    assert evidence["evaluable_cells"] == 94
    assert evidence["completed_evaluable_cells"] == 94
    assert evidence["paper_oot_cells"] == 2
    assert not evidence["publication_gate_passed"]


@pytest.mark.parametrize(
    ("count_name", "count_value"),
    [
        ("completed", 93),
        ("pending", 1),
        ("failed", 1),
    ],
)
def test_appendix_stage_rejects_incomplete_scope(
    tmp_path,
    count_name,
    count_value,
):
    catalog, summary = _write_appendix_stage_outputs(tmp_path, REVISION)
    payload = json.loads(summary.read_text(encoding="utf-8"))
    payload["counts"][count_name] = count_value
    summary.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="structurally incomplete"):
        pipeline._validate_appendix_stage_outputs(
            catalog,
            summary,
            REVISION,
            pipeline.APPENDIX_METHOD_REVISION,
            REVISION,
        )


def test_audit_passes_catalog_bound_recovery_receipt(tmp_path, monkeypatch):
    args = _args(tmp_path)
    args.primary_summary = "outputs/finalized/a10g.json"
    args.confirmation_summary = "outputs/finalized/confirmation.json"
    receipt = tmp_path / "outputs" / "finalized" / "composition.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_text('{"verified":true}\n', encoding="utf-8")
    state = _state(
        args,
        ["prepare", "dry-run", "submit", "import", "appendix"],
    )
    state["reviewed_catalog"] = {
        "numerical_recovery_receipt": {
            "path": str(receipt.relative_to(tmp_path)),
            "size_bytes": receipt.stat().st_size,
            "sha256": pipeline._sha256_file(receipt),
        }
    }
    state["stages"]["import"]["details"] = {
        "receipt": str(tmp_path / "outputs" / "replay" / "receipt.json"),
    }
    state["stages"]["appendix"]["details"] = {
        "bundle_catalog": str(tmp_path / "outputs" / "appendix" / "catalog.json"),
        "appendix_summary": str(tmp_path / "outputs" / "appendix" / "summary.json"),
        "appendix_launcher_revision": REVISION,
    }
    monkeypatch.setattr(pipeline, "_load_state", lambda _path: state)
    commands = []

    def run_audit(command, _cwd):
        commands.append(command)
        output = Path(command[command.index("--output") + 1])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "publication_ready": True,
                }
            ),
            encoding="utf-8",
        )
        return 0

    monkeypatch.setattr(pipeline, "_run_publication_audit", run_audit)
    monkeypatch.setattr(
        pipeline,
        "_record_stage",
        lambda *_args, **_kwargs: None,
    )

    pipeline._audit(args)

    command = commands[0]
    assert command[command.index("--recovery-receipt") + 1] == str(receipt)
    assert command[command.index("--recovery-protocol") + 1].endswith(
        pipeline.DEFAULT_RECOVERY_PROTOCOL
    )
    assert (
        command[command.index("--appendix-method-revision") + 1]
        == pipeline.APPENDIX_METHOD_REVISION
    )
    assert (
        command[command.index("--appendix-launcher-revision") + 1]
        == REVISION
    )


def test_audit_rejects_catalog_bound_recovery_receipt_drift(
    tmp_path,
    monkeypatch,
):
    args = _args(tmp_path)
    args.primary_summary = "outputs/finalized/a10g.json"
    args.confirmation_summary = "outputs/finalized/confirmation.json"
    receipt = tmp_path / "outputs" / "finalized" / "composition.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_text('{"verified":true}\n', encoding="utf-8")
    state = _state(
        args,
        ["prepare", "dry-run", "submit", "import", "appendix"],
    )
    state["reviewed_catalog"] = {
        "numerical_recovery_receipt": {
            "path": str(receipt.relative_to(tmp_path)),
            "size_bytes": receipt.stat().st_size,
            "sha256": "0" * 64,
        }
    }
    state["stages"]["import"]["details"] = {}
    state["stages"]["appendix"]["details"] = {}
    monkeypatch.setattr(pipeline, "_load_state", lambda _path: state)
    monkeypatch.setattr(
        pipeline,
        "_run_publication_audit",
        lambda *_args, **_kwargs: pytest.fail("audit must not launch"),
    )

    with pytest.raises(ValueError, match="drifted after catalog review"):
        pipeline._audit(args)


def test_audit_records_valid_below_threshold_outcome(tmp_path, monkeypatch):
    args = _args(tmp_path)
    args.primary_summary = "outputs/finalized/a10g.json"
    args.confirmation_summary = "outputs/finalized/confirmation.json"
    receipt = tmp_path / "outputs" / "finalized" / "composition.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_text('{"verified":true}\n', encoding="utf-8")
    state = _state(
        args,
        ["prepare", "dry-run", "submit", "import", "appendix"],
    )
    state["reviewed_catalog"] = {
        "numerical_recovery_receipt": {
            "path": str(receipt.relative_to(tmp_path)),
            "size_bytes": receipt.stat().st_size,
            "sha256": pipeline._sha256_file(receipt),
        }
    }
    state["stages"]["import"]["details"] = {
        "receipt": str(tmp_path / "outputs" / "replay" / "receipt.json"),
    }
    state["stages"]["appendix"]["details"] = {
        "bundle_catalog": str(tmp_path / "outputs" / "appendix" / "catalog.json"),
        "appendix_summary": str(tmp_path / "outputs" / "appendix" / "summary.json"),
        "appendix_launcher_revision": REVISION,
    }
    monkeypatch.setattr(pipeline, "_load_state", lambda _path: state)
    recorded = {}

    def run_audit(command, _cwd):
        output = Path(command[command.index("--output") + 1])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "publication_ready": False,
                    "criteria": {"appendix_94_gate_passed": False},
                }
            ),
            encoding="utf-8",
        )
        return 1

    def record_stage(_state, _path, stage, _command, details):
        recorded["stage"] = stage
        recorded["details"] = details

    monkeypatch.setattr(pipeline, "_run_publication_audit", run_audit)
    monkeypatch.setattr(pipeline, "_record_stage", record_stage)

    pipeline._audit(args)

    assert recorded["stage"] == "audit"
    assert not recorded["details"]["publication_ready"]
    assert recorded["details"]["audit_return_code"] == 1


def test_audit_rejects_structural_auditor_failure(tmp_path, monkeypatch):
    args = _args(tmp_path)
    args.primary_summary = "outputs/finalized/a10g.json"
    args.confirmation_summary = "outputs/finalized/confirmation.json"
    receipt = tmp_path / "outputs" / "finalized" / "composition.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_text('{"verified":true}\n', encoding="utf-8")
    state = _state(
        args,
        ["prepare", "dry-run", "submit", "import", "appendix"],
    )
    state["reviewed_catalog"] = {
        "numerical_recovery_receipt": {
            "path": str(receipt.relative_to(tmp_path)),
            "size_bytes": receipt.stat().st_size,
            "sha256": pipeline._sha256_file(receipt),
        }
    }
    state["stages"]["import"]["details"] = {
        "receipt": str(tmp_path / "outputs" / "replay" / "receipt.json"),
    }
    state["stages"]["appendix"]["details"] = {
        "bundle_catalog": str(tmp_path / "outputs" / "appendix" / "catalog.json"),
        "appendix_summary": str(tmp_path / "outputs" / "appendix" / "summary.json"),
        "appendix_launcher_revision": REVISION,
    }
    monkeypatch.setattr(pipeline, "_load_state", lambda _path: state)
    monkeypatch.setattr(
        pipeline,
        "_run_publication_audit",
        lambda *_args, **_kwargs: 2,
    )
    monkeypatch.setattr(
        pipeline,
        "_record_stage",
        lambda *_args, **_kwargs: pytest.fail("failed audit must not record"),
    )

    with pytest.raises(RuntimeError, match="failed structurally"):
        pipeline._audit(args)
