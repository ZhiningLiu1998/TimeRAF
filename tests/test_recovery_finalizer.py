import json
import os
import sys
from pathlib import Path

import pytest

from scripts import finalize_timefuse_numerical_recovery as finalizer


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_project_python_preserves_efs_venv_entry_symlink(tmp_path):
    project = tmp_path / "project"
    venv = project / ".venv-gpu312"
    binary = venv / "bin" / "python"
    binary.parent.mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text("home = /opt/conda\n", encoding="utf-8")
    binary.symlink_to(sys.executable)

    invocation = finalizer._project_python(project, binary)

    assert invocation == binary
    assert invocation.resolve() == Path(sys.executable).resolve()
    assert os.access(invocation, os.X_OK)


def test_project_python_rejects_external_interpreter_and_unbound_symlink(
    tmp_path,
):
    project = tmp_path / "project"
    project.mkdir()
    with pytest.raises(ValueError, match="venv-gpu312"):
        finalizer._project_python(project, sys.executable)

    binary = project / ".venv-gpu312" / "bin" / "python"
    binary.parent.mkdir(parents=True)
    binary.symlink_to(sys.executable)
    with pytest.raises(ValueError, match="venv-gpu312"):
        finalizer._project_python(project, binary)


def test_command_plan_binds_composition_and_recovered_checkpoints(tmp_path):
    project = tmp_path / "project"
    source = project / "operations" / "source-revision" / "source"
    output = project / "outputs" / "recovery-final"
    checkpoint_root = project / "checkpoints" / "replay"
    stage = checkpoint_root / "staging" / "controller-revision"
    python = project / ".venv" / "bin" / "python"
    inputs = {
        name: project / "inputs" / f"{name}.json"
        for name in (
            "protocol",
            "a10g_primary",
            "a100_primary",
            "a10g_exact",
            "a100_exact",
            "a10g_fallback",
            "a100_fallback",
            "a10g_exact_metadata",
            "a10g_fallback_metadata",
            "a100_exact_receipt",
            "a100_fallback_receipt",
            "a100_final_status",
        )
    }

    commands, outputs = finalizer.build_command_plan(
        python=python,
        source_root=source,
        project_root=project,
        manifest=source / "docs" / "timefuse_experiment_manifest.jsonl",
        replay_manifest=(source / "docs" / "timefuse_checkpoint_replay_manifest.jsonl"),
        confirmation_protocol=(source / "docs" / "timefuse_confirmation_protocol.json"),
        inputs=inputs,
        output_root=output,
        checkpoint_stage_root=stage,
        checkpoint_path_root=checkpoint_root,
    )

    assert len(commands) == 4
    compose, confirmation, comparison, catalog = commands
    assert compose[1].endswith("scripts/compose_timefuse_numerical_recovery.py")
    assert compose[compose.index("--a100-exact-receipt") + 1] == str(
        inputs["a100_exact_receipt"]
    )
    assert compose[compose.index("--a100-fallback-receipt") + 1] == str(
        inputs["a100_fallback_receipt"]
    )
    assert confirmation[confirmation.index("--matrix-summary") + 1] == str(
        outputs["a10g_composed_summary"]
    )
    assert comparison[comparison.index("--a10g-summary") + 1] == str(
        outputs["a10g_composed_summary"]
    )
    assert comparison[comparison.index("--a100-summary") + 1] == str(
        outputs["a100_composed_summary"]
    )
    assert comparison[comparison.index("--a10g-runtime-summary") + 1] == str(
        inputs["a10g_primary"]
    )
    assert comparison[comparison.index("--a100-runtime-summary") + 1] == str(
        inputs["a100_primary"]
    )
    assert catalog[catalog.index("--recovery-receipt") + 1] == str(
        outputs["composition_receipt"]
    )
    assert catalog[catalog.index("--recovery-protocol") + 1] == str(
        inputs["protocol"]
    )
    assert catalog[catalog.index("--stage-root") + 1] == (
        "checkpoints/replay/staging/controller-revision"
    )
    assert catalog[catalog.index("--checkpoint-path-root") + 1] == (
        "checkpoints/replay"
    )
    assert "--hash" in catalog


def test_validate_outputs_accepts_below_threshold_experimental_outcome(
    tmp_path,
):
    outputs = {
        name: tmp_path / filename
        for name, filename in {
            "a10g_composed_summary": "a10g.json",
            "a100_composed_summary": "a100.json",
            "composition_receipt": "composition.json",
            "confirmation_summary": "confirmation.json",
            "hardware_comparison": "comparison.json",
            "checkpoint_catalog": "catalog.json",
        }.items()
    }
    summary = {
        "source_revision": "d9be338",
        "counts": {
            "expected": 585,
            "completed": 585,
            "failed": 0,
            "pending": 0,
            "running": 0,
            "incomplete": 0,
        },
        "numerical_recovery": {},
        "publication_gate": {"development_gate_passed": False},
    }
    _write_json(outputs["a10g_composed_summary"], summary)
    _write_json(outputs["a100_composed_summary"], summary)
    _write_json(
        outputs["composition_receipt"],
        {
            "method_revision": "d9be338",
            "selection_uses_metric_quality": False,
            "selected_profiles": {"cell": "fallback-v1"},
        },
    )
    _write_json(
        outputs["confirmation_summary"],
        {
            "scope": {"confirmatory": 462},
            "confirmatory_gate": {"confirmatory_gate_passed": False},
        },
    )
    comparison = {
        "schema_version": 1,
        "method_revision": "d9be338",
        "seed": 2021,
        "manifest": {
            "expected_cells": 585,
            "expected_metric_values_per_run": 1326,
        },
        "tolerances": {"absolute": 1e-5, "relative": 1e-4},
        "scope": {
            "paired_cells": 585,
            "runtime_paired_cells": 585,
        },
        "consistency_gates": {
            "complete_585_cell_scope": True,
            "all_passed": False,
        },
        "metric_consistency": {
            "baseline": {"count": 1326},
            "corrected": {"count": 1326},
        },
        "runtime": {
            "comparison_scope": "first_pass_summaries",
            "gates": {
                "complete_585_cell_scope": True,
                "all_cells_have_finite_positive_timing": True,
                "a100_reported_matrix_seconds_present": True,
                "all_passed": True,
            },
            "a10g_4gpu": {"terminal_cells": 585},
            "a100_8gpu": {"terminal_cells": 585},
            "all_paired_cells": {"cell_count": 585},
        },
        "cells": [{} for _ in range(585)],
    }
    _write_json(outputs["hardware_comparison"], comparison)
    _write_json(
        outputs["checkpoint_catalog"],
        {
            "selected_count": 208,
            "checkpoint_count": 208,
            "usable_count": 208,
            "missing_count": 0,
            "missing_selected_count": 0,
            "hashes_included": True,
            "numerical_recovery_receipt": {},
            "numerical_recovery_cohort": {},
            "checkpoints": {
                f"cell-{index}": {
                    "numerical_recovery": (
                        {} if index < 4 else None
                    ),
                    "numerical_recovery_cohort": (
                        {} if index < 6 else None
                    ),
                }
                for index in range(208)
            },
        },
    )

    evidence = finalizer._validate_outputs(outputs)

    assert not evidence["primary_gate_passed"]
    assert not evidence["confirmation_gate_passed"]
    assert not evidence["consistency_gates"]["all_passed"]
    assert evidence["checkpoint_count"] == 208
    assert evidence["recovered_checkpoint_count"] == 4
    assert evidence["recovery_cohort_checkpoint_count"] == 6

    comparison["tolerances"]["relative"] = 1e-3
    _write_json(outputs["hardware_comparison"], comparison)
    with pytest.raises(ValueError, match="Hardware comparison scope drifted"):
        finalizer._validate_outputs(outputs)
    comparison["tolerances"]["relative"] = 1e-4
    _write_json(outputs["hardware_comparison"], comparison)

    markdown = tmp_path / "comparison.md"
    markdown.write_text("comparison", encoding="utf-8")
    expected_artifacts = {
        **outputs,
        "hardware_comparison_markdown": markdown,
    }
    inputs = {"manifest": {"sha256": "a" * 64}}
    receipt = {
        "schema_version": 1,
        "controller_revision": "b" * 40,
        "method_revision": "d9be338",
        "inputs": inputs,
        "artifacts": {
            name: finalizer._file_record(path)
            for name, path in expected_artifacts.items()
        },
        "evidence": evidence,
    }
    receipt_path = _write_json(tmp_path / "receipt.json", receipt)
    assert (
        finalizer._verify_completed_receipt(
            receipt_path,
            controller_revision="b" * 40,
            input_records=inputs,
            expected_artifacts=expected_artifacts,
        )
        == receipt
    )

    receipt["evidence"]["checkpoint_count"] = 207
    _write_json(receipt_path, receipt)
    with pytest.raises(ValueError, match="evidence drifted"):
        finalizer._verify_completed_receipt(
            receipt_path,
            controller_revision="b" * 40,
            input_records=inputs,
            expected_artifacts=expected_artifacts,
        )


def test_completed_receipt_is_idempotent_and_rehashes_artifacts(tmp_path):
    artifact_names = {
        "a10g_composed_summary",
        "a100_composed_summary",
        "composition_receipt",
        "confirmation_summary",
        "hardware_comparison",
        "hardware_comparison_markdown",
        "checkpoint_catalog",
    }
    artifacts = {}
    for name in artifact_names:
        path = tmp_path / name
        path.write_text(name, encoding="utf-8")
        artifacts[name] = finalizer._file_record(path)
    inputs = {"manifest": {"sha256": "a" * 64}}
    receipt = {
        "schema_version": 1,
        "controller_revision": "b" * 40,
        "method_revision": "d9be338",
        "inputs": inputs,
        "artifacts": artifacts,
    }
    receipt_path = _write_json(tmp_path / "receipt.json", receipt)

    assert (
        finalizer._verify_completed_receipt(
            receipt_path,
            controller_revision="b" * 40,
            input_records=inputs,
        )
        == receipt
    )

    Path(artifacts["checkpoint_catalog"]["path"]).write_text(
        "drifted",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="artifact drifted"):
        finalizer._verify_completed_receipt(
            receipt_path,
            controller_revision="b" * 40,
            input_records=inputs,
        )


def test_project_paths_fail_closed(tmp_path):
    project = tmp_path / "project"
    project.mkdir()

    with pytest.raises(ValueError, match="below project-root"):
        finalizer._under(project, tmp_path / "outside", "outside")


def test_finalization_worker_stops_at_reviewable_catalog():
    worker = Path(
        "scripts/timefuse_recovery_finalization_worker.sh"
    ).read_text(encoding="utf-8")

    assert "user-default-efs/workspace/TimeRAF" in worker
    assert "TIMERAF_FINALIZER_CONTROLLER_REVISION" in worker
    assert "TIMERAF_FINALIZER_PREFLIGHT_ONLY" in worker
    assert "a10g_primary_authoritative.json" in worker
    assert "--require-publication-scope" in worker
    assert "finalize_timefuse_numerical_recovery.py" in worker
    assert "--recovery-receipt" not in worker
    assert "--checkpoint-stage-root" in worker
    assert 'checkpoints/replay"' in worker
    assert 'staging/$CONTROLLER_REVISION' in worker
    assert "--checkpoint-path-root" in worker
    assert (
        'import_acceptance.mode == \\"terminal-nonfinite-first-pass\\"'
        in worker
    )
    assert "a100-primary-summary-drift" in worker
    assert "CATALOG_READY" in worker
    assert "checkpoint_count == 208" in worker
    assert "--submit" not in worker
    assert "submit_greenland_job.py" not in worker
