import json
from pathlib import Path

import pytest

from scripts import compose_timefuse_numerical_recovery
from scripts import run_timefuse_numerical_recovery
from scripts.greenland_common import (
    NUMERICAL_RECOVERY_CELL_IDS,
    NUMERICAL_RECOVERY_PROTOCOL,
    NUMERICAL_RECOVERY_PROTOCOL_SHA256,
)
from ts_rag.matrix import load_manifest


def _command(tmp_path, profile):
    return run_timefuse_numerical_recovery.build_recovery_command(
        python_executable="/python",
        source_root=tmp_path / "source",
        artifact_root=tmp_path,
        manifest=tmp_path / "source" / "manifest.jsonl",
        output_root=tmp_path / "outputs" / profile,
        checkpoint_root=tmp_path / "checkpoints" / profile,
        profile=profile,
        launcher_revision="a" * 40,
    )


def test_a10g_exact_recovery_uses_all_nine_cells_and_four_gpus(tmp_path):
    command = _command(tmp_path, "exact")

    assert command.count("--cell-id") == 9
    cell_ids = [
        command[index + 1]
        for index, value in enumerate(command)
        if value == "--cell-id"
    ]
    assert cell_ids == list(NUMERICAL_RECOVERY_CELL_IDS)
    assert "--override" not in command
    assert command[command.index("--instance-type") + 1] == "ml.g5.12xlarge"
    assert command[command.index("--reserved-gpus-per-host") + 1] == "4"
    assert command[command.index("--processes-per-host") + 1] == "4"
    assert "--max-failures=0" in command
    assert command[1] == str(
        tmp_path / "source" / "scripts" / "run_benchmark_matrix.py"
    )
    working_root_index = command.index("--working-root")
    assert command[working_root_index + 1] == str(tmp_path)


def test_a10g_fallback_recovery_uses_only_frozen_override(tmp_path):
    command = _command(tmp_path, "fallback-v1")

    assert command.count("--override") == 1
    index = command.index("--override")
    assert command[index + 1] == "learning_rate=0.0001"
    assert command.count("--cell-id") == 9


def test_recovery_worker_is_versioned_and_efs_scoped():
    script = Path("scripts/timefuse_numerical_recovery_worker.sh").read_text(
        encoding="utf-8"
    )

    assert "user-default-efs/workspace/TimeRAF" in script
    assert "TIMERAF_RECOVERY_REVISION" in script
    assert "outputs/numerical_recovery/a10g" in script
    assert "processes-per-host" not in script
    assert 'error_type == \\"NumericalIntegrityError\\"' in script
    assert "resume_args=(--dry-run)" in script
    assert script.index("resume_args=(--dry-run)") < script.index(
        '\\"\\${resume_args[@]}\\"'
    )
    assert "profile_runner_state" in script
    assert "PROFILE_CONTROL_CHANNEL_LOST" in script
    assert "PROFILE_CONTROL_RECONNECT" in script
    assert "TIMERAF_RECOVERY_MAX_CONTROL_RECONNECTS" in script
    assert "run_timefuse_numerical_recovery.py' >/dev/null ||" in script
    assert "\\$1 ~ /^python/ && (" in script
    assert script.index("profile_runner_state()") < script.index(
        "run_profile()"
    )


def test_recovery_pipeline_worker_is_pinned_and_serial():
    script = Path(
        "scripts/timefuse_recovery_pipeline_worker.sh"
    ).read_text(encoding="utf-8")

    assert "TIMERAF_LIFECYCLE_REVISION" in script
    assert "TIMERAF_RECOVERY_PREDECESSOR_PID" in script
    assert "TIMERAF_EXACT_PREPARATION_SHA256" in script
    assert "TIMERAF_FALLBACK_PREPARATION_SHA256" in script
    assert "TIMERAF_DRY_RUN_AUDIT_SHA256" in script
    assert 'kill -0 "$PREDECESSOR_PID"' in script
    assert "timefuse_numerical_recovery_worker.sh" in script
    assert "timefuse_greenland_recovery_worker.sh" in script
    assert "timefuse_recovery_finalization_worker.sh" in script
    assert script.index("timefuse_numerical_recovery_worker.sh") < (
        script.index("timefuse_greenland_recovery_worker.sh")
    )
    assert script.index("timefuse_greenland_recovery_worker.sh") < (
        script.index("timefuse_recovery_finalization_worker.sh")
    )


def test_greenland_recovery_worker_is_gated_idempotent_and_efs_scoped():
    script = Path(
        "scripts/timefuse_greenland_recovery_worker.sh"
    ).read_text(encoding="utf-8")

    assert "user-default-efs/workspace/TimeRAF" in script
    assert "TIMERAF_REPO_ROOT" in script
    assert "TIMERAF_RECOVERY_CONTROLLER_REVISION" in script
    assert "TIMERAF_RECOVERY_PREFLIGHT_ONLY" in script
    assert "a100-primary-not-imported" in script
    assert "a10g-primary-runner-active" in script
    assert "a10g-exact-incomplete" in script
    assert "a10g-fallback-incomplete" in script
    assert "check_greenland_capacity.py" in script
    assert "capacity-pre-submit-$profile.json" in script
    assert "--submit" in script
    assert "--confirm SUBMIT" in script
    assert "SUBMISSION_RECONCILIATION_REQUIRED" in script
    assert "PREPARATION_INVALID" in script
    assert "explicit-numerical-integrity-error-v1" in script
    assert "numerical_integrity_source_evidence" in script
    assert "TIMERAF_EXACT_PREPARATION_SHA256" in script
    assert "TIMERAF_FALLBACK_PREPARATION_SHA256" in script
    assert "TIMERAF_DRY_RUN_AUDIT_SHA256" in script
    assert "DRY_RUN_AUDIT_INVALID" in script
    assert "sha256sum '$REMOTE_PREP/prepare-$profile.json'" in script
    assert "sha256sum '$REMOTE_PREP/scheduler-dry-run-audit.json'" in script
    assert script.count('type == \\"array\\"') == 2
    assert script.count('\\"$profile\\" != \\"exact\\"') == 2
    assert script.count(
        '.error_type == \\"NumericalIntegrityError\\"'
    ) == 2
    assert "__TIMERAF_SUBMISSION_RECEIPT_ABSENT__" in script
    assert "__TIMERAF_SUBMISSION_RECEIPT_VALID__" in script
    assert "SUBMISSION_RECEIPT_INVALID" in script
    assert "__TIMERAF_IMPORT_RECEIPT_ABSENT__" in script
    assert "__TIMERAF_IMPORT_RECEIPT_VALID__" in script
    assert "IMPORT_RECEIPT_INVALID" in script
    assert 'if "${SSH[@]}" "jq -e' not in script
    assert "verify_greenland_startup.py" in script
    assert "fetch_greenland_numerical_recovery.py" in script
    assert (
        "--protocol "
        "'$REMOTE_CONTROLLER_SOURCE/docs/"
        "timefuse_numerical_recovery_protocol.json'"
        in script
    )
    assert (
        '"\\$ROOT/docs/timefuse_numerical_recovery_protocol.json"'
        not in script
    )
    assert "recomputed_matrix_summary.sha256" in script
    assert (
        'import_acceptance.mode == \\"terminal-nonfinite-first-pass\\"'
        in script
    )
    assert "a100-primary-summary-drift" in script
    assert script.count("submit_profile ") == 2
    assert script.count("verify_startup ") == 2
    assert script.count("import_profile ") == 2
    assert script.index("PREREQUISITES_READY") < script.index(
        "submit_profile exact"
    )
    pipeline = Path(
        "scripts/timefuse_recovery_pipeline_worker.sh"
    ).read_text(encoding="utf-8")
    assert 'git -C "$REPO_ROOT" archive "$CONTROLLER_REVISION"' in pipeline
    assert "timefuse_greenland_recovery_worker.sh" in pipeline
    assert "timefuse_recovery_finalization_worker.sh" in pipeline


def _completed_row(cell, artifact, corrected=0.9):
    baseline = {metric: 1.0 for metric in cell["metrics"]}
    return {
        "cell_id": cell["id"],
        "task_family": cell["task_family"],
        "dataset": cell["dataset"],
        "model": cell["model"],
        "pred_len": cell["pred_len"],
        "state": "completed",
        "artifact_dir": artifact,
        "started_unix": 1.0,
        "elapsed_seconds": 1.0,
        "all_test_metrics_improve": corrected < 1.0,
        "method": "identity",
        "test_baseline": baseline,
        "test_corrected": {
            metric: corrected for metric in cell["metrics"]
        },
        "metric_gain_percent": {
            metric: 100.0 * (1.0 - corrected)
            for metric in cell["metrics"]
        },
        "minimum_metric_gain_percent": 100.0 * (1.0 - corrected),
    }


def _summary(rows):
    return {
        "source_revision": "d9be338",
        "cell_states": rows,
    }


def test_composition_selects_profiles_only_from_cross_hardware_finiteness():
    manifest = load_manifest("docs/timefuse_experiment_manifest.jsonl")
    by_id = {cell["id"]: cell for cell in manifest}
    cohort = [by_id[cell_id] for cell_id in NUMERICAL_RECOVERY_CELL_IDS]
    first, second = NUMERICAL_RECOVERY_CELL_IDS[:2]

    primary_rows = {
        hardware: [
            _completed_row(cell, f"/{hardware}/primary/{cell['id']}")
            for cell in manifest
        ]
        for hardware in ("a10g", "a100")
    }
    for hardware, cell_id in (("a10g", first), ("a100", second)):
        index = next(
            index
            for index, row in enumerate(primary_rows[hardware])
            if row["cell_id"] == cell_id
        )
        invalid = dict(primary_rows[hardware][index])
        invalid["state"] = "incomplete"
        invalid["numerical_integrity_error"] = "non-finite"
        for key in (
            "all_test_metrics_improve",
            "method",
            "test_baseline",
            "test_corrected",
            "metric_gain_percent",
            "minimum_metric_gain_percent",
        ):
            invalid.pop(key, None)
        invalid["started_unix"] = 100.0 + index
        primary_rows[hardware][index] = invalid

    exact_rows = {
        hardware: [
            _completed_row(cell, f"/{hardware}/exact/{cell['id']}")
            for cell in cohort
        ]
        for hardware in ("a10g", "a100")
    }
    exact_failed = next(
        row for row in exact_rows["a100"] if row["cell_id"] == second
    )
    exact_failed["state"] = "failed"
    exact_failed["error_type"] = "NumericalIntegrityError"
    fallback_rows = {
        hardware: [
            _completed_row(
                cell,
                f"/{hardware}/fallback/{cell['id']}",
                corrected=1.1,
            )
            for cell in cohort
        ]
        for hardware in ("a10g", "a100")
    }
    protocol = json.loads(
        Path(NUMERICAL_RECOVERY_PROTOCOL).read_text(encoding="utf-8")
    )

    result = compose_timefuse_numerical_recovery.compose_numerical_recovery(
        manifest,
        protocol,
        _summary(primary_rows["a10g"]),
        _summary(primary_rows["a100"]),
        _summary(exact_rows["a10g"]),
        _summary(exact_rows["a100"]),
        _summary(fallback_rows["a10g"]),
        _summary(fallback_rows["a100"]),
    )

    assert result["affected_cell_ids"] == sorted([first, second])
    assert result["selected_profiles"] == {
        first: "exact",
        second: "fallback-v1",
    }
    assert not result["selection_uses_metric_quality"]
    assert result["composed"]["a10g"]["counts"]["completed"] == 585
    assert result["composed"]["a100"]["counts"]["completed"] == 585
    composed_a10g = {
        row["cell_id"]: row
        for row in result["composed"]["a10g"]["cell_states"]
    }
    first_row = composed_a10g[first]
    assert first_row["started_unix"] == (
        next(
            row["started_unix"]
            for row in primary_rows["a10g"]
            if row["cell_id"] == first
        )
    )
    assert first_row["numerical_recovery"]["recovery_started_unix"] == 1.0


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _composition_provenance(tmp_path):
    protocol = json.loads(
        Path(NUMERICAL_RECOVERY_PROTOCOL).read_text(encoding="utf-8")
    )
    revision = "a" * 40
    summaries = {}
    metadata = {}
    receipts = {}
    for hardware in ("a10g", "a100"):
        for profile in ("exact", "fallback-v1"):
            root = tmp_path / hardware / profile
            summary = _write_json(
                root / "recomputed_matrix_summary.json",
                {"source_revision": "d9be338", "cell_states": []},
            )
            summaries[f"{hardware}_{profile}"] = summary
            if hardware == "a10g":
                metadata[profile] = {
                    "schema_version": 1,
                    "cohort_id": protocol["cohort_id"],
                    "profile": profile,
                    "overrides": protocol["profiles"][profile]["overrides"],
                    "method_revision": "d9be338",
                    "recovery_revision": revision,
                    "parent_run_id": "a10g-primary-d9be338",
                    "protocol_sha256": NUMERICAL_RECOVERY_PROTOCOL_SHA256,
                    "cell_ids": protocol["cell_ids"],
                    "cell_ids_sha256": protocol["cell_ids_sha256"],
                    "seed": 2021,
                    "instance_type": "ml.g5.12xlarge",
                    "instance_count": 1,
                    "reserved_gpus_per_host": 4,
                    "processes_per_host": 4,
                    "total_gpus": 4,
                    "world_size": 4,
                    "inactive_reserved_gpus": 0,
                    "output_root": str(root),
                }
            else:
                receipts[profile] = {
                    "schema_version": 1,
                    "run_id": f"recovery-{profile}",
                    "job_kind": "supplemental_numerical_recovery",
                    "profile": profile,
                    "method_revision": "d9be338",
                    "source_revision": revision,
                    "launcher_revision": revision,
                    "recovery_revision": revision,
                    "protocol_sha256": NUMERICAL_RECOVERY_PROTOCOL_SHA256,
                    "cell_ids_sha256": protocol["cell_ids_sha256"],
                    "topology_gate_passed": True,
                    "matrix_counts": {
                        "expected": 9,
                        "completed": 9,
                        "failed": 0,
                        "pending": 0,
                        "running": 0,
                        "incomplete": 0,
                    },
                    "recomputed_matrix_summary": {
                        "path": str(summary),
                        "sha256": (
                            compose_timefuse_numerical_recovery._sha256(
                                summary
                            )
                        ),
                        "size_bytes": summary.stat().st_size,
                    },
                }
    return {
        "a10g_exact_metadata": metadata["exact"],
        "a10g_fallback_metadata": metadata["fallback-v1"],
        "a100_exact_receipt": receipts["exact"],
        "a100_fallback_receipt": receipts["fallback-v1"],
        "a10g_exact_summary": summaries["a10g_exact"],
        "a10g_fallback_summary": summaries["a10g_fallback-v1"],
        "a100_exact_summary": summaries["a100_exact"],
        "a100_fallback_summary": summaries["a100_fallback-v1"],
    }


def test_composition_provenance_accepts_separate_hardware_revisions(tmp_path):
    evidence = _composition_provenance(tmp_path)
    evidence["a100_exact_receipt"]["source_revision"] = "b" * 40
    evidence["a100_exact_receipt"]["launcher_revision"] = "b" * 40
    evidence["a100_exact_receipt"]["recovery_revision"] = "b" * 40
    evidence["a100_fallback_receipt"]["source_revision"] = "b" * 40
    evidence["a100_fallback_receipt"]["launcher_revision"] = "b" * 40
    evidence["a100_fallback_receipt"]["recovery_revision"] = "b" * 40

    result = (
        compose_timefuse_numerical_recovery
        .validate_composition_provenance(**evidence)
    )

    assert result["a10g"]["recovery_revision"] == "a" * 40
    assert result["a100"]["recovery_revision"] == "b" * 40
    assert not result["cross_hardware_recovery_revision_match_required"]


def test_composition_provenance_rejects_swapped_profile(tmp_path):
    evidence = _composition_provenance(tmp_path)
    evidence["a100_exact_receipt"]["profile"] = "fallback-v1"

    with pytest.raises(ValueError, match="A100 exact"):
        (
            compose_timefuse_numerical_recovery
            .validate_composition_provenance(**evidence)
        )


def test_composition_provenance_rejects_summary_hash_drift(tmp_path):
    evidence = _composition_provenance(tmp_path)
    evidence["a100_exact_summary"].write_text("tampered", encoding="utf-8")

    with pytest.raises(ValueError, match="A100 exact"):
        (
            compose_timefuse_numerical_recovery
            .validate_composition_provenance(**evidence)
        )


def test_composition_provenance_rejects_profile_revision_drift(tmp_path):
    evidence = _composition_provenance(tmp_path)
    evidence["a10g_fallback_metadata"]["recovery_revision"] = "b" * 40

    with pytest.raises(ValueError, match="A10G recovery profile revisions"):
        (
            compose_timefuse_numerical_recovery
            .validate_composition_provenance(**evidence)
        )


def test_composition_provenance_rejects_non_integrity_exact_failure(tmp_path):
    evidence = _composition_provenance(tmp_path)
    exact = evidence["a100_exact_summary"]
    exact.write_text(
        json.dumps(
            {
                "source_revision": "d9be338",
                "cell_states": [
                    {
                        "cell_id": NUMERICAL_RECOVERY_CELL_IDS[0],
                        "state": "failed",
                        "error_type": "FileNotFoundError",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    evidence["a100_exact_receipt"]["recomputed_matrix_summary"].update(
        {
            "sha256": compose_timefuse_numerical_recovery._sha256(exact),
            "size_bytes": exact.stat().st_size,
        }
    )

    with pytest.raises(ValueError, match="explicit NumericalIntegrityError"):
        (
            compose_timefuse_numerical_recovery
            .validate_composition_provenance(**evidence)
        )
