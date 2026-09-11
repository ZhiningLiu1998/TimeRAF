"""Finalize composed TimeFuse recovery evidence without scheduler side effects."""

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
from ts_rag.matrix import save_json_atomic


EXPECTED_CELLS = 585
EXPECTED_METRIC_VALUES = 1326
EXPECTED_REPLAY_CHECKPOINTS = 208
EXPECTED_REPLAY_RECOVERY_COHORT_CHECKPOINTS = 6


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path, label):
    try:
        with Path(path).open("r", encoding="utf-8") as source:
            payload = json.load(source)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is missing or invalid: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def _under(root, value, label, *, require_exists=False):
    root = Path(root).resolve()
    path = Path(value)
    path = path.resolve() if path.is_absolute() else (root / path).resolve()
    if path == root or root not in path.parents:
        raise ValueError(f"{label} must live below project-root")
    if require_exists and not path.is_file():
        raise ValueError(f"{label} does not exist: {path}")
    return path


def _file_record(path):
    path = Path(path).resolve()
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _project_python(project_root, value):
    project_root = Path(project_root).resolve()
    declared = Path(value) if value else project_root / ".venv-gpu312/bin/python"
    if not declared.is_absolute():
        declared = project_root / declared
    expected_venv = (project_root / ".venv-gpu312").resolve()
    if (
        declared.name != "python"
        or declared.parent.name != "bin"
        or declared.parent.parent.resolve() != expected_venv
        or not (expected_venv / "pyvenv.cfg").is_file()
    ):
        raise ValueError(
            "python must be the project .venv-gpu312/bin/python entry point"
        )
    invocation = expected_venv / "bin" / "python"
    if not invocation.is_file() or not os.access(invocation, os.X_OK):
        raise ValueError("python must be an executable file")
    return invocation


def build_command_plan(
    *,
    python,
    source_root,
    project_root,
    manifest,
    replay_manifest,
    confirmation_protocol,
    inputs,
    output_root,
    checkpoint_stage_root,
    checkpoint_path_root=None,
):
    python = Path(os.path.abspath(python))
    source_root = Path(source_root).resolve()
    project_root = Path(project_root).resolve()
    output_root = Path(output_root).resolve()
    checkpoint_stage_root = Path(checkpoint_stage_root).resolve()
    checkpoint_path_root = (
        checkpoint_stage_root
        if checkpoint_path_root is None
        else Path(checkpoint_path_root).resolve()
    )
    scripts = source_root / "scripts"
    outputs = {
        "a10g_composed_summary": output_root / "a10g_composed_summary.json",
        "a100_composed_summary": output_root / "a100_composed_summary.json",
        "composition_receipt": output_root / "composition_receipt.json",
        "confirmation_summary": output_root / "confirmation_summary.json",
        "hardware_comparison": output_root / "hardware_comparison.json",
        "hardware_comparison_markdown": (output_root / "hardware_comparison.md"),
        "checkpoint_catalog": output_root / "checkpoint_catalog.json",
        "finalization_receipt": output_root / "finalization_receipt.json",
    }
    checkpoint_stage_relative = checkpoint_stage_root.relative_to(project_root)
    checkpoint_path_relative = checkpoint_path_root.relative_to(project_root)
    commands = [
        [
            str(python),
            str(scripts / "compose_timefuse_numerical_recovery.py"),
            "--manifest",
            str(manifest),
            "--protocol",
            str(inputs["protocol"]),
            "--a10g-primary",
            str(inputs["a10g_primary"]),
            "--a100-primary",
            str(inputs["a100_primary"]),
            "--a10g-exact",
            str(inputs["a10g_exact"]),
            "--a100-exact",
            str(inputs["a100_exact"]),
            "--a10g-fallback",
            str(inputs["a10g_fallback"]),
            "--a100-fallback",
            str(inputs["a100_fallback"]),
            "--a10g-exact-metadata",
            str(inputs["a10g_exact_metadata"]),
            "--a10g-fallback-metadata",
            str(inputs["a10g_fallback_metadata"]),
            "--a100-exact-receipt",
            str(inputs["a100_exact_receipt"]),
            "--a100-fallback-receipt",
            str(inputs["a100_fallback_receipt"]),
            "--a10g-output",
            str(outputs["a10g_composed_summary"]),
            "--a100-output",
            str(outputs["a100_composed_summary"]),
            "--receipt",
            str(outputs["composition_receipt"]),
        ],
        [
            str(python),
            str(scripts / "summarize_timefuse_confirmation.py"),
            "--manifest",
            str(manifest),
            "--matrix-summary",
            str(outputs["a10g_composed_summary"]),
            "--protocol",
            str(confirmation_protocol),
            "--output",
            str(outputs["confirmation_summary"]),
        ],
        [
            str(python),
            str(scripts / "compare_timefuse_matrix_runs.py"),
            "--manifest",
            str(manifest),
            "--a10g-summary",
            str(outputs["a10g_composed_summary"]),
            "--a100-summary",
            str(outputs["a100_composed_summary"]),
            "--a10g-runtime-summary",
            str(inputs["a10g_primary"]),
            "--a100-runtime-summary",
            str(inputs["a100_primary"]),
            "--a100-final-status",
            str(inputs["a100_final_status"]),
            "--method-revision",
            METHOD_REVISION,
            "--output",
            str(outputs["hardware_comparison"]),
            "--markdown",
            str(outputs["hardware_comparison_markdown"]),
        ],
        [
            str(python),
            str(scripts / "index_matrix_checkpoints.py"),
            str(outputs["a10g_composed_summary"]),
            "--project-root",
            str(project_root),
            "--manifest",
            str(replay_manifest),
            "--stage-root",
            str(checkpoint_stage_relative),
            "--checkpoint-path-root",
            str(checkpoint_path_relative),
            "--output",
            str(outputs["checkpoint_catalog"]),
            "--recovery-receipt",
            str(outputs["composition_receipt"]),
            "--recovery-protocol",
            str(inputs["protocol"]),
            "--hash",
        ],
    ]
    return commands, outputs


def _validate_outputs(outputs):
    a10g = _load_json(
        outputs["a10g_composed_summary"],
        "Composed A10G summary",
    )
    a100 = _load_json(
        outputs["a100_composed_summary"],
        "Composed A100 summary",
    )
    for label, summary in (("A10G", a10g), ("A100", a100)):
        counts = summary.get("counts", {})
        if (
            summary.get("source_revision") != METHOD_REVISION
            or counts.get("expected") != EXPECTED_CELLS
            or counts.get("completed") != EXPECTED_CELLS
            or any(
                counts.get(state) != 0
                for state in ("failed", "pending", "running", "incomplete")
            )
            or not isinstance(summary.get("numerical_recovery"), dict)
        ):
            raise ValueError(f"Composed {label} summary is incomplete")

    composition = _load_json(
        outputs["composition_receipt"],
        "Composition receipt",
    )
    confirmation = _load_json(
        outputs["confirmation_summary"],
        "Confirmation summary",
    )
    comparison = _load_json(
        outputs["hardware_comparison"],
        "Hardware comparison",
    )
    catalog = _load_json(
        outputs["checkpoint_catalog"],
        "Checkpoint catalog",
    )
    if (
        composition.get("method_revision") != METHOD_REVISION
        or composition.get("selection_uses_metric_quality") is not False
        or not isinstance(composition.get("selected_profiles"), dict)
    ):
        raise ValueError("Composition receipt is incomplete")
    if confirmation.get("scope", {}).get("confirmatory") != 462 or not isinstance(
        confirmation.get("confirmatory_gate"), dict
    ):
        raise ValueError("Confirmation summary scope drifted")
    comparison_scope = comparison.get("scope", {})
    comparison_manifest = comparison.get("manifest", {})
    comparison_tolerances = comparison.get("tolerances", {})
    consistency_gates = comparison.get("consistency_gates", {})
    metric_consistency = comparison.get("metric_consistency", {})
    runtime = comparison.get("runtime", {})
    runtime_gates = runtime.get("gates", {})
    if (
        comparison.get("schema_version") != 1
        or comparison.get("method_revision") != METHOD_REVISION
        or comparison.get("seed") != 2021
        or comparison_manifest.get("expected_cells") != EXPECTED_CELLS
        or comparison_manifest.get("expected_metric_values_per_run")
        != EXPECTED_METRIC_VALUES
        or comparison_tolerances.get("absolute") != 1e-5
        or comparison_tolerances.get("relative") != 1e-4
        or comparison_scope.get("paired_cells") != EXPECTED_CELLS
        or comparison_scope.get("runtime_paired_cells") != EXPECTED_CELLS
        or consistency_gates.get("complete_585_cell_scope") is not True
        or metric_consistency.get("baseline", {}).get("count")
        != EXPECTED_METRIC_VALUES
        or metric_consistency.get("corrected", {}).get("count")
        != EXPECTED_METRIC_VALUES
        or len(comparison.get("cells", [])) != EXPECTED_CELLS
        or runtime.get("comparison_scope") != "first_pass_summaries"
        or runtime_gates.get("complete_585_cell_scope") is not True
        or runtime_gates.get("all_cells_have_finite_positive_timing")
        is not True
        or runtime_gates.get("a100_reported_matrix_seconds_present")
        is not True
        or runtime_gates.get("all_passed") is not True
        or runtime.get("a10g_4gpu", {}).get("terminal_cells")
        != EXPECTED_CELLS
        or runtime.get("a100_8gpu", {}).get("terminal_cells")
        != EXPECTED_CELLS
        or runtime.get("all_paired_cells", {}).get("cell_count")
        != EXPECTED_CELLS
    ):
        raise ValueError("Hardware comparison scope drifted")
    if (
        catalog.get("selected_count") != EXPECTED_REPLAY_CHECKPOINTS
        or catalog.get("checkpoint_count") != EXPECTED_REPLAY_CHECKPOINTS
        or catalog.get("usable_count") != EXPECTED_REPLAY_CHECKPOINTS
        or catalog.get("missing_count") != 0
        or catalog.get("missing_selected_count") != 0
        or catalog.get("hashes_included") is not True
        or not isinstance(catalog.get("numerical_recovery_receipt"), dict)
        or not isinstance(catalog.get("numerical_recovery_cohort"), dict)
        or sum(
            record.get("numerical_recovery_cohort") is not None
            for record in catalog.get("checkpoints", {}).values()
        )
        != EXPECTED_REPLAY_RECOVERY_COHORT_CHECKPOINTS
    ):
        raise ValueError("Recovery-aware checkpoint catalog is incomplete")
    recovered_checkpoint_count = sum(
        record.get("numerical_recovery") is not None
        for record in catalog["checkpoints"].values()
    )
    return {
        "primary_gate_passed": a10g["publication_gate"]["development_gate_passed"],
        "confirmation_gate_passed": confirmation["confirmatory_gate"][
            "confirmatory_gate_passed"
        ],
        "consistency_gates": consistency_gates,
        "selected_profiles": composition["selected_profiles"],
        "checkpoint_count": catalog["checkpoint_count"],
        "recovered_checkpoint_count": recovered_checkpoint_count,
        "recovery_cohort_checkpoint_count": (
            EXPECTED_REPLAY_RECOVERY_COHORT_CHECKPOINTS
        ),
    }


def _verify_completed_receipt(
    receipt_path,
    *,
    controller_revision,
    input_records,
    expected_artifacts=None,
):
    receipt = _load_json(receipt_path, "Finalization receipt")
    expected_names = {
        "a10g_composed_summary",
        "a100_composed_summary",
        "composition_receipt",
        "confirmation_summary",
        "hardware_comparison",
        "hardware_comparison_markdown",
        "checkpoint_catalog",
    }
    if (
        receipt.get("schema_version") != 1
        or receipt.get("controller_revision") != controller_revision
        or receipt.get("method_revision") != METHOD_REVISION
        or receipt.get("inputs") != input_records
        or set(receipt.get("artifacts", {})) != expected_names
    ):
        raise ValueError("Existing finalization receipt identity drifted")
    for name, record in receipt["artifacts"].items():
        path = Path(record.get("path", ""))
        if (
            not path.is_file()
            or record.get("sha256") != _sha256(path)
            or record.get("size_bytes") != path.stat().st_size
            or (
                expected_artifacts is not None
                and path.resolve() != Path(expected_artifacts[name]).resolve()
            )
        ):
            raise ValueError("Existing finalization artifact drifted")
    if expected_artifacts is not None:
        evidence = _validate_outputs(expected_artifacts)
        if receipt.get("evidence") != evidence:
            raise ValueError("Existing finalization evidence drifted")
    return receipt


def finalize(args, *, runner=subprocess.run):
    project_root = Path(args.project_root).resolve()
    if not project_root.is_dir():
        raise ValueError("project-root must be an existing directory")
    if args.required_filesystem:
        completed = runner(
            ["findmnt", "-T", str(project_root), "-o", "FSTYPE", "-n"],
            check=True,
            capture_output=True,
            text=True,
        )
        if completed.stdout.strip() != args.required_filesystem:
            raise ValueError(f"project-root must use {args.required_filesystem}")
    if not re.fullmatch(r"[0-9a-f]{40}", args.controller_revision):
        raise ValueError("controller-revision must be a full Git commit")
    source_root = _under(
        project_root,
        args.source_root,
        "source-root",
    )
    try:
        recorded_revision = (
            (source_root.parent / "source_revision").read_text(encoding="utf-8").strip()
        )
    except OSError as error:
        raise ValueError("source-root has no revision marker") from error
    if recorded_revision != args.controller_revision:
        raise ValueError("source-root revision marker drifted")
    python = _project_python(project_root, args.python)
    required_scripts = (
        "compose_timefuse_numerical_recovery.py",
        "summarize_timefuse_confirmation.py",
        "compare_timefuse_matrix_runs.py",
        "index_matrix_checkpoints.py",
    )
    for name in required_scripts:
        if not (source_root / "scripts" / name).is_file():
            raise ValueError(f"source-root is missing scripts/{name}")

    manifest = _under(
        project_root,
        args.manifest,
        "manifest",
        require_exists=True,
    )
    replay_manifest = _under(
        project_root,
        args.replay_manifest,
        "replay-manifest",
        require_exists=True,
    )
    confirmation_protocol = _under(
        project_root,
        args.confirmation_protocol,
        "confirmation-protocol",
        require_exists=True,
    )
    input_names = (
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
    inputs = {
        name: _under(
            project_root,
            getattr(args, name),
            name.replace("_", "-"),
            require_exists=True,
        )
        for name in input_names
    }
    output_root = _under(project_root, args.output_root, "output-root")
    checkpoint_stage_root = _under(
        project_root,
        args.checkpoint_stage_root,
        "checkpoint-stage-root",
    )
    checkpoint_path_root = _under(
        project_root,
        args.checkpoint_path_root or args.checkpoint_stage_root,
        "checkpoint-path-root",
    )
    if (
        checkpoint_stage_root != checkpoint_path_root
        and checkpoint_path_root not in checkpoint_stage_root.parents
    ):
        raise ValueError(
            "checkpoint-path-root must contain checkpoint-stage-root"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_stage_root.mkdir(parents=True, exist_ok=True)
    input_records = {
        "manifest": _file_record(manifest),
        "replay_manifest": _file_record(replay_manifest),
        "confirmation_protocol": _file_record(confirmation_protocol),
        **{name: _file_record(path) for name, path in inputs.items()},
    }
    commands, outputs = build_command_plan(
        python=python,
        source_root=source_root,
        project_root=project_root,
        manifest=manifest,
        replay_manifest=replay_manifest,
        confirmation_protocol=confirmation_protocol,
        inputs=inputs,
        output_root=output_root,
        checkpoint_stage_root=checkpoint_stage_root,
        checkpoint_path_root=checkpoint_path_root,
    )
    receipt_path = outputs["finalization_receipt"]
    if receipt_path.exists():
        return _verify_completed_receipt(
            receipt_path,
            controller_revision=args.controller_revision,
            input_records=input_records,
            expected_artifacts={
                name: path
                for name, path in outputs.items()
                if name != "finalization_receipt"
            },
        )

    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(source_root)
    for command in commands:
        runner(command, cwd=source_root, env=environment, check=True)

    evidence = _validate_outputs(outputs)
    artifact_records = {
        name: _file_record(path)
        for name, path in outputs.items()
        if name != "finalization_receipt"
    }
    receipt = {
        "schema_version": 1,
        "generated_unix": time.time(),
        "controller_revision": args.controller_revision,
        "method_revision": METHOD_REVISION,
        "project_root": str(project_root),
        "source_root": str(source_root),
        "checkpoint_stage_root": str(checkpoint_stage_root),
        "checkpoint_path_root": str(checkpoint_path_root),
        "inputs": input_records,
        "artifacts": artifact_records,
        "evidence": evidence,
        "commands": commands,
    }
    save_json_atomic(receipt, receipt_path)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Compose numerical recovery, compare hardware, summarize the "
            "confirmation set, and stage the 208 replay checkpoints"
        )
    )
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--python")
    parser.add_argument("--controller-revision", required=True)
    parser.add_argument(
        "--manifest",
        default="docs/timefuse_experiment_manifest.jsonl",
    )
    parser.add_argument(
        "--replay-manifest",
        default="docs/timefuse_checkpoint_replay_manifest.jsonl",
    )
    parser.add_argument(
        "--confirmation-protocol",
        default="docs/timefuse_confirmation_protocol.json",
    )
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--a10g-primary", required=True)
    parser.add_argument("--a100-primary", required=True)
    parser.add_argument("--a10g-exact", required=True)
    parser.add_argument("--a100-exact", required=True)
    parser.add_argument("--a10g-fallback", required=True)
    parser.add_argument("--a100-fallback", required=True)
    parser.add_argument("--a10g-exact-metadata", required=True)
    parser.add_argument("--a10g-fallback-metadata", required=True)
    parser.add_argument("--a100-exact-receipt", required=True)
    parser.add_argument("--a100-fallback-receipt", required=True)
    parser.add_argument("--a100-final-status", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--checkpoint-stage-root", required=True)
    parser.add_argument("--checkpoint-path-root")
    parser.add_argument("--required-filesystem", default="nfs4")
    args = parser.parse_args(argv)

    receipt = finalize(args)
    print(
        json.dumps(
            {
                "controller_revision": receipt["controller_revision"],
                "method_revision": receipt["method_revision"],
                "evidence": receipt["evidence"],
                "finalization_receipt": str(
                    _under(
                        args.project_root,
                        args.output_root,
                        "output-root",
                    )
                    / "finalization_receipt.json"
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
