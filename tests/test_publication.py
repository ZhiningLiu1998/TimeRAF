import hashlib
import json
from pathlib import Path

import pytest

from scripts import compose_timefuse_numerical_recovery
from scripts.greenland_common import (
    METHOD_REVISION,
    NUMERICAL_RECOVERY_CELL_IDS,
    NUMERICAL_RECOVERY_CELL_IDS_SHA256,
    NUMERICAL_RECOVERY_COHORT_ID,
    NUMERICAL_RECOVERY_JOB,
    NUMERICAL_RECOVERY_PROFILES,
    NUMERICAL_RECOVERY_PROTOCOL,
    NUMERICAL_RECOVERY_PROTOCOL_SHA256,
)
from ts_rag.appendix_benchmark import (
    autogluon_checkpoint_compatibility,
)
from ts_rag.appendix_matrix import (
    APPENDIX_EXECUTION_TOPOLOGY,
    APPENDIX_METHOD_REVISION,
    APPENDIX_SELECTOR_POLICY,
    APPENDIX_SELECTOR_PROTOCOL_SHA256,
    _publication_gate as _appendix_publication_gate,
)
from ts_rag.appendix_rag import load_appendix_manifest
from ts_rag.matrix import (
    _publication_gate as _matrix_publication_gate,
    build_confirmation_summary,
    load_manifest,
)
from ts_rag import publication


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_REVISION = "a" * 40
LAUNCHER_REVISION = "b" * 40
APPENDIX_REVISION = "appendix-source-revision"
APPENDIX_LAUNCHER_REVISION = "e" * 40
RUN_ID = "timefuse-replay-fixture"
IMAGE_URI = (
    "<ECR_REGISTRY>/"
    "timeraf-greenland@sha256:" + "c" * 64
)


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _relative(path, root):
    return Path(path).relative_to(root).as_posix()


def _cell_id_sha256(cell_ids):
    payload = "".join(f"{cell_id}\n" for cell_id in sorted(cell_ids))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _primary_evidence(project):
    manifest = project / "docs" / "timefuse_experiment_manifest.jsonl"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_bytes(
        (
            REPOSITORY_ROOT
            / "docs"
            / "timefuse_experiment_manifest.jsonl"
        ).read_bytes()
    )
    rows = load_manifest(manifest)
    states = []
    for index, cell in enumerate(rows):
        started = float(index + 1 if index < 123 else index + 1000)
        states.append(
            {
                "cell_id": cell["id"],
                "task_family": cell["task_family"],
                "dataset": cell["dataset"],
                "model": cell["model"],
                "pred_len": cell["pred_len"],
                "state": "completed",
                "started_unix": started,
                "all_test_metrics_improve": True,
                "metric_gain_percent": {
                    metric: 10.0 for metric in cell["metrics"]
                },
            }
        )
    counts = {
        "expected": 585,
        "completed": 585,
        "improved": 585,
        "not_improved": 0,
        "failed": 0,
        "running": 0,
        "pending": 0,
        "incomplete": 0,
    }
    summary = {
        "generated_unix": 1.0,
        "output_root": str(project / "outputs" / "primary"),
        "smoke": False,
        "seed": 2021,
        "source_revision": METHOD_REVISION,
        "export_only": False,
        "counts": counts,
        "all_completed": True,
        "all_improved": True,
        "publication_gate": _matrix_publication_gate(
            rows,
            states,
            required_cells=585,
        ),
        "cell_states": states,
    }
    summary_path = _write_json(
        project / "operations" / "primary_summary.json",
        summary,
    )

    development_ids = [row["cell_id"] for row in states[:123]]
    protocol = {
        "schema_version": 1,
        "method_source_revision": METHOD_REVISION,
        "full_matrix_cell_count": 585,
        "development_cell_count": 123,
        "confirmatory_cell_count": 462,
        "development_selection": {
            "rule": "first_completed_cells_by_started_unix",
            "tie_breaker": "cell_id",
            "cell_ids_sha256": _cell_id_sha256(development_ids),
            "last_started_unix": states[122]["started_unix"],
        },
        "confirmatory_selection": {
            "rule": "full matrix complement of frozen development cells",
            "first_started_unix": states[123]["started_unix"],
            "minimum_boundary_gap_seconds": 1.0,
        },
        "thresholds": {
            "overall_strict_improvement_rate": 0.8,
            "task_family_strict_improvement_rate": 0.7,
            "positive_task_family_metric_means": True,
            "positive_overall_metric_means_and_medians": True,
            "complete_without_runtime_failures": True,
        },
        "independence": {
            "selection_uses_outcomes": False,
            "method_changes_after_boundary_allowed": False,
            "development_cells_excluded_from_confirmatory_denominators": True,
        },
    }
    protocol_path = _write_json(
        project / "docs" / "timefuse_confirmation_protocol.json",
        protocol,
    )
    confirmation = build_confirmation_summary(rows, summary, protocol)
    confirmation_path = _write_json(
        project / "operations" / "confirmation_summary.json",
        confirmation,
    )
    return {
        "manifest": manifest,
        "summary": summary_path,
        "protocol": protocol_path,
        "confirmation": confirmation_path,
    }


def _recovery_evidence(project, primary):
    recovery_protocol = project / NUMERICAL_RECOVERY_PROTOCOL
    recovery_protocol.write_bytes(
        (REPOSITORY_ROOT / NUMERICAL_RECOVERY_PROTOCOL).read_bytes()
    )
    primary_payload = json.loads(primary["summary"].read_text())
    states_by_id = {
        row["cell_id"]: row for row in primary_payload["cell_states"]
    }
    profile_rows = [
        states_by_id[cell_id] for cell_id in NUMERICAL_RECOVERY_CELL_IDS
    ]
    operations = project / "operations" / "recovery"
    summaries = {}
    metadata = {}
    receipts = {}
    a10g_revision = "c" * 40
    a100_revision = "d" * 40
    for hardware in ("a10g", "a100"):
        for profile in ("exact", "fallback-v1"):
            root = operations / hardware / profile
            summary = _write_json(
                root / "recomputed_matrix_summary.json",
                {
                    "source_revision": METHOD_REVISION,
                    "cell_states": profile_rows,
                },
            )
            summaries[f"{hardware}_{profile}"] = summary
            if hardware == "a10g":
                metadata[profile] = _write_json(
                    root / "recovery_metadata.json",
                    {
                        "schema_version": 1,
                        "cohort_id": NUMERICAL_RECOVERY_COHORT_ID,
                        "profile": profile,
                        "overrides": NUMERICAL_RECOVERY_PROFILES[profile],
                        "method_revision": METHOD_REVISION,
                        "recovery_revision": a10g_revision,
                        "parent_run_id": "a10g-primary-d9be338",
                        "protocol_sha256": (
                            NUMERICAL_RECOVERY_PROTOCOL_SHA256
                        ),
                        "cell_ids": list(NUMERICAL_RECOVERY_CELL_IDS),
                        "cell_ids_sha256": (
                            NUMERICAL_RECOVERY_CELL_IDS_SHA256
                        ),
                        "seed": 2021,
                        "instance_type": "ml.g5.12xlarge",
                        "instance_count": 1,
                        "reserved_gpus_per_host": 4,
                        "processes_per_host": 4,
                        "total_gpus": 4,
                        "world_size": 4,
                        "inactive_reserved_gpus": 0,
                        "output_root": str(root),
                    },
                )
            else:
                receipts[profile] = _write_json(
                    root / "fetch_receipt.json",
                    {
                        "schema_version": 1,
                        "run_id": f"numerical-recovery-{profile}-fixture",
                        "job_kind": NUMERICAL_RECOVERY_JOB,
                        "profile": profile,
                        "method_revision": METHOD_REVISION,
                        "source_revision": a100_revision,
                        "launcher_revision": a100_revision,
                        "recovery_revision": a100_revision,
                        "protocol_sha256": (
                            NUMERICAL_RECOVERY_PROTOCOL_SHA256
                        ),
                        "cell_ids_sha256": (
                            NUMERICAL_RECOVERY_CELL_IDS_SHA256
                        ),
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
                            "sha256": _sha256(summary),
                            "size_bytes": summary.stat().st_size,
                        },
                    },
                )

    a10g_output = operations / "composed_a10g.json"
    a100_output = operations / "composed_a100.json"
    receipt = operations / "composition_receipt.json"
    compose_timefuse_numerical_recovery.main(
        [
            "--manifest",
            str(primary["manifest"]),
            "--protocol",
            str(recovery_protocol),
            "--a10g-primary",
            str(primary["summary"]),
            "--a100-primary",
            str(primary["summary"]),
            "--a10g-exact",
            str(summaries["a10g_exact"]),
            "--a100-exact",
            str(summaries["a100_exact"]),
            "--a10g-fallback",
            str(summaries["a10g_fallback-v1"]),
            "--a100-fallback",
            str(summaries["a100_fallback-v1"]),
            "--a10g-exact-metadata",
            str(metadata["exact"]),
            "--a10g-fallback-metadata",
            str(metadata["fallback-v1"]),
            "--a100-exact-receipt",
            str(receipts["exact"]),
            "--a100-fallback-receipt",
            str(receipts["fallback-v1"]),
            "--a10g-output",
            str(a10g_output),
            "--a100-output",
            str(a100_output),
            "--receipt",
            str(receipt),
        ]
    )
    return {
        "summary": a10g_output,
        "a100_summary": a100_output,
        "receipt": receipt,
        "protocol": recovery_protocol,
        "a10g_exact": summaries["a10g_exact"],
    }


def _job_spec():
    digests = {
        "source": "1" * 64,
        "dataset": "2" * 64,
        "checkpoints": "3" * 64,
    }
    return {
        "schema_version": 1,
        "run_id": RUN_ID,
        "job_kind": "checkpoint_replay",
        "method_revision": METHOD_REVISION,
        "image_uri": IMAGE_URI,
        "source_revision": SOURCE_REVISION,
        "launcher_revision": LAUNCHER_REVISION,
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
            for name, digest in digests.items()
        ],
        "output_s3_uri": (
            "s3://<DEV_BUCKET>/timeraf/greenland/"
            f"runs/{RUN_ID}/"
        ),
        "matrix": {
            "manifest": "docs/timefuse_checkpoint_replay_manifest.jsonl",
            "output_root": "outputs/checkpoint_replay",
            "checkpoint_root": "checkpoints/replay",
            "checkpoint_catalog": "docs/checkpoint_catalog.json",
            "release_checkpoint_root": "checkpoints/replay",
            "summary": "outputs/checkpoint_replay/matrix_summary.json",
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


def _replay_evidence(project):
    manifest = (
        project / "docs" / "timefuse_checkpoint_replay_manifest.jsonl"
    )
    manifest.write_bytes(
        (
            REPOSITORY_ROOT
            / "docs"
            / "timefuse_checkpoint_replay_manifest.jsonl"
        ).read_bytes()
    )
    rows = load_manifest(manifest)
    destination = (
        project / "outputs" / "greenland_runs" / RUN_ID
    )
    objects = {}

    def add_object(relative, payload):
        path = destination / relative
        if isinstance(payload, dict):
            _write_json(path, payload)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
        objects[relative] = {
            "key": f"timeraf/greenland/runs/{RUN_ID}/{relative}",
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
            "etag": None,
            "version_id": None,
            "path": str(path),
            "mode": "downloaded",
        }
        return path

    spec = _job_spec()
    add_object("evidence/accepted_job_spec.json", spec)
    topology = {
        "sample_count": 2,
        "expected_gpu_ids": list(range(8)),
        "eight_distinct_bindings_observed": True,
        "binding_evidence": {
            "recorded_unix": 1.0,
            "pids": list(range(100, 108)),
            "gpu_ids": list(range(8)),
        },
        "max_utilization_gpu_percent": {
            index: 50 for index in range(8)
        },
        "all_eight_gpus_utilized": True,
        "topology_gate_passed": True,
    }
    add_object("evidence/topology_summary.json", topology)
    final_status = {
        "run_id": RUN_ID,
        "method_revision": METHOD_REVISION,
        "source_revision": SOURCE_REVISION,
        "launcher_revision": LAUNCHER_REVISION,
        "exit_code": 0,
        "status": "succeeded",
        "topology_gate_passed": True,
    }
    add_object("final_status.json", final_status)
    replay_counts = {
        "expected": 208,
        "completed": 208,
        "failed": 0,
        "pending": 0,
        "running": 0,
        "incomplete": 0,
    }
    add_object(
        "checkpoint_replay/matrix_summary.json",
        {
            "source_revision": SOURCE_REVISION,
            "export_only": True,
            "all_completed": True,
            "counts": replay_counts,
        },
    )

    bundles = {}
    for index, cell in enumerate(rows):
        relative = (
            f"checkpoint_replay/cells/{index:03d}/"
            "prediction_bundle.npz"
        )
        path = add_object(relative, f"bundle:{cell['id']}".encode())
        bundles[cell["id"]] = {
            "usable": True,
            "reason": None,
            "path": _relative(path, project),
            "recorded_path": f"/workspace/timeraf/outputs/{relative}",
            "relocated": True,
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
            "source_revision": SOURCE_REVISION,
            "task_family": cell["task_family"],
            "dataset": cell["dataset"],
            "model": cell["model"],
            "pred_len": cell["pred_len"],
        }
    catalog_path = _write_json(
        destination / "replay_bundle_catalog.json",
        {
            "schema_version": 1,
            "source_revision": SOURCE_REVISION,
            "output_root": str(destination / "checkpoint_replay"),
            "project_root": str(project),
            "expected_cells": 208,
            "cataloged_cells": 208,
            "usable_count": 208,
            "missing_count": 0,
            "hashes_verified": True,
            "bundles": bundles,
        },
    )
    receipt = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "source_s3_prefix": (
            "s3://<DEV_BUCKET>/timeraf/greenland/"
            f"runs/{RUN_ID}/"
        ),
        "project_root": str(project),
        "destination": str(destination),
        "method_revision": METHOD_REVISION,
        "source_revision": SOURCE_REVISION,
        "launcher_revision": LAUNCHER_REVISION,
        "object_count": len(objects),
        "total_size_bytes": sum(
            record["size_bytes"] for record in objects.values()
        ),
        "downloaded_count": len(objects),
        "reused_count": 0,
        "topology_gate_passed": True,
        "replay_counts": replay_counts,
        "replay_bundle_catalog": str(catalog_path),
        "replay_bundle_catalog_sha256": _sha256(catalog_path),
        "objects": objects,
        "final_status": final_status,
    }
    receipt_path = _write_json(
        destination / "fetch_receipt.json",
        receipt,
    )
    return {
        "manifest": manifest,
        "receipt": receipt_path,
        "catalog": catalog_path,
        "first_bundle": Path(next(iter(bundles.values()))["path"]),
    }


def _appendix_counts(states):
    evaluable = [row for row in states if row["paper_evaluable"]]
    completed = [row for row in evaluable if row["state"] == "completed"]
    return {
        "completed": len(completed),
        "paper_oot": sum(row["state"] == "paper_oot" for row in states),
        "failed": sum(row["state"] == "failed" for row in states),
        "running": sum(row["state"] == "running" for row in states),
        "pending": sum(row["state"] == "pending" for row in states),
        "incomplete": sum(row["state"] == "incomplete" for row in states),
        "reported": len(states),
        "evaluable": len(evaluable),
        "improved": sum(
            bool(row.get("all_test_metrics_improve")) for row in completed
        ),
        "not_improved": sum(
            not bool(row.get("all_test_metrics_improve"))
            for row in completed
        ),
    }


def _appendix_evidence(project, replay_catalog):
    manifest = (
        project / "docs" / "timefuse_appendix_experiment_manifest.jsonl"
    )
    manifest.write_bytes(
        (
            REPOSITORY_ROOT
            / "docs"
            / "timefuse_appendix_experiment_manifest.jsonl"
        ).read_bytes()
    )
    rows = load_appendix_manifest(manifest)
    replay_hash = _sha256(replay_catalog)
    bundles = {}
    states = []
    for index, cell in enumerate(rows):
        evaluable = bool(cell["paper_status"]["evaluable"])
        state = {
            "cell_id": cell["id"],
            "task_family": cell["task_family"],
            "dataset": cell["dataset"],
            "baseline": cell["baseline"],
            "pred_len": cell["pred_len"],
            "paper_evaluable": evaluable,
            "state": "completed" if evaluable else "paper_oot",
            "method_revision": APPENDIX_METHOD_REVISION,
        }
        if evaluable:
            state.update(
                {
                    "all_test_metrics_improve": True,
                    "metric_gain_percent": {
                        metric: 10.0 for metric in cell["metrics"]
                    },
                }
            )
            bundle = (
                project
                / "outputs"
                / "appendix"
                / f"{index:03d}"
                / "prediction_bundle.npz"
            )
            bundle.parent.mkdir(parents=True, exist_ok=True)
            bundle.write_bytes(f"appendix:{cell['id']}".encode())
            bundle_hash = _sha256(bundle)
            identity = {"source_revision": APPENDIX_REVISION}
            if cell["baseline"] in {
                "forward_selection",
                "portfolio_ensemble",
                "zeroshot_ensemble",
            }:
                identity["replay_catalog_sha256"] = replay_hash
            external = cell["baseline"] in {
                "autogluon_high_quality",
                "chronos_bolt_finetuned",
                "chronos_bolt_zeroshot",
            }
            compatibility = None
            if external:
                identity["autogluon_timeseries_version"] = "1.4.0"
            if cell["baseline"] == "autogluon_high_quality":
                compatibility = autogluon_checkpoint_compatibility(
                    cell["baseline"]
                )
                identity["checkpoint_compatibility"] = compatibility
            metadata = _write_json(
                Path(f"{bundle}.json"),
                {
                    "cell_id": cell["id"],
                    "prediction_bundle_sha256": bundle_hash,
                    "run_identity": identity,
                    **(
                        {"checkpoint_compatibility": compatibility}
                        if compatibility is not None
                        else {}
                    ),
                    **(
                        {
                            "predictor": {
                                "autogluon_timeseries_version": "1.4.0",
                                **(
                                    {
                                        "checkpoint_compatibility": (
                                            compatibility
                                        )
                                    }
                                    if compatibility is not None
                                    else {}
                                ),
                            }
                        }
                        if external
                        else {}
                    ),
                },
            )
            bundles[cell["id"]] = {
                "path": _relative(bundle, project),
                "sha256": bundle_hash,
                "size_bytes": bundle.stat().st_size,
                "metadata_path": _relative(metadata, project),
                "metadata_sha256": _sha256(metadata),
                "baseline": cell["baseline"],
                "dataset": cell["dataset"],
                "task_family": cell["task_family"],
                "pred_len": cell["pred_len"],
                **(
                    {"autogluon_timeseries_version": "1.4.0"}
                    if external
                    else {}
                ),
                **(
                    {"checkpoint_compatibility": compatibility}
                    if compatibility is not None
                    else {}
                ),
            }
        states.append(state)
    catalog = {
        "schema_version": 2,
        "source_revision": APPENDIX_REVISION,
        "project_root": str(project),
        "manifest": _relative(manifest, project),
        "manifest_sha256": _sha256(manifest),
        "replay_catalog": _relative(replay_catalog, project),
        "replay_catalog_sha256": replay_hash,
        "hashes_verified": True,
        "counts": {
            "reported": 96,
            "evaluable": 94,
            "cataloged": 94,
            "missing_evaluable": 0,
        },
        "bundles": bundles,
    }
    catalog_path = _write_json(
        project / "outputs" / "appendix" / "bundle_catalog.json",
        catalog,
    )
    summary = {
        "generated_unix": 1.0,
        "output_root": str(project / "outputs" / "appendix-rag"),
        "source_revision": APPENDIX_REVISION,
        "method_revision": APPENDIX_METHOD_REVISION,
        "launcher_revision": APPENDIX_LAUNCHER_REVISION,
        "selector_policy": APPENDIX_SELECTOR_POLICY,
        "selector_protocol_sha256": APPENDIX_SELECTOR_PROTOCOL_SHA256,
        "execution_topology": APPENDIX_EXECUTION_TOPOLOGY,
        "counts": _appendix_counts(states),
        "all_evaluable_completed": True,
        "publication_gate": _appendix_publication_gate(
            rows,
            states,
            expected_total=96,
            expected_evaluable=94,
        ),
        "cell_states": states,
    }
    summary_path = _write_json(
        project / "outputs" / "appendix" / "appendix_summary.json",
        summary,
    )
    return {
        "manifest": manifest,
        "catalog": catalog_path,
        "summary": summary_path,
    }


@pytest.fixture
def publication_fixture(tmp_path, monkeypatch):
    project = tmp_path / "TimeRAF"
    project.mkdir()
    primary = _primary_evidence(project)
    recovery = _recovery_evidence(project, primary)
    primary["summary"] = recovery["summary"]
    monkeypatch.setattr(
        publication,
        "CONFIRMATION_PROTOCOL_SHA256",
        _sha256(primary["protocol"]),
    )
    replay = _replay_evidence(project)
    appendix = _appendix_evidence(project, replay["catalog"])
    return {
        "project_root": project,
        "primary_manifest": primary["manifest"],
        "primary_summary": primary["summary"],
        "recovery_receipt": recovery["receipt"],
        "recovery_protocol": recovery["protocol"],
        "recovery_input": recovery["a10g_exact"],
        "confirmation_protocol": primary["protocol"],
        "confirmation_summary": primary["confirmation"],
        "replay_manifest": replay["manifest"],
        "replay_receipt": replay["receipt"],
        "replay_bundle": project / replay["first_bundle"],
        "appendix_manifest": appendix["manifest"],
        "appendix_catalog": appendix["catalog"],
        "appendix_summary": appendix["summary"],
        "appendix_launcher_revision": APPENDIX_LAUNCHER_REVISION,
    }


def _audit(paths):
    inputs = dict(paths)
    inputs.pop("replay_bundle")
    inputs.pop("recovery_input")
    return publication.audit_timefuse_publication(**inputs)


def test_publication_audit_accepts_complete_hash_verified_evidence(
    publication_fixture,
):
    report = _audit(publication_fixture)

    assert report["publication_ready"]
    assert all(report["criteria"].values())
    assert report["method_revision"] == METHOD_REVISION
    assert report["appendix_method_revision"] == APPENDIX_METHOD_REVISION
    assert (
        report["appendix_launcher_revision"]
        == APPENDIX_LAUNCHER_REVISION
    )
    assert report["numerical_recovery"]["verified"]
    assert report["numerical_recovery"]["required"]
    assert report["replay"]["verified_bundles"] == 208
    assert report["appendix"]["verified_bundles"] == 94


def test_publication_audit_reports_a_real_gate_failure(
    publication_fixture,
):
    summary_path = publication_fixture["appendix_summary"]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    regressed = [
        row
        for row in summary["cell_states"]
        if row["baseline"] == "forward_selection"
        and row["paper_evaluable"]
    ][:5]
    for row in regressed:
        row["all_test_metrics_improve"] = False
        row["metric_gain_percent"] = {
            metric: -1.0 for metric in row["metric_gain_percent"]
        }
    summary["counts"] = _appendix_counts(summary["cell_states"])
    summary["publication_gate"] = _appendix_publication_gate(
        load_appendix_manifest(publication_fixture["appendix_manifest"]),
        summary["cell_states"],
        expected_total=96,
        expected_evaluable=94,
    )
    _write_json(summary_path, summary)

    report = _audit(publication_fixture)

    assert not report["publication_ready"]
    assert not report["criteria"]["appendix_94_gate_passed"]
    assert all(
        value
        for name, value in report["criteria"].items()
        if name != "appendix_94_gate_passed"
    )


def test_publication_audit_rejects_replay_artifact_hash_drift(
    publication_fixture,
):
    publication_fixture["replay_bundle"].write_bytes(b"tampered")

    with pytest.raises(ValueError, match="Greenland object .* mismatch"):
        _audit(publication_fixture)


def test_publication_audit_requires_composition_receipt(
    publication_fixture,
):
    publication_fixture["recovery_receipt"] = None

    with pytest.raises(ValueError, match="requires its receipt"):
        _audit(publication_fixture)


def test_publication_audit_rejects_recovery_input_hash_drift(
    publication_fixture,
):
    publication_fixture["recovery_input"].write_text(
        "tampered",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Numerical recovery input a10g_exact SHA-256 mismatch",
    ):
        _audit(publication_fixture)


def test_publication_audit_rejects_appendix_method_revision_drift(
    publication_fixture,
):
    publication_fixture["appendix_method_revision"] = METHOD_REVISION

    with pytest.raises(
        ValueError,
        match="Appendix method revision drifted",
    ):
        _audit(publication_fixture)
