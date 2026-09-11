import hashlib
import json
from types import SimpleNamespace

import pytest

from scripts import run_appendix_bundle_pipeline as pipeline
from ts_rag.appendix_rag import load_appendix_manifest
from ts_rag.matrix import load_manifest


def _sha256(content):
    return hashlib.sha256(content).hexdigest()


def _write_replay_catalog(tmp_path):
    rows = load_manifest("docs/timefuse_checkpoint_replay_manifest.jsonl")
    bundles = {}
    for index, row in enumerate(rows):
        content = f"bundle-{index}".encode()
        path = tmp_path / "replay" / f"{index}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        bundles[row["id"]] = {
            "usable": True,
            "path": str(path.relative_to(tmp_path)),
            "size_bytes": len(content),
            "sha256": _sha256(content),
        }
    catalog_path = tmp_path / "replay_catalog.json"
    catalog_path.write_text(json.dumps({"bundles": bundles}))
    return catalog_path, bundles


def test_replay_preflight_requires_and_hashes_all_208_bundles(tmp_path):
    catalog_path, _ = _write_replay_catalog(tmp_path)

    context = pipeline.load_replay_libraries(
        "docs/timefuse_checkpoint_replay_manifest.jsonl",
        catalog_path,
        "docs/timefuse_appendix_experiment_manifest.jsonl",
        tmp_path,
    )

    assert len(context["verified_bundles"]) == 208
    assert len(context["libraries"]) == 16
    assert all(len(library) == 13 for library in context["libraries"].values())


def test_replay_preflight_rejects_hash_drift(tmp_path):
    catalog_path, bundles = _write_replay_catalog(tmp_path)
    first = next(iter(bundles.values()))
    (tmp_path / first["path"]).write_bytes(b"changed")
    first["size_bytes"] = len(b"changed")
    catalog_path.write_text(json.dumps({"bundles": bundles}))

    with pytest.raises(ValueError, match="hash mismatch"):
        pipeline.load_replay_libraries(
            "docs/timefuse_checkpoint_replay_manifest.jsonl",
            catalog_path,
            "docs/timefuse_appendix_experiment_manifest.jsonl",
            tmp_path,
        )


def test_external_runtime_preflight_is_exact_and_fail_closed(monkeypatch):
    monkeypatch.setattr(
        pipeline,
        "require_autogluon_timeseries_version",
        lambda: "1.4.0",
    )
    assert pipeline.preflight_external_runtime() == {
        "autogluon_timeseries": {
            "installed_version": "1.4.0",
            "required_version": "1.4.0",
            "exact_version_match": True,
        }
    }

    def reject_version():
        raise RuntimeError("version drift")

    monkeypatch.setattr(
        pipeline,
        "require_autogluon_timeseries_version",
        reject_version,
    )
    with pytest.raises(RuntimeError, match="version drift"):
        pipeline.run_external_exports(
            {"appendix_rows": []},
            SimpleNamespace(),
        )


def test_external_main_preflights_before_lock_or_input_reads(
    tmp_path,
    monkeypatch,
):
    def reject_version():
        raise RuntimeError("version drift")

    def reject_side_effect(*args, **kwargs):
        raise AssertionError("pipeline work must not start")

    output_root = tmp_path / "output"
    monkeypatch.setattr(
        pipeline,
        "require_autogluon_timeseries_version",
        reject_version,
    )
    monkeypatch.setattr(pipeline, "_acquire_lock", reject_side_effect)
    monkeypatch.setattr(pipeline, "load_replay_libraries", reject_side_effect)
    monkeypatch.setattr(
        pipeline.sys,
        "argv",
        [
            "run_appendix_bundle_pipeline.py",
            "--replay-catalog",
            str(tmp_path / "missing.json"),
            "--output-root",
            str(output_root),
            "--source-revision",
            "source-revision",
            "--launcher-revision",
            "launcher-revision",
            "--phase",
            "external",
            "--dry-run",
        ],
    )

    with pytest.raises(RuntimeError, match="version drift"):
        pipeline.main()

    assert not output_root.exists()


def test_p4de_external_topology_requires_eight_active_a100_workers(
    monkeypatch,
):
    monkeypatch.setattr(pipeline.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(pipeline.torch.cuda, "device_count", lambda: 8)

    topology = pipeline.build_external_topology(
        "ml.p4de.24xlarge",
        reserved_gpus_per_host=8,
        processes_per_host=8,
        gpu_ids=list(range(8)),
        queued_chronos_cells=32,
        queued_evaluable_cells=46,
    )

    assert topology["world_size"] == 8
    assert topology["worker_gpu_ids"] == list(range(8))
    assert topology["inactive_reserved_gpus"] == 0

    with pytest.raises(ValueError, match="requires eight visible"):
        pipeline.build_external_topology(
            "ml.p4de.24xlarge",
            reserved_gpus_per_host=8,
            processes_per_host=4,
            gpu_ids=list(range(4)),
            queued_chronos_cells=32,
            queued_evaluable_cells=46,
        )
    with pytest.raises(ValueError, match="at least eight queued Chronos"):
        pipeline.build_external_topology(
            "ml.p4de.24xlarge",
            reserved_gpus_per_host=8,
            processes_per_host=8,
            gpu_ids=list(range(8)),
            queued_chronos_cells=7,
            queued_evaluable_cells=7,
        )
    with pytest.raises(ValueError, match="at least eight queued Chronos"):
        pipeline.build_external_topology(
            "ml.p4de.24xlarge",
            reserved_gpus_per_host=8,
            processes_per_host=8,
            gpu_ids=list(range(8)),
            queued_chronos_cells=0,
            queued_evaluable_cells=14,
        )


def test_external_worker_isolates_one_physical_gpu(monkeypatch, tmp_path):
    captured = {}

    class _Process:
        stdout = iter(())

        @staticmethod
        def wait():
            return 0

    def fake_popen(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return _Process()

    monkeypatch.setattr(pipeline.subprocess, "Popen", fake_popen)
    cell = next(
        row
        for row in load_appendix_manifest(
            "docs/timefuse_appendix_experiment_manifest.jsonl"
        )
        if row["baseline"] == "chronos_bolt_zeroshot"
    )
    args = SimpleNamespace(
        output_root=str(tmp_path),
        appendix_manifest="manifest.jsonl",
        source_revision="source-revision",
        launcher_revision="launcher-revision",
        seed=2021,
        batch_windows=8,
        max_prediction_items=32,
        fine_tune_steps=1000,
        inference_batch_size=16,
        fine_tune_batch_size=16,
        time_limit=None,
        data_roots={},
        cache_root=str(tmp_path / ".cache" / "appendix"),
    )

    assert pipeline._run_external_cell(cell, args, gpu_id=3) == 0

    assert captured["kwargs"]["env"]["CUDA_VISIBLE_DEVICES"] == "3"
    assert captured["kwargs"]["env"]["HF_HOME"].startswith(str(tmp_path))
    assert captured["kwargs"]["env"]["TMPDIR"].startswith(str(tmp_path))
    command = captured["command"]
    assert command[command.index("--physical-gpu-id") + 1] == "3"
    assert (
        command[command.index("--source-revision") + 1]
        == "source-revision"
    )


def test_external_resume_rejects_autogluon_version_drift(tmp_path):
    cell = next(
        row
        for row in load_appendix_manifest(
            "docs/timefuse_appendix_experiment_manifest.jsonl"
        )
        if row["baseline"] == "chronos_bolt_zeroshot"
    )
    bundle = pipeline._bundle_path(tmp_path, cell)
    bundle.parent.mkdir(parents=True, exist_ok=True)
    bundle.write_bytes(b"prediction-bundle")
    metadata_path = pipeline._metadata_path(bundle)
    metadata = {
        "evaluation_status": "exported",
        "prediction_bundle_sha256": pipeline.sha256_file(bundle),
        "run_identity": {
            "source_revision": "source-revision",
            "autogluon_timeseries_version": "1.5.0",
        },
        "predictor": {
            "autogluon_timeseries_version": "1.4.0",
        },
    }
    pipeline.save_json_atomic(metadata, metadata_path)

    with pytest.raises(ValueError, match="AutoGluon version mismatch"):
        pipeline._external_complete(
            cell,
            tmp_path,
            "source-revision",
        )

    metadata["run_identity"]["autogluon_timeseries_version"] = "1.4.0"
    pipeline.save_json_atomic(metadata, metadata_path)
    assert pipeline._external_complete(
        cell,
        tmp_path,
        "source-revision",
    )


def test_gpu_evidence_requires_distinct_bindings_and_nonzero_utilization():
    sample = {
        "recorded_unix": 1.0,
        "gpus": [
            {
                "index": index,
                "utilization_gpu_percent": 30,
                "processes": [
                    {
                        "pid": 100 + index,
                        "cuda_visible_devices": str(index),
                    }
                ],
            }
            for index in range(8)
        ],
    }

    summary = pipeline._summarize_gpu_samples([sample], range(8))

    assert summary["distinct_bindings_observed"]
    assert summary["all_expected_gpus_utilized"]
    assert summary["topology_gate_passed"]

    sample["gpus"][7]["utilization_gpu_percent"] = 0
    idle = pipeline._summarize_gpu_samples([sample], range(8))
    assert not idle["topology_gate_passed"]


def test_completed_external_resume_preserves_gpu_evidence(
    monkeypatch,
    tmp_path,
):
    cell = next(
        row
        for row in load_appendix_manifest(
            "docs/timefuse_appendix_experiment_manifest.jsonl"
        )
        if row["baseline"] == "chronos_bolt_zeroshot"
    )
    gpu_ids = [0, 1, 2, 3]
    metadata_path = tmp_path / "external_run_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "source_revision": "source-revision",
                "queued_evaluable_cells": 1,
                "topology": {"worker_gpu_ids": gpu_ids},
            }
        ),
        encoding="utf-8",
    )
    evidence_path = tmp_path / "gpu_evidence" / "summary.json"
    evidence_path.parent.mkdir(parents=True)
    evidence_path.write_text(
        json.dumps(
            {
                "expected_gpu_ids": gpu_ids,
                "sample_count": 1,
                "topology_gate_passed": True,
            }
        ),
        encoding="utf-8",
    )
    original_metadata = metadata_path.read_bytes()
    original_evidence = evidence_path.read_bytes()
    monkeypatch.setattr(
        pipeline,
        "preflight_external_runtime",
        lambda: {
            "autogluon_timeseries": {
                "installed_version": "1.4.0",
                "required_version": "1.4.0",
                "exact_version_match": True,
            }
        },
    )
    monkeypatch.setattr(
        pipeline,
        "_external_complete",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        pipeline,
        "build_external_topology",
        lambda *_args, **_kwargs: {"worker_gpu_ids": gpu_ids},
    )
    monkeypatch.setattr(
        pipeline,
        "_GpuSampler",
        lambda *_args, **_kwargs: pytest.fail(
            "a completed resume must not start a new sampler"
        ),
    )
    args = SimpleNamespace(
        output_root=str(tmp_path),
        appendix_manifest=(
            "docs/timefuse_appendix_experiment_manifest.jsonl"
        ),
        source_revision="source-revision",
        launcher_revision="launcher-revision",
        seed=2021,
        cache_root=str(tmp_path / ".cache"),
        dry_run=False,
        max_external_cells=0,
        instance_type="ml.g5.12xlarge",
        reserved_gpus_per_host=4,
        processes_per_host=4,
        gpu_ids=gpu_ids,
    )

    assert (
        pipeline.run_external_exports({"appendix_rows": [cell]}, args) == 0
    )
    assert metadata_path.read_bytes() == original_metadata
    assert evidence_path.read_bytes() == original_evidence


def test_static_pipeline_generates_complete_48_bundle_set(
    monkeypatch,
    tmp_path,
):
    rows = load_appendix_manifest(
        "docs/timefuse_appendix_experiment_manifest.jsonl"
    )
    static_rows = [
        row
        for row in rows
        if row["baseline"] in pipeline.STATIC_BASELINES
    ]
    targets = {
        (row["task_family"], row["dataset"], row["pred_len"])
        for row in static_rows
    }
    context = {
        "appendix_rows": static_rows,
        "replay_catalog_sha256": "catalog-hash",
        "replay_manifest_sha256": pipeline.EXPECTED_REPLAY_MANIFEST_SHA256,
        "libraries": {target: {} for target in targets},
    }

    def fake_advanced(
        base_bundles,
        task_family,
        method,
        output_path,
        forward_ensemble_size,
    ):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(f"{task_family}-{method}".encode())
        return {
            "prediction_bundle": str(output_path),
            "prediction_bundle_sha256": pipeline.sha256_file(output_path),
        }

    def fake_zeroshot(task_bundles, task_family, output_root):
        outputs = {}
        for dataset in task_bundles:
            path = output_root / dataset / "prediction_bundle.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"{task_family}-{dataset}".encode())
            outputs[dataset] = {
                "prediction_bundle": str(path),
                "prediction_bundle_sha256": pipeline.sha256_file(path),
            }
        return {
            "task_family": task_family,
            "protocol": {"task_holdout": "test"},
            "outputs": outputs,
        }

    monkeypatch.setattr(
        pipeline,
        "build_advanced_ensemble_bundle",
        fake_advanced,
    )
    monkeypatch.setattr(
        pipeline,
        "build_zeroshot_ensemble_bundles",
        fake_zeroshot,
    )

    outputs = pipeline.build_static_bundles(
        context,
        tmp_path,
        source_revision="revision",
    )

    assert len(outputs) == 48
    assert all(path.is_file() for path in outputs.values())


def test_complete_catalog_requires_all_94_evaluable_bundles(tmp_path):
    rows = load_appendix_manifest(
        "docs/timefuse_appendix_experiment_manifest.jsonl"
    )
    manifest = tmp_path / "appendix.jsonl"
    manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    replay_catalog = tmp_path / "replay.json"
    replay_catalog.write_text("{}")
    source_revision = "source-revision"
    for cell in rows:
        if not cell["paper_status"]["evaluable"]:
            continue
        bundle = pipeline._bundle_path(tmp_path, cell)
        bundle.parent.mkdir(parents=True, exist_ok=True)
        bundle.write_bytes(cell["id"].encode())
        identity = {"source_revision": source_revision}
        if cell["baseline"] in pipeline.STATIC_BASELINES:
            identity["replay_catalog_sha256"] = pipeline.sha256_file(
                replay_catalog
            )
        predictor = None
        if cell["baseline"] in pipeline.EXTERNAL_BASELINES:
            identity["autogluon_timeseries_version"] = "1.4.0"
            predictor = {
                "autogluon_timeseries_version": "1.4.0",
            }
            if cell["baseline"] == "autogluon_high_quality":
                compatibility = (
                    pipeline.autogluon_checkpoint_compatibility(
                        cell["baseline"]
                    )
                )
                identity["checkpoint_compatibility"] = compatibility
                predictor["checkpoint_compatibility"] = compatibility
        pipeline.save_json_atomic(
            {
                "prediction_bundle_sha256": pipeline.sha256_file(bundle),
                "run_identity": identity,
                **(
                    {"checkpoint_compatibility": compatibility}
                    if cell["baseline"] == "autogluon_high_quality"
                    else {}
                ),
                **({"predictor": predictor} if predictor else {}),
            },
            pipeline._metadata_path(bundle),
        )

    catalog = pipeline.build_complete_catalog(
        rows,
        manifest,
        replay_catalog,
        tmp_path,
        tmp_path,
        source_revision,
    )

    assert catalog["counts"] == {
        "reported": 96,
        "evaluable": 94,
        "cataloged": 94,
        "missing_evaluable": 0,
    }
    assert catalog["hashes_verified"]
    assert catalog["project_root"] == str(tmp_path.resolve())

    external_id = next(
        cell_id
        for cell_id, record in catalog["bundles"].items()
        if record["baseline"] in pipeline.EXTERNAL_BASELINES
    )
    external_metadata_path = (
        tmp_path / catalog["bundles"][external_id]["metadata_path"]
    )
    external_metadata = json.loads(external_metadata_path.read_text())
    external_metadata["run_identity"][
        "autogluon_timeseries_version"
    ] = "1.5.0"
    external_metadata_path.write_text(json.dumps(external_metadata))
    with pytest.raises(ValueError, match="AutoGluon version mismatch"):
        pipeline.build_complete_catalog(
            rows,
            manifest,
            replay_catalog,
            tmp_path,
            tmp_path,
            source_revision,
        )
    external_metadata["run_identity"][
        "autogluon_timeseries_version"
    ] = "1.4.0"
    external_metadata_path.write_text(json.dumps(external_metadata))

    high_quality_id = next(
        cell_id
        for cell_id, record in catalog["bundles"].items()
        if record["baseline"] == "autogluon_high_quality"
    )
    high_quality_metadata_path = (
        tmp_path / catalog["bundles"][high_quality_id]["metadata_path"]
    )
    high_quality_metadata = json.loads(
        high_quality_metadata_path.read_text()
    )
    del high_quality_metadata["predictor"]["checkpoint_compatibility"]
    high_quality_metadata_path.write_text(
        json.dumps(high_quality_metadata)
    )
    with pytest.raises(
        ValueError,
        match="checkpoint compatibility mismatch",
    ):
        pipeline.build_complete_catalog(
            rows,
            manifest,
            replay_catalog,
            tmp_path,
            tmp_path,
            source_revision,
        )
    high_quality_metadata["predictor"]["checkpoint_compatibility"] = (
        pipeline.autogluon_checkpoint_compatibility(
            "autogluon_high_quality"
        )
    )
    high_quality_metadata_path.write_text(
        json.dumps(high_quality_metadata)
    )

    missing = next(iter(catalog["bundles"]))
    bundle_path = tmp_path / catalog["bundles"][missing]["path"]
    bundle_path.unlink()
    with pytest.raises(ValueError, match="Incomplete existing output"):
        pipeline.build_complete_catalog(
            rows,
            manifest,
            replay_catalog,
            tmp_path,
            tmp_path,
            source_revision,
        )
