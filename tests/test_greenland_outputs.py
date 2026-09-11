import hashlib
import json
from pathlib import Path

import pytest

from scripts import fetch_greenland_outputs
from ts_rag.matrix import load_manifest


RUN_ID = "replay-fixture"
METHOD_REVISION = "d9be338"
SOURCE_REVISION = "a" * 40
LAUNCHER_REVISION = "b" * 40
IMAGE_URI = (
    "<ECR_REGISTRY>/"
    "timeraf-greenland@sha256:" + "c" * 64
)


def _sha256(payload):
    return hashlib.sha256(payload).hexdigest()


class _Paginator:
    def __init__(self, objects):
        self.objects = objects

    def paginate(self, Bucket, Prefix):
        del Bucket
        yield {
            "Contents": [
                {"Key": key, "Size": len(payload)}
                for key, payload in sorted(self.objects.items())
                if key.startswith(Prefix)
            ]
        }


class _S3:
    def __init__(self, objects):
        self.objects = objects

    def get_paginator(self, name):
        assert name == "list_objects_v2"
        return _Paginator(self.objects)

    def head_object(self, Bucket, Key):
        del Bucket
        payload = self.objects[Key]
        return {
            "ContentLength": len(payload),
            "Metadata": {"sha256": _sha256(payload)},
            "ETag": _sha256(payload),
        }

    def download_file(self, bucket, key, filename):
        del bucket
        Path(filename).write_bytes(self.objects[key])


def _spec():
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


def _add_json(objects, prefix, relative, payload):
    objects[prefix + relative] = (
        json.dumps(payload, sort_keys=True) + "\n"
    ).encode()


def _run_objects(rows):
    prefix = (
        "timeraf/greenland/"
        f"runs/{RUN_ID}/"
    )
    objects = {}
    _add_json(
        objects,
        prefix,
        "evidence/accepted_job_spec.json",
        _spec(),
    )
    _add_json(
        objects,
        prefix,
        "evidence/topology_summary.json",
        {
            "topology_gate_passed": True,
            "eight_distinct_bindings_observed": True,
            "all_eight_gpus_utilized": True,
            "expected_gpu_ids": list(range(8)),
            "max_utilization_gpu_percent": {
                index: 20 for index in range(8)
            },
            "binding_evidence": {
                "gpu_ids": list(range(8)),
                "pids": list(range(100, 108)),
            },
        },
    )
    _add_json(
        objects,
        prefix,
        "final_status.json",
        {
            "run_id": RUN_ID,
            "method_revision": METHOD_REVISION,
            "source_revision": SOURCE_REVISION,
            "launcher_revision": LAUNCHER_REVISION,
            "exit_code": 0,
            "status": "succeeded",
            "topology_gate_passed": True,
        },
    )
    _add_json(
        objects,
        prefix,
        "checkpoint_replay/matrix_summary.json",
        {
            "source_revision": SOURCE_REVISION,
            "export_only": True,
            "all_completed": True,
            "counts": {
                "expected": 208,
                "completed": 208,
                "failed": 0,
                "pending": 0,
                "running": 0,
                "incomplete": 0,
            },
        },
    )
    for index, row in enumerate(rows):
        artifact = f"checkpoint_replay/cells/{index:03d}"
        bundle = f"bundle-{row['id']}".encode()
        bundle_hash = _sha256(bundle)
        objects[f"{prefix}{artifact}/prediction_bundle.npz"] = bundle
        _add_json(
            objects,
            prefix,
            f"{artifact}/status.json",
            {
                "cell_id": row["id"],
                "source_revision": SOURCE_REVISION,
                "status": "completed",
                "export_only": True,
                "started_unix": index,
            },
        )
        _add_json(
            objects,
            prefix,
            f"{artifact}/result.json",
            {
                "export_only": True,
                "prediction_bundle": (
                    "/workspace/timeraf/outputs/"
                    f"{artifact}/prediction_bundle.npz"
                ),
                "prediction_bundle_sha256": bundle_hash,
            },
        )
    return objects


def test_fetch_greenland_run_rebases_and_catalogs_all_208_bundles(tmp_path):
    project = tmp_path / "project"
    manifest = project / "docs" / "timefuse_checkpoint_replay_manifest.jsonl"
    manifest.parent.mkdir(parents=True)
    manifest.write_bytes(
        Path("docs/timefuse_checkpoint_replay_manifest.jsonl").read_bytes()
    )
    rows = load_manifest(manifest)
    destination = project / "outputs" / "greenland_runs" / RUN_ID

    receipt = fetch_greenland_outputs.fetch_greenland_run(
        _S3(_run_objects(rows)),
        RUN_ID,
        project,
        destination,
        manifest,
    )

    catalog = json.loads(
        Path(receipt["replay_bundle_catalog"]).read_text(encoding="utf-8")
    )
    assert receipt["topology_gate_passed"]
    assert receipt["object_count"] == 4 + 3 * 208
    assert receipt["downloaded_count"] == receipt["object_count"]
    assert catalog["cataloged_cells"] == 208
    assert catalog["usable_count"] == 208
    assert catalog["missing_count"] == 0
    first = next(iter(catalog["bundles"].values()))
    assert (project / first["path"]).is_file()
    assert str(project / first["path"]).startswith(str(destination))


def test_greenland_output_key_cannot_escape_destination():
    with pytest.raises(ValueError, match="Unsafe"):
        fetch_greenland_outputs._safe_object_relative(
            "timeraf/greenland/runs/replay-fixture/../outside",
            "timeraf/greenland/runs/replay-fixture/",
        )


def test_greenland_output_import_rejects_failed_topology_gate(tmp_path):
    destination = tmp_path / "run"
    evidence = destination / "evidence"
    replay = destination / "checkpoint_replay"
    evidence.mkdir(parents=True)
    replay.mkdir()
    (evidence / "accepted_job_spec.json").write_text(json.dumps(_spec()))
    (evidence / "topology_summary.json").write_text(
        json.dumps(
            {
                "topology_gate_passed": False,
                "eight_distinct_bindings_observed": False,
                "all_eight_gpus_utilized": True,
                "expected_gpu_ids": list(range(8)),
                "binding_evidence": None,
            }
        )
    )
    (destination / "final_status.json").write_text(
        json.dumps(
            {
                "run_id": RUN_ID,
                "method_revision": METHOD_REVISION,
                "source_revision": SOURCE_REVISION,
                "launcher_revision": LAUNCHER_REVISION,
                "exit_code": 0,
                "status": "succeeded",
                "topology_gate_passed": True,
            }
        )
    )
    (replay / "matrix_summary.json").write_text(
        json.dumps(
            {
                "source_revision": SOURCE_REVISION,
                "export_only": True,
                "all_completed": True,
                "counts": {
                    "expected": 208,
                    "completed": 208,
                    "failed": 0,
                    "pending": 0,
                    "running": 0,
                    "incomplete": 0,
                },
            }
        )
    )

    with pytest.raises(ValueError, match="utilization evidence is invalid"):
        fetch_greenland_outputs._validate_downloaded_run(
            destination,
            RUN_ID,
        )


def test_greenland_output_import_rejects_forged_topology_flags(tmp_path):
    rows = load_manifest("docs/timefuse_checkpoint_replay_manifest.jsonl")
    objects = _run_objects(rows)
    prefix = f"timeraf/greenland/runs/{RUN_ID}/"
    topology_key = prefix + "evidence/topology_summary.json"
    topology = json.loads(objects[topology_key])
    topology["binding_evidence"]["pids"] = [100] * 8
    objects[topology_key] = (
        json.dumps(topology, sort_keys=True) + "\n"
    ).encode()

    project = tmp_path / "project"
    manifest = project / "docs" / "timefuse_checkpoint_replay_manifest.jsonl"
    manifest.parent.mkdir(parents=True)
    manifest.write_bytes(
        Path("docs/timefuse_checkpoint_replay_manifest.jsonl").read_bytes()
    )

    with pytest.raises(ValueError, match="eight active distinct A100"):
        fetch_greenland_outputs.fetch_greenland_run(
            _S3(objects),
            RUN_ID,
            project,
            project / "outputs" / "greenland_runs" / RUN_ID,
            manifest,
        )
