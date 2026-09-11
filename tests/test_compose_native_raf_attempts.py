from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.compose_native_raf_attempts import (
    AttemptSpec,
    compose_raf_attempts,
    validate_composite_raf_summary,
)
from scripts.generate_native_retrieval_baseline_manifest import build_manifest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_REVISION = "raf-method-revision"
UPSTREAM_COMMIT = "425e2af35797d0d32b63cfc554c7d97b3c54b390"


def write_json(path: Path, document: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def topology_document(
    pid_offset: int,
    *,
    interrupted: bool,
) -> dict:
    worker_pids = {str(index): pid_offset + index for index in range(6)}
    bindings = {
        pid_offset + index: gpu for index, gpu in enumerate(range(2, 8))
    }
    return {
        "schema_version": 1,
        "instance_type": "ml.p5.48xlarge",
        "reserved_gpus_per_host": 8,
        "processes_per_host": 6,
        "worker_gpu_ids": [2, 3, 4, 5, 6, 7],
        "total_gpus": 8,
        "world_size": 6,
        "worker_pids": worker_pids,
        "worker_returncodes": {
            worker: (-15 if interrupted and int(worker) >= 2 else 0)
            for worker in worker_pids
        },
        "expected_bindings": bindings,
        "topology_valid": True,
        "observations": [
            {
                "processes": [
                    {"pid": pid, "physical_gpu": gpu}
                    for pid, gpu in bindings.items()
                ]
            }
        ],
    }


def result_document(
    row: dict,
    *,
    index: int,
    protocol_sha256: str,
    pid: int,
    gpu: int,
) -> dict:
    base = {"wql": 2.0 + index, "mase": 3.0 + index}
    retrieval = {"wql": 1.5 + index, "mase": 2.25 + index}
    return {
        "schema_version": 1,
        "status": "completed",
        "cell_id": row["id"],
        "method": "RAF",
        "base_method": "Chronos",
        "backbone": row["backbone"],
        "dataset": row["dataset"],
        "benchmark": row["benchmark"],
        "context_length": row["context_length"],
        "prediction_length": row["prediction_length"],
        "base": {"metrics": base},
        "retrieval": {"metrics": retrieval},
        "delta": {
            metric: retrieval[metric] - base[metric]
            for metric in ("wql", "mase")
        },
        "protocol_sha256": protocol_sha256,
        "source_revision": SOURCE_REVISION,
        "upstream_commit": UPSTREAM_COMMIT,
        "pid": pid,
        "cuda_visible_devices": str(gpu),
    }


def write_attempt(
    root: Path,
    rows: list[dict],
    *,
    protocol_sha256: str,
    launcher_revision: str,
    pid_offset: int,
    interrupted: bool,
) -> tuple[Path, Path]:
    topology_path = write_json(
        root / "startup_topology.json",
        topology_document(pid_offset, interrupted=interrupted),
    )
    results = []
    for index, row in enumerate(rows):
        worker = index % 6
        result = result_document(
            row,
            index=index,
            protocol_sha256=protocol_sha256,
            pid=pid_offset + worker,
            gpu=2 + worker,
        )
        cell_root = root / "cells" / row["id"]
        write_json(cell_root / "result.json", result)
        write_json(
            cell_root / "status.json",
            {
                "status": "completed",
                "cell_id": row["id"],
                "protocol_sha256": protocol_sha256,
            },
        )
        results.append(result)
    summary = {
        "schema_version": 1,
        "method": "RAF",
        "expected_pairs": 44 if interrupted else len(rows),
        "completed_pairs": len(rows),
        "failed_or_missing_pairs": 44 - len(rows) if interrupted else 0,
        "all_completed": not interrupted,
        "protocol_sha256": protocol_sha256,
        "source_revision": SOURCE_REVISION,
        "launcher_revision": launcher_revision,
        "upstream_commit": UPSTREAM_COMMIT,
        "topology_path": str(topology_path),
        "topology_valid": True,
        "results": results,
        "failures": (
            [{"status": "running"}] if interrupted else []
        ),
    }
    return write_json(root / "summary.json", summary), topology_path


def complete_attempt_inputs(
    tmp_path: Path,
) -> tuple[Path, list[dict], list[AttemptSpec]]:
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_bytes(
        (
            PROJECT_ROOT / "docs/native_retrieval_baseline_protocol.json"
        ).read_bytes()
    )
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol_sha256 = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    rows = [
        row for row in build_manifest(protocol) if row["method"] == "raf"
    ]
    summary_a, topology_a = write_attempt(
        tmp_path / "attempt-a",
        rows[:24],
        protocol_sha256=protocol_sha256,
        launcher_revision="launcher-a",
        pid_offset=1_000,
        interrupted=True,
    )
    summary_b, topology_b = write_attempt(
        tmp_path / "attempt-b",
        rows[24:],
        protocol_sha256=protocol_sha256,
        launcher_revision="launcher-b",
        pid_offset=2_000,
        interrupted=False,
    )
    return protocol_path, rows, [
        AttemptSpec("attempt-a", "interrupted", summary_a, topology_a),
        AttemptSpec("attempt-b", "completed", summary_b, topology_b),
    ]


def test_compose_accepts_interrupted_then_completed_attempt(tmp_path) -> None:
    protocol, _, attempts = complete_attempt_inputs(tmp_path)

    document = compose_raf_attempts(protocol, attempts)
    topology_paths = validate_composite_raf_summary(document, protocol)

    assert document["schema_version"] == 2
    assert document["completed_pairs"] == 44
    assert document["all_completed"] is True
    assert document["launcher_revision"] == "launcher-b"
    assert [
        attempt["accepted_result_count"]
        for attempt in document["composite_attempts"]
    ] == [24, 20]
    assert len(topology_paths) == 2
    assert all(
        {
            "attempt_id",
            "result_path",
            "result_sha256",
            "launcher_revision",
            "topology_sha256",
        }
        <= set(result)
        for result in document["results"]
    )


def test_compose_rejects_duplicate_cell(tmp_path) -> None:
    protocol, rows, attempts = complete_attempt_inputs(tmp_path)
    summary_path = attempts[1].summary_path
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    replacement = result_document(
        rows[0],
        index=99,
        protocol_sha256=summary["protocol_sha256"],
        pid=2_000,
        gpu=2,
    )
    summary["results"][0] = replacement
    write_json(summary_path, summary)
    cell_root = summary_path.parent / "cells" / rows[0]["id"]
    write_json(cell_root / "result.json", replacement)
    write_json(
        cell_root / "status.json",
        {
            "status": "completed",
            "cell_id": rows[0]["id"],
            "protocol_sha256": summary["protocol_sha256"],
        },
    )

    with pytest.raises(ValueError, match="Duplicate RAF cell"):
        compose_raf_attempts(protocol, attempts)


def test_compose_rejects_missing_cell(tmp_path) -> None:
    protocol, _, attempts = complete_attempt_inputs(tmp_path)
    summary_path = attempts[1].summary_path
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["results"].pop()
    summary["completed_pairs"] = len(summary["results"])
    summary["expected_pairs"] = len(summary["results"])
    write_json(summary_path, summary)

    with pytest.raises(ValueError, match="does not exactly cover 44"):
        compose_raf_attempts(protocol, attempts)


def test_composite_validation_rejects_tampered_result_hash(tmp_path) -> None:
    protocol, _, attempts = complete_attempt_inputs(tmp_path)
    document = compose_raf_attempts(protocol, attempts)
    result_path = Path(document["results"][0]["result_path"])
    result_path.write_text(
        result_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="summary or receipt drifted"):
        validate_composite_raf_summary(document, protocol)


def test_composite_validation_accepts_identical_protocol_at_new_path(
    tmp_path,
) -> None:
    protocol, _, attempts = complete_attempt_inputs(tmp_path)
    document = compose_raf_attempts(protocol, attempts)
    protocol_copy = tmp_path / "new-source" / "protocol.json"
    protocol_copy.parent.mkdir(parents=True)
    protocol_copy.write_bytes(protocol.read_bytes())

    validate_composite_raf_summary(document, protocol_copy)


def test_composite_validation_rejects_current_protocol_drift(tmp_path) -> None:
    protocol, _, attempts = complete_attempt_inputs(tmp_path)
    document = compose_raf_attempts(protocol, attempts)
    protocol_copy = tmp_path / "new-source" / "protocol.json"
    protocol_copy.parent.mkdir(parents=True)
    protocol_copy.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="protocol artifact drifted"):
        validate_composite_raf_summary(document, protocol_copy)


def test_compose_rejects_unobserved_result_pid(tmp_path) -> None:
    protocol, _, attempts = complete_attempt_inputs(tmp_path)
    topology_path = attempts[0].topology_path
    topology = json.loads(topology_path.read_text(encoding="utf-8"))
    topology["observations"][0]["processes"] = [
        record
        for record in topology["observations"][0]["processes"]
        if record["pid"] != 1_000
    ]
    write_json(topology_path, topology)

    with pytest.raises(ValueError, match="PID/GPU was not observed"):
        compose_raf_attempts(protocol, attempts)


def test_compose_rejects_undeclared_interruption(tmp_path) -> None:
    protocol, _, attempts = complete_attempt_inputs(tmp_path)
    attempts[0] = AttemptSpec(
        "attempt-a",
        "completed",
        attempts[0].summary_path,
        attempts[0].topology_path,
    )

    with pytest.raises(
        ValueError, match="require explicit interrupted status"
    ):
        compose_raf_attempts(protocol, attempts)


def test_compose_rejects_nonzero_final_returncode(tmp_path) -> None:
    protocol, _, attempts = complete_attempt_inputs(tmp_path)
    topology_path = attempts[1].topology_path
    topology = json.loads(topology_path.read_text(encoding="utf-8"))
    topology["worker_returncodes"]["0"] = -15
    write_json(topology_path, topology)

    with pytest.raises(ValueError, match="require explicit interrupted status"):
        compose_raf_attempts(protocol, attempts)
