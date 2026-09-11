from __future__ import annotations

import json
from pathlib import Path

import pytest

import scripts.run_native_ratd_sharded_eval as sharded_eval
from scripts.aggregate_native_retrieval_baselines import (
    validate_topology as validate_aggregate_topology,
)
from scripts.run_native_ratd_sharded_eval import (
    aggregate_origin_records,
    build_contiguous_shards,
    canonical_json_sha256,
    compare_verification_results,
    encode_rng_state,
    finalize_pair,
    plan_rng_boundaries,
    save_json_atomic,
    sha256_file,
    validate_epoch100_checkpoint,
    validate_topology_document,
    validate_training_topology,
)


class FakeObservedData:
    shape = (1, 321, 264)

    @staticmethod
    def stride():
        return (84_744, 1, 321)


class FakeCuda:
    def __init__(self):
        self.state = 0

    def set_rng_state(self, state, device=None):
        del device
        self.state = int.from_bytes(state, "little")

    def get_rng_state(self, device=None):
        del device
        return self.state.to_bytes(8, "little")

    def synchronize(self, device=None):
        del device


class FakeTorch:
    def __init__(self):
        self.cuda = FakeCuda()
        self.calls = []

    def randn_like(self, observed):
        self.calls.append((tuple(observed.shape), tuple(observed.stride())))
        self.cuda.state += 1
        return object()


def write_json(path: Path, document: dict) -> Path:
    save_json_atomic(path, document)
    return path


def file_record(path: Path) -> dict:
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def test_noise_planner_records_exact_contiguous_boundaries() -> None:
    shards = build_contiguous_shards(5, [2, 4])
    fake = FakeTorch()
    start_state = (10).to_bytes(8, "little")

    planned = plan_rng_boundaries(
        fake,
        FakeObservedData(),
        start_state,
        shards,
        samples=2,
        diffusion_steps=3,
    )

    assert [(row["start_index"], row["end_index"]) for row in planned] == [
        (0, 3),
        (3, 5),
    ]
    assert planned[0]["start_rng"] == encode_rng_state(start_state)
    assert planned[0]["end_rng"] == encode_rng_state((28).to_bytes(8, "little"))
    assert planned[1]["start_rng"] == planned[0]["end_rng"]
    assert planned[1]["end_rng"] == encode_rng_state((40).to_bytes(8, "little"))
    assert planned[0]["rng_call_count"] == 18
    assert planned[1]["rng_call_count"] == 12
    assert fake.calls == [((1, 321, 264), (84_744, 1, 321))] * 30


def test_origin_aggregation_sorts_before_python_float_accumulation() -> None:
    records = [
        {"index": 2, "squared": 1.0, "absolute": 3.0, "points": 1},
        {"index": 0, "squared": 1.0e16, "absolute": 1.0e16, "points": 1},
        {"index": 1, "squared": 1.0, "absolute": 2.0, "points": 1},
    ]

    result = aggregate_origin_records(records, 3)

    expected_squared = 0.0
    expected_absolute = 0.0
    for record in sorted(records, key=lambda row: row["index"]):
        expected_squared += record["squared"]
        expected_absolute += record["absolute"]
    assert [row["index"] for row in result["records"]] == [0, 1, 2]
    assert result["squared"] == expected_squared
    assert result["absolute"] == expected_absolute


@pytest.mark.parametrize(
    ("records", "message"),
    [
        (
            [
                {"index": 0, "squared": 1.0, "absolute": 1.0, "points": 1},
                {"index": 0, "squared": 2.0, "absolute": 2.0, "points": 1},
            ],
            "Duplicate origin",
        ),
        (
            [
                {"index": 0, "squared": 1.0, "absolute": 1.0, "points": 1},
            ],
            "coverage is incomplete",
        ),
    ],
)
def test_origin_aggregation_rejects_duplicate_or_missing(
    records: list[dict], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        aggregate_origin_records(records, 2)


def verification_worker_document(role: str, sample_suffix: str = "") -> dict:
    start_rng = encode_rng_state(b"start")
    first_end = encode_rng_state(b"first")
    final_rng = encode_rng_state(b"final")
    records = [
        {
            "index": 10,
            "sample_sha256": ("a" * 63) + (sample_suffix or "a"),
            "median_sha256": "b" * 64,
            "squared": 1.25,
            "absolute": 2.5,
            "points": 53_928,
            "end_rng": first_end,
        },
        {
            "index": 11,
            "sample_sha256": "c" * 64,
            "median_sha256": "d" * 64,
            "squared": 3.75,
            "absolute": 4.5,
            "points": 53_928,
            "end_rng": final_rng,
        },
    ]
    return {
        "schema_version": 1,
        "job_kind": "native_ratd_exact_equivalence_worker",
        "status": "completed",
        "method": "RATD",
        "role": role,
        "samples": 100,
        "diffusion_steps": 50,
        "start_rng": start_rng,
        "origin_records": records,
        "final_rng": final_rng,
    }


def test_verification_comparison_requires_exact_origin_and_rng_evidence() -> None:
    oracle = verification_worker_document("oracle")
    comparison = verification_worker_document("comparison")

    compared = compare_verification_results(oracle, comparison)

    assert [record["index"] for record in compared["origin_records"]] == [
        10,
        11,
    ]
    comparison["origin_records"][0]["median_sha256"] = "e" * 64
    with pytest.raises(ValueError, match="not byte-identical"):
        compare_verification_results(oracle, comparison)


def topology_document(method: str, pid: int) -> dict:
    gpu_ids = (
        list(range(8)) if method == "RATD" else list(range(1, 8))
    )
    worker_pids = {
        f"{index:04d}": pid + index for index in range(len(gpu_ids))
    }
    return {
        "schema_version": 2,
        "job_kind": "native_ratd_origin_sharded_eval_topology",
        "method": method,
        "instance_type": "ml.p5.48xlarge",
        "reserved_gpus_per_host": 8,
        "processes_per_host": len(gpu_ids),
        "worker_gpu_ids": gpu_ids,
        "world_size": len(gpu_ids),
        "inactive_reserved_gpus": 8 - len(gpu_ids),
        "worker_pids": worker_pids,
        "worker_returncodes": {key: 0 for key in worker_pids},
        "expected_bindings": {
            str(pid + index): worker_gpu
            for index, worker_gpu in enumerate(gpu_ids)
        },
        "observed_bindings": {
            str(pid + index): [worker_gpu]
            for index, worker_gpu in enumerate(gpu_ids)
        },
        "topology_valid": True,
    }


def test_formal_evaluation_topology_requires_every_worker() -> None:
    topology = topology_document("RATD", 100)
    topology["worker_pids"].pop("0007")
    topology["worker_returncodes"].pop("0007")
    topology["expected_bindings"].pop("107")
    topology["observed_bindings"].pop("107")
    topology["processes_per_host"] = 7
    topology["worker_gpu_ids"] = list(range(7))
    topology["world_size"] = 7
    topology["inactive_reserved_gpus"] = 1

    with pytest.raises(
        ValueError, match="worker completion evidence is invalid"
    ):
        validate_topology_document(topology)


def test_training_topology_requires_explicit_post_epoch100_policy(
    tmp_path: Path,
) -> None:
    topology = write_json(
        tmp_path / "training.json",
        {
            "instance_type": "ml.p5.48xlarge",
            "reserved_gpus_per_host": 8,
            "processes_per_host": 2,
            "worker_gpu_ids": [0, 1],
            "worker_pids": {"ratd": 11, "csdi": 12},
            "worker_returncodes": {"ratd": -15, "csdi": -15},
            "expected_bindings": {"11": 0, "12": 1},
            "observed_bindings": {"11": [0], "12": [1]},
            "topology_valid": True,
        },
    )
    checkpoint_evidence = {
        method: {"method": method.upper(), "epoch": 100} for method in ("ratd", "csdi")
    }

    with pytest.raises(ValueError, match="explicit post-epoch100"):
        validate_training_topology(
            topology,
            exit_policy="require-zero",
            checkpoint_evidence=checkpoint_evidence,
        )
    _, termination = validate_training_topology(
        topology,
        exit_policy="intentional-post-epoch100-termination",
        checkpoint_evidence=checkpoint_evidence,
    )
    assert termination["accepted"]
    checkpoint_evidence["csdi"]["epoch"] = 99
    with pytest.raises(ValueError, match="Both training checkpoints"):
        validate_training_topology(
            topology,
            exit_policy="intentional-post-epoch100-termination",
            checkpoint_evidence=checkpoint_evidence,
        )


def test_checkpoint_gate_rejects_nonfinal_epoch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    result = {
        "source_revision": "frozen-method",
        "upstream_commit": "released-upstream",
        "protocol_sha256": "a" * 64,
        "reference_map_sha256": "b" * 64,
    }
    monkeypatch.setattr(
        sharded_eval,
        "load_checkpoint_header",
        lambda _path: {
            "epoch": 99,
            "identity": {
                "method": "RATD",
                "source_revision": "frozen-method",
                "upstream_commit": "released-upstream",
                "protocol_sha256": "a" * 64,
                "reference_map_sha256": "b" * 64,
                "seed": 1,
                "epochs": 100,
                "diffusion_steps": 50,
                "samples": 100,
            },
        },
    )

    with pytest.raises(ValueError, match="not epoch 100"):
        validate_epoch100_checkpoint(file_record(checkpoint), result, "RATD")


def method_bundle(
    tmp_path: Path,
    method: str,
    metrics: dict,
    pid: int,
    shared: dict,
) -> tuple[Path, Path]:
    root = tmp_path / method.lower()
    topology = write_json(
        root / "topology.json", topology_document(method, pid)
    )
    plan = write_json(root / "plan.json", {"method": method})
    checkpoint = root / "checkpoint.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(f"{method}-checkpoint".encode())
    runner = root / "runner.py"
    runner.write_text("# frozen runner\n", encoding="utf-8")
    sharded_runner = root / "sharded_runner.py"
    sharded_runner.write_text("# sharded launcher\n", encoding="utf-8")
    upstream_main_model = root / "main_model.py"
    upstream_main_model.write_text("# released model\n", encoding="utf-8")
    upstream_diff_models = root / "diff_models.py"
    upstream_diff_models.write_text("# released diffusion model\n", encoding="utf-8")
    protocol = write_json(root / "protocol.json", {"frozen": True})
    config = root / "config.yaml"
    config.write_text("frozen: true\n", encoding="utf-8")
    data = root / "electricity.csv"
    data.write_text("date,value\n", encoding="utf-8")
    reference_map = Path(shared["reference_map"])
    reference_metadata = Path(shared["reference_metadata"])
    gpu_ids = (
        list(range(8)) if method == "RATD" else list(range(1, 8))
    )
    worker_records = []
    for index, worker_gpu in enumerate(gpu_ids):
        worker_pid = pid + index
        shard_id = f"{index:04d}"
        worker_result = write_json(
            root / f"worker-{shard_id}.json",
            {
                "shard_id": shard_id,
                "pid": worker_pid,
                "physical_gpu": worker_gpu,
                "started_unix": 1.0,
                "finished_unix": 2.0,
                "boundary_sample_sha256": {
                    "first": {"origin_index": 0, "sha256": "a" * 64},
                    "last": {"origin_index": 5092, "sha256": "b" * 64},
                },
                "end_rng_sha256": "c" * 64,
            },
        )
        worker_records.append(
            {
                **file_record(worker_result),
                "shard_id": shard_id,
                "pid": worker_pid,
                "physical_gpu": worker_gpu,
                "started_unix": 1.0,
                "finished_unix": 2.0,
                "boundary_sample_sha256": json.loads(
                    worker_result.read_text()
                )["boundary_sample_sha256"],
                "end_rng_sha256": "c" * 64,
            }
        )
    result = {
        "schema_version": 2,
        "status": "completed",
        "method": method,
        "evaluation_mode": "origin-sharded-exact-rng",
        "checkpoint_epoch": 100,
        "test_origins": 5_093,
        "test_points": 274_655_304,
        "samples": 100,
        "diffusion_steps": 50,
        "metrics": metrics,
        "source_revision": "frozen-method",
        "launcher_revision": "sharded-launcher",
        "upstream_commit": "released-upstream",
        "protocol_sha256": sha256_file(protocol),
        "reference_map_sha256": sha256_file(reference_map),
        "checkpoint_sha256": sha256_file(checkpoint),
        "seed": 1,
        "epochs": 100,
    }
    result_path = write_json(root / "result.json", result)
    receipt_payload = {
        "schema_version": 1,
        "job_kind": "native_ratd_origin_sharded_eval_receipt",
        "method": method,
        "created_unix": 3.0,
        "plan": file_record(plan),
        "topology": file_record(topology),
        "method_result": file_record(result_path),
        "checkpoint": file_record(checkpoint),
        "sharded_runner": file_record(sharded_runner),
        "runner": file_record(runner),
        "upstream_main_model": file_record(upstream_main_model),
        "upstream_diff_models": file_record(upstream_diff_models),
        "protocol": file_record(protocol),
        "config": file_record(config),
        "data": file_record(data),
        "reference_map": file_record(reference_map),
        "reference_metadata": file_record(reference_metadata),
        "worker_results": worker_records,
    }
    receipt = {
        **receipt_payload,
        "receipt_sha256": canonical_json_sha256(receipt_payload),
    }
    receipt_path = write_json(root / "receipt.json", receipt)
    return result_path, receipt_path


def test_finalizer_emits_metric_compatible_schema_v2_topology(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference_map = tmp_path / "reference_map.npz"
    reference_map.write_bytes(b"reference-map")
    reference_metadata = {
        "schema_version": 1,
        "seed": 1,
        "context_length": 96,
        "prediction_length": 168,
        "top_k": 3,
        "train_end": 18_149,
        "query_counts": {
            "train": 17_886,
            "validation": 2_463,
            "test": 5_093,
        },
        "audits": {
            split: {
                "latest_reference_future_end": 18_149,
                "train_end": 18_149,
            }
            for split in ("train", "validation", "test")
        },
        "reference_map": str(reference_map),
        "reference_map_sha256": sha256_file(reference_map),
    }
    metadata_path = write_json(tmp_path / "reference_map.json", reference_metadata)
    shared = {
        "reference_map": str(reference_map),
        "reference_metadata": str(metadata_path),
    }
    ratd_result, ratd_receipt = method_bundle(
        tmp_path,
        "RATD",
        {"rmse": 1.5, "mae": 0.75},
        101,
        shared,
    )
    csdi_result, csdi_receipt = method_bundle(
        tmp_path,
        "CSDI",
        {"rmse": 2.0, "mae": 1.0},
        102,
        shared,
    )
    training_topology = write_json(
        tmp_path / "training_topology.json",
        {
            "instance_type": "ml.p5.48xlarge",
            "reserved_gpus_per_host": 8,
            "processes_per_host": 2,
            "worker_gpu_ids": [0, 1],
            "worker_pids": {"ratd": 11, "csdi": 12},
            "worker_returncodes": {"ratd": -15, "csdi": -15},
            "expected_bindings": {"11": 0, "12": 1},
            "observed_bindings": {"11": [0], "12": [1]},
            "topology_valid": True,
        },
    )
    topology_output = tmp_path / "combined_topology.json"
    summary_output = tmp_path / "summary.json"
    receipt_output = tmp_path / "finalizer_receipt.json"
    checkpoint_identities = {
        "ratd": {
            "method": "RATD",
            "source_revision": "frozen-method",
            "upstream_commit": "released-upstream",
            "protocol_sha256": json.loads(ratd_result.read_text())["protocol_sha256"],
            "reference_map_sha256": sha256_file(reference_map),
            "seed": 1,
            "epochs": 100,
            "diffusion_steps": 50,
            "samples": 100,
        },
        "csdi": {
            "method": "CSDI",
            "source_revision": "frozen-method",
            "upstream_commit": "released-upstream",
            "protocol_sha256": json.loads(csdi_result.read_text())["protocol_sha256"],
            "reference_map_sha256": sha256_file(reference_map),
            "seed": 1,
            "epochs": 100,
            "diffusion_steps": 50,
            "samples": 100,
        },
    }

    def checkpoint_header(path: Path) -> dict:
        method = path.parent.name
        return {
            "epoch": 100,
            "identity": checkpoint_identities[method],
        }

    monkeypatch.setattr(sharded_eval, "load_checkpoint_header", checkpoint_header)

    summary = finalize_pair(
        ratd_result_path=ratd_result,
        ratd_receipt_path=ratd_receipt,
        csdi_result_path=csdi_result,
        csdi_receipt_path=csdi_receipt,
        reference_metadata_path=metadata_path,
        training_topology_path=training_topology,
        topology_output=topology_output,
        summary_output=summary_output,
        receipt_output=receipt_output,
        training_eval_exit_policy=("intentional-post-epoch100-termination"),
    )

    assert summary["pair"] == {
        "cell_id": "ratd/electricity/c96/h168",
        "base": {"rmse": 2.0, "mae": 1.0},
        "retrieval": {"rmse": 1.5, "mae": 0.75},
        "delta": {"rmse": -0.5, "mae": -0.25},
    }
    assert summary["results"]["ratd"]["test_origins"] == 5_093
    assert summary["aggregator_compatibility"]["requires_topology_schema_v2_support"]
    combined = json.loads(topology_output.read_text())
    assert set(combined["evaluation"]) == {"ratd", "csdi"}
    assert combined["training"]["worker_returncodes"] == {
        "ratd": -15,
        "csdi": -15,
    }
    assert combined["training"]["termination"]["classification"] == (
        "original-serial-evaluation-terminated-after-epoch100"
    )
    assert set(combined["training"]["termination"]["checkpoint_evidence"]) == {
        "ratd",
        "csdi",
    }
    finalizer_receipt = json.loads(receipt_output.read_text())
    payload = {
        key: value
        for key, value in finalizer_receipt.items()
        if key != "receipt_sha256"
    }
    assert finalizer_receipt["receipt_sha256"] == canonical_json_sha256(payload)
    assert (
        validate_aggregate_topology("ratd", summary_output, summary)
        == topology_output
    )
