from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.aggregate_native_retrieval_baselines import (
    EXPECTED,
    EXPECTED_TOPOLOGY,
    aggregate,
    sha256_file,
    validate_ratd_equivalence_evidence,
)
from scripts.compose_native_raf_attempts import AttemptSpec, compose_raf_attempts
from scripts.generate_native_retrieval_baseline_manifest import build_manifest
from scripts.run_native_ratd_sharded_eval import (
    canonical_json_sha256,
    encode_rng_state,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def write_json(path: Path, document: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def file_record(path: Path) -> dict:
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def equivalence_worker(method: str, role: str, pid: int, gpu: int) -> dict:
    start_rng = encode_rng_state(b"start")
    first_rng = encode_rng_state(b"first")
    final_rng = encode_rng_state(b"final")
    return {
        "schema_version": 1,
        "job_kind": "native_ratd_exact_equivalence_worker",
        "status": "completed",
        "method": method,
        "role": role,
        "pid": pid,
        "physical_gpu": gpu,
        "samples": 100,
        "diffusion_steps": 50,
        "start_rng": start_rng,
        "origin_records": [
            {
                "index": 0,
                "sample_sha256": "a" * 64,
                "median_sha256": "b" * 64,
                "squared": 1.0,
                "absolute": 2.0,
                "points": 3,
                "end_rng": first_rng,
            },
            {
                "index": 1,
                "sample_sha256": "c" * 64,
                "median_sha256": "d" * 64,
                "squared": 4.0,
                "absolute": 5.0,
                "points": 6,
                "end_rng": final_rng,
            },
        ],
        "final_rng": final_rng,
    }


def install_equivalence_verification(
    root: Path,
    method: str,
    oracle_gpu: int,
    comparison_gpu: int,
) -> Path:
    oracle = equivalence_worker(
        method, "oracle", 10_000 + oracle_gpu, oracle_gpu
    )
    comparison = equivalence_worker(
        method,
        "comparison",
        20_000 + comparison_gpu,
        comparison_gpu,
    )
    oracle_path = write_json(root / "oracle.json", oracle)
    comparison_path = write_json(root / "comparison.json", comparison)
    payload = {
        "schema_version": 1,
        "job_kind": "native_ratd_exact_cross_gpu_verification",
        "status": "verified",
        "method": method,
        "strict_byte_equivalence": True,
        "verification_range": {
            "start_index": 0,
            "end_index": 2,
            "origin_count": 2,
        },
        "samples": 100,
        "diffusion_steps": 50,
        "identity": {
            "source_revision": "ratd-source",
            "launcher_revision": "ratd-launcher",
            "upstream_commit": "ratd-upstream",
            "protocol": {"sha256": "a" * 64},
        },
        "start_rng": oracle["start_rng"],
        "expected_final_rng": oracle["final_rng"],
        "oracle": {
            "role": "oracle",
            "pid": oracle["pid"],
            "physical_gpu": oracle_gpu,
            "returncode": 0,
            "observed_gpus": [oracle_gpu],
            "result": file_record(oracle_path),
            "origin_records": oracle["origin_records"],
            "start_rng": oracle["start_rng"],
            "final_rng": oracle["final_rng"],
        },
        "comparison": {
            "role": "comparison",
            "pid": comparison["pid"],
            "physical_gpu": comparison_gpu,
            "returncode": 0,
            "observed_gpus": [comparison_gpu],
            "result": file_record(comparison_path),
            "origin_records": comparison["origin_records"],
            "start_rng": comparison["start_rng"],
            "final_rng": comparison["final_rng"],
        },
    }
    return write_json(
        root / "verification.json",
        {
            **payload,
            "verification_sha256": canonical_json_sha256(payload),
        },
    )


def topology_document(
    *,
    worker_gpu_ids: list[int],
    pid_offset: int,
    topology_valid: bool | None = None,
) -> dict:
    worker_pids = {
        str(worker): pid_offset + worker
        for worker in range(len(worker_gpu_ids))
    }
    expected_bindings = {
        pid_offset + worker: gpu
        for worker, gpu in enumerate(worker_gpu_ids)
    }
    document = {
        "instance_type": "ml.p5.48xlarge",
        "reserved_gpus_per_host": 8,
        "processes_per_host": len(worker_gpu_ids),
        "worker_gpu_ids": worker_gpu_ids,
        "total_gpus": 8,
        "world_size": len(worker_gpu_ids),
        "worker_pids": worker_pids,
        "worker_returncodes": {
            worker: 0 for worker in worker_pids
        },
        "expected_bindings": expected_bindings,
        "observations": [
            {
                "processes": [
                    {
                        "pid": pid,
                        "physical_gpu": gpu,
                    }
                    for pid, gpu in expected_bindings.items()
                ]
            }
        ],
    }
    if topology_valid is not None:
        document["topology_valid"] = topology_valid
    return document


def standard_summary(
    method: str,
    rows: list[dict],
    protocol_sha256: str,
    upstream_commit: str,
    base_method: str,
) -> dict:
    results = []
    method_key = method.lower().replace("-", "_")
    for index, row in enumerate(rows):
        base = {
            metric: 2.0 + index for metric in EXPECTED[method_key]["metrics"]
        }
        retrieval = {
            metric: 1.5 + index
            for metric in EXPECTED[method_key]["metrics"]
        }
        result = {
            "status": "completed",
            "cell_id": row["id"],
            "method": method,
            "base_method": base_method,
            "backbone": row["backbone"],
            "dataset": row["dataset"],
            "context_length": row["context_length"],
            "prediction_length": row["prediction_length"],
            "base": {"metrics": base},
            "retrieval": {"metrics": retrieval},
            "delta": {
                metric: retrieval[metric] - base[metric]
                for metric in EXPECTED[method_key]["metrics"]
            },
            "source_revision": f"{method_key}-source-revision",
            "upstream_commit": upstream_commit,
        }
        if method_key == "ts_rag":
            result["retrieval_index_audit"] = {
                "all_reference_futures_end_in_training": True
            }
        results.append(result)
    summary = {
        "method": method,
        "expected_pairs": len(rows),
        "completed_pairs": len(rows),
        "failed_or_missing_pairs": 0,
        "all_completed": True,
        "protocol_sha256": protocol_sha256,
        "source_revision": f"{method_key}-source-revision",
        "upstream_commit": upstream_commit,
        "results": results,
        "failures": [],
    }
    if method_key == "raf":
        summary["launcher_revision"] = "raf-launcher-revision"
    return summary


def complete_inputs(
    tmp_path: Path,
) -> tuple[Path, dict[str, Path], Path]:
    protocol = tmp_path / "protocol.json"
    protocol.write_bytes(
        (
            PROJECT_ROOT / "docs/native_retrieval_baseline_protocol.json"
        ).read_bytes()
    )
    protocol_document = json.loads(protocol.read_text(encoding="utf-8"))
    protocol_sha256 = sha256_file(protocol)
    manifest = build_manifest(protocol_document)
    rows = {
        method: [row for row in manifest if row["method"] == method]
        for method in ("raf", "ts_rag", "ratd")
    }
    raf = standard_summary(
        "RAF",
        rows["raf"],
        protocol_sha256,
        protocol_document["sources"]["raf"]["commit"],
        protocol_document["systems"]["raf"]["base_system"],
    )
    ts_rag = standard_summary(
        "TS-RAG",
        rows["ts_rag"],
        protocol_sha256,
        protocol_document["sources"]["ts_rag"]["commit"],
        protocol_document["systems"]["ts_rag"]["base_system"],
    )
    asset_manifest = json.loads(
        (
            PROJECT_ROOT / "docs/native_ts_rag_assets.json"
        ).read_text(encoding="utf-8")
    )
    asset_verification = write_json(
        tmp_path / "ts-rag-assets.json",
        {
            "manifest": str(
                PROJECT_ROOT / "docs/native_ts_rag_assets.json"
            ),
            "files": [
                {
                    "path": str(Path(asset_manifest[root_key]) / row["path"]),
                    "size_bytes": row["size_bytes"],
                    "sha256": row["sha256"],
                }
                for root_key, files_key in (
                    ("checkpoint_root", "checkpoint_files"),
                    ("data_root", "data_files"),
                )
                for row in asset_manifest[files_key]
            ],
        },
    )
    ts_rag["asset_verification_path"] = str(asset_verification)

    reference_map = tmp_path / "ratd-reference-map.npz"
    reference_map.write_bytes(b"reference-map")
    reference_metadata = {
        "schema_version": 1,
        "seed": 1,
        "context_length": 96,
        "prediction_length": 168,
        "top_k": 3,
        "train_end": 18_412,
        "query_counts": {
            "train": 17_886,
            "validation": 2_463,
            "test": 5_093,
        },
        "audits": {
            split: {
                "minimum_reference_start": 0,
                "maximum_reference_start": 18_148,
                "latest_reference_future_end": 18_412,
                "train_end": 18_412,
            }
            for split in ("train", "validation", "test")
        },
        "reference_map": str(reference_map),
        "reference_map_sha256": sha256_file(reference_map),
    }
    write_json(reference_map.with_suffix(".json"), reference_metadata)
    base_metrics = {"rmse": 2.0, "mae": 1.0}
    retrieval_metrics = {"rmse": 1.5, "mae": 0.75}
    ratd = {
        "method": "RATD",
        "base_method": "CSDI",
        "expected_pairs": 1,
        "completed_pairs": 1,
        "failed_or_missing_pairs": 0,
        "all_completed": True,
        "protocol_sha256": protocol_sha256,
        "source_revision": "ratd-source-revision",
        "topology_valid": True,
        "reference_metadata": reference_metadata,
        "failures": [],
        "results": {
            "ratd": {
                "status": "completed",
                "test_origins": 5_093,
                "protocol_sha256": protocol_sha256,
                "metrics": retrieval_metrics,
                "source_revision": "ratd-source-revision",
                "upstream_commit": protocol_document["sources"]["ratd"][
                    "commit"
                ],
            },
            "csdi": {
                "status": "completed",
                "test_origins": 5_093,
                "protocol_sha256": protocol_sha256,
                "metrics": base_metrics,
                "source_revision": "ratd-source-revision",
                "upstream_commit": protocol_document["sources"]["ratd"][
                    "commit"
                ],
            },
        },
        "pair": {
            "cell_id": rows["ratd"][0]["id"],
            "base": base_metrics,
            "retrieval": retrieval_metrics,
            "delta": {"rmse": -0.5, "mae": -0.25},
        },
    }
    ratd["upstream_commit"] = protocol_document["sources"]["ratd"]["commit"]
    topologies = {
        "raf": write_json(
            tmp_path / "raf-topology.json",
            topology_document(
                worker_gpu_ids=[2, 3, 4, 5, 6, 7],
                pid_offset=1_000,
                topology_valid=True,
            ),
        ),
        "ts_rag": write_json(
            tmp_path / "ts-rag-topology.json",
            topology_document(
                worker_gpu_ids=[0, 1, 2, 3, 4, 5, 6],
                pid_offset=2_000,
            ),
        ),
        "ratd": write_json(
            tmp_path / "ratd-topology.json",
            topology_document(
                worker_gpu_ids=[0, 1],
                pid_offset=3_000,
                topology_valid=True,
            ),
        ),
    }
    raf["topology_path"] = str(topologies["raf"])
    ts_rag["topology_path"] = str(topologies["ts_rag"])
    ratd["topology_path"] = str(topologies["ratd"])
    paths = {
        "raf": write_json(tmp_path / "raf" / "summary.json", raf),
        "ts_rag": write_json(tmp_path / "ts-rag" / "summary.json", ts_rag),
        "ratd": write_json(tmp_path / "ratd" / "summary.json", ratd),
    }
    execution_record = write_json(
        tmp_path / "execution-record.json",
        {
            "schema_version": 1,
            "protocol_sha256": protocol_sha256,
            "platform": {
                "instance_type": "ml.p5.48xlarge",
                "reserved_gpus_per_host": 8,
                "project_root": str(tmp_path),
            },
            "runs": {
                method: {
                    "processes_per_host": topology["processes_per_host"],
                    "worker_gpu_ids": topology["worker_gpu_ids"],
                    "expected_pairs": EXPECTED[method]["pairs"],
                    "output_root": paths[method].parent.name,
                    "method_source_revision": json.loads(
                        paths[method].read_text(encoding="utf-8")
                    )["source_revision"],
                    "upstream_commit": protocol_document["sources"][method][
                        "commit"
                    ],
                    "summary_sha256": sha256_file(paths[method]),
                    "topology_sha256": sha256_file(
                        Path(
                            json.loads(
                                paths[method].read_text(encoding="utf-8")
                            )["topology_path"]
                        )
                    ),
                    **(
                        {"launcher_revision": "raf-launcher-revision"}
                        if method == "raf"
                        else {}
                    ),
                    **(
                        {
                            "asset_verification_sha256": sha256_file(
                                asset_verification
                            )
                        }
                        if method == "ts_rag"
                        else {}
                    ),
                }
                for method, topology in EXPECTED_TOPOLOGY.items()
            },
        },
    )
    return protocol, paths, execution_record


def install_composite_raf(
    protocol_path: Path,
    paths: dict[str, Path],
    execution_record_path: Path,
    tmp_path: Path,
) -> dict:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol_sha256 = sha256_file(protocol_path)
    upstream_commit = protocol["sources"]["raf"]["commit"]
    rows = [
        row for row in build_manifest(protocol) if row["method"] == "raf"
    ]
    specs = []
    for attempt_index, (attempt_id, attempt_rows, interrupted) in enumerate(
        (
            ("attempt-a", rows[:24], True),
            ("attempt-b", rows[24:], False),
        )
    ):
        root = tmp_path / attempt_id
        pid_offset = 4_000 + attempt_index * 100
        topology = topology_document(
            worker_gpu_ids=[2, 3, 4, 5, 6, 7],
            pid_offset=pid_offset,
            topology_valid=True,
        )
        if interrupted:
            topology["worker_returncodes"]["2"] = -15
        topology_path = write_json(root / "startup_topology.json", topology)
        results = []
        for index, row in enumerate(attempt_rows):
            worker = index % 6
            base = {"wql": 2.0 + index, "mase": 3.0 + index}
            retrieval = {"wql": 1.0 + index, "mase": 2.5 + index}
            result = {
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
                "source_revision": "raf-source-revision",
                "upstream_commit": upstream_commit,
                "pid": pid_offset + worker,
                "cuda_visible_devices": str(2 + worker),
            }
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
            "expected_pairs": 44 if interrupted else len(attempt_rows),
            "completed_pairs": len(attempt_rows),
            "failed_or_missing_pairs": (
                44 - len(attempt_rows) if interrupted else 0
            ),
            "all_completed": not interrupted,
            "protocol_sha256": protocol_sha256,
            "source_revision": "raf-source-revision",
            "launcher_revision": f"launcher-{attempt_index}",
            "upstream_commit": upstream_commit,
            "topology_path": str(topology_path),
            "topology_valid": True,
            "results": results,
            "failures": [{"status": "running"}] if interrupted else [],
        }
        summary_path = write_json(root / "summary.json", summary)
        specs.append(
            AttemptSpec(
                attempt_id,
                "interrupted" if interrupted else "completed",
                summary_path,
                topology_path,
            )
        )

    composite = compose_raf_attempts(protocol_path, specs)
    composite_path = write_json(
        tmp_path / "raf-composite" / "summary.json", composite
    )
    paths["raf"] = composite_path
    execution_record = json.loads(
        execution_record_path.read_text(encoding="utf-8")
    )
    execution_record["runs"]["raf"].update(
        {
            "output_root": "raf-composite",
            "launcher_revision": "launcher-1",
            "upstream_commit": upstream_commit,
            "summary_sha256": sha256_file(composite_path),
            "topology_sha256": composite["composite_attempts"][-1][
                "topology_sha256"
            ],
        }
    )
    write_json(execution_record_path, execution_record)
    return composite


def test_aggregate_requires_all_52_pairs(tmp_path) -> None:
    protocol, paths, execution_record = complete_inputs(tmp_path)

    document = aggregate(protocol, paths, execution_record)

    assert document["schema_version"] == 2
    assert document["expected_pairs"] == 52
    assert document["completed_pairs"] == 52
    assert document["all_completed"] is True
    assert len(document["pairs"]) == 52
    assert document["execution_record"]["sha256"] == sha256_file(
        execution_record
    )
    assert (
        document["source_summaries"]["ts_rag"][
            "asset_verification_sha256"
        ]
        == sha256_file(
            Path(
                json.loads(
                    paths["ts_rag"].read_text(encoding="utf-8")
                )["asset_verification_path"]
            )
        )
    )
    assert (
        document["source_summaries"]["ratd"]["reference_map_sha256"]
        == json.loads(paths["ratd"].read_text(encoding="utf-8"))[
            "reference_metadata"
        ]["reference_map_sha256"]
    )


def test_aggregate_accepts_composite_raf_provenance(tmp_path) -> None:
    protocol, paths, execution_record = complete_inputs(tmp_path)
    install_composite_raf(protocol, paths, execution_record, tmp_path)

    document = aggregate(protocol, paths, execution_record)

    raf_source = document["source_summaries"]["raf"]
    assert [
        attempt["accepted_result_count"]
        for attempt in raf_source["composite_attempts"]
    ] == [24, 20]
    assert raf_source["composite_receipt"]["result_count"] == 44


def test_aggregate_rejects_tampered_composite_result(tmp_path) -> None:
    protocol, paths, execution_record = complete_inputs(tmp_path)
    composite = install_composite_raf(
        protocol, paths, execution_record, tmp_path
    )
    result_path = Path(composite["results"][0]["result_path"])
    result_path.write_text(
        result_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="summary or receipt drifted"):
        aggregate(protocol, paths, execution_record)


def test_aggregate_rejects_incomplete_summary(tmp_path) -> None:
    protocol, paths, execution_record = complete_inputs(tmp_path)
    summary = json.loads(paths["ts_rag"].read_text(encoding="utf-8"))
    summary["completed_pairs"] = 6
    write_json(paths["ts_rag"], summary)

    with pytest.raises(ValueError, match="incomplete pair count"):
        aggregate(protocol, paths, execution_record)


def test_aggregate_rejects_incorrect_delta(tmp_path) -> None:
    protocol, paths, execution_record = complete_inputs(tmp_path)
    summary = json.loads(paths["raf"].read_text(encoding="utf-8"))
    summary["results"][0]["delta"]["wql"] = 0.0
    write_json(paths["raf"], summary)

    with pytest.raises(ValueError, match="delta mismatch"):
        aggregate(protocol, paths, execution_record)


def test_aggregate_rejects_nonfinite_metric(tmp_path) -> None:
    protocol, paths, execution_record = complete_inputs(tmp_path)
    summary = json.loads(paths["ts_rag"].read_text(encoding="utf-8"))
    summary["results"][0]["retrieval"]["metrics"]["mse"] = float("nan")
    summary["results"][0]["delta"]["mse"] = float("nan")
    write_json(paths["ts_rag"], summary)

    with pytest.raises(ValueError, match="non-finite"):
        aggregate(protocol, paths, execution_record)


def test_aggregate_rejects_incorrect_gpu_binding(tmp_path) -> None:
    protocol, paths, execution_record = complete_inputs(tmp_path)
    summary = json.loads(paths["raf"].read_text(encoding="utf-8"))
    topology_path = Path(summary["topology_path"])
    topology = json.loads(topology_path.read_text(encoding="utf-8"))
    topology["observations"][0]["processes"][0]["physical_gpu"] = 1
    write_json(topology_path, topology)

    with pytest.raises(ValueError, match="observed worker bindings"):
        aggregate(protocol, paths, execution_record)


@pytest.mark.parametrize(
    ("field", "value"),
    (("dataset", "wrong-dataset"), ("context_length", 999)),
)
def test_aggregate_rejects_manifest_field_drift(
    tmp_path, field: str, value
) -> None:
    protocol, paths, execution_record = complete_inputs(tmp_path)
    summary = json.loads(paths["raf"].read_text(encoding="utf-8"))
    summary["results"][0][field] = value
    write_json(paths["raf"], summary)

    with pytest.raises(
        ValueError, match=rf"canonical manifest {field} mismatch"
    ):
        aggregate(protocol, paths, execution_record)


def test_aggregate_rejects_tampered_ratd_pair(tmp_path) -> None:
    protocol, paths, execution_record = complete_inputs(tmp_path)
    summary = json.loads(paths["ratd"].read_text(encoding="utf-8"))
    summary["pair"]["base"]["rmse"] = 3.0
    summary["pair"]["delta"]["rmse"] = -1.5
    write_json(paths["ratd"], summary)

    with pytest.raises(ValueError, match="do not match CSDI"):
        aggregate(protocol, paths, execution_record)


def test_aggregate_rejects_ts_rag_reference_leakage(tmp_path) -> None:
    protocol, paths, execution_record = complete_inputs(tmp_path)
    summary = json.loads(paths["ts_rag"].read_text(encoding="utf-8"))
    summary["results"][0]["retrieval_index_audit"][
        "all_reference_futures_end_in_training"
    ] = False
    write_json(paths["ts_rag"], summary)

    with pytest.raises(ValueError, match="crosses training boundary"):
        aggregate(protocol, paths, execution_record)


def test_aggregate_rejects_ratd_reference_leakage(tmp_path) -> None:
    protocol, paths, execution_record = complete_inputs(tmp_path)
    summary = json.loads(paths["ratd"].read_text(encoding="utf-8"))
    summary["reference_metadata"]["audits"]["test"][
        "latest_reference_future_end"
    ] += 1
    write_json(paths["ratd"], summary)

    with pytest.raises(ValueError, match="crosses training boundary"):
        aggregate(protocol, paths, execution_record)


def test_aggregate_rejects_execution_topology_drift(tmp_path) -> None:
    protocol, paths, execution_record = complete_inputs(tmp_path)
    record = json.loads(execution_record.read_text(encoding="utf-8"))
    record["runs"]["raf"]["processes_per_host"] = 5
    write_json(execution_record, record)

    with pytest.raises(ValueError, match="execution record raf topology"):
        aggregate(protocol, paths, execution_record)


def test_aggregate_rejects_unapproved_summary_path(tmp_path) -> None:
    protocol, paths, execution_record = complete_inputs(tmp_path)
    unapproved = tmp_path / "unapproved" / "summary.json"
    write_json(
        unapproved,
        json.loads(paths["ts_rag"].read_text(encoding="utf-8")),
    )
    paths["ts_rag"] = unapproved

    with pytest.raises(ValueError, match="summary path is not the approved run"):
        aggregate(protocol, paths, execution_record)


def test_aggregate_rejects_unapproved_source_revision(tmp_path) -> None:
    protocol, paths, execution_record = complete_inputs(tmp_path)
    summary = json.loads(paths["raf"].read_text(encoding="utf-8"))
    summary["source_revision"] = "unapproved-source"
    write_json(paths["raf"], summary)

    with pytest.raises(ValueError, match="source revision is not approved"):
        aggregate(protocol, paths, execution_record)


def test_aggregate_rejects_unapproved_result_upstream(tmp_path) -> None:
    protocol, paths, execution_record = complete_inputs(tmp_path)
    summary = json.loads(paths["ratd"].read_text(encoding="utf-8"))
    summary["results"]["ratd"]["upstream_commit"] = "unapproved-upstream"
    write_json(paths["ratd"], summary)

    with pytest.raises(ValueError, match="result upstream commit"):
        aggregate(protocol, paths, execution_record)


def test_aggregate_rejects_execution_upstream_not_in_protocol(
    tmp_path: Path,
) -> None:
    protocol, paths, execution_record = complete_inputs(tmp_path)
    record = json.loads(execution_record.read_text(encoding="utf-8"))
    record["runs"]["ts_rag"]["upstream_commit"] = "unapproved-upstream"
    write_json(execution_record, record)

    with pytest.raises(ValueError, match="upstream commit drifted"):
        aggregate(protocol, paths, execution_record)


def test_aggregate_rejects_wrong_native_base_method(tmp_path: Path) -> None:
    protocol, paths, execution_record = complete_inputs(tmp_path)
    summary = json.loads(paths["ts_rag"].read_text(encoding="utf-8"))
    summary["results"][0]["base_method"] = "UnapprovedBase"
    write_json(paths["ts_rag"], summary)

    with pytest.raises(
        ValueError, match="canonical manifest base_method mismatch"
    ):
        aggregate(protocol, paths, execution_record)


def test_aggregate_rejects_empty_ts_rag_asset_verification(
    tmp_path: Path,
) -> None:
    protocol, paths, execution_record = complete_inputs(tmp_path)
    summary = json.loads(paths["ts_rag"].read_text(encoding="utf-8"))
    asset_path = Path(summary["asset_verification_path"])
    write_json(asset_path, {"manifest": "assets.json", "files": []})

    with pytest.raises(ValueError, match="does not match manifest"):
        aggregate(protocol, paths, execution_record)


def test_aggregate_rejects_unapproved_summary_bytes(tmp_path: Path) -> None:
    protocol, paths, execution_record = complete_inputs(tmp_path)
    paths["raf"].write_text(
        paths["raf"].read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="approved summary hash drifted"):
        aggregate(protocol, paths, execution_record)


def test_ratd_equivalence_evidence_is_hash_bound_and_byte_exact(
    tmp_path: Path,
) -> None:
    paths = {
        "ratd": install_equivalence_verification(
            tmp_path / "ratd", "RATD", 0, 1
        ),
        "csdi": install_equivalence_verification(
            tmp_path / "csdi", "CSDI", 1, 2
        ),
    }
    execution_record = {
        "runs": {
            "ratd": {
                "launcher_revision": "ratd-launcher",
                "equivalence_verifications": {
                    method: {
                        "path": str(path),
                        "sha256": sha256_file(path),
                    }
                    for method, path in paths.items()
                },
            }
        }
    }
    summary = {
        "source_revision": "ratd-source",
        "launcher_revision": "ratd-launcher",
        "upstream_commit": "ratd-upstream",
        "protocol_sha256": "a" * 64,
    }

    validated = validate_ratd_equivalence_evidence(
        execution_record,
        summary,
    )

    assert validated["ratd"]["strict_byte_equivalence"] is True
    verification = json.loads(paths["ratd"].read_text(encoding="utf-8"))
    comparison_path = Path(verification["comparison"]["result"]["path"])
    comparison = json.loads(
        comparison_path.read_text(encoding="utf-8")
    )
    comparison["origin_records"][0]["sample_sha256"] = "e" * 64
    write_json(comparison_path, comparison)
    verification["comparison"]["origin_records"][0][
        "sample_sha256"
    ] = "e" * 64
    verification["comparison"]["result"] = file_record(comparison_path)
    verification["verification_sha256"] = canonical_json_sha256(
        {
            key: value
            for key, value in verification.items()
            if key != "verification_sha256"
        }
    )
    write_json(paths["ratd"], verification)
    execution_record["runs"]["ratd"]["equivalence_verifications"]["ratd"][
        "sha256"
    ] = sha256_file(paths["ratd"])
    with pytest.raises(ValueError, match="cross-GPU equivalence"):
        validate_ratd_equivalence_evidence(execution_record, summary)
