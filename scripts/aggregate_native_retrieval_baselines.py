#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

try:
    from scripts.compose_native_raf_attempts import (
        validate_composite_raf_summary,
    )
    from scripts.generate_native_retrieval_baseline_manifest import (
        build_manifest,
    )
    from scripts.run_native_ratd_sharded_eval import (
        canonical_json_sha256,
        validate_file_record,
        validate_receipt as validate_sharded_eval_receipt,
        validate_topology_document as validate_sharded_eval_topology,
        verification_projection,
    )
except ModuleNotFoundError:
    from compose_native_raf_attempts import validate_composite_raf_summary
    from generate_native_retrieval_baseline_manifest import build_manifest
    from run_native_ratd_sharded_eval import (
        canonical_json_sha256,
        validate_file_record,
        validate_receipt as validate_sharded_eval_receipt,
        validate_topology_document as validate_sharded_eval_topology,
        verification_projection,
    )


EXPECTED = {
    "raf": {
        "label": "RAF",
        "pairs": 44,
        "metrics": ("wql", "mase"),
    },
    "ts_rag": {
        "label": "TS-RAG",
        "pairs": 7,
        "metrics": ("mse", "mae"),
    },
    "ratd": {
        "label": "RATD",
        "pairs": 1,
        "metrics": ("rmse", "mae"),
    },
}
EXPECTED_TOPOLOGY = {
    "raf": {
        "instance_type": "ml.p5.48xlarge",
        "reserved_gpus_per_host": 8,
        "processes_per_host": 6,
        "worker_gpu_ids": [2, 3, 4, 5, 6, 7],
    },
    "ts_rag": {
        "instance_type": "ml.p5.48xlarge",
        "reserved_gpus_per_host": 8,
        "processes_per_host": 7,
        "worker_gpu_ids": [0, 1, 2, 3, 4, 5, 6],
    },
    "ratd": {
        "instance_type": "ml.p5.48xlarge",
        "reserved_gpus_per_host": 8,
        "processes_per_host": 2,
        "worker_gpu_ids": [0, 1],
    },
}
TS_RAG_ASSET_MANIFEST_PATH = (
    Path(__file__).resolve().parents[1] / "docs" / "native_ts_rag_assets.json"
)
SHARDED_EVALUATION_GPU_IDS = {
    "ratd": list(range(8)),
    "csdi": list(range(1, 8)),
}


def is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_execution_record(
    execution_record_path: Path,
    protocol_sha256: str,
    protocol: dict,
) -> dict:
    record = load_json(execution_record_path)
    if record.get("schema_version") != 1:
        raise ValueError("execution record schema drifted")
    if record.get("protocol_sha256") != protocol_sha256:
        raise ValueError("execution record protocol hash drifted")
    platform = record.get("platform")
    if (
        not isinstance(platform, dict)
        or platform.get("instance_type") != "ml.p5.48xlarge"
        or platform.get("reserved_gpus_per_host") != 8
        or not isinstance(platform.get("project_root"), str)
        or not platform["project_root"]
    ):
        raise ValueError("execution record platform identity drifted")
    runs = record.get("runs")
    if not isinstance(runs, dict):
        raise ValueError("execution record runs are missing")
    sources = protocol.get("sources")
    if not isinstance(sources, dict):
        raise ValueError("native protocol sources are missing")
    for method, expected in EXPECTED_TOPOLOGY.items():
        run = runs.get(method)
        if not isinstance(run, dict):
            raise ValueError(f"execution record {method} run is missing")
        if (
            run.get("processes_per_host")
            != expected["processes_per_host"]
            or run.get("worker_gpu_ids") != expected["worker_gpu_ids"]
        ):
            raise ValueError(
                f"execution record {method} topology drifted"
            )
        if (
            run.get("expected_pairs") != EXPECTED[method]["pairs"]
            or not isinstance(run.get("output_root"), str)
            or not run["output_root"]
            or not isinstance(run.get("method_source_revision"), str)
            or not run["method_source_revision"]
            or not isinstance(run.get("upstream_commit"), str)
            or not run["upstream_commit"]
        ):
            raise ValueError(f"execution record {method} identity drifted")
        source = sources.get(method)
        if (
            not isinstance(source, dict)
            or run["upstream_commit"] != source.get("commit")
        ):
            raise ValueError(
                f"execution record {method} upstream commit drifted"
            )
        for field in ("summary_sha256", "topology_sha256"):
            if not is_sha256(run.get(field)):
                raise ValueError(
                    f"execution record {method} {field} is invalid"
                )
        if method == "raf" and (
            not isinstance(run.get("launcher_revision"), str)
            or not run["launcher_revision"]
        ):
            raise ValueError(
                "execution record raf launcher identity drifted"
            )
    approved_roots = {
        run["output_root"] for run in runs.values() if isinstance(run, dict)
    }
    for invalidated in record.get("invalidated_runs", []):
        if (
            isinstance(invalidated, dict)
            and invalidated.get("output_root") in approved_roots
        ):
            raise ValueError(
                "execution record approves an invalidated output root"
            )
    return record


def validate_run_identity(
    method: str,
    summary_path: Path,
    summary: dict,
    execution_record: dict,
) -> None:
    run = execution_record["runs"][method]
    expected_path = (
        Path(execution_record["platform"]["project_root"])
        / run["output_root"]
        / "summary.json"
    )
    if summary_path.resolve() != expected_path.resolve():
        raise ValueError(f"{method}: summary path is not the approved run")
    if summary.get("source_revision") != run["method_source_revision"]:
        raise ValueError(f"{method}: source revision is not approved")
    launcher_revision = run.get("launcher_revision")
    if (
        launcher_revision is not None
        and summary.get("launcher_revision") != launcher_revision
    ):
        raise ValueError(f"{method}: launcher revision is not approved")
    upstream_commit = run["upstream_commit"]
    if (
        summary.get("upstream_commit") is not None
        and summary.get("upstream_commit") != upstream_commit
    ):
        raise ValueError(f"{method}: upstream commit is not approved")

    results = summary.get("results")
    if isinstance(results, dict):
        records = list(results.values())
    elif isinstance(results, list):
        records = results
    else:
        raise ValueError(f"{method}: result records are missing")
    for result in records:
        if result.get("source_revision") != run["method_source_revision"]:
            raise ValueError(
                f"{method}: result source revision is not approved"
            )
        if result.get("upstream_commit") != upstream_commit:
            raise ValueError(
                f"{method}: result upstream commit is not approved"
            )


def validate_sharded_ratd_evidence(
    summary_path: Path,
    summary: dict,
    topology_path: Path,
    topology: dict,
) -> None:
    if (
        topology.get("schema_version") != 2
        or topology.get("job_kind")
        != "native_ratd_sharded_training_eval_topology"
        or topology.get("instance_type") != "ml.p5.48xlarge"
        or topology.get("reserved_gpus_per_host") != 8
        or topology.get("topology_valid") is not True
    ):
        raise ValueError("ratd: sharded topology identity drifted")

    training = topology.get("training")
    if not isinstance(training, dict):
        raise ValueError("ratd: sharded training evidence is missing")
    training_path = validate_file_record(
        training.get("topology"), "RATD training topology"
    )
    training_topology = load_json(training_path)
    if (
        training_topology.get("instance_type") != "ml.p5.48xlarge"
        or training_topology.get("reserved_gpus_per_host") != 8
        or training_topology.get("processes_per_host") != 2
        or training_topology.get("worker_gpu_ids") != [0, 1]
        or training_topology.get("topology_valid") is not True
    ):
        raise ValueError("ratd: training topology drifted")
    termination = training.get("termination")
    if (
        not isinstance(termination, dict)
        or termination.get("accepted") is not True
        or termination.get("classification")
        != "original-serial-evaluation-terminated-after-epoch100"
        or set(termination.get("checkpoint_evidence", {}))
        != {"ratd", "csdi"}
        or any(
            evidence.get("epoch") != 100
            for evidence in termination["checkpoint_evidence"].values()
        )
    ):
        raise ValueError("ratd: epoch-100 training transition is invalid")
    for key, evidence in termination["checkpoint_evidence"].items():
        validate_file_record(
            evidence.get("checkpoint"), f"RATD {key} checkpoint"
        )

    evaluation = topology.get("evaluation")
    if not isinstance(evaluation, dict) or set(evaluation) != {"ratd", "csdi"}:
        raise ValueError("ratd: sharded evaluation evidence is incomplete")
    for key, evidence in evaluation.items():
        if not isinstance(evidence, dict):
            raise ValueError(f"ratd: {key} evaluation evidence is invalid")
        method_topology_path = validate_file_record(
            evidence.get("topology"), f"RATD {key} evaluation topology"
        )
        method_topology = load_json(method_topology_path)
        validate_sharded_eval_topology(method_topology)
        expected_gpu_ids = SHARDED_EVALUATION_GPU_IDS[key]
        if (
            method_topology.get("method") != key.upper()
            or method_topology.get("instance_type") != "ml.p5.48xlarge"
            or method_topology.get("reserved_gpus_per_host") != 8
            or method_topology.get("processes_per_host")
            != len(expected_gpu_ids)
            or method_topology.get("worker_gpu_ids") != expected_gpu_ids
            or method_topology.get("world_size") != len(expected_gpu_ids)
            or method_topology.get("inactive_reserved_gpus")
            != 8 - len(expected_gpu_ids)
        ):
            raise ValueError(
                f"ratd: {key} formal evaluation topology drifted"
            )
        receipt_path = validate_file_record(
            evidence.get("receipt"), f"RATD {key} evaluation receipt"
        )
        receipt = load_json(receipt_path)
        result_record = receipt.get("method_result")
        result_path = validate_file_record(
            result_record, f"RATD {key} method result"
        )
        validate_sharded_eval_receipt(receipt_path, result_path)
        if evidence.get("checkpoint") != receipt.get("checkpoint"):
            raise ValueError(f"ratd: {key} checkpoint receipt drifted")
        if evidence.get("plan") != receipt.get("plan"):
            raise ValueError(f"ratd: {key} plan receipt drifted")

    receipt_value = summary.get("receipt_path")
    if not isinstance(receipt_value, str) or not receipt_value:
        raise ValueError("ratd: finalizer receipt path is missing")
    finalizer_receipt_path = Path(receipt_value)
    finalizer_receipt = load_json(finalizer_receipt_path)
    receipt_hash = finalizer_receipt.get("receipt_sha256")
    receipt_payload = {
        key: value
        for key, value in finalizer_receipt.items()
        if key != "receipt_sha256"
    }
    if (
        receipt_hash != canonical_json_sha256(receipt_payload)
        or finalizer_receipt.get("schema_version") != 1
        or finalizer_receipt.get("job_kind")
        != "native_ratd_sharded_pair_finalizer_receipt"
    ):
        raise ValueError("ratd: finalizer receipt self-hash drifted")
    recorded_summary = validate_file_record(
        finalizer_receipt.get("summary"), "RATD final summary"
    )
    if (
        recorded_summary.resolve() != summary_path.resolve()
        or load_json(recorded_summary) != summary
    ):
        raise ValueError("ratd: finalizer receipt binds another summary")
    recorded_topology = validate_file_record(
        finalizer_receipt.get("combined_topology"),
        "RATD combined topology",
    )
    if recorded_topology.resolve() != topology_path.resolve():
        raise ValueError("ratd: finalizer receipt binds another topology")
    direct_records = {
        "ratd": {
            "result": "ratd_result",
            "receipt": "ratd_receipt",
        },
        "csdi": {
            "result": "csdi_result",
            "receipt": "csdi_receipt",
        },
    }
    for key, labels in direct_records.items():
        evidence = evaluation[key]
        receipt_path = Path(evidence["receipt"]["path"])
        receipt = load_json(receipt_path)
        for record_kind, finalizer_key in labels.items():
            actual = validate_file_record(
                finalizer_receipt.get(finalizer_key),
                f"RATD finalizer {key} {record_kind}",
            )
            expected_record = (
                evidence["receipt"]
                if record_kind == "receipt"
                else receipt["method_result"]
            )
            expected = validate_file_record(
                expected_record,
                f"RATD {key} {record_kind}",
            )
            if actual.resolve() != expected.resolve():
                raise ValueError(
                    f"ratd: finalizer binds another {key} {record_kind}"
                )
    training_topology = validate_file_record(
        finalizer_receipt.get("training_topology"),
        "RATD finalizer training topology",
    )
    if training_topology.resolve() != training_path.resolve():
        raise ValueError("ratd: finalizer binds another training topology")
    reference_map = Path(summary["reference_metadata"]["reference_map"])
    reference_metadata = reference_map.with_suffix(".json")
    for key, expected in (
        ("reference_map", reference_map),
        ("reference_metadata", reference_metadata),
    ):
        actual = validate_file_record(
            finalizer_receipt.get(key), f"RATD finalizer {key}"
        )
        if actual.resolve() != expected.resolve():
            raise ValueError(f"ratd: finalizer binds another {key}")


def validate_ratd_equivalence_evidence(
    execution_record: dict,
    summary: dict,
) -> dict[str, dict]:
    run = execution_record["runs"]["ratd"]
    if run.get("launcher_revision") != summary.get("launcher_revision"):
        raise ValueError("ratd: equivalence launcher revision drifted")
    records = run.get("equivalence_verifications")
    if not isinstance(records, dict) or set(records) != {"ratd", "csdi"}:
        raise ValueError(
            "ratd: equivalence verification records are incomplete"
        )

    validated = {}
    for key, expected_method in (("ratd", "RATD"), ("csdi", "CSDI")):
        record = records[key]
        if (
            not isinstance(record, dict)
            or not isinstance(record.get("path"), str)
            or not record["path"]
            or not isinstance(record.get("sha256"), str)
            or len(record["sha256"]) != 64
        ):
            raise ValueError(
                f"ratd: {key} equivalence record is invalid"
            )
        path = Path(record["path"])
        if sha256_file(path) != record["sha256"]:
            raise ValueError(
                f"ratd: {key} equivalence file hash drifted"
            )
        verification = load_json(path)
        payload = {
            field: value
            for field, value in verification.items()
            if field != "verification_sha256"
        }
        identity = verification.get("identity")
        if (
            verification.get("schema_version") != 1
            or verification.get("job_kind")
            != "native_ratd_exact_cross_gpu_verification"
            or verification.get("status") != "verified"
            or verification.get("method") != expected_method
            or verification.get("strict_byte_equivalence") is not True
            or verification.get("samples") != 100
            or verification.get("diffusion_steps") != 50
            or verification.get("verification_range", {}).get(
                "origin_count"
            )
            != 2
            or verification.get("verification_sha256")
            != canonical_json_sha256(payload)
            or not isinstance(identity, dict)
            or identity.get("source_revision")
            != summary.get("source_revision")
            or identity.get("launcher_revision")
            != summary.get("launcher_revision")
            or identity.get("upstream_commit")
            != summary.get("upstream_commit")
            or identity.get("protocol", {}).get("sha256")
            != summary.get("protocol_sha256")
        ):
            raise ValueError(
                f"ratd: {key} equivalence identity drifted"
            )

        projections = {}
        pids = set()
        physical_gpus = set()
        for role in ("oracle", "comparison"):
            evidence = verification.get(role)
            if (
                not isinstance(evidence, dict)
                or evidence.get("role") != role
                or not isinstance(evidence.get("pid"), int)
                or evidence["pid"] <= 0
                or not isinstance(evidence.get("physical_gpu"), int)
                or evidence.get("returncode") != 0
                or evidence.get("observed_gpus")
                != [evidence["physical_gpu"]]
            ):
                raise ValueError(
                    f"ratd: {key} {role} equivalence topology is invalid"
                )
            result_path = validate_file_record(
                evidence.get("result"),
                f"RATD {key} {role} equivalence result",
            )
            worker_result = load_json(result_path)
            projection = verification_projection(worker_result)
            if (
                worker_result.get("role") != role
                or worker_result.get("pid") != evidence["pid"]
                or worker_result.get("physical_gpu")
                != evidence["physical_gpu"]
                or evidence.get("origin_records")
                != projection["origin_records"]
                or evidence.get("start_rng") != projection["start_rng"]
                or evidence.get("final_rng") != projection["final_rng"]
            ):
                raise ValueError(
                    f"ratd: {key} {role} equivalence result drifted"
                )
            projections[role] = projection
            pids.add(evidence["pid"])
            physical_gpus.add(evidence["physical_gpu"])
        if (
            len(pids) != 2
            or len(physical_gpus) != 2
            or projections["oracle"] != projections["comparison"]
            or verification.get("start_rng")
            != projections["oracle"]["start_rng"]
            or verification.get("expected_final_rng")
            != projections["oracle"]["final_rng"]
        ):
            raise ValueError(
                f"ratd: {key} cross-GPU equivalence is not exact"
            )
        validated[key] = {
            "path": str(path),
            "sha256": record["sha256"],
            "oracle_gpu": verification["oracle"]["physical_gpu"],
            "comparison_gpu": verification["comparison"]["physical_gpu"],
            "origin_count": 2,
            "strict_byte_equivalence": True,
        }
    return validated


def validate_topology(
    method: str, summary_path: Path, summary: dict
) -> Path:
    topology_value = summary.get("topology_path")
    if not isinstance(topology_value, str) or not topology_value:
        raise ValueError(f"{method}: topology path is missing")
    topology_path = Path(topology_value)
    topology = load_json(topology_path)
    if (
        method == "ratd"
        and topology.get("job_kind")
        == "native_ratd_sharded_training_eval_topology"
    ):
        validate_sharded_ratd_evidence(
            summary_path, summary, topology_path, topology
        )
        return topology_path
    expected = EXPECTED_TOPOLOGY[method]
    for key, value in expected.items():
        if topology.get(key) != value:
            raise ValueError(f"{method}: topology {key} drifted")
    if (
        topology.get("total_gpus") != expected["reserved_gpus_per_host"]
        or topology.get("world_size") != expected["processes_per_host"]
    ):
        raise ValueError(f"{method}: topology world size drifted")
    worker_pids = topology.get("worker_pids")
    returncodes = topology.get("worker_returncodes")
    if (
        not isinstance(worker_pids, dict)
        or len(worker_pids) != expected["processes_per_host"]
        or not isinstance(returncodes, dict)
        or set(returncodes) != set(worker_pids)
        or any(int(code) != 0 for code in returncodes.values())
    ):
        raise ValueError(f"{method}: worker completion evidence is invalid")
    pids = [int(pid) for pid in worker_pids.values()]
    if len(set(pids)) != len(pids) or any(pid <= 0 for pid in pids):
        raise ValueError(f"{method}: worker PIDs are invalid")

    expected_bindings = topology.get("expected_bindings")
    if isinstance(expected_bindings, dict):
        expected_by_pid = {
            int(pid): int(gpu) for pid, gpu in expected_bindings.items()
        }
    else:
        ordered_workers = sorted(worker_pids, key=int)
        expected_by_pid = {
            int(worker_pids[worker]): gpu
            for worker, gpu in zip(
                ordered_workers,
                expected["worker_gpu_ids"],
                strict=True,
            )
        }
    if (
        set(expected_by_pid) != set(pids)
        or sorted(expected_by_pid.values())
        != expected["worker_gpu_ids"]
    ):
        raise ValueError(f"{method}: expected worker bindings drifted")

    observed_by_pid = {pid: set() for pid in pids}
    observations = topology.get("observations")
    if not isinstance(observations, list) or not observations:
        raise ValueError(f"{method}: GPU observations are missing")
    for observation in observations:
        for record in observation.get("processes", []):
            pid = int(record["pid"])
            physical_gpu = record.get("physical_gpu")
            if pid in observed_by_pid and physical_gpu is not None:
                observed_by_pid[pid].add(int(physical_gpu))
    if any(
        observed_by_pid[pid] != {expected_gpu}
        for pid, expected_gpu in expected_by_pid.items()
    ):
        raise ValueError(f"{method}: observed worker bindings are invalid")
    if method in ("raf", "ratd") and topology.get("topology_valid") is not True:
        raise ValueError(f"{method}: topology validity gate is false")
    return topology_path


def validate_composite_raf_topologies(
    summary: dict,
    protocol_path: Path,
    execution_record: dict,
) -> list[Path]:
    run = execution_record["runs"]["raf"]
    topology_paths = validate_composite_raf_summary(
        summary,
        protocol_path,
        expected_source_revision=run["method_source_revision"],
        expected_launcher_revision=run["launcher_revision"],
    )
    platform = execution_record["platform"]
    for topology_path in topology_paths:
        topology = load_json(topology_path)
        if (
            topology.get("instance_type") != platform["instance_type"]
            or topology.get("reserved_gpus_per_host")
            != platform["reserved_gpus_per_host"]
        ):
            raise ValueError("raf: composite attempt platform drifted")
    return topology_paths


def validate_metrics(
    method: str,
    cell_id: str,
    base: dict,
    retrieval: dict,
    reported_delta: dict,
) -> dict:
    expected_metrics = EXPECTED[method]["metrics"]
    if set(base) != set(expected_metrics):
        raise ValueError(f"{cell_id}: base metric coverage drifted")
    if set(retrieval) != set(expected_metrics):
        raise ValueError(f"{cell_id}: retrieval metric coverage drifted")
    if set(reported_delta) != set(expected_metrics):
        raise ValueError(f"{cell_id}: delta metric coverage drifted")
    delta = {}
    for metric in expected_metrics:
        base_value = float(base[metric])
        retrieval_value = float(retrieval[metric])
        reported_value = float(reported_delta[metric])
        values = (base_value, retrieval_value, reported_value)
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"{cell_id}: non-finite {metric}")
        expected_delta = retrieval_value - base_value
        if not math.isclose(
            reported_value,
            expected_delta,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError(f"{cell_id}: {metric} delta mismatch")
        delta[metric] = expected_delta
    return delta


def validate_summary_header(
    method: str,
    summary: dict,
    protocol_sha256: str,
) -> None:
    expected = EXPECTED[method]
    if summary.get("method") != expected["label"]:
        raise ValueError(f"{method}: method identity drifted")
    if summary.get("expected_pairs") != expected["pairs"]:
        raise ValueError(f"{method}: expected pair count drifted")
    if summary.get("completed_pairs") != expected["pairs"]:
        raise ValueError(f"{method}: incomplete pair count")
    if summary.get("failed_or_missing_pairs") != 0:
        raise ValueError(f"{method}: failed or missing pairs remain")
    if summary.get("all_completed") is not True:
        raise ValueError(f"{method}: all_completed gate is false")
    if summary.get("failures"):
        raise ValueError(f"{method}: failure records are present")
    if summary.get("protocol_sha256") != protocol_sha256:
        raise ValueError(f"{method}: protocol hash drifted")
    source_revision = summary.get("source_revision")
    if not isinstance(source_revision, str) or not source_revision.strip():
        raise ValueError(f"{method}: source revision is missing")


def validate_ts_rag_assets(summary: dict) -> Path:
    asset_value = summary.get("asset_verification_path")
    if not isinstance(asset_value, str) or not asset_value:
        raise ValueError("ts_rag: asset verification path is missing")
    asset_path = Path(asset_value)
    asset_verification = load_json(asset_path)
    if not isinstance(asset_verification, dict):
        raise ValueError("ts_rag: asset verification is invalid")
    manifest = load_json(TS_RAG_ASSET_MANIFEST_PATH)
    if (
        manifest.get("schema_version") != 1
        or manifest.get("source", {}).get("upstream_commit")
        != summary["results"][0]["upstream_commit"]
    ):
        raise ValueError("ts_rag: canonical asset manifest identity drifted")
    expected_records = {}
    for root_key, files_key in (
        ("checkpoint_root", "checkpoint_files"),
        ("data_root", "data_files"),
    ):
        root = Path(manifest[root_key])
        records = manifest.get(files_key)
        if not isinstance(records, list) or not records:
            raise ValueError("ts_rag: canonical asset manifest is incomplete")
        for record in records:
            path = str(root / record["path"])
            expected_records[path] = {
                "path": path,
                "size_bytes": record["size_bytes"],
                "sha256": record["sha256"],
            }
    observed_records = asset_verification.get("files")
    if not isinstance(observed_records, list):
        raise ValueError("ts_rag: asset verification files are missing")
    observed_by_path = {
        record.get("path"): record
        for record in observed_records
        if isinstance(record, dict)
    }
    if (
        len(observed_by_path) != len(observed_records)
        or observed_by_path != expected_records
    ):
        raise ValueError("ts_rag: asset verification does not match manifest")
    return asset_path


def validate_ratd_references(summary: dict) -> tuple[Path, Path]:
    metadata = summary.get("reference_metadata")
    if not isinstance(metadata, dict):
        raise ValueError("ratd: reference metadata is missing")
    if (
        metadata.get("context_length") != 96
        or metadata.get("prediction_length") != 168
        or metadata.get("top_k") != 3
        or metadata.get("query_counts")
        != {"train": 17_886, "validation": 2_463, "test": 5_093}
    ):
        raise ValueError("ratd: reference metadata identity drifted")
    train_end = metadata.get("train_end")
    if not isinstance(train_end, int) or train_end <= 0:
        raise ValueError("ratd: reference training boundary is invalid")
    audits = metadata.get("audits")
    if not isinstance(audits, dict) or set(audits) != {
        "train",
        "validation",
        "test",
    }:
        raise ValueError("ratd: reference audits are incomplete")
    for split, audit in audits.items():
        if not isinstance(audit, dict):
            raise ValueError(f"ratd: {split} reference audit is invalid")
        minimum = audit.get("minimum_reference_start")
        maximum = audit.get("maximum_reference_start")
        latest_end = audit.get("latest_reference_future_end")
        if (
            not all(
                isinstance(value, int)
                for value in (minimum, maximum, latest_end)
            )
            or minimum < 0
            or maximum < minimum
            or latest_end != maximum + 96 + 168
            or audit.get("train_end") != train_end
            or latest_end > train_end
        ):
            raise ValueError(
                f"ratd: {split} reference crosses training boundary"
            )

    reference_map_value = metadata.get("reference_map")
    reference_map_sha256 = metadata.get("reference_map_sha256")
    if (
        not isinstance(reference_map_value, str)
        or not reference_map_value
        or not isinstance(reference_map_sha256, str)
        or not reference_map_sha256
    ):
        raise ValueError("ratd: reference map identity is missing")
    reference_map_path = Path(reference_map_value)
    if sha256_file(reference_map_path) != reference_map_sha256:
        raise ValueError("ratd: reference map hash drifted")
    metadata_path = reference_map_path.with_suffix(".json")
    if load_json(metadata_path) != metadata:
        raise ValueError("ratd: reference metadata file drifted")
    return reference_map_path, metadata_path


def normalize_standard_summary(method: str, summary: dict) -> list[dict]:
    results = summary.get("results")
    if not isinstance(results, list) or len(results) != EXPECTED[method]["pairs"]:
        raise ValueError(f"{method}: result records are incomplete")
    normalized = []
    for result in results:
        cell_id = result.get("cell_id")
        if not isinstance(cell_id, str) or not cell_id:
            raise ValueError(f"{method}: result is missing a cell ID")
        if result.get("status") != "completed":
            raise ValueError(f"{cell_id}: result is not completed")
        if result.get("method") != EXPECTED[method]["label"]:
            raise ValueError(f"{cell_id}: method identity drifted")
        if (
            method == "ts_rag"
            and result.get("retrieval_index_audit", {}).get(
                "all_reference_futures_end_in_training"
            )
            is not True
        ):
            raise ValueError(
                f"{cell_id}: retrieval index crosses training boundary"
            )
        base = result.get("base", {}).get("metrics")
        retrieval = result.get("retrieval", {}).get("metrics")
        if not isinstance(base, dict) or not isinstance(retrieval, dict):
            raise ValueError(f"{cell_id}: paired metrics are missing")
        delta = validate_metrics(
            method,
            cell_id,
            base,
            retrieval,
            result.get("delta", {}),
        )
        normalized.append(
            {
                "cell_id": cell_id,
                "method": EXPECTED[method]["label"],
                "base_method": result["base_method"],
                "backbone": result["backbone"],
                "dataset": result["dataset"],
                "context_length": result["context_length"],
                "prediction_length": result["prediction_length"],
                "metrics": list(EXPECTED[method]["metrics"]),
                "base": base,
                "retrieval": retrieval,
                "delta": delta,
            }
        )
    return normalized


def normalize_ratd_summary(summary: dict) -> list[dict]:
    if summary.get("topology_valid") is not True:
        raise ValueError("ratd: topology validation failed")
    results = summary.get("results")
    if not isinstance(results, dict) or set(results) != {"ratd", "csdi"}:
        raise ValueError("ratd: native paired results are missing")
    for method in ("ratd", "csdi"):
        result = results[method]
        if (
            result.get("status") != "completed"
            or result.get("test_origins") != 5_093
            or result.get("protocol_sha256")
            != summary.get("protocol_sha256")
        ):
            raise ValueError(f"ratd: invalid {method} result identity")
    pair = summary.get("pair")
    if not isinstance(pair, dict):
        raise ValueError("ratd: paired result is missing")
    cell_id = pair.get("cell_id")
    if not isinstance(cell_id, str) or not cell_id:
        raise ValueError("ratd: paired result is missing a cell ID")
    if pair.get("base") != results["csdi"].get("metrics"):
        raise ValueError(
            "ratd: paired base metrics do not match CSDI result"
        )
    if pair.get("retrieval") != results["ratd"].get("metrics"):
        raise ValueError(
            "ratd: paired retrieval metrics do not match RATD result"
        )
    delta = validate_metrics(
        "ratd",
        cell_id,
        pair.get("base", {}),
        pair.get("retrieval", {}),
        pair.get("delta", {}),
    )
    return [
        {
            "cell_id": cell_id,
            "method": "RATD",
            "base_method": summary["base_method"],
            "backbone": summary["base_method"],
            "dataset": "electricity",
            "context_length": 96,
            "prediction_length": 168,
            "metrics": list(EXPECTED["ratd"]["metrics"]),
            "base": pair["base"],
            "retrieval": pair["retrieval"],
            "delta": delta,
        }
    ]


def validate_canonical_manifest(protocol: dict, pairs: list[dict]) -> None:
    manifest = build_manifest(protocol)
    if len(manifest) != 52:
        raise ValueError("Canonical native baseline manifest is not 52 cells")
    expected_by_id = {row["id"]: row for row in manifest}
    actual_by_id = {pair["cell_id"]: pair for pair in pairs}
    if (
        len(actual_by_id) != len(pairs)
        or set(actual_by_id) != set(expected_by_id)
    ):
        raise ValueError(
            "Native baseline cell IDs do not match canonical manifest"
        )
    for cell_id, row in expected_by_id.items():
        pair = actual_by_id[cell_id]
        expected_fields = {
            "method": EXPECTED[row["method"]]["label"],
            "base_method": protocol["systems"][row["method"]]["base_system"],
            "dataset": row["dataset"],
            "backbone": row["backbone"],
            "context_length": row["context_length"],
            "prediction_length": row["prediction_length"],
            "metrics": row["metrics"],
        }
        for field, expected_value in expected_fields.items():
            if pair.get(field) != expected_value:
                raise ValueError(
                    f"{cell_id}: canonical manifest {field} mismatch"
                )


def aggregate(
    protocol_path: Path,
    summary_paths: dict[str, Path],
    execution_record_path: Path,
) -> dict:
    protocol = load_json(protocol_path)
    protocol_sha256 = sha256_file(protocol_path)
    execution_record = validate_execution_record(
        execution_record_path, protocol_sha256, protocol
    )
    summaries = {
        method: load_json(path) for method, path in summary_paths.items()
    }
    pairs = []
    sources = {}
    for method in ("raf", "ts_rag", "ratd"):
        summary = summaries[method]
        validate_run_identity(
            method,
            summary_paths[method],
            summary,
            execution_record,
        )
        validate_summary_header(method, summary, protocol_sha256)
        is_composite_raf = (
            method == "raf"
            and summary.get("schema_version") == 2
            and "composite_attempts" in summary
        )
        if is_composite_raf:
            topology_paths = validate_composite_raf_topologies(
                summary,
                protocol_path,
                execution_record,
            )
            topology_path = topology_paths[-1]
        else:
            topology_paths = []
            topology_path = validate_topology(
                method, summary_paths[method], summary
            )
        normalized = (
            normalize_ratd_summary(summary)
            if method == "ratd"
            else normalize_standard_summary(method, summary)
        )
        pairs.extend(normalized)
        source = {
            "path": str(summary_paths[method]),
            "sha256": sha256_file(summary_paths[method]),
            "source_revision": summary["source_revision"],
            "launcher_revision": summary.get("launcher_revision"),
            "topology_path": str(topology_path),
            "topology_sha256": sha256_file(topology_path),
            "pairs": len(normalized),
        }
        if is_composite_raf:
            source.update(
                {
                    "composite_attempts": [
                        {
                            "attempt_id": record["attempt_id"],
                            "status": record["status"],
                            "summary_path": record["summary_path"],
                            "summary_sha256": record["summary_sha256"],
                            "topology_path": record["topology_path"],
                            "topology_sha256": record["topology_sha256"],
                            "accepted_result_count": record[
                                "accepted_result_count"
                            ],
                        }
                        for record in summary["composite_attempts"]
                    ],
                    "composite_receipt": summary["receipt"],
                }
            )
        elif method == "ts_rag":
            asset_path = validate_ts_rag_assets(summary)
            source.update(
                {
                    "asset_verification_path": str(asset_path),
                    "asset_verification_sha256": sha256_file(asset_path),
                }
            )
        elif method == "ratd":
            reference_map_path, reference_metadata_path = (
                validate_ratd_references(summary)
            )
            source.update(
                {
                    "reference_map_path": str(reference_map_path),
                    "reference_map_sha256": sha256_file(reference_map_path),
                    "reference_metadata_path": str(
                        reference_metadata_path
                    ),
                    "reference_metadata_sha256": sha256_file(
                        reference_metadata_path
                    ),
                }
            )
            if summary.get("schema_version") == 2:
                source["equivalence_verifications"] = (
                    validate_ratd_equivalence_evidence(
                        execution_record,
                        summary,
                    )
                )
        sources[method] = source
    validate_canonical_manifest(protocol, pairs)
    for method, summary_path in summary_paths.items():
        run = execution_record["runs"][method]
        if sha256_file(summary_path) != run["summary_sha256"]:
            raise ValueError(f"{method}: approved summary hash drifted")
        if sources[method]["topology_sha256"] != run["topology_sha256"]:
            raise ValueError(f"{method}: approved topology hash drifted")
    ts_rag_asset_path = Path(
        summaries["ts_rag"]["asset_verification_path"]
    )
    expected_asset_hash = execution_record["runs"]["ts_rag"].get(
        "asset_verification_sha256"
    )
    if (
        not is_sha256(expected_asset_hash)
        or sha256_file(ts_rag_asset_path) != expected_asset_hash
    ):
        raise ValueError("ts_rag: approved asset verification hash drifted")
    if summaries["ratd"].get("schema_version") == 2:
        run = execution_record["runs"]["ratd"]
        finalizer_path = Path(summaries["ratd"]["receipt_path"])
        expected_finalizer_hash = run.get("finalizer_receipt_sha256")
        if (
            not is_sha256(expected_finalizer_hash)
            or sha256_file(finalizer_path) != expected_finalizer_hash
        ):
            raise ValueError("ratd: approved finalizer receipt hash drifted")
    return {
        "schema_version": 2,
        "protocol_path": str(protocol_path),
        "protocol_sha256": protocol_sha256,
        "execution_record": {
            "path": str(execution_record_path),
            "sha256": sha256_file(execution_record_path),
        },
        "expected_pairs": 52,
        "completed_pairs": 52,
        "all_completed": True,
        "source_summaries": sources,
        "pairs": sorted(
            pairs, key=lambda pair: (pair["method"], pair["cell_id"])
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--raf-summary", required=True, type=Path)
    parser.add_argument("--ts-rag-summary", required=True, type=Path)
    parser.add_argument("--ratd-summary", required=True, type=Path)
    parser.add_argument("--execution-record", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    document = aggregate(
        args.protocol,
        {
            "raf": args.raf_summary,
            "ts_rag": args.ts_rag_summary,
            "ratd": args.ratd_summary,
        },
        args.execution_record,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
