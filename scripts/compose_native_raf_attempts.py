#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

try:
    from scripts.generate_native_retrieval_baseline_manifest import (
        build_manifest,
    )
except ModuleNotFoundError:
    from generate_native_retrieval_baseline_manifest import build_manifest


METRICS = ("wql", "mase")
ATTEMPT_STATUSES = {"completed", "interrupted"}


@dataclass(frozen=True)
class AttemptSpec:
    attempt_id: str
    status: str
    summary_path: Path
    topology_path: Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot load JSON from {path}: {error}") from error
    if not isinstance(document, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return document


def canonical_json_sha256(document: object) -> str:
    encoded = json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def save_json_atomic(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def canonical_raf_rows(protocol: dict) -> dict[str, dict]:
    rows = [
        row for row in build_manifest(protocol) if row["method"] == "raf"
    ]
    if len(rows) != 44:
        raise ValueError(f"Expected 44 canonical RAF cells, found {len(rows)}")
    return {row["id"]: row for row in rows}


def _require_nonempty_string(document: dict, field: str, context: str) -> str:
    value = document.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context}: {field} is missing")
    return value


def _worker_topology(topology: dict, attempt_id: str) -> dict:
    worker_pids = topology.get("worker_pids")
    returncodes = topology.get("worker_returncodes")
    expected_bindings = topology.get("expected_bindings")
    observations = topology.get("observations")
    if (
        not isinstance(worker_pids, dict)
        or not worker_pids
        or not isinstance(returncodes, dict)
        or set(returncodes) != set(worker_pids)
    ):
        raise ValueError(f"{attempt_id}: invalid worker completion evidence")
    if not isinstance(expected_bindings, dict):
        raise ValueError(f"{attempt_id}: expected GPU bindings are missing")
    if not isinstance(observations, list) or not observations:
        raise ValueError(f"{attempt_id}: GPU observations are missing")

    pids_by_worker: dict[str, int] = {}
    returncodes_by_pid: dict[int, int] = {}
    for worker, raw_pid in worker_pids.items():
        pid = int(raw_pid)
        if pid <= 0 or pid in returncodes_by_pid:
            raise ValueError(f"{attempt_id}: worker PIDs are invalid")
        pids_by_worker[str(worker)] = pid
        returncodes_by_pid[pid] = int(returncodes[worker])

    bindings_by_pid = {
        int(raw_pid): int(raw_gpu)
        for raw_pid, raw_gpu in expected_bindings.items()
    }
    if set(bindings_by_pid) != set(returncodes_by_pid):
        raise ValueError(f"{attempt_id}: expected GPU bindings drifted")
    reserved = topology.get("reserved_gpus_per_host")
    if (
        not isinstance(reserved, int)
        or reserved <= 0
        or any(gpu < 0 or gpu >= reserved for gpu in bindings_by_pid.values())
    ):
        raise ValueError(f"{attempt_id}: declared physical GPUs are invalid")

    observed_by_pid = {pid: set() for pid in returncodes_by_pid}
    for observation in observations:
        if not isinstance(observation, dict):
            raise ValueError(f"{attempt_id}: invalid GPU observation")
        processes = observation.get("processes", [])
        if not isinstance(processes, list):
            raise ValueError(f"{attempt_id}: invalid GPU process observation")
        for record in processes:
            if not isinstance(record, dict) or "pid" not in record:
                raise ValueError(f"{attempt_id}: invalid GPU process record")
            pid = int(record["pid"])
            physical_gpu = record.get("physical_gpu")
            if pid in observed_by_pid and physical_gpu is not None:
                observed_by_pid[pid].add(int(physical_gpu))

    return {
        "pids_by_worker": pids_by_worker,
        "returncodes_by_pid": returncodes_by_pid,
        "bindings_by_pid": bindings_by_pid,
        "observed_by_pid": observed_by_pid,
    }


def _declared_result_gpu(result: dict, cell_id: str) -> int:
    raw_gpu = result.get("physical_gpu")
    if raw_gpu is None:
        raw_gpu = result.get("cuda_visible_devices")
    if isinstance(raw_gpu, int):
        return raw_gpu
    if isinstance(raw_gpu, str):
        fields = [field.strip() for field in raw_gpu.split(",") if field.strip()]
        if len(fields) == 1:
            try:
                return int(fields[0])
            except ValueError:
                pass
    raise ValueError(f"{cell_id}: result does not declare one physical GPU")


def _validate_result_metrics(result: dict, cell_id: str) -> None:
    base = result.get("base", {}).get("metrics")
    retrieval = result.get("retrieval", {}).get("metrics")
    delta = result.get("delta")
    if (
        not isinstance(base, dict)
        or set(base) != set(METRICS)
        or not isinstance(retrieval, dict)
        or set(retrieval) != set(METRICS)
        or not isinstance(delta, dict)
        or set(delta) != set(METRICS)
    ):
        raise ValueError(f"{cell_id}: RAF metric coverage drifted")
    for metric in METRICS:
        values = (
            float(base[metric]),
            float(retrieval[metric]),
            float(delta[metric]),
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"{cell_id}: non-finite {metric}")
        expected = values[1] - values[0]
        if not math.isclose(
            values[2],
            expected,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError(f"{cell_id}: {metric} delta mismatch")


def _validate_result_identity(
    result: dict,
    row: dict,
    protocol_sha256: str,
    source_revision: str,
    upstream_commit: str,
) -> None:
    cell_id = row["id"]
    expected = {
        "status": "completed",
        "cell_id": cell_id,
        "method": "RAF",
        "base_method": "Chronos",
        "backbone": row["backbone"],
        "dataset": row["dataset"],
        "benchmark": row["benchmark"],
        "context_length": row["context_length"],
        "prediction_length": row["prediction_length"],
        "protocol_sha256": protocol_sha256,
        "source_revision": source_revision,
        "upstream_commit": upstream_commit,
    }
    for field, value in expected.items():
        if result.get(field) != value:
            raise ValueError(f"{cell_id}: result {field} drifted")
    _validate_result_metrics(result, cell_id)


def _validate_completed_status(
    status_path: Path,
    cell_id: str,
    protocol_sha256: str,
) -> str:
    status = load_json(status_path)
    if (
        status.get("status") != "completed"
        or status.get("cell_id") != cell_id
        or status.get("protocol_sha256") != protocol_sha256
    ):
        raise ValueError(f"{cell_id}: atomic completion status is invalid")
    return sha256_file(status_path)


def _validate_attempt(
    spec: AttemptSpec,
    *,
    is_final: bool,
    protocol_sha256: str,
    upstream_commit: str,
    canonical_rows: dict[str, dict],
    expected_source_revision: str | None,
) -> tuple[dict, list[dict], str]:
    if not spec.attempt_id.strip():
        raise ValueError("Attempt ID is missing")
    if spec.status not in ATTEMPT_STATUSES:
        raise ValueError(
            f"{spec.attempt_id}: status must be completed or interrupted"
        )

    summary_path = spec.summary_path.resolve()
    topology_path = spec.topology_path.resolve()
    summary = load_json(summary_path)
    topology = load_json(topology_path)
    if summary.get("method") != "RAF":
        raise ValueError(f"{spec.attempt_id}: summary is not RAF")
    if summary.get("protocol_sha256") != protocol_sha256:
        raise ValueError(f"{spec.attempt_id}: protocol hash drifted")
    declared_topology = summary.get("topology_path")
    if (
        not isinstance(declared_topology, str)
        or Path(declared_topology).resolve() != topology_path
    ):
        raise ValueError(f"{spec.attempt_id}: topology path drifted")

    source_revision = _require_nonempty_string(
        summary, "source_revision", spec.attempt_id
    )
    launcher_revision = _require_nonempty_string(
        summary, "launcher_revision", spec.attempt_id
    )
    if (
        expected_source_revision is not None
        and source_revision != expected_source_revision
    ):
        raise ValueError(f"{spec.attempt_id}: source revision drifted")
    if (
        summary.get("upstream_commit") is not None
        and summary.get("upstream_commit") != upstream_commit
    ):
        raise ValueError(f"{spec.attempt_id}: upstream commit drifted")

    worker = _worker_topology(topology, spec.attempt_id)
    nonzero_returncodes = [
        code for code in worker["returncodes_by_pid"].values() if code != 0
    ]
    if nonzero_returncodes and spec.status != "interrupted":
        raise ValueError(
            f"{spec.attempt_id}: nonzero return codes require explicit "
            "interrupted status"
        )
    if spec.status == "interrupted" and summary.get("all_completed") is True:
        raise ValueError(
            f"{spec.attempt_id}: interrupted attempt claims all_completed"
        )
    if is_final:
        if spec.status != "completed":
            raise ValueError("Final attempt must have completed status")
        if (
            summary.get("all_completed") is not True
            or summary.get("topology_valid") is not True
            or topology.get("topology_valid") is not True
            or nonzero_returncodes
        ):
            raise ValueError(
                f"{spec.attempt_id}: final attempt is not fully successful"
            )
        if (
            summary.get("failed_or_missing_pairs") != 0
            or summary.get("failures")
        ):
            raise ValueError(
                f"{spec.attempt_id}: final attempt retains failures"
            )

    inline_results = summary.get("results")
    if not isinstance(inline_results, list):
        raise ValueError(f"{spec.attempt_id}: results are missing")
    if summary.get("completed_pairs") != len(inline_results):
        raise ValueError(f"{spec.attempt_id}: completed pair count drifted")
    if is_final and summary.get("expected_pairs") != len(inline_results):
        raise ValueError(f"{spec.attempt_id}: final pair count drifted")

    topology_sha256 = sha256_file(topology_path)
    accepted = []
    accepted_ids = []
    for inline_result in inline_results:
        if not isinstance(inline_result, dict):
            raise ValueError(f"{spec.attempt_id}: invalid result record")
        cell_id = inline_result.get("cell_id")
        if cell_id not in canonical_rows:
            raise ValueError(f"{spec.attempt_id}: noncanonical RAF cell")
        result_path = (
            summary_path.parent / "cells" / cell_id / "result.json"
        ).resolve()
        status_path = result_path.with_name("status.json")
        raw_result = load_json(result_path)
        if raw_result != inline_result:
            raise ValueError(f"{cell_id}: summary result differs from artifact")
        _validate_result_identity(
            raw_result,
            canonical_rows[cell_id],
            protocol_sha256,
            source_revision,
            upstream_commit,
        )
        if (
            raw_result.get("launcher_revision") is not None
            and raw_result.get("launcher_revision") != launcher_revision
        ):
            raise ValueError(f"{cell_id}: result launcher revision drifted")
        status_sha256 = _validate_completed_status(
            status_path, cell_id, protocol_sha256
        )

        pid = raw_result.get("pid")
        if not isinstance(pid, int) or pid not in worker["returncodes_by_pid"]:
            raise ValueError(f"{cell_id}: result PID is not a topology worker")
        declared_gpu = _declared_result_gpu(raw_result, cell_id)
        if worker["bindings_by_pid"][pid] != declared_gpu:
            raise ValueError(f"{cell_id}: result GPU binding drifted")
        if worker["observed_by_pid"][pid] != {declared_gpu}:
            raise ValueError(f"{cell_id}: result PID/GPU was not observed")

        enriched = {
            **raw_result,
            "attempt_id": spec.attempt_id,
            "attempt_status": spec.status,
            "launcher_revision": launcher_revision,
            "result_path": str(result_path),
            "result_sha256": sha256_file(result_path),
            "status_path": str(status_path),
            "status_sha256": status_sha256,
            "topology_path": str(topology_path),
            "topology_sha256": topology_sha256,
        }
        accepted.append(enriched)
        accepted_ids.append(cell_id)

    attempt_record = {
        "attempt_id": spec.attempt_id,
        "status": spec.status,
        "summary_path": str(summary_path),
        "summary_sha256": sha256_file(summary_path),
        "topology_path": str(topology_path),
        "topology_sha256": topology_sha256,
        "source_revision": source_revision,
        "launcher_revision": launcher_revision,
        "upstream_commit": upstream_commit,
        "reported_expected_pairs": summary.get("expected_pairs"),
        "reported_completed_pairs": summary.get("completed_pairs"),
        "accepted_result_count": len(accepted),
        "accepted_cell_ids": sorted(accepted_ids),
        "topology_valid": topology.get("topology_valid"),
        "worker_pids": worker["pids_by_worker"],
        "worker_returncodes": {
            worker_id: int(topology["worker_returncodes"][worker_id])
            for worker_id in topology["worker_pids"]
        },
    }
    return attempt_record, accepted, source_revision


def compose_raf_attempts(
    protocol_path: Path,
    attempts: list[AttemptSpec],
) -> dict:
    if len(attempts) < 2:
        raise ValueError("RAF composite requires at least two attempts")
    attempt_ids = [attempt.attempt_id for attempt in attempts]
    if len(attempt_ids) != len(set(attempt_ids)):
        raise ValueError("RAF attempt IDs must be unique")

    protocol_path = protocol_path.resolve()
    protocol = load_json(protocol_path)
    protocol_sha256 = sha256_file(protocol_path)
    canonical_rows = canonical_raf_rows(protocol)
    upstream_commit = _require_nonempty_string(
        protocol["sources"]["raf"], "commit", "RAF protocol source"
    )

    attempt_records = []
    results = []
    source_revision = None
    seen: dict[str, str] = {}
    for index, spec in enumerate(attempts):
        record, accepted, attempt_source_revision = _validate_attempt(
            spec,
            is_final=index == len(attempts) - 1,
            protocol_sha256=protocol_sha256,
            upstream_commit=upstream_commit,
            canonical_rows=canonical_rows,
            expected_source_revision=source_revision,
        )
        if source_revision is None:
            source_revision = attempt_source_revision
        for result in accepted:
            cell_id = result["cell_id"]
            if cell_id in seen:
                raise ValueError(
                    f"Duplicate RAF cell {cell_id} in attempts "
                    f"{seen[cell_id]} and {spec.attempt_id}"
                )
            seen[cell_id] = spec.attempt_id
            results.append(result)
        attempt_records.append(record)

    expected_ids = set(canonical_rows)
    actual_ids = set(seen)
    if actual_ids != expected_ids:
        missing = sorted(expected_ids - actual_ids)
        extra = sorted(actual_ids - expected_ids)
        raise ValueError(
            "RAF composite does not exactly cover 44 canonical cells: "
            f"missing={missing} extra={extra}"
        )

    results.sort(key=lambda result: result["cell_id"])
    final_attempt = attempt_records[-1]
    inventory = [
        {
            "cell_id": result["cell_id"],
            "attempt_id": result["attempt_id"],
            "result_path": result["result_path"],
            "result_sha256": result["result_sha256"],
            "status_sha256": result["status_sha256"],
            "launcher_revision": result["launcher_revision"],
            "topology_sha256": result["topology_sha256"],
        }
        for result in results
    ]
    receipt_payload = {
        "schema_version": 1,
        "protocol_path": str(protocol_path),
        "protocol_sha256": protocol_sha256,
        "source_revision": source_revision,
        "upstream_commit": upstream_commit,
        "final_attempt_id": final_attempt["attempt_id"],
        "attempts": [
            {
                "attempt_id": record["attempt_id"],
                "status": record["status"],
                "summary_sha256": record["summary_sha256"],
                "topology_sha256": record["topology_sha256"],
                "accepted_result_count": record["accepted_result_count"],
            }
            for record in attempt_records
        ],
        "result_inventory_sha256": canonical_json_sha256(inventory),
        "result_count": len(results),
    }
    receipt = {
        **receipt_payload,
        "receipt_sha256": canonical_json_sha256(receipt_payload),
    }
    return {
        "schema_version": 2,
        "method": "RAF",
        "expected_pairs": 44,
        "completed_pairs": 44,
        "failed_or_missing_pairs": 0,
        "all_completed": True,
        "protocol_sha256": protocol_sha256,
        "source_revision": source_revision,
        "launcher_revision": final_attempt["launcher_revision"],
        "upstream_commit": upstream_commit,
        "topology_path": final_attempt["topology_path"],
        "topology_valid": True,
        "results": results,
        "failures": [],
        "composite_attempts": attempt_records,
        "receipt": receipt,
    }


def validate_composite_raf_summary(
    summary: dict,
    protocol_path: Path,
    *,
    expected_source_revision: str | None = None,
    expected_launcher_revision: str | None = None,
) -> list[Path]:
    if summary.get("schema_version") != 2:
        raise ValueError("RAF composite schema version drifted")
    receipt = summary.get("receipt")
    if not isinstance(receipt, dict):
        raise ValueError("RAF composite receipt is missing")
    recorded_protocol_path = Path(
        _require_nonempty_string(
            receipt, "protocol_path", "RAF composite receipt"
        )
    )
    if sha256_file(recorded_protocol_path) != sha256_file(protocol_path):
        raise ValueError("RAF composite protocol artifact drifted")
    records = summary.get("composite_attempts")
    if not isinstance(records, list) or len(records) < 2:
        raise ValueError("RAF composite attempts are missing")
    attempts = []
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("RAF composite attempt record is invalid")
        attempts.append(
            AttemptSpec(
                attempt_id=_require_nonempty_string(
                    record, "attempt_id", "RAF composite attempt"
                ),
                status=_require_nonempty_string(
                    record, "status", "RAF composite attempt"
                ),
                summary_path=Path(
                    _require_nonempty_string(
                        record, "summary_path", "RAF composite attempt"
                    )
                ),
                topology_path=Path(
                    _require_nonempty_string(
                        record, "topology_path", "RAF composite attempt"
                    )
                ),
            )
        )
    recomputed = compose_raf_attempts(recorded_protocol_path, attempts)
    if recomputed != summary:
        raise ValueError("RAF composite summary or receipt drifted")
    if (
        expected_source_revision is not None
        and summary.get("source_revision") != expected_source_revision
    ):
        raise ValueError("RAF composite source revision is not approved")
    if (
        expected_launcher_revision is not None
        and summary.get("launcher_revision") != expected_launcher_revision
    ):
        raise ValueError("RAF composite launcher revision is not approved")
    return [Path(record["topology_path"]) for record in records]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--attempt-a-id", required=True)
    parser.add_argument(
        "--attempt-a-status",
        required=True,
        choices=sorted(ATTEMPT_STATUSES),
    )
    parser.add_argument("--attempt-a-summary", required=True, type=Path)
    parser.add_argument("--attempt-a-topology", required=True, type=Path)
    parser.add_argument("--attempt-b-id", required=True)
    parser.add_argument(
        "--attempt-b-status",
        required=True,
        choices=sorted(ATTEMPT_STATUSES),
    )
    parser.add_argument("--attempt-b-summary", required=True, type=Path)
    parser.add_argument("--attempt-b-topology", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    document = compose_raf_attempts(
        args.protocol,
        [
            AttemptSpec(
                args.attempt_a_id,
                args.attempt_a_status,
                args.attempt_a_summary,
                args.attempt_a_topology,
            ),
            AttemptSpec(
                args.attempt_b_id,
                args.attempt_b_status,
                args.attempt_b_summary,
                args.attempt_b_topology,
            ),
        ],
    )
    save_json_atomic(args.output, document)


if __name__ == "__main__":
    main()
