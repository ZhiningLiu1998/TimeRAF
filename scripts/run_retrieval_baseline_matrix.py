#!/usr/bin/env python3

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from statistics import fmean

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch

from ts_rag.matrix import load_manifest, resolve_source_revision, save_json_atomic
from ts_rag.retrieval_baselines import SYSTEM_IDS, load_protocol


EXPECTED_CELLS = 585


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        while chunk := source.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _acquire_lock(output_root: Path):
    path = output_root / "matrix.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError(f"Another baseline runner holds {path}") from error
    lock.seek(0)
    lock.truncate()
    lock.write(f"pid={os.getpid()} host={socket.gethostname()}\n")
    lock.flush()
    return lock


def _validate_scope(
    catalog: dict,
    rows: list[dict],
    protocol: dict,
) -> list[dict]:
    if len(rows) != EXPECTED_CELLS:
        raise ValueError(f"Expected {EXPECTED_CELLS} manifest rows")
    row_ids = {row["id"] for row in rows}
    if len(row_ids) != EXPECTED_CELLS:
        raise ValueError("Replay manifest contains duplicate cell IDs")
    records = catalog.get("bundles", {})
    if set(records) != row_ids:
        raise ValueError("Replay catalog and manifest cell IDs differ")
    if (
        catalog.get("usable_count") != EXPECTED_CELLS
        or any(record.get("usable") is not True for record in records.values())
    ):
        raise ValueError(
            f"Replay catalog does not contain {EXPECTED_CELLS} usable bundles"
        )
    if int(protocol["scope"]["expected_cells"]) != EXPECTED_CELLS:
        raise ValueError("Protocol expected-cell count drifted")
    expected_horizons = {
        family: sorted(
            {
                int(row["pred_len"])
                for row in rows
                if row["task_family"] == family
            }
        )
        for family in ("long_term", "pems", "epf")
    }
    protocol_horizons = {
        family: sorted(int(value) for value in values)
        for family, values in protocol["scope"]["horizons"].items()
    }
    if protocol_horizons != expected_horizons:
        raise ValueError("Protocol and manifest horizons differ")
    return sorted(rows, key=lambda row: row["id"])


def _topology(args: argparse.Namespace) -> dict:
    visible = torch.cuda.device_count()
    if args.cpu:
        return {
            "mode": "cpu",
            "visible_cuda_devices": visible,
            "processes_per_host": args.processes_per_host,
        }
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required unless --cpu is set")
    if (
        args.reserved_gpus_per_host != visible
        or args.processes_per_host != visible
        or visible not in (4, 8)
    ):
        raise ValueError(
            "A retrieval matrix requires one worker on every reserved GPU "
            "of a supported four- or eight-GPU host"
        )
    worker_gpu_ids = list(range(visible))
    listing = subprocess.run(
        ["nvidia-smi", "-L"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {
        "recorded_unix": time.time(),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "instance_type": args.instance_type,
        "instance_count": 1,
        "reserved_gpus_per_host": args.reserved_gpus_per_host,
        "visible_cuda_devices": visible,
        "processes_per_host": args.processes_per_host,
        "worker_gpu_ids": worker_gpu_ids,
        "total_gpus": args.reserved_gpus_per_host,
        "world_size": args.processes_per_host,
        "inactive_reserved_gpus": 0,
        "child_device": "cuda:0",
        "nvidia_smi_list": listing,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
    }


def _artifact_dir(output_root: Path, cell: dict) -> Path:
    return (
        output_root
        / "cells"
        / cell["task_family"]
        / cell["dataset"]
        / cell["model"]
        / f"pl{cell['pred_len']}"
    )


def _load_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _completed(
    output_root: Path,
    cell: dict,
    source_revision: str,
    protocol_sha256: str,
    catalog_sha256: str,
) -> bool:
    artifact = _artifact_dir(output_root, cell)
    status = _load_json(artifact / "status.json")
    result = _load_json(artifact / "result.json")
    return bool(
        status
        and result
        and status.get("status") == "completed"
        and status.get("source_revision") == source_revision
        and status.get("protocol_sha256") == protocol_sha256
        and status.get("catalog_sha256") == catalog_sha256
        and set(result.get("comparison", {}).get("systems", {}))
        == set(SYSTEM_IDS)
        and len(result["comparison"]["systems"]) == len(SYSTEM_IDS)
    )


def _cell_command(args, cell, protocol_sha256, catalog_sha256):
    command = [
        sys.executable,
        str(Path(__file__).with_name("run_retrieval_baseline_cell.py")),
        "--cell-id",
        cell["id"],
        "--catalog",
        str(args.catalog),
        "--catalog-sha256",
        catalog_sha256,
        "--manifest",
        str(args.manifest),
        "--protocol",
        str(args.protocol),
        "--protocol-sha256",
        protocol_sha256,
        "--a10g-summary",
        str(args.a10g_summary),
        "--project-root",
        str(args.project_root),
        "--output-root",
        str(args.output_root),
        "--source-revision",
        args.source_revision,
        "--launcher-revision",
        args.launcher_revision,
    ]
    if args.cpu:
        command.append("--cpu")
    return command


def _terminate(process: subprocess.Popen) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=10)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def _run_cell(
    args,
    cell,
    gpu_id,
    protocol_sha256,
    catalog_sha256,
    active_pids,
    active_lock,
):
    command = _cell_command(args, cell, protocol_sha256, catalog_sha256)
    if not args.cpu:
        command.extend(["--physical-gpu", str(gpu_id)])
    environment = dict(os.environ)
    environment["PYTHONUNBUFFERED"] = "1"
    if not args.cpu:
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    log_path = args.output_root / "logs" / f"{cell['id'].replace('/', '__')}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        header = (
            f"\n[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] "
            f"worker_gpu={gpu_id} command={shlex.join(command)}\n"
        )
        log.write(header)
        log.flush()
        process = None
        try:
            process = subprocess.Popen(
                command,
                cwd=args.source_root,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            with active_lock:
                active_pids[gpu_id] = process.pid
            return process.wait()
        except BaseException:
            if process is not None:
                _terminate(process)
            raise
        finally:
            with active_lock:
                active_pids.pop(gpu_id, None)


def _nvidia_process_map() -> dict[int, str]:
    output = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,gpu_uuid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    mapping = {}
    for line in output.splitlines():
        if not line.strip():
            continue
        pid, uuid = [value.strip() for value in line.split(",", 1)]
        mapping[int(pid)] = uuid
    return mapping


def _gpu_uuid_map() -> dict[int, str]:
    output = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {
        int(index.strip()): uuid.strip()
        for index, uuid in (
            line.split(",", 1) for line in output.splitlines() if line.strip()
        )
    }


def _capture_startup_topology(
    output_root: Path,
    active_pids: dict,
    active_lock: threading.Lock,
    worker_count: int,
    timeout: float = 180.0,
) -> dict:
    deadline = time.monotonic() + timeout
    expected_uuids = _gpu_uuid_map()
    while time.monotonic() < deadline:
        with active_lock:
            pids = dict(active_pids)
        process_map = _nvidia_process_map()
        if len(pids) == worker_count and all(
            process_map.get(pid) == expected_uuids[gpu]
            for gpu, pid in pids.items()
        ):
            evidence = {
                "recorded_unix": time.time(),
                "worker_bindings": [
                    {
                        "worker_gpu": gpu,
                        "pid": pid,
                        "gpu_uuid": process_map[pid],
                    }
                    for gpu, pid in sorted(pids.items())
                ],
                "expected_worker_count": worker_count,
                "distinct_positive_pids": len(set(pids.values()))
                == worker_count
                and all(pid > 0 for pid in pids.values()),
                "gpu_ids": sorted(pids),
                "all_bindings_observed": True,
            }
            save_json_atomic(evidence, output_root / "startup_topology.json")
            return evidence
        time.sleep(2)
    raise RuntimeError(
        f"Did not observe {worker_count} distinct baseline GPU bindings"
    )


def build_summary(rows: list[dict], output_root: Path, identities: dict) -> dict:
    states = []
    by_system = defaultdict(list)
    for cell in rows:
        artifact = _artifact_dir(output_root, cell)
        status = _load_json(artifact / "status.json")
        result = _load_json(artifact / "result.json")
        if not status:
            state = "pending"
        else:
            state = status.get("status", "incomplete")
        row = {
            "cell_id": cell["id"],
            "task_family": cell["task_family"],
            "dataset": cell["dataset"],
            "model": cell["model"],
            "pred_len": cell["pred_len"],
            "state": state,
            "artifact_dir": str(artifact),
        }
        if state == "completed" and result:
            systems = result["comparison"]["systems"]
            if set(systems) != set(SYSTEM_IDS) or len(systems) != len(SYSTEM_IDS):
                row["state"] = "incomplete"
            else:
                row["systems"] = systems
                row["elapsed_seconds"] = result["elapsed_seconds"]
                row["base_metric_recomputation"] = result[
                    "base_metric_recomputation"
                ]
                row["a10g_metric_reference"] = result[
                    "a10g_metric_reference"
                ]
                row["no_lookahead"] = result["comparison"]["no_lookahead"]
                for system_id, system in systems.items():
                    by_system[system_id].append((cell, system))
        elif state == "failed":
            row["error_type"] = status.get("error_type")
            row["error"] = status.get("error")
        states.append(row)

    counts = {
        name: sum(row["state"] == name for row in states)
        for name in ("completed", "failed", "running", "pending", "incomplete")
    }
    system_summary = {}
    for system_id in SYSTEM_IDS:
        entries = by_system[system_id]
        gains = defaultdict(list)
        for _, system in entries:
            for metric, value in system["metric_gain_percent"].items():
                gains[metric].append(float(value))
        system_summary[system_id] = {
            "completed_cells": len(entries),
            "strict_wins": sum(
                bool(system["strict_win"]) for _, system in entries
            ),
            "mean_paired_gain_percent": {
                metric: fmean(values) for metric, values in sorted(gains.items())
            },
            "by_task_family": {
                family: {
                    "cells": sum(
                        cell["task_family"] == family for cell, _ in entries
                    ),
                    "strict_wins": sum(
                        cell["task_family"] == family
                        and bool(system["strict_win"])
                        for cell, system in entries
                    ),
                }
                for family in ("long_term", "pems", "epf")
            },
        }
    return {
        "schema_version": 1,
        **identities,
        "expected_cells": len(rows),
        "expected_systems": list(SYSTEM_IDS),
        "counts": counts,
        "all_completed": counts["completed"] == len(rows),
        "cell_states": states,
        "system_summary": system_summary,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the frozen 585-cell retrieval-baseline matrix"
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("docs/timefuse_experiment_manifest.jsonl"),
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("docs/retrieval_baseline_protocol.json"),
    )
    parser.add_argument(
        "--a10g-summary",
        type=Path,
        default=Path("docs/publication_results/a10g_composed_summary.json"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/retrieval_baselines/full-horizon-585"),
    )
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--source-revision")
    parser.add_argument("--launcher-revision")
    parser.add_argument("--instance-type", default="ml.p5.48xlarge")
    parser.add_argument("--reserved-gpus-per-host", type=int, default=8)
    parser.add_argument("--processes-per-host", type=int, default=8)
    parser.add_argument("--cell-id", action="append")
    parser.add_argument("--max-cells", type=int, default=0)
    parser.add_argument("--rerun-completed", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.source_root = (args.source_root or Path.cwd()).resolve()
    args.project_root = (args.project_root or Path.cwd()).resolve()
    args.catalog = args.catalog.resolve()
    args.manifest = args.manifest.resolve()
    args.protocol = args.protocol.resolve()
    args.a10g_summary = args.a10g_summary.resolve()
    args.output_root = args.output_root.resolve()
    args.source_revision = args.source_revision or resolve_source_revision()
    args.launcher_revision = args.launcher_revision or args.source_revision

    catalog_sha256 = sha256_file(args.catalog)
    protocol_sha256 = sha256_file(args.protocol)
    protocol = load_protocol(args.protocol)
    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
    rows = _validate_scope(catalog, load_manifest(args.manifest), protocol)
    if args.cell_id:
        requested = set(args.cell_id)
        unknown = requested - {row["id"] for row in rows}
        if unknown:
            raise ValueError(f"Unknown requested cells: {sorted(unknown)}")
        rows_to_consider = [row for row in rows if row["id"] in requested]
    else:
        rows_to_consider = list(rows)

    lock = _acquire_lock(args.output_root)
    topology = _topology(args)
    identities = {
        "source_revision": args.source_revision,
        "launcher_revision": args.launcher_revision,
        "catalog": str(args.catalog),
        "catalog_sha256": catalog_sha256,
        "protocol": str(args.protocol),
        "protocol_sha256": protocol_sha256,
        "manifest": str(args.manifest),
        "a10g_summary": str(args.a10g_summary),
        "source_root": str(args.source_root),
        "project_root": str(args.project_root),
    }
    save_json_atomic(
        {
            **identities,
            "recorded_unix": time.time(),
            "topology": topology,
            "selected_cells": len(rows_to_consider),
            "module_sha256": sha256_file(
                args.source_root / "ts_rag/retrieval_baselines.py"
            ),
            "cell_runner_sha256": sha256_file(
                args.source_root / "scripts/run_retrieval_baseline_cell.py"
            ),
            "matrix_runner_sha256": sha256_file(Path(__file__)),
        },
        args.output_root / "run_metadata.json",
    )
    save_json_atomic(
        build_summary(rows, args.output_root, identities),
        args.output_root / "matrix_summary.json",
    )

    pending = [
        row
        for row in rows_to_consider
        if args.rerun_completed
        or not _completed(
            args.output_root,
            row,
            args.source_revision,
            protocol_sha256,
            catalog_sha256,
        )
    ]
    if args.max_cells:
        pending = pending[: args.max_cells]
    if args.dry_run:
        for row in pending:
            print(row["id"])
        return 0
    if not pending:
        print("No retrieval baseline cells are pending")
        return 0
    if (
        not args.cpu
        and len(pending) < args.processes_per_host
        and not args.cell_id
        and not (args.output_root / "startup_topology.json").is_file()
    ):
        raise ValueError(
            "A GPU matrix launch requires enough pending cells to occupy "
            "every worker unless an earlier launch in this output root "
            "already recorded full startup topology"
        )

    worker_ids = (
        [None] * args.processes_per_host
        if args.cpu
        else list(range(args.processes_per_host))
    )
    available = list(worker_ids)
    remaining = iter(pending)
    active = {}
    active_pids = {}
    active_lock = threading.Lock()
    exhausted = False
    failures = 0
    startup_captured = (
        args.cpu or (args.output_root / "startup_topology.json").is_file()
    )
    with ThreadPoolExecutor(max_workers=args.processes_per_host) as executor:
        while active or not exhausted:
            while available and not exhausted and failures == 0:
                try:
                    cell = next(remaining)
                except StopIteration:
                    exhausted = True
                    break
                gpu_id = available.pop(0)
                print(f"launch {cell['id']} worker_gpu={gpu_id}", flush=True)
                future = executor.submit(
                    _run_cell,
                    args,
                    cell,
                    gpu_id,
                    protocol_sha256,
                    catalog_sha256,
                    active_pids,
                    active_lock,
                )
                active[future] = (cell, gpu_id)

            if (
                not startup_captured
                and len(active) == args.processes_per_host
            ):
                _capture_startup_topology(
                    args.output_root,
                    active_pids,
                    active_lock,
                    args.processes_per_host,
                )
                startup_captured = True
            if not active:
                break
            done, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in done:
                cell, gpu_id = active.pop(future)
                available.append(gpu_id)
                try:
                    return_code = future.result()
                except BaseException as error:
                    return_code = 1
                    print(
                        f"worker exception {cell['id']}: "
                        f"{type(error).__name__}: {error}",
                        flush=True,
                    )
                if return_code:
                    failures += 1
                    print(f"failed {cell['id']} rc={return_code}", flush=True)
                else:
                    print(f"completed {cell['id']}", flush=True)
                save_json_atomic(
                    build_summary(rows, args.output_root, identities),
                    args.output_root / "matrix_summary.json",
                )

    summary = build_summary(rows, args.output_root, identities)
    save_json_atomic(summary, args.output_root / "matrix_summary.json")
    print(
        json.dumps(
            {
                "failures_this_run": failures,
                "counts": summary["counts"],
                "all_completed": summary["all_completed"],
                "summary": str(args.output_root / "matrix_summary.json"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
