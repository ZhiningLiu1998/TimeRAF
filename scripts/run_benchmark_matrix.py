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
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch

from ts_rag.matrix import (
    build_matrix_summary,
    load_manifest,
    resolve_source_revision,
    save_json_atomic,
)
from ts_rag.matrix_checkpoints import validate_selected_checkpoint_catalog


P4DE_INSTANCE_TYPE = "ml.p4de.24xlarge"
P4DE_GPUS_PER_HOST = 8
PROCESS_TERM_TIMEOUT_SECONDS = 10.0
PROCESS_GROUP_POLL_SECONDS = 0.1


def _filter_manifest(rows, args):
    selected = []
    for row in rows:
        if args.cell_id and row["id"] not in args.cell_id:
            continue
        if args.family and row["task_family"] not in args.family:
            continue
        if args.dataset and row["dataset"] not in args.dataset:
            continue
        if args.model and row["model"] not in args.model:
            continue
        if args.pred_len and row["pred_len"] not in args.pred_len:
            continue
        selected.append(row)
    return selected


def _sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        while block := source.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def _load_and_verify_checkpoint_catalog(path, checkpoint_root, manifest):
    path = Path(path)
    checkpoint_root = Path(checkpoint_root)
    with path.open("r", encoding="utf-8") as source:
        catalog = json.load(source)
    manifest_ids = [row["id"] for row in manifest]
    evidence = validate_selected_checkpoint_catalog(
        catalog,
        manifest_ids,
        expected_source_revision=catalog.get("source_revision"),
        require_hashes=True,
    )
    for cell_id, record in catalog["checkpoints"].items():
        checkpoint = checkpoint_root / record["relative_path"]
        if checkpoint.is_symlink() or not checkpoint.is_file():
            raise ValueError(f"Checkpoint is not a regular file: {cell_id}")
        if checkpoint.stat().st_size != record["file_size"]:
            raise ValueError(f"Checkpoint size drifted: {cell_id}")
        if _sha256_file(checkpoint) != record["sha256"]:
            raise ValueError(f"Checkpoint SHA-256 drifted: {cell_id}")
    return catalog, {
        **evidence,
        "catalog_path": str(path.resolve()),
        "catalog_sha256": _sha256_file(path),
        "checkpoint_root": str(checkpoint_root.resolve()),
        "file_hashes_recomputed": True,
    }


def _acquire_lock(output_root):
    lock_path = Path(output_root) / "matrix.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError(f"Another matrix runner holds {lock_path}") from error
    lock.seek(0)
    lock.truncate()
    lock.write(f"pid={os.getpid()} host={socket.gethostname()}\n")
    lock.flush()
    return lock


def _topology(args):
    visible_gpus = torch.cuda.device_count()
    worker_gpu_ids = []
    if args.cpu and args.instance_type == P4DE_INSTANCE_TYPE:
        raise ValueError("ml.p4de.24xlarge cannot run in CPU mode")
    if not args.cpu:
        if not torch.cuda.is_available() or visible_gpus < 1:
            raise RuntimeError("CUDA is required unless --cpu is set")
        if args.instance_type == P4DE_INSTANCE_TYPE and (
            args.gpu != 0
            or args.reserved_gpus_per_host != P4DE_GPUS_PER_HOST
            or args.processes_per_host != P4DE_GPUS_PER_HOST
            or visible_gpus != P4DE_GPUS_PER_HOST
        ):
            raise ValueError(
                "ml.p4de.24xlarge requires eight visible, reserved, and "
                "active worker GPUs numbered from zero"
            )
        if not 1 <= args.processes_per_host <= args.reserved_gpus_per_host:
            raise ValueError(
                "processes_per_host must be within the reserved GPU count"
            )
        worker_gpu_ids = list(
            range(args.gpu, args.gpu + args.processes_per_host)
        )
        if any(gpu_id >= visible_gpus for gpu_id in worker_gpu_ids):
            raise ValueError(
                "Worker GPU assignments must identify visible CUDA devices"
            )
        if visible_gpus > args.reserved_gpus_per_host:
            raise ValueError(
                "Visible CUDA devices exceed --reserved-gpus-per-host"
            )
    try:
        gpu_listing = subprocess.run(
            ["nvidia-smi", "-L"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        gpu_listing = None
    return {
        "recorded_unix": time.time(),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "instance_type": args.instance_type,
        "instance_count": 1,
        "reserved_gpus_per_host": args.reserved_gpus_per_host,
        "visible_cuda_devices": visible_gpus,
        "processes_per_host": args.processes_per_host,
        "launcher_mode": "independent_cell_workers",
        "worker_gpu_ids": worker_gpu_ids,
        "total_gpus": args.reserved_gpus_per_host,
        "world_size": args.processes_per_host,
        "inactive_reserved_gpus": (
            args.reserved_gpus_per_host
            - (0 if args.cpu else args.processes_per_host)
        ),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "nvidia_smi_list": gpu_listing,
    }


def _validate_queued_work(args, queued_cells):
    if (
        not args.cpu
        and not args.dry_run
        and args.instance_type == P4DE_INSTANCE_TYPE
        and 0 < queued_cells < P4DE_GPUS_PER_HOST
    ):
        raise ValueError(
            "ml.p4de.24xlarge requires at least eight queued cells so all "
            "eight A100 workers start with useful work"
        )


def _nvidia_process_map():
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
    return {
        int(pid.strip()): uuid.strip()
        for pid, uuid in (
            line.split(",", 1) for line in output.splitlines() if line.strip()
        )
    }


def _gpu_uuid_map():
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
    output_root,
    active_pids,
    active_lock,
    worker_gpu_ids,
    timeout=300.0,
):
    deadline = time.monotonic() + timeout
    expected_uuids = _gpu_uuid_map()
    expected_gpu_ids = sorted(worker_gpu_ids)
    while time.monotonic() < deadline:
        with active_lock:
            pids = dict(active_pids)
        process_map = _nvidia_process_map()
        if (
            sorted(pids) == expected_gpu_ids
            and all(
                process_map.get(pid) == expected_uuids[gpu]
                for gpu, pid in pids.items()
            )
        ):
            evidence = {
                "recorded_unix": time.time(),
                "expected_worker_count": len(expected_gpu_ids),
                "worker_bindings": [
                    {
                        "worker_gpu": gpu,
                        "pid": pid,
                        "gpu_uuid": process_map[pid],
                    }
                    for gpu, pid in sorted(pids.items())
                ],
                "distinct_positive_pids": (
                    len(set(pids.values())) == len(expected_gpu_ids)
                    and all(pid > 0 for pid in pids.values())
                ),
                "gpu_ids": sorted(pids),
                "all_bindings_observed": True,
            }
            save_json_atomic(
                evidence,
                Path(output_root) / "startup_topology.json",
            )
            return evidence
        time.sleep(2)
    raise RuntimeError(
        f"Did not observe {len(expected_gpu_ids)} distinct matrix GPU bindings"
    )


def _worker_process_config(cpu, gpu_id):
    environment = dict(os.environ)
    environment["PYTHONUNBUFFERED"] = "1"
    if cpu:
        return None, environment
    if gpu_id is None or gpu_id < 0:
        raise ValueError("GPU workers require a non-negative physical GPU ID")

    # Isolate the child before CUDA initializes; the assigned physical GPU is
    # then addressed as cuda:0 inside the worker.
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    return 0, environment


def _mirror_stdout(text):
    try:
        print(text, end="")
    except (OSError, ValueError):
        return False
    return True


def _process_group_exists(process_group_id):
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    return True


def _wait_for_process_group_exit(process, process_group_id, timeout):
    deadline = None if timeout is None else time.monotonic() + timeout
    while True:
        process.poll()
        if not _process_group_exists(process_group_id):
            return True
        if deadline is not None and time.monotonic() >= deadline:
            return False
        time.sleep(PROCESS_GROUP_POLL_SECONDS)


def _terminate_process_group(
    process,
    terminate_timeout=PROCESS_TERM_TIMEOUT_SECONDS,
):
    process_group_id = process.pid
    try:
        os.killpg(process_group_id, signal.SIGTERM)
    except ProcessLookupError:
        pass
    if not _wait_for_process_group_exit(
        process,
        process_group_id,
        terminate_timeout,
    ):
        try:
            os.killpg(process_group_id, signal.SIGKILL)
        except ProcessLookupError:
            pass
        # Stay fail-closed after SIGKILL: the scheduler must not reuse this
        # worker's GPU while any member of its process group remains alive.
        _wait_for_process_group_exit(process, process_group_id, None)
    process.wait()


def _write_summary(rows, args):
    summary = build_matrix_summary(
        rows,
        args.output_root,
        smoke=args.smoke,
        seed=args.seed,
        source_revision=args.source_revision,
        export_only=args.export_only,
    )
    save_json_atomic(summary, args.summary)
    return summary


def _run_cell(
    cell,
    args,
    gpu_id,
    active_pids=None,
    active_lock=None,
):
    active_pids = {} if active_pids is None else active_pids
    active_lock = threading.Lock() if active_lock is None else active_lock
    local_gpu_id, environment = _worker_process_config(args.cpu, gpu_id)
    command = [
        sys.executable,
        str(Path(__file__).resolve().parent / "run_benchmark_cell.py"),
        "--manifest",
        args.manifest,
        "--cell-id",
        cell["id"],
        "--seed",
        str(args.seed),
        "--source-revision",
        args.source_revision,
        "--output-root",
        args.output_root,
        "--checkpoint-root",
        args.checkpoint_root,
        "--num-workers",
        str(args.num_workers),
    ]
    if args.cpu:
        command.append("--cpu")
    else:
        command.extend(["--gpu", str(local_gpu_id)])
    if args.smoke:
        command.extend(
            [
                "--smoke",
                "--smoke-train-batches",
                str(args.smoke_train_batches),
                "--smoke-eval-samples",
                str(args.smoke_eval_samples),
            ]
        )
    if args.force_train:
        command.append("--force-train")
    if args.development_diagnostics:
        command.append("--development-diagnostics")
    if args.no_train:
        command.append("--no-train")
    if args.save_arrays:
        command.append("--save-arrays")
    if args.export_only:
        command.append("--export-only")
    for value in args.override:
        command.extend(["--override", value])
    release_record = args.release_checkpoints.get(cell["id"])
    if release_record is not None:
        checkpoint = Path(args.release_checkpoint_root) / release_record[
            "relative_path"
        ]
        if checkpoint.is_file():
            command.extend(["--checkpoint", str(checkpoint)])
        elif getattr(args, "release_only", False):
            raise ValueError(f"Catalog checkpoint missing: {checkpoint}")
        else:
            print(f"catalog checkpoint missing: {checkpoint}")
    elif getattr(args, "release_only", False):
        raise ValueError(f"Catalog has no checkpoint for {cell['id']}")

    log_path = Path(args.log_root) / f"{cell['id'].replace('/', '__')}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        header = (
            f"\n[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] "
            f"worker_gpu={gpu_id} local_gpu={local_gpu_id} "
            f"CUDA_VISIBLE_DEVICES="
            f"{environment.get('CUDA_VISIBLE_DEVICES')} "
            f"{shlex.join(command)}\n"
        )
        mirror_stdout = _mirror_stdout(header)
        log.write(header)
        process = None
        try:
            process = subprocess.Popen(
                command,
                cwd=args.working_root,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                start_new_session=True,
            )
            if gpu_id is not None:
                with active_lock:
                    active_pids[gpu_id] = process.pid
            assert process.stdout is not None
            for line in process.stdout:
                if mirror_stdout:
                    mirror_stdout = _mirror_stdout(line)
                log.write(line)
                log.flush()
            return process.wait()
        except BaseException:
            if process is not None:
                _terminate_process_group(process)
            raise
        finally:
            if gpu_id is not None:
                with active_lock:
                    active_pids.pop(gpu_id, None)
            if process is not None and process.stdout is not None:
                process.stdout.close()


def main():
    parser = argparse.ArgumentParser(
        description="Run the TimeFuse benchmark matrix on a GPU worker pool"
    )
    parser.add_argument(
        "--manifest", default="./docs/timefuse_experiment_manifest.jsonl"
    )
    parser.add_argument(
        "--output-root", default="./ts_rag_outputs/timefuse_matrix_full"
    )
    parser.add_argument(
        "--checkpoint-root", default="./checkpoints/timefuse_matrix_full"
    )
    parser.add_argument("--summary")
    parser.add_argument("--log-root")
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--source-revision")
    parser.add_argument("--launcher-revision")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-train-batches", type=int, default=1)
    parser.add_argument("--smoke-eval-samples", type=int, default=256)
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--no-train", action="store_true")
    parser.add_argument("--save-arrays", action="store_true")
    parser.add_argument("--export-only", action="store_true")
    parser.add_argument("--development-diagnostics", action="store_true")
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--rerun-completed", action="store_true")
    parser.add_argument("--max-cells", type=int, default=0)
    parser.add_argument("--max-failures", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--cell-id", action="append")
    parser.add_argument("--family", action="append")
    parser.add_argument("--dataset", action="append")
    parser.add_argument("--model", action="append")
    parser.add_argument("--pred-len", action="append", type=int)
    parser.add_argument("--instance-type", default="unknown")
    parser.add_argument("--reserved-gpus-per-host", type=int, default=1)
    parser.add_argument("--processes-per-host", type=int, default=1)
    parser.add_argument("--checkpoint-catalog")
    parser.add_argument("--release-checkpoint-root")
    parser.add_argument("--release-only", action="store_true")
    parser.add_argument("--working-root")
    args = parser.parse_args()
    args.source_revision = args.source_revision or resolve_source_revision()
    args.launcher_revision = (
        args.launcher_revision or resolve_source_revision()
    )
    args.working_root = str(
        Path(
            args.working_root or Path(__file__).resolve().parent.parent
        ).resolve()
    )
    if not Path(args.working_root).is_dir():
        parser.error("--working-root must identify an existing directory")

    args.summary = args.summary or str(
        Path(args.output_root) / "matrix_summary.json"
    )
    args.log_root = args.log_root or str(Path(args.output_root) / "logs")
    args.release_checkpoints = {}
    checkpoint_catalog_evidence = None
    if args.force_train and args.no_train:
        parser.error("--force-train and --no-train are mutually exclusive")
    if args.export_only:
        args.save_arrays = True
    manifest = load_manifest(args.manifest)
    if args.checkpoint_catalog:
        if not args.release_checkpoint_root:
            parser.error(
                "--release-checkpoint-root is required with --checkpoint-catalog"
            )
        catalog, checkpoint_catalog_evidence = (
            _load_and_verify_checkpoint_catalog(
                args.checkpoint_catalog,
                args.release_checkpoint_root,
                manifest,
            )
        )
        args.release_checkpoints = {
            cell_id: record
            for cell_id, record in catalog["checkpoints"].items()
            if record.get("usable") is True
        }
    rows = _filter_manifest(manifest, args)
    if args.release_only:
        if not args.checkpoint_catalog:
            parser.error("--release-only requires --checkpoint-catalog")
        rows = [
            row for row in rows if row["id"] in args.release_checkpoints
        ]
    if not rows:
        parser.error("No manifest cells match the requested filters")

    lock = _acquire_lock(args.output_root)
    topology = _topology(args)
    save_json_atomic(
        {
            "topology": topology,
            "manifest": args.manifest,
            "selected_cells": len(rows),
            "usable_release_checkpoints": len(args.release_checkpoints),
            "checkpoint_catalog_evidence": checkpoint_catalog_evidence,
            "seed": args.seed,
            "source_revision": args.source_revision,
            "launcher_revision": args.launcher_revision,
            "working_root": args.working_root,
            "smoke": args.smoke,
            "no_train": args.no_train,
            "save_arrays": args.save_arrays,
            "export_only": args.export_only,
            "overrides": args.override,
        },
        Path(args.output_root) / "run_metadata.json",
    )
    summary = _write_summary(rows, args)
    completed_ids = {
        row["cell_id"]
        for row in summary["cell_states"]
        if row["state"] == "completed"
    }

    cells_to_run = []
    for cell in rows:
        if (
            cell["id"] in completed_ids
            and not args.rerun_completed
            and not args.force_train
        ):
            print(f"skip completed {cell['id']}")
            continue
        if args.max_cells and len(cells_to_run) >= args.max_cells:
            break
        cells_to_run.append(cell)
    _validate_queued_work(args, len(cells_to_run))

    attempted = 0
    failures = 0
    if args.dry_run:
        for cell in cells_to_run:
            print(f"[{attempted + 1}/{len(rows)}] {cell['id']}")
            attempted += 1
    else:
        worker_gpu_ids = (
            [None] * args.processes_per_host
            if args.cpu
            else topology["worker_gpu_ids"]
        )
        available_workers = list(worker_gpu_ids)
        remaining = iter(cells_to_run)
        active = {}
        active_pids = {}
        active_lock = threading.Lock()
        exhausted = False
        stop_scheduling = False
        startup_captured = args.cpu

        with ThreadPoolExecutor(
            max_workers=args.processes_per_host
        ) as executor:
            while active or not exhausted:
                while available_workers and not exhausted and not stop_scheduling:
                    try:
                        cell = next(remaining)
                    except StopIteration:
                        exhausted = True
                        break
                    gpu_id = available_workers.pop(0)
                    print(
                        f"[{attempted + 1}/{len(rows)}] {cell['id']} "
                        f"worker_gpu={gpu_id}"
                    )
                    future = executor.submit(
                        _run_cell,
                        cell,
                        args,
                        gpu_id,
                        active_pids,
                        active_lock,
                    )
                    active[future] = (cell, gpu_id)
                    attempted += 1

                if (
                    not startup_captured
                    and len(active) == args.processes_per_host
                ):
                    _capture_startup_topology(
                        args.output_root,
                        active_pids,
                        active_lock,
                        topology["worker_gpu_ids"],
                    )
                    startup_captured = True

                if not active:
                    break

                done, _ = wait(active, return_when=FIRST_COMPLETED)
                for future in done:
                    cell, gpu_id = active.pop(future)
                    available_workers.append(gpu_id)
                    try:
                        return_code = future.result()
                    except Exception as error:
                        return_code = 1
                        print(
                            f"worker failed for {cell['id']}: "
                            f"{type(error).__name__}: {error}"
                        )
                    if return_code != 0:
                        failures += 1
                        print(f"cell failed rc={return_code}: {cell['id']}")
                    summary = _write_summary(rows, args)
                    if args.max_failures and failures >= args.max_failures:
                        stop_scheduling = True

                if stop_scheduling and active:
                    print(
                        f"stopping new work after {failures} failed cell(s); "
                        f"waiting for {len(active)} active worker(s)"
                    )

    summary = _write_summary(rows, args)
    counts = summary["counts"]
    print(
        json.dumps(
            {
                "attempted_this_run": attempted,
                "failures_this_run": failures,
                "counts": counts,
                "all_completed": summary["all_completed"],
                "all_improved": summary["all_improved"],
                "development_gate_passed": summary["publication_gate"][
                    "development_gate_passed"
                ],
                "summary": args.summary,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
