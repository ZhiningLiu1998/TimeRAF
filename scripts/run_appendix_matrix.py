import argparse
import fcntl
import json
import os
import shlex
import socket
import subprocess
import sys
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from ts_rag.appendix_matrix import (
    APPENDIX_METHOD_REVISION,
    APPENDIX_SELECTOR_POLICY,
    APPENDIX_SELECTOR_PROTOCOL_SHA256,
    build_appendix_summary,
    load_bundle_catalog,
    validate_catalog_scope,
)
from ts_rag.appendix_rag import load_appendix_manifest, sha256_file
from ts_rag.matrix import resolve_source_revision, save_json_atomic


def _filter_manifest(rows, args):
    selected = []
    for row in rows:
        if args.cell_id and row["id"] not in args.cell_id:
            continue
        if args.family and row["task_family"] not in args.family:
            continue
        if args.dataset and row["dataset"] not in args.dataset:
            continue
        if args.baseline and row["baseline"] not in args.baseline:
            continue
        selected.append(row)
    return selected


def _acquire_lock(output_root):
    path = Path(output_root) / "appendix_matrix.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError(f"Another Appendix runner holds {path}") from error
    lock.seek(0)
    lock.truncate()
    lock.write(f"pid={os.getpid()} host={socket.gethostname()}\n")
    lock.flush()
    return lock


def _write_summary(rows, args):
    summary = build_appendix_summary(
        rows,
        args.output_root,
        source_revision=args.source_revision,
        method_revision=args.method_revision,
    )
    summary["execution_topology"] = {
        "instance_type": args.instance_type,
        "instance_count": 1,
        "reserved_gpus_per_host": args.reserved_gpus_per_host,
        "active_gpu_workers": 0,
        "inactive_reserved_gpus": args.inactive_reserved_gpus,
        "cpu_workers": args.workers,
        "gpu_inactivity_reason": args.gpu_inactivity_reason,
    }
    summary["launcher_revision"] = args.launcher_revision
    summary["selector_policy"] = APPENDIX_SELECTOR_POLICY
    summary["selector_protocol_sha256"] = (
        APPENDIX_SELECTOR_PROTOCOL_SHA256
    )
    save_json_atomic(summary, args.summary)
    return summary


def _run_cell(cell, bundle_record, args):
    command = [
        sys.executable,
        "scripts/run_appendix_rag_cell.py",
        "--manifest",
        args.manifest,
        "--cell-id",
        cell["id"],
        "--output-root",
        args.output_root,
        "--source-revision",
        args.source_revision,
        "--method-revision",
        args.method_revision,
    ]
    if cell["paper_status"]["evaluable"]:
        bundle_path = Path(bundle_record["path"])
        if not bundle_path.is_absolute():
            bundle_path = Path(args.project_root) / bundle_path
        command.extend(["--prediction-bundle", str(bundle_path.resolve())])
    if args.development_diagnostics:
        command.append("--development-diagnostics")
    if args.save_arrays:
        command.append("--save-arrays")

    log_path = Path(args.log_root) / (
        cell["id"].replace("/", "__") + ".log"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        header = f"\n{shlex.join(command)}\n"
        print(header, end="")
        log.write(header)
        process = subprocess.Popen(
            command,
            cwd=Path(__file__).resolve().parent.parent,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
            log.flush()
        return process.wait()


def main():
    parser = argparse.ArgumentParser(
        description="Run validation-selected RAG over Appendix baselines"
    )
    parser.add_argument(
        "--manifest",
        default="./docs/timefuse_appendix_experiment_manifest.jsonl",
    )
    parser.add_argument("--bundle-catalog", required=True)
    parser.add_argument("--project-root", default=".")
    parser.add_argument(
        "--output-root",
        default="./ts_rag_outputs/timefuse_appendix_rag",
    )
    parser.add_argument("--summary")
    parser.add_argument("--log-root")
    parser.add_argument("--source-revision")
    parser.add_argument("--launcher-revision", required=True)
    parser.add_argument(
        "--method-revision",
        default=APPENDIX_METHOD_REVISION,
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--instance-type", default="unreserved-cpu")
    parser.add_argument("--reserved-gpus-per-host", type=int, default=0)
    parser.add_argument("--inactive-reserved-gpus", type=int, default=0)
    parser.add_argument("--gpu-inactivity-reason")
    parser.add_argument("--max-failures", type=int, default=1)
    parser.add_argument("--max-cells", type=int, default=0)
    parser.add_argument("--rerun-completed", action="store_true")
    parser.add_argument("--development-diagnostics", action="store_true")
    parser.add_argument("--save-arrays", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--cell-id", action="append")
    parser.add_argument("--family", action="append")
    parser.add_argument("--dataset", action="append")
    parser.add_argument("--baseline", action="append")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if (
        args.reserved_gpus_per_host < 0
        or args.inactive_reserved_gpus < 0
        or args.inactive_reserved_gpus != args.reserved_gpus_per_host
    ):
        parser.error(
            "Appendix RAG has no GPU workers, so every reserved GPU must be "
            "recorded as inactive"
        )
    if args.reserved_gpus_per_host and not args.gpu_inactivity_reason:
        parser.error(
            "--gpu-inactivity-reason is required when GPUs are reserved"
        )
    if args.method_revision != APPENDIX_METHOD_REVISION:
        parser.error(
            "Appendix method revision must remain frozen at "
            f"{APPENDIX_METHOD_REVISION}"
        )
    if not (
        len(args.launcher_revision) == 40
        and all(
            character in "0123456789abcdef"
            for character in args.launcher_revision
        )
    ):
        parser.error("--launcher-revision must be a full lowercase commit SHA")
    args.source_revision = args.source_revision or resolve_source_revision()
    args.project_root = str(Path(args.project_root).resolve())
    args.summary = args.summary or str(
        Path(args.output_root) / "appendix_summary.json"
    )
    args.log_root = args.log_root or str(Path(args.output_root) / "logs")

    rows = _filter_manifest(load_appendix_manifest(args.manifest), args)
    if not rows:
        parser.error("No Appendix cells match the requested filters")
    catalog = load_bundle_catalog(
        args.bundle_catalog,
        project_root=args.project_root,
    )
    validate_catalog_scope(
        rows,
        catalog,
        require_files=not args.dry_run,
        allow_extra=True,
    )
    lock = _acquire_lock(args.output_root)

    save_json_atomic(
        {
            "manifest": args.manifest,
            "project_root": args.project_root,
            "bundle_catalog": args.bundle_catalog,
            "bundle_catalog_sha256": sha256_file(args.bundle_catalog),
            "bundle_source_revision": catalog.get("source_revision"),
            "selected_cells": len(rows),
            "source_revision": args.source_revision,
            "launcher_revision": args.launcher_revision,
            "method_revision": args.method_revision,
            "selector_policy": APPENDIX_SELECTOR_POLICY,
            "selector_protocol_sha256": (
                APPENDIX_SELECTOR_PROTOCOL_SHA256
            ),
            "workers": args.workers,
            "launcher_mode": "independent_cpu_appendix_rag_cells",
            "execution_topology": {
                "instance_type": args.instance_type,
                "instance_count": 1,
                "reserved_gpus_per_host": args.reserved_gpus_per_host,
                "active_gpu_workers": 0,
                "inactive_reserved_gpus": args.inactive_reserved_gpus,
                "cpu_workers": args.workers,
                "gpu_inactivity_reason": args.gpu_inactivity_reason,
            },
        },
        Path(args.output_root) / "run_metadata.json",
    )
    summary = _write_summary(rows, args)
    finished = {
        row["cell_id"]
        for row in summary["cell_states"]
        if row["state"] in {"completed", "paper_oot"}
    }
    cells = [
        cell
        for cell in rows
        if args.rerun_completed or cell["id"] not in finished
    ]
    if args.max_cells:
        cells = cells[: args.max_cells]

    if args.dry_run:
        for cell in cells:
            print(cell["id"])
        return 0

    failures = 0
    attempted = 0
    iterator = iter(cells)
    active = {}
    exhausted = False
    stop_scheduling = False
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        while active or not exhausted:
            while (
                len(active) < args.workers
                and not exhausted
                and not stop_scheduling
            ):
                try:
                    cell = next(iterator)
                except StopIteration:
                    exhausted = True
                    break
                record = catalog["bundles"].get(cell["id"], {})
                future = executor.submit(_run_cell, cell, record, args)
                active[future] = cell
                attempted += 1
            if not active:
                break
            done, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in done:
                cell = active.pop(future)
                try:
                    return_code = future.result()
                except Exception as error:
                    return_code = 1
                    print(
                        f"worker failed for {cell['id']}: "
                        f"{type(error).__name__}: {error}"
                    )
                if return_code:
                    failures += 1
                _write_summary(rows, args)
                if args.max_failures and failures >= args.max_failures:
                    stop_scheduling = True

    summary = _write_summary(rows, args)
    print(
        json.dumps(
            {
                "attempted_this_run": attempted,
                "failures_this_run": failures,
                "counts": summary["counts"],
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
