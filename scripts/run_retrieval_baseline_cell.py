#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import socket
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch

from ts_rag.evaluation import compute_protocol_metrics
from ts_rag.matrix import load_manifest, save_json_atomic
from ts_rag.retrieval_baselines import (
    SYSTEM_IDS,
    load_protocol,
    load_replay_bundle,
    run_retrieval_baseline_cell,
)


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        while chunk := source.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _select_cell(rows: list[dict], cell_id: str) -> dict:
    matches = [row for row in rows if row["id"] == cell_id]
    if len(matches) != 1:
        raise ValueError(f"Expected one manifest row for {cell_id}")
    return matches[0]


def _select_expected(summary: dict, cell_id: str) -> dict:
    matches = [
        row for row in summary["cell_states"] if row["cell_id"] == cell_id
    ]
    if len(matches) != 1 or matches[0]["state"] != "completed":
        raise ValueError(f"Expected one completed A10G result for {cell_id}")
    return matches[0]


def _compare_base_metrics(
    actual: dict,
    expected: dict,
    metric_names: tuple[str, ...],
) -> dict:
    differences = {}
    relative_differences = {}
    for metric in metric_names:
        observed = float(actual[metric])
        reference = float(expected["test_baseline"][metric])
        if not math.isfinite(observed) or not math.isfinite(reference):
            raise ValueError(f"Non-finite base metric reference for {metric}")
        differences[metric] = observed - reference
        relative_differences[metric] = (
            (observed - reference) / abs(reference)
            if reference != 0
            else None
        )
    return {
        "comparison_only": True,
        "equality_required": False,
        "absolute_difference": differences,
        "relative_difference": relative_differences,
        "reference": expected["test_baseline"],
    }


def _runtime_device(cpu: bool, physical_gpu: int | None) -> tuple[str, dict]:
    if cpu:
        return "cpu", {
            "mode": "cpu",
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
        }
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(physical_gpu):
        raise ValueError("CUDA_VISIBLE_DEVICES does not match --physical-gpu")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("A GPU child must see exactly one CUDA device")
    if torch.cuda.current_device() != 0:
        raise RuntimeError("A GPU child must use local cuda:0")
    probe = torch.ones(1, device="cuda:0")
    torch.cuda.synchronize()
    return "cuda:0", {
        "mode": "cuda",
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "physical_gpu": physical_gpu,
        "local_gpu": 0,
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "device_name": torch.cuda.get_device_name(0),
        "probe": float(probe.item()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one frozen retrieval-baseline replay cell"
    )
    parser.add_argument("--cell-id", required=True)
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--catalog-sha256", required=True)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--a10g-summary", required=True, type=Path)
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--launcher-revision", required=True)
    parser.add_argument("--physical-gpu", type=int)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cell = _select_cell(load_manifest(args.manifest), args.cell_id)
    artifact_dir = (
        args.output_root
        / "cells"
        / cell["task_family"]
        / cell["dataset"]
        / cell["model"]
        / f"pl{cell['pred_len']}"
    )
    status_path = artifact_dir / "status.json"
    result_path = artifact_dir / "result.json"
    start = time.time()
    base_status = {
        "cell_id": cell["id"],
        "source_revision": args.source_revision,
        "launcher_revision": args.launcher_revision,
        "catalog_sha256": args.catalog_sha256,
        "protocol_sha256": args.protocol_sha256,
        "started_unix": start,
        "status": "running",
    }
    save_json_atomic(base_status, status_path)

    try:
        if sha256_file(args.catalog) != args.catalog_sha256:
            raise ValueError("Replay catalog hash drifted")
        if sha256_file(args.protocol) != args.protocol_sha256:
            raise ValueError("Retrieval protocol hash drifted")
        catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
        record = catalog["bundles"].get(cell["id"])
        if not record or record.get("usable") is not True:
            raise ValueError(f"Catalog has no usable bundle for {cell['id']}")
        bundle_path = args.project_root / record["path"]
        if bundle_path.stat().st_size != int(record["size_bytes"]):
            raise ValueError("Replay bundle size drifted")
        bundle_sha256 = sha256_file(bundle_path)
        if bundle_sha256 != record["sha256"]:
            raise ValueError("Replay bundle hash drifted")

        device, runtime = _runtime_device(args.cpu, args.physical_gpu)
        validation_bundle, test_bundle = load_replay_bundle(
            bundle_path,
            cell["task_family"],
        )
        protocol = load_protocol(args.protocol)
        metric_names = tuple(protocol["metrics"][cell["task_family"]])
        if metric_names != tuple(cell["metrics"]):
            raise ValueError("Protocol and manifest metrics differ")
        if test_bundle["y"].shape[1] != int(cell["pred_len"]):
            raise ValueError("Replay bundle prediction horizon drifted")

        expected_summary = json.loads(
            args.a10g_summary.read_text(encoding="utf-8")
        )
        expected = _select_expected(expected_summary, cell["id"])
        base_metrics = compute_protocol_metrics(
            test_bundle["y_base"],
            test_bundle["y"],
            cell["task_family"],
        )
        base_reference = _compare_base_metrics(
            base_metrics,
            expected,
            metric_names,
        )

        comparison = run_retrieval_baseline_cell(
            validation_bundle,
            test_bundle,
            cell["task_family"],
            metric_names,
            int(cell["pred_len"]),
            protocol,
            device,
        )
        if tuple(comparison["systems"]) != SYSTEM_IDS:
            raise AssertionError("Cell did not produce every frozen system")
        if comparison["systems"]["base"]["test_metrics"] != base_metrics:
            raise AssertionError("Base metric recomputation is inconsistent")

        elapsed = time.time() - start
        payload = {
            "schema_version": 1,
            "cell": {
                key: cell[key]
                for key in (
                    "id",
                    "task_family",
                    "dataset",
                    "model",
                    "pred_len",
                    "metrics",
                )
            },
            "source_revision": args.source_revision,
            "launcher_revision": args.launcher_revision,
            "catalog_sha256": args.catalog_sha256,
            "protocol_sha256": args.protocol_sha256,
            "bundle": {
                "path": str(bundle_path),
                "size_bytes": bundle_path.stat().st_size,
                "sha256": bundle_sha256,
            },
            "runtime": runtime,
            "base_metric_recomputation": {
                "passed": True,
                "metrics": base_metrics,
            },
            "a10g_metric_reference": base_reference,
            "comparison": comparison,
            "elapsed_seconds": elapsed,
        }
        save_json_atomic(payload, result_path)
        save_json_atomic(
            {
                **base_status,
                "status": "completed",
                "elapsed_seconds": elapsed,
                "result": str(result_path),
                "bundle_sha256": bundle_sha256,
            },
            status_path,
        )
        print(
            json.dumps(
                {
                    "cell_id": cell["id"],
                    "status": "completed",
                    "elapsed_seconds": elapsed,
                    "result": str(result_path),
                },
                sort_keys=True,
            )
        )
        return 0
    except BaseException as error:
        save_json_atomic(
            {
                **base_status,
                "status": "failed",
                "elapsed_seconds": time.time() - start,
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            },
            status_path,
        )
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
