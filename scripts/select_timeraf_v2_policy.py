#!/usr/bin/env python3
"""Evaluate selection policies over dumped validation candidates.

The candidate search is the expensive part of a fixed-forecast cell, and every
policy considered here reads only validation quantities. This script therefore
reuses one dumped candidate pool per cell and applies exactly one selected
candidate per policy to the test split.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import multiprocessing as mp
import os
import socket
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from scripts.run_timeraf_v2_matrix import (
    _no_lookahead_check,
    _prediction_sha256,
    _rung_admits,
    sha256_file,
    sha256_text,
    save_json_atomic,
    summarize,
)
from ts_rag.benchmark_rag import apply_selected_rag, selection_policy_from_dict
from ts_rag.retrieval_baselines import load_replay_bundle

def choose(rows, policy, pred_len=None):
    pool = [
        row for row in rows if _rung_admits(row, {"portfolio": policy["portfolio"]})
    ]
    selected = selection_policy_from_dict(
        {key: value for key, value in policy.items() if key not in ("id", "portfolio")}
    ).select(pool, pred_len=pred_len)
    return selected, len(pool)


def run_cell(job):
    from ts_rag.benchmark_rag import protocol_metrics

    started = time.time()
    cell_id = job["cell_id"]
    output_dir = Path(job["output_root"]) / "cells" / cell_id.replace("/", "__")
    result_path = output_dir / "result.json"
    if result_path.exists() and not job["force"]:
        try:
            existing = json.loads(result_path.read_text())
            if existing.get("state") == "completed" and set(
                existing.get("rungs", {})
            ) == {policy["id"] for policy in job["policies"]}:
                return existing
        except (json.JSONDecodeError, OSError):
            pass

    try:
        with gzip.open(job["candidate_path"], "rt") as handle:
            dump = json.load(handle)
        if dump["cell_id"] != cell_id:
            raise ValueError("Candidate dump cell id mismatch")

        task_family = dump["task_family"]
        metric_names = tuple(dump["metric_names"])
        pred_len = int(dump["pred_len"])
        bundle_path = Path(job["artifact_root"]) / dump["bundle_path"]
        digest = sha256_file(bundle_path)
        if digest != dump["bundle_sha256"]:
            raise ValueError(f"Prediction bundle hash drifted for {cell_id}")
        validation_bundle, test_bundle = load_replay_bundle(bundle_path, task_family)
        base_metrics = protocol_metrics(
            test_bundle, test_bundle["y_base"], task_family
        )
        if dump.get("test_baseline") is not None:
            for name, value in base_metrics.items():
                if float(value) != float(dump["test_baseline"][name]):
                    raise ValueError("Recomputed base metric differs from the dump")

        rungs = {}
        cache = {}
        for policy in job["policies"]:
            selected, pool_size = choose(
                dump["candidates"], policy, pred_len=pred_len
            )
            signature = json.dumps(selected, sort_keys=True)
            if signature in cache:
                prediction, metrics, no_lookahead = cache[signature]
            else:
                prediction = apply_selected_rag(
                    validation_bundle,
                    test_bundle,
                    selected,
                    task_family,
                    pred_len,
                )
                if not np.isfinite(prediction).all():
                    raise ValueError("Corrected prediction contains non-finite values")
                metrics = protocol_metrics(test_bundle, prediction, task_family)
                for name, value in metrics.items():
                    if not math.isfinite(float(value)):
                        raise ValueError(f"Non-finite corrected metric {name}")
                no_lookahead = _no_lookahead_check(
                    validation_bundle,
                    test_bundle,
                    selected,
                    task_family,
                    pred_len,
                    prediction,
                )
                cache[signature] = (prediction, metrics, no_lookahead)
            rungs[policy["id"]] = {
                "policy": policy,
                "candidate_count": pool_size,
                "selected": selected,
                "test_corrected": metrics,
                "metric_gain_percent": {
                    name: 100.0
                    * (float(base_metrics[name]) - float(metrics[name]))
                    / float(base_metrics[name])
                    for name in metric_names
                },
                "strict_win": all(
                    float(metrics[name]) < float(base_metrics[name])
                    for name in metric_names
                ),
                "prediction_sha256": _prediction_sha256(prediction),
                "no_lookahead": no_lookahead,
            }

        payload = {
            "state": "completed",
            "cell_id": cell_id,
            "task_family": task_family,
            "dataset": dump["dataset"],
            "model": dump["model"],
            "pred_len": pred_len,
            "metric_names": list(metric_names),
            "bundle_sha256": digest,
            "test_baseline": base_metrics,
            "widest_candidate_count": len(dump["candidates"]),
            "validation_baseline": dump["validation_baseline"],
            "validation_memory_count": dump["validation_memory_count"],
            "validation_query_count": dump["validation_query_count"],
            "rungs": rungs,
            "elapsed_seconds": time.time() - started,
            "worker_pid": os.getpid(),
            "worker_host": socket.gethostname(),
        }
    except Exception as error:  # noqa: BLE001 - recorded as a cell failure
        payload = {
            "state": "failed",
            "cell_id": cell_id,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "elapsed_seconds": time.time() - started,
        }
    save_json_atomic(payload, result_path)
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--candidate-root", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--policies", required=True)
    parser.add_argument(
        "--cell-file",
        default="",
        help="optional newline separated cell ids to restrict this pass to",
    )
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--method-revision", required=True)
    parser.add_argument("--launcher-revision", required=True)
    args = parser.parse_args()

    protocol_path = Path(args.protocol)
    protocol = json.loads(protocol_path.read_text())
    available = {policy["id"]: policy for policy in protocol["selection_policies"]}
    policy_ids = [entry for entry in args.policies.split(",") if entry]
    unknown = [entry for entry in policy_ids if entry not in available]
    if unknown:
        raise SystemExit(f"Unknown policies {unknown}")
    policies = [available[entry] for entry in policy_ids]

    dev_list_path = protocol_path.parent / Path(
        protocol["development_split"]["cell_list"]
    ).name
    dev_text = dev_list_path.read_text()
    if sha256_text(dev_text) != protocol["development_split"]["cell_list_sha256"]:
        raise SystemExit("Development cell list hash does not match the protocol")
    dev_cells = {line for line in dev_text.split("\n") if line}

    candidate_root = Path(args.candidate_root)
    dumps = sorted(candidate_root.glob("cells/*/candidates.json.gz"))
    if args.cell_file:
        wanted = {
            line
            for line in Path(args.cell_file).read_text().split("\n")
            if line
        }
        dumps = [
            path for path in dumps if path.parent.name.replace("__", "/") in wanted
        ]
    if not dumps:
        raise SystemExit(f"No candidate dumps under {candidate_root}")

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    run_metadata = {
        "policies": policies,
        "candidate_root": str(candidate_root),
        "candidate_run_metadata_sha256": sha256_file(
            candidate_root / "run_metadata.json"
        ),
        "protocol": str(protocol_path),
        "protocol_sha256": sha256_file(protocol_path),
        "development_cell_list_sha256": sha256_text(dev_text),
        "method_revision": args.method_revision,
        "launcher_revision": args.launcher_revision,
        "cells_requested": len(dumps),
        "cpu_workers": args.workers,
        "started_unix": time.time(),
        "host": socket.gethostname(),
    }
    save_json_atomic(run_metadata, output_root / "run_metadata.json")

    jobs = [
        {
            "cell_id": path.parent.name.replace("__", "/"),
            "candidate_path": str(path),
            "policies": policies,
            "artifact_root": args.artifact_root,
            "output_root": str(output_root),
            "force": args.force,
        }
        for path in dumps
    ]
    jobs.sort(key=lambda job: -Path(job["candidate_path"]).stat().st_size)

    payloads = []
    context = mp.get_context("spawn")
    with context.Pool(processes=args.workers) as pool:
        for payload in pool.imap_unordered(run_cell, jobs):
            payloads.append(payload)
            print(
                f"{len(payloads)}/{len(jobs)} {payload['cell_id']} {payload['state']} "
                f"{payload['elapsed_seconds']:.0f}s",
                flush=True,
            )

    summary = summarize(payloads, dev_cells, policy_ids)
    summary["run_metadata"] = run_metadata
    summary["completed_unix"] = time.time()
    summary["cell_states"] = sorted(payloads, key=lambda row: row["cell_id"])
    save_json_atomic(summary, output_root / "summary.json")
    print(json.dumps({"counts": summary["counts"]}, indent=2))
    return 0 if not summary["failed_cells"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
