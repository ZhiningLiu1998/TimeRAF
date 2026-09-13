#!/usr/bin/env python3
"""Fixed-forecast evaluation of the unified residual/drift retrieval portfolio.

Every cell revises the byte-identical frozen replay forecast of the accepted
585-cell comparison, so the only difference against the recorded TimeRAF row is
the candidate portfolio. Rungs and the development split come from
``docs/timeraf_v2_unified_retrieval_protocol.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from ts_rag.benchmark_rag import PortfolioConfig, apply_selected_rag
from ts_rag.retrieval_baselines import load_replay_bundle

METRIC_NAMES = {
    "long_term": ("mse", "mae"),
    "epf": ("mse", "mae"),
    "pems": ("mae", "rmse", "mape"),
}


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        while chunk := source.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def save_json_atomic(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    temporary.replace(path)


def portfolio_from_payload(payload):
    payload = dict(payload)
    for key in ("drift_betas", "drift_alphas", "drift_ramps"):
        if key in payload:
            payload[key] = tuple(payload[key])
    if "drift_aggregations" in payload:
        payload["drift_aggregations"] = tuple(
            tuple(entry) for entry in payload["drift_aggregations"]
        )
    return PortfolioConfig(**payload)


def _rung_admits(row, rung):
    if row["method"] != "residual_drift":
        return True
    portfolio = rung["portfolio"]
    if not portfolio.get("enable_drift"):
        return False
    params = row["params"]
    aggregations = {
        tuple(entry) for entry in portfolio["drift_aggregations"]
    }
    if (params["k"], params["temperature"], params["shrinkage"]) not in aggregations:
        return False
    if params["ramp"] not in portfolio["drift_ramps"]:
        return False
    if params["beta"] not in portfolio["drift_betas"]:
        return False
    if params["alpha"] not in portfolio["drift_alphas"]:
        return False
    if not portfolio.get("shape_descriptors") and not params["feature_config"].get(
        "include_level_features", True
    ):
        return False
    return True


def search_portfolio(protocol, ladder_ids):
    """One candidate pool that contains every laddered rung's pool.

    Rungs differ only by which candidates they admit and by how they score
    them, so a single search per cell serves the whole ladder. Validation folds
    are computed whenever any rung selects on them; the whole-window score of
    every row is unaffected, which keeps the ``v1_reference`` rung byte
    identical to the accepted result.
    """

    rungs = {rung["id"]: rung for rung in protocol["rungs"]}
    widest = None
    for rung_id in ladder_ids:
        portfolio = rungs[rung_id]["portfolio"]
        if not portfolio.get("enable_drift"):
            continue
        score = (
            len(portfolio["drift_ramps"]),
            1 if portfolio.get("shape_descriptors") else 0,
            len(portfolio["drift_aggregations"]),
            len(portfolio["drift_betas"]),
            len(portfolio["drift_alphas"]),
        )
        if widest is None or score > widest[0]:
            widest = (score, dict(portfolio))
    payload = widest[1] if widest else dict(rungs[ladder_ids[0]]["portfolio"])
    payload["conservative_tolerance"] = 0.0
    payload["fold_selection"] = (
        "minimax2"
        if any(
            rungs[rung_id]["portfolio"].get("fold_selection", "none") != "none"
            for rung_id in ladder_ids
        )
        else "none"
    )
    return payload


def _prediction_sha256(prediction):
    contiguous = np.ascontiguousarray(prediction, dtype=np.float32)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def _no_lookahead_check(
    validation_bundle, test_bundle, selected, task_family, pred_len, reference
):
    """Reuse the accepted no-lookahead check of the retrieval-baseline runner."""

    from ts_rag.retrieval_baselines import _time_raf_no_lookahead

    return _time_raf_no_lookahead(
        validation_bundle,
        test_bundle,
        selected,
        task_family,
        pred_len,
        reference,
    )


def run_cell(job):
    from ts_rag.benchmark_rag import (
        _choose_best,
        protocol_metrics,
        search_validation_rag,
    )

    started = time.time()
    record = dict(job["record"])
    cell_id = job["cell_id"]
    output_dir = Path(job["output_root"]) / "cells" / cell_id.replace("/", "__")
    result_path = output_dir / "result.json"
    if job.get("search_only"):
        dump_path = output_dir / "candidates.json.gz"
        if dump_path.exists() and not job["force"]:
            return {
                "state": "completed",
                "cell_id": cell_id,
                "task_family": record["task_family"],
                "resumed": True,
                "rungs": {},
                "elapsed_seconds": 0.0,
            }
    elif result_path.exists() and not job["force"]:
        try:
            existing = json.loads(result_path.read_text())
            if existing.get("state") == "completed" and set(
                existing.get("rungs", {})
            ) == set(rung["id"] for rung in job["ladder"]):
                return existing
        except (json.JSONDecodeError, OSError):
            pass

    try:
        bundle_path = Path(job["artifact_root"]) / record["path"]
        digest = sha256_file(bundle_path)
        if digest != record["sha256"]:
            raise ValueError(f"Prediction bundle hash drifted for {cell_id}")

        task_family = record["task_family"]
        metric_names = METRIC_NAMES[task_family]
        pred_len = int(record["pred_len"])
        search_only = bool(job.get("search_only"))
        if search_only:
            (validation_bundle,) = load_replay_bundle(
                bundle_path, task_family, splits=("validation",)
            )
            test_bundle = None
        else:
            validation_bundle, test_bundle = load_replay_bundle(
                bundle_path, task_family
            )

        search = search_validation_rag(
            validation_bundle,
            task_family,
            metric_names,
            pred_len,
            portfolio=portfolio_from_payload(job["search_portfolio"]),
        )
        base_metrics = (
            None
            if search_only
            else protocol_metrics(test_bundle, test_bundle["y_base"], task_family)
        )

        if job.get("dump_candidates"):
            import gzip

            dump_path = output_dir / "candidates.json.gz"
            dump_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "cell_id": cell_id,
                "task_family": task_family,
                "dataset": record["dataset"],
                "model": record["model"],
                "pred_len": pred_len,
                "metric_names": list(metric_names),
                "bundle_sha256": digest,
                "bundle_path": record["path"],
                "test_baseline": base_metrics,
                "validation_baseline": search["baseline"],
                "validation_memory_count": search["memory_count"],
                "validation_query_count": search["query_count"],
                "search_portfolio": job["search_portfolio"],
                "candidates": search["candidates"],
            }
            temporary = dump_path.with_suffix(".tmp")
            with gzip.open(temporary, "wt") as handle:
                json.dump(payload, handle, sort_keys=True, allow_nan=False)
            temporary.replace(dump_path)

        rungs = {}
        for rung in [] if search_only else job["ladder"]:
            rows = [row for row in search["candidates"] if _rung_admits(row, rung)]
            if not rows:
                raise ValueError(f"Rung {rung['id']} admitted no candidate")
            tolerance = float(
                rung["portfolio"].get("conservative_tolerance", 0.0)
            )
            selected = _choose_best(
                rows,
                tolerance,
                rung["portfolio"].get("fold_selection", "none"),
            )
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
            rungs[rung["id"]] = {
                "candidate_count": len(rows),
                "conservative_tolerance": tolerance,
                "fold_selection": rung["portfolio"].get("fold_selection", "none"),
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
                "no_lookahead": _no_lookahead_check(
                    validation_bundle,
                    test_bundle,
                    selected,
                    task_family,
                    pred_len,
                    prediction,
                ),
            }

        payload = {
            "state": "completed",
            "cell_id": cell_id,
            "task_family": task_family,
            "dataset": record["dataset"],
            "model": record["model"],
            "pred_len": pred_len,
            "metric_names": list(metric_names),
            "bundle_sha256": digest,
            "test_baseline": base_metrics,
            "search_only": search_only,
            "widest_candidate_count": len(search["candidates"]),
            "validation_baseline": search["baseline"],
            "validation_memory_count": search["memory_count"],
            "validation_query_count": search["query_count"],
            "rungs": rungs,
            "elapsed_seconds": time.time() - started,
            "worker_pid": os.getpid(),
            "worker_host": socket.gethostname(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        }
    except Exception as error:  # noqa: BLE001 - recorded as a cell failure
        payload = {
            "state": "failed",
            "cell_id": cell_id,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "elapsed_seconds": time.time() - started,
            "worker_pid": os.getpid(),
        }
    save_json_atomic(payload, result_path)
    return payload


def _block(entries):
    """Aggregate (cell, rung outcome) pairs."""

    if not entries:
        return None
    worst = [min(entry["metric_gain_percent"].values()) for entry in entries]
    metrics = sorted({name for entry in entries for name in entry["metric_gain_percent"]})

    def per_metric(reduce_fn, name):
        return float(
            reduce_fn(
                [
                    entry["metric_gain_percent"][name]
                    for entry in entries
                    if name in entry["metric_gain_percent"]
                ]
            )
        )

    return {
        "cells": len(entries),
        "strict_wins": sum(1 for entry in entries if entry["strict_win"]),
        "median_worst_metric_gain_percent": float(np.median(worst)),
        "mean_worst_metric_gain_percent": float(np.mean(worst)),
        "median_gain_percent": {name: per_metric(np.median, name) for name in metrics},
        "mean_gain_percent": {name: per_metric(np.mean, name) for name in metrics},
        "selected_method_counts": {
            method: sum(
                1 for entry in entries if entry["selected"]["method"] == method
            )
            for method in sorted({entry["selected"]["method"] for entry in entries})
        },
        "selected_beta_counts": {
            str(beta): sum(
                1
                for entry in entries
                if entry["selected"]["params"].get("beta") == beta
            )
            for beta in sorted(
                {
                    entry["selected"]["params"].get("beta")
                    for entry in entries
                    if entry["selected"]["params"].get("beta") is not None
                }
            )
        },
    }


def summarize(cell_payloads, dev_cells, ladder_ids):
    families = ("long_term", "pems", "epf")
    completed = [row for row in cell_payloads if row.get("state") == "completed"]
    failed = [row for row in cell_payloads if row.get("state") != "completed"]
    summary = {
        "counts": {
            "completed": len(completed),
            "failed": len(failed),
            "total": len(cell_payloads),
        },
        "failed_cells": sorted(row["cell_id"] for row in failed),
        "no_lookahead_all_passed": all(
            outcome["no_lookahead"]["passed"]
            for row in completed
            for outcome in row["rungs"].values()
        ),
        "rungs": {},
    }
    for rung_id in ladder_ids:
        entries = [
            dict(row["rungs"][rung_id], cell_id=row["cell_id"], task_family=row["task_family"])
            for row in completed
            if rung_id in row["rungs"]
        ]
        summary["rungs"][rung_id] = {
            "overall": _block(entries),
            "by_task_family": {
                family: _block(
                    [entry for entry in entries if entry["task_family"] == family]
                )
                for family in families
            },
            "splits": {
                "development": _block(
                    [entry for entry in entries if entry["cell_id"] in dev_cells]
                ),
                "confirmatory": _block(
                    [entry for entry in entries if entry["cell_id"] not in dev_cells]
                ),
            },
        }
    return summary


def gpu_evidence():
    try:
        listing = subprocess.run(
            ["nvidia-smi", "-L"], capture_output=True, text=True, timeout=60
        )
        gpus = [line for line in listing.stdout.splitlines() if line.strip()]
    except (OSError, subprocess.SubprocessError):
        gpus = []
    return {
        "reserved_gpus": len(gpus),
        "gpu_workers": 0,
        "inactive_reserved_gpus": len(gpus),
        "reason": (
            "Fixed-forecast portfolio search is a numpy numerical replay over "
            "existing prediction bundles; no model inference occurs."
        ),
        "nvidia_smi_l": gpus,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--ladder",
        required=True,
        help="comma separated rung ids evaluated from one shared candidate search",
    )
    parser.add_argument("--cells", choices=("all", "development", "confirmatory"), default="all")
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--search-only",
        action="store_true",
        help="dump validation candidates without evaluating any rung on test, so "
        "the test arrays are never loaded",
    )
    parser.add_argument(
        "--min-bundle-bytes", type=int, default=0
    )
    parser.add_argument(
        "--max-bundle-bytes", type=int, default=0
    )
    parser.add_argument(
        "--dump-candidates",
        action="store_true",
        help="store every scored validation candidate so selection policies can "
        "be evaluated without repeating the search",
    )
    parser.add_argument("--method-revision", required=True)
    parser.add_argument("--launcher-revision", required=True)
    parser.add_argument("--comparison-summary", default="")
    args = parser.parse_args()

    protocol_path = Path(args.protocol)
    protocol = json.loads(protocol_path.read_text())
    rungs = {rung["id"]: rung for rung in protocol["rungs"]}
    ladder_ids = [entry for entry in args.ladder.split(",") if entry]
    unknown = [entry for entry in ladder_ids if entry not in rungs]
    if unknown:
        raise SystemExit(f"Unknown rungs {unknown}")
    ladder = [rungs[entry] for entry in ladder_ids]
    search_grid = search_portfolio(protocol, ladder_ids)

    catalog_path = Path(args.catalog)
    catalog = json.loads(catalog_path.read_text())
    bundles = catalog["bundles"]
    if len(bundles) != protocol["fixed_forecast_scope"]["cells"]:
        raise SystemExit("Catalog cell count does not match the protocol scope")
    catalog_sha = sha256_file(catalog_path)
    if catalog_sha != protocol["fixed_forecast_scope"]["catalog_sha256"]:
        raise SystemExit("Catalog hash does not match the frozen protocol")

    dev_list_path = protocol_path.parent / Path(
        protocol["development_split"]["cell_list"]
    ).name
    dev_text = dev_list_path.read_text()
    if sha256_text(dev_text) != protocol["development_split"]["cell_list_sha256"]:
        raise SystemExit("Development cell list hash does not match the protocol")
    dev_cells = set(line for line in dev_text.split("\n") if line)
    if len(dev_cells) != protocol["development_split"]["cells"]:
        raise SystemExit("Development cell count does not match the protocol")

    selected_ids = sorted(bundles)
    if args.cells == "development":
        selected_ids = [cell for cell in selected_ids if cell in dev_cells]
    elif args.cells == "confirmatory":
        selected_ids = [cell for cell in selected_ids if cell not in dev_cells]
    if args.min_bundle_bytes:
        selected_ids = [
            cell
            for cell in selected_ids
            if bundles[cell]["size_bytes"] >= args.min_bundle_bytes
        ]
    if args.max_bundle_bytes:
        selected_ids = [
            cell
            for cell in selected_ids
            if bundles[cell]["size_bytes"] < args.max_bundle_bytes
        ]
    if args.limit:
        selected_ids = selected_ids[: args.limit]

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    run_metadata = {
        "ladder": ladder,
        "search_portfolio": search_grid,
        "cells_requested": len(selected_ids),
        "cell_scope": args.cells,
        "catalog": str(catalog_path),
        "catalog_sha256": catalog_sha,
        "protocol": str(protocol_path),
        "protocol_sha256": sha256_file(protocol_path),
        "development_cell_list_sha256": sha256_text(dev_text),
        "method_revision": args.method_revision,
        "launcher_revision": args.launcher_revision,
        "artifact_root": args.artifact_root,
        "output_root": str(output_root),
        "cpu_workers": args.workers,
        "search_only": args.search_only,
        "dump_candidates": args.dump_candidates,
        "min_bundle_bytes": args.min_bundle_bytes,
        "max_bundle_bytes": args.max_bundle_bytes,
        "topology": gpu_evidence(),
        "started_unix": time.time(),
        "host": socket.gethostname(),
        "python": sys.version,
        "numpy": np.__version__,
    }
    if args.comparison_summary:
        run_metadata["comparison_summary"] = args.comparison_summary
        run_metadata["comparison_summary_sha256"] = sha256_file(args.comparison_summary)
    save_json_atomic(run_metadata, output_root / "run_metadata.json")

    jobs = [
        {
            "cell_id": cell_id,
            "record": bundles[cell_id],
            "ladder": ladder,
            "search_portfolio": search_grid,
            "artifact_root": args.artifact_root,
            "output_root": str(output_root),
            "force": args.force,
            "dump_candidates": args.dump_candidates,
            "search_only": args.search_only,
        }
        # Largest bundles first so the makespan is not dominated by a late
        # heavy cell while workers idle.
        for cell_id in sorted(
            selected_ids, key=lambda name: -bundles[name]["size_bytes"]
        )
    ]

    payloads = []
    if args.workers <= 1:
        for job in jobs:
            payloads.append(run_cell(job))
            print(
                f"{len(payloads)}/{len(jobs)} {payloads[-1]['cell_id']} "
                f"{payloads[-1]['state']}",
                flush=True,
            )
    else:
        context = mp.get_context("spawn")
        with context.Pool(processes=args.workers) as pool:
            for payload in pool.imap_unordered(run_cell, jobs):
                payloads.append(payload)
                print(
                    f"{len(payloads)}/{len(jobs)} {payload['cell_id']} "
                    f"{payload['state']} {payload['elapsed_seconds']:.0f}s",
                    flush=True,
                )

    summary = summarize(payloads, dev_cells, ladder_ids)
    summary["run_metadata"] = run_metadata
    summary["completed_unix"] = time.time()
    summary["cell_states"] = sorted(payloads, key=lambda row: row["cell_id"])
    save_json_atomic(summary, output_root / "summary.json")
    print(json.dumps({"counts": summary["counts"], "rungs": summary["rungs"]}, indent=2))
    return 0 if not summary["failed_cells"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
