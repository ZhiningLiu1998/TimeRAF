#!/usr/bin/env python3

from __future__ import annotations

import argparse
from bisect import bisect_right
import hashlib
import json
import os
import random
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.generate_native_retrieval_baseline_manifest import build_manifest


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def save_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def raf_rows(protocol: dict) -> list[dict]:
    return [
        row
        for row in build_manifest(protocol)
        if row["method"] == "raf"
    ]


def filter_rows_by_cell_ids(rows: list[dict], cell_ids_path: Path) -> list[dict]:
    text = cell_ids_path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError("cell IDs input is empty")

    if cell_ids_path.suffix.lower() == ".jsonl":
        cell_ids = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                raise ValueError(f"cell IDs JSONL has an empty line at {line_number}")
            try:
                cell_ids.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid cell ID JSONL at line {line_number}"
                ) from error
    else:
        try:
            cell_ids = json.loads(text)
        except json.JSONDecodeError as error:
            raise ValueError("invalid cell IDs JSON") from error
        if not isinstance(cell_ids, list):
            raise ValueError("cell IDs JSON must be an array")

    known_ids = {row["id"] for row in rows}
    selected_ids: set[str] = set()
    for cell_id in cell_ids:
        if not isinstance(cell_id, str):
            raise ValueError("every cell ID must be a string")
        if not cell_id.strip():
            raise ValueError("cell ID must not be empty")
        if cell_id in selected_ids:
            raise ValueError(f"duplicate cell ID: {cell_id}")
        if cell_id not in known_ids:
            raise ValueError(f"unknown RAF cell ID: {cell_id}")
        selected_ids.add(cell_id)

    if not selected_ids:
        raise ValueError("cell IDs input selected no RAF rows")
    return [row for row in rows if row["id"] in selected_ids]


def cell_dir(output_root: Path, row: dict) -> Path:
    return output_root / "cells" / row["id"]


def is_completed(output_root: Path, row: dict, protocol_sha256: str) -> bool:
    path = cell_dir(output_root, row) / "result.json"
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(
        result.get("status") == "completed"
        and result.get("cell_id") == row["id"]
        and result.get("protocol_sha256") == protocol_sha256
        and set(result.get("base", {}).get("metrics", {})) == {"wql", "mase"}
        and set(result.get("retrieval", {}).get("metrics", {}))
        == {"wql", "mase"}
        and all(
            np.isfinite(value)
            for system in ("base", "retrieval")
            for value in result[system]["metrics"].values()
        )
    )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def native_config(row: dict) -> dict:
    return {
        "name": row["dataset"],
        "hf_repo": (
            "autogluon/chronos_datasets_extra"
            if row["dataset"] == "ETTh"
            else "autogluon/chronos_datasets"
        ),
        "offset": -row["prediction_length"],
        "prediction_length": row["prediction_length"],
        "num_rolls": 1,
        "max_history": row["context_length"],
        "distance": 20,
    }


def memory_safe_best_matches(
    train_df,
    context_tensor_matrix,
    test_length,
    prediction_length,
    pipeline,
    top_n=1,
    candidate_batch_size=512,
):
    """Reference query-major implementation retained for equivalence tests."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with torch.inference_mode():
        contexts = torch.stack(context_tensor_matrix).to(device)
        target_embeddings, _ = pipeline.embed(contexts)
        target_embeddings = target_embeddings.unsqueeze(1)
        del contexts

        query_count = len(context_tensor_matrix)
        best_errors = torch.full(
            (query_count, top_n),
            float("inf"),
            dtype=torch.float32,
        )
        best_locations: list[list[tuple[int, int] | None]] = [
            [None] * top_n for _ in range(query_count)
        ]
        segment_batch = []
        location_batch: list[tuple[int, int]] = []

        def consume_batch() -> None:
            nonlocal best_errors, best_locations
            if not segment_batch:
                return
            candidates = torch.stack(segment_batch).to(device)
            candidate_embeddings, _ = pipeline.embed(candidates)
            errors = torch.norm(
                target_embeddings - candidate_embeddings.unsqueeze(0),
                dim=3,
                p=2,
            ).sum(dim=2)
            prior = best_errors.to(device)
            combined = torch.cat([prior, errors], dim=1)
            values, indices = torch.topk(
                combined, top_n, dim=1, largest=False, sorted=True
            )
            next_locations: list[list[tuple[int, int] | None]] = []
            for query_index in range(query_count):
                query_locations = []
                for selected in indices[query_index].cpu().tolist():
                    if selected < top_n:
                        query_locations.append(
                            best_locations[query_index][selected]
                        )
                    else:
                        query_locations.append(
                            location_batch[selected - top_n]
                        )
                next_locations.append(query_locations)
            best_errors = values.cpu()
            best_locations = next_locations
            segment_batch.clear()
            location_batch.clear()
            del candidates, candidate_embeddings, errors, prior, combined
            torch.cuda.empty_cache()

        for series_index, series in enumerate(train_df):
            values = np.asarray(series["target"], dtype=float)
            final_start = len(values) - test_length - prediction_length
            for start in range(final_start + 1):
                segment_batch.append(
                    torch.tensor(values[start : start + test_length])
                )
                location_batch.append((series_index, start))
                if len(segment_batch) == candidate_batch_size:
                    consume_batch()
        consume_batch()

    matches = []
    total_length = test_length + prediction_length
    for query_locations in best_locations:
        for location in query_locations:
            if location is None:
                raise RuntimeError("RAF retrieval found no candidate")
            series_index, start = location
            matches.append(
                train_df[series_index]["target"][start : start + total_length]
            )
    return matches


def loop_interchanged_best_matches(
    train_df,
    context_batches,
    test_length,
    prediction_length,
    pipeline,
    top_n=1,
    candidate_batch_size=512,
):
    if not 1 <= candidate_batch_size <= 512:
        raise ValueError("candidate_batch_size must be between 1 and 512")
    if not context_batches or any(not batch for batch in context_batches):
        raise ValueError("RAF retrieval requires non-empty query batches")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with torch.inference_mode():
        target_embedding_batches = []
        for context_tensor_matrix in context_batches:
            contexts = torch.stack(context_tensor_matrix).to(device)
            target_embeddings, _ = pipeline.embed(contexts)
            target_embedding_batches.append(target_embeddings.unsqueeze(1))
            del contexts

        best_errors = [
            torch.full(
                (len(context_tensor_matrix), top_n),
                float("inf"),
                dtype=torch.float32,
                device=device,
            )
            for context_tensor_matrix in context_batches
        ]
        best_indices = [
            torch.full(
                (len(context_tensor_matrix), top_n),
                -1,
                dtype=torch.long,
                device=device,
            )
            for context_tensor_matrix in context_batches
        ]
        segment_batch = []
        candidate_offset = 0
        enumerated_candidates = 0
        series_starts = []
        series_ends = []

        def consume_batch() -> None:
            nonlocal candidate_offset
            if not segment_batch:
                return
            candidates = torch.stack(segment_batch).to(device)
            candidate_embeddings, _ = pipeline.embed(candidates)
            chunk_indices = torch.arange(
                candidate_offset,
                candidate_offset + len(segment_batch),
                dtype=torch.long,
                device=device,
            )

            for batch_index, target_embeddings in enumerate(target_embedding_batches):
                errors = torch.norm(
                    target_embeddings - candidate_embeddings.unsqueeze(0),
                    dim=3,
                    p=2,
                ).sum(dim=2)
                combined = torch.cat([best_errors[batch_index], errors], dim=1)
                values, indices = torch.topk(
                    combined,
                    top_n,
                    dim=1,
                    largest=False,
                    sorted=True,
                )
                candidate_indices = chunk_indices.expand(
                    len(context_batches[batch_index]), -1
                )
                combined_indices = torch.cat(
                    [best_indices[batch_index], candidate_indices],
                    dim=1,
                )
                best_errors[batch_index] = values
                best_indices[batch_index] = torch.gather(
                    combined_indices, 1, indices
                )
                del errors, combined, values, indices, combined_indices

            candidate_offset += len(segment_batch)
            segment_batch.clear()
            del candidates, candidate_embeddings, chunk_indices

        for series_index, series in enumerate(train_df):
            values = np.asarray(series["target"], dtype=float)
            final_start = len(values) - test_length - prediction_length
            series_starts.append(enumerated_candidates)
            for start in range(final_start + 1):
                segment_batch.append(torch.tensor(values[start : start + test_length]))
                enumerated_candidates += 1
                if len(segment_batch) == candidate_batch_size:
                    consume_batch()
            series_ends.append(enumerated_candidates)
        consume_batch()

    matches_by_batch = []
    total_length = test_length + prediction_length
    for batch_indices in best_indices:
        batch_matches = []
        for query_indices in batch_indices.cpu().tolist():
            for absolute_index in query_indices:
                if absolute_index < 0:
                    raise RuntimeError("RAF retrieval found no candidate")
                series_index = bisect_right(series_ends, absolute_index)
                if series_index >= len(train_df):
                    raise RuntimeError("RAF retrieval candidate index is invalid")
                start = absolute_index - series_starts[series_index]
                batch_matches.append(
                    train_df[series_index]["target"][start : start + total_length]
                )
        matches_by_batch.append(batch_matches)
    return matches_by_batch


def generate_sample_forecasts_loop_interchanged(
    run_chronos,
    train_df,
    top_n,
    test_data_input,
    pipeline,
    prediction_length,
    batch_size,
    num_samples,
    candidate_batch_size=512,
    **predict_kwargs,
):
    test_entries = list(test_data_input)
    entry_batches = [
        list(batch)
        for batch in run_chronos.batcher(test_entries, batch_size=batch_size)
    ]
    context_batches = [
        [torch.tensor(entry["target"]) for entry in batch] for batch in entry_batches
    ]
    if not context_batches:
        raise ValueError("RAF retrieval requires at least one query")
    test_length = len(context_batches[0][0])

    cpu_rng_state = torch.get_rng_state()
    cuda_rng_states = (
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    )
    matches_by_batch = loop_interchanged_best_matches(
        train_df,
        context_batches,
        test_length,
        prediction_length,
        pipeline,
        top_n=top_n,
        candidate_batch_size=candidate_batch_size,
    )
    if not torch.equal(cpu_rng_state, torch.get_rng_state()):
        raise RuntimeError("RAF embedding changed the CPU RNG state")
    if torch.cuda.is_available() and any(
        not torch.equal(before, after)
        for before, after in zip(cuda_rng_states, torch.cuda.get_rng_state_all())
    ):
        raise RuntimeError("RAF embedding changed a CUDA RNG state")

    finder_globals = run_chronos.augment_time_series.__globals__
    finder_name = "find_best_matches_full_series_batch"
    reference_finder = finder_globals[finder_name]
    next_batch = 0

    def use_precomputed_matches(
        _train_df,
        context_tensor_matrix,
        received_test_length,
        received_prediction_length,
        _pipeline,
        received_top_n=1,
    ):
        nonlocal next_batch
        if next_batch >= len(matches_by_batch):
            raise RuntimeError("RAF requested too many precomputed batches")
        if (
            len(context_tensor_matrix) != len(context_batches[next_batch])
            or received_test_length != test_length
            or received_prediction_length != prediction_length
            or received_top_n != top_n
        ):
            raise RuntimeError("RAF precomputed query batch identity changed")
        matches = matches_by_batch[next_batch]
        next_batch += 1
        return matches

    finder_globals[finder_name] = use_precomputed_matches
    try:
        forecasts = run_chronos.generate_sample_forecasts(
            train_df,
            True,
            top_n=top_n,
            test_data_input=test_entries,
            pipeline=pipeline,
            prediction_length=prediction_length,
            batch_size=batch_size,
            num_samples=num_samples,
            **predict_kwargs,
        )
    finally:
        finder_globals[finder_name] = reference_finder

    if next_batch != len(matches_by_batch):
        raise RuntimeError("RAF did not consume every precomputed query batch")
    return forecasts


def evaluate_forecasts_native(run_chronos, forecasts, test_data) -> dict:
    metrics = (
        run_chronos.evaluate_forecasts(
            forecasts,
            test_data=test_data,
            metrics=[
                run_chronos.MASE(),
                run_chronos.MeanWeightedSumQuantileLoss(
                    np.arange(0.1, 1.0, 0.1)
                ),
            ],
            batch_size=5000,
        )
        .reset_index(drop=True)
        .to_dict(orient="records")[0]
    )
    return {
        "wql": float(metrics["mean_weighted_sum_quantile_loss"]),
        "mase": float(metrics["MASE[0.5]"]),
    }


def run_pair(
    run_chronos,
    pipeline,
    row: dict,
    protocol: dict,
    protocol_sha256: str,
    output_root: Path,
    source_revision: str,
) -> None:
    artifact = cell_dir(output_root, row)
    artifact.mkdir(parents=True, exist_ok=True)
    started = time.time()
    save_json_atomic(
        artifact / "status.json",
        {
            "status": "running",
            "cell_id": row["id"],
            "pid": os.getpid(),
            "started_unix": started,
            "protocol_sha256": protocol_sha256,
        },
    )
    try:
        config = native_config(row)
        test_data, train_df = run_chronos.load_and_split_dataset(config)
        settings = protocol["systems"]["raf"]
        common = {
            "pipeline": pipeline,
            "prediction_length": row["prediction_length"],
            "batch_size": 30,
            "num_samples": int(settings["num_samples"]),
            "temperature": None,
            "top_k": None,
            "top_p": None,
            "top_n": int(settings["retrieval_top_n"]),
        }

        seed_everything(int(settings["seed"]))
        base_forecasts = run_chronos.generate_sample_forecasts(
            train_df,
            False,
            test_data_input=test_data.input,
            **common,
        )
        base_metrics = evaluate_forecasts_native(
            run_chronos, base_forecasts, test_data
        )

        seed_everything(int(settings["seed"]))
        retrieval_forecasts = generate_sample_forecasts_loop_interchanged(
            run_chronos,
            train_df,
            test_data_input=test_data.input,
            **common,
        )
        retrieval_metrics = evaluate_forecasts_native(
            run_chronos, retrieval_forecasts, test_data
        )
        values = list(base_metrics.values()) + list(retrieval_metrics.values())
        if not all(np.isfinite(value) for value in values):
            raise ValueError("RAF produced a non-finite metric")

        finished = time.time()
        result = {
            "schema_version": 1,
            "status": "completed",
            "cell_id": row["id"],
            "method": "RAF",
            "base_method": "Chronos",
            "backbone": settings["backbone"],
            "dataset": row["dataset"],
            "benchmark": row["benchmark"],
            "context_length": row["context_length"],
            "prediction_length": row["prediction_length"],
            "seed": settings["seed"],
            "num_samples": settings["num_samples"],
            "retrieval_top_n": settings["retrieval_top_n"],
            "base": {"metrics": base_metrics},
            "retrieval": {"metrics": retrieval_metrics},
            "delta": {
                metric: retrieval_metrics[metric] - base_metrics[metric]
                for metric in ("wql", "mase")
            },
            "protocol_sha256": protocol_sha256,
            "source_revision": source_revision,
            "upstream_commit": protocol["sources"]["raf"]["commit"],
            "started_unix": started,
            "finished_unix": finished,
            "elapsed_seconds": finished - started,
            "pid": os.getpid(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "torch_device": "cuda:0",
            "released_config_correction": (
                protocol["systems"]["raf"]["released_config_correction"]
                if row["dataset"] == "ETTh"
                and row["context_length"] == 50
                else None
            ),
        }
        save_json_atomic(artifact / "result.json", result)
        save_json_atomic(
            artifact / "status.json",
            {
                "status": "completed",
                "cell_id": row["id"],
                "finished_unix": finished,
                "protocol_sha256": protocol_sha256,
            },
        )
    except BaseException as error:
        save_json_atomic(
            artifact / "status.json",
            {
                "status": "failed",
                "cell_id": row["id"],
                "error_type": type(error).__name__,
                "error": str(error),
                "finished_unix": time.time(),
                "protocol_sha256": protocol_sha256,
            },
        )
        raise
    finally:
        torch.cuda.empty_cache()


def import_upstream(upstream_root: Path):
    sys.path.insert(0, str(upstream_root))
    import run_chronos  # type: ignore

    run_chronos.augment_time_series.__globals__[
        "find_best_matches_full_series_batch"
    ] = memory_safe_best_matches
    return run_chronos


def run_worker(args: argparse.Namespace, protocol: dict, rows: list[dict]) -> int:
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Each RAF worker must see exactly one GPU")
    if args.worker_id is None or args.worker_count is None:
        raise ValueError("Worker identity is required")

    assigned = [
        row
        for index, row in enumerate(rows)
        if index % args.worker_count == args.worker_id
    ]
    if not assigned:
        return 0

    run_chronos = import_upstream(args.upstream_root)
    settings = protocol["systems"]["raf"]
    protocol_sha256 = sha256_file(args.protocol)
    pipeline = run_chronos.ChronosPipeline.from_pretrained(
        settings["backbone"],
        device_map="cuda:0",
        torch_dtype=torch.bfloat16,
    )

    for row in assigned:
        if is_completed(args.output_root, row, protocol_sha256):
            continue
        run_pair(
            run_chronos,
            pipeline,
            row,
            protocol,
            protocol_sha256,
            args.output_root,
            args.source_revision,
        )
    return 0


def query_gpu_processes() -> list[dict]:
    output = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,gpu_uuid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    rows = []
    for line in output.splitlines():
        if not line.strip():
            continue
        pid, uuid, memory = [item.strip() for item in line.split(",", 2)]
        rows.append(
            {"pid": int(pid), "gpu_uuid": uuid, "used_memory_mib": int(memory)}
        )
    return rows


def query_gpu_uuids() -> dict[str, int]:
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
        uuid.strip(): int(index)
        for index, uuid in (
            line.split(",", 1) for line in output.splitlines() if line.strip()
        )
    }


def terminate(process: subprocess.Popen) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=20)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def aggregate(
    output_root: Path,
    rows: list[dict],
    protocol_sha256: str,
    source_revision: str,
    launcher_revision: str,
    topology_path: Path,
    topology_valid: bool,
) -> dict:
    results = []
    failed = []
    for row in rows:
        artifact = cell_dir(output_root, row)
        if is_completed(output_root, row, protocol_sha256):
            results.append(
                json.loads(
                    (artifact / "result.json").read_text(encoding="utf-8")
                )
            )
        else:
            status_path = artifact / "status.json"
            status = (
                json.loads(status_path.read_text(encoding="utf-8"))
                if status_path.exists()
                else {"status": "missing"}
            )
            failed.append({"cell_id": row["id"], **status})

    summary = {
        "schema_version": 1,
        "method": "RAF",
        "expected_pairs": len(rows),
        "completed_pairs": len(results),
        "failed_or_missing_pairs": len(failed),
        "all_completed": (
            len(results) == len(rows) and not failed and topology_valid
        ),
        "protocol_sha256": protocol_sha256,
        "source_revision": source_revision,
        "launcher_revision": launcher_revision,
        "topology_path": str(topology_path),
        "topology_valid": topology_valid,
        "results": results,
        "failures": failed,
    }
    save_json_atomic(output_root / "summary.json", summary)
    return summary


def run_parent(
    args: argparse.Namespace, protocol: dict, rows: list[dict]
) -> int:
    visible_gpus = torch.cuda.device_count()
    if visible_gpus != args.reserved_gpus_per_host:
        raise RuntimeError(
            "Visible GPU count does not match reserved topology: "
            f"{visible_gpus} != {args.reserved_gpus_per_host}"
        )
    if not 1 <= args.worker_count <= args.reserved_gpus_per_host:
        raise ValueError(
            "worker_count must be between one and the reserved GPU count"
        )
    worker_gpu_ids = (
        args.worker_gpu_ids
        if args.worker_gpu_ids is not None
        else tuple(range(args.worker_count))
    )
    if len(worker_gpu_ids) != args.worker_count:
        raise ValueError("worker_gpu_ids must match worker_count")
    if (
        len(set(worker_gpu_ids)) != len(worker_gpu_ids)
        or min(worker_gpu_ids) < 0
        or max(worker_gpu_ids) >= args.reserved_gpus_per_host
    ):
        raise ValueError("worker_gpu_ids are invalid")
    if args.worker_count > len(rows):
        raise ValueError("worker_count exceeds the number of RAF cells")

    args.output_root.mkdir(parents=True, exist_ok=True)
    logs = args.output_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    protocol_sha256 = sha256_file(args.protocol)
    uuid_to_gpu = query_gpu_uuids()
    processes: dict[int, subprocess.Popen] = {}
    log_files = {}
    observations: list[dict] = []
    command_base = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--protocol",
        str(args.protocol),
        "--upstream-root",
        str(args.upstream_root),
        "--output-root",
        str(args.output_root),
        "--source-revision",
        args.source_revision,
        "--worker-count",
        str(args.worker_count),
    ]
    if args.max_cells:
        command_base.extend(["--max-cells", str(args.max_cells)])
    if args.cell_ids_path is not None:
        command_base.extend(
            ["--cell-ids-path", str(args.cell_ids_path.resolve())]
        )

    try:
        for worker_id in range(args.worker_count):
            log_file = (logs / f"worker-{worker_id}.log").open(
                "a", encoding="utf-8"
            )
            environment = dict(os.environ)
            environment["CUDA_VISIBLE_DEVICES"] = str(
                worker_gpu_ids[worker_id]
            )
            environment["PYTHONUNBUFFERED"] = "1"
            command = command_base + ["--worker-id", str(worker_id)]
            process = subprocess.Popen(
                command,
                cwd=args.upstream_root,
                env=environment,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            processes[worker_id] = process
            log_files[worker_id] = log_file

        while any(process.poll() is None for process in processes.values()):
            process_rows = query_gpu_processes()
            observations.append(
                {
                    "recorded_unix": time.time(),
                    "processes": [
                        {
                            **record,
                            "physical_gpu": uuid_to_gpu.get(record["gpu_uuid"]),
                        }
                        for record in process_rows
                    ],
                }
            )
            time.sleep(args.poll_seconds)
    finally:
        for process in processes.values():
            if process.poll() is None:
                terminate(process)
        for log_file in log_files.values():
            log_file.close()

    expected_bindings = {
        process.pid: worker_gpu_ids[worker_id]
        for worker_id, process in processes.items()
    }
    observed_bindings = {
        pid: sorted(
            {
                int(record["physical_gpu"])
                for observation in observations
                for record in observation["processes"]
                if record["pid"] == pid
                and record["physical_gpu"] is not None
            }
        )
        for pid in expected_bindings
    }
    topology_valid = all(
        observed_bindings[pid] == [expected_gpu]
        for pid, expected_gpu in expected_bindings.items()
    )
    co_located_gpu_ids = sorted(
        set(range(args.reserved_gpus_per_host)) - set(worker_gpu_ids)
    )
    topology = {
        "schema_version": 1,
        "instance_type": args.instance_type,
        "instance_count": 1,
        "reserved_gpus_per_host": args.reserved_gpus_per_host,
        "processes_per_host": args.worker_count,
        "worker_gpu_ids": list(worker_gpu_ids),
        "total_gpus": args.reserved_gpus_per_host,
        "world_size": args.worker_count,
        "co_located_gpu_ids": co_located_gpu_ids,
        "co_located_reason": (
            "RATD and CSDI occupy the remaining P5 GPUs."
            if co_located_gpu_ids
            else None
        ),
        "child_device": "cuda:0",
        "worker_pids": {
            str(worker): process.pid
            for worker, process in processes.items()
        },
        "worker_returncodes": {
            str(worker): process.returncode
            for worker, process in processes.items()
        },
        "expected_bindings": expected_bindings,
        "observed_bindings": observed_bindings,
        "topology_valid": topology_valid,
        "observations": observations,
    }
    topology_path = args.output_root / "startup_topology.json"
    save_json_atomic(topology_path, topology)
    summary = aggregate(
        args.output_root,
        rows,
        protocol_sha256,
        args.source_revision,
        args.launcher_revision or args.source_revision,
        topology_path,
        topology_valid,
    )
    if any(process.returncode != 0 for process in processes.values()):
        return 1
    return 0 if summary["all_completed"] else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("docs/native_retrieval_baseline_protocol.json"),
    )
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--launcher-revision")
    parser.add_argument("--instance-type", default="ml.g5.12xlarge")
    parser.add_argument("--reserved-gpus-per-host", type=int, default=4)
    parser.add_argument("--worker-count", type=int, default=4)
    parser.add_argument(
        "--worker-gpu-ids",
        type=lambda value: tuple(int(item) for item in value.split(",")),
    )
    parser.add_argument("--worker-id", type=int)
    parser.add_argument("--max-cells", type=int, default=0)
    parser.add_argument("--cell-ids-path", type=Path)
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    rows = raf_rows(protocol)
    if args.cell_ids_path is not None:
        rows = filter_rows_by_cell_ids(rows, args.cell_ids_path)
    if args.max_cells:
        rows = rows[: args.max_cells]
    if args.cell_ids_path is not None:
        args.worker_count = min(args.worker_count, len(rows))
        if args.worker_gpu_ids is not None:
            args.worker_gpu_ids = args.worker_gpu_ids[: args.worker_count]
    if args.worker_id is not None:
        return run_worker(args, protocol, rows)
    return run_parent(args, protocol, rows)


if __name__ == "__main__":
    raise SystemExit(main())
