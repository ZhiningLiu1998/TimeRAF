#!/usr/bin/env python3

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
import os
import pickle
import random
import signal
import subprocess
import sys
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.generate_native_retrieval_baseline_manifest import build_manifest


DATASETS = {
    "ETTh1": {
        "csv": "ETTh1_retrieve_ETTh1_512_only_self_train_None.csv",
        "database": "ETTh1_hour_512.pkl",
        "split": "ett_hour",
    },
    "ETTh2": {
        "csv": "ETTh2_retrieve_ETTh2_512_only_self_train_None.csv",
        "database": "ETTh2_hour_512.pkl",
        "split": "ett_hour",
    },
    "ETTm1": {
        "csv": "ETTm1_retrieve_ETTm1_512_only_self_train_None.csv",
        "database": "ETTm1_minute_512.pkl",
        "split": "ett_minute",
    },
    "ETTm2": {
        "csv": "ETTm2_retrieve_ETTm2_512_only_self_train_None.csv",
        "database": "ETTm2_minute_512.pkl",
        "split": "ett_minute",
    },
    "electricity": {
        "csv": "electricity_retrieve_electricity_512_only_self_train_None.csv",
        "database": "electricity_hour_512.pkl",
        "split": "custom",
    },
    "exchange_rate": {
        "csv": "exchange_rate_retrieve_exchange_rate_512_only_self_train_None.csv",
        "database": "exchange_rate_hour_512.pkl",
        "split": "custom",
    },
    "weather": {
        "csv": "weather_retrieve_weather_512_only_self_train_None.csv",
        "database": "weather_10minutes_512.pkl",
        "split": "custom",
    },
}


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


def ts_rag_rows(protocol: dict) -> list[dict]:
    return [
        row for row in build_manifest(protocol) if row["method"] == "ts_rag"
    ]


def split_boundaries(kind: str, length: int, context_length: int) -> dict:
    if kind == "ett_hour":
        train_end = 12 * 30 * 24
        test_start = (12 + 4) * 30 * 24
        test_end = (12 + 8) * 30 * 24
    elif kind == "ett_minute":
        train_end = 12 * 30 * 24 * 4
        test_start = (12 + 4) * 30 * 24 * 4
        test_end = (12 + 8) * 30 * 24 * 4
    elif kind == "custom":
        train_end = int(length * 0.7)
        test_length = int(length * 0.2)
        test_start = length - test_length
        test_end = length
    else:
        raise ValueError(f"Unknown split kind: {kind}")
    if test_end > length:
        raise ValueError(
            f"Split end {test_end} exceeds series length {length}"
        )
    return {
        "train_end": train_end,
        "test_context_start": test_start - context_length,
        "test_end": test_end,
    }


def assert_train_only_indices(
    indices: np.ndarray,
    train_end: int,
    context_length: int,
    prediction_length: int,
) -> dict:
    if indices.ndim != 2:
        raise ValueError("Retrieval indices must have shape [origins, top_k]")
    if not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("Retrieval indices must be integers")
    if indices.size == 0:
        raise ValueError("Retrieval index array is empty")
    minimum = int(indices.min())
    maximum = int(indices.max())
    if minimum < 0:
        raise ValueError(f"Negative retrieval index: {minimum}")
    latest_end = maximum + context_length + prediction_length
    if latest_end > train_end:
        raise ValueError(
            "Retrieved future crosses the training boundary: "
            f"latest_end={latest_end}, train_end={train_end}"
        )
    return {
        "minimum_index": minimum,
        "maximum_index": maximum,
        "latest_retrieved_future_end": latest_end,
        "train_end": train_end,
    }


def select_safe_released_candidates(
    indices: np.ndarray,
    distances: np.ndarray,
    maximum_start: int,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if indices.shape != distances.shape or indices.ndim != 2:
        raise ValueError("Released indices and distances must have equal 2D shapes")
    selected_indices = np.full((len(indices), top_k), -1, dtype=np.int64)
    selected_distances = np.full(
        (len(indices), top_k), np.inf, dtype=np.float32
    )
    fallback = []
    for row_index, (row_indices, row_distances) in enumerate(
        zip(indices, distances, strict=True)
    ):
        valid = np.flatnonzero(
            (row_indices >= 0) & (row_indices <= maximum_start)
        )
        if len(valid) < top_k:
            fallback.append(row_index)
            continue
        chosen = valid[:top_k]
        selected_indices[row_index] = row_indices[chosen]
        selected_distances[row_index] = row_distances[chosen]
    return (
        selected_indices,
        selected_distances,
        np.asarray(fallback, dtype=np.int64),
    )


def exact_l2_top_k(
    queries: np.ndarray, candidates: np.ndarray, top_k: int
) -> tuple[np.ndarray, np.ndarray]:
    if queries.ndim != 2 or candidates.ndim != 2:
        raise ValueError("Exact retrieval inputs must be matrices")
    if queries.shape[1] != candidates.shape[1]:
        raise ValueError("Query and candidate embedding dimensions differ")
    if len(candidates) < top_k:
        raise ValueError("Fewer safe candidates than requested top_k")
    query_norm = np.square(queries, dtype=np.float32).sum(
        axis=1, keepdims=True
    )
    candidate_norm = np.square(candidates, dtype=np.float32).sum(
        axis=1, keepdims=True
    ).T
    distances = query_norm + candidate_norm - 2.0 * queries @ candidates.T
    np.maximum(distances, 0.0, out=distances)
    partition = np.argpartition(distances, top_k - 1, axis=1)[:, :top_k]
    partition_distances = np.take_along_axis(distances, partition, axis=1)
    order = np.argsort(partition_distances, axis=1, kind="stable")
    selected = np.take_along_axis(partition, order, axis=1)
    selected_distances = np.take_along_axis(
        partition_distances, order, axis=1
    )
    return selected.astype(np.int64), selected_distances.astype(np.float32)


def verify_assets(asset_manifest_path: Path) -> dict:
    manifest = json.loads(asset_manifest_path.read_text(encoding="utf-8"))
    verified = []
    for root_key, files_key in (
        ("checkpoint_root", "checkpoint_files"),
        ("data_root", "data_files"),
    ):
        root = Path(manifest[root_key])
        for record in manifest[files_key]:
            path = root / record["path"]
            stat = path.stat()
            if stat.st_size != record["size_bytes"]:
                raise ValueError(
                    f"Size mismatch for {path}: {stat.st_size} != "
                    f"{record['size_bytes']}"
                )
            digest = sha256_file(path)
            if digest != record["sha256"]:
                raise ValueError(
                    f"SHA-256 mismatch for {path}: {digest} != "
                    f"{record['sha256']}"
                )
            verified.append(
                {
                    "path": str(path),
                    "size_bytes": stat.st_size,
                    "sha256": digest,
                }
            )
    return {"manifest": str(asset_manifest_path), "files": verified}


def seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cell_dir(output_root: Path, row: dict) -> Path:
    return output_root / "cells" / row["id"]


def completed(
    output_root: Path, row: dict, protocol_sha256: str
) -> bool:
    path = cell_dir(output_root, row) / "result.json"
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    values = [
        result.get(system, {}).get("metrics", {}).get(metric)
        for system in ("base", "retrieval")
        for metric in ("mse", "mae")
    ]
    return bool(
        result.get("status") == "completed"
        and result.get("cell_id") == row["id"]
        and result.get("protocol_sha256") == protocol_sha256
        and all(value is not None and np.isfinite(value) for value in values)
    )


def load_ts_rag_model_module(upstream_root: Path):
    package_name = "_timeraf_native_ts_rag_models"
    model_root = upstream_root / "TS-RAG" / "models"
    for module_name in list(sys.modules):
        if module_name == package_name or module_name.startswith(
            f"{package_name}."
        ):
            del sys.modules[module_name]
    package = types.ModuleType(package_name)
    package.__path__ = [str(model_root)]
    package.__package__ = package_name
    sys.modules[package_name] = package
    return importlib.import_module(f"{package_name}.ChronosBolt")


def install_autogluon_import_compatibility() -> None:
    old_name = (
        "autogluon.timeseries.models.gluonts.abstract_gluonts"
    )
    try:
        importlib.import_module(old_name)
        return
    except ModuleNotFoundError as error:
        if error.name != old_name:
            raise
    dataset_module = importlib.import_module(
        "autogluon.timeseries.models.gluonts.dataset"
    )
    compatibility = types.ModuleType(old_name)
    compatibility.SimpleGluonTSDataset = (
        dataset_module.SimpleGluonTSDataset
    )
    sys.modules[old_name] = compatibility


def load_models(
    upstream_root: Path, checkpoint_root: Path, chronos_root: Path
):
    import torch
    from transformers import AutoConfig

    sys.path.insert(0, str(chronos_root))
    from chronos import ChronosPipeline  # type: ignore

    install_autogluon_import_compatibility()
    model_module = load_ts_rag_model_module(upstream_root)
    ChronosBoltModelForForecasting = (
        model_module.ChronosBoltModelForForecasting
    )
    ChronosBoltModelForForecastingWithRetrieval = (
        model_module.ChronosBoltModelForForecastingWithRetrieval
    )

    base_root = checkpoint_root / "base"
    config = AutoConfig.from_pretrained(
        str(base_root), local_files_only=True
    )
    base = ChronosBoltModelForForecasting.from_pretrained(
        str(base_root), config=config, local_files_only=True
    )
    base_state = torch.load(
        base_root / "autogluon_model.pth",
        map_location="cpu",
        weights_only=True,
    )
    base.load_state_dict(base_state)

    retrieval = (
        ChronosBoltModelForForecastingWithRetrieval.from_pretrained(
            str(base_root),
            config=config,
            augment="moe",
            local_files_only=True,
        )
    )
    retrieval_state = torch.load(
        checkpoint_root / "chronos-bolt" / "best.pth",
        map_location="cpu",
        weights_only=True,
    )
    retrieval_state = {
        key.removeprefix("module."): value
        for key, value in retrieval_state.items()
    }
    retrieval.load_state_dict(retrieval_state)

    device = torch.device("cuda:0")
    base.to(device).eval()
    retrieval.to(device).eval()
    embedding = ChronosPipeline.from_pretrained(
        "amazon/chronos-t5-base",
        device_map="cuda:0",
        torch_dtype=torch.bfloat16,
    )
    return base, retrieval, embedding


def dataset_paths(data_root: Path, dataset: str) -> tuple[Path, Path]:
    config = DATASETS[dataset]
    return (
        data_root / "datasets_512" / config["csv"],
        data_root / "database_512" / config["database"],
    )


def read_dataset(
    data_root: Path,
    dataset: str,
    top_k: int,
    context_length: int,
    prediction_length: int,
    embedding_model,
) -> dict:
    import torch

    csv_path, database_path = dataset_paths(data_root, dataset)
    columns = pd.read_csv(csv_path, nrows=0).columns.tolist()
    raw_columns = [
        column
        for column in columns
        if not column.startswith(
            ("boundary_idx_", "timestamp_idx_", "distance_")
        )
    ]
    channel_columns = raw_columns[1:]
    released_top_k = 20
    timestamp_columns = [
        [
            f"timestamp_idx_{channel}_{rank}"
            for rank in range(released_top_k)
        ]
        for channel in channel_columns
    ]
    distance_columns = [
        [f"distance_{channel}_{rank}" for rank in range(released_top_k)]
        for channel in channel_columns
    ]
    required = raw_columns + [
        item for group in timestamp_columns + distance_columns for item in group
    ]
    missing = sorted(set(required).difference(columns))
    if missing:
        raise ValueError(f"Missing TS-RAG columns in {csv_path}: {missing[:5]}")

    frame = pd.read_csv(csv_path, usecols=required)
    raw = frame[channel_columns].to_numpy(dtype=np.float32)
    boundaries = split_boundaries(
        DATASETS[dataset]["split"], len(frame), context_length
    )
    train_end = boundaries["train_end"]
    train = raw[:train_end]
    means = train.mean(axis=0, dtype=np.float64)
    scales = train.std(axis=0, dtype=np.float64)
    scales[scales == 0] = 1.0
    scaled = ((raw - means) / scales).astype(np.float32)

    with database_path.open("rb") as source:
        database = pickle.load(source)
    database_raw = []
    database_embeddings = []
    for channel_index, channel in enumerate(channel_columns):
        if channel not in database:
            raise ValueError(
                f"Database {database_path} has no channel {channel}"
            )
        values = np.asarray(database[channel]["raw_data"], dtype=np.float32)
        database_raw.append(
            ((values - means[channel_index]) / scales[channel_index]).astype(
                np.float32
            )
        )
        database_embeddings.append(
            np.asarray(database[channel]["embeddings"], dtype=np.float32)
        )

    first = boundaries["test_context_start"]
    final = boundaries["test_end"] - context_length - prediction_length
    origins = np.arange(first, final + 1, dtype=np.int64)
    audits = []
    timestamp_values = []
    distance_values = []
    fallback_total = 0
    maximum_start = train_end - context_length - prediction_length
    for channel_index, channel in enumerate(channel_columns):
        released_indices = frame.loc[
            origins, timestamp_columns[channel_index]
        ].to_numpy(dtype=np.int64)
        released_distances = frame.loc[
            origins, distance_columns[channel_index]
        ].to_numpy(dtype=np.float32)
        indices, distances, fallback = select_safe_released_candidates(
            released_indices,
            released_distances,
            maximum_start,
            top_k,
        )
        if len(fallback):
            query_embeddings = []
            for offset in range(0, len(fallback), 512):
                selected_rows = fallback[offset : offset + 512]
                contexts = np.stack(
                    [
                        raw[
                            origins[row] : origins[row] + context_length,
                            channel_index,
                        ]
                        for row in selected_rows
                    ]
                )
                embedded, _ = embedding_model.embed(
                    torch.from_numpy(contexts)
                )
                query_embeddings.append(
                    embedded[:, -1, :].float().numpy()
                )
            query_embeddings_array = np.concatenate(
                query_embeddings, axis=0
            ).astype(np.float32)
            safe_embeddings = database_embeddings[channel_index][
                : maximum_start + 1
            ]
            exact_indices, exact_distances = exact_l2_top_k(
                query_embeddings_array,
                safe_embeddings,
                top_k,
            )
            indices[fallback] = exact_indices
            distances[fallback] = exact_distances
        fallback_total += len(fallback)
        audits.append(
            {
                "channel": channel,
                "released_top_20_rows_recomputed": int(len(fallback)),
                **assert_train_only_indices(
                    indices,
                    train_end,
                    context_length,
                    prediction_length,
                ),
            }
        )
        timestamp_values.append(indices)
        distance_values.append(distances)
        database_embeddings[channel_index] = None
    del frame, database, database_embeddings
    gc.collect()
    return {
        "scaled": scaled,
        "database_raw": database_raw,
        "channels": channel_columns,
        "origins": origins,
        "indices": timestamp_values,
        "distances": distance_values,
        "boundaries": boundaries,
        "audits": audits,
        "released_top_20_rows_recomputed": fallback_total,
        "csv_path": str(csv_path),
        "database_path": str(database_path),
    }


def evaluate_dataset(
    base_model,
    retrieval_model,
    prepared: dict,
    batch_size: int,
    context_length: int,
    prediction_length: int,
) -> dict:
    import torch

    device = torch.device("cuda:0")
    quantiles = retrieval_model.quantiles.detach().cpu()
    median_index = int(torch.abs(quantiles - 0.5).argmin())
    totals = {
        "base_squared": 0.0,
        "base_absolute": 0.0,
        "retrieval_squared": 0.0,
        "retrieval_absolute": 0.0,
        "points": 0,
    }
    channel_results = []
    for channel_index, channel in enumerate(prepared["channels"]):
        channel_totals = {key: 0.0 for key in totals}
        channel_totals["points"] = 0
        series = prepared["scaled"][:, channel_index]
        database = prepared["database_raw"][channel_index]
        indices = prepared["indices"][channel_index]
        distances = prepared["distances"][channel_index]
        origins = prepared["origins"]
        for offset in range(0, len(origins), batch_size):
            batch_origins = origins[offset : offset + batch_size]
            batch_indices = indices[offset : offset + batch_size]
            context = np.stack(
                [
                    series[start : start + context_length]
                    for start in batch_origins
                ]
            )
            target = np.stack(
                [
                    series[
                        start
                        + context_length : start
                        + context_length
                        + prediction_length
                    ]
                    for start in batch_origins
                ]
            )
            retrieved = np.stack(
                [
                    np.stack(
                        [
                            database[
                                start : start
                                + context_length
                                + prediction_length
                            ]
                            for start in row
                        ]
                    )
                    for row in batch_indices
                ]
            )
            context_tensor = torch.from_numpy(context).to(device)
            retrieved_tensor = torch.from_numpy(retrieved).to(device)
            distance_tensor = torch.from_numpy(
                distances[offset : offset + batch_size]
            ).to(device)
            with torch.inference_mode():
                base_prediction = base_model(
                    context=context_tensor
                ).quantile_preds[:, median_index]
                retrieval_prediction = retrieval_model(
                    context=context_tensor,
                    retrieved_seq=retrieved_tensor,
                    distances=distance_tensor,
                ).quantile_preds[:, median_index]
            target_tensor = torch.from_numpy(target).to(device)
            base_error = base_prediction - target_tensor
            retrieval_error = retrieval_prediction - target_tensor
            values = {
                "base_squared": float(base_error.square().sum().item()),
                "base_absolute": float(base_error.abs().sum().item()),
                "retrieval_squared": float(
                    retrieval_error.square().sum().item()
                ),
                "retrieval_absolute": float(
                    retrieval_error.abs().sum().item()
                ),
                "points": int(target_tensor.numel()),
            }
            for key, value in values.items():
                channel_totals[key] += value
                totals[key] += value
            del (
                context_tensor,
                target_tensor,
                retrieved_tensor,
                distance_tensor,
                base_prediction,
                retrieval_prediction,
            )
        channel_results.append(
            {
                "channel": channel,
                "test_origins": len(origins),
                "base": {
                    "mse": channel_totals["base_squared"]
                    / channel_totals["points"],
                    "mae": channel_totals["base_absolute"]
                    / channel_totals["points"],
                },
                "retrieval": {
                    "mse": channel_totals["retrieval_squared"]
                    / channel_totals["points"],
                    "mae": channel_totals["retrieval_absolute"]
                    / channel_totals["points"],
                },
            }
        )
    return {
        "base": {
            "mse": totals["base_squared"] / totals["points"],
            "mae": totals["base_absolute"] / totals["points"],
        },
        "retrieval": {
            "mse": totals["retrieval_squared"] / totals["points"],
            "mae": totals["retrieval_absolute"] / totals["points"],
        },
        "points": totals["points"],
        "channel_results": channel_results,
    }


def run_pair(
    row: dict,
    protocol: dict,
    protocol_sha256: str,
    source_revision: str,
    output_root: Path,
    data_root: Path,
    base_model,
    retrieval_model,
    embedding_model,
    batch_size: int,
) -> None:
    artifact = cell_dir(output_root, row)
    artifact.mkdir(parents=True, exist_ok=True)
    started = time.time()
    prepared = None
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
        settings = protocol["systems"]["ts_rag"]
        prepared = read_dataset(
            data_root,
            row["dataset"],
            int(settings["retrieval_top_k"]),
            row["context_length"],
            row["prediction_length"],
            embedding_model,
        )
        metrics = evaluate_dataset(
            base_model,
            retrieval_model,
            prepared,
            batch_size,
            row["context_length"],
            row["prediction_length"],
        )
        base_metrics = metrics["base"]
        retrieval_metrics = metrics["retrieval"]
        values = list(base_metrics.values()) + list(retrieval_metrics.values())
        if not all(np.isfinite(value) for value in values):
            raise ValueError("TS-RAG produced a non-finite metric")
        finished = time.time()
        result = {
            "schema_version": 1,
            "status": "completed",
            "cell_id": row["id"],
            "method": "TS-RAG",
            "base_method": "Chronos-Bolt",
            "backbone": settings["backbone"],
            "dataset": row["dataset"],
            "context_length": row["context_length"],
            "prediction_length": row["prediction_length"],
            "seed": settings["seed"],
            "retrieval_top_k": settings["retrieval_top_k"],
            "metric_space": settings["metric_space"],
            "base": {"metrics": base_metrics},
            "retrieval": {"metrics": retrieval_metrics},
            "delta": {
                metric: retrieval_metrics[metric] - base_metrics[metric]
                for metric in ("mse", "mae")
            },
            "test_points": metrics["points"],
            "test_origins_per_channel": len(prepared["origins"]),
            "channel_count": len(prepared["channels"]),
            "channel_results": metrics["channel_results"],
            "retrieval_index_audit": {
                "all_reference_futures_end_in_training": True,
                "released_top_20_rows_recomputed": prepared[
                    "released_top_20_rows_recomputed"
                ],
                "boundaries": prepared["boundaries"],
                "channels": prepared["audits"],
            },
            "test_label_forward_argument": "omitted",
            "csv_path": prepared["csv_path"],
            "database_path": prepared["database_path"],
            "protocol_sha256": protocol_sha256,
            "source_revision": source_revision,
            "upstream_commit": protocol["sources"]["ts_rag"]["commit"],
            "started_unix": started,
            "finished_unix": finished,
            "elapsed_seconds": finished - started,
            "pid": os.getpid(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "torch_device": "cuda:0",
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
        if prepared is not None:
            del prepared
        gc.collect()
        import torch

        torch.cuda.empty_cache()


def run_worker(args: argparse.Namespace, protocol: dict, rows: list[dict]) -> int:
    import torch

    if torch.cuda.device_count() != 1:
        raise RuntimeError("Each TS-RAG worker must see exactly one GPU")
    assigned = [
        row
        for index, row in enumerate(rows)
        if index % args.worker_count == args.worker_id
    ]
    if not assigned:
        return 0
    seed_everything(int(protocol["systems"]["ts_rag"]["seed"]))
    base_model, retrieval_model, embedding_model = load_models(
        args.upstream_root, args.checkpoint_root, args.chronos_root
    )
    protocol_sha256 = sha256_file(args.protocol)
    for row in assigned:
        if completed(args.output_root, row, protocol_sha256):
            continue
        run_pair(
            row,
            protocol,
            protocol_sha256,
            args.source_revision,
            args.output_root,
            args.data_root,
            base_model,
            retrieval_model,
            embedding_model,
            args.batch_size,
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
        if line.strip():
            pid, uuid, memory = [
                item.strip() for item in line.split(",", 2)
            ]
            rows.append(
                {
                    "pid": int(pid),
                    "gpu_uuid": uuid,
                    "used_memory_mib": int(memory),
                }
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
    topology_path: Path,
    asset_verification_path: Path,
) -> dict:
    results = []
    failures = []
    for row in rows:
        artifact = cell_dir(output_root, row)
        if completed(output_root, row, protocol_sha256):
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
            failures.append({"cell_id": row["id"], **status})
    summary = {
        "schema_version": 1,
        "method": "TS-RAG",
        "expected_pairs": len(rows),
        "completed_pairs": len(results),
        "failed_or_missing_pairs": len(failures),
        "all_completed": len(results) == len(rows) and not failures,
        "protocol_sha256": protocol_sha256,
        "source_revision": source_revision,
        "topology_path": str(topology_path),
        "asset_verification_path": str(asset_verification_path),
        "results": results,
        "failures": failures,
    }
    save_json_atomic(output_root / "summary.json", summary)
    return summary


def run_parent(
    args: argparse.Namespace, protocol: dict, rows: list[dict]
) -> int:
    import torch

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
    if args.worker_count > len(rows):
        raise ValueError("worker_count exceeds the number of TS-RAG cells")
    args.output_root.mkdir(parents=True, exist_ok=True)
    logs = args.output_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    asset_verification = verify_assets(args.asset_manifest)
    asset_verification_path = args.output_root / "asset_verification.json"
    save_json_atomic(asset_verification_path, asset_verification)
    protocol_sha256 = sha256_file(args.protocol)
    uuid_to_gpu = query_gpu_uuids()
    processes = {}
    log_files = {}
    observations = []
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--protocol",
        str(args.protocol),
        "--asset-manifest",
        str(args.asset_manifest),
        "--upstream-root",
        str(args.upstream_root),
        "--checkpoint-root",
        str(args.checkpoint_root),
        "--chronos-root",
        str(args.chronos_root),
        "--data-root",
        str(args.data_root),
        "--output-root",
        str(args.output_root),
        "--source-revision",
        args.source_revision,
        "--worker-count",
        str(args.worker_count),
        "--batch-size",
        str(args.batch_size),
    ]
    try:
        for worker_id in range(args.worker_count):
            log_file = (logs / f"worker-{worker_id}.log").open(
                "a", encoding="utf-8"
            )
            environment = dict(os.environ)
            environment["CUDA_VISIBLE_DEVICES"] = str(worker_id)
            environment["PYTHONUNBUFFERED"] = "1"
            process = subprocess.Popen(
                command + ["--worker-id", str(worker_id)],
                cwd=args.upstream_root / "TS-RAG",
                env=environment,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            processes[worker_id] = process
            log_files[worker_id] = log_file
        while any(process.poll() is None for process in processes.values()):
            observations.append(
                {
                    "recorded_unix": time.time(),
                    "processes": [
                        {
                            **record,
                            "physical_gpu": uuid_to_gpu.get(
                                record["gpu_uuid"]
                            ),
                        }
                        for record in query_gpu_processes()
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
    topology = {
        "schema_version": 1,
        "instance_type": args.instance_type,
        "instance_count": 1,
        "reserved_gpus_per_host": args.reserved_gpus_per_host,
        "processes_per_host": args.worker_count,
        "worker_gpu_ids": list(range(args.worker_count)),
        "total_gpus": args.reserved_gpus_per_host,
        "world_size": args.worker_count,
        "inactive_reserved_gpus": (
            args.reserved_gpus_per_host - args.worker_count
        ),
        "inactive_reason": (
            None
            if args.reserved_gpus_per_host == args.worker_count
            else (
                "The frozen TS-RAG matrix has seven independent dataset "
                "cells, so no eighth cell is available."
            )
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
        "observations": observations,
    }
    topology_path = args.output_root / "startup_topology.json"
    save_json_atomic(topology_path, topology)
    summary = aggregate(
        args.output_root,
        rows,
        protocol_sha256,
        args.source_revision,
        topology_path,
        asset_verification_path,
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
    parser.add_argument(
        "--asset-manifest",
        type=Path,
        default=Path("docs/native_ts_rag_assets.json"),
    )
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--chronos-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--worker-count", type=int, default=4)
    parser.add_argument(
        "--instance-type", default="ml.g5.12xlarge"
    )
    parser.add_argument(
        "--reserved-gpus-per-host", type=int, default=4
    )
    parser.add_argument("--worker-id", type=int)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    rows = ts_rag_rows(protocol)
    if args.worker_id is not None:
        return run_worker(args, protocol, rows)
    return run_parent(args, protocol, rows)


if __name__ == "__main__":
    raise SystemExit(main())
