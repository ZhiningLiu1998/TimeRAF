#!/usr/bin/env python3

from __future__ import annotations

import argparse
import base64
import gc
import hashlib
import importlib.util
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

FROZEN_SEED = 1
FROZEN_EPOCHS = 100
FROZEN_DIFFUSION_STEPS = 50
FROZEN_SAMPLES = 100
FROZEN_TEST_ORIGINS = 5_093
FROZEN_CONTEXT_LENGTH = 96
FROZEN_PREDICTION_LENGTH = 168
FROZEN_FEATURES = 321
FROZEN_SEQUENCE_LENGTH = 264
RNG_CALLS_PER_ORIGIN = FROZEN_SAMPLES * FROZEN_DIFFUSION_STEPS


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(document: dict) -> str:
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


def load_json(path: Path) -> dict:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return document


def file_record(path: Path) -> dict:
    resolved = path.resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size_bytes": stat.st_size,
        "sha256": sha256_file(resolved),
    }


def validate_file_record(record: dict, label: str) -> Path:
    if not isinstance(record, dict):
        raise ValueError(f"{label} file record is missing")
    path_value = record.get("path")
    if not isinstance(path_value, str) or not path_value:
        raise ValueError(f"{label} path is missing")
    path = Path(path_value)
    stat = path.stat()
    if stat.st_size != record.get("size_bytes"):
        raise ValueError(f"{label} size drifted")
    if sha256_file(path) != record.get("sha256"):
        raise ValueError(f"{label} SHA-256 drifted")
    return path


def rng_state_bytes(state: Any) -> bytes:
    if isinstance(state, bytes):
        return state
    if isinstance(state, bytearray):
        return bytes(state)
    try:
        array = state.detach().cpu().contiguous().numpy()
    except AttributeError as error:
        raise TypeError("Unsupported RNG state representation") from error
    return array.tobytes()


def encode_rng_state(state: Any) -> dict:
    payload = rng_state_bytes(state)
    return {
        "encoding": "base64-raw-uint8",
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "data": base64.b64encode(payload).decode("ascii"),
    }


def validate_rng_state_record(record: dict) -> bytes:
    if not isinstance(record, dict):
        raise ValueError("RNG state record is missing")
    if record.get("encoding") != "base64-raw-uint8":
        raise ValueError("RNG state encoding drifted")
    try:
        payload = base64.b64decode(record["data"], validate=True)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("RNG state payload is invalid") from error
    if len(payload) != record.get("size_bytes"):
        raise ValueError("RNG state size drifted")
    if hashlib.sha256(payload).hexdigest() != record.get("sha256"):
        raise ValueError("RNG state SHA-256 drifted")
    return payload


def decode_rng_state(record: dict, torch_module) -> Any:
    payload = validate_rng_state_record(record)
    return torch_module.frombuffer(bytearray(payload), dtype=torch_module.uint8).clone()


def build_contiguous_shards(total_origins: int, gpu_ids: list[int]) -> list[dict]:
    if total_origins <= 0:
        raise ValueError("total_origins must be positive")
    if not gpu_ids:
        raise ValueError("At least one GPU ID is required")
    if len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError("GPU IDs must be unique")
    if any(not isinstance(gpu_id, int) or gpu_id < 0 for gpu_id in gpu_ids):
        raise ValueError("GPU IDs must be non-negative integers")
    if len(gpu_ids) > total_origins:
        raise ValueError("There cannot be more workers than test origins")

    quotient, remainder = divmod(total_origins, len(gpu_ids))
    shards = []
    start = 0
    for worker_index, gpu_id in enumerate(gpu_ids):
        count = quotient + int(worker_index < remainder)
        end = start + count
        shards.append(
            {
                "shard_id": f"{worker_index:04d}",
                "worker_index": worker_index,
                "physical_gpu": gpu_id,
                "start_index": start,
                "end_index": end,
                "origin_count": count,
            }
        )
        start = end
    validate_contiguous_shards(shards, total_origins)
    return shards


def validate_contiguous_shards(shards: list[dict], total_origins: int) -> None:
    if not isinstance(shards, list) or not shards:
        raise ValueError("Shard plan is empty")
    shard_ids = [record.get("shard_id") for record in shards]
    if any(
        not isinstance(shard_id, str) or not shard_id for shard_id in shard_ids
    ) or len(set(shard_ids)) != len(shard_ids):
        raise ValueError("Shard IDs are missing or duplicated")
    expected_start = 0
    for shard in shards:
        start = shard.get("start_index")
        end = shard.get("end_index")
        if (
            not isinstance(start, int)
            or not isinstance(end, int)
            or start != expected_start
            or end <= start
            or shard.get("origin_count") != end - start
        ):
            raise ValueError("Shards are not a complete contiguous partition")
        expected_start = end
    if expected_start != total_origins:
        raise ValueError("Shard plan does not cover every test origin")


def _cuda_set_rng_state(torch_module, state: Any, device: Any) -> None:
    try:
        torch_module.cuda.set_rng_state(state, device=device)
    except TypeError:
        torch_module.cuda.set_rng_state(state)


def _cuda_get_rng_state(torch_module, device: Any) -> Any:
    try:
        return torch_module.cuda.get_rng_state(device=device)
    except TypeError:
        return torch_module.cuda.get_rng_state()


def _cuda_synchronize(torch_module, device: Any) -> None:
    synchronize = getattr(torch_module.cuda, "synchronize", None)
    if synchronize is None:
        return
    try:
        synchronize(device)
    except TypeError:
        synchronize()


def plan_rng_boundaries(
    torch_module,
    observed_data: Any,
    start_state: Any,
    shards: list[dict],
    *,
    samples: int,
    diffusion_steps: int,
    device: Any = None,
) -> list[dict]:
    if samples <= 0 or diffusion_steps <= 0:
        raise ValueError("Sampling dimensions must be positive")
    total_origins = shards[-1]["end_index"]
    validate_contiguous_shards(shards, total_origins)
    calls_per_origin = samples * diffusion_steps
    _cuda_set_rng_state(torch_module, start_state, device)

    planned = []
    for shard in shards:
        start_rng = encode_rng_state(_cuda_get_rng_state(torch_module, device))
        last_noise = None
        for _origin_index in range(shard["start_index"], shard["end_index"]):
            for _ in range(calls_per_origin):
                last_noise = torch_module.randn_like(observed_data)
        del last_noise
        _cuda_synchronize(torch_module, device)
        end_rng = encode_rng_state(_cuda_get_rng_state(torch_module, device))
        planned.append(
            {
                **shard,
                "rng_calls_per_origin": calls_per_origin,
                "rng_call_count": shard["origin_count"] * calls_per_origin,
                "start_rng": start_rng,
                "end_rng": end_rng,
            }
        )
    return planned


def aggregate_origin_records(records: list[dict], total_origins: int) -> dict:
    if not isinstance(records, list):
        raise ValueError("Origin records must be a list")
    by_index: dict[int, dict] = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Origin record is invalid")
        index = record.get("index")
        if not isinstance(index, int) or index < 0:
            raise ValueError("Origin index is invalid")
        if index in by_index:
            raise ValueError(f"Duplicate origin index: {index}")
        squared = record.get("squared")
        absolute = record.get("absolute")
        points = record.get("points")
        if (
            not isinstance(squared, (int, float))
            or not math.isfinite(float(squared))
            or float(squared) < 0
            or not isinstance(absolute, (int, float))
            or not math.isfinite(float(absolute))
            or float(absolute) < 0
            or not isinstance(points, int)
            or points <= 0
        ):
            raise ValueError(f"Origin {index} metrics are invalid")
        by_index[index] = record

    expected = set(range(total_origins))
    actual = set(by_index)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise ValueError(
            "Origin coverage is incomplete: "
            f"missing={missing[:10]}, unexpected={unexpected[:10]}"
        )

    squared_total = 0.0
    absolute_total = 0.0
    points_total = 0
    ordered = []
    for index in range(total_origins):
        record = by_index[index]
        squared_total += float(record["squared"])
        absolute_total += float(record["absolute"])
        points_total += int(record["points"])
        ordered.append(record)
    mse = squared_total / points_total
    metrics = {
        "rmse": float(math.sqrt(mse)),
        "mae": absolute_total / points_total,
    }
    if not all(math.isfinite(value) for value in metrics.values()):
        raise ValueError("Aggregated metrics are non-finite")
    return {
        "records": ordered,
        "squared": squared_total,
        "absolute": absolute_total,
        "points": points_total,
        "metrics": metrics,
    }


def load_runner_module(path: Path):
    module_name = f"_timeraf_frozen_ratd_{sha256_file(path)[:12]}"
    specification = importlib.util.spec_from_file_location(module_name, path)
    if specification is None or specification.loader is None:
        raise ImportError(f"Cannot load frozen RATD runner from {path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def expected_checkpoint_identity(
    *,
    method: str,
    source_revision: str,
    upstream_commit: str,
    protocol_sha256: str,
    config_sha256: str,
    data_sha256: str,
    reference_map_sha256: str,
) -> dict:
    return {
        "method": method,
        "source_revision": source_revision,
        "upstream_commit": upstream_commit,
        "protocol_sha256": protocol_sha256,
        "config_sha256": config_sha256,
        "data_sha256": data_sha256,
        "reference_map_sha256": reference_map_sha256,
        "seed": FROZEN_SEED,
        "epochs": FROZEN_EPOCHS,
        "diffusion_steps": FROZEN_DIFFUSION_STEPS,
        "samples": FROZEN_SAMPLES,
    }


def build_identity(args: argparse.Namespace, torch_module) -> tuple[dict, dict]:
    method = args.method.upper()
    if method not in {"RATD", "CSDI"}:
        raise ValueError("Method must be RATD or CSDI")
    reference_metadata = args.reference_map.with_suffix(".json")
    upstream_main_model = args.upstream_root / "main_model.py"
    upstream_diff_models = args.upstream_root / "diff_models.py"
    files = {
        "sharded_runner": file_record(Path(__file__)),
        "runner": file_record(args.runner),
        "upstream_main_model": file_record(upstream_main_model),
        "upstream_diff_models": file_record(upstream_diff_models),
        "config": file_record(args.config),
        "data": file_record(args.data),
        "reference_map": file_record(args.reference_map),
        "reference_metadata": file_record(reference_metadata),
        "protocol": file_record(args.protocol),
        "checkpoint": file_record(args.checkpoint),
    }
    checkpoint = torch_module.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )
    if checkpoint.get("epoch") != FROZEN_EPOCHS:
        raise ValueError("Evaluation requires an epoch-100 checkpoint")
    expected = expected_checkpoint_identity(
        method=method,
        source_revision=args.source_revision,
        upstream_commit=args.upstream_commit,
        protocol_sha256=files["protocol"]["sha256"],
        config_sha256=files["config"]["sha256"],
        data_sha256=files["data"]["sha256"],
        reference_map_sha256=files["reference_map"]["sha256"],
    )
    if checkpoint.get("identity") != expected:
        raise ValueError("Checkpoint identity does not match the frozen run")
    cuda_rng = checkpoint.get("cuda_rng")
    if cuda_rng is None:
        raise ValueError("Checkpoint CUDA RNG state is missing")
    identity = {
        "method": method,
        "use_reference": method == "RATD",
        "source_revision": args.source_revision,
        "launcher_revision": args.launcher_revision,
        "upstream_commit": args.upstream_commit,
        "upstream_root": str(args.upstream_root.resolve()),
        "files": files,
        "checkpoint_epoch": FROZEN_EPOCHS,
        "checkpoint_identity": expected,
        "seed": FROZEN_SEED,
        "epochs": FROZEN_EPOCHS,
        "diffusion_steps": FROZEN_DIFFUSION_STEPS,
        "samples": FROZEN_SAMPLES,
        "context_length": FROZEN_CONTEXT_LENGTH,
        "prediction_length": FROZEN_PREDICTION_LENGTH,
        "test_origins": FROZEN_TEST_ORIGINS,
    }
    return identity, checkpoint


def verify_identity(identity: dict) -> dict[str, Path]:
    if identity.get("method") not in {"RATD", "CSDI"}:
        raise ValueError("Plan method identity drifted")
    expected_constants = {
        "seed": FROZEN_SEED,
        "epochs": FROZEN_EPOCHS,
        "diffusion_steps": FROZEN_DIFFUSION_STEPS,
        "samples": FROZEN_SAMPLES,
        "context_length": FROZEN_CONTEXT_LENGTH,
        "prediction_length": FROZEN_PREDICTION_LENGTH,
        "test_origins": FROZEN_TEST_ORIGINS,
        "checkpoint_epoch": FROZEN_EPOCHS,
    }
    for key, expected in expected_constants.items():
        if identity.get(key) != expected:
            raise ValueError(f"Plan {key} drifted")
    paths = {
        label: validate_file_record(record, label)
        for label, record in identity.get("files", {}).items()
    }
    expected_labels = {
        "sharded_runner",
        "runner",
        "upstream_main_model",
        "upstream_diff_models",
        "config",
        "data",
        "reference_map",
        "reference_metadata",
        "protocol",
        "checkpoint",
    }
    if set(paths) != expected_labels:
        raise ValueError("Plan file coverage drifted")
    if not Path(identity["upstream_root"]).is_dir():
        raise ValueError("Upstream RATD root is missing")
    return paths


def load_model_and_dataset(identity: dict, device: Any):
    import torch
    import yaml

    paths = verify_identity(identity)
    runner = load_runner_module(paths["runner"])
    model_class = runner.install_repairs(Path(identity["upstream_root"]))
    config = yaml.safe_load(paths["config"].read_text(encoding="utf-8"))
    config["train"]["epochs"] = FROZEN_EPOCHS
    config["model"]["use_reference"] = identity["use_reference"]
    config["model"]["is_unconditional"] = False
    config["diffusion"]["h_size"] = FROZEN_CONTEXT_LENGTH
    config["diffusion"]["ref_size"] = FROZEN_PREDICTION_LENGTH
    if config["diffusion"].get("num_steps") != FROZEN_DIFFUSION_STEPS:
        raise ValueError("Diffusion-step configuration drifted")

    checkpoint = torch.load(paths["checkpoint"], map_location="cpu", weights_only=False)
    if checkpoint.get("epoch") != FROZEN_EPOCHS:
        raise ValueError("Checkpoint epoch drifted")
    if checkpoint.get("identity") != identity.get("checkpoint_identity"):
        raise ValueError("Checkpoint identity drifted")

    values = runner.load_electricity(paths["data"])
    reference_map = runner.load_reference_map(paths["reference_map"])
    split = runner.split_starts(
        len(values), FROZEN_CONTEXT_LENGTH, FROZEN_PREDICTION_LENGTH
    )
    dataset_class = runner.make_dataset_class()
    dataset = dataset_class(
        values,
        reference_map["test_starts"],
        reference_map["test_references"],
        int(split["train_end"]),
        identity["use_reference"],
    )
    if len(dataset) != FROZEN_TEST_ORIGINS:
        raise ValueError("Test-origin count drifted")

    model = model_class(config, device, FROZEN_FEATURES).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return {
        "torch": torch,
        "runner": runner,
        "model": model,
        "dataset": dataset,
        "checkpoint": checkpoint,
        "paths": paths,
    }


def observed_data_from_real_batch(runtime: dict, device: Any):
    torch_module = runtime["torch"]
    loader = torch_module.utils.data.DataLoader(
        runtime["dataset"],
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )
    batch = next(iter(loader))
    observed_data = runtime["model"].process_data(batch)[0]
    shape = list(observed_data.shape)
    stride = list(observed_data.stride())
    if shape != [1, FROZEN_FEATURES, FROZEN_SEQUENCE_LENGTH]:
        raise ValueError(f"Observed-data shape drifted: {shape}")
    if str(observed_data.dtype) != "torch.float32":
        raise ValueError("Observed-data dtype drifted")
    if observed_data.device != device:
        raise ValueError("Observed data is on the wrong planner device")
    return observed_data, {
        "shape": shape,
        "stride": stride,
        "dtype": str(observed_data.dtype),
        "numel": observed_data.numel(),
    }


def create_evaluation_plan(args: argparse.Namespace, output_path: Path) -> dict:
    import torch

    visible_environment = os.environ.get("CUDA_VISIBLE_DEVICES")
    expected_visibility = ",".join(
        str(index) for index in range(args.reserved_gpus_per_host)
    )
    if visible_environment not in (None, "", expected_visibility):
        raise ValueError("Parent CUDA_VISIBLE_DEVICES must preserve physical GPU order")
    if torch.cuda.device_count() != args.reserved_gpus_per_host:
        raise ValueError("Visible GPU count does not match reserved topology")
    if args.planner_gpu not in args.gpu_ids:
        raise ValueError("Planner GPU must be one of the worker GPU IDs")
    if args.planner_gpu >= torch.cuda.device_count():
        raise ValueError("Planner GPU is outside the visible topology")

    identity, checkpoint = build_identity(args, torch)
    device = torch.device(f"cuda:{args.planner_gpu}")
    runtime = None
    try:
        runtime = load_model_and_dataset(identity, device)
        observed_data, observed_metadata = observed_data_from_real_batch(
            runtime, device
        )
        checkpoint_rng = checkpoint["cuda_rng"].detach().cpu()
        shards = build_contiguous_shards(FROZEN_TEST_ORIGINS, args.gpu_ids)
        planned_shards = plan_rng_boundaries(
            torch,
            observed_data,
            checkpoint_rng,
            shards,
            samples=FROZEN_SAMPLES,
            diffusion_steps=FROZEN_DIFFUSION_STEPS,
            device=device,
        )
        document = {
            "schema_version": 1,
            "job_kind": "native_ratd_origin_sharded_eval_plan",
            "created_unix": time.time(),
            "identity": identity,
            "instance_type": args.instance_type,
            "instance_count": 1,
            "reserved_gpus_per_host": args.reserved_gpus_per_host,
            "worker_gpu_ids": args.gpu_ids,
            "processes_per_host": len(args.gpu_ids),
            "world_size": len(args.gpu_ids),
            "planner": {
                "pid": os.getpid(),
                "physical_gpu": args.planner_gpu,
                "torch_device": str(device),
            },
            "observed_data": observed_metadata,
            "checkpoint_cuda_rng": encode_rng_state(checkpoint_rng),
            "rng_calls_per_origin": RNG_CALLS_PER_ORIGIN,
            "total_rng_calls": (FROZEN_TEST_ORIGINS * RNG_CALLS_PER_ORIGIN),
            "shards": planned_shards,
        }
        save_json_atomic(output_path, document)
        return document
    finally:
        if runtime is not None:
            del runtime
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def validate_plan(plan: dict) -> dict[str, Path]:
    if (
        plan.get("schema_version") != 1
        or plan.get("job_kind") != "native_ratd_origin_sharded_eval_plan"
    ):
        raise ValueError("Evaluation plan schema drifted")
    identity = plan.get("identity")
    if not isinstance(identity, dict):
        raise ValueError("Evaluation plan identity is missing")
    paths = verify_identity(identity)
    if plan.get("rng_calls_per_origin") != RNG_CALLS_PER_ORIGIN:
        raise ValueError("Plan RNG-call count drifted")
    if plan.get("total_rng_calls") != (FROZEN_TEST_ORIGINS * RNG_CALLS_PER_ORIGIN):
        raise ValueError("Plan total RNG-call count drifted")
    shards = plan.get("shards")
    validate_contiguous_shards(shards, FROZEN_TEST_ORIGINS)
    if [shard["physical_gpu"] for shard in shards] != plan.get("worker_gpu_ids"):
        raise ValueError("Plan worker GPU assignment drifted")
    if len(shards) != plan.get("processes_per_host"):
        raise ValueError("Plan process count drifted")
    observed = plan.get("observed_data")
    if (
        not isinstance(observed, dict)
        or observed.get("shape") != [1, FROZEN_FEATURES, FROZEN_SEQUENCE_LENGTH]
        or observed.get("dtype") != "torch.float32"
        or observed.get("numel") != FROZEN_FEATURES * FROZEN_SEQUENCE_LENGTH
    ):
        raise ValueError("Plan observed-data identity drifted")
    checkpoint_rng = plan.get("checkpoint_cuda_rng")
    validate_rng_state_record(checkpoint_rng)
    for index, shard in enumerate(shards):
        if shard.get("rng_calls_per_origin") != RNG_CALLS_PER_ORIGIN:
            raise ValueError("Shard RNG-call count drifted")
        expected_calls = shard["origin_count"] * RNG_CALLS_PER_ORIGIN
        if shard.get("rng_call_count") != expected_calls:
            raise ValueError("Shard total RNG-call count drifted")
        validate_rng_state_record(shard.get("start_rng"))
        validate_rng_state_record(shard.get("end_rng"))
        if index == 0 and shard["start_rng"] != checkpoint_rng:
            raise ValueError("First shard does not start at checkpoint RNG")
        if index > 0 and (shard["start_rng"] != shards[index - 1]["end_rng"]):
            raise ValueError("Adjacent shard RNG boundaries do not match")
    return paths


def tensor_sha256(tensor: Any) -> str:
    payload = tensor.detach().cpu().contiguous().numpy().tobytes(order="C")
    return hashlib.sha256(payload).hexdigest()


def run_worker(plan_path: Path, shard_id: str, output_path: Path) -> dict:
    import torch

    if output_path.exists() or output_path.with_suffix(".failed.json").exists():
        raise FileExistsError("Refusing to overwrite a shard result")
    plan = load_json(plan_path)
    validate_plan(plan)
    plan_sha256 = sha256_file(plan_path)
    matching = [shard for shard in plan["shards"] if shard["shard_id"] == shard_id]
    if len(matching) != 1:
        raise ValueError(f"Unknown or duplicated shard ID: {shard_id}")
    shard = matching[0]
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != str(shard["physical_gpu"]):
        raise ValueError("Worker CUDA_VISIBLE_DEVICES does not match the shard plan")
    if torch.cuda.device_count() != 1:
        raise ValueError("A shard worker must see exactly one GPU")
    device = torch.device("cuda:0")
    runtime = load_model_and_dataset(plan["identity"], device)
    model = runtime["model"]
    dataset = runtime["dataset"]
    subset = torch.utils.data.Subset(
        dataset, range(shard["start_index"], shard["end_index"])
    )
    loader = torch.utils.data.DataLoader(
        subset, batch_size=1, shuffle=False, num_workers=0
    )

    start_rng = decode_rng_state(shard["start_rng"], torch)
    torch.cuda.set_rng_state(start_rng, device=device)
    started = time.time()
    origin_records = []
    first_sample = None
    last_sample = None
    with torch.inference_mode():
        for offset, batch in enumerate(loader):
            origin_index = shard["start_index"] + offset
            samples, target, target_mask = model.evaluate(batch, FROZEN_SAMPLES)
            median = samples.median(dim=1).values
            error = (median - target) * target_mask
            origin_records.append(
                {
                    "index": origin_index,
                    "squared": float(error.square().sum().item()),
                    "absolute": float(error.abs().sum().item()),
                    "points": int(target_mask.sum().item()),
                }
            )
            if offset == 0:
                first_sample = {
                    "origin_index": origin_index,
                    "sha256": tensor_sha256(samples),
                }
            if offset == shard["origin_count"] - 1:
                last_sample = {
                    "origin_index": origin_index,
                    "sha256": tensor_sha256(samples),
                }
    torch.cuda.synchronize(device)
    end_rng = encode_rng_state(torch.cuda.get_rng_state(device=device))
    if end_rng != shard["end_rng"]:
        raise ValueError(
            "Worker end RNG does not match the noise-only planner boundary"
        )
    if len(origin_records) != shard["origin_count"]:
        raise ValueError("Worker did not evaluate every assigned origin")
    if first_sample is None or last_sample is None:
        raise ValueError("Worker boundary sample hashes are missing")
    finished = time.time()
    result = {
        "schema_version": 1,
        "job_kind": "native_ratd_origin_shard_result",
        "status": "completed",
        "method": plan["identity"]["method"],
        "shard_id": shard_id,
        "shard": shard,
        "plan_path": str(plan_path.resolve()),
        "plan_sha256": plan_sha256,
        "pid": os.getpid(),
        "physical_gpu": shard["physical_gpu"],
        "cuda_visible_devices": visible,
        "torch_device": "cuda:0",
        "started_unix": started,
        "finished_unix": finished,
        "elapsed_seconds": finished - started,
        "origin_records": origin_records,
        "boundary_sample_sha256": {
            "first": first_sample,
            "last": last_sample,
        },
        "end_rng": end_rng,
    }
    save_json_atomic(output_path, result)
    return result


def validate_verification_plan(plan: dict) -> dict[str, Path]:
    if (
        plan.get("schema_version") != 1
        or plan.get("job_kind") != "native_ratd_exact_equivalence_plan"
    ):
        raise ValueError("Verification plan schema drifted")
    identity = plan.get("identity")
    if not isinstance(identity, dict):
        raise ValueError("Verification plan identity is missing")
    paths = verify_identity(identity)
    verification_range = plan.get("verification_range")
    if (
        not isinstance(verification_range, dict)
        or verification_range.get("origin_count") != 2
        or verification_range.get("end_index")
        != verification_range.get("start_index", -2) + 2
        or verification_range.get("start_index", -1) < 0
        or verification_range.get("end_index", FROZEN_TEST_ORIGINS + 1)
        > FROZEN_TEST_ORIGINS
    ):
        raise ValueError("Verification must cover two consecutive origins")
    gpu_ids = plan.get("physical_gpu_ids")
    if (
        not isinstance(gpu_ids, dict)
        or set(gpu_ids) != {"oracle", "comparison"}
        or len(set(gpu_ids.values())) != 2
        or any(not isinstance(gpu_id, int) or gpu_id < 0 for gpu_id in gpu_ids.values())
    ):
        raise ValueError("Verification GPU assignments are invalid")
    validate_rng_state_record(plan.get("checkpoint_cuda_rng"))
    validate_rng_state_record(plan.get("start_rng"))
    validate_rng_state_record(plan.get("expected_final_rng"))
    observed = plan.get("observed_data")
    if (
        not isinstance(observed, dict)
        or observed.get("shape") != [1, FROZEN_FEATURES, FROZEN_SEQUENCE_LENGTH]
        or observed.get("dtype") != "torch.float32"
        or observed.get("numel") != FROZEN_FEATURES * FROZEN_SEQUENCE_LENGTH
    ):
        raise ValueError("Verification observed-data identity drifted")
    if plan.get("samples") != FROZEN_SAMPLES:
        raise ValueError("Verification sample count drifted")
    if plan.get("diffusion_steps") != FROZEN_DIFFUSION_STEPS:
        raise ValueError("Verification diffusion-step count drifted")
    return paths


def create_verification_plan(args: argparse.Namespace, output_path: Path) -> dict:
    import torch

    visible_environment = os.environ.get("CUDA_VISIBLE_DEVICES")
    expected_visibility = ",".join(
        str(index) for index in range(args.reserved_gpus_per_host)
    )
    if visible_environment not in (None, "", expected_visibility):
        raise ValueError("Parent CUDA_VISIBLE_DEVICES must preserve physical GPU order")
    if torch.cuda.device_count() != args.reserved_gpus_per_host:
        raise ValueError("Visible GPU count does not match reserved topology")
    if args.oracle_gpu == args.comparison_gpu:
        raise ValueError("Oracle and comparison GPUs must be distinct")
    if any(
        gpu_id < 0 or gpu_id >= torch.cuda.device_count()
        for gpu_id in (args.oracle_gpu, args.comparison_gpu)
    ):
        raise ValueError("Verification GPU is outside the visible topology")
    if args.origin_start < 0 or args.origin_start + 2 > FROZEN_TEST_ORIGINS:
        raise ValueError("Verification range must contain two real origins")

    identity, checkpoint = build_identity(args, torch)
    device = torch.device(f"cuda:{args.oracle_gpu}")
    runtime = None
    try:
        runtime = load_model_and_dataset(identity, device)
        observed_data, observed_metadata = observed_data_from_real_batch(
            runtime, device
        )
        checkpoint_rng = checkpoint["cuda_rng"].detach().cpu()
        ranges = []
        if args.origin_start:
            ranges.append(
                {
                    "shard_id": "prefix",
                    "worker_index": 0,
                    "physical_gpu": args.oracle_gpu,
                    "start_index": 0,
                    "end_index": args.origin_start,
                    "origin_count": args.origin_start,
                }
            )
        ranges.append(
            {
                "shard_id": "verification",
                "worker_index": len(ranges),
                "physical_gpu": args.oracle_gpu,
                "start_index": args.origin_start,
                "end_index": args.origin_start + 2,
                "origin_count": 2,
            }
        )
        planned = plan_rng_boundaries(
            torch,
            observed_data,
            checkpoint_rng,
            ranges,
            samples=FROZEN_SAMPLES,
            diffusion_steps=FROZEN_DIFFUSION_STEPS,
            device=device,
        )
        verification_boundary = planned[-1]
        document = {
            "schema_version": 1,
            "job_kind": "native_ratd_exact_equivalence_plan",
            "created_unix": time.time(),
            "identity": identity,
            "instance_type": args.instance_type,
            "reserved_gpus_per_host": args.reserved_gpus_per_host,
            "physical_gpu_ids": {
                "oracle": args.oracle_gpu,
                "comparison": args.comparison_gpu,
            },
            "planner": {
                "pid": os.getpid(),
                "physical_gpu": args.oracle_gpu,
                "torch_device": str(device),
            },
            "verification_range": {
                "start_index": args.origin_start,
                "end_index": args.origin_start + 2,
                "origin_count": 2,
            },
            "samples": FROZEN_SAMPLES,
            "diffusion_steps": FROZEN_DIFFUSION_STEPS,
            "rng_calls_per_origin": RNG_CALLS_PER_ORIGIN,
            "prefix_rng_calls": args.origin_start * RNG_CALLS_PER_ORIGIN,
            "observed_data": observed_metadata,
            "checkpoint_cuda_rng": encode_rng_state(checkpoint_rng),
            "start_rng": verification_boundary["start_rng"],
            "expected_final_rng": verification_boundary["end_rng"],
        }
        save_json_atomic(output_path, document)
        validate_verification_plan(document)
        return document
    finally:
        if runtime is not None:
            del runtime
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_verification_worker(plan_path: Path, role: str, output_path: Path) -> dict:
    import torch

    if role not in {"oracle", "comparison"}:
        raise ValueError("Verification role is invalid")
    if output_path.exists() or output_path.with_suffix(".failed.json").exists():
        raise FileExistsError("Refusing to overwrite verification output")
    plan = load_json(plan_path)
    validate_verification_plan(plan)
    physical_gpu = plan["physical_gpu_ids"][role]
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(physical_gpu):
        raise ValueError("Verification CUDA_VISIBLE_DEVICES does not match the plan")
    if torch.cuda.device_count() != 1:
        raise ValueError("Verification worker must see exactly one GPU")

    device = torch.device("cuda:0")
    runtime = load_model_and_dataset(plan["identity"], device)
    verification_range = plan["verification_range"]
    subset = torch.utils.data.Subset(
        runtime["dataset"],
        range(
            verification_range["start_index"],
            verification_range["end_index"],
        ),
    )
    loader = torch.utils.data.DataLoader(
        subset, batch_size=1, shuffle=False, num_workers=0
    )
    start_rng = decode_rng_state(plan["start_rng"], torch)
    torch.cuda.set_rng_state(start_rng, device=device)
    started = time.time()
    origin_records = []
    with torch.inference_mode():
        for offset, batch in enumerate(loader):
            origin_index = verification_range["start_index"] + offset
            samples, target, target_mask = runtime["model"].evaluate(
                batch, FROZEN_SAMPLES
            )
            median = samples.median(dim=1).values
            error = (median - target) * target_mask
            origin_records.append(
                {
                    "index": origin_index,
                    "sample_sha256": tensor_sha256(samples),
                    "median_sha256": tensor_sha256(median),
                    "squared": float(error.square().sum().item()),
                    "absolute": float(error.abs().sum().item()),
                    "points": int(target_mask.sum().item()),
                    "end_rng": encode_rng_state(
                        torch.cuda.get_rng_state(device=device)
                    ),
                }
            )
    torch.cuda.synchronize(device)
    final_rng = encode_rng_state(torch.cuda.get_rng_state(device=device))
    if len(origin_records) != 2:
        raise ValueError("Verification worker did not evaluate two origins")
    if final_rng != origin_records[-1]["end_rng"]:
        raise ValueError("Per-origin and final RNG evidence differ")
    if final_rng != plan["expected_final_rng"]:
        raise ValueError("Real evaluator RNG does not match the noise-only planner")
    finished = time.time()
    result = {
        "schema_version": 1,
        "job_kind": "native_ratd_exact_equivalence_worker",
        "status": "completed",
        "method": plan["identity"]["method"],
        "role": role,
        "plan_path": str(plan_path.resolve()),
        "plan_sha256": sha256_file(plan_path),
        "pid": os.getpid(),
        "physical_gpu": physical_gpu,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_device": "cuda:0",
        "samples": FROZEN_SAMPLES,
        "diffusion_steps": FROZEN_DIFFUSION_STEPS,
        "origin_records": origin_records,
        "start_rng": plan["start_rng"],
        "final_rng": final_rng,
        "started_unix": started,
        "finished_unix": finished,
        "elapsed_seconds": finished - started,
    }
    save_json_atomic(output_path, result)
    return result


def verification_projection(result: dict) -> dict:
    if (
        result.get("schema_version") != 1
        or result.get("job_kind") != "native_ratd_exact_equivalence_worker"
        or result.get("status") != "completed"
        or result.get("samples") != FROZEN_SAMPLES
        or result.get("diffusion_steps") != FROZEN_DIFFUSION_STEPS
    ):
        raise ValueError("Verification worker result identity drifted")
    records = result.get("origin_records")
    if (
        not isinstance(records, list)
        or len(records) != 2
        or any(
            set(record)
            != {
                "index",
                "sample_sha256",
                "median_sha256",
                "squared",
                "absolute",
                "points",
                "end_rng",
            }
            for record in records
        )
    ):
        raise ValueError("Verification origin evidence is incomplete")
    for record in records:
        if (
            not isinstance(record["sample_sha256"], str)
            or len(record["sample_sha256"]) != 64
            or not isinstance(record["median_sha256"], str)
            or len(record["median_sha256"]) != 64
            or not math.isfinite(float(record["squared"]))
            or not math.isfinite(float(record["absolute"]))
            or not isinstance(record["points"], int)
            or record["points"] <= 0
        ):
            raise ValueError("Verification origin evidence is invalid")
        validate_rng_state_record(record["end_rng"])
    validate_rng_state_record(result.get("start_rng"))
    validate_rng_state_record(result.get("final_rng"))
    return {
        "method": result["method"],
        "samples": result["samples"],
        "diffusion_steps": result["diffusion_steps"],
        "start_rng": result["start_rng"],
        "origin_records": records,
        "final_rng": result["final_rng"],
    }


def compare_verification_results(oracle: dict, comparison: dict) -> dict:
    oracle_projection = verification_projection(oracle)
    comparison_projection = verification_projection(comparison)
    if oracle_projection != comparison_projection:
        raise ValueError("Cross-GPU frozen evaluator results are not byte-identical")
    return oracle_projection


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


def query_gpu_processes() -> list[dict]:
    uuid_to_gpu = query_gpu_uuids()
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
    records = []
    for line in output.splitlines():
        if not line.strip():
            continue
        pid, uuid, memory = [part.strip() for part in line.split(",", 2)]
        records.append(
            {
                "pid": int(pid),
                "gpu_uuid": uuid,
                "physical_gpu": uuid_to_gpu.get(uuid),
                "used_memory_mib": int(memory),
            }
        )
    return records


def terminate_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
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


def run_verification_process(
    *,
    role: str,
    physical_gpu: int,
    plan_path: Path,
    output_path: Path,
    log_path: Path,
    poll_seconds: float,
) -> dict:
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)
    environment["PYTHONUNBUFFERED"] = "1"
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "verify-worker",
        "--plan",
        str(plan_path),
        "--role",
        role,
        "--output",
        str(output_path),
    ]
    observations = []
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            command,
            cwd=Path(
                load_json(plan_path)["identity"]["files"]["runner"]["path"]
            ).parent.parent,
            env=environment,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        try:
            while process.poll() is None:
                observations.append(
                    {
                        "recorded_unix": time.time(),
                        "processes": query_gpu_processes(),
                    }
                )
                time.sleep(poll_seconds)
        finally:
            terminate_process(process)
    observed_gpus = sorted(
        {
            int(record["physical_gpu"])
            for observation in observations
            for record in observation["processes"]
            if record["pid"] == process.pid and record["physical_gpu"] is not None
        }
    )
    if process.returncode != 0:
        raise RuntimeError(f"{role} verification worker exited {process.returncode}")
    if observed_gpus != [physical_gpu]:
        raise ValueError(f"{role} verification PID-to-GPU binding is invalid")
    result = load_json(output_path)
    if (
        result.get("pid") != process.pid
        or result.get("physical_gpu") != physical_gpu
        or result.get("role") != role
    ):
        raise ValueError(f"{role} verification process identity drifted")
    return {
        "role": role,
        "pid": process.pid,
        "physical_gpu": physical_gpu,
        "returncode": process.returncode,
        "observed_gpus": observed_gpus,
        "observations": observations,
        "result": file_record(output_path),
    }


def run_verification(args: argparse.Namespace) -> dict:
    if args.output.exists() or args.output.with_suffix(".failed.json").exists():
        raise FileExistsError("Refusing to overwrite verification output")
    artifact_root = args.output.parent / f"{args.output.stem}_artifacts"
    if artifact_root.exists():
        raise FileExistsError("Refusing to overwrite verification artifacts")
    if args.poll_seconds <= 0:
        raise ValueError("Verification poll interval must be positive")

    gpu_inventory = query_gpu_uuids()
    if len(gpu_inventory) != args.reserved_gpus_per_host:
        raise ValueError("Physical GPU count does not match reserved topology")
    selected_gpus = {args.oracle_gpu, args.comparison_gpu}
    foreign = [
        record
        for record in query_gpu_processes()
        if record["physical_gpu"] in selected_gpus
    ]
    if foreign:
        raise RuntimeError(f"Selected verification GPUs are already active: {foreign}")

    artifact_root.mkdir(parents=True)
    plan_path = artifact_root / "verification_plan.json"
    oracle_path = artifact_root / "oracle.json"
    comparison_path = artifact_root / "comparison.json"
    create_verification_plan(args, plan_path)
    plan = load_json(plan_path)
    validate_verification_plan(plan)

    oracle_process = run_verification_process(
        role="oracle",
        physical_gpu=args.oracle_gpu,
        plan_path=plan_path,
        output_path=oracle_path,
        log_path=artifact_root / "oracle.log",
        poll_seconds=args.poll_seconds,
    )
    comparison_process = run_verification_process(
        role="comparison",
        physical_gpu=args.comparison_gpu,
        plan_path=plan_path,
        output_path=comparison_path,
        log_path=artifact_root / "comparison.log",
        poll_seconds=args.poll_seconds,
    )
    oracle = load_json(oracle_path)
    comparison = load_json(comparison_path)
    compared = compare_verification_results(oracle, comparison)
    if (
        compared["start_rng"] != plan["start_rng"]
        or compared["final_rng"] != plan["expected_final_rng"]
    ):
        raise ValueError("Verification output does not match planned RNG")

    payload = {
        "schema_version": 1,
        "job_kind": "native_ratd_exact_cross_gpu_verification",
        "status": "verified",
        "created_unix": time.time(),
        "method": plan["identity"]["method"],
        "strict_byte_equivalence": True,
        "verification_range": plan["verification_range"],
        "samples": FROZEN_SAMPLES,
        "diffusion_steps": FROZEN_DIFFUSION_STEPS,
        "identity": {
            "source_revision": plan["identity"]["source_revision"],
            "launcher_revision": plan["identity"]["launcher_revision"],
            "upstream_commit": plan["identity"]["upstream_commit"],
            "runner": plan["identity"]["files"]["runner"],
            "sharded_runner": plan["identity"]["files"]["sharded_runner"],
            "checkpoint": plan["identity"]["files"]["checkpoint"],
            "protocol": plan["identity"]["files"]["protocol"],
            "data": plan["identity"]["files"]["data"],
            "reference_map": plan["identity"]["files"]["reference_map"],
        },
        "plan": file_record(plan_path),
        "start_rng": plan["start_rng"],
        "expected_final_rng": plan["expected_final_rng"],
        "oracle": {
            **oracle_process,
            "origin_records": oracle["origin_records"],
            "start_rng": oracle["start_rng"],
            "final_rng": oracle["final_rng"],
        },
        "comparison": {
            **comparison_process,
            "origin_records": comparison["origin_records"],
            "start_rng": comparison["start_rng"],
            "final_rng": comparison["final_rng"],
        },
    }
    verification = {
        **payload,
        "verification_sha256": canonical_json_sha256(payload),
    }
    save_json_atomic(args.output, verification)
    return verification


def validate_worker_outputs(
    plan_path: Path, worker_paths: dict[str, Path]
) -> tuple[list[dict], list[dict]]:
    plan = load_json(plan_path)
    validate_plan(plan)
    plan_sha256 = sha256_file(plan_path)
    expected_ids = {shard["shard_id"] for shard in plan["shards"]}
    if set(worker_paths) != expected_ids:
        raise ValueError("Worker output coverage does not match the plan")
    worker_results = []
    origin_records = []
    for shard in plan["shards"]:
        shard_id = shard["shard_id"]
        path = worker_paths[shard_id]
        result = load_json(path)
        if (
            result.get("schema_version") != 1
            or result.get("job_kind") != "native_ratd_origin_shard_result"
            or result.get("status") != "completed"
            or result.get("method") != plan["identity"]["method"]
            or result.get("shard_id") != shard_id
            or result.get("shard") != shard
            or result.get("plan_sha256") != plan_sha256
            or result.get("end_rng") != shard["end_rng"]
        ):
            raise ValueError(f"Invalid worker result for shard {shard_id}")
        records = result.get("origin_records")
        if not isinstance(records, list) or len(records) != shard["origin_count"]:
            raise ValueError(f"Shard {shard_id} origin records are incomplete")
        boundary_hashes = result.get("boundary_sample_sha256")
        if (
            not isinstance(boundary_hashes, dict)
            or set(boundary_hashes) != {"first", "last"}
            or any(
                not isinstance(boundary_hashes[key].get("sha256"), str)
                or len(boundary_hashes[key]["sha256"]) != 64
                for key in ("first", "last")
            )
        ):
            raise ValueError(f"Shard {shard_id} sample hashes are invalid")
        worker_results.append(
            {
                "shard_id": shard_id,
                "path": str(path.resolve()),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "pid": result["pid"],
                "physical_gpu": result["physical_gpu"],
                "started_unix": result["started_unix"],
                "finished_unix": result["finished_unix"],
                "boundary_sample_sha256": boundary_hashes,
                "end_rng_sha256": result["end_rng"]["sha256"],
            }
        )
        origin_records.extend(records)
    aggregate_origin_records(origin_records, FROZEN_TEST_ORIGINS)
    return worker_results, origin_records


def build_method_result(
    plan_path: Path,
    topology_path: Path,
    worker_paths: dict[str, Path],
) -> tuple[dict, list[dict]]:
    plan = load_json(plan_path)
    validate_plan(plan)
    worker_results, origin_records = validate_worker_outputs(plan_path, worker_paths)
    topology = load_json(topology_path)
    validate_topology_document(topology)
    if (
        topology.get("method") != plan["identity"]["method"]
        or topology.get("plan_sha256") != sha256_file(plan_path)
        or topology.get("worker_gpu_ids") != plan["worker_gpu_ids"]
        or topology.get("processes_per_host") != len(plan["shards"])
    ):
        raise ValueError("Method topology does not match the evaluation plan")
    for worker in worker_results:
        shard_id = worker["shard_id"]
        if (
            topology["worker_pids"].get(shard_id) != worker["pid"]
            or int(topology["expected_bindings"].get(str(worker["pid"]), -1))
            != worker["physical_gpu"]
            or topology["observed_bindings"].get(str(worker["pid"]))
            != [worker["physical_gpu"]]
        ):
            raise ValueError(
                f"Shard {shard_id} result does not match topology evidence"
            )
    aggregated = aggregate_origin_records(origin_records, FROZEN_TEST_ORIGINS)
    expected_points = FROZEN_TEST_ORIGINS * FROZEN_PREDICTION_LENGTH * FROZEN_FEATURES
    if aggregated["points"] != expected_points:
        raise ValueError("Aggregated test-point count drifted")
    identity = plan["identity"]
    result = {
        "schema_version": 2,
        "status": "completed",
        "method": identity["method"],
        "use_reference": identity["use_reference"],
        "backbone": "CSDI",
        "dataset": "electricity",
        "context_length": FROZEN_CONTEXT_LENGTH,
        "prediction_length": FROZEN_PREDICTION_LENGTH,
        "seed": FROZEN_SEED,
        "epochs": FROZEN_EPOCHS,
        "diffusion_steps": FROZEN_DIFFUSION_STEPS,
        "samples": FROZEN_SAMPLES,
        "test_origins": FROZEN_TEST_ORIGINS,
        "test_points": aggregated["points"],
        "metrics": aggregated["metrics"],
        "metric_totals": {
            "squared": aggregated["squared"],
            "absolute": aggregated["absolute"],
            "points": aggregated["points"],
        },
        "protocol_sha256": identity["files"]["protocol"]["sha256"],
        "source_revision": identity["source_revision"],
        "launcher_revision": identity["launcher_revision"],
        "upstream_commit": identity["upstream_commit"],
        "reference_map_sha256": identity["files"]["reference_map"]["sha256"],
        "checkpoint_epoch": identity["checkpoint_epoch"],
        "checkpoint_path": identity["files"]["checkpoint"]["path"],
        "checkpoint_sha256": identity["files"]["checkpoint"]["sha256"],
        "evaluation_mode": "origin-sharded-exact-rng",
        "plan_path": str(plan_path.resolve()),
        "plan_sha256": sha256_file(plan_path),
        "topology_path": str(topology_path.resolve()),
        "topology_sha256": sha256_file(topology_path),
        "worker_results": worker_results,
        "started_unix": min(result["started_unix"] for result in worker_results),
        "finished_unix": max(result["finished_unix"] for result in worker_results),
    }
    result["elapsed_seconds"] = result["finished_unix"] - result["started_unix"]
    return result, worker_results


def make_receipt(
    *,
    plan_path: Path,
    topology_path: Path,
    method_result_path: Path,
    worker_results: list[dict],
) -> dict:
    plan = load_json(plan_path)
    payload = {
        "schema_version": 1,
        "job_kind": "native_ratd_origin_sharded_eval_receipt",
        "method": plan["identity"]["method"],
        "created_unix": time.time(),
        "plan": file_record(plan_path),
        "topology": file_record(topology_path),
        "method_result": file_record(method_result_path),
        "checkpoint": plan["identity"]["files"]["checkpoint"],
        "sharded_runner": plan["identity"]["files"]["sharded_runner"],
        "runner": plan["identity"]["files"]["runner"],
        "upstream_main_model": plan["identity"]["files"]["upstream_main_model"],
        "upstream_diff_models": plan["identity"]["files"]["upstream_diff_models"],
        "protocol": plan["identity"]["files"]["protocol"],
        "config": plan["identity"]["files"]["config"],
        "data": plan["identity"]["files"]["data"],
        "reference_map": plan["identity"]["files"]["reference_map"],
        "reference_metadata": plan["identity"]["files"]["reference_metadata"],
        "worker_results": worker_results,
    }
    return {
        **payload,
        "receipt_sha256": canonical_json_sha256(payload),
    }


def validate_topology_document(topology: dict) -> None:
    if (
        topology.get("schema_version") != 2
        or topology.get("job_kind") != "native_ratd_origin_sharded_eval_topology"
    ):
        raise ValueError("Evaluation topology schema drifted")
    worker_pids = topology.get("worker_pids")
    returncodes = topology.get("worker_returncodes")
    expected = topology.get("expected_bindings")
    observed = topology.get("observed_bindings")
    expected_gpu_ids = {
        "RATD": list(range(8)),
        "CSDI": list(range(1, 8)),
    }.get(topology.get("method"))
    if (
        expected_gpu_ids is None
        or topology.get("instance_type") != "ml.p5.48xlarge"
        or topology.get("reserved_gpus_per_host") != 8
        or topology.get("processes_per_host") != len(expected_gpu_ids)
        or topology.get("worker_gpu_ids") != expected_gpu_ids
        or topology.get("world_size") != len(expected_gpu_ids)
        or topology.get("inactive_reserved_gpus")
        != 8 - len(expected_gpu_ids)
        or not isinstance(worker_pids, dict)
        or len(worker_pids) != len(expected_gpu_ids)
        or not isinstance(returncodes, dict)
        or set(returncodes) != set(worker_pids)
        or any(int(code) != 0 for code in returncodes.values())
        or not isinstance(expected, dict)
        or not isinstance(observed, dict)
    ):
        raise ValueError("Evaluation worker completion evidence is invalid")
    pids = [int(pid) for pid in worker_pids.values()]
    if len(set(pids)) != len(pids) or any(pid <= 0 for pid in pids):
        raise ValueError("Evaluation worker PIDs are invalid")
    expected_by_pid = {int(pid): int(gpu) for pid, gpu in expected.items()}
    observed_by_pid = {
        int(pid): [int(gpu) for gpu in gpu_ids] for pid, gpu_ids in observed.items()
    }
    if set(expected_by_pid) != set(pids) or any(
        observed_by_pid.get(pid) != [gpu] for pid, gpu in expected_by_pid.items()
    ):
        raise ValueError("Evaluation worker GPU bindings are invalid")
    if topology.get("topology_valid") is not True:
        raise ValueError("Evaluation topology validity gate is false")


def run_orchestrator(args: argparse.Namespace) -> dict:
    method_root = args.output_root / args.method.lower()
    status_path = method_root / "status.json"
    plan_path = method_root / "eval_plan.json"
    topology_path = method_root / "startup_topology.json"
    result_path = method_root / "result.json"
    receipt_path = method_root / "receipt.json"
    if method_root.exists() and any(method_root.iterdir()):
        raise FileExistsError(
            "Refusing to overwrite an existing sharded evaluation attempt"
        )

    gpu_inventory = query_gpu_uuids()
    if len(gpu_inventory) != args.reserved_gpus_per_host:
        raise ValueError("Physical GPU count does not match reserved topology")
    foreign = [
        record
        for record in query_gpu_processes()
        if record["physical_gpu"] in args.gpu_ids
    ]
    if foreign:
        raise RuntimeError(
            f"Selected worker GPUs already have compute processes: {foreign}"
        )
    save_json_atomic(
        status_path,
        {
            "status": "planning",
            "method": args.method.upper(),
            "started_unix": time.time(),
            "pid": os.getpid(),
        },
    )
    create_evaluation_plan(args, plan_path)
    plan = load_json(plan_path)
    validate_plan(plan)

    logs = method_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    worker_paths = {
        shard["shard_id"]: method_root / "shards" / f'{shard["shard_id"]}.json'
        for shard in plan["shards"]
    }
    processes: dict[str, subprocess.Popen] = {}
    log_handles = {}
    observations = []
    try:
        for shard in plan["shards"]:
            shard_id = shard["shard_id"]
            log_handle = (logs / f"{shard_id}.log").open("a", encoding="utf-8")
            environment = dict(os.environ)
            environment["CUDA_VISIBLE_DEVICES"] = str(shard["physical_gpu"])
            environment["PYTHONUNBUFFERED"] = "1"
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "worker",
                "--plan",
                str(plan_path),
                "--shard-id",
                shard_id,
                "--output",
                str(worker_paths[shard_id]),
            ]
            process = subprocess.Popen(
                command,
                cwd=Path(plan["identity"]["files"]["runner"]["path"]).parent.parent,
                env=environment,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            processes[shard_id] = process
            log_handles[shard_id] = log_handle
        save_json_atomic(
            status_path,
            {
                "status": "running",
                "method": args.method.upper(),
                "plan_path": str(plan_path),
                "plan_sha256": sha256_file(plan_path),
                "worker_pids": {
                    shard_id: process.pid for shard_id, process in processes.items()
                },
            },
        )
        while any(process.poll() is None for process in processes.values()):
            observations.append(
                {
                    "recorded_unix": time.time(),
                    "processes": query_gpu_processes(),
                }
            )
            time.sleep(args.poll_seconds)
    finally:
        for process in processes.values():
            terminate_process(process)
        for log_handle in log_handles.values():
            log_handle.close()

    expected_bindings = {
        process.pid: plan["shards"][index]["physical_gpu"]
        for index, process in enumerate(processes.values())
    }
    observed_bindings = {
        pid: sorted(
            {
                int(record["physical_gpu"])
                for observation in observations
                for record in observation["processes"]
                if record["pid"] == pid and record["physical_gpu"] is not None
            }
        )
        for pid in expected_bindings
    }
    topology_valid = all(
        observed_bindings[pid] == [gpu] for pid, gpu in expected_bindings.items()
    )
    topology = {
        "schema_version": 2,
        "job_kind": "native_ratd_origin_sharded_eval_topology",
        "method": args.method.upper(),
        "instance_type": args.instance_type,
        "instance_count": 1,
        "reserved_gpus_per_host": args.reserved_gpus_per_host,
        "processes_per_host": len(args.gpu_ids),
        "worker_gpu_ids": args.gpu_ids,
        "total_gpus": args.reserved_gpus_per_host,
        "world_size": len(args.gpu_ids),
        "inactive_reserved_gpus": (args.reserved_gpus_per_host - len(args.gpu_ids)),
        "inactive_reason": (
            "Only explicitly selected origin-shard workers are launched."
        ),
        "child_device": "cuda:0",
        "plan_path": str(plan_path.resolve()),
        "plan_sha256": sha256_file(plan_path),
        "worker_pids": {
            shard_id: process.pid for shard_id, process in processes.items()
        },
        "worker_returncodes": {
            shard_id: process.returncode for shard_id, process in processes.items()
        },
        "expected_bindings": expected_bindings,
        "observed_bindings": observed_bindings,
        "observations": observations,
        "topology_valid": topology_valid,
    }
    save_json_atomic(topology_path, topology)
    validate_topology_document(topology)

    method_result, worker_results = build_method_result(
        plan_path, topology_path, worker_paths
    )
    save_json_atomic(result_path, method_result)
    receipt = make_receipt(
        plan_path=plan_path,
        topology_path=topology_path,
        method_result_path=result_path,
        worker_results=worker_results,
    )
    save_json_atomic(receipt_path, receipt)
    save_json_atomic(
        status_path,
        {
            "status": "completed",
            "method": args.method.upper(),
            "result": file_record(result_path),
            "topology": file_record(topology_path),
            "receipt": file_record(receipt_path),
            "finished_unix": time.time(),
        },
    )
    return method_result


def validate_receipt(path: Path, method_result_path: Path) -> dict:
    receipt = load_json(path)
    receipt_hash = receipt.get("receipt_sha256")
    payload = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if receipt_hash != canonical_json_sha256(payload):
        raise ValueError("Evaluation receipt self-hash drifted")
    if (
        receipt.get("schema_version") != 1
        or receipt.get("job_kind") != "native_ratd_origin_sharded_eval_receipt"
    ):
        raise ValueError("Evaluation receipt schema drifted")
    recorded_result = validate_file_record(
        receipt.get("method_result"), "method result"
    )
    if recorded_result.resolve() != method_result_path.resolve():
        raise ValueError("Evaluation receipt binds a different method result")
    method_result = load_json(method_result_path)
    if receipt.get("method") != method_result.get("method"):
        raise ValueError("Evaluation receipt method identity drifted")
    if (
        receipt.get("checkpoint", {}).get("sha256")
        != method_result.get("checkpoint_sha256")
        or receipt.get("protocol", {}).get("sha256")
        != method_result.get("protocol_sha256")
        or receipt.get("reference_map", {}).get("sha256")
        != method_result.get("reference_map_sha256")
    ):
        raise ValueError("Evaluation receipt and method identity differ")
    for label in (
        "plan",
        "topology",
        "checkpoint",
        "sharded_runner",
        "runner",
        "upstream_main_model",
        "upstream_diff_models",
        "protocol",
        "config",
        "data",
        "reference_map",
        "reference_metadata",
    ):
        validate_file_record(receipt.get(label), label)
    topology_path = Path(receipt["topology"]["path"])
    validate_topology_document(load_json(topology_path))
    worker_results = receipt.get("worker_results")
    if not isinstance(worker_results, list) or not worker_results:
        raise ValueError("Evaluation receipt worker results are missing")
    for worker in worker_results:
        validate_file_record(worker, f'worker {worker.get("shard_id")}')
    return receipt


def load_checkpoint_header(path: Path) -> dict:
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return {
        "epoch": checkpoint.get("epoch"),
        "identity": checkpoint.get("identity"),
    }


def validate_epoch100_checkpoint(
    checkpoint_record: dict,
    method_result: dict,
    expected_method: str,
) -> dict:
    checkpoint_path = validate_file_record(
        checkpoint_record, f"{expected_method} checkpoint"
    )
    header = load_checkpoint_header(checkpoint_path)
    identity = header.get("identity")
    expected_identity = {
        "method": expected_method,
        "source_revision": method_result["source_revision"],
        "upstream_commit": method_result["upstream_commit"],
        "protocol_sha256": method_result["protocol_sha256"],
        "reference_map_sha256": method_result["reference_map_sha256"],
        "seed": FROZEN_SEED,
        "epochs": FROZEN_EPOCHS,
        "diffusion_steps": FROZEN_DIFFUSION_STEPS,
        "samples": FROZEN_SAMPLES,
    }
    if header.get("epoch") != FROZEN_EPOCHS:
        raise ValueError(f"{expected_method} checkpoint is not epoch 100")
    if not isinstance(identity, dict) or any(
        identity.get(key) != value for key, value in expected_identity.items()
    ):
        raise ValueError(f"{expected_method} checkpoint frozen identity drifted")
    return {
        "method": expected_method,
        "epoch": FROZEN_EPOCHS,
        "checkpoint": checkpoint_record,
        "identity": expected_identity,
    }


def validate_training_topology(
    path: Path,
    *,
    exit_policy: str,
    checkpoint_evidence: dict[str, dict],
) -> tuple[dict, dict]:
    topology = load_json(path)
    if (
        topology.get("instance_type") != "ml.p5.48xlarge"
        or topology.get("reserved_gpus_per_host") != 8
        or topology.get("processes_per_host") != 2
        or topology.get("worker_gpu_ids") != [0, 1]
        or topology.get("topology_valid") is not True
    ):
        raise ValueError("Training topology identity or binding gate failed")
    worker_pids = topology.get("worker_pids")
    expected = topology.get("expected_bindings")
    observed = topology.get("observed_bindings")
    if (
        not isinstance(worker_pids, dict)
        or set(worker_pids) != {"ratd", "csdi"}
        or not isinstance(expected, dict)
        or not isinstance(observed, dict)
    ):
        raise ValueError("Training topology evidence is incomplete")
    pids = {int(pid) for pid in worker_pids.values()}
    expected_by_pid = {int(pid): int(gpu) for pid, gpu in expected.items()}
    observed_by_pid = {
        int(pid): [int(gpu) for gpu in gpu_ids] for pid, gpu_ids in observed.items()
    }
    if set(expected_by_pid) != pids or any(
        observed_by_pid.get(pid) != [gpu] for pid, gpu in expected_by_pid.items()
    ):
        raise ValueError("Training PID-to-GPU bindings are invalid")
    if set(checkpoint_evidence) != {"ratd", "csdi"} or any(
        evidence.get("epoch") != FROZEN_EPOCHS
        for evidence in checkpoint_evidence.values()
    ):
        raise ValueError("Both training checkpoints must be epoch 100")
    returncodes = topology.get("worker_returncodes")
    if (
        not isinstance(returncodes, dict)
        or set(returncodes) != {"ratd", "csdi"}
        or any(not isinstance(code, int) for code in returncodes.values())
    ):
        raise ValueError("Training worker returncodes are incomplete")
    nonzero = {method: code for method, code in returncodes.items() if code != 0}
    if not nonzero:
        if exit_policy != "require-zero":
            raise ValueError(
                "Intentional-termination policy requires a nonzero returncode"
            )
        classification = "training-and-original-evaluation-completed"
    else:
        if exit_policy != "intentional-post-epoch100-termination":
            raise ValueError(
                "Nonzero training worker returncodes require explicit "
                "post-epoch100 termination policy"
            )
        allowed_returncodes = {
            -signal.SIGTERM,
            -signal.SIGKILL,
            128 + signal.SIGTERM,
            128 + signal.SIGKILL,
        }
        if any(code not in allowed_returncodes for code in nonzero.values()):
            raise ValueError(
                "Training worker returncode is not an intentional signal exit"
            )
        classification = "original-serial-evaluation-terminated-after-epoch100"
    termination = {
        "exit_policy": exit_policy,
        "classification": classification,
        "worker_returncodes": returncodes,
        "nonzero_returncodes": nonzero,
        "checkpoint_evidence": checkpoint_evidence,
        "accepted": True,
    }
    return topology, termination


def validate_reference_metadata(path: Path) -> dict:
    metadata = load_json(path)
    if (
        metadata.get("seed") != FROZEN_SEED
        or metadata.get("context_length") != FROZEN_CONTEXT_LENGTH
        or metadata.get("prediction_length") != FROZEN_PREDICTION_LENGTH
        or metadata.get("top_k") != 3
        or metadata.get("query_counts")
        != {"train": 17_886, "validation": 2_463, "test": 5_093}
    ):
        raise ValueError("Reference metadata identity drifted")
    reference_map = Path(metadata.get("reference_map", ""))
    if not reference_map.is_file() or sha256_file(reference_map) != metadata.get(
        "reference_map_sha256"
    ):
        raise ValueError("Reference map hash drifted")
    audits = metadata.get("audits")
    if not isinstance(audits, dict) or set(audits) != {
        "train",
        "validation",
        "test",
    }:
        raise ValueError("Reference audits are incomplete")
    train_end = metadata.get("train_end")
    for split, audit in audits.items():
        if audit.get("latest_reference_future_end", train_end + 1) > train_end:
            raise ValueError(f"{split} references cross the training boundary")
    return metadata


def validate_method_result(result: dict, expected_method: str) -> None:
    if (
        result.get("schema_version") != 2
        or result.get("status") != "completed"
        or result.get("method") != expected_method
        or result.get("evaluation_mode") != "origin-sharded-exact-rng"
        or result.get("checkpoint_epoch") != FROZEN_EPOCHS
        or result.get("test_origins") != FROZEN_TEST_ORIGINS
        or result.get("samples") != FROZEN_SAMPLES
        or result.get("diffusion_steps") != FROZEN_DIFFUSION_STEPS
        or set(result.get("metrics", {})) != {"rmse", "mae"}
        or not all(
            math.isfinite(float(value)) for value in result.get("metrics", {}).values()
        )
    ):
        raise ValueError(f"Invalid {expected_method} method result")


def finalize_pair(
    *,
    ratd_result_path: Path,
    ratd_receipt_path: Path,
    csdi_result_path: Path,
    csdi_receipt_path: Path,
    reference_metadata_path: Path,
    training_topology_path: Path,
    topology_output: Path,
    summary_output: Path,
    receipt_output: Path,
    training_eval_exit_policy: str,
) -> dict:
    if any(path.exists() for path in (topology_output, summary_output, receipt_output)):
        raise FileExistsError("Refusing to overwrite finalizer outputs")
    ratd = load_json(ratd_result_path)
    csdi = load_json(csdi_result_path)
    validate_method_result(ratd, "RATD")
    validate_method_result(csdi, "CSDI")
    ratd_receipt = validate_receipt(ratd_receipt_path, ratd_result_path)
    csdi_receipt = validate_receipt(csdi_receipt_path, csdi_result_path)
    metadata = validate_reference_metadata(reference_metadata_path)
    checkpoint_evidence = {
        "ratd": validate_epoch100_checkpoint(ratd_receipt["checkpoint"], ratd, "RATD"),
        "csdi": validate_epoch100_checkpoint(csdi_receipt["checkpoint"], csdi, "CSDI"),
    }
    training_topology, training_termination = validate_training_topology(
        training_topology_path,
        exit_policy=training_eval_exit_policy,
        checkpoint_evidence=checkpoint_evidence,
    )

    shared_fields = (
        "source_revision",
        "launcher_revision",
        "upstream_commit",
        "protocol_sha256",
        "reference_map_sha256",
        "seed",
        "epochs",
        "diffusion_steps",
        "samples",
        "test_origins",
        "test_points",
    )
    for field in shared_fields:
        if ratd.get(field) != csdi.get(field):
            raise ValueError(f"RATD/CSDI {field} identity drifted")
    if ratd["reference_map_sha256"] != metadata["reference_map_sha256"]:
        raise ValueError("Method results use a different reference map")

    evaluation = {}
    for method, receipt in (
        ("ratd", ratd_receipt),
        ("csdi", csdi_receipt),
    ):
        topology_path = Path(receipt["topology"]["path"])
        evaluation[method] = {
            "topology": file_record(topology_path),
            "receipt": file_record(
                ratd_receipt_path if method == "ratd" else csdi_receipt_path
            ),
            "checkpoint": receipt["checkpoint"],
            "plan": receipt["plan"],
            "worker_results": receipt["worker_results"],
        }
    combined_topology = {
        "schema_version": 2,
        "job_kind": "native_ratd_sharded_training_eval_topology",
        "instance_type": "ml.p5.48xlarge",
        "instance_count": 1,
        "reserved_gpus_per_host": 8,
        "training": {
            "topology": file_record(training_topology_path),
            "worker_pids": training_topology["worker_pids"],
            "worker_returncodes": training_topology.get("worker_returncodes", {}),
            "expected_bindings": training_topology["expected_bindings"],
            "observed_bindings": training_topology["observed_bindings"],
            "termination": training_termination,
            "topology_valid": True,
        },
        "evaluation": evaluation,
        "topology_valid": True,
    }
    save_json_atomic(topology_output, combined_topology)

    pair = {
        "cell_id": "ratd/electricity/c96/h168",
        "base": csdi["metrics"],
        "retrieval": ratd["metrics"],
        "delta": {
            metric: ratd["metrics"][metric] - csdi["metrics"][metric]
            for metric in ("rmse", "mae")
        },
    }
    summary = {
        "schema_version": 2,
        "method": "RATD",
        "base_method": "CSDI",
        "expected_pairs": 1,
        "completed_pairs": 1,
        "failed_or_missing_pairs": 0,
        "all_completed": True,
        "protocol_sha256": ratd["protocol_sha256"],
        "source_revision": ratd["source_revision"],
        "launcher_revision": ratd["launcher_revision"],
        "upstream_commit": ratd["upstream_commit"],
        "topology_path": str(topology_output.resolve()),
        "topology_valid": True,
        "reference_metadata": metadata,
        "results": {"ratd": ratd, "csdi": csdi},
        "pair": pair,
        "failures": [],
        "sharded_evaluation": evaluation,
        "receipt_path": str(receipt_output.resolve()),
        "aggregator_compatibility": {
            "metric_and_pair_schema_compatible": True,
            "requires_topology_schema_v2_support": True,
            "minimal_change": (
                "In validate_topology, accept RATD topology job_kind "
                "native_ratd_sharded_training_eval_topology and validate its "
                "training topology plus both hash-bound evaluation topologies "
                "instead of requiring one two-process [0,1] topology."
            ),
        },
    }
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_temporary = summary_output.with_suffix(
        summary_output.suffix + f".tmp.{os.getpid()}"
    )
    summary_temporary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary_stat = summary_temporary.stat()
    finalizer_payload = {
        "schema_version": 1,
        "job_kind": "native_ratd_sharded_pair_finalizer_receipt",
        "created_unix": time.time(),
        "ratd_result": file_record(ratd_result_path),
        "ratd_receipt": file_record(ratd_receipt_path),
        "csdi_result": file_record(csdi_result_path),
        "csdi_receipt": file_record(csdi_receipt_path),
        "reference_metadata": file_record(reference_metadata_path),
        "reference_map": file_record(Path(metadata["reference_map"])),
        "training_topology": file_record(training_topology_path),
        "training_termination": training_termination,
        "combined_topology": file_record(topology_output),
        "summary": {
            "path": str(summary_output.resolve()),
            "size_bytes": summary_stat.st_size,
            "sha256": sha256_file(summary_temporary),
        },
    }
    finalizer_receipt = {
        **finalizer_payload,
        "receipt_sha256": canonical_json_sha256(finalizer_payload),
    }
    save_json_atomic(receipt_output, finalizer_receipt)
    os.replace(summary_temporary, summary_output)
    return summary


def parse_gpu_ids(value: str) -> list[int]:
    try:
        gpu_ids = [int(item) for item in value.split(",") if item != ""]
    except ValueError as error:
        raise argparse.ArgumentTypeError("GPU IDs must be integers") from error
    if not gpu_ids or len(set(gpu_ids)) != len(gpu_ids):
        raise argparse.ArgumentTypeError("GPU IDs must be a non-empty unique list")
    if any(gpu_id < 0 for gpu_id in gpu_ids):
        raise argparse.ArgumentTypeError("GPU IDs must be non-negative")
    return gpu_ids


def add_identity_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--method", choices=("ratd", "csdi"), required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--reference-map", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--launcher-revision", required=True)
    parser.add_argument("--upstream-commit", required=True)


def add_run_arguments(parser: argparse.ArgumentParser) -> None:
    add_identity_arguments(parser)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gpu-ids", type=parse_gpu_ids, required=True)
    parser.add_argument("--planner-gpu", type=int, required=True)
    parser.add_argument("--instance-type", default="ml.p5.48xlarge")
    parser.add_argument("--reserved-gpus-per-host", type=int, default=8)
    parser.add_argument("--poll-seconds", type=float, default=30.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run")
    add_run_arguments(run_parser)

    worker_parser = subparsers.add_parser("worker")
    worker_parser.add_argument("--plan", type=Path, required=True)
    worker_parser.add_argument("--shard-id", required=True)
    worker_parser.add_argument("--output", type=Path, required=True)

    verify_parser = subparsers.add_parser("verify")
    add_identity_arguments(verify_parser)
    verify_parser.add_argument("--origin-start", type=int, required=True)
    verify_parser.add_argument("--oracle-gpu", type=int, required=True)
    verify_parser.add_argument("--comparison-gpu", type=int, required=True)
    verify_parser.add_argument("--output", type=Path, required=True)
    verify_parser.add_argument("--instance-type", default="ml.p5.48xlarge")
    verify_parser.add_argument("--reserved-gpus-per-host", type=int, default=8)
    verify_parser.add_argument("--poll-seconds", type=float, default=5.0)

    verify_worker = subparsers.add_parser("verify-worker")
    verify_worker.add_argument("--plan", type=Path, required=True)
    verify_worker.add_argument(
        "--role", choices=("oracle", "comparison"), required=True
    )
    verify_worker.add_argument("--output", type=Path, required=True)

    finalizer = subparsers.add_parser("finalize")
    finalizer.add_argument("--ratd-result", type=Path, required=True)
    finalizer.add_argument("--ratd-receipt", type=Path, required=True)
    finalizer.add_argument("--csdi-result", type=Path, required=True)
    finalizer.add_argument("--csdi-receipt", type=Path, required=True)
    finalizer.add_argument("--reference-metadata", type=Path, required=True)
    finalizer.add_argument("--training-topology", type=Path, required=True)
    finalizer.add_argument("--topology-output", type=Path, required=True)
    finalizer.add_argument("--output", type=Path, required=True)
    finalizer.add_argument("--receipt-output", type=Path, required=True)
    finalizer.add_argument(
        "--training-eval-exit-policy",
        choices=(
            "require-zero",
            "intentional-post-epoch100-termination",
        ),
        default="require-zero",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "run":
        try:
            run_orchestrator(args)
        except BaseException as error:
            if not isinstance(error, FileExistsError):
                method_root = args.output_root / args.method.lower()
                save_json_atomic(
                    method_root / "status.json",
                    {
                        "status": "failed",
                        "method": args.method.upper(),
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "finished_unix": time.time(),
                    },
                )
            raise
        return 0
    if args.command == "worker":
        try:
            run_worker(args.plan, args.shard_id, args.output)
        except BaseException as error:
            if not isinstance(error, FileExistsError):
                save_json_atomic(
                    args.output.with_suffix(".failed.json"),
                    {
                        "status": "failed",
                        "shard_id": args.shard_id,
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "finished_unix": time.time(),
                    },
                )
            raise
        return 0
    if args.command == "verify":
        try:
            run_verification(args)
        except BaseException as error:
            if not isinstance(error, FileExistsError):
                save_json_atomic(
                    args.output.with_suffix(".failed.json"),
                    {
                        "status": "failed",
                        "method": args.method.upper(),
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "finished_unix": time.time(),
                    },
                )
            raise
        return 0
    if args.command == "verify-worker":
        try:
            run_verification_worker(args.plan, args.role, args.output)
        except BaseException as error:
            if not isinstance(error, FileExistsError):
                save_json_atomic(
                    args.output.with_suffix(".failed.json"),
                    {
                        "status": "failed",
                        "role": args.role,
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "finished_unix": time.time(),
                    },
                )
            raise
        return 0
    finalize_pair(
        ratd_result_path=args.ratd_result,
        ratd_receipt_path=args.ratd_receipt,
        csdi_result_path=args.csdi_result,
        csdi_receipt_path=args.csdi_receipt,
        reference_metadata_path=args.reference_metadata,
        training_topology_path=args.training_topology,
        topology_output=args.topology_output,
        summary_output=args.output,
        receipt_output=args.receipt_output,
        training_eval_exit_policy=args.training_eval_exit_policy,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
