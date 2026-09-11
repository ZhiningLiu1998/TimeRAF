"""Build every prediction bundle required by the Appendix RAG matrix."""

import argparse
import csv
import fcntl
import io
import json
import os
import shlex
import shutil
import socket
import subprocess
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch

from ts_rag.appendix_autogluon import (
    AUTOGLOUON_TIMESERIES_VERSION,
    autogluon_checkpoint_compatibility,
    require_autogluon_timeseries_version,
)
from ts_rag.appendix_ensembles import build_advanced_ensemble_bundle
from ts_rag.appendix_rag import load_appendix_manifest, sha256_file
from ts_rag.appendix_zeroshot import build_zeroshot_ensemble_bundles
from ts_rag.benchmark import MODELS
from ts_rag.matrix import load_manifest, resolve_source_revision, save_json_atomic


STATIC_BASELINES = (
    "forward_selection",
    "portfolio_ensemble",
    "zeroshot_ensemble",
)
EXTERNAL_BASELINES = (
    "chronos_bolt_finetuned",
    "chronos_bolt_zeroshot",
    "autogluon_high_quality",
)
P4DE_INSTANCE_TYPE = "ml.p4de.24xlarge"
P4DE_GPUS_PER_HOST = 8
EXPECTED_REPLAY_CELLS = 208
EXPECTED_REPLAY_MANIFEST_SHA256 = (
    "447391d99c46bc1dd4170e71a8388bad5edb0a4c48a3048e230e404b96a0a77a"
)
EXPECTED_APPENDIX_ROWS = 96
EXPECTED_APPENDIX_EVALUABLE = 94


def preflight_external_runtime():
    installed = require_autogluon_timeseries_version()
    return {
        "autogluon_timeseries": {
            "installed_version": installed,
            "required_version": AUTOGLOUON_TIMESERIES_VERSION,
            "exact_version_match": True,
        }
    }


def _load_json(path):
    with Path(path).open("r", encoding="utf-8") as source:
        return json.load(source)


def _manifest_sha256(path):
    return sha256_file(path)


def _acquire_lock(output_root):
    lock_path = Path(output_root) / "appendix_bundle_pipeline.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError(
            f"Another Appendix bundle pipeline holds {lock_path}"
        ) from error
    lock.seek(0)
    lock.truncate()
    lock.write(f"pid={os.getpid()} host={socket.gethostname()}\n")
    lock.flush()
    return lock


def _expected_replay_keys(appendix_rows):
    targets = {
        (
            row["task_family"],
            row["dataset"],
            int(row["pred_len"]),
        )
        for row in appendix_rows
    }
    return {
        (*target, model)
        for target in targets
        for model in MODELS
    }


def _replay_key(row):
    return (
        row["task_family"],
        row["dataset"],
        int(row["pred_len"]),
        row["model"],
    )


def _resolve_catalog_path(record, project_root):
    path = Path(record["path"])
    if path.is_absolute():
        return path
    return Path(project_root) / path


def load_replay_libraries(
    replay_manifest,
    replay_catalog,
    appendix_manifest,
    project_root,
    verify_hashes=True,
):
    """Validate the complete replay library and index it by target dataset."""
    replay_rows = load_manifest(replay_manifest)
    appendix_rows = load_appendix_manifest(appendix_manifest)
    replay_manifest_sha256 = sha256_file(replay_manifest)
    if replay_manifest_sha256 != EXPECTED_REPLAY_MANIFEST_SHA256:
        raise ValueError(
            "Replay manifest SHA-256 does not match the frozen 208-cell "
            "manifest"
        )
    if len(replay_rows) != EXPECTED_REPLAY_CELLS:
        raise ValueError(
            f"Replay manifest must contain {EXPECTED_REPLAY_CELLS} cells"
        )
    if (
        len(appendix_rows) != EXPECTED_APPENDIX_ROWS
        or len({row["id"] for row in appendix_rows})
        != EXPECTED_APPENDIX_ROWS
    ):
        raise ValueError("Appendix manifest must contain 96 unique rows")
    evaluable_count = sum(
        row["paper_status"]["evaluable"] for row in appendix_rows
    )
    if evaluable_count != EXPECTED_APPENDIX_EVALUABLE:
        raise ValueError("Appendix manifest must contain 94 evaluable rows")
    expected_keys = _expected_replay_keys(appendix_rows)
    rows_by_key = {_replay_key(row): row for row in replay_rows}
    if len(rows_by_key) != len(replay_rows):
        raise ValueError("Replay manifest contains duplicate logical cells")
    missing_keys = expected_keys - set(rows_by_key)
    extra_keys = set(rows_by_key) - expected_keys
    if missing_keys or extra_keys:
        raise ValueError(
            "Replay manifest does not exactly cover Appendix base libraries: "
            f"missing={len(missing_keys)}, extra={len(extra_keys)}"
        )

    catalog = _load_json(replay_catalog)
    records = catalog.get("bundles")
    if not isinstance(records, dict):
        raise ValueError("Replay catalog requires a bundles object")
    expected_ids = {row["id"] for row in replay_rows}
    if set(records) != expected_ids:
        raise ValueError(
            "Replay catalog must contain exactly the 208 replay cells: "
            f"missing={len(expected_ids - set(records))}, "
            f"extra={len(set(records) - expected_ids)}"
        )

    libraries = {}
    verified = {}
    for key, row in sorted(rows_by_key.items()):
        record = records[row["id"]]
        if record.get("usable") is not True:
            raise ValueError(f"Replay bundle is not usable: {row['id']}")
        if not record.get("path"):
            raise ValueError(f"Replay bundle has no path: {row['id']}")
        path = _resolve_catalog_path(record, project_root)
        if not path.is_file():
            raise FileNotFoundError(f"Replay bundle is missing: {path}")
        size_bytes = path.stat().st_size
        if record.get("size_bytes") != size_bytes:
            raise ValueError(f"Replay bundle size mismatch: {row['id']}")
        recorded_hash = record.get("sha256")
        if not recorded_hash:
            raise ValueError(f"Replay bundle has no SHA-256: {row['id']}")
        if verify_hashes:
            actual_hash = sha256_file(path)
            if actual_hash != recorded_hash:
                raise ValueError(f"Replay bundle hash mismatch: {row['id']}")
        target = key[:3]
        libraries.setdefault(target, {})[row["model"]] = path
        verified[row["id"]] = {
            "path": str(path),
            "sha256": recorded_hash,
            "size_bytes": size_bytes,
        }

    for target, library in libraries.items():
        if set(library) != set(MODELS):
            raise ValueError(f"Incomplete base-model library for {target}")
    return {
        "appendix_rows": appendix_rows,
        "replay_rows": replay_rows,
        "replay_catalog": catalog,
        "replay_catalog_sha256": sha256_file(replay_catalog),
        "replay_manifest_sha256": replay_manifest_sha256,
        "libraries": libraries,
        "verified_bundles": verified,
    }


def _bundle_path(output_root, cell):
    return (
        Path(output_root)
        / "bundles"
        / cell["baseline"]
        / cell["task_family"]
        / cell["dataset"]
        / f"pl{cell['pred_len']}"
        / "prediction_bundle.npz"
    )


def _metadata_path(bundle_path):
    return Path(f"{bundle_path}.json")


def _status_path(bundle_path):
    return Path(f"{bundle_path}.status.json")


def _catalog_path(path, project_root):
    path = Path(path).resolve()
    try:
        return str(path.relative_to(Path(project_root).resolve()))
    except ValueError:
        return str(path)


def _validate_output_metadata(
    cell,
    bundle_path,
    source_revision,
    replay_catalog_sha256=None,
):
    metadata_path = _metadata_path(bundle_path)
    if not bundle_path.exists() and not metadata_path.exists():
        return None
    if not bundle_path.is_file() or not metadata_path.is_file():
        raise ValueError(f"Incomplete existing output for {cell['id']}")
    metadata = _load_json(metadata_path)
    identity = metadata.get("run_identity", {})
    if identity.get("source_revision") != source_revision:
        raise ValueError(f"Source revision mismatch for {cell['id']}")
    if (
        replay_catalog_sha256 is not None
        and identity.get("replay_catalog_sha256")
        != replay_catalog_sha256
    ):
        raise ValueError(f"Replay catalog mismatch for {cell['id']}")
    recorded_hash = metadata.get("prediction_bundle_sha256")
    if not recorded_hash or sha256_file(bundle_path) != recorded_hash:
        raise ValueError(f"Generated bundle hash mismatch for {cell['id']}")
    return metadata


def _static_cells(appendix_rows, baseline):
    return sorted(
        (row for row in appendix_rows if row["baseline"] == baseline),
        key=lambda row: row["id"],
    )


def build_static_bundles(
    context,
    output_root,
    source_revision,
    forward_ensemble_size=50,
):
    """Build or verify all 48 advanced-ensemble Appendix bundles."""
    rows = context["appendix_rows"]
    catalog_hash = context["replay_catalog_sha256"]
    outputs = {}
    for baseline in ("forward_selection", "portfolio_ensemble"):
        for cell in _static_cells(rows, baseline):
            bundle_path = _bundle_path(output_root, cell)
            metadata = _validate_output_metadata(
                cell,
                bundle_path,
                source_revision,
                replay_catalog_sha256=catalog_hash,
            )
            if metadata is None:
                target = (
                    cell["task_family"],
                    cell["dataset"],
                    int(cell["pred_len"]),
                )
                metadata = build_advanced_ensemble_bundle(
                    context["libraries"][target],
                    cell["task_family"],
                    baseline,
                    bundle_path,
                    forward_ensemble_size=forward_ensemble_size,
                )
                metadata.update(
                    {
                        "cell_id": cell["id"],
                        "run_identity": {
                            "source_revision": source_revision,
                            "replay_catalog_sha256": catalog_hash,
                            "replay_manifest_sha256": context[
                                "replay_manifest_sha256"
                            ],
                        },
                    }
                )
                save_json_atomic(metadata, _metadata_path(bundle_path))
            outputs[cell["id"]] = bundle_path

    for family in sorted({row["task_family"] for row in rows}):
        cells = [
            row
            for row in _static_cells(rows, "zeroshot_ensemble")
            if row["task_family"] == family
        ]
        existing = {}
        for cell in cells:
            bundle_path = _bundle_path(output_root, cell)
            metadata = _validate_output_metadata(
                cell,
                bundle_path,
                source_revision,
                replay_catalog_sha256=catalog_hash,
            )
            if metadata is not None:
                existing[cell["id"]] = bundle_path
        if existing and len(existing) != len(cells):
            raise ValueError(
                f"Partial ZeroShot output set already exists for {family}"
            )
        if len(existing) == len(cells):
            outputs.update(existing)
            continue

        stage_root = (
            Path(output_root)
            / ".staging"
            / f"zeroshot-{family}-{os.getpid()}-{time.time_ns()}"
        )
        task_bundles = {
            cell["dataset"]: context["libraries"][
                (
                    cell["task_family"],
                    cell["dataset"],
                    int(cell["pred_len"]),
                )
            ]
            for cell in cells
        }
        try:
            metadata = build_zeroshot_ensemble_bundles(
                task_bundles,
                family,
                stage_root,
            )
            for cell in cells:
                staged = (
                    stage_root
                    / cell["dataset"]
                    / "prediction_bundle.npz"
                )
                output = _bundle_path(output_root, cell)
                output.parent.mkdir(parents=True, exist_ok=True)
                if output.exists():
                    raise FileExistsError(
                        f"Refusing to replace ZeroShot bundle: {output}"
                    )
                os.replace(staged, output)
                output_metadata = {
                    **metadata["outputs"][cell["dataset"]],
                    "cell_id": cell["id"],
                    "task_family": family,
                    "protocol": metadata["protocol"],
                    "run_identity": {
                        "source_revision": source_revision,
                        "replay_catalog_sha256": catalog_hash,
                        "replay_manifest_sha256": context[
                            "replay_manifest_sha256"
                        ],
                    },
                }
                output_metadata["prediction_bundle"] = str(output)
                save_json_atomic(
                    output_metadata,
                    _metadata_path(output),
                )
                outputs[cell["id"]] = output
        finally:
            if stage_root.exists():
                shutil.rmtree(stage_root)

    expected = sum(
        row["baseline"] in STATIC_BASELINES for row in rows
    )
    if len(outputs) != expected:
        raise ValueError(
            f"Expected {expected} static bundles, found {len(outputs)}"
        )
    return outputs


def _parse_gpu_ids(value):
    try:
        gpu_ids = [int(item) for item in value.split(",") if item != ""]
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "--gpu-ids must be comma-separated integers"
        ) from error
    if not gpu_ids or len(gpu_ids) != len(set(gpu_ids)):
        raise argparse.ArgumentTypeError(
            "--gpu-ids must contain distinct GPU IDs"
        )
    return gpu_ids


def build_external_topology(
    instance_type,
    reserved_gpus_per_host,
    processes_per_host,
    gpu_ids,
    queued_chronos_cells,
    queued_evaluable_cells,
    dry_run=False,
):
    visible_gpus = torch.cuda.device_count()
    if not torch.cuda.is_available() or visible_gpus < 1:
        raise RuntimeError("Appendix external exports require CUDA")
    if not 1 <= processes_per_host <= reserved_gpus_per_host:
        raise ValueError(
            "processes_per_host must be within the reserved GPU count"
        )
    if len(gpu_ids) != processes_per_host:
        raise ValueError("One distinct GPU ID is required per worker process")
    if any(gpu_id < 0 or gpu_id >= visible_gpus for gpu_id in gpu_ids):
        raise ValueError("Worker GPU IDs must identify visible CUDA devices")
    if visible_gpus > reserved_gpus_per_host:
        raise ValueError(
            "Visible CUDA devices exceed the reserved GPU count"
        )
    if instance_type == P4DE_INSTANCE_TYPE:
        if (
            reserved_gpus_per_host != P4DE_GPUS_PER_HOST
            or processes_per_host != P4DE_GPUS_PER_HOST
            or visible_gpus != P4DE_GPUS_PER_HOST
            or gpu_ids != list(range(P4DE_GPUS_PER_HOST))
        ):
            raise ValueError(
                "ml.p4de.24xlarge requires eight visible, reserved, and "
                "active worker GPUs numbered zero through seven"
            )
        if (
            not dry_run
            and queued_evaluable_cells > 0
            and queued_chronos_cells < P4DE_GPUS_PER_HOST
        ):
            raise ValueError(
                "ml.p4de.24xlarge requires at least eight queued Chronos "
                "cells at launch"
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
        "instance_type": instance_type,
        "instance_count": 1,
        "reserved_gpus_per_host": reserved_gpus_per_host,
        "visible_cuda_devices": visible_gpus,
        "processes_per_host": processes_per_host,
        "worker_gpu_ids": gpu_ids,
        "total_gpus": reserved_gpus_per_host,
        "world_size": processes_per_host,
        "inactive_reserved_gpus": (
            reserved_gpus_per_host - processes_per_host
        ),
        "launcher_mode": "independent_appendix_export_workers",
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "nvidia_smi_list": gpu_listing,
    }


def _external_complete(cell, output_root, source_revision):
    bundle_path = _bundle_path(output_root, cell)
    metadata = _validate_output_metadata(
        cell,
        bundle_path,
        source_revision,
    )
    if metadata is None:
        return False
    if metadata.get("evaluation_status") != "exported":
        raise ValueError(f"External export is not complete: {cell['id']}")
    _validate_external_runtime_metadata(cell, metadata)
    return True


def _validate_external_runtime_metadata(cell, metadata):
    if cell["baseline"] not in EXTERNAL_BASELINES:
        return None
    identity = metadata.get("run_identity")
    if not isinstance(identity, dict):
        raise ValueError(
            f"External run identity is invalid for {cell['id']}"
        )
    if (
        identity.get("autogluon_timeseries_version")
        != AUTOGLOUON_TIMESERIES_VERSION
    ):
        raise ValueError(
            "External run identity AutoGluon version mismatch for "
            f"{cell['id']}"
        )
    predictor = metadata.get("predictor")
    if not isinstance(predictor, dict):
        raise ValueError(
            f"External predictor metadata is invalid for {cell['id']}"
        )
    if (
        predictor.get("autogluon_timeseries_version")
        != AUTOGLOUON_TIMESERIES_VERSION
    ):
        raise ValueError(
            "External predictor AutoGluon version mismatch for "
            f"{cell['id']}"
        )
    if cell["baseline"] == "autogluon_high_quality":
        expected_compatibility = autogluon_checkpoint_compatibility(
            cell["baseline"]
        )
        if (
            identity.get("checkpoint_compatibility")
            != expected_compatibility
            or predictor.get("checkpoint_compatibility")
            != expected_compatibility
            or metadata.get("checkpoint_compatibility")
            != expected_compatibility
        ):
            raise ValueError(
                "External predictor checkpoint compatibility mismatch for "
                f"{cell['id']}"
            )
    return AUTOGLOUON_TIMESERIES_VERSION


def _mark_paper_oot(
    cell,
    output_root,
    source_revision,
    launcher_revision,
    autogluon_timeseries_version,
):
    bundle_path = _bundle_path(output_root, cell)
    metadata_path = _metadata_path(bundle_path)
    status_path = _status_path(bundle_path)
    identity = {
        "source_revision": source_revision,
        "launcher_revision": launcher_revision,
        "physical_gpu_id": None,
        "cuda_visible_devices": None,
        "autogluon_timeseries_version": autogluon_timeseries_version,
    }
    payload = {
        "cell_id": cell["id"],
        "evaluation_status": "paper_oot",
        "paper_status": cell["paper_status"],
        "run_identity": identity,
    }
    save_json_atomic(payload, metadata_path)
    save_json_atomic(
        {
            "cell_id": cell["id"],
            "status": "paper_oot",
            "run_identity": identity,
        },
        status_path,
    )


def _worker_environment(gpu_id, cache_root):
    if gpu_id < 0:
        raise ValueError("GPU workers require a non-negative physical GPU ID")
    cache_root = Path(cache_root)
    paths = {
        "XDG_CACHE_HOME": cache_root / "xdg",
        "HF_HOME": cache_root / "huggingface",
        "HF_HUB_CACHE": cache_root / "huggingface" / "hub",
        "TRANSFORMERS_CACHE": cache_root / "huggingface" / "transformers",
        "TORCH_HOME": cache_root / "torch",
        "MPLCONFIGDIR": cache_root / "matplotlib",
        "JOBLIB_TEMP_FOLDER": cache_root / "joblib",
        "RAY_TMPDIR": cache_root / "ray",
        "TMPDIR": cache_root / "tmp",
        "TEMP": cache_root / "tmp",
        "TMP": cache_root / "tmp",
    }
    for path in set(paths.values()):
        path.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment["PYTHONUNBUFFERED"] = "1"
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    environment.update({key: str(path) for key, path in paths.items()})
    return environment


def _parse_csv_rows(payload, expected_columns):
    rows = []
    for row in csv.reader(io.StringIO(payload.strip())):
        if not row:
            continue
        if len(row) != expected_columns:
            raise ValueError(
                f"Expected {expected_columns} GPU columns, found {len(row)}"
            )
        rows.append([item.strip() for item in row])
    return rows


def _cuda_visible_devices(pid):
    try:
        values = (Path("/proc") / str(pid) / "environ").read_bytes().split(
            b"\0"
        )
    except (OSError, PermissionError):
        return None
    for value in values:
        key, separator, raw = value.partition(b"=")
        if separator and key == b"CUDA_VISIBLE_DEVICES":
            return raw.decode(errors="replace")
    return None


def _capture_gpu_sample():
    gpu_payload = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,utilization.gpu,memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout
    try:
        process_payload = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout
    except subprocess.CalledProcessError:
        process_payload = ""
    gpus = []
    by_uuid = {}
    for row in _parse_csv_rows(gpu_payload, 5):
        gpu = {
            "index": int(row[0]),
            "uuid": row[1],
            "name": row[2],
            "utilization_gpu_percent": int(row[3]),
            "memory_used_mib": int(row[4]),
            "processes": [],
        }
        gpus.append(gpu)
        by_uuid[gpu["uuid"]] = gpu
    for row in _parse_csv_rows(process_payload, 4):
        pid = int(row[1])
        process = {
            "gpu_uuid": row[0],
            "pid": pid,
            "process_name": row[2],
            "memory_used_mib": int(row[3]),
            "cuda_visible_devices": _cuda_visible_devices(pid),
        }
        gpu = by_uuid.get(process["gpu_uuid"])
        if gpu is not None:
            gpu["processes"].append(process)
    return {
        "recorded_unix": time.time(),
        "hostname": socket.gethostname(),
        "gpus": gpus,
    }


def _summarize_gpu_samples(samples, expected_gpu_ids):
    expected = set(expected_gpu_ids)
    maximum_utilization = {index: 0 for index in expected}
    binding_evidence = None
    for sample in samples:
        gpus = {gpu["index"]: gpu for gpu in sample.get("gpus", [])}
        for index in expected & set(gpus):
            maximum_utilization[index] = max(
                maximum_utilization[index],
                int(gpus[index]["utilization_gpu_percent"]),
            )
        if not expected.issubset(gpus):
            continue
        pids = []
        for index in sorted(expected):
            matching = [
                process
                for process in gpus[index]["processes"]
                if process.get("cuda_visible_devices") == str(index)
            ]
            if not matching:
                break
            pids.append(matching[0]["pid"])
        else:
            if len(set(pids)) == len(expected):
                binding_evidence = {
                    "recorded_unix": sample["recorded_unix"],
                    "pids": pids,
                    "gpu_ids": sorted(expected),
                }
    utilization_verified = all(
        value > 0 for value in maximum_utilization.values()
    )
    return {
        "sample_count": len(samples),
        "expected_gpu_ids": sorted(expected),
        "distinct_bindings_observed": binding_evidence is not None,
        "binding_evidence": binding_evidence,
        "max_utilization_gpu_percent": maximum_utilization,
        "all_expected_gpus_utilized": utilization_verified,
        "topology_gate_passed": (
            binding_evidence is not None and utilization_verified
        ),
    }


class _GpuSampler:
    def __init__(self, path, expected_gpu_ids, interval_seconds):
        self.path = Path(path)
        self.expected_gpu_ids = list(expected_gpu_ids)
        self.interval_seconds = interval_seconds
        self.samples = []
        self.errors = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=self.interval_seconds + 30)

    def _run(self):
        while not self._stop.is_set():
            try:
                sample = _capture_gpu_sample()
                self.samples.append(sample)
                with self.path.open("a", encoding="utf-8") as output:
                    output.write(json.dumps(sample, sort_keys=True) + "\n")
            except Exception as error:
                self.errors.append(
                    {
                        "recorded_unix": time.time(),
                        "type": type(error).__name__,
                        "message": str(error),
                    }
                )
            self._stop.wait(self.interval_seconds)

    def summary(self):
        return {
            **_summarize_gpu_samples(
                self.samples,
                self.expected_gpu_ids,
            ),
            "sampler_errors": self.errors,
        }


def _run_external_cell(cell, args, gpu_id):
    bundle_path = _bundle_path(args.output_root, cell)
    metadata_path = _metadata_path(bundle_path)
    status_path = _status_path(bundle_path)
    predictor_path = (
        Path(args.output_root)
        / "predictors"
        / cell["baseline"]
        / cell["task_family"]
        / cell["dataset"]
        / f"pl{cell['pred_len']}"
        / args.source_revision[:12]
    )
    staging_root = (
        Path(args.output_root)
        / ".staging"
        / "external"
        / cell["id"].replace("/", "__")
        / f"{os.getpid()}-{time.time_ns()}"
    )
    command = [
        sys.executable,
        "scripts/export_appendix_autogluon_bundle.py",
        "--manifest",
        args.appendix_manifest,
        "--cell-id",
        cell["id"],
        "--output",
        str(bundle_path),
        "--predictor-path",
        str(predictor_path),
        "--metadata",
        str(metadata_path),
        "--status",
        str(status_path),
        "--staging-root",
        str(staging_root),
        "--seed",
        str(args.seed),
        "--batch-windows",
        str(args.batch_windows),
        "--max-prediction-items",
        str(args.max_prediction_items),
        "--fine-tune-steps",
        str(args.fine_tune_steps),
        "--inference-batch-size",
        str(args.inference_batch_size),
        "--fine-tune-batch-size",
        str(args.fine_tune_batch_size),
        "--source-revision",
        args.source_revision,
        "--launcher-revision",
        args.launcher_revision,
        "--physical-gpu-id",
        str(gpu_id),
    ]
    if args.time_limit is not None:
        command.extend(["--time-limit", str(args.time_limit)])
    data_root = args.data_roots.get(cell["task_family"])
    if data_root is not None:
        command.extend(["--data-root", str(data_root)])

    environment = _worker_environment(gpu_id, args.cache_root)
    log_path = (
        Path(args.output_root)
        / "logs"
        / "external"
        / f"{cell['id'].replace('/', '__')}.log"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        header = (
            f"\n[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] "
            f"worker_gpu={gpu_id} local_gpu=0 "
            f"CUDA_VISIBLE_DEVICES={environment['CUDA_VISIBLE_DEVICES']} "
            f"cache_root={args.cache_root} "
            f"{shlex.join(command)}\n"
        )
        print(header, end="")
        log.write(header)
        process = subprocess.Popen(
            command,
            cwd=Path(__file__).resolve().parent.parent,
            env=environment,
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


def _external_cells(rows):
    priority = {
        "chronos_bolt_finetuned": 0,
        "chronos_bolt_zeroshot": 1,
        "autogluon_high_quality": 2,
    }
    return sorted(
        (row for row in rows if row["baseline"] in EXTERNAL_BASELINES),
        key=lambda row: (priority[row["baseline"]], row["id"]),
    )


def _load_completed_external_evidence(args):
    metadata_path = Path(args.output_root) / "external_run_metadata.json"
    evidence_path = (
        Path(args.output_root) / "gpu_evidence" / "summary.json"
    )
    try:
        metadata = _load_json(metadata_path)
        evidence = _load_json(evidence_path)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            "Completed external exports require preserved run metadata "
            "and GPU evidence"
        ) from error
    topology = metadata.get("topology")
    if (
        metadata.get("source_revision") != args.source_revision
        or not isinstance(topology, dict)
        or topology.get("worker_gpu_ids") != list(args.gpu_ids)
        or metadata.get("queued_evaluable_cells", 0) < 1
    ):
        raise ValueError(
            "Preserved external run metadata does not match this run"
        )
    if (
        evidence.get("expected_gpu_ids") != list(args.gpu_ids)
        or evidence.get("sample_count", 0) < 1
        or evidence.get("topology_gate_passed") is not True
    ):
        raise ValueError(
            "Preserved external GPU evidence did not pass the topology gate"
        )
    return {
        "external_run_metadata": str(metadata_path),
        "external_run_metadata_sha256": sha256_file(metadata_path),
        "gpu_evidence": str(evidence_path),
        "gpu_evidence_sha256": sha256_file(evidence_path),
        "topology_gate_passed": True,
    }


def run_external_exports(context, args):
    external_runtime = preflight_external_runtime()
    autogluon_timeseries_version = external_runtime[
        "autogluon_timeseries"
    ]["installed_version"]
    rows = _external_cells(context["appendix_rows"])
    if not args.dry_run:
        for cell in rows:
            if not cell["paper_status"]["evaluable"]:
                _mark_paper_oot(
                    cell,
                    args.output_root,
                    args.source_revision,
                    args.launcher_revision,
                    autogluon_timeseries_version,
                )
    pending = [
        cell
        for cell in rows
        if cell["paper_status"]["evaluable"]
        and not _external_complete(
            cell,
            args.output_root,
            args.source_revision,
        )
    ]
    if args.max_external_cells:
        pending = pending[: args.max_external_cells]
    queued_chronos = sum(
        cell["baseline"].startswith("chronos_") for cell in pending
    )
    topology = build_external_topology(
        args.instance_type,
        args.reserved_gpus_per_host,
        args.processes_per_host,
        args.gpu_ids,
        queued_chronos,
        len(pending),
        dry_run=args.dry_run,
    )
    run_metadata = {
        "topology": topology,
        "appendix_manifest": args.appendix_manifest,
        "appendix_manifest_sha256": _manifest_sha256(
            args.appendix_manifest
        ),
        "selected_external_cells": len(rows),
        "queued_evaluable_cells": len(pending),
        "queued_chronos_cells": queued_chronos,
        "source_revision": args.source_revision,
        "launcher_revision": args.launcher_revision,
        "seed": args.seed,
        "external_runtime": external_runtime,
        "cache_root": str(args.cache_root),
    }
    if args.dry_run:
        save_json_atomic(
            run_metadata,
            Path(args.output_root) / "external_run_metadata.json",
        )
        for cell in pending:
            print(cell["id"])
        return 0
    if not pending:
        preserved = _load_completed_external_evidence(args)
        print(
            json.dumps(
                {
                    "attempted_this_run": 0,
                    "failures_this_run": 0,
                    "pending_before_run": 0,
                    "gpu_topology_gate_passed": True,
                    "preserved_evidence": preserved,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    save_json_atomic(
        run_metadata,
        Path(args.output_root) / "external_run_metadata.json",
    )

    failures = 0
    attempted = 0
    sampler = _GpuSampler(
        Path(args.output_root) / "gpu_evidence" / "samples.jsonl",
        args.gpu_ids,
        args.gpu_sample_seconds,
    )
    sampler.start()
    available_gpus = list(args.gpu_ids)
    remaining = iter(pending)
    active = {}
    exhausted = False
    stop_scheduling = False
    try:
        with ThreadPoolExecutor(
            max_workers=args.processes_per_host
        ) as executor:
            while active or not exhausted:
                while (
                    available_gpus
                    and not exhausted
                    and not stop_scheduling
                ):
                    try:
                        cell = next(remaining)
                    except StopIteration:
                        exhausted = True
                        break
                    gpu_id = available_gpus.pop(0)
                    future = executor.submit(
                        _run_external_cell,
                        cell,
                        args,
                        gpu_id,
                    )
                    active[future] = (cell, gpu_id)
                    attempted += 1
                if not active:
                    break
                done, _ = wait(active, return_when=FIRST_COMPLETED)
                for future in done:
                    cell, gpu_id = active.pop(future)
                    available_gpus.append(gpu_id)
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
                    if args.max_failures and failures >= args.max_failures:
                        stop_scheduling = True
    finally:
        sampler.stop()
    gpu_evidence = sampler.summary()
    save_json_atomic(
        gpu_evidence,
        Path(args.output_root) / "gpu_evidence" / "summary.json",
    )
    if (
        args.instance_type == P4DE_INSTANCE_TYPE
        and pending
        and not gpu_evidence["topology_gate_passed"]
    ):
        failures += 1
        print(
            "p4de topology gate failed: eight distinct bindings and "
            "nonzero utilization on every A100 were not observed"
        )
    print(
        json.dumps(
            {
                "attempted_this_run": attempted,
                "failures_this_run": failures,
                "pending_before_run": len(pending),
                "gpu_topology_gate_passed": gpu_evidence[
                    "topology_gate_passed"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return failures


def build_complete_catalog(
    appendix_rows,
    appendix_manifest,
    replay_catalog,
    output_root,
    project_root,
    source_revision,
):
    if len(appendix_rows) != EXPECTED_APPENDIX_ROWS:
        raise ValueError(
            f"Appendix manifest must contain {EXPECTED_APPENDIX_ROWS} rows"
        )
    evaluable = [
        row for row in appendix_rows if row["paper_status"]["evaluable"]
    ]
    if len(evaluable) != EXPECTED_APPENDIX_EVALUABLE:
        raise ValueError(
            "Appendix manifest must contain "
            f"{EXPECTED_APPENDIX_EVALUABLE} evaluable rows"
        )
    bundles = {}
    for cell in sorted(evaluable, key=lambda row: row["id"]):
        bundle_path = _bundle_path(output_root, cell)
        replay_hash = (
            sha256_file(replay_catalog)
            if cell["baseline"] in STATIC_BASELINES
            else None
        )
        metadata = _validate_output_metadata(
            cell,
            bundle_path,
            source_revision,
            replay_catalog_sha256=replay_hash,
        )
        if metadata is None:
            raise ValueError(f"Missing Appendix bundle: {cell['id']}")
        autogluon_version = _validate_external_runtime_metadata(
            cell,
            metadata,
        )
        metadata_path = _metadata_path(bundle_path)
        bundles[cell["id"]] = {
            "path": _catalog_path(bundle_path, project_root),
            "sha256": metadata["prediction_bundle_sha256"],
            "size_bytes": bundle_path.stat().st_size,
            "metadata_path": _catalog_path(metadata_path, project_root),
            "metadata_sha256": sha256_file(metadata_path),
            "baseline": cell["baseline"],
            "dataset": cell["dataset"],
            "task_family": cell["task_family"],
            "pred_len": cell["pred_len"],
            **(
                {
                    "autogluon_timeseries_version": autogluon_version,
                }
                if autogluon_version is not None
                else {}
            ),
            **(
                {
                    "checkpoint_compatibility": metadata[
                        "checkpoint_compatibility"
                    ],
                }
                if cell["baseline"] == "autogluon_high_quality"
                else {}
            ),
        }
    return {
        "schema_version": 2,
        "source_revision": source_revision,
        "project_root": str(Path(project_root).resolve()),
        "manifest": _catalog_path(appendix_manifest, project_root),
        "manifest_sha256": sha256_file(appendix_manifest),
        "replay_catalog": _catalog_path(replay_catalog, project_root),
        "replay_catalog_sha256": sha256_file(replay_catalog),
        "hashes_verified": True,
        "counts": {
            "reported": len(appendix_rows),
            "evaluable": len(evaluable),
            "cataloged": len(bundles),
            "missing_evaluable": len(evaluable) - len(bundles),
        },
        "bundles": bundles,
    }


def _data_root(value):
    family, separator, path = value.partition("=")
    if not separator or family not in {"long_term", "pems", "epf"} or not path:
        raise argparse.ArgumentTypeError(
            "--data-root must use FAMILY=PATH"
        )
    return family, path


def _require_project_path(project_root, value, description, require_exists=False):
    project_root = Path(project_root).resolve()
    path = Path(
        os.path.abspath(os.path.expanduser(os.fspath(value)))
    )
    if path.exists() and path.is_symlink():
        raise ValueError(f"{description} cannot be a symlink")
    resolved = path.resolve()
    if resolved == project_root or project_root not in resolved.parents:
        raise ValueError(f"{description} must live below --project-root")
    if require_exists and not path.exists():
        raise ValueError(f"{description} does not exist: {path}")
    return path


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Build and verify all Appendix baseline prediction bundles"
        )
    )
    parser.add_argument(
        "--appendix-manifest",
        default="./docs/timefuse_appendix_experiment_manifest.jsonl",
    )
    parser.add_argument(
        "--replay-manifest",
        default="./docs/timefuse_checkpoint_replay_manifest.jsonl",
    )
    parser.add_argument("--replay-catalog", required=True)
    parser.add_argument(
        "--output-root",
        default="./ts_rag_outputs/timefuse_appendix_bundles",
    )
    parser.add_argument(
        "--bundle-catalog",
        default="./docs/timefuse_appendix_bundle_catalog.json",
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument(
        "--cache-root",
        help=(
            "EFS root for Hugging Face, Torch, Ray, and temporary worker files"
        ),
    )
    parser.add_argument("--source-revision")
    parser.add_argument("--launcher-revision")
    parser.add_argument(
        "--phase",
        choices=("all", "ensembles", "external", "catalog"),
        default="all",
    )
    parser.add_argument("--forward-ensemble-size", type=int, default=50)
    parser.add_argument("--instance-type", default="ml.g5.12xlarge")
    parser.add_argument("--reserved-gpus-per-host", type=int, default=4)
    parser.add_argument("--processes-per-host", type=int, default=4)
    parser.add_argument(
        "--gpu-ids",
        type=_parse_gpu_ids,
        default=_parse_gpu_ids("0,1,2,3"),
    )
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--time-limit", type=int)
    parser.add_argument("--batch-windows", type=int, default=64)
    parser.add_argument("--max-prediction-items", type=int, default=2048)
    parser.add_argument("--fine-tune-steps", type=int, default=1000)
    parser.add_argument("--inference-batch-size", type=int, default=32)
    parser.add_argument("--fine-tune-batch-size", type=int, default=32)
    parser.add_argument("--data-root", action="append", type=_data_root, default=[])
    parser.add_argument("--max-external-cells", type=int, default=0)
    parser.add_argument("--max-failures", type=int, default=0)
    parser.add_argument("--gpu-sample-seconds", type=float, default=2.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.source_revision = args.source_revision or resolve_source_revision()
    args.launcher_revision = (
        args.launcher_revision or resolve_source_revision()
    )
    external_runtime = None
    if args.phase in {"all", "external"}:
        external_runtime = preflight_external_runtime()

    project_root = Path(args.project_root).resolve()
    cache_root = _require_project_path(
        project_root,
        args.cache_root
        or project_root
        / ".cache"
        / "appendix-runtime"
        / args.source_revision[:12],
        "cache-root",
    )
    output_root = _require_project_path(
        project_root,
        args.output_root,
        "output-root",
    )
    bundle_catalog = _require_project_path(
        project_root,
        args.bundle_catalog,
        "bundle-catalog",
    )
    replay_catalog = _require_project_path(
        project_root,
        args.replay_catalog,
        "replay-catalog",
        require_exists=True,
    )
    appendix_manifest = _require_project_path(
        project_root,
        args.appendix_manifest,
        "appendix-manifest",
        require_exists=True,
    )
    replay_manifest = _require_project_path(
        project_root,
        args.replay_manifest,
        "replay-manifest",
        require_exists=True,
    )
    args.project_root = str(project_root)
    args.cache_root = str(cache_root)
    args.output_root = str(output_root)
    args.bundle_catalog = str(bundle_catalog)
    args.replay_catalog = str(replay_catalog)
    args.appendix_manifest = str(appendix_manifest)
    args.replay_manifest = str(replay_manifest)
    args.data_roots = dict(args.data_root)
    if len(args.data_roots) != len(args.data_root):
        parser.error("Each --data-root family may be specified only once")
    if args.forward_ensemble_size < 1:
        parser.error("--forward-ensemble-size must be positive")
    if args.max_failures < 0 or args.max_external_cells < 0:
        parser.error("Failure and cell limits cannot be negative")
    if args.gpu_sample_seconds <= 0:
        parser.error("--gpu-sample-seconds must be positive")
    args.data_roots = {
        family: str(
            _require_project_path(
                project_root,
                path,
                f"{family} data-root",
                require_exists=True,
            )
        )
        for family, path in args.data_roots.items()
    }

    lock = _acquire_lock(args.output_root)
    context = load_replay_libraries(
        args.replay_manifest,
        args.replay_catalog,
        args.appendix_manifest,
        args.project_root,
        verify_hashes=True,
    )
    save_json_atomic(
        {
            "source_revision": args.source_revision,
            "launcher_revision": args.launcher_revision,
            "appendix_manifest": args.appendix_manifest,
            "appendix_manifest_sha256": sha256_file(
                args.appendix_manifest
            ),
            "replay_manifest": args.replay_manifest,
            "replay_manifest_sha256": context[
                "replay_manifest_sha256"
            ],
            "replay_catalog": args.replay_catalog,
            "replay_catalog_sha256": context[
                "replay_catalog_sha256"
            ],
            "replay_bundles_verified": len(
                context["verified_bundles"]
            ),
            "phase": args.phase,
            "external_runtime": external_runtime,
        },
        Path(args.output_root) / "pipeline_metadata.json",
    )

    if args.dry_run and args.phase in {"all", "ensembles"}:
        print("48 static ensemble bundles")
    elif args.phase in {"all", "ensembles"}:
        build_static_bundles(
            context,
            args.output_root,
            args.source_revision,
            forward_ensemble_size=args.forward_ensemble_size,
        )

    failures = 0
    if args.phase in {"all", "external"}:
        failures = run_external_exports(context, args)

    catalog = None
    if (
        not args.dry_run
        and failures == 0
        and args.phase in {"all", "catalog"}
    ):
        catalog = build_complete_catalog(
            context["appendix_rows"],
            args.appendix_manifest,
            args.replay_catalog,
            args.output_root,
            args.project_root,
            args.source_revision,
        )
        save_json_atomic(catalog, args.bundle_catalog)
    print(
        json.dumps(
            {
                "phase": args.phase,
                "failures": failures,
                "replay_bundles_verified": len(
                    context["verified_bundles"]
                ),
                "catalog_counts": (
                    catalog["counts"] if catalog is not None else None
                ),
                "bundle_catalog": (
                    args.bundle_catalog if catalog is not None else None
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
