"""Run one hash-pinned TimeRAF matrix job inside a Greenland p4de pod."""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from urllib.parse import urlparse

import boto3
from boto3.s3.transfer import TransferConfig

from greenland_common import (
    CHECKPOINT_REPLAY_JOB,
    FULL_MATRIX_REPLICATION_JOB,
    GPUS_PER_HOST,
    NUMERICAL_RECOVERY_JOB,
    build_matrix_command,
    canonical_json_bytes,
    parse_compute_rows,
    parse_gpu_rows,
    parse_s3_uri,
    safe_extract_tar,
    sha256_bytes,
    sha256_file,
    summarize_gpu_samples,
    validate_full_matrix_manifest,
    validate_job_spec,
    validate_numerical_recovery_protocol,
    validate_replay_checkpoints,
    validate_replay_manifest,
)


TRANSFER_CONFIG = TransferConfig(
    multipart_threshold=64 * 1024 * 1024,
    multipart_chunksize=64 * 1024 * 1024,
    max_concurrency=8,
    use_threads=True,
)


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(canonical_json_bytes(payload))
    temporary.replace(path)


def _download_object(s3, uri, destination):
    bucket, key = parse_s3_uri(uri)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    s3.download_file(
        bucket,
        key,
        str(destination),
        Config=TRANSFER_CONFIG,
    )
    return destination


def _read_process_metadata(pid):
    proc = Path("/proc") / str(pid)
    metadata = {"pid": pid}
    try:
        values = proc.joinpath("environ").read_bytes().split(b"\0")
        environment = {}
        for value in values:
            key, separator, raw = value.partition(b"=")
            if separator:
                environment[key.decode(errors="replace")] = raw.decode(
                    errors="replace"
                )
        metadata["cuda_visible_devices"] = environment.get(
            "CUDA_VISIBLE_DEVICES"
        )
    except (OSError, PermissionError):
        metadata["cuda_visible_devices"] = None
    try:
        metadata["cwd"] = str(proc.joinpath("cwd").resolve())
    except (OSError, PermissionError):
        metadata["cwd"] = None
    try:
        metadata["command"] = (
            proc.joinpath("cmdline")
            .read_bytes()
            .replace(b"\0", b" ")
            .decode(errors="replace")
            .strip()
        )
    except (OSError, PermissionError):
        metadata["command"] = None
    return metadata


def _run_nvidia_smi(arguments):
    return subprocess.run(
        ["nvidia-smi", *arguments],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout


def capture_gpu_sample():
    gpu_payload = _run_nvidia_smi(
        [
            "--query-gpu=index,uuid,name,utilization.gpu,memory.used",
            "--format=csv,noheader,nounits",
        ]
    )
    try:
        process_payload = _run_nvidia_smi(
            [
                "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ]
        )
    except subprocess.CalledProcessError:
        process_payload = ""
    gpus = parse_gpu_rows(gpu_payload)
    by_uuid = {gpu["uuid"]: gpu for gpu in gpus}
    for process in parse_compute_rows(process_payload):
        metadata = _read_process_metadata(process["pid"])
        process.update(metadata)
        gpu = by_uuid.get(process["gpu_uuid"])
        if gpu is not None:
            gpu["processes"].append(process)
    return {
        "recorded_unix": time.time(),
        "hostname": socket.gethostname(),
        "gpus": gpus,
    }


class GpuSampler:
    def __init__(self, path, interval_seconds):
        self.path = Path(path)
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
                sample = capture_gpu_sample()
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
            **summarize_gpu_samples(self.samples),
            "sampler_errors": self.errors,
        }


def _upload_tree(s3, local_root, output_s3_uri):
    bucket, prefix = parse_s3_uri(output_s3_uri)
    local_root = Path(local_root)
    uploaded = []
    for path in sorted(local_root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(local_root).as_posix()
        key = f"{prefix.rstrip('/')}/{relative}"
        digest = sha256_file(path)
        s3.upload_file(
            str(path),
            bucket,
            key,
            ExtraArgs={"Metadata": {"sha256": digest}},
            Config=TRANSFER_CONFIG,
        )
        uploaded.append(
            {
                "relative_path": relative,
                "s3_uri": f"s3://{bucket}/{key}",
                "sha256": digest,
                "size_bytes": path.stat().st_size,
            }
        )
    return uploaded


def _upload_marker(s3, output_s3_uri, name, payload):
    bucket, prefix = parse_s3_uri(output_s3_uri)
    body = canonical_json_bytes(payload)
    digest = sha256_bytes(body)
    key = f"{prefix.rstrip('/')}/{name}"
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=body,
        ContentType="application/json",
        Metadata={"sha256": digest},
    )
    return {
        "relative_path": name,
        "s3_uri": f"s3://{bucket}/{key}",
        "sha256": digest,
        "size_bytes": len(body),
    }


def _load_job_spec(s3, uri, expected_sha256, destination):
    _download_object(s3, uri, destination)
    actual = sha256_file(destination)
    if actual != expected_sha256:
        raise ValueError(
            f"Job spec hash mismatch: expected {expected_sha256}, found {actual}"
        )
    spec = json.loads(Path(destination).read_text(encoding="utf-8"))
    return validate_job_spec(
        spec, expected_image_uri=os.environ.get("TIMERAF_IMAGE_URI")
    )


def _prepare_inputs(s3, spec, work_root):
    project_root = work_root / "project"
    downloads = work_root / "downloads"
    project_root.mkdir(parents=True, exist_ok=True)
    records = []
    for record in spec["inputs"]:
        archive = downloads / f"{record['name']}-{record['sha256']}.tar.gz"
        _download_object(s3, record["s3_uri"], archive)
        actual = sha256_file(archive)
        if actual != record["sha256"]:
            raise ValueError(
                f"{record['name']} hash mismatch: "
                f"expected {record['sha256']}, found {actual}"
            )
        safe_extract_tar(archive, project_root)
        records.append(
            {
                **record,
                "download_path": str(archive),
                "size_bytes": archive.stat().st_size,
            }
        )
    return records


def _stream_process(command, cwd, log_path, process_holder):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as output:
        output.write(
            json.dumps(
                {
                    "event": "matrix_start",
                    "recorded_unix": time.time(),
                    "command": command,
                    "cwd": str(cwd),
                },
                sort_keys=True,
            )
            + "\n"
        )
        output.flush()
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        process_holder["process"] = process
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            output.write(line)
            output.flush()
        return process.wait()


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run a TimeRAF p4de job from immutable S3 inputs"
    )
    parser.add_argument(
        "--job-spec-s3-uri",
        default=os.environ.get("TIMERAF_JOB_SPEC_S3_URI"),
    )
    parser.add_argument(
        "--job-spec-sha256",
        default=os.environ.get("TIMERAF_JOB_SPEC_SHA256"),
    )
    parser.add_argument("--work-root", default="/workspace/timeraf")
    parser.add_argument("--sample-seconds", type=float, default=2.0)
    args = parser.parse_args(argv)
    if not args.job_spec_s3_uri or not args.job_spec_sha256:
        parser.error("Job spec S3 URI and SHA-256 are required")

    work_root = Path(args.work_root).resolve()
    output_root = work_root / "outputs"
    evidence_root = output_root / "evidence"
    output_root.mkdir(parents=True, exist_ok=True)
    process_holder = {}
    interrupted = {"signal": None}

    def stop_handler(signum, _frame):
        interrupted["signal"] = signum
        process = process_holder.get("process")
        if process and process.poll() is None:
            process.terminate()

    signal.signal(signal.SIGTERM, stop_handler)
    signal.signal(signal.SIGINT, stop_handler)

    started = time.time()
    s3 = boto3.client("s3", region_name="us-east-1")
    spec = None
    sampler = None
    matrix_started = None
    matrix_finished = None
    exit_code = 1
    error_payload = None
    uploaded = []
    completion_marker = None
    try:
        spec = _load_job_spec(
            s3,
            args.job_spec_s3_uri,
            args.job_spec_sha256,
            work_root / "job-spec.json",
        )
        _write_json(evidence_root / "accepted_job_spec.json", spec)
        input_records = _prepare_inputs(s3, spec, work_root)
        _write_json(evidence_root / "input_records.json", input_records)
        manifest_path = (
            work_root / "project" / spec["matrix"]["manifest"]
        )
        if spec["job_kind"] == CHECKPOINT_REPLAY_JOB:
            manifest_evidence = validate_replay_manifest(manifest_path)
            _write_json(
                evidence_root / "replay_manifest.json",
                manifest_evidence,
            )
            checkpoint_evidence = validate_replay_checkpoints(
                work_root / "project",
                spec["matrix"]["manifest"],
                spec["matrix"]["checkpoint_catalog"],
                spec["matrix"]["release_checkpoint_root"],
                expected_method_revision=spec["method_revision"],
            )
            _write_json(
                evidence_root / "replay_checkpoints.json",
                checkpoint_evidence,
            )
        elif spec["job_kind"] == FULL_MATRIX_REPLICATION_JOB:
            manifest_evidence = validate_full_matrix_manifest(manifest_path)
            _write_json(
                evidence_root / "full_matrix_manifest.json",
                manifest_evidence,
            )
        elif spec["job_kind"] == NUMERICAL_RECOVERY_JOB:
            manifest_evidence = validate_full_matrix_manifest(manifest_path)
            _write_json(
                evidence_root / "full_matrix_manifest.json",
                manifest_evidence,
            )
            recovery_evidence = validate_numerical_recovery_protocol(
                work_root / "project" / spec["recovery"]["protocol"]
            )
            if (
                recovery_evidence["sha256"]
                != spec["recovery"]["protocol_sha256"]
                or recovery_evidence["cohort_id"]
                != spec["recovery"]["cohort_id"]
                or recovery_evidence["cell_ids"]
                != spec["recovery"]["cell_ids"]
                or recovery_evidence["cell_ids_sha256"]
                != spec["recovery"]["cell_ids_sha256"]
                or recovery_evidence["profiles"][
                    spec["recovery"]["profile"]
                ]
                != spec["recovery"]["overrides"]
            ):
                raise ValueError(
                    "Accepted numerical recovery spec drifted from protocol"
                )
            _write_json(
                evidence_root / "numerical_recovery_protocol.json",
                recovery_evidence,
            )
        else:
            raise ValueError(f"Unsupported job kind: {spec['job_kind']}")

        initial = capture_gpu_sample()
        if len(initial["gpus"]) != GPUS_PER_HOST:
            raise RuntimeError(
                f"Expected {GPUS_PER_HOST} visible GPUs, "
                f"found {len(initial['gpus'])}"
            )
        _write_json(evidence_root / "initial_gpu_inventory.json", initial)

        sampler = GpuSampler(
            evidence_root / "gpu_samples.jsonl",
            args.sample_seconds,
        )
        sampler.start()
        command = build_matrix_command(spec, work_root, sys.executable)
        matrix_started = time.time()
        exit_code = _stream_process(
            command,
            work_root / "project",
            output_root / "driver.log",
            process_holder,
        )
        matrix_finished = time.time()
        if interrupted["signal"] is not None:
            raise InterruptedError(
                f"Received signal {interrupted['signal']}"
            )
    except BaseException as error:
        error_payload = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
        print(
            f"Greenland runtime failed: {type(error).__name__}: {error}",
            file=sys.stderr,
            flush=True,
        )
        exit_code = exit_code or 1
    finally:
        if sampler is not None:
            sampler.stop()
            topology = sampler.summary()
        else:
            topology = summarize_gpu_samples([])
        _write_json(evidence_root / "topology_summary.json", topology)
        if exit_code == 0 and not topology["topology_gate_passed"]:
            exit_code = 3
            error_payload = {
                "type": "TopologyGateFailed",
                "message": (
                    "The run did not prove eight distinct GPU bindings and "
                    "nonzero utilization on all eight A100 GPUs"
                ),
            }
        final_status = {
            "run_id": spec.get("run_id") if spec else None,
            "job_kind": spec.get("job_kind") if spec else None,
            "method_revision": (
                spec.get("method_revision") if spec else None
            ),
            "source_revision": spec.get("source_revision") if spec else None,
            "launcher_revision": (
                spec.get("launcher_revision") if spec else None
            ),
            "recovery_revision": (
                spec.get("recovery_revision") if spec else None
            ),
            "recovery_profile": (
                spec.get("recovery", {}).get("profile") if spec else None
            ),
            "started_unix": started,
            "finished_unix": time.time(),
            "elapsed_seconds": time.time() - started,
            "matrix_started_unix": matrix_started,
            "matrix_finished_unix": matrix_finished,
            "matrix_elapsed_seconds": (
                matrix_finished - matrix_started
                if matrix_started is not None and matrix_finished is not None
                else None
            ),
            "hostname": socket.gethostname(),
            "exit_code": exit_code,
            "status": "succeeded" if exit_code == 0 else "failed",
            "signal": interrupted["signal"],
            "topology_gate_passed": topology["topology_gate_passed"],
            "error": error_payload,
        }
        _write_json(output_root / "final_status.json", final_status)
        if spec is not None:
            try:
                uploaded = _upload_tree(
                    s3, output_root, spec["output_s3_uri"]
                )
                completion_marker = _upload_marker(
                    s3,
                    spec["output_s3_uri"],
                    "upload_complete.json",
                    {
                        "schema_version": 1,
                        "run_id": spec["run_id"],
                        "job_kind": spec["job_kind"],
                        "method_revision": spec["method_revision"],
                        "source_revision": spec["source_revision"],
                        "launcher_revision": spec["launcher_revision"],
                        "recovery_revision": spec.get(
                            "recovery_revision"
                        ),
                        "recovery_profile": spec.get(
                            "recovery", {}
                        ).get("profile"),
                        "completed_unix": time.time(),
                        "uploaded_file_count": len(uploaded),
                        "uploaded_size_bytes": sum(
                            record["size_bytes"] for record in uploaded
                        ),
                        "objects": uploaded,
                    },
                )
            except Exception as upload_error:
                upload_failure = {
                    "schema_version": 1,
                    "run_id": spec["run_id"],
                    "job_kind": spec["job_kind"],
                    "method_revision": spec["method_revision"],
                    "source_revision": spec["source_revision"],
                    "launcher_revision": spec["launcher_revision"],
                    "recovery_revision": spec.get("recovery_revision"),
                    "recovery_profile": spec.get(
                        "recovery", {}
                    ).get("profile"),
                    "failed_unix": time.time(),
                    "uploaded_file_count_before_failure": len(uploaded),
                    "error": {
                        "type": type(upload_error).__name__,
                        "message": str(upload_error),
                    },
                }
                print(
                    f"Output upload failed: {type(upload_error).__name__}: "
                    f"{upload_error}",
                    file=sys.stderr,
                    flush=True,
                )
                exit_code = 4
                try:
                    _upload_marker(
                        s3,
                        spec["output_s3_uri"],
                        "upload_failed.json",
                        upload_failure,
                    )
                except Exception as marker_error:
                    print(
                        "Upload failure marker also failed: "
                        f"{type(marker_error).__name__}: {marker_error}",
                        file=sys.stderr,
                        flush=True,
                    )
        print(
            json.dumps(
                {
                    **final_status,
                    "exit_code": exit_code,
                    "uploaded_files": len(uploaded),
                    "upload_completion_marker": completion_marker,
                    "output_s3_uri": (
                        spec.get("output_s3_uri") if spec else None
                    ),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
