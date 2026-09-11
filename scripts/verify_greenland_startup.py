"""Verify that a Greenland EKS matrix started one worker on every GPU."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


EXPECTED_GPU_IDS = tuple(range(8))
WORKER_HEADER = re.compile(
    r"^\[(?P<recorded_at>[^\]]+)\] "
    r"worker_gpu=(?P<worker_gpu>[0-7]) "
    r"local_gpu=(?P<local_gpu>\d+) "
    r"CUDA_VISIBLE_DEVICES=(?P<visible_gpu>[0-7]) "
)
SCHEDULED_CELL = re.compile(r"^\[(?P<scheduled>\d+)/585\]")
COMPLETED_RESULT = re.compile(r'^\s*"cell_id"\s*:')
ERROR_PATTERNS = (
    "cell failed",
    "Greenland runtime failed",
    "Traceback",
)


class AwsCliLogsClient:
    def __init__(self, profile, region):
        self.profile = profile
        self.region = region

    def _run(self, *arguments):
        command = [
            "aws",
            "--profile",
            self.profile,
            "--region",
            self.region,
            "logs",
            *arguments,
            "--output",
            "json",
            "--no-cli-pager",
        ]
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
        return json.loads(completed.stdout)

    def describe_streams(self, log_group, job_name):
        try:
            payload = self._run(
                "describe-log-streams",
                "--log-group-name",
                log_group,
                "--log-stream-name-prefix",
                job_name,
            )
        except subprocess.CalledProcessError as error:
            if "ResourceNotFoundException" in error.stderr:
                return []
            raise
        return payload.get("logStreams", ())

    def filter_events(self, log_group, log_stream, pattern):
        payload = self._run(
            "filter-log-events",
            "--log-group-name",
            log_group,
            "--log-stream-names",
            log_stream,
            "--filter-pattern",
            json.dumps(pattern),
        )
        return payload.get("events", ())

    def latest_events(self, log_group, log_stream):
        payload = self._run(
            "get-log-events",
            "--log-group-name",
            log_group,
            "--log-stream-name",
            log_stream,
            "--limit",
            "1",
            "--no-start-from-head",
        )
        return payload.get("events", ())


def _iso_from_millis(timestamp):
    return (
        datetime.fromtimestamp(
            timestamp / 1000,
            tz=timezone.utc,
        )
        .isoformat()
        .replace("+00:00", "Z")
    )


def _submitted_millis(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("--submitted-at must include a timezone")
    return int(parsed.timestamp() * 1000)


def extract_worker_bindings(events):
    bindings = {}
    for event in sorted(events, key=lambda item: item["timestamp"]):
        match = WORKER_HEADER.match(event["message"])
        if match is None:
            continue
        worker_gpu = int(match.group("worker_gpu"))
        binding = {
            "worker_gpu": worker_gpu,
            "local_gpu": int(match.group("local_gpu")),
            "cuda_visible_devices": int(match.group("visible_gpu")),
            "recorded_at": match.group("recorded_at"),
            "cloudwatch_timestamp": _iso_from_millis(event["timestamp"]),
        }
        prior = bindings.setdefault(worker_gpu, binding)
        if (
            prior["local_gpu"] != binding["local_gpu"]
            or prior["cuda_visible_devices"] != binding["cuda_visible_devices"]
        ):
            raise ValueError(f"Worker GPU {worker_gpu} binding drifted")

    missing = sorted(set(EXPECTED_GPU_IDS) - set(bindings))
    if missing:
        raise ValueError(f"Missing Greenland worker headers for GPUs {missing}")
    ordered = [bindings[gpu_id] for gpu_id in EXPECTED_GPU_IDS]
    if any(
        binding["local_gpu"] != 0
        or binding["cuda_visible_devices"] != binding["worker_gpu"]
        for binding in ordered
    ):
        raise ValueError(
            "Greenland workers must use distinct physical GPUs and local cuda:0"
        )
    return ordered


def summarize_progress(schedule_events, result_events, error_events):
    scheduled = []
    for event in schedule_events:
        match = SCHEDULED_CELL.match(event["message"])
        if match is not None:
            scheduled.append(
                (
                    int(match.group("scheduled")),
                    event["timestamp"],
                    event["message"],
                )
            )
    latest = max(scheduled, default=None)
    completed = sum(
        COMPLETED_RESULT.match(event["message"]) is not None for event in result_events
    )
    errors = {pattern: len(error_events.get(pattern, ())) for pattern in ERROR_PATTERNS}
    return {
        "scheduled_cells": latest[0] if latest else 0,
        "completed_result_lines": completed,
        "latest_schedule_at": (_iso_from_millis(latest[1]) if latest else None),
        "latest_schedule_message": latest[2] if latest else None,
        "error_counts": errors,
    }


def summarize_activity(events, observed_millis=None):
    latest = max(events, key=lambda event: event["timestamp"], default=None)
    if latest is None:
        return {
            "latest_log_at": None,
            "latest_log_message": None,
            "age_seconds": None,
        }
    observed_millis = (
        int(time.time() * 1000)
        if observed_millis is None
        else observed_millis
    )
    return {
        "latest_log_at": _iso_from_millis(latest["timestamp"]),
        "latest_log_message": latest["message"][-1000:],
        "age_seconds": max(
            0.0,
            (observed_millis - latest["timestamp"]) / 1000,
        ),
    }


def choose_log_stream(streams, job_name, submitted_millis):
    candidates = [
        stream
        for stream in streams
        if stream["logStreamName"].startswith(job_name)
        and stream.get("firstEventTimestamp", 0) >= submitted_millis - 300_000
    ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda stream: (
            stream.get("lastIngestionTime", 0),
            stream["logStreamName"],
        ),
    )["logStreamName"]


def _describe_streams(client, log_group, job_name):
    return client.describe_streams(log_group, job_name)


def _filter_events(client, log_group, log_stream, pattern):
    return client.filter_events(log_group, log_stream, pattern)


def _latest_events(client, log_group, log_stream):
    return client.latest_events(log_group, log_stream)


def wait_for_startup(
    client,
    *,
    job_uuid,
    job_name,
    initiative_id,
    region,
    submitted_at,
    poll_seconds,
    timeout_seconds,
    log_group=None,
):
    log_group = log_group or (
        f"/aws/eks/greenland-eks-{region}/greenland-{initiative_id}"
    )
    submitted_millis = _submitted_millis(submitted_at)
    deadline = time.monotonic() + timeout_seconds
    last_reason = "no matching CloudWatch log stream"
    while time.monotonic() < deadline:
        streams = _describe_streams(client, log_group, job_name)
        log_stream = choose_log_stream(
            streams,
            job_name,
            submitted_millis,
        )
        if log_stream is not None:
            worker_events = _filter_events(
                client,
                log_group,
                log_stream,
                "worker_gpu=",
            )
            try:
                bindings = extract_worker_bindings(worker_events)
            except ValueError as error:
                last_reason = str(error)
            else:
                schedule_events = _filter_events(
                    client,
                    log_group,
                    log_stream,
                    "/585]",
                )
                result_events = _filter_events(
                    client,
                    log_group,
                    log_stream,
                    "cell_id",
                )
                error_events = {
                    pattern: _filter_events(
                        client,
                        log_group,
                        log_stream,
                        pattern,
                    )
                    for pattern in ERROR_PATTERNS
                }
                observed = datetime.now(timezone.utc)
                activity = summarize_activity(
                    _latest_events(client, log_group, log_stream),
                    observed_millis=int(observed.timestamp() * 1000),
                )
                return {
                    "schema_version": 1,
                    "job_uuid": job_uuid,
                    "job_name": job_name,
                    "initiative_id": initiative_id,
                    "region": region,
                    "submitted_at": submitted_at,
                    "observed_at": observed.isoformat().replace(
                        "+00:00",
                        "Z",
                    ),
                    "log_group": log_group,
                    "log_stream": log_stream,
                    "worker_bindings": bindings,
                    "startup_topology_gate_passed": True,
                    "progress": summarize_progress(
                        schedule_events,
                        result_events,
                        error_events,
                    ),
                    "activity": activity,
                    "final_topology_acceptance": (
                        "pending uploaded PID and utilization sampler evidence"
                    ),
                }
        time.sleep(poll_seconds)
    raise TimeoutError(
        "Timed out waiting for Greenland startup evidence: " + last_reason
    )


def _write_json_atomic(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Verify all eight Greenland EKS matrix worker headers"
    )
    parser.add_argument("--job-uuid", required=True)
    parser.add_argument("--job-name", required=True)
    parser.add_argument("--submitted-at", required=True)
    parser.add_argument("--initiative-id", default="feedml-sp-os")
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--log-group")
    parser.add_argument(
        "--aws-profile",
        default=os.environ.get(
            "TIMERAF_GREENLAND_LOG_PROFILE",
            "timeraf-greenland-console",
        ),
    )
    parser.add_argument("--poll-seconds", type=float, default=60)
    parser.add_argument("--timeout-seconds", type=float, default=3600)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    client = AwsCliLogsClient(args.aws_profile, args.region)
    evidence = wait_for_startup(
        client,
        job_uuid=args.job_uuid,
        job_name=args.job_name,
        initiative_id=args.initiative_id,
        region=args.region,
        submitted_at=args.submitted_at,
        poll_seconds=args.poll_seconds,
        timeout_seconds=args.timeout_seconds,
        log_group=args.log_group,
    )
    _write_json_atomic(args.output, evidence)
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
