"""Report whether a hash-verified Greenland output upload has completed."""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from scripts.greenland_common import (
    SERVICE_ROLE_ARN,
    S3_ROOT,
    parse_s3_uri,
    sha256_bytes,
)


def _assumed_s3_client(region):
    import boto3

    sts = boto3.client("sts", region_name=region)
    identity = sts.get_caller_identity()
    if identity["Account"] != "<AWS_ACCOUNT_ID>":
        raise RuntimeError(
            "Greenland status must start in modeldev account "
            f"<AWS_ACCOUNT_ID>, found {identity['Account']}"
        )
    response = sts.assume_role(
        RoleArn=SERVICE_ROLE_ARN,
        RoleSessionName="timeraf-greenland-run-status",
        DurationSeconds=3600,
    )
    credentials = response["Credentials"]
    return boto3.client(
        "s3",
        region_name=region,
        aws_access_key_id=credentials["AccessKeyId"],
        aws_secret_access_key=credentials["SecretAccessKey"],
        aws_session_token=credentials["SessionToken"],
    )


def _read_marker(s3, run_id, name):
    uri = f"{S3_ROOT}runs/{run_id}/{name}"
    bucket, key = parse_s3_uri(uri)
    try:
        response = s3.get_object(Bucket=bucket, Key=key)
    except Exception as error:
        response = getattr(error, "response", {})
        code = str(response.get("Error", {}).get("Code", ""))
        if code in {"404", "NoSuchKey", "NotFound"}:
            return None
        raise
    payload = response["Body"].read()
    digest = sha256_bytes(payload)
    recorded = response.get("Metadata", {}).get("sha256")
    if digest != recorded:
        raise ValueError(f"Greenland {name} marker hash mismatch")
    marker = json.loads(payload)
    if marker.get("run_id") != run_id or marker.get("schema_version") != 1:
        raise ValueError(f"Greenland {name} marker identity drifted")
    return uri, digest, marker


def inspect_run(s3, run_id):
    completed = _read_marker(s3, run_id, "upload_complete.json")
    if completed is None:
        failed = _read_marker(s3, run_id, "upload_failed.json")
        if failed is None:
            return {
                "run_id": run_id,
                "upload_complete": False,
                "upload_failed": False,
            }
        uri, digest, marker = failed
        return {
            "run_id": run_id,
            "upload_complete": False,
            "upload_failed": True,
            "upload_failed_uri": uri,
            "upload_failed_sha256": digest,
            "job_kind": marker.get("job_kind"),
            "method_revision": marker.get("method_revision"),
            "source_revision": marker.get("source_revision"),
            "launcher_revision": marker.get("launcher_revision"),
            "recovery_revision": marker.get("recovery_revision"),
            "recovery_profile": marker.get("recovery_profile"),
            "error": marker.get("error"),
        }
    uri, digest, marker = completed
    return {
        "run_id": run_id,
        "upload_complete": True,
        "upload_failed": False,
        "upload_complete_uri": uri,
        "upload_complete_sha256": digest,
        "job_kind": marker.get("job_kind"),
        "method_revision": marker.get("method_revision"),
        "source_revision": marker.get("source_revision"),
        "launcher_revision": marker.get("launcher_revision"),
        "recovery_revision": marker.get("recovery_revision"),
        "recovery_profile": marker.get("recovery_profile"),
        "uploaded_file_count": marker.get("uploaded_file_count"),
        "uploaded_size_bytes": marker.get("uploaded_size_bytes"),
        "completed_unix": marker.get("completed_unix"),
    }


def validate_terminal_identity(
    status,
    *,
    job_kind,
    method_revision,
    source_revision,
    launcher_revision,
    recovery_revision=None,
    recovery_profile=None,
):
    if not status.get("upload_complete") and not status.get("upload_failed"):
        return status
    expected = {
        "job_kind": job_kind,
        "method_revision": method_revision,
        "source_revision": source_revision,
        "launcher_revision": launcher_revision,
    }
    if recovery_revision is not None or recovery_profile is not None:
        if recovery_revision is None or recovery_profile is None:
            raise ValueError(
                "Recovery revision and profile must be validated together"
            )
        expected.update(
            {
                "recovery_revision": recovery_revision,
                "recovery_profile": recovery_profile,
            }
        )
    drifted = {
        field: {"expected": value, "found": status.get(field)}
        for field, value in expected.items()
        if status.get(field) != value
    }
    if drifted:
        raise ValueError(
            "Greenland terminal marker identity drifted: "
            + json.dumps(drifted, sort_keys=True)
        )
    return status


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Check a Greenland run's final S3 upload marker"
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--expected-job-kind")
    parser.add_argument("--expected-method-revision")
    parser.add_argument("--expected-source-revision")
    parser.add_argument("--expected-launcher-revision")
    parser.add_argument("--expected-recovery-revision")
    parser.add_argument("--expected-recovery-profile")
    args = parser.parse_args(argv)
    status = inspect_run(_assumed_s3_client(args.region), args.run_id)
    expected = (
        args.expected_job_kind,
        args.expected_method_revision,
        args.expected_source_revision,
        args.expected_launcher_revision,
    )
    if any(expected):
        if not all(expected):
            parser.error("all expected terminal identity fields are required")
        validate_terminal_identity(
            status,
            job_kind=args.expected_job_kind,
            method_revision=args.expected_method_revision,
            source_revision=args.expected_source_revision,
            launcher_revision=args.expected_launcher_revision,
            recovery_revision=args.expected_recovery_revision,
            recovery_profile=args.expected_recovery_profile,
        )
    print(json.dumps(status, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
