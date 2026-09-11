"""Render and optionally submit a fully utilized TimeRAF Greenland job."""

from __future__ import annotations

import argparse
import base64
import json
import os
import secrets
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

from greenland_common import (
    APPROVED_IMAGE_RECORD,
    GPUS_PER_HOST,
    INITIATIVE_ID,
    INSTANCE_TYPE,
    REGION,
    SERVICE_ROLE_ARN,
    canonical_json_bytes,
    parse_s3_uri,
    sha256_bytes,
    sha256_file,
    validate_approved_image_uri,
    validate_job_spec,
)


def _assumed_s3_client():
    import boto3

    sts = boto3.client("sts", region_name=REGION)
    identity = sts.get_caller_identity()
    if identity["Account"] != "<AWS_ACCOUNT_ID>":
        raise RuntimeError(
            f"Expected modeldev account <AWS_ACCOUNT_ID>, found {identity['Account']}"
        )
    response = sts.assume_role(
        RoleArn=SERVICE_ROLE_ARN,
        RoleSessionName="timeraf-greenland-submit",
        DurationSeconds=3600,
    )
    credentials = response["Credentials"]
    return boto3.client(
        "s3",
        region_name=REGION,
        aws_access_key_id=credentials["AccessKeyId"],
        aws_secret_access_key=credentials["SecretAccessKey"],
        aws_session_token=credentials["SessionToken"],
    )


def _verify_s3_object(s3, uri, expected_sha256):
    bucket, key = parse_s3_uri(uri)
    response = s3.head_object(Bucket=bucket, Key=key)
    recorded = response.get("Metadata", {}).get("sha256")
    if recorded != expected_sha256:
        raise ValueError(
            f"S3 metadata SHA-256 mismatch for {uri}: "
            f"expected {expected_sha256}, found {recorded}"
        )
    return {
        "s3_uri": uri,
        "sha256": recorded,
        "size_bytes": response["ContentLength"],
        "version_id": response.get("VersionId"),
        "etag": response.get("ETag"),
    }


def verify_remote_inputs(s3, spec, spec_uri, spec_sha256):
    records = [_verify_s3_object(s3, spec_uri, spec_sha256)]
    records.extend(
        _verify_s3_object(s3, record["s3_uri"], record["sha256"])
        for record in spec["inputs"]
    )
    return records


def _job_name(run_id):
    suffix = sha256_bytes(run_id.encode("utf-8"))[:8]
    return f"timeraf-{run_id[:14]}-{suffix}"


def build_app_definition(
    spec,
    spec_uri,
    spec_sha256,
    *,
    torchx_specs=None,
    set_overlay=None,
    auth_token=None,
):
    validate_job_spec(spec)
    if torchx_specs is None:
        from torchx import specs as torchx_specs
    if set_overlay is None:
        from torchx.specs.overlays import set_overlay

    auth_token = auth_token or secrets.token_hex(16)
    job_name = _job_name(spec["run_id"])
    environment = {
        "AWS_CONTAINER_CREDENTIALS_FULL_URI": "http://127.0.0.1:9090/creds",
        "AWS_CONTAINER_AUTHORIZATION_TOKEN": auth_token,
        "AWS_ROLE_ARN": "",
        "AWS_WEB_IDENTITY_TOKEN_FILE": "",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUNBUFFERED": "1",
        "TIMERAF_IMAGE_URI": spec["image_uri"],
        "TIMERAF_JOB_SPEC_S3_URI": spec_uri,
        "TIMERAF_JOB_SPEC_SHA256": spec_sha256,
    }
    role = torchx_specs.Role(
        name="timeraf",
        image=spec["image_uri"],
        entrypoint="python",
        args=[
            "/opt/timeraf-runtime/run_greenland_job.py",
            "--sample-seconds",
            "2",
        ],
        env=environment,
        num_replicas=1,
        min_replicas=1,
        max_retries=0,
        resource=torchx_specs.Resource(
            cpu=90,
            gpu=GPUS_PER_HOST,
            memMB=1_000_000,
            capabilities={
                "node.kubernetes.io/instance-type": INSTANCE_TYPE,
            },
        ),
    )
    set_overlay(
        role,
        "kubernetes",
        "V1Pod",
        {
            "spec": {
                "shareProcessNamespace": True,
                "containers": [
                    {
                        "name": "credential-proxy",
                        "image": spec["image_uri"],
                        "command": [
                            "python",
                            "/opt/timeraf-runtime/"
                            "greenland_credential_proxy.py",
                        ],
                        "env": [
                            {
                                "name": "AWS_CONFIG_FILE",
                                "value": "/greenland/.aws/config",
                            },
                            {
                                "name": "PROXY_AUTH_TOKEN",
                                "value": auth_token,
                            },
                            {"name": "PROXY_BIND_PORT", "value": "9090"},
                        ],
                        "volumeMounts": [
                            {
                                "name": "greenland-aws-config",
                                "mountPath": "/greenland/.aws",
                                "readOnly": True,
                            }
                        ],
                        "resources": {
                            "requests": {"cpu": "50m", "memory": "128Mi"},
                            "limits": {"cpu": "200m", "memory": "256Mi"},
                        },
                    }
                ],
            }
        },
    )
    annotations = {
        "timeraf.amazon.com/method-revision": spec["method_revision"],
        "timeraf.amazon.com/source-revision": spec["source_revision"],
        "timeraf.amazon.com/job-spec-sha256": spec_sha256,
    }
    if spec.get("recovery_revision"):
        annotations.update(
            {
                "timeraf.amazon.com/recovery-revision": spec[
                    "recovery_revision"
                ],
                "timeraf.amazon.com/recovery-profile": spec["recovery"][
                    "profile"
                ],
            }
        )
    app = torchx_specs.AppDef(
        name=job_name,
        roles=[role],
        metadata={
            "labels": {
                "timeraf.amazon.com/run-id": spec["run_id"],
                "timeraf.amazon.com/job-kind": spec["job_kind"],
                "timeraf.amazon.com/world-size": str(GPUS_PER_HOST),
            },
            "annotations": annotations,
        },
    )
    return app


def scheduler_config():
    return {
        "initiative_id": INITIATIVE_ID,
        "region": REGION,
        "instance_type": INSTANCE_TYPE,
        "runtime": "EKS",
        "cs_role_arn": SERVICE_ROLE_ARN,
        "is_production": False,
        "retry_limit": 0,
    }


def validate_scheduler_manifest(dryrun_info, expected_image_uri):
    import yaml

    request = dryrun_info.request
    if request.instance_type != INSTANCE_TYPE or request.instance_count != 1:
        raise ValueError(
            "Greenland request must resolve to one p4de.24xlarge host"
        )
    manifest = yaml.safe_load(request.eks_manifest_yaml)
    if manifest["spec"].get("minAvailable") != 1:
        raise ValueError("Volcano manifest must require its one GPU pod")
    tasks = manifest["spec"].get("tasks", [])
    if len(tasks) != 1 or tasks[0].get("replicas") != 1:
        raise ValueError("Volcano manifest must contain one trainer pod")
    pod = tasks[0]["template"]["spec"]
    if (
        pod.get("nodeSelector", {}).get(
            "node.kubernetes.io/instance-type"
        )
        != INSTANCE_TYPE
    ):
        raise ValueError("Volcano manifest does not select p4de.24xlarge")
    main = next(
        (
            container
            for container in pod.get("containers", [])
            if container.get("name") == "timeraf"
            or str(container.get("name", "")).startswith("timeraf-")
        ),
        None,
    )
    if main is None:
        raise ValueError("Volcano manifest has no TimeRAF container")
    limits = main.get("resources", {}).get("limits", {})
    requests = main.get("resources", {}).get("requests", {})
    if limits.get("nvidia.com/gpu") != "8":
        raise ValueError("TimeRAF container does not limit eight GPUs")
    if requests.get("nvidia.com/gpu") != "8":
        raise ValueError("TimeRAF container does not request eight GPUs")
    if main.get("image") != expected_image_uri:
        raise ValueError("Manifest image is not the hash-pinned image")
    if not any(
        container.get("name") == "credential-proxy"
        for container in pod.get("containers", [])
    ):
        raise ValueError("Manifest has no credential proxy sidecar")
    return manifest


def _put_control_object(s3, output_uri, name, payload, content_type):
    bucket, prefix = parse_s3_uri(output_uri)
    body = payload if isinstance(payload, bytes) else payload.encode("utf-8")
    digest = sha256_bytes(body)
    key = f"{prefix.rstrip('/')}/control/{name}"
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=body,
        ContentType=content_type,
        Metadata={"sha256": digest},
    )
    return {
        "s3_uri": f"s3://{bucket}/{key}",
        "sha256": digest,
        "size_bytes": len(body),
    }


def _parse_handle(handle):
    encoded = handle.rsplit("/", 1)[-1]
    parts = encoded.split("_", 3)
    job_uuid = parts[2] if len(parts) >= 3 else encoded
    return {
        "app_handle": handle,
        "job_uuid": job_uuid,
        "console_url": (
            "https://greenland.harmony.a2z.com/job/details/"
            f"{REGION}/{INITIATIVE_ID}/{job_uuid}"
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Dry-run or submit one TimeRAF p4de job"
    )
    parser.add_argument("--job-spec", required=True)
    parser.add_argument("--job-spec-s3-uri", required=True)
    parser.add_argument("--image-uri", required=True)
    parser.add_argument("--receipt")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--confirm", default="")
    args = parser.parse_args(argv)

    approval_record = (
        Path(__file__).resolve().parents[1] / APPROVED_IMAGE_RECORD
    )
    spec_path = Path(args.job_spec)
    spec_bytes = spec_path.read_bytes()
    spec_sha256 = sha256_bytes(spec_bytes)
    spec = validate_job_spec(
        json.loads(spec_bytes), expected_image_uri=args.image_uri
    )
    validate_approved_image_uri(
        args.image_uri,
        approval_record,
        required_job_kind=spec["job_kind"],
    )

    s3 = _assumed_s3_client()
    remote_inputs = verify_remote_inputs(
        s3, spec, args.job_spec_s3_uri, spec_sha256
    )

    from torchx.runner import get_runner

    app = build_app_definition(
        spec,
        args.job_spec_s3_uri,
        spec_sha256,
    )
    runner = get_runner("timeraf-greenland")
    dryrun_info = runner.dryrun(
        app,
        scheduler="greenland",
        cfg=scheduler_config(),
    )
    manifest = validate_scheduler_manifest(
        dryrun_info, expected_image_uri=args.image_uri
    )
    import yaml

    plan = {
        "recorded_unix": time.time(),
        "dry_run": not args.submit,
        "run_id": spec["run_id"],
        "job_kind": spec["job_kind"],
        "job_name": app.name,
        "job_spec_s3_uri": args.job_spec_s3_uri,
        "job_spec_sha256": spec_sha256,
        "image_uri": args.image_uri,
        "initiative_id": INITIATIVE_ID,
        "region": REGION,
        "runtime": "EKS",
        "instance_type": INSTANCE_TYPE,
        "instance_count": 1,
        "reserved_gpus_per_host": GPUS_PER_HOST,
        "processes_per_host": GPUS_PER_HOST,
        "total_gpus": GPUS_PER_HOST,
        "world_size": GPUS_PER_HOST,
        "inactive_reserved_gpus": 0,
        "recovery_revision": spec.get("recovery_revision"),
        "recovery_profile": spec.get("recovery", {}).get("profile"),
        "remote_inputs": remote_inputs,
    }
    control_records = [
        _put_control_object(
            s3,
            spec["output_s3_uri"],
            "submission_plan.json",
            canonical_json_bytes(plan),
            "application/json",
        ),
        _put_control_object(
            s3,
            spec["output_s3_uri"],
            "scheduler_manifest.yaml",
            yaml.safe_dump(manifest, sort_keys=True),
            "application/yaml",
        ),
    ]

    if args.submit and args.confirm != "SUBMIT":
        parser.error("A real launch requires --submit --confirm SUBMIT")

    if args.submit:
        handle = runner.schedule(dryrun_info)
        receipt = {
            **plan,
            **_parse_handle(handle),
            "dry_run": False,
            "submitted_unix": time.time(),
            "control_records": control_records,
        }
        control_records.append(
            _put_control_object(
                s3,
                spec["output_s3_uri"],
                "submission_receipt.json",
                canonical_json_bytes(receipt),
                "application/json",
            )
        )
        receipt["control_records"] = control_records
    else:
        receipt = {**plan, "control_records": control_records}

    if args.receipt:
        Path(args.receipt).write_bytes(canonical_json_bytes(receipt))
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
