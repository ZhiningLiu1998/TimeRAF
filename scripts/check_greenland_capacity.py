"""Read and validate live Greenland p4de capacity before submission."""

from __future__ import annotations

import argparse
import http.cookiejar
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib import request


BASE_URL = "https://<GREENLAND_ENDPOINT>"
INITIATIVE_ID = "feedml-sp-os"
INSTANCE_TYPE = "p4de.24xlarge"
REGION = "us-east-1"
RUNTIME = "EKS"


def parse_midway_cookie(path):
    tokens = [
        token
        for token in re.split(r"\t|\n", Path(path).read_text(encoding="utf-8"))
        if token
    ]
    values = {}
    for name in ("user_name", "session"):
        try:
            values[name] = tokens[tokens.index(name) + 1]
        except (ValueError, IndexError) as error:
            raise ValueError(f"Midway cookie is missing {name}") from error
    return values


def summarize_capacity(
    allocations_payload,
    running_payload,
    fleet_payload,
    *,
    requested_hosts=1,
    collected_at=None,
):
    if not isinstance(requested_hosts, int) or requested_hosts < 1:
        raise ValueError("requested_hosts must be a positive integer")
    running_by_type = {
        resource["InstanceType"]: sum(
            int(allocation["RunningInstanceCount"])
            for allocation in resource.get("Allocations", ())
            if allocation.get("Region") == REGION
        )
        for resource in running_payload.get("Resources", ())
    }
    allocations = [
        {
            "allocated_nodes": int(row["InstanceCount"]),
            "allocation_id": row["AllocationId"],
            "availability_zone_id": row["AvailabilityZoneId"],
            "instance_type": row["InstanceType"],
            "running_nodes": running_by_type.get(row["InstanceType"], 0),
            "runtime": row["Runtime"],
            "status": row["Status"],
        }
        for row in allocations_payload.get("resourceAllocations", ())
        if row.get("Region") == REGION
        and row.get("InstanceType") == INSTANCE_TYPE
    ]
    raw_fleet_rows = fleet_payload.get(
        "data",
        fleet_payload.get(
            "rows",
            fleet_payload if isinstance(fleet_payload, list) else (),
        ),
    )
    fleet = [
        {
            "available_nodes": int(row["available"]),
            "instance_type": row["instanceType"],
            "max_job_size": int(row["maxJobSize"]),
            "running_nodes": int(row["running"]),
            "runtime": row["runtime"],
            "total_nodes": int(row["total"]),
        }
        for row in raw_fleet_rows
        if row.get("region") == REGION
        and row.get("runtime") == RUNTIME
        and row.get("instanceType") == INSTANCE_TYPE
    ]
    if not allocations:
        raise ValueError("No initiative p4de allocation was returned")
    if any(
        row["status"] != "Approved"
        or row["runtime"] != RUNTIME
        or row["allocated_nodes"] < 1
        for row in allocations
    ):
        raise ValueError("Initiative p4de allocation is not approved and active")
    if len(fleet) != 1:
        raise ValueError("Expected exactly one topology-aware EKS p4de fleet row")
    if fleet[0]["max_job_size"] < requested_hosts:
        raise ValueError("Requested host count exceeds the live maximum job size")

    allocated_nodes = sum(row["allocated_nodes"] for row in allocations)
    running_nodes = max(row["running_nodes"] for row in allocations)
    free_allocated_nodes = max(0, allocated_nodes - running_nodes)
    immediate_pool_capacity = fleet[0]["available_nodes"] >= requested_hosts
    return {
        "schema_version": 1,
        "collected_at_utc": collected_at
        or datetime.now(timezone.utc).isoformat(),
        "fleet_snapshot_timestamp_utc": fleet_payload.get("timestamp")
        if isinstance(fleet_payload, dict)
        else None,
        "initiative": INITIATIVE_ID,
        "region": REGION,
        "instance_type": INSTANCE_TYPE,
        "runtime": RUNTIME,
        "requested_hosts": requested_hosts,
        "initiative_allocations": allocations,
        "initiative_summary": {
            "allocated_nodes": allocated_nodes,
            "running_nodes": running_nodes,
            "free_allocated_nodes": free_allocated_nodes,
        },
        "topology_aware_eks_fleet": fleet,
        "submission_eligible": True,
        "immediate_allocated_capacity": free_allocated_nodes >= requested_hosts,
        "immediate_pool_capacity": immediate_pool_capacity,
        "decision": (
            "use-free-initiative-allocation"
            if free_allocated_nodes >= requested_hosts
            else (
                "borrow-pool-capacity"
                if immediate_pool_capacity
                else "submit-and-queue"
            )
        ),
    }


def collect_capacity(cookie_path, *, requested_hosts=1, timeout=30):
    source_cookies = parse_midway_cookie(cookie_path)
    cookie_jar = http.cookiejar.CookieJar()
    opener = request.build_opener(request.HTTPCookieProcessor(cookie_jar))

    def read(url, *, data=None, cookie=None):
        headers = {}
        if data is not None:
            headers["Content-Type"] = "application/json"
        if cookie is not None:
            headers["Cookie"] = cookie
        req = request.Request(
            url,
            data=data.encode("utf-8") if data is not None else None,
            headers=headers,
            method="POST" if data is not None else "GET",
        )
        with opener.open(req, timeout=timeout) as response:
            return response.read().decode("utf-8")

    login = json.loads(read(f"{BASE_URL}/sso/login"))
    rfp = next(
        (
            cookie.value
            for cookie in cookie_jar
            if cookie.name == "amzn_sso_rfp"
        ),
        None,
    )
    if not rfp:
        raise RuntimeError("Greenland SSO did not return an RFP cookie")
    authn_endpoint = login.get("authn_endpoint")
    if not isinstance(authn_endpoint, str) or not authn_endpoint:
        raise RuntimeError("Greenland SSO did not return an auth endpoint")
    source_cookie_header = "; ".join(
        f"{name}={value}" for name, value in source_cookies.items()
    )
    token = read(authn_endpoint, cookie=source_cookie_header)
    if any(
        marker in token
        for marker in (
            "missing posture error",
            "More authentication needed",
            "Unauthenticated",
        )
    ):
        raise RuntimeError("Midway SSO returned a posture/authentication error")
    sso_cookie_header = f"amzn_sso_rfp={rfp}; amzn_sso_token={token}"
    body = json.dumps({"InitiativeId": INITIATIVE_ID, "region": REGION})
    payloads = [
        json.loads(
            read(
                f"{BASE_URL}/v1/getinitiativedetailsandresourceallocations",
                data=body,
                cookie=sso_cookie_header,
            )
        ),
        json.loads(
            read(
                f"{BASE_URL}/v2/getrunninginstancecounts",
                data=body,
                cookie=sso_cookie_header,
            )
        ),
        json.loads(
            read(
                f"{BASE_URL}/v2/getavailableinstances",
                data="{}",
                cookie=sso_cookie_header,
            )
        ),
    ]
    return summarize_capacity(
        *payloads,
        requested_hosts=requested_hosts,
    )


def save_json_atomic(payload, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(
        f".{output.name}.tmp-{int(time.time() * 1_000_000)}"
    )
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Record live feedml-sp-os initiative and pool-wide EKS p4de "
            "capacity"
        )
    )
    parser.add_argument(
        "--midway-cookie",
        default=str(Path.home() / ".midway" / "cookie"),
    )
    parser.add_argument("--requested-hosts", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--output")
    args = parser.parse_args(argv)

    payload = collect_capacity(
        args.midway_cookie,
        requested_hosts=args.requested_hosts,
        timeout=args.timeout,
    )
    if args.output:
        save_json_atomic(payload, args.output)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
