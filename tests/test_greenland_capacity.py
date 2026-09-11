from pathlib import Path

import pytest

from scripts import check_greenland_capacity


def _payloads(*, available=19, maximum=19, running=2):
    allocations = {
        "resourceAllocations": [
            {
                "AllocationId": "feedml-sp-os#allocation",
                "AvailabilityZoneId": "use1-az4",
                "InstanceCount": 2,
                "InstanceType": "p4de.24xlarge",
                "Region": "us-east-1",
                "Runtime": "EKS",
                "Status": "Approved",
            }
        ]
    }
    running_counts = {
        "Resources": [
            {
                "InstanceType": "p4de.24xlarge",
                "Allocations": [
                    {
                        "Region": "us-east-1",
                        "RunningInstanceCount": running,
                    }
                ],
            }
        ]
    }
    fleet = {
        "timestamp": "2026-08-01T03:00:00Z",
        "data": [
            {
                "available": available,
                "instanceType": "p4de.24xlarge",
                "maxJobSize": maximum,
                "region": "us-east-1",
                "running": 103 - available,
                "runtime": "EKS",
                "total": 103,
            }
        ],
    }
    return allocations, running_counts, fleet


def test_capacity_summary_records_initiative_and_pool_state():
    summary = check_greenland_capacity.summarize_capacity(
        *_payloads(),
        requested_hosts=1,
        collected_at="2026-08-01T03:01:00Z",
    )

    assert summary["submission_eligible"] is True
    assert summary["initiative_summary"] == {
        "allocated_nodes": 2,
        "running_nodes": 2,
        "free_allocated_nodes": 0,
    }
    assert summary["immediate_allocated_capacity"] is False
    assert summary["immediate_pool_capacity"] is True
    assert summary["decision"] == "borrow-pool-capacity"
    assert summary["topology_aware_eks_fleet"][0]["max_job_size"] == 19


def test_capacity_summary_allows_queue_but_rejects_unsupported_host_count():
    payloads = _payloads(available=0, maximum=1)
    summary = check_greenland_capacity.summarize_capacity(
        *payloads,
        requested_hosts=1,
    )

    assert summary["submission_eligible"] is True
    assert summary["immediate_pool_capacity"] is False
    assert summary["decision"] == "submit-and-queue"
    with pytest.raises(ValueError, match="maximum job size"):
        check_greenland_capacity.summarize_capacity(
            *payloads,
            requested_hosts=2,
        )


def test_capacity_summary_requires_approved_allocation():
    payloads = list(_payloads())
    payloads[0]["resourceAllocations"][0]["Status"] = "Expired"

    with pytest.raises(ValueError, match="not approved"):
        check_greenland_capacity.summarize_capacity(*payloads)


def test_midway_cookie_parser_does_not_return_unrelated_fields(tmp_path):
    cookie = tmp_path / "cookie"
    cookie.write_text(
        "user_name\tznliu\nsession\tsecret\nother\tignored\n",
        encoding="utf-8",
    )

    assert check_greenland_capacity.parse_midway_cookie(cookie) == {
        "user_name": "znliu",
        "session": "secret",
    }


def test_capacity_snapshot_is_written_atomically(tmp_path):
    output = tmp_path / "operations" / "capacity.json"
    check_greenland_capacity.save_json_atomic({"schema_version": 1}, output)

    assert output.read_text(encoding="utf-8") == (
        '{\n  "schema_version": 1\n}\n'
    )
    assert list(output.parent.glob(f".{output.name}.tmp-*")) == []
