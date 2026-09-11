from pathlib import Path

import pytest

from scripts import verify_greenland_startup as startup


def test_progress_monitor_is_finite_and_fail_closed():
    monitor = Path(
        "scripts/timefuse_greenland_progress_monitor.sh"
    ).read_text(encoding="utf-8")

    assert "verify_greenland_startup.py" in monitor
    assert ".startup_topology_gate_passed == true" in monitor
    assert "(.worker_bindings | length) == 8" in monitor
    assert "MAX_ACTIVITY_AGE_SECONDS" in monitor
    assert "ALERT stale_log_activity" in monitor
    assert "RECOVERY log_activity_resumed" in monitor
    assert "completed >= EXPECTED_CELLS" in monitor
    assert 'event "COMPLETE job_uuid=$JOB_UUID' in monitor
    assert "sleep \"$POLL_SECONDS\"" in monitor


def _event(gpu_id, timestamp=1_785_478_107_000):
    return {
        "timestamp": timestamp + gpu_id,
        "message": (
            "[2026-07-31T06:08:27Z] "
            f"worker_gpu={gpu_id} local_gpu=0 "
            f"CUDA_VISIBLE_DEVICES={gpu_id} python cell.py"
        ),
    }


def test_extract_worker_bindings_requires_distinct_physical_gpus():
    bindings = startup.extract_worker_bindings([_event(gpu_id) for gpu_id in range(8)])

    assert [binding["worker_gpu"] for binding in bindings] == list(range(8))
    assert [binding["cuda_visible_devices"] for binding in bindings] == list(range(8))
    assert {binding["local_gpu"] for binding in bindings} == {0}

    missing = [_event(gpu_id) for gpu_id in range(7)]
    with pytest.raises(ValueError, match="GPUs \\[7\\]"):
        startup.extract_worker_bindings(missing)

    mismatched = [_event(gpu_id) for gpu_id in range(8)]
    mismatched[-1]["message"] = mismatched[-1]["message"].replace(
        "CUDA_VISIBLE_DEVICES=7",
        "CUDA_VISIBLE_DEVICES=6",
    )
    with pytest.raises(ValueError, match="distinct physical GPUs"):
        startup.extract_worker_bindings(mismatched)


def test_summarize_progress_counts_results_and_failures():
    summary = startup.summarize_progress(
        [
            {"timestamp": 1000, "message": "[8/585] cell worker_gpu=7"},
            {"timestamp": 2000, "message": "[9/585] cell worker_gpu=0"},
        ],
        [
            {"timestamp": 1000, "message": '  "cell_id": "one",'},
            {"timestamp": 2000, "message": '  "cell_id": "two",'},
            {"timestamp": 3000, "message": "--cell-id is not a result"},
        ],
        {
            "cell failed": [{"timestamp": 3000, "message": "cell failed"}],
            "Greenland runtime failed": [],
            "Traceback": [],
        },
    )

    assert summary["scheduled_cells"] == 9
    assert summary["completed_result_lines"] == 2
    assert summary["error_counts"]["cell failed"] == 1


def test_summarize_activity_reports_latest_log_age():
    summary = startup.summarize_activity(
        [
            {"timestamp": 1000, "message": "old"},
            {"timestamp": 2500, "message": "latest training iteration"},
        ],
        observed_millis=4000,
    )

    assert summary["latest_log_at"] == "1970-01-01T00:00:02.500000Z"
    assert summary["latest_log_message"] == "latest training iteration"
    assert summary["age_seconds"] == 1.5

    missing = startup.summarize_activity([], observed_millis=4000)
    assert missing == {
        "latest_log_at": None,
        "latest_log_message": None,
        "age_seconds": None,
    }


def test_choose_log_stream_rejects_stale_matching_run():
    streams = [
        {
            "logStreamName": "timeraf-full-matrix-old",
            "firstEventTimestamp": 1000,
            "lastIngestionTime": 9000,
        },
        {
            "logStreamName": "timeraf-full-matrix-current",
            "firstEventTimestamp": 10_000,
            "lastIngestionTime": 11_000,
        },
    ]

    assert (
        startup.choose_log_stream(streams, "timeraf-full-matrix", 10_000)
        == "timeraf-full-matrix-current"
    )
