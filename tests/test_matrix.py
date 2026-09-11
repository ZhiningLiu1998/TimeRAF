import hashlib
import json
import signal
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import pytest

from scripts import run_benchmark_matrix
from scripts import summarize_timefuse_matrix
from ts_rag.matrix import build_matrix_summary


def _cell(cell_id, family, dataset, model, metrics):
    return {
        "id": cell_id,
        "task_family": family,
        "dataset": dataset,
        "model": model,
        "pred_len": 96,
        "metrics": metrics,
    }


def _run_cell_args(tmp_path):
    return SimpleNamespace(
        cpu=True,
        manifest="manifest.jsonl",
        seed=2021,
        source_revision="revision",
        output_root=str(tmp_path / "output"),
        checkpoint_root=str(tmp_path / "checkpoints"),
        num_workers=0,
        smoke=False,
        force_train=False,
        no_train=False,
        save_arrays=False,
        export_only=False,
        development_diagnostics=False,
        override=[],
        release_checkpoints={},
        release_checkpoint_root=None,
        log_root=str(tmp_path / "logs"),
        working_root=str(tmp_path),
    )


def _write_attempt(root, name, status, result=None):
    artifact = root / name
    artifact.mkdir(parents=True)
    (artifact / "status.json").write_text(json.dumps(status))
    if result is not None:
        (artifact / "result.json").write_text(json.dumps(result))


def test_matrix_summary_tracks_resume_and_improvement_states(tmp_path):
    manifest = [
        _cell("long/A/M/96", "long_term", "A", "M", ["mse", "mae"]),
        _cell("pems/B/M/96", "pems", "B", "M", ["mae", "rmse"]),
        _cell("epf/C/N/96", "epf", "C", "N", ["mse", "mae"]),
    ]
    _write_attempt(
        tmp_path,
        "completed",
        {
            "cell_id": "long/A/M/96",
            "seed": 2021,
            "smoke": False,
            "started_unix": 1,
            "status": "completed",
        },
        {
            "smoke": False,
            "all_test_metrics_improve": True,
            "test_baseline": {"mse": 2.0, "mae": 1.0},
            "test_corrected": {"mse": 1.0, "mae": 0.8},
            "validation": {"selected": {"method": "historical_residual"}},
        },
    )
    _write_attempt(
        tmp_path,
        "failed",
        {
            "cell_id": "pems/B/M/96",
            "seed": 2021,
            "smoke": False,
            "started_unix": 2,
            "status": "failed",
            "error_type": "ValueError",
            "error": "bad shape",
        },
    )

    summary = build_matrix_summary(manifest, tmp_path)

    assert summary["counts"]["expected"] == 3
    assert summary["counts"]["completed"] == 1
    assert summary["counts"]["improved"] == 1
    assert summary["counts"]["failed"] == 1
    assert summary["counts"]["pending"] == 1
    assert summary["groups"]["model"]["M"]["expected"] == 2
    assert summary["method_counts"] == {"historical_residual": 1}
    assert summary["best_cells"][0]["minimum_metric_gain_percent"] == pytest.approx(
        20.0
    )
    assert not summary["all_completed"]
    assert not summary["all_improved"]
    assert not summary["publication_gate"]["development_gate_passed"]
    assert not summary["publication_gate"]["criteria"][
        "matrix_complete_without_runtime_failures"
    ]


def test_publication_gate_accepts_frozen_threshold_boundaries(tmp_path):
    manifest = []
    for family, improved_count in (("long_term", 9), ("pems", 7)):
        metrics = ["mse", "mae"] if family == "long_term" else ["mae", "rmse"]
        for index in range(10):
            cell_id = f"{family}/D/M/{index}"
            manifest.append(_cell(cell_id, family, "D", "M", metrics))
            improved = index < improved_count
            baseline = {metric: 100.0 for metric in metrics}
            corrected = {
                metric: 90.0 if improved else 101.0 for metric in metrics
            }
            _write_attempt(
                tmp_path,
                f"{family}-{index}",
                {
                    "cell_id": cell_id,
                    "seed": 2021,
                    "smoke": False,
                    "started_unix": index,
                    "status": "completed",
                },
                {
                    "smoke": False,
                    "all_test_metrics_improve": improved,
                    "test_baseline": baseline,
                    "test_corrected": corrected,
                    "validation": {"selected": {"method": "test"}},
                },
            )

    gate = build_matrix_summary(
        manifest,
        tmp_path,
        publication_expected_cells=len(manifest),
    )["publication_gate"]

    assert gate["scope"]["is_full_matrix"]
    assert gate["overall"]["improvement_rate"] == pytest.approx(0.80)
    assert gate["task_families"]["long_term"][
        "improvement_rate"
    ] == pytest.approx(0.90)
    assert gate["task_families"]["pems"]["improvement_rate"] == pytest.approx(
        0.70
    )
    assert all(gate["criteria"].values())
    assert gate["development_gate_passed"]
    assert gate["confirmatory_evidence_required"]


def test_standalone_matrix_summarizer_preserves_incomplete_denominator(
    tmp_path,
):
    output = tmp_path / "publication-summary.json"

    assert summarize_timefuse_matrix.main(
        [
            "--output-root",
            str(tmp_path / "matrix"),
            "--source-revision",
            "method-revision",
            "--summary",
            str(output),
            "--require-publication-scope",
        ]
    ) == 0

    summary = json.loads(output.read_text(encoding="utf-8"))
    assert summary["counts"] == {
        "completed": 0,
        "expected": 585,
        "failed": 0,
        "improved": 0,
        "incomplete": 0,
        "not_improved": 0,
        "pending": 585,
        "running": 0,
    }
    assert not summary["publication_gate"]["development_gate_passed"]


def test_matrix_summary_does_not_mix_smoke_and_full_results(tmp_path):
    manifest = [
        _cell("long/A/M/96", "long_term", "A", "M", ["mse", "mae"])
    ]
    _write_attempt(
        tmp_path,
        "smoke",
        {
            "cell_id": "long/A/M/96",
            "seed": 2021,
            "smoke": True,
            "started_unix": 1,
            "status": "completed",
        },
        {
            "smoke": True,
            "all_test_metrics_improve": True,
            "test_baseline": {"mse": 2.0, "mae": 1.0},
            "test_corrected": {"mse": 1.0, "mae": 0.8},
            "validation": {"selected": {"method": "bias"}},
        },
    )

    full = build_matrix_summary(manifest, tmp_path, smoke=False)
    smoke = build_matrix_summary(manifest, tmp_path, smoke=True)

    assert full["counts"]["pending"] == 1
    assert smoke["counts"]["completed"] == 1


def test_matrix_summary_does_not_mix_source_revisions(tmp_path):
    manifest = [
        _cell("long/A/M/96", "long_term", "A", "M", ["mse", "mae"])
    ]
    result = {
        "smoke": False,
        "all_test_metrics_improve": True,
        "test_baseline": {"mse": 2.0, "mae": 1.0},
        "test_corrected": {"mse": 1.0, "mae": 0.8},
        "validation": {"selected": {"method": "historical_residual"}},
    }
    _write_attempt(
        tmp_path,
        "old",
        {
            "cell_id": "long/A/M/96",
            "seed": 2021,
            "source_revision": "old",
            "smoke": False,
            "started_unix": 2,
            "status": "completed",
        },
        result,
    )
    _write_attempt(
        tmp_path,
        "current",
        {
            "cell_id": "long/A/M/96",
            "seed": 2021,
            "source_revision": "current",
            "smoke": False,
            "started_unix": 1,
            "status": "completed",
        },
        result,
    )

    summary = build_matrix_summary(
        manifest,
        tmp_path,
        source_revision="current",
    )

    assert summary["source_revision"] == "current"
    assert summary["counts"]["completed"] == 1
    assert summary["cell_states"][0]["artifact_dir"].endswith("current")


def test_matrix_summary_does_not_mix_export_only_results(tmp_path):
    manifest = [
        _cell("long/A/M/96", "long_term", "A", "M", ["mse", "mae"])
    ]
    _write_attempt(
        tmp_path,
        "export",
        {
            "cell_id": "long/A/M/96",
            "seed": 2021,
            "source_revision": "current",
            "smoke": False,
            "export_only": True,
            "started_unix": 2,
            "status": "completed",
        },
        {
            "smoke": False,
            "export_only": True,
            "prediction_bundle_sha256": "digest",
        },
    )

    regular = build_matrix_summary(
        manifest,
        tmp_path,
        source_revision="current",
    )
    exported = build_matrix_summary(
        manifest,
        tmp_path,
        source_revision="current",
        export_only=True,
    )

    assert regular["counts"]["pending"] == 1
    assert exported["counts"]["completed"] == 1


def test_matrix_topology_assigns_one_distinct_worker_per_gpu(monkeypatch):
    monkeypatch.setattr(
        run_benchmark_matrix.torch.cuda,
        "is_available",
        lambda: True,
    )
    monkeypatch.setattr(
        run_benchmark_matrix.torch.cuda,
        "device_count",
        lambda: 4,
    )
    args = SimpleNamespace(
        cpu=False,
        gpu=0,
        processes_per_host=4,
        reserved_gpus_per_host=4,
        instance_type="ml.g5.12xlarge",
    )

    topology = run_benchmark_matrix._topology(args)

    assert topology["worker_gpu_ids"] == [0, 1, 2, 3]
    assert topology["world_size"] == 4
    assert topology["inactive_reserved_gpus"] == 0
    assert topology["launcher_mode"] == "independent_cell_workers"


def test_p4de_topology_requires_all_eight_worker_gpus(monkeypatch):
    monkeypatch.setattr(
        run_benchmark_matrix.torch.cuda,
        "is_available",
        lambda: True,
    )
    monkeypatch.setattr(
        run_benchmark_matrix.torch.cuda,
        "device_count",
        lambda: 8,
    )
    args = SimpleNamespace(
        cpu=False,
        gpu=0,
        processes_per_host=8,
        reserved_gpus_per_host=8,
        instance_type="ml.p4de.24xlarge",
    )

    topology = run_benchmark_matrix._topology(args)

    assert topology["worker_gpu_ids"] == list(range(8))
    assert topology["world_size"] == 8
    assert topology["inactive_reserved_gpus"] == 0

    args.processes_per_host = 4
    with pytest.raises(ValueError, match="requires eight visible"):
        run_benchmark_matrix._topology(args)

    args.cpu = True
    with pytest.raises(ValueError, match="cannot run in CPU mode"):
        run_benchmark_matrix._topology(args)


def test_p4de_requires_enough_queued_cells_for_all_workers():
    args = SimpleNamespace(
        cpu=False,
        dry_run=False,
        instance_type="ml.p4de.24xlarge",
    )

    with pytest.raises(ValueError, match="at least eight queued cells"):
        run_benchmark_matrix._validate_queued_work(args, queued_cells=7)

    run_benchmark_matrix._validate_queued_work(args, queued_cells=8)


def test_checkpoint_catalog_rehashes_every_selected_file(tmp_path):
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"checkpoint")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    cell = _cell(
        "long/A/M/96",
        "long_term",
        "A",
        "M",
        ["mse", "mae"],
    )
    catalog = {
        "source_revision": "method",
        "selected_count": 1,
        "checkpoint_count": 1,
        "usable_count": 1,
        "missing_count": 0,
        "missing_selected_count": 0,
        "hashes_included": True,
        "numerical_recovery": None,
        "numerical_recovery_cohort": None,
        "numerical_recovery_receipt": None,
        "checkpoints": {
            cell["id"]: {
                "usable": True,
                "relative_path": "checkpoint.pth",
                "file_size": checkpoint.stat().st_size,
                "sha256": digest,
                "source_revision": "method",
                "numerical_recovery": None,
                "numerical_recovery_cohort": None,
            },
        },
    }
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(json.dumps(catalog), encoding="utf-8")

    _, evidence = run_benchmark_matrix._load_and_verify_checkpoint_catalog(
        catalog_path,
        tmp_path,
        [cell],
    )

    assert evidence["selected_count"] == 1
    assert evidence["file_hashes_recomputed"]

    checkpoint.write_bytes(b"checkpoinu")
    with pytest.raises(ValueError, match="SHA-256 drifted"):
        run_benchmark_matrix._load_and_verify_checkpoint_catalog(
            catalog_path,
            tmp_path,
            [cell],
        )


def test_export_only_replay_is_forwarded_to_cells(monkeypatch, tmp_path):
    captured = {}

    class _Stdout:
        def __iter__(self):
            return iter(())

        @staticmethod
        def close():
            return None

    class _Process:
        stdout = _Stdout()

        @staticmethod
        def wait():
            return 0

    def fake_popen(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return _Process()

    monkeypatch.setattr(run_benchmark_matrix.subprocess, "Popen", fake_popen)
    args = SimpleNamespace(
        cpu=True,
        manifest="manifest.jsonl",
        seed=2021,
        source_revision="revision",
        output_root=str(tmp_path / "output"),
        checkpoint_root=str(tmp_path / "checkpoints"),
        num_workers=0,
        smoke=False,
        force_train=False,
        no_train=True,
        save_arrays=True,
        export_only=True,
        development_diagnostics=False,
        override=["learning_rate=0.0001"],
        release_checkpoints={},
        release_checkpoint_root=None,
        log_root=str(tmp_path / "logs"),
        working_root=str(tmp_path),
    )

    return_code = run_benchmark_matrix._run_cell(
        _cell("long/A/M/96", "long_term", "A", "M", ["mse", "mae"]),
        args,
        gpu_id=None,
    )

    assert return_code == 0
    assert "--no-train" in captured["command"]
    assert "--save-arrays" in captured["command"]
    assert "--export-only" in captured["command"]
    assert captured["command"][1] == str(
        Path(run_benchmark_matrix.__file__).resolve().parent
        / "run_benchmark_cell.py"
    )
    assert captured["kwargs"]["cwd"] == str(tmp_path)
    assert captured["kwargs"]["start_new_session"] is True
    override_index = captured["command"].index("--override")
    assert captured["command"][override_index + 1] == "learning_rate=0.0001"


def test_run_cell_ignores_broken_stdout_and_waits_for_child(
    monkeypatch,
    tmp_path,
):
    events = []

    class _Stdout:
        def __iter__(self):
            yield "cell output\n"

        def close(self):
            events.append("stdout-closed")

    class _Process:
        pid = 123
        stdout = _Stdout()

        @staticmethod
        def wait():
            events.append("waited")
            return 0

    monkeypatch.setattr(
        run_benchmark_matrix.subprocess,
        "Popen",
        lambda *args, **kwargs: _Process(),
    )
    monkeypatch.setattr(
        run_benchmark_matrix,
        "_mirror_stdout",
        lambda text: events.append("mirror") or False,
    )
    args = _run_cell_args(tmp_path)

    return_code = run_benchmark_matrix._run_cell(
        _cell("long/A/M/96", "long_term", "A", "M", ["mse", "mae"]),
        args,
        gpu_id=None,
    )

    assert return_code == 0
    assert events == ["mirror", "waited", "stdout-closed"]
    assert "cell output\n" in (
        tmp_path / "logs" / "long__A__M__96.log"
    ).read_text()


def test_stdout_mirror_treats_broken_pipe_as_nonfatal(monkeypatch):
    def broken_print(*args, **kwargs):
        raise BrokenPipeError

    monkeypatch.setattr("builtins.print", broken_print)

    assert run_benchmark_matrix._mirror_stdout("output\n") is False


def test_terminate_process_group_escalates_and_reaps(monkeypatch):
    events = []
    process_group_alive = True

    class _Process:
        pid = 2468

        @staticmethod
        def poll():
            events.append("poll")
            return None

        @staticmethod
        def wait():
            events.append("wait")
            return -signal.SIGKILL

    def fake_killpg(process_group_id, sent_signal):
        nonlocal process_group_alive
        assert process_group_id == _Process.pid
        if sent_signal == 0:
            if not process_group_alive:
                raise ProcessLookupError
            return
        events.append(sent_signal)
        if sent_signal == signal.SIGKILL:
            process_group_alive = False

    monkeypatch.setattr(run_benchmark_matrix.os, "killpg", fake_killpg)

    run_benchmark_matrix._terminate_process_group(
        _Process(),
        terminate_timeout=0,
    )

    assert signal.SIGTERM in events
    assert signal.SIGKILL in events
    assert events[-1] == "wait"


def test_run_cell_reaps_process_group_before_reader_error_returns_worker(
    monkeypatch,
    tmp_path,
):
    events = []
    cleanup_started = Event()
    allow_child_exit = Event()

    class _BrokenStdout:
        def __iter__(self):
            raise OSError("reader failed")

        def close(self):
            events.append("stdout-closed")

    class _Process:
        pid = 5678
        stdout = _BrokenStdout()
        alive = True

    process = _Process()

    def fake_terminate(received_process):
        assert received_process is process
        events.append("cleanup-start")
        cleanup_started.set()
        assert allow_child_exit.wait(timeout=1)
        process.alive = False
        events.append("child-exited")

    monkeypatch.setattr(
        run_benchmark_matrix.subprocess,
        "Popen",
        lambda *args, **kwargs: process,
    )
    monkeypatch.setattr(
        run_benchmark_matrix,
        "_terminate_process_group",
        fake_terminate,
    )
    args = _run_cell_args(tmp_path)

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            run_benchmark_matrix._run_cell,
            _cell("long/A/M/96", "long_term", "A", "M", ["mse", "mae"]),
            args,
            None,
        )
        assert cleanup_started.wait(timeout=1)
        assert process.alive is True
        assert future.done() is False
        allow_child_exit.set()
        with pytest.raises(OSError, match="reader failed"):
            future.result(timeout=1)

    assert events == [
        "cleanup-start",
        "child-exited",
        "stdout-closed",
    ]


def test_run_cell_reaps_process_group_after_log_error(
    monkeypatch,
    tmp_path,
):
    events = []

    class _Log:
        write_count = 0

        def __enter__(self):
            return self

        @staticmethod
        def __exit__(*args):
            return False

        def write(self, text):
            self.write_count += 1
            if self.write_count > 1:
                raise OSError("log write failed")

        @staticmethod
        def flush():
            return None

    class _Stdout:
        def __iter__(self):
            yield "cell output\n"

        @staticmethod
        def close():
            events.append("stdout-closed")

    class _Process:
        pid = 6789
        stdout = _Stdout()

    process = _Process()
    monkeypatch.setattr(Path, "open", lambda *args, **kwargs: _Log())
    monkeypatch.setattr(
        run_benchmark_matrix.subprocess,
        "Popen",
        lambda *args, **kwargs: process,
    )
    monkeypatch.setattr(
        run_benchmark_matrix,
        "_terminate_process_group",
        lambda received_process: events.append(
            ("reaped", received_process.pid)
        ),
    )

    with pytest.raises(OSError, match="log write failed"):
        run_benchmark_matrix._run_cell(
            _cell("long/A/M/96", "long_term", "A", "M", ["mse", "mae"]),
            _run_cell_args(tmp_path),
            gpu_id=None,
        )

    assert events == [("reaped", process.pid), "stdout-closed"]


def test_matrix_topology_rejects_assignments_past_visible_gpus(monkeypatch):
    monkeypatch.setattr(
        run_benchmark_matrix.torch.cuda,
        "is_available",
        lambda: True,
    )
    monkeypatch.setattr(
        run_benchmark_matrix.torch.cuda,
        "device_count",
        lambda: 4,
    )
    args = SimpleNamespace(
        cpu=False,
        gpu=2,
        processes_per_host=4,
        reserved_gpus_per_host=4,
        instance_type="ml.g5.12xlarge",
    )

    with pytest.raises(ValueError, match="visible CUDA devices"):
        run_benchmark_matrix._topology(args)


def test_gpu_worker_exposes_only_assigned_physical_gpu(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")

    local_gpu_id, environment = run_benchmark_matrix._worker_process_config(
        cpu=False,
        gpu_id=2,
    )

    assert local_gpu_id == 0
    assert environment["CUDA_VISIBLE_DEVICES"] == "2"


def test_cpu_worker_does_not_require_gpu_assignment(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    local_gpu_id, environment = run_benchmark_matrix._worker_process_config(
        cpu=True,
        gpu_id=None,
    )

    assert local_gpu_id is None
    assert "CUDA_VISIBLE_DEVICES" not in environment
