import argparse
import copy
import json

import numpy as np
import pytest

from scripts.run_retrieval_baseline_cell import _compare_base_metrics
from scripts import run_retrieval_baseline_matrix
from scripts.run_retrieval_baseline_matrix import (
    _topology,
    _validate_scope,
    build_summary,
)
from ts_rag.matrix import load_manifest
from ts_rag.benchmark_rag import prediction_in_metric_space
from ts_rag.retrieval_baselines import (
    SYSTEM_IDS,
    _compatible_raft_periods,
    _deterministic_mmr,
    load_protocol,
    run_retrieval_baseline_cell,
    validation_partition,
)


def _bundle(sample_count, seed, pred_len=8):
    rng = np.random.default_rng(seed)
    seq_len = 24
    channels = 2
    x = rng.normal(size=(sample_count, seq_len, channels)).astype(np.float32)
    y_base = np.repeat(x[:, -1:, :], pred_len, axis=1)
    residual = (
        0.15
        + 0.05 * x[:, -1:, :]
        + rng.normal(
            scale=0.01,
            size=(sample_count, pred_len, channels),
        )
    ).astype(np.float32)
    marks = rng.normal(size=(sample_count, seq_len, 4)).astype(np.float32)
    return {
        "x": x,
        "y": y_base + residual,
        "y_base": y_base,
        "x_mark": marks,
        "y_mark": marks[:, :pred_len],
    }


def test_validation_partition_preserves_prediction_horizon_gap():
    memory, query = validation_partition(80, pred_len=8)

    assert memory[-1] + 8 < query[0]
    assert len(memory) > 0
    assert len(query) > 0


def test_reported_space_pems_bundle_is_not_inverse_transformed_again():
    bundle = _bundle(4, seed=1)
    bundle["metric_space"] = "inverse_scaled"
    prediction = bundle["y_base"]

    actual_prediction, actual_true = prediction_in_metric_space(
        bundle,
        prediction,
        "pems",
    )

    np.testing.assert_array_equal(actual_prediction, prediction)
    np.testing.assert_array_equal(actual_true, bundle["y"])


def test_deterministic_mmr_breaks_ties_by_memory_index():
    candidate_indices = np.array([[9, 3, 7, 1]])
    candidate_scores = np.array([[0.8, 0.8, 0.8, 0.8]], dtype=np.float32)

    selected, scores = _deterministic_mmr(
        candidate_indices,
        candidate_scores,
        k=3,
        lambda_value=0.5,
    )

    np.testing.assert_array_equal(selected, [[1, 3, 7]])
    np.testing.assert_allclose(scores, 0.8)


def test_all_retrieval_systems_produce_finite_results_without_lookahead():
    validation = _bundle(64, seed=2)
    test = _bundle(20, seed=3)
    protocol = load_protocol("docs/retrieval_baseline_protocol.json")

    result = run_retrieval_baseline_cell(
        validation,
        test,
        task_family="long_term",
        metric_names=("mse", "mae"),
        pred_len=8,
        protocol=protocol,
        device="cpu",
    )

    assert tuple(result["systems"]) == SYSTEM_IDS
    assert result["no_lookahead"]["retrieval_systems"]["passed"]
    assert result["no_lookahead"]["timeraf"]["passed"]
    for system in result["systems"].values():
        assert all(
            np.isfinite(value)
            for value in system["test_metrics"].values()
        )


def test_test_truth_does_not_affect_noncausal_timeraf_prediction():
    validation = _bundle(64, seed=4)
    test = _bundle(20, seed=5)
    modified = copy.deepcopy(test)
    modified["y"] += 1000.0
    protocol = load_protocol("docs/retrieval_baseline_protocol.json")

    original = run_retrieval_baseline_cell(
        validation,
        test,
        task_family="long_term",
        metric_names=("mse", "mae"),
        pred_len=8,
        protocol=protocol,
        device="cpu",
    )
    changed = run_retrieval_baseline_cell(
        validation,
        modified,
        task_family="long_term",
        metric_names=("mse", "mae"),
        pred_len=8,
        protocol=protocol,
        device="cpu",
    )

    for system_id in (
        "analog_future",
        "raft_adapted",
        "saraf_adapted",
        "residual_retrieval",
    ):
        assert (
            original["systems"][system_id]["prediction_sha256"]
            == changed["systems"][system_id]["prediction_sha256"]
        )


def test_matrix_summary_accepts_json_sorted_system_keys(tmp_path):
    cell = {
        "id": "long_term/ETTh1/Autoformer/96",
        "task_family": "long_term",
        "dataset": "ETTh1",
        "model": "Autoformer",
        "pred_len": 96,
    }
    artifact = (
        tmp_path / "cells/long_term/ETTh1/Autoformer/pl96"
    )
    artifact.mkdir(parents=True)
    (artifact / "status.json").write_text(
        json.dumps({"status": "completed"}),
        encoding="utf-8",
    )
    systems = {
        system_id: {
            "strict_win": False,
            "metric_gain_percent": {"mse": 0.0, "mae": 0.0},
        }
        for system_id in sorted(SYSTEM_IDS)
    }
    (artifact / "result.json").write_text(
        json.dumps(
            {
                "elapsed_seconds": 1.0,
                "base_metric_recomputation": {"passed": True},
                "a10g_metric_reference": {
                    "comparison_only": True,
                },
                "comparison": {
                    "systems": systems,
                    "no_lookahead": {"passed": True},
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    summary = build_summary([cell], tmp_path, identities={})

    assert summary["counts"]["completed"] == 1
    assert summary["counts"]["incomplete"] == 0
    assert summary["all_completed"]


def test_cross_hardware_base_difference_is_recorded_not_rejected():
    comparison = _compare_base_metrics(
        {"mse": 0.5021056532859802, "mae": 0.4},
        {"test_baseline": {"mse": 0.5016162395477295, "mae": 0.39}},
        ("mse", "mae"),
    )

    assert comparison["comparison_only"]
    assert not comparison["equality_required"]
    assert comparison["absolute_difference"]["mse"] > 0.0001


def test_raft_period_sets_drop_only_shape_incompatible_periods():
    assert _compatible_raft_periods([1], 96, 6) == [1]
    assert _compatible_raft_periods([4, 2, 1], 96, 6) == [2, 1]
    assert _compatible_raft_periods([4, 2, 1], 96, 12) == [4, 2, 1]
    assert _compatible_raft_periods([4, 2, 1], 96, 24) == [4, 2, 1]


def test_full_horizon_scope_requires_all_585_cells():
    rows = load_manifest("docs/timefuse_experiment_manifest.jsonl")
    protocol = load_protocol("docs/retrieval_baseline_protocol.json")
    catalog = {
        "usable_count": len(rows),
        "bundles": {
            row["id"]: {"usable": True}
            for row in rows
        },
    }

    selected = _validate_scope(catalog, rows, protocol)

    assert len(selected) == 585
    assert {
        family: sorted(
            {
                row["pred_len"]
                for row in selected
                if row["task_family"] == family
            }
        )
        for family in ("long_term", "pems", "epf")
    } == {
        "long_term": [96, 192, 336, 720],
        "pems": [6, 12, 24],
        "epf": [24],
    }


def test_p5_topology_uses_all_eight_h100_workers(monkeypatch):
    monkeypatch.setattr(
        run_retrieval_baseline_matrix.torch.cuda,
        "is_available",
        lambda: True,
    )
    monkeypatch.setattr(
        run_retrieval_baseline_matrix.torch.cuda,
        "device_count",
        lambda: 8,
    )
    monkeypatch.setattr(
        run_retrieval_baseline_matrix.subprocess,
        "run",
        lambda *args, **kwargs: argparse.Namespace(
            stdout="\n".join(f"GPU {index}: H100" for index in range(8))
        ),
    )
    args = argparse.Namespace(
        cpu=False,
        instance_type="ml.p5.48xlarge",
        reserved_gpus_per_host=8,
        processes_per_host=8,
    )

    topology = _topology(args)

    assert topology["worker_gpu_ids"] == list(range(8))
    assert topology["world_size"] == 8
    assert topology["inactive_reserved_gpus"] == 0

    args.processes_per_host = 7
    with pytest.raises(ValueError, match="one worker on every reserved GPU"):
        _topology(args)
