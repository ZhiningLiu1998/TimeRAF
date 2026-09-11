import json
import math
from pathlib import Path

import pytest
import torch

from ts_rag.matrix import build_matrix_summary
from utils.tools import EarlyStopping

try:
    from scripts.run_benchmark_cell import (
        NumericalIntegrityError,
        _train_or_load,
        _validate_finite_result,
    )
except ImportError:
    NumericalIntegrityError = None
    _train_or_load = None
    _validate_finite_result = None


def _model_with_weight(value):
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(value)
    return model


def test_early_stopping_nan_does_not_replace_finite_checkpoint(tmp_path):
    model = _model_with_weight(1.0)
    stopping = EarlyStopping(patience=2)

    stopping(1.0, model, str(tmp_path))
    checkpoint = tmp_path / "checkpoint.pth"
    finite_checkpoint = checkpoint.read_bytes()

    with torch.no_grad():
        model.weight.fill_(2.0)
    stopping(math.nan, model, str(tmp_path))
    assert checkpoint.read_bytes() == finite_checkpoint
    assert not stopping.early_stop

    stopping(math.nan, model, str(tmp_path))
    assert checkpoint.read_bytes() == finite_checkpoint
    assert stopping.early_stop


def test_early_stopping_first_nan_does_not_create_checkpoint(tmp_path):
    stopping = EarlyStopping(patience=2)

    stopping(math.nan, _model_with_weight(1.0), str(tmp_path))

    assert not (tmp_path / "checkpoint.pth").exists()


def _cell():
    return {
        "id": "long_term/example/Model/96",
        "task_family": "long_term",
        "dataset": "example",
        "model": "Model",
        "pred_len": 96,
        "metrics": ["mse", "mae"],
    }


def _write_completed_result(tmp_path, baseline, corrected):
    artifact = tmp_path / "attempt"
    artifact.mkdir()
    (artifact / "status.json").write_text(
        json.dumps(
            {
                "cell_id": _cell()["id"],
                "seed": 2021,
                "smoke": False,
                "started_unix": 1.0,
                "status": "completed",
            }
        ),
        encoding="utf-8",
    )
    (artifact / "result.json").write_text(
        json.dumps(
            {
                "smoke": False,
                "all_test_metrics_improve": True,
                "test_baseline": baseline,
                "test_corrected": corrected,
                "validation": {"selected": {"method": "test"}},
            }
        ),
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    ("baseline", "corrected"),
    [
        ({"mse": math.nan, "mae": 1.0}, {"mse": 0.5, "mae": 0.5}),
        ({"mse": 1.0, "mae": 1.0}, {"mse": math.inf, "mae": 0.5}),
        ({"mse": 1.0, "mae": -math.inf}, {"mse": 0.5, "mae": 0.5}),
        (
            {"mse": 5e-324, "mae": 1.0},
            {"mse": 1.0, "mae": 0.5},
        ),
    ],
    ids=[
        "baseline-nan",
        "corrected-positive-inf",
        "baseline-negative-inf",
        "derived-gain-overflow",
    ],
)
def test_non_finite_matrix_result_is_not_completed(
    tmp_path,
    baseline,
    corrected,
):
    _write_completed_result(tmp_path, baseline, corrected)

    summary = build_matrix_summary(
        [_cell()],
        tmp_path,
        publication_expected_cells=1,
    )

    assert summary["counts"]["completed"] == 0
    assert summary["cell_states"][0]["state"] != "completed"
    assert not summary["all_completed"]
    assert not summary["publication_gate"]["criteria"][
        "matrix_complete_without_runtime_failures"
    ]


def test_matrix_monitor_exits_after_terminal_numerical_outcome():
    monitor = Path("scripts/timefuse_matrix_monitor.sh").read_text(
        encoding="utf-8"
    )

    assert "terminal_count=$((completed + failed + incomplete))" in monitor
    assert "COMPLETE_REQUIRES_NUMERICAL_RECOVERY" in monitor
    assert "TERMINAL_WITH_FAILURES" in monitor


@pytest.mark.skipif(
    _validate_finite_result is None,
    reason=(
        "waiting for scripts.run_benchmark_cell."
        "_validate_finite_result(result, metrics)"
    ),
)
def test_run_benchmark_cell_finite_result_validator():
    finite = {
        "test_baseline": {"mse": 1.0, "mae": 0.5},
        "test_corrected": {"mse": 0.9, "mae": 0.4},
    }
    _validate_finite_result(finite, ("mse", "mae"))

    non_finite = {
        "test_baseline": {"mse": 1.0, "mae": 0.5},
        "test_corrected": {"mse": math.nan, "mae": 0.4},
    }
    with pytest.raises(NumericalIntegrityError, match="finite"):
        _validate_finite_result(non_finite, ("mse", "mae"))


@pytest.mark.skipif(
    _train_or_load is None,
    reason="waiting for scripts.run_benchmark_cell._train_or_load",
)
def test_training_without_finite_checkpoint_is_numerical_failure(tmp_path):
    checkpoint_root = tmp_path / "checkpoints"
    runtime_args = type(
        "RuntimeArgs",
        (),
        {"checkpoints": str(checkpoint_root)},
    )()
    cli = type(
        "Cli",
        (),
        {
            "checkpoint": None,
            "force_train": False,
            "no_train": False,
            "smoke": False,
        },
    )()

    class _Experiment:
        @staticmethod
        def train(setting):
            raise FileNotFoundError(
                2,
                "No such file or directory",
                checkpoint_root / setting / "checkpoint.pth",
            )

    with pytest.raises(
        NumericalIntegrityError,
        match="no finite validation checkpoint",
    ):
        _train_or_load(_Experiment(), runtime_args, "cell", cli)
