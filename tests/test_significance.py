import numpy as np

from ts_rag.significance import (
    paired_moving_block_bootstrap,
    paired_multi_seed_moving_block_bootstrap,
)


def test_bootstrap_detects_uniform_improvement():
    true = np.zeros((500, 4, 2), dtype=np.float32)
    baseline = np.ones_like(true)
    candidate = np.full_like(true, 0.8)
    result = paired_moving_block_bootstrap(
        baseline,
        candidate,
        true,
        block_length=20,
        n_resamples=500,
    )
    assert result["candidate_minus_baseline"] < 0
    assert result["confidence_interval_95"][1] < 0


def test_bootstrap_rejects_unknown_loss():
    true = np.zeros((20, 2, 1), dtype=np.float32)
    try:
        paired_moving_block_bootstrap(true, true, true, loss="unknown")
    except ValueError as error:
        assert "Unsupported loss" in str(error)
    else:
        raise AssertionError("Expected an unsupported loss to raise ValueError")


def test_multi_seed_bootstrap_averages_aligned_origins():
    true = np.zeros((40, 2, 1), dtype=np.float32)
    baselines = [
        np.ones_like(true),
        np.full_like(true, 2.0),
    ]
    candidates = [
        np.full_like(true, 0.5),
        np.full_like(true, 1.0),
    ]

    result = paired_multi_seed_moving_block_bootstrap(
        baselines,
        candidates,
        [true, true],
        n_resamples=200,
        block_length=5,
    )

    assert result["model_seed_count"] == 2
    assert result["forecast_origin_count"] == 40
    assert result["candidate_minus_baseline"] < 0.0
    assert result["confidence_interval_95"][1] < 0.0
