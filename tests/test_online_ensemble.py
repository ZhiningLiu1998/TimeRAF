import numpy as np

from ts_rag.online_ensemble import (
    OnlineEnsembleConfig,
    apply_online_ensemble,
    build_online_statistics,
    causal_online_weights,
    overlapping_forecast_ensemble,
)


def _synthetic_bundle():
    y_base = np.zeros((5, 3, 1), dtype=np.float32)
    y = np.arange(15, dtype=np.float32).reshape(5, 3, 1)
    return {"y_base": y_base, "y": y}


def _auxiliary(bundle):
    shape = bundle["y"].shape
    return (
        np.ones(shape, dtype=np.float32),
        np.full(shape, 2.0, dtype=np.float32),
        np.full(shape, -1.0, dtype=np.float32),
    )


def test_online_weights_do_not_read_unobserved_targets():
    bundle = _synthetic_bundle()
    auxiliary = _auxiliary(bundle)
    origins = np.arange(len(bundle["y"]))
    config = OnlineEnsembleConfig(4, 0.1, (0.5, 0.25, 0.1))

    original = build_online_statistics([bundle], [auxiliary], [origins])
    original_weights = causal_online_weights(original, np.array([2]), config)

    modified_bundle = {key: value.copy() for key, value in bundle.items()}
    target_times = origins[:, None] + np.arange(bundle["y"].shape[1])[None, :]
    modified_bundle["y"][target_times >= 2] += 10_000.0
    modified = build_online_statistics(
        [modified_bundle],
        [auxiliary],
        [origins],
    )
    modified_weights = causal_online_weights(modified, np.array([2]), config)

    np.testing.assert_allclose(original_weights, modified_weights)


def test_horizon_block_weights_apply_to_their_own_steps():
    bundle = _synthetic_bundle()
    auxiliary = _auxiliary(bundle)
    weights = np.zeros((5, 3, 1, 3), dtype=np.float64)
    weights[:, 0, :, 0] = 1.0
    weights[:, 1, :, 1] = 1.0
    weights[:, 2, :, 2] = 1.0

    corrected = apply_online_ensemble(bundle, auxiliary, weights)

    np.testing.assert_allclose(corrected[:, 0], 1.0)
    np.testing.assert_allclose(corrected[:, 1], 2.0)
    np.testing.assert_allclose(corrected[:, 2], -1.0)


def test_overlapping_ensemble_uses_only_earlier_origins():
    base = np.arange(5 * 3, dtype=np.float32).reshape(5, 3, 1)
    original = overlapping_forecast_ensemble(base, max_age=2)
    modified = base.copy()
    modified[3:] += 10_000.0

    changed = overlapping_forecast_ensemble(modified, max_age=2)

    np.testing.assert_allclose(original[:3], changed[:3])
