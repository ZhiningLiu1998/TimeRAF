"""Identity and legality tests for the unified residual/drift retrieval family.

The unified family is defined by

    prediction = base + alpha * (residual_correction + beta * ramp * drift)

with drift completing the residual into an endpoint-aligned analogue transfer.
Both end points of the beta dial must reproduce an independently computed
reference: beta = 0 is the frozen historical residual family, and
beta = 1 with alpha = 1, a flat ramp and no shrinkage is the analogue mean.
"""

import numpy as np
import pytest

from ts_rag.benchmark_rag import (
    PortfolioConfig,
    apply_selected_rag,
    search_validation_rag,
)
from ts_rag.historical_retrieval import (
    HistoricalResidualIndex,
    RetrievalFeatureConfig,
)


def _bundle(sample_count=64, seq_len=32, pred_len=8, channels=3, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(sample_count, seq_len, channels)).astype(np.float32)
    x = np.cumsum(x, axis=1)
    y = (
        x[:, -1:, :]
        + rng.normal(size=(sample_count, pred_len, channels)).astype(np.float32)
    )
    y_base = y + 0.4 * rng.normal(
        size=(sample_count, pred_len, channels)
    ).astype(np.float32)
    return {
        "x": x.astype(np.float32),
        "y": y.astype(np.float32),
        "y_base": y_base.astype(np.float32),
        "x_mark": rng.normal(size=(sample_count, seq_len, 4)).astype(np.float32),
        "y_mark": rng.normal(size=(sample_count, pred_len, 4)).astype(np.float32),
    }


def _feature_config(**overrides):
    params = {
        "tail_len": 16,
        "pooled_steps": 8,
        "include_differences": True,
        "include_forecast": False,
        "include_calendar": False,
        "max_channels": 16,
    }
    params.update(overrides)
    return RetrievalFeatureConfig(**params)


def _selected(method, config, **params):
    payload = {
        "feature_config": config.to_dict(),
        "k": 4,
        "temperature": 0.1,
        "shrinkage": 0.0,
    }
    payload.update(params)
    return {"method": method, "params": payload}


def test_beta_zero_matches_historical_residual():
    validation, test = _bundle(seed=1), _bundle(seed=2)
    config = _feature_config()
    residual = apply_selected_rag(
        validation,
        test,
        _selected("historical_residual", config, alpha=0.3),
        "long_term",
        test["y"].shape[1],
    )
    drift = apply_selected_rag(
        validation,
        test,
        _selected(
            "residual_drift", config, alpha=0.3, beta=0.0, ramp="flat"
        ),
        "long_term",
        test["y"].shape[1],
    )
    np.testing.assert_allclose(residual, drift, rtol=0, atol=0)


def test_beta_one_matches_endpoint_aligned_analogue_mean():
    validation, test = _bundle(seed=3), _bundle(seed=4)
    config = _feature_config()
    prediction = apply_selected_rag(
        validation,
        test,
        _selected(
            "residual_drift", config, alpha=1.0, beta=1.0, ramp="flat"
        ),
        "long_term",
        test["y"].shape[1],
    )

    index = HistoricalResidualIndex(config).fit(validation)
    neighbors, scores = index.query_neighbors(test, max_k=4)
    weights = index.neighbor_weights(scores, 4, 0.1)
    aligned = (
        validation["y"][neighbors]
        - validation["x"][neighbors][:, :, -1:, :]
        + test["x"][:, None, -1:, :]
    )
    expected = np.einsum("nk,nkpc->npc", weights, aligned)

    np.testing.assert_allclose(prediction, expected, rtol=1e-5, atol=1e-5)


def test_linear_ramp_scales_only_the_drift_term():
    validation, test = _bundle(seed=5), _bundle(seed=6)
    config = _feature_config()
    pred_len = test["y"].shape[1]
    flat = apply_selected_rag(
        validation,
        test,
        _selected("residual_drift", config, alpha=1.0, beta=1.0, ramp="flat"),
        "long_term",
        pred_len,
    )
    ramped = apply_selected_rag(
        validation,
        test,
        _selected("residual_drift", config, alpha=1.0, beta=1.0, ramp="linear"),
        "long_term",
        pred_len,
    )
    residual_only = apply_selected_rag(
        validation,
        test,
        _selected("residual_drift", config, alpha=1.0, beta=0.0, ramp="flat"),
        "long_term",
        pred_len,
    )
    ramp = (np.arange(1, pred_len + 1, dtype=np.float32) / pred_len).reshape(
        1, pred_len, 1
    )
    expected = residual_only + ramp * (flat - residual_only)
    np.testing.assert_allclose(ramped, expected, rtol=1e-5, atol=1e-5)


def test_drift_prediction_ignores_test_labels():
    validation, test = _bundle(seed=7), _bundle(seed=8)
    config = _feature_config()
    selected = _selected(
        "residual_drift", config, alpha=0.5, beta=0.5, ramp="linear"
    )
    reference = apply_selected_rag(
        validation, test, selected, "long_term", test["y"].shape[1]
    )
    perturbed = dict(test)
    perturbed["y"] = test["y"] + 12345.0
    repeated = apply_selected_rag(
        validation, perturbed, selected, "long_term", test["y"].shape[1]
    )
    np.testing.assert_array_equal(reference, repeated)


def test_shape_descriptor_drops_level_features():
    bundle = _bundle(seed=9)
    from ts_rag.historical_retrieval import extract_retrieval_features

    with_level = extract_retrieval_features(bundle, _feature_config())
    without_level = extract_retrieval_features(
        bundle, _feature_config(include_level_features=False)
    )
    assert without_level.shape[1] < with_level.shape[1]
    assert without_level.shape[1] == with_level.shape[1] - 4 * bundle["x"].shape[2]


def test_default_portfolio_matches_frozen_candidate_set():
    bundle = _bundle(seed=10)
    pred_len = bundle["y"].shape[1]
    frozen = search_validation_rag(bundle, "long_term", ("mse", "mae"), pred_len)
    assert not any(
        row["method"] == "residual_drift" for row in frozen["candidates"]
    )
    assert frozen["selection_policy"] == "validation_argmin_v3"

    extended = search_validation_rag(
        bundle,
        "long_term",
        ("mse", "mae"),
        pred_len,
        portfolio=PortfolioConfig(enable_drift=True),
    )
    frozen_rows = {
        (row["method"], repr(row["params"])) for row in frozen["candidates"]
    }
    extended_rows = {
        (row["method"], repr(row["params"])) for row in extended["candidates"]
    }
    assert frozen_rows < extended_rows
    assert extended["selected"]["selection_score"] <= (
        frozen["selected"]["selection_score"]
    )


def test_conservative_tolerance_prefers_the_smaller_correction():
    rows = [
        {"method": "identity", "params": {}, "selection_score": 1.0},
        {
            "method": "residual_drift",
            "params": {"alpha": 1.0, "beta": 1.0},
            "selection_score": 0.900,
        },
        {
            "method": "historical_residual",
            "params": {"alpha": 0.2},
            "selection_score": 0.904,
        },
    ]
    from ts_rag.benchmark_rag import _choose_best

    assert _choose_best(rows)["method"] == "residual_drift"
    conservative = _choose_best(rows, conservative_tolerance=0.05)
    assert conservative["method"] == "historical_residual"
    assert _choose_best(rows, conservative_tolerance=0.01)["method"] == (
        "residual_drift"
    )


def test_unsupported_ramp_is_rejected():
    from ts_rag.benchmark_rag import _drift_ramp

    with pytest.raises(ValueError):
        _drift_ramp("quadratic", 8)


def test_selection_policy_reproduces_argmin_by_default():
    rows = [
        {
            "method": "identity",
            "params": {},
            "selection_score": 1.0,
            "fold_selection_score": 1.0,
        },
        {
            "method": "historical_residual",
            "params": {"alpha": 0.2},
            "selection_score": 0.95,
            "fold_selection_score": 0.99,
        },
        {
            "method": "residual_drift",
            "params": {"alpha": 1.0, "beta": 1.0, "ramp": "flat", "shrinkage": 0.0},
            "selection_score": 0.90,
            "fold_selection_score": 1.02,
        },
    ]
    from ts_rag.benchmark_rag import SelectionPolicy, selection_policy_from_dict

    assert SelectionPolicy().select(rows)["method"] == "residual_drift"
    folded = SelectionPolicy(fold_selection="minimax2").select(rows)
    assert folded["method"] == "historical_residual"

    policy = selection_policy_from_dict(
        {
            "fold_selection": "none",
            "horizon_conditional_prior": {
                "threshold": 24,
                "margin_at_or_below": 0.0,
                "margin_above": 1.0,
            },
        }
    )
    assert policy.select(rows, pred_len=24)["method"] == "residual_drift"
    assert policy.select(rows, pred_len=96)["method"] == "historical_residual"


def test_family_prior_margin_requires_clear_drift_evidence():
    from ts_rag.benchmark_rag import SelectionPolicy

    rows = [
        {
            "method": "historical_residual",
            "params": {"alpha": 0.2},
            "selection_score": 0.950,
            "fold_selection_score": 0.950,
        },
        {
            "method": "residual_drift",
            "params": {"alpha": 0.3, "beta": 0.5, "ramp": "flat", "shrinkage": 0.0},
            "selection_score": 0.930,
            "fold_selection_score": 0.930,
        },
    ]
    assert SelectionPolicy().select(rows)["method"] == "residual_drift"
    assert (
        SelectionPolicy(family_prior_margin=0.05).select(rows)["method"]
        == "historical_residual"
    )
    assert (
        SelectionPolicy(family_prior_margin=0.01).select(rows)["method"]
        == "residual_drift"
    )
