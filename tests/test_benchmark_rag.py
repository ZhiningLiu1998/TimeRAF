import numpy as np

from ts_rag.benchmark_rag import (
    _choose_best,
    _origin_layout,
    build_causal_bias_statistics,
    causal_residual_bias,
    evaluate_method_leaders,
    prediction_in_metric_space,
    run_validation_selected_rag,
)


def _bundle(sample_count=80, seq_len=24, pred_len=6, channels=2):
    rng = np.random.default_rng(17)
    x = rng.normal(size=(sample_count, seq_len, channels)).astype(np.float32)
    y_base = np.repeat(x[:, -1:, :], pred_len, axis=1)
    residual = 0.2 + rng.normal(
        scale=0.01, size=(sample_count, pred_len, channels)
    ).astype(np.float32)
    y = y_base + residual
    marks = np.zeros((sample_count, seq_len, 1), dtype=np.float32)
    return {
        "x": x,
        "x_mark": marks,
        "y_mark": marks[:, :pred_len],
        "y": y,
        "y_base": y_base,
        "output_scale": np.ones(channels),
        "output_mean": np.zeros(channels),
        "y_inv": y,
    }


def test_pems_metric_space_uses_inverse_scaler():
    bundle = _bundle(sample_count=2)
    bundle["output_scale"] = np.array([10.0, 20.0])
    bundle["output_mean"] = np.array([3.0, 4.0])
    bundle["y_inv"] = (
        bundle["y"] * bundle["output_scale"][None, None, :]
        + bundle["output_mean"][None, None, :]
    )

    prediction, true = prediction_in_metric_space(
        bundle, bundle["y_base"], "pems"
    )

    np.testing.assert_allclose(
        prediction,
        bundle["y_base"] * bundle["output_scale"][None, None, :]
        + bundle["output_mean"][None, None, :],
    )
    np.testing.assert_allclose(true, bundle["y_inv"])


def test_validation_selected_rag_can_learn_constant_residual_bias():
    validation = _bundle()
    test = _bundle()

    result, corrected = run_validation_selected_rag(
        validation,
        test,
        task_family="long_term",
        metric_names=("mse", "mae"),
        pred_len=6,
    )

    assert corrected.shape == test["y_base"].shape
    assert result["validation"]["selected"]["method"] != "identity"
    assert result["all_test_metrics_improve"]


def test_causal_bias_does_not_read_unobserved_targets():
    bundle = _bundle(sample_count=20)
    origins = np.arange(len(bundle["y"]))
    query_origin = np.array([10])
    original = build_causal_bias_statistics([bundle], [origins])
    original_bias = causal_residual_bias(
        original, query_origin, pred_len=6, window=12
    )

    modified = {
        key: value.copy() if isinstance(value, np.ndarray) else value
        for key, value in bundle.items()
    }
    target_times = origins[:, None] + np.arange(6)[None, :]
    modified["y"][target_times >= query_origin[0]] += 10_000.0
    changed = build_causal_bias_statistics([modified], [origins])
    changed_bias = causal_residual_bias(
        changed, query_origin, pred_len=6, window=12
    )

    np.testing.assert_allclose(original_bias, changed_bias)


def test_pems_origin_layout_accounts_for_fresh_test_context():
    validation = _bundle(sample_count=20, seq_len=96, pred_len=24)
    test = _bundle(sample_count=3, seq_len=96, pred_len=24)

    validation_origins, test_origins = _origin_layout(
        validation,
        test,
        task_family="pems",
        pred_len=24,
    )

    assert validation_origins[-1] == 19
    np.testing.assert_array_equal(test_origins, [139, 151, 163])


def test_overlapping_split_origin_layout_reuses_context_history():
    validation = _bundle(sample_count=20, seq_len=96, pred_len=24)
    test = _bundle(sample_count=3, seq_len=96, pred_len=24)

    _, test_origins = _origin_layout(
        validation,
        test,
        task_family="long_term",
        pred_len=24,
    )

    np.testing.assert_array_equal(test_origins, [43, 44, 45])


def test_development_diagnostics_do_not_change_selection():
    validation = _bundle()
    test = _bundle()
    result, corrected = run_validation_selected_rag(
        validation,
        test,
        task_family="long_term",
        metric_names=("mse", "mae"),
        pred_len=6,
    )
    selected_before = result["validation"]["selected"].copy()

    diagnostics = evaluate_method_leaders(
        validation,
        test,
        result["validation"],
        task_family="long_term",
        metric_names=("mse", "mae"),
        pred_len=6,
    )

    assert diagnostics
    assert result["validation"]["selected"] == selected_before
    assert corrected.shape == test["y"].shape


def _candidate(method, score, improves=True, params=None):
    return {
        "method": method,
        "params": params or {},
        "metrics": {},
        "selection_score": score,
        "all_validation_metrics_improve": improves,
    }


def test_validation_argmin_selects_weakest_worst_relative_metric():
    selected = _choose_best(
        [
            _candidate("seasonal_blend", 0.985),
            _candidate("overlap_blend", 0.992),
            _candidate("identity", 1.0, improves=False),
        ]
    )

    assert selected["method"] == "seasonal_blend"


def test_validation_argmin_keeps_strong_validation_winner():
    selected = _choose_best(
        [
            _candidate("historical_residual", 0.9803),
            _candidate("overlap_blend", 0.99),
        ]
    )

    assert selected["method"] == "historical_residual"


def test_validation_argmin_uses_score_not_redundant_improvement_flag():
    selected = _choose_best(
        [
            _candidate("historical_residual", 0.987),
            _candidate("overlap_blend", 1.001, improves=False),
            _candidate("seasonal_blend", 0.9995),
        ]
    )

    assert selected["method"] == "historical_residual"


def test_validation_argmin_prefers_identity_on_an_exact_tie():
    selected = _choose_best(
        [
            _candidate("seasonal_blend", 1.0),
            _candidate("identity", 1.0, improves=False),
        ]
    )

    assert selected["method"] == "identity"


def test_validation_argmin_ties_are_independent_of_candidate_order():
    candidates = [
        _candidate("seasonal_blend", 0.99, params={"alpha": 0.1}),
        _candidate("historical_residual", 0.99, params={"alpha": 0.2}),
        _candidate("historical_residual", 0.99, params={"alpha": 0.1}),
    ]

    selected = _choose_best(candidates)
    reversed_selected = _choose_best(list(reversed(candidates)))

    assert selected["method"] == "historical_residual"
    assert selected["params"] == {"alpha": 0.1}
    assert reversed_selected == selected
