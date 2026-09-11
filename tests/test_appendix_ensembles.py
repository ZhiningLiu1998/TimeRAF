import numpy as np

from ts_rag.appendix_ensembles import (
    apply_static_weights,
    build_advanced_ensemble_bundle,
    forward_selection_weights,
    portfolio_weights,
)
from ts_rag.appendix_rag import load_prediction_bundles, save_prediction_bundles
from ts_rag.benchmark import MODELS


def _library():
    truth = np.zeros((8, 2, 1), dtype=np.float32)
    best = np.zeros_like(truth)
    biased = np.full_like(truth, 2.0)
    return np.stack([best, biased]), truth


def test_forward_selection_uses_validation_only():
    predictions, truth = _library()

    selection = forward_selection_weights(
        predictions,
        truth,
        task_family="long_term",
        ensemble_size=5,
    )

    assert selection["counts"] == [5, 0]
    assert selection["weights"] == [1.0, 0.0]
    assert selection["selection_uses"] == "validation_labels_only"


def test_portfolio_selects_validation_ranked_prefix():
    truth = np.zeros((8, 2, 1), dtype=np.float32)
    predictions = np.stack(
        [
            np.full_like(truth, 1.0),
            np.full_like(truth, -1.0),
            np.full_like(truth, 4.0),
        ]
    )

    selection = portfolio_weights(
        predictions,
        truth,
        task_family="long_term",
    )

    assert selection["selected_subset_size"] == 2
    assert selection["selected_model_indices"] == [0, 1]
    np.testing.assert_allclose(selection["weights"], [0.5, 0.5, 0.0])


def test_static_weights_apply_without_truth():
    predictions, _ = _library()

    combined = apply_static_weights(predictions, [0.25, 0.75])

    np.testing.assert_allclose(combined, 1.5)


def _write_model_bundle(path, offset):
    payload = {}
    for split, sample_count in (("validation", 8), ("test", 4)):
        x = np.arange(sample_count * 6, dtype=np.float32).reshape(
            sample_count,
            3,
            2,
        )
        truth = np.zeros((sample_count, 2, 2), dtype=np.float32)
        bundle = {
            "x": x,
            "y": truth,
            "y_base": truth + offset,
            "x_mark": np.zeros((sample_count, 3, 1), dtype=np.float32),
            "y_mark": np.zeros((sample_count, 2, 1), dtype=np.float32),
        }
        payload[split] = bundle
    save_prediction_bundles(
        path,
        payload["validation"],
        payload["test"],
        task_family="long_term",
        already_reported=True,
    )


def test_advanced_ensemble_bundle_preserves_aligned_arrays(tmp_path):
    base_bundles = {}
    for index, model in enumerate(MODELS):
        path = tmp_path / f"{model}.npz"
        _write_model_bundle(path, offset=float(index))
        base_bundles[model] = path
    output = tmp_path / "forward.npz"

    metadata = build_advanced_ensemble_bundle(
        base_bundles,
        task_family="long_term",
        method="forward_selection",
        output_path=output,
        forward_ensemble_size=3,
    )
    validation, test = load_prediction_bundles(output, "long_term")

    assert metadata["selection"]["weights"][0] == 1.0
    np.testing.assert_array_equal(validation["y_base"], validation["y"])
    np.testing.assert_array_equal(test["y_base"], test["y"])
