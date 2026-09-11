import json

import numpy as np
import pytest

from ts_rag.appendix_benchmark import iter_appendix_manifest
from ts_rag.appendix_rag import (
    evaluate_appendix_cell,
    load_prediction_bundles,
    save_prediction_bundles,
)


def _arrays(sample_count, seq_len, pred_len, channels):
    rng = np.random.default_rng(23)
    x = rng.normal(size=(sample_count, seq_len, channels)).astype(np.float32)
    y_base = np.repeat(x[:, -1:, :], pred_len, axis=1)
    y = y_base + 0.2
    return {
        "x": x,
        "y": y,
        "y_base": y_base,
        "x_mark": np.zeros((sample_count, seq_len, 1), dtype=np.float32),
        "y_mark": np.zeros((sample_count, pred_len, 1), dtype=np.float32),
    }


def _write_bundle(path, pred_len=96, channels=7):
    payload = {}
    for split, sample_count in (("validation", 220), ("test", 40)):
        for key, value in _arrays(
            sample_count,
            seq_len=96,
            pred_len=pred_len,
            channels=channels,
        ).items():
            payload[f"{split}_{key}"] = value
    np.savez(path, **payload)


def _cell(dataset, baseline):
    return next(
        row
        for row in iter_appendix_manifest()
        if row["dataset"] == dataset and row["baseline"] == baseline
    )


def test_external_prediction_bundle_runs_validation_selected_rag(tmp_path):
    path = tmp_path / "predictions.npz"
    _write_bundle(path)

    result, corrected = evaluate_appendix_cell(
        _cell("ETTh1", "forward_selection"),
        path,
    )

    assert result["evaluation_status"] == "completed"
    assert result["prediction_bundle_sha256"]
    assert result["all_test_metrics_improve"]
    assert corrected.shape == (40, 96, 7)


def test_prediction_bundle_rejects_misaligned_baseline_shape(tmp_path):
    path = tmp_path / "bad.npz"
    _write_bundle(path)
    with np.load(path) as archive:
        payload = {key: archive[key] for key in archive.files}
    payload["test_y_base"] = payload["test_y_base"][:, :-1]
    np.savez(path, **payload)

    with pytest.raises(ValueError, match="shapes differ"):
        load_prediction_bundles(path, task_family="long_term")


def test_prediction_bundle_rejects_split_shape_drift(tmp_path):
    path = tmp_path / "bad-splits.npz"
    _write_bundle(path)
    with np.load(path) as archive:
        payload = {key: archive[key] for key in archive.files}
    for key in ("test_x", "test_y", "test_y_base"):
        payload[key] = payload[key][..., :-1]
    np.savez(path, **payload)

    with pytest.raises(ValueError, match="Validation/test x shapes differ"):
        load_prediction_bundles(path, task_family="long_term")


def test_pems_bundle_is_already_in_reported_metric_space(tmp_path):
    path = tmp_path / "pems.npz"
    _write_bundle(path, pred_len=24, channels=358)

    validation, test = load_prediction_bundles(path, task_family="pems")

    np.testing.assert_array_equal(validation["y_inv"], validation["y"])
    np.testing.assert_array_equal(test["y_base_inv"], test["y_base"])
    np.testing.assert_array_equal(test["output_scale"], np.ones(358))


def test_writer_exports_pems_inverse_arrays(tmp_path):
    path = tmp_path / "reported-pems.npz"
    validation = _arrays(220, seq_len=96, pred_len=24, channels=3)
    test = _arrays(40, seq_len=96, pred_len=24, channels=3)
    for bundle in (validation, test):
        bundle["x_inv"] = bundle["x"] + 100.0
        bundle["y_inv"] = bundle["y"] + 100.0
        bundle["y_base_inv"] = bundle["y_base"] + 100.0

    digest = save_prediction_bundles(
        path,
        validation,
        test,
        task_family="pems",
    )
    loaded_validation, loaded_test = load_prediction_bundles(
        path,
        task_family="pems",
    )

    assert digest
    np.testing.assert_array_equal(
        loaded_validation["x"],
        validation["x_inv"],
    )
    np.testing.assert_array_equal(
        loaded_test["y_base"],
        test["y_base_inv"],
    )


def test_paper_oot_cell_records_status_without_predictions():
    result, corrected = evaluate_appendix_cell(
        _cell("traffic", "autogluon_high_quality"),
        prediction_bundle=None,
    )

    assert result["evaluation_status"] == "paper_oot"
    assert corrected is None
    assert json.dumps(result)
