import numpy as np

import ts_rag.appendix_zeroshot as appendix_zeroshot
from ts_rag.appendix_rag import load_prediction_bundles
from ts_rag.appendix_zeroshot import (
    build_zeroshot_ensemble_bundles,
    dataset_meta_feature_vector,
    zeroshot_weights,
)


MODEL_ORDER = ["model_a", "model_b"]
FEATURE_NAMES = ["feature_a", "feature_b"]


def _profile(name, features, scores):
    return {
        "dataset": name,
        "model_order": MODEL_ORDER,
        "meta_feature_names": FEATURE_NAMES,
        "meta_feature_values": features,
        "source_validation_performance": {
            "reciprocal_rank_scores": scores,
        },
    }


def test_meta_features_are_deterministic_and_finite():
    rng = np.random.default_rng(17)
    x = rng.normal(size=(20, 24, 5)).astype(np.float32)

    first = dataset_meta_feature_vector(x, max_samples=8, max_channels=3)
    second = dataset_meta_feature_vector(x, max_samples=8, max_channels=3)

    assert first["feature_names"] == second["feature_names"]
    np.testing.assert_array_equal(first["values"], second["values"])
    assert np.isfinite(first["values"]).all()
    assert first["sampling"]["selected_samples"] == 8
    assert first["sampling"]["selected_channels"] == 3


def test_zeroshot_excludes_target_performance():
    profiles = {
        "target": _profile("target", [0.0, 0.0], [0.0, 1.0]),
        "near": _profile("near", [0.1, 0.1], [0.9, 0.1]),
        "far": _profile("far", [10.0, 10.0], [0.1, 0.9]),
    }

    selection = zeroshot_weights("target", profiles)

    assert selection["source_datasets"] == ["far", "near"]
    assert selection["selection_uses"].startswith("target_validation_inputs")
    assert selection["weights"][0] > selection["weights"][1]
    assert np.isclose(sum(selection["weights"]), 1.0)

    profiles["target"]["source_validation_performance"][
        "reciprocal_rank_scores"
    ] = [1.0, 0.0]
    repeated = zeroshot_weights("target", profiles)
    np.testing.assert_allclose(repeated["weights"], selection["weights"])


def _synthetic_library(dataset_index):
    validation_count = 6
    test_count = 3

    def bundle(sample_count):
        x = np.full(
            (sample_count, 8, 1),
            float(dataset_index),
            dtype=np.float32,
        )
        y = np.zeros((sample_count, 2, 1), dtype=np.float32)
        return {
            "x": x,
            "y": y,
            "y_base": y.copy(),
            "x_mark": np.zeros((sample_count, 8, 1), dtype=np.float32),
            "y_mark": np.zeros((sample_count, 2, 1), dtype=np.float32),
        }

    validation = bundle(validation_count)
    test = bundle(test_count)
    validation_predictions = np.stack(
        [
            validation["y"] + dataset_index,
            validation["y"] + (2 - dataset_index),
        ]
    )
    test_predictions = np.stack(
        [
            test["y"] + dataset_index,
            test["y"] + (2 - dataset_index),
        ]
    )
    return {
        "models": MODEL_ORDER,
        "validation": validation,
        "test": test,
        "validation_predictions": validation_predictions,
        "test_predictions": test_predictions,
        "base_prediction_bundles": [],
    }


def test_builds_leave_one_task_out_bundles(monkeypatch, tmp_path):
    datasets = ["task_a", "task_b", "task_c"]
    libraries = {
        dataset: _synthetic_library(index)
        for index, dataset in enumerate(datasets)
    }
    monkeypatch.setattr(
        appendix_zeroshot,
        "_expected_family_datasets",
        lambda task_family: datasets,
    )
    monkeypatch.setattr(
        appendix_zeroshot,
        "load_base_prediction_library",
        lambda identifier, task_family: libraries[identifier],
    )

    metadata = build_zeroshot_ensemble_bundles(
        {dataset: dataset for dataset in datasets},
        task_family="long_term",
        output_root=tmp_path,
    )
    validation, test = load_prediction_bundles(
        tmp_path / "task_a" / "prediction_bundle.npz",
        "long_term",
    )

    assert sorted(metadata["outputs"]) == datasets
    assert metadata["outputs"]["task_a"]["selection"]["source_datasets"] == [
        "task_b",
        "task_c",
    ]
    assert validation["y_base"].shape == validation["y"].shape
    assert test["y_base"].shape == test["y"].shape
