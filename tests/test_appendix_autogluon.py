import errno
import os

import numpy as np
import pandas as pd
import pytest

from ts_rag.appendix_autogluon import (
    AUTOGLOUON_TIMESERIES_VERSION,
    _close_staged_arrays,
    _remove_staging_tree,
    autogluon_checkpoint_compatibility,
    autogluon_fit_recipe,
    autogluon_frequency,
    export_autogluon_prediction_bundle,
    extract_window_batch,
    make_context_frame,
    make_panel_frame,
    require_autogluon_timeseries_version,
    training_target_matrix,
    trusted_autogluon_checkpoint_loads,
    unpack_mean_predictions,
)


class _Scaler:
    n_features_in_ = 2
    scale_ = np.asarray([10.0, 20.0])
    mean_ = np.asarray([100.0, 200.0])

    def inverse_transform(self, values):
        return values * self.scale_ + self.mean_


class _Dataset:
    def __init__(self):
        self.scaler = _Scaler()
        self.data_x = np.arange(24, dtype=np.float32).reshape(12, 2)

    def __len__(self):
        return 3

    def __getitem__(self, index):
        x = np.arange(8, dtype=np.float32).reshape(4, 2) + index * 100
        y = np.arange(8, dtype=np.float32).reshape(4, 2) + index * 1000
        x_mark = np.arange(4, dtype=np.float32)[:, None]
        y_mark = np.arange(4, dtype=np.float32)[:, None] + 10
        return x, y, x_mark, y_mark

    def inverse_transform(self, values):
        return self.scaler.inverse_transform(values)


def _cell(family="long_term", features="M", output_variates=2):
    return {
        "id": "appendix/test",
        "dataset": "ETTm1",
        "task_family": family,
        "baseline": "chronos_bolt_zeroshot",
        "seq_len": 4,
        "pred_len": 2,
        "protocol": {
            "features": features,
            "output_variates": output_variates,
            "data_path": "fake.csv",
        },
    }


def test_versioned_fit_recipes_match_paper_systems():
    high_quality = autogluon_fit_recipe("autogluon_high_quality")
    zeroshot = autogluon_fit_recipe("chronos_bolt_zeroshot")
    finetuned = autogluon_fit_recipe(
        "chronos_bolt_finetuned",
        fine_tune_steps=77,
        inference_batch_size=11,
        fine_tune_batch_size=13,
    )

    assert AUTOGLOUON_TIMESERIES_VERSION == "1.4.0"
    assert high_quality["presets"] == "high_quality"
    assert zeroshot["fit_kwargs"]["hyperparameters"]["Chronos"] == {
        "model_path": "bolt_base",
        "batch_size": 32,
    }
    chronos = finetuned["fit_kwargs"]["hyperparameters"]["Chronos"]
    assert chronos["model_path"] == "bolt_base"
    assert chronos["fine_tune"] is True
    assert chronos["fine_tune_steps"] == 77
    assert chronos["batch_size"] == 11
    assert chronos["fine_tune_batch_size"] == 13


def test_runtime_preflight_requires_exact_autogluon_version(monkeypatch):
    import ts_rag.appendix_autogluon as exporter

    monkeypatch.setattr(
        exporter.importlib.metadata,
        "version",
        lambda package: "1.4.0",
    )
    assert require_autogluon_timeseries_version() == "1.4.0"

    monkeypatch.setattr(
        exporter.importlib.metadata,
        "version",
        lambda package: "1.5.0",
    )
    with pytest.raises(RuntimeError, match="version drift"):
        require_autogluon_timeseries_version()


def test_trusted_checkpoint_compatibility_is_scoped_and_restored(monkeypatch):
    variable = "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"
    monkeypatch.delenv(variable, raising=False)

    with trusted_autogluon_checkpoint_loads(
        "chronos_bolt_zeroshot"
    ) as disabled:
        assert not disabled["enabled"]
        assert variable not in os.environ

    with trusted_autogluon_checkpoint_loads(
        "autogluon_high_quality"
    ) as enabled:
        assert enabled == autogluon_checkpoint_compatibility(
            "autogluon_high_quality"
        )
        assert os.environ[variable] == "1"

    assert variable not in os.environ


def test_trusted_checkpoint_compatibility_rejects_conflicting_force(
    monkeypatch,
):
    monkeypatch.setenv("TORCH_FORCE_WEIGHTS_ONLY_LOAD", "1")
    with pytest.raises(RuntimeError, match="Conflicting"):
        with trusted_autogluon_checkpoint_loads(
            "autogluon_high_quality"
        ):
            pass


def test_export_preflight_runs_before_dataset_or_output_work(
    tmp_path,
    monkeypatch,
):
    import ts_rag.appendix_autogluon as exporter

    def reject_version():
        raise RuntimeError("version drift")

    def reject_dataset_work(*args, **kwargs):
        raise AssertionError("dataset loading must not start")

    monkeypatch.setattr(
        exporter,
        "require_autogluon_timeseries_version",
        reject_version,
    )
    monkeypatch.setattr(
        exporter,
        "load_tslib_dataset",
        reject_dataset_work,
    )
    output = tmp_path / "bundle.npz"

    with pytest.raises(RuntimeError, match="version drift"):
        export_autogluon_prediction_bundle(
            {
                **_cell(),
                "paper_status": {
                    "evaluable": True,
                    "status": "reported",
                },
            },
            output,
            predictor_path=tmp_path / "predictor",
        )

    assert not output.exists()


def test_exporter_cli_preflights_before_writing_status(tmp_path, monkeypatch):
    from scripts import export_appendix_autogluon_bundle as cli

    def reject_version():
        raise RuntimeError("version drift")

    output = tmp_path / "bundle.npz"
    status = tmp_path / "status.json"
    monkeypatch.setattr(
        cli,
        "require_autogluon_timeseries_version",
        reject_version,
    )
    monkeypatch.setattr(
        cli.sys,
        "argv",
        [
            "export_appendix_autogluon_bundle.py",
            "--cell-id",
            "unused",
            "--output",
            str(output),
            "--predictor-path",
            str(tmp_path / "predictor"),
            "--status",
            str(status),
        ],
    )

    with pytest.raises(RuntimeError, match="version drift"):
        cli.main()

    assert not output.exists()
    assert not status.exists()


def test_window_adapter_preserves_every_origin_and_metric_space():
    dataset = _Dataset()

    batch = extract_window_batch(
        dataset,
        _cell(family="pems"),
        start=0,
        stop=3,
    )

    assert batch["model_x"].shape == (3, 4, 2)
    assert batch["y"].shape == (3, 2, 2)
    np.testing.assert_array_equal(
        batch["model_x"][:, 0, 0],
        np.asarray([0.0, 100.0, 200.0]),
    )
    np.testing.assert_array_equal(
        batch["x"][:, 0, 0],
        np.asarray([100.0, 1100.0, 2100.0]),
    )
    np.testing.assert_array_equal(
        batch["y"][0, :, 1],
        np.asarray([300.0, 340.0]),
    )


def test_ms_adapter_uses_only_target_channel():
    dataset = _Dataset()
    cell = _cell(features="MS", output_variates=1)

    batch = extract_window_batch(dataset, cell, start=0, stop=2)
    train = training_target_matrix(dataset, cell)

    assert batch["model_x"].shape == (2, 4, 1)
    assert batch["y"].shape == (2, 2, 1)
    np.testing.assert_array_equal(train[:, 0], dataset.data_x[:, -1])


def test_context_frame_round_trip_keeps_window_channel_order():
    contexts = np.asarray(
        [
            [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]],
            [[4.0, 40.0], [5.0, 50.0], [6.0, 60.0]],
        ],
        dtype=np.float32,
    )

    frame, item_ids = make_context_frame(
        contexts,
        frequency="h",
        first_window=7,
    )

    assert item_ids.tolist() == [14, 15, 16, 17]
    np.testing.assert_array_equal(
        frame.xs(14, level="item_id")["target"].to_numpy(),
        [1.0, 2.0, 3.0],
    )
    np.testing.assert_array_equal(
        frame.xs(17, level="item_id")["target"].to_numpy(),
        [40.0, 50.0, 60.0],
    )


def test_prediction_unpacking_restores_window_horizon_channel_axes():
    item_ids = np.asarray([14, 15, 16, 17])
    index = pd.MultiIndex.from_product(
        [item_ids, pd.date_range("2000-01-01", periods=2, freq="h")],
        names=["item_id", "timestamp"],
    )
    predictions = pd.DataFrame(
        {"mean": np.arange(8, dtype=np.float32)},
        index=index,
    )

    restored = unpack_mean_predictions(
        predictions,
        item_ids,
        window_count=2,
        channel_count=2,
        pred_len=2,
    )

    assert restored.shape == (2, 2, 2)
    np.testing.assert_array_equal(restored[0], [[0.0, 2.0], [1.0, 3.0]])
    np.testing.assert_array_equal(restored[1], [[4.0, 6.0], [5.0, 7.0]])


def test_panel_frame_rejects_duplicate_item_ids():
    with pytest.raises(ValueError, match="unique"):
        make_panel_frame(
            np.ones((4, 2), dtype=np.float32),
            frequency="h",
            item_ids=[1, 1],
        )


def test_frequency_uses_actual_ett_minute_and_pems_cadence():
    assert autogluon_frequency(_cell()) == "15min"
    assert autogluon_frequency(
        {**_cell(family="pems"), "dataset": "PEMS03"}
    ) == "5min"


def test_staging_cleanup_closes_memmaps_and_retries_nfs_race(
    tmp_path,
    monkeypatch,
):
    import ts_rag.appendix_autogluon as exporter

    path = tmp_path / "staging"
    path.mkdir()
    array = np.lib.format.open_memmap(
        path / "values.npy",
        mode="w+",
        dtype=np.float32,
        shape=(4,),
    )
    mmap = array._mmap
    arrays = {"validation": {"x": array}}

    _close_staged_arrays(arrays)

    assert mmap.closed
    assert arrays == {}
    real_rmtree = exporter.shutil.rmtree
    attempts = []

    def flaky_rmtree(target):
        attempts.append(target)
        if len(attempts) == 1:
            raise OSError(errno.ENOTEMPTY, "NFS directory not empty")
        real_rmtree(target)

    monkeypatch.setattr(exporter.shutil, "rmtree", flaky_rmtree)
    monkeypatch.setattr(exporter.time, "sleep", lambda _seconds: None)

    _remove_staging_tree(path)

    assert len(attempts) == 2
    assert not path.exists()


def test_exporter_writes_complete_bundle_without_dropping_origins(
    tmp_path,
    monkeypatch,
):
    import ts_rag.appendix_autogluon as exporter

    cell = {
        **_cell(),
        "paper_status": {"evaluable": True, "status": "reported"},
    }
    dataset = _Dataset()

    class FakePredictor:
        def predict(self, frame, **kwargs):
            del kwargs
            item_ids = frame.index.unique(level="item_id")
            index = pd.MultiIndex.from_product(
                [
                    item_ids,
                    pd.date_range("2000-01-05", periods=2, freq="h"),
                ],
                names=["item_id", "timestamp"],
            )
            last = frame["target"].groupby(level="item_id").last()
            mean = np.repeat(last.reindex(item_ids).to_numpy(), 2)
            return pd.DataFrame({"mean": mean}, index=index)

    monkeypatch.setattr(
        exporter,
        "load_tslib_dataset",
        lambda *args, **kwargs: dataset,
    )
    monkeypatch.setattr(
        exporter,
        "fit_or_load_predictor",
        lambda *args, **kwargs: (
            FakePredictor(),
            {
                "autogluon_timeseries_version": "1.4.0",
                "model_names": ["Fake"],
                "model_count": 1,
            },
        ),
    )
    monkeypatch.setattr(
        exporter,
        "load_autogluon_backend",
        lambda: (lambda frame: frame, object, "1.4.0"),
    )
    monkeypatch.setattr(
        exporter,
        "require_autogluon_timeseries_version",
        lambda: "1.4.0",
    )
    output = tmp_path / "bundle.npz"
    progress = []

    metadata = export_autogluon_prediction_bundle(
        cell,
        output,
        predictor_path=tmp_path / "predictor",
        staging_root=tmp_path / "staging",
        requested_batch_windows=100,
        max_prediction_items=4,
        progress_callback=progress.append,
    )

    assert metadata["evaluation_status"] == "exported"
    assert metadata["splits"]["validation"]["samples"] == 3
    assert metadata["splits"]["test"]["samples"] == 3
    assert metadata["splits"]["validation"]["batch_windows"] == 2
    assert metadata["splits"]["test"]["batch_windows"] == 2
    assert not (tmp_path / "staging").exists()
    assert progress[-1]["completed_windows"] == 3
    with np.load(output) as bundle:
        assert bundle["validation_x"].shape == (3, 4, 2)
        assert bundle["validation_y_base"].shape == (3, 2, 2)
        np.testing.assert_array_equal(
            bundle["validation_y_base"][:, 0, 0],
            np.asarray([6.0, 106.0, 206.0]),
        )
