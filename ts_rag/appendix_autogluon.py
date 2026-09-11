import errno
import importlib.metadata
import os
import shutil
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from ts_rag.appendix_benchmark import (
    AUTOGLOUON_TIMESERIES_VERSION,
    TORCH_CHECKPOINT_COMPATIBILITY_ENV,
    autogluon_checkpoint_compatibility,
)
from ts_rag.appendix_rag import save_prediction_bundles, sha256_file


AUTOGLOUON_BASELINES = {
    "autogluon_high_quality",
    "chronos_bolt_finetuned",
    "chronos_bolt_zeroshot",
}
DEFAULT_DATA_ROOTS = {
    "long_term": Path("./dataset/timefuse/long_term_forecast"),
    "pems": Path("./dataset/timefuse/short_term_forecast/PEMS"),
    "epf": Path("./dataset/timefuse/short_term_forecast/EPF"),
}


@contextmanager
def trusted_autogluon_checkpoint_loads(baseline):
    policy = autogluon_checkpoint_compatibility(baseline)
    if not policy["enabled"]:
        yield policy
        return

    force_weights_only = os.environ.get(
        "TORCH_FORCE_WEIGHTS_ONLY_LOAD",
        "",
    ).lower()
    if force_weights_only in {"1", "y", "yes", "true"}:
        raise RuntimeError(
            "Conflicting PyTorch checkpoint compatibility environment"
        )
    previous = os.environ.get(TORCH_CHECKPOINT_COMPATIBILITY_ENV)
    os.environ[TORCH_CHECKPOINT_COMPATIBILITY_ENV] = "1"
    try:
        yield policy
    finally:
        if previous is None:
            os.environ.pop(TORCH_CHECKPOINT_COMPATIBILITY_ENV, None)
        else:
            os.environ[TORCH_CHECKPOINT_COMPATIBILITY_ENV] = previous


def autogluon_fit_recipe(
    baseline,
    fine_tune_steps=1000,
    inference_batch_size=32,
    fine_tune_batch_size=32,
):
    if baseline == "autogluon_high_quality":
        return {
            "presets": "high_quality",
            "fit_kwargs": {},
            "paper_description": "AutoGluon high_quality (24 models)",
        }
    if baseline == "chronos_bolt_zeroshot":
        return {
            "presets": None,
            "fit_kwargs": {
                "hyperparameters": {
                    "Chronos": {
                        "model_path": "bolt_base",
                        "batch_size": int(inference_batch_size),
                    }
                },
                "skip_model_selection": True,
                "enable_ensemble": False,
            },
            "paper_description": "Chronos-Bolt-Base zeroshot",
        }
    if baseline == "chronos_bolt_finetuned":
        return {
            "presets": None,
            "fit_kwargs": {
                "hyperparameters": {
                    "Chronos": {
                        "model_path": "bolt_base",
                        "batch_size": int(inference_batch_size),
                        "fine_tune": True,
                        "fine_tune_steps": int(fine_tune_steps),
                        "fine_tune_batch_size": int(fine_tune_batch_size),
                    }
                },
                "skip_model_selection": True,
                "enable_ensemble": False,
            },
            "paper_description": "Chronos-Bolt-Base finetuned",
        }
    raise ValueError(f"Unsupported AutoGluon appendix baseline: {baseline}")


def autogluon_frequency(cell):
    if cell["dataset"].startswith("ETTm"):
        return "15min"
    if cell["task_family"] == "pems":
        return "5min"
    return "h"


def _output_slice(cell):
    return slice(-1, None) if cell["protocol"]["features"] == "MS" else slice(None)


def _as_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _inverse_output(dataset, values):
    values = np.asarray(values)
    shape = values.shape
    feature_count = int(dataset.scaler.n_features_in_)
    if shape[-1] == feature_count:
        return dataset.inverse_transform(
            values.reshape(-1, feature_count)
        ).reshape(shape)
    if shape[-1] == 1:
        return (
            values * np.asarray(dataset.scaler.scale_[-1])
            + np.asarray(dataset.scaler.mean_[-1])
        )
    raise ValueError(
        f"Cannot inverse {shape[-1]} channels with a "
        f"{feature_count}-feature scaler"
    )


def extract_window_batch(dataset, cell, start, stop):
    if not 0 <= start < stop <= len(dataset):
        raise ValueError(
            f"Invalid window range [{start}, {stop}) for {len(dataset)} samples"
        )
    output_slice = _output_slice(cell)
    pred_len = int(cell["pred_len"])
    xs, ys, x_marks, y_marks = [], [], [], []
    for index in range(start, stop):
        seq_x, seq_y, seq_x_mark, seq_y_mark = dataset[index]
        xs.append(_as_numpy(seq_x)[:, output_slice])
        ys.append(_as_numpy(seq_y)[-pred_len:, output_slice])
        x_marks.append(_as_numpy(seq_x_mark))
        y_marks.append(_as_numpy(seq_y_mark)[-pred_len:])

    model_x = np.asarray(xs, dtype=np.float32)
    model_y = np.asarray(ys, dtype=np.float32)
    if cell["task_family"] == "pems":
        reported_x = _inverse_output(dataset, model_x)
        reported_y = _inverse_output(dataset, model_y)
    else:
        reported_x = model_x
        reported_y = model_y
    return {
        "model_x": model_x,
        "x": np.asarray(reported_x, dtype=np.float32),
        "y": np.asarray(reported_y, dtype=np.float32),
        "x_mark": np.asarray(x_marks, dtype=np.float32),
        "y_mark": np.asarray(y_marks, dtype=np.float32),
    }


def training_target_matrix(dataset, cell):
    values = np.asarray(dataset.data_x[:, _output_slice(cell)], dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != cell["protocol"]["output_variates"]:
        raise ValueError(
            "Training target matrix does not match the appendix protocol: "
            f"{values.shape}"
        )
    return values


def make_panel_frame(values, frequency, item_ids=None):
    values = np.asarray(values)
    if values.ndim != 2:
        raise ValueError("Panel values must have shape [time, item]")
    time_steps, item_count = values.shape
    if item_ids is None:
        item_ids = np.arange(item_count, dtype=np.int64)
    item_ids = np.asarray(item_ids)
    if len(item_ids) != item_count or len(np.unique(item_ids)) != item_count:
        raise ValueError("Panel item IDs must be unique and match item count")
    timestamps = pd.date_range("2000-01-01", periods=time_steps, freq=frequency)
    index = pd.MultiIndex.from_product(
        [item_ids, timestamps],
        names=["item_id", "timestamp"],
    )
    return pd.DataFrame(
        {"target": values.T.reshape(-1)},
        index=index,
    )


def make_context_frame(contexts, frequency, first_window=0):
    contexts = np.asarray(contexts)
    if contexts.ndim != 3:
        raise ValueError("Contexts must have shape [window, time, channel]")
    window_count, _, channel_count = contexts.shape
    item_ids = np.arange(
        first_window * channel_count,
        (first_window + window_count) * channel_count,
        dtype=np.int64,
    )
    values = contexts.transpose(1, 0, 2).reshape(
        contexts.shape[1],
        window_count * channel_count,
    )
    return make_panel_frame(values, frequency, item_ids=item_ids), item_ids


def unpack_mean_predictions(predictions, item_ids, window_count, channel_count, pred_len):
    if "mean" not in predictions:
        raise KeyError("AutoGluon predictions do not contain a mean forecast")
    mean = predictions["mean"]
    rows = []
    for item_id in item_ids:
        try:
            values = np.asarray(
                mean.xs(item_id, level="item_id"),
                dtype=np.float32,
            ).reshape(-1)
        except KeyError as error:
            raise ValueError(f"Missing forecast for item_id={item_id}") from error
        if len(values) != pred_len:
            raise ValueError(
                f"item_id={item_id} has {len(values)} predictions, "
                f"expected {pred_len}"
            )
        rows.append(values)
    flat = np.asarray(rows, dtype=np.float32)
    expected = window_count * channel_count
    if flat.shape != (expected, pred_len):
        raise ValueError(
            f"Forecast matrix has shape {flat.shape}, "
            f"expected {(expected, pred_len)}"
        )
    return flat.reshape(window_count, channel_count, pred_len).transpose(0, 2, 1)


def _dataset_args(cell):
    protocol = cell["protocol"]
    return SimpleNamespace(
        augmentation_ratio=0,
        embed="timeF",
        seasonal_patterns="Monthly",
        task_name="long_term_forecast",
        data=protocol["data"],
        root_path=None,
        data_path=protocol["data_path"],
        features=protocol["features"],
        target=protocol["target"],
        freq=protocol["freq"],
        seq_len=int(cell["seq_len"]),
        label_len=int(protocol["label_len"]),
        pred_len=int(cell["pred_len"]),
    )


def load_tslib_dataset(cell, split, data_root=None):
    from data_provider.data_factory import data_dict

    if split not in {"train", "val", "test"}:
        raise ValueError(f"Unsupported split: {split}")
    root = Path(data_root or DEFAULT_DATA_ROOTS[cell["task_family"]])
    args = _dataset_args(cell)
    args.root_path = str(root)
    dataset_type = data_dict[args.data]
    return dataset_type(
        args=args,
        root_path=str(root),
        data_path=args.data_path,
        flag=split,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=1,
        freq=args.freq,
        seasonal_patterns=args.seasonal_patterns,
    )


def require_autogluon_timeseries_version():
    """Require the exact AutoGluon version frozen by the Appendix protocol."""
    try:
        installed = importlib.metadata.version("autogluon.timeseries")
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError(
            "AutoGluon version preflight failed: "
            "autogluon.timeseries is not installed; "
            f"required {AUTOGLOUON_TIMESERIES_VERSION}"
        ) from error
    if installed != AUTOGLOUON_TIMESERIES_VERSION:
        raise RuntimeError(
            "AutoGluon version drift: "
            f"installed {installed}, required {AUTOGLOUON_TIMESERIES_VERSION}"
        )
    return installed


def load_autogluon_backend():
    installed = require_autogluon_timeseries_version()
    try:
        from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
    except ImportError as error:
        raise ImportError(
            "Appendix AutoGluon export requires "
            f"autogluon.timeseries=={AUTOGLOUON_TIMESERIES_VERSION}; "
            "install requirements-appendix.txt"
        ) from error
    return TimeSeriesDataFrame, TimeSeriesPredictor, installed


def fit_or_load_predictor(
    cell,
    train_dataset,
    predictor_path,
    seed=2021,
    time_limit=None,
    fine_tune_steps=1000,
    inference_batch_size=32,
    fine_tune_batch_size=32,
):
    TimeSeriesDataFrame, TimeSeriesPredictor, version = load_autogluon_backend()
    predictor_path = Path(predictor_path)
    recipe = autogluon_fit_recipe(
        cell["baseline"],
        fine_tune_steps=fine_tune_steps,
        inference_batch_size=inference_batch_size,
        fine_tune_batch_size=fine_tune_batch_size,
    )
    if predictor_path.exists():
        predictor = TimeSeriesPredictor.load(str(predictor_path))
        mode = "loaded"
    else:
        frequency = autogluon_frequency(cell)
        train_frame = make_panel_frame(
            training_target_matrix(train_dataset, cell),
            frequency,
        )
        train_data = TimeSeriesDataFrame(train_frame)
        predictor = TimeSeriesPredictor(
            target="target",
            prediction_length=int(cell["pred_len"]),
            freq=frequency,
            eval_metric=(
                "MAE" if cell["task_family"] == "pems" else "MSE"
            ),
            path=str(predictor_path),
            cache_predictions=False,
        )
        fit_kwargs = dict(recipe["fit_kwargs"])
        fit_kwargs.update(
            {
                "train_data": train_data,
                "random_seed": int(seed),
            }
        )
        if recipe["presets"] is not None:
            fit_kwargs["presets"] = recipe["presets"]
        if time_limit is not None:
            fit_kwargs["time_limit"] = int(time_limit)
        predictor.fit(**fit_kwargs)
        mode = "fit"

    if int(predictor.prediction_length) != int(cell["pred_len"]):
        raise ValueError(
            "Loaded predictor horizon does not match the appendix cell"
        )
    model_names = list(predictor.model_names())
    return predictor, {
        "autogluon_timeseries_version": version,
        "mode": mode,
        "path": str(predictor_path),
        "recipe": recipe,
        "model_names": model_names,
        "model_count": len(model_names),
        "paper_model_count_claim": (
            24 if cell["baseline"] == "autogluon_high_quality" else 1
        ),
    }


class _SplitStager:
    def __init__(self, root, split, sample_count, sample):
        self.root = Path(root) / split
        self.root.mkdir(parents=True, exist_ok=True)
        self.arrays = {}
        for key in ("x", "y", "x_mark", "y_mark"):
            shape = (sample_count, *sample[key].shape[1:])
            self.arrays[key] = np.lib.format.open_memmap(
                self.root / f"{key}.npy",
                mode="w+",
                dtype=np.float32,
                shape=shape,
            )
        self.arrays["y_base"] = np.lib.format.open_memmap(
            self.root / "y_base.npy",
            mode="w+",
            dtype=np.float32,
            shape=self.arrays["y"].shape,
        )

    def write(self, start, stop, batch, predictions):
        for key in ("x", "y", "x_mark", "y_mark"):
            self.arrays[key][start:stop] = batch[key]
        self.arrays["y_base"][start:stop] = predictions
        for value in self.arrays.values():
            value.flush()


def _close_staged_arrays(split_arrays):
    for arrays in split_arrays.values():
        for value in arrays.values():
            value.flush()
            mmap = getattr(value, "_mmap", None)
            if mmap is not None and not mmap.closed:
                mmap.close()
        arrays.clear()
    split_arrays.clear()


def _remove_staging_tree(path, attempts=8, delay_seconds=0.25):
    for attempt in range(attempts):
        try:
            shutil.rmtree(path)
            return
        except FileNotFoundError:
            return
        except OSError as error:
            if error.errno != errno.ENOTEMPTY or attempt + 1 == attempts:
                raise
            time.sleep(delay_seconds * (attempt + 1))


def _predict_split(
    predictor,
    TimeSeriesDataFrame,
    dataset,
    cell,
    split,
    staging_root,
    requested_batch_windows,
    max_prediction_items,
    seed,
    progress_callback,
):
    sample_count = len(dataset)
    first = extract_window_batch(dataset, cell, 0, 1)
    stager = _SplitStager(staging_root, split, sample_count, first)
    channel_count = int(first["model_x"].shape[-1])
    batch_windows = min(
        int(requested_batch_windows),
        max(1, int(max_prediction_items) // channel_count),
    )
    started = time.time()
    for start in range(0, sample_count, batch_windows):
        stop = min(sample_count, start + batch_windows)
        batch = extract_window_batch(dataset, cell, start, stop)
        frame, item_ids = make_context_frame(
            batch["model_x"],
            autogluon_frequency(cell),
            first_window=start,
        )
        forecasts = predictor.predict(
            TimeSeriesDataFrame(frame),
            use_cache=False,
            random_seed=int(seed),
        )
        predictions = unpack_mean_predictions(
            forecasts,
            item_ids,
            window_count=stop - start,
            channel_count=channel_count,
            pred_len=int(cell["pred_len"]),
        )
        if cell["task_family"] == "pems":
            predictions = _inverse_output(dataset, predictions)
        stager.write(start, stop, batch, predictions)
        if progress_callback is not None:
            progress_callback(
                {
                    "split": split,
                    "completed_windows": stop,
                    "total_windows": sample_count,
                    "batch_windows": batch_windows,
                    "channels": channel_count,
                    "elapsed_seconds": time.time() - started,
                }
            )
    return stager.arrays, {
        "samples": sample_count,
        "channels": channel_count,
        "batch_windows": batch_windows,
        "elapsed_seconds": time.time() - started,
    }


def export_autogluon_prediction_bundle(
    cell,
    output_path,
    predictor_path,
    data_root=None,
    staging_root=None,
    seed=2021,
    time_limit=None,
    requested_batch_windows=64,
    max_prediction_items=2048,
    fine_tune_steps=1000,
    inference_batch_size=32,
    fine_tune_batch_size=32,
    progress_callback=None,
):
    if cell["baseline"] not in AUTOGLOUON_BASELINES:
        raise ValueError(
            f"Cell {cell['id']} is not an AutoGluon/Chronos appendix cell"
        )
    require_autogluon_timeseries_version()
    if not cell["paper_status"]["evaluable"]:
        return {
            "cell_id": cell["id"],
            "evaluation_status": "paper_oot",
            "paper_status": cell["paper_status"],
        }

    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite prediction bundle: {output_path}"
        )
    staging_root = Path(
        staging_root or output_path.parent / f".{output_path.stem}.staging"
    )
    if staging_root.exists():
        raise FileExistsError(
            f"Staging directory already exists: {staging_root}"
        )

    with trusted_autogluon_checkpoint_loads(cell["baseline"]) as compatibility:
        train_dataset = load_tslib_dataset(cell, "train", data_root=data_root)
        predictor, predictor_metadata = fit_or_load_predictor(
            cell,
            train_dataset,
            predictor_path,
            seed=seed,
            time_limit=time_limit,
            fine_tune_steps=fine_tune_steps,
            inference_batch_size=inference_batch_size,
            fine_tune_batch_size=fine_tune_batch_size,
        )
        predictor_metadata["checkpoint_compatibility"] = compatibility
        TimeSeriesDataFrame, _, _ = load_autogluon_backend()
        split_arrays = {}
        split_metadata = {}
        started = time.time()
        try:
            for split, dataset_split in (
                ("validation", "val"),
                ("test", "test"),
            ):
                dataset = load_tslib_dataset(
                    cell,
                    dataset_split,
                    data_root=data_root,
                )
                split_arrays[split], split_metadata[split] = _predict_split(
                    predictor,
                    TimeSeriesDataFrame,
                    dataset,
                    cell,
                    split,
                    staging_root,
                    requested_batch_windows=requested_batch_windows,
                    max_prediction_items=max_prediction_items,
                    seed=seed,
                    progress_callback=progress_callback,
                )
            digest = save_prediction_bundles(
                output_path,
                split_arrays["validation"],
                split_arrays["test"],
                cell["task_family"],
                already_reported=True,
            )
        except Exception:
            raise
        else:
            _close_staged_arrays(split_arrays)
            _remove_staging_tree(staging_root)

    return {
        "cell_id": cell["id"],
        "evaluation_status": "exported",
        "prediction_bundle": str(output_path),
        "prediction_bundle_sha256": digest,
        "prediction_bundle_size_bytes": output_path.stat().st_size,
        "predictor": predictor_metadata,
        "checkpoint_compatibility": compatibility,
        "splits": split_metadata,
        "seed": int(seed),
        "frequency": autogluon_frequency(cell),
        "window_protocol": "all_tslib_validation_and_test_forecast_origins",
        "elapsed_seconds": time.time() - started,
        "source_files": {
            "data_path": str(
                Path(data_root or DEFAULT_DATA_ROOTS[cell["task_family"]])
                / cell["protocol"]["data_path"]
            ),
        },
        "output_sha256_recheck": sha256_file(output_path),
    }
