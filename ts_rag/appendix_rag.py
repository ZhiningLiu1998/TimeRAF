import hashlib
import json
import os
from pathlib import Path

import numpy as np

from ts_rag.benchmark_rag import (
    evaluate_method_leaders,
    run_validation_selected_rag,
)


ARRAY_KEYS = ("x", "y", "y_base", "x_mark", "y_mark")


def _all_finite(value, chunk_size=64):
    value = np.asarray(value)
    if value.ndim == 0:
        return bool(np.isfinite(value))
    return all(
        np.isfinite(value[start : start + chunk_size]).all()
        for start in range(0, len(value), chunk_size)
    )


def sha256_file(path, block_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def load_appendix_manifest(path):
    with Path(path).open("r", encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def select_appendix_cell(rows, cell_id):
    matches = [row for row in rows if row["id"] == cell_id]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one appendix cell for {cell_id!r}, found {len(matches)}"
        )
    return matches[0]


def _validate_bundle(bundle, split):
    sample_count = len(bundle["y"])
    if sample_count < 2:
        raise ValueError(f"{split} bundle requires at least two samples")
    if bundle["y"].shape != bundle["y_base"].shape:
        raise ValueError(
            f"{split} truth and baseline prediction shapes differ: "
            f"{bundle['y'].shape} != {bundle['y_base'].shape}"
        )
    if bundle["x"].ndim != 3 or bundle["y"].ndim != 3:
        raise ValueError(f"{split} x and y arrays must be rank three")
    for key in ARRAY_KEYS:
        value = bundle[key]
        if len(value) != sample_count:
            raise ValueError(
                f"{split}_{key} has {len(value)} samples, expected {sample_count}"
            )
        if not _all_finite(value):
            raise ValueError(f"{split}_{key} contains non-finite values")
    if bundle["x_mark"].shape[1] != bundle["x"].shape[1]:
        raise ValueError(f"{split} input marks do not align with x")
    if bundle["y_mark"].shape[1] != bundle["y"].shape[1]:
        raise ValueError(f"{split} output marks do not align with y")


def load_prediction_bundles(path, task_family):
    bundles = {}
    with np.load(path, allow_pickle=False) as archive:
        for split in ("validation", "test"):
            missing = [
                f"{split}_{key}"
                for key in ARRAY_KEYS
                if f"{split}_{key}" not in archive
            ]
            if missing:
                raise KeyError(
                    f"Prediction bundle is missing keys: {', '.join(missing)}"
                )
            bundle = {
                key: np.asarray(archive[f"{split}_{key}"])
                for key in ARRAY_KEYS
            }
            _validate_bundle(bundle, split)
            output_channels = bundle["y"].shape[-1]
            bundle.update(
                {
                    "split": "val" if split == "validation" else "test",
                    "future_residual": bundle["y"] - bundle["y_base"],
                    "output_scale": np.ones(output_channels, dtype=np.float64),
                    "output_mean": np.zeros(output_channels, dtype=np.float64),
                }
            )
            if task_family == "pems":
                bundle.update(
                    {
                        "x_inv": bundle["x"],
                        "y_inv": bundle["y"],
                        "y_base_inv": bundle["y_base"],
                        "future_residual_inv": (
                            bundle["y"] - bundle["y_base"]
                        ),
                    }
                )
            bundles[split] = bundle
    validation = bundles["validation"]
    test = bundles["test"]
    for key in ARRAY_KEYS:
        if validation[key].shape[1:] != test[key].shape[1:]:
            raise ValueError(
                f"Validation/test {key} shapes differ after the sample axis: "
                f"{validation[key].shape[1:]} != {test[key].shape[1:]}"
            )
    return validation, test


def _reported_split_arrays(bundle, task_family):
    if task_family == "pems":
        keys = {
            "x": "x_inv",
            "y": "y_inv",
            "y_base": "y_base_inv",
            "x_mark": "x_mark",
            "y_mark": "y_mark",
        }
    else:
        keys = {key: key for key in ARRAY_KEYS}
    missing = [source for source in keys.values() if source not in bundle]
    if missing:
        raise KeyError(
            "Evaluation bundle is missing reported-space arrays: "
            + ", ".join(missing)
        )
    return {key: np.asarray(bundle[source]) for key, source in keys.items()}


def save_prediction_bundles(
    path,
    validation_bundle,
    test_bundle,
    task_family,
    already_reported=False,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {}
    for split, bundle in (
        ("validation", validation_bundle),
        ("test", test_bundle),
    ):
        arrays = (
            {key: np.asarray(bundle[key]) for key in ARRAY_KEYS}
            if already_reported
            else _reported_split_arrays(bundle, task_family)
        )
        _validate_bundle(arrays, split)
        payload.update(
            {
                f"{split}_{key}": value
                for key, value in arrays.items()
            }
        )
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as output:
        np.savez_compressed(output, **payload)
    os.replace(temporary, path)
    return sha256_file(path)


def evaluate_appendix_cell(
    cell,
    prediction_bundle,
    development_diagnostics=False,
):
    if not cell["paper_status"]["evaluable"]:
        return {
            "cell": cell,
            "paper_status": cell["paper_status"],
            "evaluation_status": "paper_oot",
        }, None

    validation_bundle, test_bundle = load_prediction_bundles(
        prediction_bundle,
        cell["task_family"],
    )
    if validation_bundle["y"].shape[1] != cell["pred_len"]:
        raise ValueError(
            "Prediction horizon does not match appendix manifest: "
            f"{validation_bundle['y'].shape[1]} != {cell['pred_len']}"
        )

    result, corrected = run_validation_selected_rag(
        validation_bundle,
        test_bundle,
        cell["task_family"],
        tuple(cell["metrics"]),
        cell["pred_len"],
    )
    if development_diagnostics:
        result["development_diagnostics"] = evaluate_method_leaders(
            validation_bundle,
            test_bundle,
            result["validation"],
            cell["task_family"],
            tuple(cell["metrics"]),
            cell["pred_len"],
        )
    return {
        "cell": cell,
        "paper_status": cell["paper_status"],
        "evaluation_status": "completed",
        "prediction_bundle": str(prediction_bundle),
        "prediction_bundle_sha256": sha256_file(prediction_bundle),
        "validation_samples": int(len(validation_bundle["y"])),
        "test_samples": int(len(test_bundle["y"])),
        **result,
    }, corrected
