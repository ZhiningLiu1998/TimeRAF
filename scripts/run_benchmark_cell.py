import argparse
import hashlib
import json
import math
import numbers
import os
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import torch

from ts_rag.benchmark_rag import (
    evaluate_method_leaders,
    run_validation_selected_rag,
)
from ts_rag.appendix_rag import save_prediction_bundles
from ts_rag.config import set_seed
from ts_rag.evaluation import save_json
from ts_rag.exp_wrapper import ExpLongTermForecastRAG
from ts_rag.matrix import resolve_source_revision


class NumericalIntegrityError(ValueError):
    """The cell cannot produce a finite experimental artifact."""


def _load_manifest(path):
    with open(path, "r", encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def _select_cell(rows, args):
    if args.cell_id:
        matches = [row for row in rows if row["id"] == args.cell_id]
    else:
        matches = [
            row
            for row in rows
            if row["dataset"] == args.dataset
            and row["model"] == args.model
            and row["pred_len"] == args.pred_len
        ]
    if len(matches) != 1:
        raise ValueError(f"Expected one manifest cell, found {len(matches)}")
    return matches[0]


def _parse_override(value):
    key, separator, raw = value.partition("=")
    if not separator or not key:
        raise argparse.ArgumentTypeError("Overrides must use KEY=VALUE")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = raw
    return key, parsed


def _jsonable_args(args):
    return {
        key: value
        if isinstance(value, (str, int, float, bool, list, dict, type(None)))
        else str(value)
        for key, value in vars(args).items()
    }


def _setting(cell, runtime_args):
    config = json.dumps(vars(runtime_args), sort_keys=True, default=str).encode()
    suffix = hashlib.sha256(config).hexdigest()[:12]
    return (
        f"{cell['task_family']}/{cell['dataset']}/{cell['model']}/"
        f"pl{cell['pred_len']}/seed{runtime_args.seed}_{suffix}"
    )


def _prepare_runtime_args(cell, cli):
    values = dict(cell["args"])
    for key, value in cli.override:
        values[key] = value
    values.update(
        {
            "seed": cli.seed,
            "source_revision": cli.source_revision,
            "checkpoints": str(Path(cli.checkpoint_root)),
            "num_workers": cli.num_workers,
            "use_gpu": bool(not cli.cpu and torch.cuda.is_available()),
            "gpu": cli.gpu,
            "gpu_type": "cuda",
            "use_multi_gpu": False,
            "devices": str(cli.gpu),
            "inverse": cell["metric_space"] == "inverse_scaled",
            "use_amp": bool(values.get("use_amp", False) and not cli.cpu),
        }
    )
    if cli.smoke:
        values.update(
            {
                "batch_size": min(int(values["batch_size"]), 2),
                "train_epochs": 1,
                "patience": 1,
                "num_workers": 0,
                "use_amp": False,
            }
        )
    return SimpleNamespace(**values)


def _checkpoint_path(runtime_args, setting):
    return Path(runtime_args.checkpoints) / setting / "checkpoint.pth"


def _train_or_load(exp, runtime_args, setting, cli):
    checkpoint = Path(cli.checkpoint) if cli.checkpoint else _checkpoint_path(runtime_args, setting)
    if cli.checkpoint or (checkpoint.exists() and not cli.force_train):
        exp.load_checkpoint(setting, str(checkpoint))
        return {"mode": "loaded", "checkpoint": str(checkpoint)}
    if cli.no_train:
        raise FileNotFoundError(f"No checkpoint available at {checkpoint}")
    if cli.smoke:
        loss = exp.smoke_train(setting, max_batches=cli.smoke_train_batches)
        return {
            "mode": "smoke_train",
            "checkpoint": str(_checkpoint_path(runtime_args, setting)),
            "train_loss": loss,
        }
    try:
        exp.train(setting)
    except FileNotFoundError as error:
        missing_path = getattr(error, "filename", None)
        if (
            missing_path is not None
            and Path(missing_path).resolve() == checkpoint.resolve()
            and not checkpoint.exists()
        ):
            raise NumericalIntegrityError(
                "training produced no finite validation checkpoint at "
                f"{checkpoint}"
            ) from error
        raise
    return {
        "mode": "trained",
        "checkpoint": str(_checkpoint_path(runtime_args, setting)),
    }


def _write_status(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        json.dump(payload, output, indent=2, sort_keys=True)
        output.write("\n")
    temporary.replace(path)


def _require_finite(value, label):
    if isinstance(value, dict):
        for key, child in value.items():
            _require_finite(child, f"{label}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _require_finite(child, f"{label}[{index}]")
        return
    if isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            count = int(value.size - np.count_nonzero(np.isfinite(value)))
            raise NumericalIntegrityError(
                f"{label} contains {count} non-finite value(s)"
            )
        return
    if isinstance(value, numbers.Real) and not isinstance(value, bool):
        if not math.isfinite(float(value)):
            raise NumericalIntegrityError(
                f"{label} is non-finite: {value!r}"
            )


def _require_finite_metrics(result, metric_names):
    for section in ("test_baseline", "test_corrected"):
        values = result.get(section)
        if not isinstance(values, dict):
            raise ValueError(f"{section} is missing or is not a mapping")
        for metric in metric_names:
            if metric not in values:
                raise ValueError(f"{section}.{metric} is missing")
            _require_finite(values[metric], f"{section}.{metric}")
    _require_finite(result, "result")


def _validate_finite_result(result, metric_names):
    _require_finite_metrics(result, metric_names)


def main():
    parser = argparse.ArgumentParser(description="Run one TimeFuse benchmark cell")
    parser.add_argument("--manifest", default="./docs/timefuse_experiment_manifest.jsonl")
    parser.add_argument("--cell-id")
    parser.add_argument("--dataset")
    parser.add_argument("--model")
    parser.add_argument("--pred-len", type=int)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--source-revision")
    parser.add_argument("--output-root", default="./ts_rag_outputs/timefuse_matrix")
    parser.add_argument("--checkpoint-root", default="./checkpoints/timefuse_matrix")
    parser.add_argument("--checkpoint")
    parser.add_argument("--no-train", action="store_true")
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-train-batches", type=int, default=1)
    parser.add_argument("--smoke-eval-samples", type=int, default=256)
    parser.add_argument("--save-arrays", action="store_true")
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="Export baseline prediction bundles without running RAG",
    )
    parser.add_argument("--development-diagnostics", action="store_true")
    parser.add_argument("--override", action="append", type=_parse_override, default=[])
    cli = parser.parse_args()
    cli.source_revision = cli.source_revision or resolve_source_revision()

    if not cli.cell_id and None in (cli.dataset, cli.model, cli.pred_len):
        parser.error("Use --cell-id or all of --dataset, --model, and --pred-len")

    cell = _select_cell(_load_manifest(cli.manifest), cli)
    runtime_args = _prepare_runtime_args(cell, cli)
    set_seed(runtime_args.seed)
    setting = _setting(cell, runtime_args)
    artifact_dir = Path(cli.output_root) / setting
    status_path = artifact_dir / "status.json"
    start = time.time()
    base_status = {
        "cell_id": cell["id"],
        "setting": setting,
        "seed": runtime_args.seed,
        "source_revision": runtime_args.source_revision,
        "smoke": cli.smoke,
        "started_unix": start,
        "status": "running",
    }
    _write_status(status_path, base_status)

    try:
        exp = ExpLongTermForecastRAG(runtime_args)
        checkpoint = _train_or_load(exp, runtime_args, setting, cli)
        limit = cli.smoke_eval_samples if cli.smoke else 0
        validation_bundle = exp.predict_split(
            "val", inverse=runtime_args.inverse, limit=limit
        )
        test_bundle = exp.predict_split(
            "test", inverse=runtime_args.inverse, limit=limit
        )
        _require_finite(validation_bundle, "validation_bundle")
        _require_finite(test_bundle, "test_bundle")
        artifact_dir.mkdir(parents=True, exist_ok=True)
        prediction_bundle_sha256 = None
        if cli.save_arrays or cli.export_only:
            prediction_bundle_sha256 = save_prediction_bundles(
                artifact_dir / "prediction_bundle.npz",
                validation_bundle,
                test_bundle,
                cell["task_family"],
            )
        if cli.export_only:
            payload = {
                "cell": cell,
                "runtime_args": _jsonable_args(runtime_args),
                "checkpoint": checkpoint,
                "smoke": cli.smoke,
                "export_only": True,
                "validation_samples": int(len(validation_bundle["y"])),
                "test_samples": int(len(test_bundle["y"])),
                "prediction_bundle": str(
                    artifact_dir / "prediction_bundle.npz"
                ),
                "prediction_bundle_sha256": prediction_bundle_sha256,
                "elapsed_seconds": time.time() - start,
            }
            save_json(payload, str(artifact_dir / "result.json"))
            _write_status(
                status_path,
                {
                    **base_status,
                    "status": "completed",
                    "export_only": True,
                    "elapsed_seconds": time.time() - start,
                    "prediction_bundle_sha256": prediction_bundle_sha256,
                },
            )
            print(
                json.dumps(
                    {
                        "cell_id": cell["id"],
                        "artifact_dir": str(artifact_dir),
                        "export_only": True,
                        "prediction_bundle_sha256": (
                            prediction_bundle_sha256
                        ),
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return

        result, corrected = run_validation_selected_rag(
            validation_bundle,
            test_bundle,
            cell["task_family"],
            tuple(cell["metrics"]),
            cell["pred_len"],
        )
        _require_finite(corrected, "corrected_predictions")
        _validate_finite_result(result, tuple(cell["metrics"]))
        if cli.development_diagnostics:
            result["development_diagnostics"] = evaluate_method_leaders(
                validation_bundle,
                test_bundle,
                result["validation"],
                cell["task_family"],
                tuple(cell["metrics"]),
                cell["pred_len"],
            )
            _require_finite(
                result["development_diagnostics"],
                "development_diagnostics",
            )
        payload = {
            "cell": cell,
            "runtime_args": _jsonable_args(runtime_args),
            "checkpoint": checkpoint,
            "smoke": cli.smoke,
            "validation_samples": int(len(validation_bundle["y"])),
            "test_samples": int(len(test_bundle["y"])),
            "prediction_bundle_sha256": prediction_bundle_sha256,
            "elapsed_seconds": time.time() - start,
            **result,
        }
        save_json(payload, str(artifact_dir / "result.json"))
        if cli.save_arrays:
            np.savez_compressed(
                artifact_dir / "test_predictions.npz",
                baseline=test_bundle["y_base"],
                corrected=corrected,
                true=test_bundle["y"],
            )
        _write_status(
            status_path,
            {
                **base_status,
                "status": "completed",
                "elapsed_seconds": time.time() - start,
                "all_test_metrics_improve": result["all_test_metrics_improve"],
            },
        )
        print(
            json.dumps(
                {
                    "cell_id": cell["id"],
                    "artifact_dir": str(artifact_dir),
                    "selected": result["validation"]["selected"],
                    "test_baseline": result["test_baseline"],
                    "test_corrected": result["test_corrected"],
                    "all_test_metrics_improve": result["all_test_metrics_improve"],
                },
                indent=2,
                sort_keys=True,
            )
        )
    except Exception as error:
        _write_status(
            status_path,
            {
                **base_status,
                "status": "failed",
                "elapsed_seconds": time.time() - start,
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            },
        )
        raise


if __name__ == "__main__":
    main()
