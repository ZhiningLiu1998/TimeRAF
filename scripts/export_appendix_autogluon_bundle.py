import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from ts_rag.appendix_autogluon import (
    autogluon_checkpoint_compatibility,
    export_autogluon_prediction_bundle,
    require_autogluon_timeseries_version,
)
from ts_rag.appendix_rag import (
    load_appendix_manifest,
    select_appendix_cell,
)
from ts_rag.matrix import save_json_atomic


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Export paper-aligned AutoGluon or Chronos-Bolt appendix forecasts"
        )
    )
    parser.add_argument(
        "--manifest",
        default="./docs/timefuse_appendix_experiment_manifest.jsonl",
    )
    parser.add_argument("--cell-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--predictor-path", required=True)
    parser.add_argument("--metadata")
    parser.add_argument("--status")
    parser.add_argument("--data-root")
    parser.add_argument("--staging-root")
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--time-limit", type=int)
    parser.add_argument("--batch-windows", type=int, default=64)
    parser.add_argument("--max-prediction-items", type=int, default=2048)
    parser.add_argument("--fine-tune-steps", type=int, default=1000)
    parser.add_argument("--inference-batch-size", type=int, default=32)
    parser.add_argument("--fine-tune-batch-size", type=int, default=32)
    parser.add_argument("--source-revision")
    parser.add_argument("--launcher-revision")
    parser.add_argument("--physical-gpu-id", type=int)
    args = parser.parse_args()
    if args.batch_windows < 1 or args.max_prediction_items < 1:
        parser.error("Batch limits must be positive")

    autogluon_timeseries_version = require_autogluon_timeseries_version()
    cell = select_appendix_cell(
        load_appendix_manifest(args.manifest),
        args.cell_id,
    )
    metadata_path = Path(args.metadata or f"{args.output}.json")
    status_path = Path(args.status or f"{args.output}.status.json")
    started = time.time()
    run_identity = {
        "source_revision": args.source_revision,
        "launcher_revision": args.launcher_revision,
        "physical_gpu_id": args.physical_gpu_id,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "autogluon_timeseries_version": autogluon_timeseries_version,
        "checkpoint_compatibility": autogluon_checkpoint_compatibility(
            cell["baseline"]
        ),
    }

    def update_progress(progress):
        save_json_atomic(
            {
                "cell_id": cell["id"],
                "status": "running",
                "started_unix": started,
                "run_identity": run_identity,
                **progress,
            },
            status_path,
        )

    save_json_atomic(
        {
            "cell_id": cell["id"],
            "status": "running",
            "started_unix": started,
            "run_identity": run_identity,
        },
        status_path,
    )
    try:
        metadata = export_autogluon_prediction_bundle(
            cell,
            args.output,
            args.predictor_path,
            data_root=args.data_root,
            staging_root=args.staging_root,
            seed=args.seed,
            time_limit=args.time_limit,
            requested_batch_windows=args.batch_windows,
            max_prediction_items=args.max_prediction_items,
            fine_tune_steps=args.fine_tune_steps,
            inference_batch_size=args.inference_batch_size,
            fine_tune_batch_size=args.fine_tune_batch_size,
            progress_callback=update_progress,
        )
        metadata["run_identity"] = run_identity
        save_json_atomic(metadata, metadata_path)
        save_json_atomic(
            {
                "cell_id": cell["id"],
                "status": metadata["evaluation_status"],
                "started_unix": started,
                "elapsed_seconds": time.time() - started,
                "run_identity": run_identity,
                "prediction_bundle_sha256": metadata.get(
                    "prediction_bundle_sha256"
                ),
            },
            status_path,
        )
        print(json.dumps(metadata, indent=2, sort_keys=True))
    except Exception as error:
        save_json_atomic(
            {
                "cell_id": cell["id"],
                "status": "failed",
                "started_unix": started,
                "elapsed_seconds": time.time() - started,
                "error_type": type(error).__name__,
                "error": str(error),
                "run_identity": run_identity,
                "traceback": traceback.format_exc(),
            },
            status_path,
        )
        raise


if __name__ == "__main__":
    main()
