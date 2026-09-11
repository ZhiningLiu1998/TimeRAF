import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from ts_rag.appendix_matrix import APPENDIX_METHOD_REVISION
from ts_rag.appendix_rag import (
    evaluate_appendix_cell,
    load_appendix_manifest,
    select_appendix_cell,
)
from ts_rag.evaluation import save_json
from ts_rag.matrix import resolve_source_revision


def _save_json_atomic(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        json.dump(payload, output, indent=2, sort_keys=True)
        output.write("\n")
    os.replace(temporary, path)


def _artifact_dir(output_root, cell, source_revision):
    return (
        Path(output_root)
        / cell["task_family"]
        / cell["dataset"]
        / cell["baseline"]
        / f"pl{cell['pred_len']}"
        / source_revision[:12]
    )


def main():
    parser = argparse.ArgumentParser(
        description="Apply validation-selected RAG to one appendix baseline"
    )
    parser.add_argument(
        "--manifest",
        default="./docs/timefuse_appendix_experiment_manifest.jsonl",
    )
    parser.add_argument("--cell-id", required=True)
    parser.add_argument("--prediction-bundle")
    parser.add_argument(
        "--output-root",
        default="./ts_rag_outputs/timefuse_appendix_rag",
    )
    parser.add_argument("--source-revision")
    parser.add_argument(
        "--method-revision",
        default=APPENDIX_METHOD_REVISION,
    )
    parser.add_argument("--development-diagnostics", action="store_true")
    parser.add_argument("--save-arrays", action="store_true")
    args = parser.parse_args()
    args.source_revision = args.source_revision or resolve_source_revision()
    if args.method_revision != APPENDIX_METHOD_REVISION:
        parser.error(
            "Appendix method revision must remain frozen at "
            f"{APPENDIX_METHOD_REVISION}"
        )

    cell = select_appendix_cell(
        load_appendix_manifest(args.manifest),
        args.cell_id,
    )
    if cell["paper_status"]["evaluable"] and not args.prediction_bundle:
        parser.error("--prediction-bundle is required for evaluable cells")

    artifact_dir = _artifact_dir(
        args.output_root,
        cell,
        args.source_revision,
    )
    status_path = artifact_dir / "status.json"
    started = time.time()
    base_status = {
        "cell_id": cell["id"],
        "source_revision": args.source_revision,
        "method_revision": args.method_revision,
        "started_unix": started,
    }
    _save_json_atomic({**base_status, "status": "running"}, status_path)

    try:
        result, corrected = evaluate_appendix_cell(
            cell,
            args.prediction_bundle,
            development_diagnostics=args.development_diagnostics,
        )
        result["source_revision"] = args.source_revision
        result["method_revision"] = args.method_revision
        result["elapsed_seconds"] = time.time() - started
        save_json(result, str(artifact_dir / "result.json"))
        if args.save_arrays and corrected is not None:
            np.savez_compressed(
                artifact_dir / "corrected_predictions.npz",
                corrected=corrected,
            )
        _save_json_atomic(
            {
                **base_status,
                "status": result["evaluation_status"],
                "elapsed_seconds": time.time() - started,
                "all_test_metrics_improve": result.get(
                    "all_test_metrics_improve"
                ),
            },
            status_path,
        )
        print(
            json.dumps(
                {
                    "cell_id": cell["id"],
                    "artifact_dir": str(artifact_dir),
                    "evaluation_status": result["evaluation_status"],
                    "all_test_metrics_improve": result.get(
                        "all_test_metrics_improve"
                    ),
                },
                indent=2,
                sort_keys=True,
            )
        )
    except Exception as error:
        _save_json_atomic(
            {
                **base_status,
                "status": "failed",
                "elapsed_seconds": time.time() - started,
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            },
            status_path,
        )
        raise


if __name__ == "__main__":
    main()
