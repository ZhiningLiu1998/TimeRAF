import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from ts_rag.appendix_ensembles import build_advanced_ensemble_bundle
from ts_rag.benchmark import MODELS
from ts_rag.matrix import save_json_atomic


def _model_bundle(value):
    model, separator, path = value.partition("=")
    if not separator or model not in MODELS or not path:
        raise argparse.ArgumentTypeError(
            "--base-bundle must use a known MODEL=PATH"
        )
    return model, path


def main():
    parser = argparse.ArgumentParser(
        description="Build a paper-matched static ensemble prediction bundle"
    )
    parser.add_argument(
        "--method",
        required=True,
        choices=("forward_selection", "portfolio_ensemble"),
    )
    parser.add_argument(
        "--task-family",
        required=True,
        choices=("long_term", "pems", "epf"),
    )
    parser.add_argument(
        "--base-bundle",
        action="append",
        type=_model_bundle,
        required=True,
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--metadata")
    parser.add_argument("--forward-ensemble-size", type=int, default=50)
    args = parser.parse_args()

    base_bundles = dict(args.base_bundle)
    if len(base_bundles) != len(args.base_bundle):
        parser.error("Each base model may be specified only once")
    metadata = build_advanced_ensemble_bundle(
        base_bundles,
        args.task_family,
        args.method,
        Path(args.output),
        forward_ensemble_size=args.forward_ensemble_size,
    )
    metadata_path = args.metadata or f"{args.output}.json"
    save_json_atomic(metadata, metadata_path)
    print(
        json.dumps(
            {
                "method": args.method,
                "prediction_bundle": args.output,
                "prediction_bundle_sha256": metadata[
                    "prediction_bundle_sha256"
                ],
                "metadata": metadata_path,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
