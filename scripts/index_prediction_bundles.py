import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from ts_rag.matrix import load_manifest, save_json_atomic
from ts_rag.prediction_bundles import build_prediction_bundle_catalog


def _filter(rows, args):
    selected = []
    for row in rows:
        if args.family and row["task_family"] not in args.family:
            continue
        if args.dataset and row["dataset"] not in args.dataset:
            continue
        if args.model and row["model"] not in args.model:
            continue
        if args.pred_len and row["pred_len"] not in args.pred_len:
            continue
        selected.append(row)
    return selected


def main():
    parser = argparse.ArgumentParser(
        description="Index export-only matrix prediction bundles"
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--source-revision")
    parser.add_argument("--output", required=True)
    parser.add_argument("--hash", action="store_true")
    parser.add_argument("--family", action="append")
    parser.add_argument("--dataset", action="append")
    parser.add_argument("--model", action="append")
    parser.add_argument("--pred-len", action="append", type=int)
    args = parser.parse_args()

    catalog = build_prediction_bundle_catalog(
        _filter(load_manifest(args.manifest), args),
        args.output_root,
        args.project_root,
        source_revision=args.source_revision,
        compute_hashes=args.hash,
    )
    save_json_atomic(catalog, args.output)
    print(
        json.dumps(
            {
                "expected_cells": catalog["expected_cells"],
                "cataloged_cells": catalog["cataloged_cells"],
                "usable_count": catalog["usable_count"],
                "missing_count": catalog["missing_count"],
                "output": args.output,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
