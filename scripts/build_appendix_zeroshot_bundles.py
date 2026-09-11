import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from ts_rag.appendix_zeroshot import build_zeroshot_ensemble_bundles
from ts_rag.matrix import save_json_atomic


def main():
    parser = argparse.ArgumentParser(
        description="Build leave-one-dataset-out ZeroShot ensemble bundles"
    )
    parser.add_argument("--library-spec", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--metadata")
    args = parser.parse_args()

    with open(args.library_spec, "r", encoding="utf-8") as source:
        spec = json.load(source)
    task_family = spec["task_family"]
    task_bundles = spec["datasets"]
    metadata = build_zeroshot_ensemble_bundles(
        task_bundles,
        task_family,
        Path(args.output_root),
    )
    metadata_path = args.metadata or str(
        Path(args.output_root) / "zeroshot_metadata.json"
    )
    save_json_atomic(metadata, metadata_path)
    print(
        json.dumps(
            {
                "task_family": task_family,
                "datasets": sorted(metadata["outputs"]),
                "metadata": metadata_path,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
