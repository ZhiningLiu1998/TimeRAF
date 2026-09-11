import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from ts_rag.benchmark import DATASETS, TIMEFUSE_COMMIT
from ts_rag.data_audit import build_data_audit, sha256_file


def main():
    parser = argparse.ArgumentParser(description="Audit released TimeFuse datasets")
    parser.add_argument("--dataset_root", default="./dataset/timefuse")
    parser.add_argument("--output", default="./docs/timefuse_data_audit.json")
    parser.add_argument(
        "--release_archive",
        default="/tmp/timefuse-data-probe",
        help="Optional path to the downloaded TimeFuse release ZIP",
    )
    args = parser.parse_args()

    payload = build_data_audit(DATASETS, args.dataset_root)
    payload["timefuse_commit"] = TIMEFUSE_COMMIT
    if args.release_archive and os.path.exists(args.release_archive):
        payload["release_archive"] = {
            "path": args.release_archive,
            "bytes": os.path.getsize(args.release_archive),
            "sha256": sha256_file(args.release_archive),
        }

    output_directory = os.path.dirname(args.output)
    if output_directory:
        os.makedirs(output_directory, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as output:
        json.dump(payload, output, indent=2, sort_keys=True)
        output.write("\n")

    print(
        json.dumps(
            {
                "datasets": payload["dataset_count"],
                "all_files_present": payload["all_files_present"],
                "all_input_variates_match": payload["all_input_variates_match"],
                "output": args.output,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
