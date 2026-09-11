import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from ts_rag.matrix import load_manifest, save_json_atomic
from ts_rag.release_checkpoints import build_release_catalog


def main():
    parser = argparse.ArgumentParser(
        description="Index checkpoints in the TimeFuse release ZIP"
    )
    parser.add_argument("archive")
    parser.add_argument(
        "--manifest", default="./docs/timefuse_experiment_manifest.jsonl"
    )
    parser.add_argument(
        "--output", default="./docs/timefuse_release_checkpoint_catalog.json"
    )
    args = parser.parse_args()

    catalog = build_release_catalog(
        args.archive,
        load_manifest(args.manifest),
    )
    save_json_atomic(catalog, args.output)
    print(
        f"checkpoints={catalog['checkpoint_count']} "
        f"config_compatible={catalog['config_compatible_count']} "
        f"output={args.output}"
    )


if __name__ == "__main__":
    main()
