import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from ts_rag.release_checkpoints import extract_config_compatible


def main():
    parser = argparse.ArgumentParser(
        description="Extract config-compatible TimeFuse release checkpoints"
    )
    parser.add_argument("archive")
    parser.add_argument("catalog")
    parser.add_argument("output_root")
    args = parser.parse_args()

    with open(args.catalog, "r", encoding="utf-8") as source:
        catalog = json.load(source)
    count = extract_config_compatible(
        args.archive,
        catalog,
        args.output_root,
    )
    print(f"extracted={count} output_root={args.output_root}")


if __name__ == "__main__":
    main()
