import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from ts_rag.matrix import load_manifest, save_json_atomic
from ts_rag.release_checkpoints import audit_extracted_checkpoints


def main():
    parser = argparse.ArgumentParser(
        description="Strict-load TimeFuse release checkpoints"
    )
    parser.add_argument("catalog")
    parser.add_argument("checkpoint_root")
    parser.add_argument(
        "--manifest", default="./docs/timefuse_experiment_manifest.jsonl"
    )
    parser.add_argument(
        "--output", default="./docs/timefuse_release_checkpoint_audit.json"
    )
    args = parser.parse_args()

    with open(args.catalog, "r", encoding="utf-8") as source:
        catalog = json.load(source)
    audited = audit_extracted_checkpoints(
        catalog,
        load_manifest(args.manifest),
        args.checkpoint_root,
    )
    save_json_atomic(audited, args.output)
    print(
        f"config_compatible={audited['config_compatible_count']} "
        f"usable={audited['usable_count']} output={args.output}"
    )


if __name__ == "__main__":
    main()
