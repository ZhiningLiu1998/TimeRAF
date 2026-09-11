import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from ts_rag.benchmark import DATASETS, MODELS, TIMEFUSE_COMMIT, iter_experiment_manifest


def main():
    parser = argparse.ArgumentParser(description="Build the TimeFuse benchmark matrix")
    parser.add_argument(
        "--output",
        default="./docs/timefuse_experiment_manifest.jsonl",
        help="Destination JSONL path",
    )
    parser.add_argument(
        "--scripts_root",
        default="./scripts/long_term_forecast",
        help="TSLib shell configuration root",
    )
    args = parser.parse_args()

    rows = list(iter_experiment_manifest(args.scripts_root))
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, sort_keys=True) + "\n")

    families = Counter(row["task_family"] for row in rows)
    sources = Counter(row["config_source"] for row in rows)
    print(
        json.dumps(
            {
                "timefuse_commit": TIMEFUSE_COMMIT,
                "datasets": len(DATASETS),
                "models": len(MODELS),
                "experiments": len(rows),
                "families": dict(sorted(families.items())),
                "config_sources": dict(sorted(sources.items())),
                "output": args.output,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
