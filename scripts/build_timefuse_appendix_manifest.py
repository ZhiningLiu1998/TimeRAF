import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from ts_rag.appendix_benchmark import (
    APPENDIX_BASELINES,
    PAPER_ARXIV_ID,
    iter_appendix_manifest,
)
from ts_rag.benchmark import DATASETS


def main():
    parser = argparse.ArgumentParser(
        description="Build the TimeFuse appendix baseline-system matrix"
    )
    parser.add_argument(
        "--output",
        default="./docs/timefuse_appendix_experiment_manifest.jsonl",
    )
    args = parser.parse_args()

    rows = list(iter_appendix_manifest())
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    temporary = args.output + ".tmp"
    with open(temporary, "w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, sort_keys=True) + "\n")
    os.replace(temporary, args.output)

    print(
        json.dumps(
            {
                "arxiv_id": PAPER_ARXIV_ID,
                "datasets": len(DATASETS),
                "baseline_systems": len(APPENDIX_BASELINES),
                "reported_cells": len(rows),
                "evaluable_cells": sum(
                    row["paper_status"]["evaluable"] for row in rows
                ),
                "families": dict(
                    sorted(Counter(row["task_family"] for row in rows).items())
                ),
                "paper_status": dict(
                    sorted(
                        Counter(
                            row["paper_status"]["status"] for row in rows
                        ).items()
                    )
                ),
                "output": args.output,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
