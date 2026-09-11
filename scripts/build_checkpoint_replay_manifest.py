"""Select the 208 base-model cells needed by the Appendix reproduction."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


REPLAY_HORIZONS = {
    "long_term": 96,
    "pems": 24,
    "epf": 24,
}
EXPECTED_CELLS = 208


def select_checkpoint_replay_cells(rows):
    selected = [
        row
        for row in rows
        if row["task_family"] in REPLAY_HORIZONS
        and row["pred_len"] == REPLAY_HORIZONS[row["task_family"]]
    ]
    if len(selected) != EXPECTED_CELLS:
        raise ValueError(
            f"Expected {EXPECTED_CELLS} checkpoint replay cells, "
            f"found {len(selected)}"
        )
    return selected


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Build the Appendix-horizon checkpoint replay manifest"
    )
    parser.add_argument(
        "--input", default="docs/timefuse_experiment_manifest.jsonl"
    )
    parser.add_argument(
        "--output",
        default="docs/timefuse_checkpoint_replay_manifest.jsonl",
    )
    args = parser.parse_args(argv)

    with open(args.input, "r", encoding="utf-8") as source:
        rows = [json.loads(line) for line in source if line.strip()]
    selected = select_checkpoint_replay_cells(rows)
    payload = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        for row in selected
    ).encode("utf-8")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(payload)
    print(
        json.dumps(
            {
                "output": str(output),
                "cells": len(selected),
                "sha256": hashlib.sha256(payload).hexdigest(),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
