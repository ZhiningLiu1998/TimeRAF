#!/usr/bin/env python3
"""Development-set diagnostics for the unified retrieval portfolio."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", required=True)
    args = parser.parse_args()

    summary = json.loads(Path(args.summary).read_text())
    rows = [row for row in summary["cell_states"] if row.get("state") == "completed"]

    reference, candidate = args.reference, args.candidate
    print(f"{candidate} versus {reference}\n")
    for family in ("long_term", "pems", "epf"):
        subset = [row for row in rows if row["task_family"] == family]
        if not subset:
            continue
        lost = [
            row
            for row in subset
            if row["rungs"][reference]["strict_win"]
            and not row["rungs"][candidate]["strict_win"]
        ]
        gained = [
            row
            for row in subset
            if row["rungs"][candidate]["strict_win"]
            and not row["rungs"][reference]["strict_win"]
        ]
        worse = [
            row
            for row in subset
            if min(row["rungs"][candidate]["metric_gain_percent"].values())
            < min(row["rungs"][reference]["metric_gain_percent"].values())
        ]
        print(
            f"{family:10s} cells={len(subset):3d} lost={len(lost):3d} "
            f"gained={len(gained):3d} worse_worst_metric={len(worse):3d}"
        )
        selected = Counter(
            (
                row["rungs"][candidate]["selected"]["method"],
                row["rungs"][candidate]["selected"]["params"].get("beta"),
            )
            for row in subset
        )
        print(f"           selected: {dict(selected)}")
        drift_rows = [
            row
            for row in subset
            if row["rungs"][candidate]["selected"]["method"] == "residual_drift"
        ]
        if drift_rows:
            delta = [
                min(row["rungs"][candidate]["metric_gain_percent"].values())
                - min(row["rungs"][reference]["metric_gain_percent"].values())
                for row in drift_rows
            ]
            print(
                f"           drift cells={len(drift_rows):3d} "
                f"median worst-metric delta={np.median(delta):+.3f}pp "
                f"helped={sum(1 for value in delta if value > 0)} "
                f"hurt={sum(1 for value in delta if value < 0)}"
            )
        if lost:
            print("           lost cells:")
            for row in lost[:12]:
                selection = row["rungs"][candidate]["selected"]
                params = {
                    key: selection["params"].get(key)
                    for key in ("alpha", "beta", "ramp")
                }
                print(
                    f"             {row['cell_id']:44s} {selection['method']:20s} {params} "
                    f"gain={min(row['rungs'][candidate]['metric_gain_percent'].values()):+.3f} "
                    f"ref={min(row['rungs'][reference]['metric_gain_percent'].values()):+.3f}"
                )
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
