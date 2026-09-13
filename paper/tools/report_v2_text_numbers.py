#!/usr/bin/env python3
"""Print every number the experiments and method text quotes, from one source."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean, median

FAMILIES = ("long_term", "pems", "epf")
SYSTEMS = (
    "base",
    "analog_future",
    "raft_adapted",
    "saraf_adapted",
    "residual_retrieval",
    "timeraf_no_folds",
    "timeraf_no_drift",
    "timeraf_horizon_prior",
    "timeraf",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("docs/publication_results/retrieval_v2_summary.json"),
    )
    args = parser.parse_args()
    cells = json.loads(args.summary.read_text())["cell_states"]

    print("== strict wins and median worst-metric gain")
    for system in SYSTEMS:
        parts = []
        for family in (*FAMILIES, "all"):
            rows = [
                c
                for c in cells
                if family == "all" or c["task_family"] == family
            ]
            wins = sum(1 for c in rows if c["systems"][system]["strict_win"])
            worst = [
                min(c["systems"][system]["metric_gain_percent"].values())
                for c in rows
            ]
            parts.append(f"{family}={wins}/{len(rows)} med={median(worst):+.2f}")
        print(f"  {system:26s} " + "  ".join(parts))

    print("\n== per-metric family medians and means")
    for family in FAMILIES:
        rows = [c for c in cells if c["task_family"] == family]
        metrics = sorted(rows[0]["systems"]["timeraf"]["metric_gain_percent"])
        print(f"  {family}")
        for system in SYSTEMS:
            summary = {
                metric: (
                    round(
                        median(
                            c["systems"][system]["metric_gain_percent"][metric]
                            for c in rows
                        ),
                        2,
                    ),
                    round(
                        fmean(
                            c["systems"][system]["metric_gain_percent"][metric]
                            for c in rows
                        ),
                        2,
                    ),
                )
                for metric in metrics
            }
            print(f"    {system:26s} median/mean {summary}")

    print("\n== dataset-level wins for the method (both metrics best)")
    for family in FAMILIES:
        rows = [c for c in cells if c["task_family"] == family]
        datasets = sorted({c["dataset"] for c in rows})
        both = []
        for dataset in datasets:
            subset = [c for c in rows if c["dataset"] == dataset]
            metrics = sorted(subset[0]["systems"]["base"]["test_metrics"])
            best_all = True
            for metric in metrics:
                means = {
                    system: fmean(
                        c["systems"][system]["test_metrics"][metric] for c in subset
                    )
                    for system in SYSTEMS
                    if system in ("base", "analog_future", "raft_adapted", "saraf_adapted", "residual_retrieval", "timeraf")
                }
                if min(means, key=means.get) != "timeraf":
                    best_all = False
            both.append((dataset, best_all))
        print(f"  {family}: method best on both metrics for "
              f"{sum(1 for _, ok in both if ok)}/{len(both)} datasets "
              f"-> {[d for d, ok in both if not ok]} not best")

    print("\n== selected candidate composition for the method")
    for family in (*FAMILIES, "all"):
        rows = [c for c in cells if family == "all" or c["task_family"] == family]
        methods = Counter(c["systems"]["timeraf"]["selected"]["method"] for c in rows)
        betas = Counter(
            c["systems"]["timeraf"]["selected"]["params"].get("beta")
            for c in rows
            if c["systems"]["timeraf"]["selected"]["method"] == "residual_drift"
        )
        ramps = Counter(
            c["systems"]["timeraf"]["selected"]["params"].get("ramp")
            for c in rows
            if c["systems"]["timeraf"]["selected"]["method"] == "residual_drift"
        )
        print(f"  {family:10s} methods={dict(methods)}")
        print(f"             betas={dict(betas)} ramps={dict(ramps)}")

    print("\n== drift selection by horizon")
    by_horizon = defaultdict(lambda: [0, 0])
    for cell in cells:
        entry = by_horizon[cell["pred_len"]]
        entry[1] += 1
        entry[0] += int(
            cell["systems"]["timeraf"]["selected"]["method"] == "residual_drift"
        )
    for horizon in sorted(by_horizon):
        drift, total = by_horizon[horizon]
        print(f"  horizon {horizon:3d}: drift selected {drift}/{total}")

    print("\n== strongest five backbones by long-term base MSE")
    base_mse = {}
    for model in sorted({c["model"] for c in cells}):
        rows = [
            c
            for c in cells
            if c["model"] == model and c["task_family"] == "long_term"
        ]
        base_mse[model] = fmean(c["systems"]["base"]["test_metrics"]["mse"] for c in rows)
    strongest = sorted(base_mse, key=base_mse.get)[:5]
    print(f"  {strongest}")
    for system in ("analog_future", "raft_adapted", "saraf_adapted", "residual_retrieval", "timeraf"):
        rows = [c for c in cells if c["model"] in strongest]
        wins = sum(1 for c in rows if c["systems"][system]["strict_win"])
        print(f"  {system:22s} {wins}/{len(rows)}")

    print("\n== abstention and damage among non-strict settings")
    for system in ("raft_adapted", "saraf_adapted", "residual_retrieval", "timeraf"):
        identical = 0
        damaged = 0
        for cell in cells:
            payload = cell["systems"][system]
            if payload["strict_win"]:
                continue
            gains = payload["metric_gain_percent"].values()
            if all(abs(value) < 1e-12 for value in gains):
                identical += 1
            else:
                damaged += 1
        pems_identical = sum(
            1
            for cell in cells
            if cell["task_family"] == "pems"
            and all(
                abs(value) < 1e-12
                for value in cell["systems"][system]["metric_gain_percent"].values()
            )
        )
        print(
            f"  {system:22s} unchanged={identical} degraded_or_mixed={damaged} "
            f"unchanged_on_pems={pems_identical}"
        )

    print("\n== method versus every other system, per setting")
    for other in SYSTEMS:
        if other == "timeraf":
            continue
        better = worse = mixed = 0
        for cell in cells:
            mine = cell["systems"]["timeraf"]["test_metrics"]
            theirs = cell["systems"][other]["test_metrics"]
            if all(float(mine[k]) < float(theirs[k]) for k in mine):
                better += 1
            elif all(float(mine[k]) > float(theirs[k]) for k in mine):
                worse += 1
            else:
                mixed += 1
        print(f"  vs {other:26s} better={better} worse={worse} mixed={mixed}")


if __name__ == "__main__":
    main()
