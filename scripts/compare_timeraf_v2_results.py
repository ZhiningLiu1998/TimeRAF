#!/usr/bin/env python3
"""Compare unified-portfolio rungs against every accepted fixed-forecast system.

Both inputs revise byte-identical frozen base forecasts, so the recorded
585-cell comparison and the v2 rungs are directly paired per cell.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

FAMILIES = ("long_term", "pems", "epf")
RECORDED_SYSTEMS = (
    "analog_future",
    "raft_adapted",
    "saraf_adapted",
    "residual_retrieval",
    "timeraf",
)


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        while chunk := source.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _block(entries):
    if not entries:
        return None
    worst = [min(entry["gains"].values()) for entry in entries]
    metrics = sorted({name for entry in entries for name in entry["gains"]})
    return {
        "cells": len(entries),
        "strict_wins": sum(1 for entry in entries if entry["strict_win"]),
        "median_worst_metric_gain_percent": float(np.median(worst)),
        "mean_worst_metric_gain_percent": float(np.mean(worst)),
        "median_gain_percent": {
            name: float(np.median([e["gains"][name] for e in entries if name in e["gains"]]))
            for name in metrics
        },
        "mean_gain_percent": {
            name: float(np.mean([e["gains"][name] for e in entries if name in e["gains"]]))
            for name in metrics
        },
    }


def _grouped(entries, dev_cells):
    return {
        "overall": _block(entries),
        "by_task_family": {
            family: _block([e for e in entries if e["task_family"] == family])
            for family in FAMILIES
        },
        "splits": {
            "development": _block([e for e in entries if e["cell_id"] in dev_cells]),
            "confirmatory": _block(
                [e for e in entries if e["cell_id"] not in dev_cells]
            ),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v2-summary", required=True)
    parser.add_argument("--recorded-summary", required=True)
    parser.add_argument("--development-cells", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    v2 = json.loads(Path(args.v2_summary).read_text())
    recorded = json.loads(Path(args.recorded_summary).read_text())
    dev_cells = {
        line
        for line in Path(args.development_cells).read_text().split("\n")
        if line
    }

    recorded_cells = {row["cell_id"]: row for row in recorded["cell_states"]}
    v2_cells = {
        row["cell_id"]: row
        for row in v2["cell_states"]
        if row.get("state") == "completed"
    }

    # Paired base check: the fixed forecast must be identical in both runs.
    base_mismatches = []
    for cell_id, row in v2_cells.items():
        reference = recorded_cells[cell_id]["systems"]["base"]["test_metrics"]
        for name, value in row["test_baseline"].items():
            if float(value) != float(reference[name]):
                base_mismatches.append([cell_id, name, float(value), float(reference[name])])

    systems = {}
    for system in RECORDED_SYSTEMS:
        entries = []
        for cell_id in v2_cells:
            payload = recorded_cells[cell_id]["systems"][system]
            entries.append(
                {
                    "cell_id": cell_id,
                    "task_family": recorded_cells[cell_id]["task_family"],
                    "gains": payload["metric_gain_percent"],
                    "strict_win": bool(payload["strict_win"]),
                    "metrics": payload["test_metrics"],
                }
            )
        systems[system] = {"entries": entries, **_grouped(entries, dev_cells)}

    for rung_id in v2["rungs"]:
        entries = []
        for cell_id, row in v2_cells.items():
            outcome = row["rungs"][rung_id]
            entries.append(
                {
                    "cell_id": cell_id,
                    "task_family": row["task_family"],
                    "gains": outcome["metric_gain_percent"],
                    "strict_win": bool(outcome["strict_win"]),
                    "metrics": outcome["test_corrected"],
                }
            )
        systems[f"v2:{rung_id}"] = {"entries": entries, **_grouped(entries, dev_cells)}

    # Per-setting dominance: strictly lower on every reported metric.
    dominance = {}
    for name, payload in systems.items():
        if not name.startswith("v2:"):
            continue
        mine = {entry["cell_id"]: entry for entry in payload["entries"]}
        dominance[name] = {}
        for other, other_payload in systems.items():
            if other == name:
                continue
            theirs = {entry["cell_id"]: entry for entry in other_payload["entries"]}
            wins = ties = losses = 0
            for cell_id, entry in mine.items():
                metrics = entry["metrics"]
                reference = theirs[cell_id]["metrics"]
                better = all(
                    float(metrics[key]) < float(reference[key]) for key in metrics
                )
                worse = all(
                    float(metrics[key]) > float(reference[key]) for key in metrics
                )
                wins += better
                losses += worse
                ties += not better and not worse
            dominance[name][other] = {
                "strictly_better_cells": wins,
                "strictly_worse_cells": losses,
                "mixed_or_equal_cells": ties,
            }

    baseline = systems["timeraf"]
    targets = {}
    for name, payload in systems.items():
        if not name.startswith("v2:"):
            continue
        family_ok = {}
        for family in FAMILIES:
            mine = payload["by_task_family"][family]
            if mine is None:
                continue
            best_median = max(
                systems[other]["by_task_family"][family][
                    "median_worst_metric_gain_percent"
                ]
                for other in RECORDED_SYSTEMS
            )
            family_ok[family] = {
                "strict_wins": mine["strict_wins"],
                "timeraf_strict_wins": baseline["by_task_family"][family][
                    "strict_wins"
                ],
                "strict_wins_at_least_timeraf": mine["strict_wins"]
                >= baseline["by_task_family"][family]["strict_wins"],
                "median_worst_metric_gain_percent": mine[
                    "median_worst_metric_gain_percent"
                ],
                "best_recorded_median_worst_metric_gain_percent": best_median,
                "median_at_least_best_recorded": mine[
                    "median_worst_metric_gain_percent"
                ]
                >= best_median,
                "per_metric_median_at_least_timeraf": {
                    metric: mine["median_gain_percent"][metric]
                    >= baseline["by_task_family"][family]["median_gain_percent"][metric]
                    for metric in mine["median_gain_percent"]
                },
            }
        overall_ok = (
            payload["overall"]["strict_wins"] >= baseline["overall"]["strict_wins"]
        )
        targets[name] = {
            "overall_strict_wins": payload["overall"]["strict_wins"],
            "timeraf_overall_strict_wins": baseline["overall"]["strict_wins"],
            "overall_strict_wins_at_least_timeraf": overall_ok,
            "by_task_family": family_ok,
            "all_targets_met": overall_ok
            and all(
                entry["strict_wins_at_least_timeraf"]
                and entry["median_at_least_best_recorded"]
                and all(entry["per_metric_median_at_least_timeraf"].values())
                for entry in family_ok.values()
            ),
        }

    report = {
        "v2_summary": str(Path(args.v2_summary).resolve()),
        "v2_summary_sha256": sha256_file(args.v2_summary),
        "recorded_summary": str(Path(args.recorded_summary).resolve()),
        "recorded_summary_sha256": sha256_file(args.recorded_summary),
        "cells_compared": len(v2_cells),
        "base_forecast_mismatches": base_mismatches,
        "no_lookahead_all_passed": bool(v2["no_lookahead_all_passed"]),
        "systems": {
            name: {key: value for key, value in payload.items() if key != "entries"}
            for name, payload in systems.items()
        },
        "per_setting_dominance": dominance,
        "acceptance_targets": targets,
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2, sort_keys=True))

    order = list(RECORDED_SYSTEMS) + [f"v2:{rung}" for rung in v2["rungs"]]
    lines = [
        f"# Fixed-forecast comparison over {len(v2_cells)} cells",
        "",
        "Strict wins and median worst-metric paired gain against the frozen base.",
        "",
        "| system | overall | long-term | PEMS | EPF | confirmatory split |",
        "|---|---|---|---|---|---|",
    ]

    def cell(block):
        if block is None:
            return "--"
        return (
            f"{block['strict_wins']}/{block['cells']} "
            f"({block['median_worst_metric_gain_percent']:+.2f}%)"
        )

    for name in order:
        payload = systems[name]
        lines.append(
            f"| {name} | {cell(payload['overall'])} | "
            + " | ".join(
                cell(payload["by_task_family"][family]) for family in FAMILIES
            )
            + f" | {cell(payload['splits']['confirmatory'])} |"
        )
    lines += ["", "## Per-metric family medians", ""]
    for family in FAMILIES:
        metrics = sorted(
            systems["timeraf"]["by_task_family"][family]["median_gain_percent"]
        )
        lines += [
            f"### {family}",
            "",
            "| system | " + " | ".join(metrics) + " |",
            "|---" * (len(metrics) + 1) + "|",
        ]
        for name in order:
            block = systems[name]["by_task_family"][family]
            if block is None:
                continue
            lines.append(
                f"| {name} | "
                + " | ".join(
                    f"{block['median_gain_percent'][metric]:+.2f}%" for metric in metrics
                )
                + " |"
            )
        lines.append("")
    lines += ["## Per-setting dominance of the selected policy", ""]
    for name, payload in dominance.items():
        lines.append(f"### {name}")
        lines.append("")
        lines.append("| against | strictly better | strictly worse | mixed or equal |")
        lines.append("|---|---|---|---|")
        for other, counts in payload.items():
            lines.append(
                f"| {other} | {counts['strictly_better_cells']} | "
                f"{counts['strictly_worse_cells']} | {counts['mixed_or_equal_cells']} |"
            )
        lines.append("")
    Path(args.report).with_suffix(".md").write_text("\n".join(lines))

    header = f"{'system':28s} {'overall':>13s} {'long_term':>13s} {'pems':>13s} {'epf':>13s}"
    print(header)
    print("-" * len(header))
    for name in order:
        payload = systems[name]

        def cell(block):
            if block is None:
                return f"{'-':>13s}"
            return f"{block['strict_wins']:4d} {block['median_worst_metric_gain_percent']:+7.2f}%"

        print(
            f"{name:28s} {cell(payload['overall'])} "
            + " ".join(cell(payload["by_task_family"][family]) for family in FAMILIES)
        )
    print()
    for name, payload in targets.items():
        print(f"{name}: all_targets_met={payload['all_targets_met']}")
    if base_mismatches:
        print(f"WARNING: {len(base_mismatches)} base forecast mismatches")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
