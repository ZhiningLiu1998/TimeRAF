#!/usr/bin/env python3
"""Generate the main-paper retrieval-object comparison tables.

Every row averages the 585-cell fixed-forecast replay in which all six systems
receive identical base forecasts, validation splits, and chronological
memory boundaries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any

EXPECTED_SOURCE_SHA256 = (
    "9d592d0cd03a70e68181f80f29d363931bda048686750f106406d0ba0ff744be"
)

SYSTEMS: list[tuple[str, str]] = [
    ("base", "Base"),
    ("analog_future", "\\analogknn{}"),
    ("raft_adapted", "\\raft{}"),
    ("saraf_adapted", "\\saraf{}"),
    ("residual_retrieval", "\\residualknn{}"),
    ("timeraf", "\\method{}"),
]

SCOPE: dict[str, dict[str, Any]] = {
    "long_term": {
        "datasets": [
            "ETTh1",
            "ETTh2",
            "ETTm1",
            "ETTm2",
            "weather",
            "electricity",
            "traffic",
        ],
        "metrics": ["mse", "mae"],
        "cells": 364,
    },
    "pems": {
        "datasets": ["PEMS03", "PEMS04", "PEMS07", "PEMS08"],
        "metrics": ["mae", "mape"],
        "cells": 156,
    },
    "epf": {
        "datasets": ["NP", "PJM", "BE", "FR", "DE"],
        "metrics": ["mse", "mae"],
        "cells": 65,
    },
}

DATASET_LABELS = {
    "weather": "Weather",
    "electricity": "Electricity",
    "traffic": "Traffic",
}

METRIC_LABELS = {"mse": "MSE", "mae": "MAE", "mape": "MAPE", "rmse": "RMSE"}

DECIMALS = {"mse": 3, "mae": 3, "mape": 3, "rmse": 2}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "docs"
        / "publication_results"
        / "retrieval_v2_summary.json",
    )
    parser.add_argument("--expected-sha256", default=EXPECTED_SOURCE_SHA256)
    parser.add_argument(
        "--paper-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    return parser.parse_args()


def load_cells(args: argparse.Namespace) -> list[dict[str, Any]]:
    payload = args.input.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != args.expected_sha256:
        raise ValueError(f"source SHA-256 mismatch: {digest}")
    document = json.loads(payload)
    cells = document["cell_states"]
    if len(cells) != 585:
        raise ValueError(f"expected 585 cells, found {len(cells)}")
    for cell in cells:
        if cell["state"] != "completed":
            raise ValueError(f"non-completed cell {cell['cell_id']}")
        family = cell["cell_id"].split("/", 1)[0]
        if family not in SCOPE:
            raise ValueError(f"unexpected family {family}")
        for system, _ in SYSTEMS:
            metrics = cell["systems"][system]["test_metrics"]
            if not all(math.isfinite(value) for value in metrics.values()):
                raise ValueError(f"non-finite metric in {cell['cell_id']}/{system}")
    return cells


def group_means(
    cells: list[dict[str, Any]], family: str
) -> tuple[dict[str, dict[str, dict[str, float]]], dict[str, int]]:
    """Return means[dataset][system][metric] plus the cell count per dataset."""
    buckets: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    counts: dict[str, int] = defaultdict(int)
    metrics = SCOPE[family]["metrics"]
    for cell in cells:
        cell_family, dataset = cell["cell_id"].split("/")[:2]
        if cell_family != family:
            continue
        counts[dataset] += 1
        counts["__all__"] += 1
        for system, _ in SYSTEMS:
            values = cell["systems"][system]["test_metrics"]
            for metric in metrics:
                buckets[(dataset, system, metric)].append(values[metric])
                buckets[("__all__", system, metric)].append(values[metric])

    means: dict[str, dict[str, dict[str, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for (dataset, system, metric), values in buckets.items():
        means[dataset][system][metric] = fmean(values)
    return means, counts


def rank_marks(values: dict[str, float]) -> tuple[str | None, str | None]:
    ordered = sorted(values.items(), key=lambda item: item[1])
    best = ordered[0][0]
    second = ordered[1][0] if len(ordered) > 1 else None
    return best, second


def render_cell(value: float, metric: str, rank: int) -> str:
    text = f"{value:.{DECIMALS[metric]}f}"
    if rank == 0:
        return f"\\bs{{{text}}}"
    if rank == 1:
        return f"\\sbs{{{text}}}"
    return text


def render_block(
    cells: list[dict[str, Any]],
    family: str,
    average_label: str,
) -> list[str]:
    means, _ = group_means(cells, family)
    metrics = SCOPE[family]["metrics"]
    lines: list[str] = []
    order = list(SCOPE[family]["datasets"]) + ["__all__"]
    for dataset in order:
        label = (
            average_label
            if dataset == "__all__"
            else DATASET_LABELS.get(dataset, dataset)
        )
        entries: list[str] = []
        ranks: dict[str, dict[str, int]] = {system: {} for system, _ in SYSTEMS}
        for metric in metrics:
            by_system = {
                system: means[dataset][system][metric] for system, _ in SYSTEMS
            }
            best, second = rank_marks(by_system)
            for system, _ in SYSTEMS:
                if system == best:
                    ranks[system][metric] = 0
                elif system == second:
                    ranks[system][metric] = 1
                else:
                    ranks[system][metric] = 2
        for system, _ in SYSTEMS:
            entries.append(
                " ".join(
                    render_cell(
                        means[dataset][system][metric], metric, ranks[system][metric]
                    )
                    for metric in metrics
                )
            )
        prefix = "\\textbf{" + label + "}"
        if dataset == "__all__":
            lines.append("\\midrule")
            prefix = "\\textbf{" + label + "}"
        lines.append(prefix + " & " + " & ".join(entries) + " \\\\")
    return lines


def header_rows(metrics: list[str]) -> list[str]:
    metric_text = " / ".join(METRIC_LABELS[metric] for metric in metrics)
    labels = " & ".join(
        label if label.startswith("\\") else "\\textbf{" + label + "}"
        for _, label in SYSTEMS
    )
    return [
        "\\textbf{Dataset} & \\textbf{Uncorrected} & "
        "\\multicolumn{3}{c|}{\\textbf{Retrieve inputs / futures}} & "
        "\\multicolumn{2}{c}{\\textbf{Retrieve errors (ours)}} \\\\",
        "\\cmidrule(lr){3-5} \\cmidrule(lr){6-7}",
        "\\textbf{" + metric_text + "} & " + labels + " \\\\",
    ]


HEADER = [
    "\\textbf{Dataset} & \\textbf{Uncorrected} & "
    "\\multicolumn{3}{c|}{\\textbf{Retrieve inputs / futures}} & "
    "\\multicolumn{2}{c}{\\textbf{Retrieve errors (ours)}} \\\\",
    "\\cmidrule(lr){3-5} \\cmidrule(lr){6-7}",
]

MARKING = (
    "Lower is better; \\bs{best} and \\sbs{second best} per column group."
)


def wrap_table(
    caption: str, label: str, body: list[str], metric_header: str = "Metrics"
) -> str:
    labels = " & ".join(
        label if label.startswith("\\") else "\\textbf{" + label + "}"
        for _, label in SYSTEMS
    )
    return "\n".join(
        [
            "% Generated by paper/tools/generate_comparison_tables.py; do not edit.",
            "\\begin{table}[t]",
            "\\centering",
            "\\caption{" + caption + "}",
            "\\label{" + label + "}",
            "\\resizebox{\\textwidth}{!}{%",
            "\\setlength{\\tabcolsep}{3pt}",
            "\\begin{tabular}{@{}l|c|c|c|c|c|c@{}}",
            "\\toprule",
            *HEADER,
            "\\textbf{" + metric_header + "} & " + labels + " \\\\",
            "\\midrule",
            *body,
            "\\bottomrule",
            "\\end{tabular}%",
            "}",
            "\\end{table}",
            "",
        ]
    )


def render_long_term(cells: list[dict[str, Any]]) -> str:
    body = list(render_block(cells, "long_term", "Average"))
    caption = (
        "\\textbf{Long-term forecasting.} Each entry averages 13 backbones and "
        "horizons $\\{96,192,336,720\\}$ over the \\emph{same} 364 base forecasts; "
        "\\analogknn{} and \\residualknn{} pin Equation~\\ref{eq:dial} at $\\beta=1$ "
        "and $\\beta=0$. " + MARKING
    )
    return wrap_table(caption, "tab:main-long-term", body, "MSE / MAE")


def render_short_term(cells: list[dict[str, Any]]) -> str:
    body: list[str] = []
    blocks = [
        ("pems", "PEMS traffic flow, MAE / MAPE, inverse-scaled (156 settings)"),
        ("epf", "EPF electricity price, MSE / MAE (65 settings)"),
    ]
    for index, (family, title) in enumerate(blocks):
        if index:
            body.append("\\midrule")
        body.append("\\multicolumn{7}{@{}l}{\\textit{" + title + "}} \\\\")
        body.extend(render_block(cells, family, "Average"))
    caption = (
        "\\textbf{Short-term forecasting.} Same protocol and columns on the PEMS "
        "networks (13 backbones, horizons $\\{6,12,24\\}$) and the five EPF markets "
        "(horizon 24); PEMS RMSE is in Appendix~\\ref{app:retrieval-baselines}. "
        + MARKING
    )
    return wrap_table(caption, "tab:main-short-term", body)


def main() -> None:
    args = parse_args()
    cells = load_cells(args)
    table_dir = args.paper_root / "latex" / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    (table_dir / "main_long_term.tex").write_text(
        render_long_term(cells), encoding="utf-8"
    )
    (table_dir / "main_short_term.tex").write_text(
        render_short_term(cells), encoding="utf-8"
    )
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "source": {
            "filename": args.input.name,
            "sha256": args.expected_sha256,
        },
        "aggregation": "arithmetic mean of matched per-cell test metrics",
        "systems": [system for system, _ in SYSTEMS],
        "blocks": {},
    }
    for family in SCOPE:
        means, counts = group_means(cells, family)
        receipt["blocks"][family] = {
            "cells": counts["__all__"],
            "per_dataset_cells": {
                key: value for key, value in counts.items() if key != "__all__"
            },
            "means": {
                dataset: {system: dict(values) for system, values in systems.items()}
                for dataset, systems in means.items()
            },
        }
    data_dir = args.paper_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "comparison_table_results.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
