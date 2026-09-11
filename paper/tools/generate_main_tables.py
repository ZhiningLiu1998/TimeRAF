#!/usr/bin/env python3
"""Generate paired main-paper tables from the accepted A100 matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable


EXPECTED_SOURCE_SHA256 = (
    "924d5e851f7f940d6ef7b7d3503c5281f1d744a41320254b36950f40684952de"
)

MODELS = [
    "TimeXer",
    "TimeMixer",
    "PAttn",
    "iTransformer",
    "TimesNet",
    "PatchTST",
    "DLinear",
    "FreTS",
    "FEDformer",
    "Nonstationary_Transformer",
    "LightTS",
    "Informer",
    "Autoformer",
]

MODEL_LABELS = {
    "Nonstationary_Transformer": "Non-stat. Transformer",
}

SCOPE = {
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
        "horizons": [96, 192, 336, 720],
        "metrics": ["mse", "mae"],
    },
    "pems": {
        "datasets": ["PEMS03", "PEMS04", "PEMS07", "PEMS08"],
        "horizons": [6, 12, 24],
        "metrics": ["mae", "rmse", "mape"],
    },
    "epf": {
        "datasets": ["NP", "PJM", "BE", "FR", "DE"],
        "horizons": [24],
        "metrics": ["mse", "mae"],
    },
}

DATASET_LABELS = {
    "weather": "Weather",
    "electricity": "Electricity",
    "traffic": "Traffic",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--expected-sha256",
        default=EXPECTED_SOURCE_SHA256,
        help="Expected SHA-256 of the finalized composed summary.",
    )
    parser.add_argument(
        "--paper-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    return parser.parse_args()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def expected_cell_ids() -> set[str]:
    result = set()
    for family, scope in SCOPE.items():
        for dataset in scope["datasets"]:
            for model in MODELS:
                for horizon in scope["horizons"]:
                    result.add(f"{family}/{dataset}/{model}/{horizon}")
    return result


def validate_rows(document: dict[str, Any]) -> list[dict[str, Any]]:
    rows = document.get("cell_states")
    if not isinstance(rows, list):
        raise ValueError("cell_states must be a list")
    if len(rows) != 585:
        raise ValueError(f"expected 585 rows, found {len(rows)}")

    seen: set[str] = set()
    for row in rows:
        cell_id = row["cell_id"]
        if cell_id in seen:
            raise ValueError(f"duplicate cell: {cell_id}")
        seen.add(cell_id)
        if row["state"] != "completed":
            raise ValueError(f"non-completed cell: {cell_id}")

        family = row["task_family"]
        if family not in SCOPE:
            raise ValueError(f"unexpected family in {cell_id}: {family}")
        scope = SCOPE[family]
        if row["dataset"] not in scope["datasets"]:
            raise ValueError(f"unexpected dataset in {cell_id}")
        if row["model"] not in MODELS:
            raise ValueError(f"unexpected model in {cell_id}")
        if row["pred_len"] not in scope["horizons"]:
            raise ValueError(f"unexpected horizon in {cell_id}")

        metrics = set(scope["metrics"])
        for key in ("test_baseline", "test_corrected", "metric_gain_percent"):
            if set(row[key]) != metrics:
                raise ValueError(f"{key} metric mismatch in {cell_id}")
            if not all(math.isfinite(value) for value in row[key].values()):
                raise ValueError(f"non-finite {key} in {cell_id}")

        strict = True
        for metric in scope["metrics"]:
            baseline = row["test_baseline"][metric]
            corrected = row["test_corrected"][metric]
            if baseline <= 0:
                raise ValueError(f"non-positive baseline in {cell_id}")
            expected_gain = 100.0 * (baseline - corrected) / baseline
            if not math.isclose(
                expected_gain,
                row["metric_gain_percent"][metric],
                rel_tol=1e-10,
                abs_tol=1e-10,
            ):
                raise ValueError(f"gain mismatch in {cell_id}/{metric}")
            strict = strict and corrected < baseline
        if strict != row["all_test_metrics_improve"]:
            raise ValueError(f"strict-win mismatch in {cell_id}")

    missing = expected_cell_ids() - seen
    extra = seen - expected_cell_ids()
    if missing or extra:
        raise ValueError(
            f"matrix identity mismatch: {len(missing)} missing, {len(extra)} extra"
        )
    return rows


def aggregate(
    rows: Iterable[dict[str, Any]], keys: tuple[str, ...]
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)

    result = []
    for identity, members in groups.items():
        family = members[0]["task_family"]
        metrics = SCOPE[family]["metrics"]
        record: dict[str, Any] = dict(zip(keys, identity))
        record.update(
            {
                "cells": len(members),
                "strict_wins": sum(
                    bool(member["all_test_metrics_improve"]) for member in members
                ),
                "baseline": {},
                "corrected": {},
                "mean_cell_gain_percent": {},
            }
        )
        for metric in metrics:
            record["baseline"][metric] = fmean(
                member["test_baseline"][metric] for member in members
            )
            record["corrected"][metric] = fmean(
                member["test_corrected"][metric] for member in members
            )
            record["mean_cell_gain_percent"][metric] = fmean(
                member["metric_gain_percent"][metric] for member in members
            )
        result.append(record)
    return result


def index_by(
    records: Iterable[dict[str, Any]], keys: tuple[str, ...]
) -> dict[tuple[Any, ...], dict[str, Any]]:
    return {tuple(record[key] for key in keys): record for record in records}


def fmt_metric(metric: str, value: float) -> str:
    return f"{value:.3f}"


def fmt_gain(value: float) -> str:
    return f"{value:.1f}"


def dataset_label(dataset: str) -> str:
    return DATASET_LABELS.get(dataset, dataset)


def model_label(model: str) -> str:
    return MODEL_LABELS.get(model, model)


def fmt_absolute_metric(family: str, value: float) -> str:
    decimals = 3 if family == "pems" else 4
    return f"{value:.{decimals}f}"


def render_full_cell_tables(rows: list[dict[str, Any]]) -> str:
    by_cell = index_by(
        rows,
        ("task_family", "dataset", "model", "pred_len"),
    )
    tables = [
        "% Generated by paper/tools/generate_main_tables.py; do not edit.",
        "% Every accepted primary-matrix cell appears exactly once below.",
    ]

    for family, scope in SCOPE.items():
        for dataset in scope["datasets"]:
            metrics = scope["metrics"]
            horizons = scope["horizons"]
            columns = "ll" + "r" * (len(metrics) * len(horizons))
            task_label = {
                "long_term": "long-term",
                "pems": "PEMS",
                "epf": "EPF",
            }[family]
            label_dataset = dataset.lower().replace("_", "-")
            lines = [
                "\\begin{table*}[p]",
                "\\centering",
                "\\scriptsize",
                "\\setlength{\\tabcolsep}{3pt}",
                (
                    f"\\caption{{Complete {task_label} results on "
                    f"{dataset_label(dataset)}. Base and +\\method{{}} are "
                    "reported separately for every backbone and horizon; "
                    "bold marks a lower corrected error. No result in this "
                    "table is averaged over backbones or horizons.}"
                ),
                f"\\label{{tab:full-{family.replace('_', '-')}-{label_dataset}}}",
                f"\\begin{{tabular}}{{{columns}}}",
                "\\toprule",
            ]

            if len(horizons) > 1:
                spans = " & ".join(
                    f"\\multicolumn{{{len(metrics)}}}{{c}}{{{horizon}}}"
                    for horizon in horizons
                )
                lines.append(f"Backbone & System & {spans} \\\\")
                start = 3
                rules = []
                for _ in horizons:
                    end = start + len(metrics) - 1
                    rules.append(f"\\cmidrule(lr){{{start}-{end}}}")
                    start = end + 1
                lines.append(" ".join(rules))
                metric_header = " & ".join(
                    metric.upper()
                    for _ in horizons
                    for metric in metrics
                )
                lines.append(f" & & {metric_header} \\\\")
            else:
                lines.append(
                    "Backbone & System & "
                    + " & ".join(metric.upper() for metric in metrics)
                    + " \\\\"
                )
            lines.append("\\midrule")

            for model_index, model in enumerate(MODELS):
                baseline_values = []
                corrected_values = []
                for horizon in horizons:
                    row = by_cell[(family, dataset, model, horizon)]
                    for metric in metrics:
                        baseline = row["test_baseline"][metric]
                        corrected = row["test_corrected"][metric]
                        baseline_values.append(
                            fmt_absolute_metric(family, baseline)
                        )
                        corrected_text = fmt_absolute_metric(family, corrected)
                        if corrected < baseline:
                            corrected_text = f"\\textbf{{{corrected_text}}}"
                        corrected_values.append(corrected_text)
                lines.append(
                    f"{model_label(model)} & Base & "
                    + " & ".join(baseline_values)
                    + " \\\\"
                )
                lines.append(
                    " & +\\method{} & "
                    + " & ".join(corrected_values)
                    + " \\\\"
                )
                if model_index + 1 != len(MODELS):
                    lines.append("\\addlinespace[1pt]")

            lines += [
                "\\bottomrule",
                "\\end{tabular}",
                "\\end{table*}",
                "",
            ]
            tables.extend(lines)

    return "\n".join(tables)


def render_horizon_table(
    family: str,
    records: list[dict[str, Any]],
) -> str:
    if family == "pems":
        return render_pems_horizon_tables(records)

    scope = SCOPE[family]
    by_horizon = index_by(records, ("task_family", "dataset", "pred_len"))
    metrics = scope["metrics"]
    columns = "ll" + "r" * (3 * len(metrics)) + "r"
    label = f"tab:{family.replace('_', '-')}-horizons"
    task_name = "long-term" if family == "long_term" else "PEMS"

    lines = [
        "% Generated by paper/tools/generate_main_tables.py; do not edit.",
        "\\begin{table*}[p]",
        "\\centering",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{3pt}",
        (
            f"\\caption{{Full {task_name} results by prediction length. "
            "Absolute metrics average the same 13 paired backbones; "
            "$\\Delta$ is the mean cell-wise relative reduction (\\%).}"
        ),
        f"\\label{{{label}}}",
    ]
    lines += [
        f"\\begin{{tabular}}{{{columns}}}",
        "\\toprule",
        (
            "Dataset & Horizon & "
            + " & ".join(
                f"Base {metric.upper()} & +\\method{{}} {metric.upper()} & "
                f"$\\Delta$ {metric.upper()}"
                for metric in metrics
            )
            + " & Wins \\\\"
        ),
        "\\midrule",
    ]

    for dataset_index, dataset in enumerate(scope["datasets"]):
        for horizon in scope["horizons"]:
            record = by_horizon[(family, dataset, horizon)]
            values = []
            for metric in metrics:
                values += [
                    fmt_metric(metric, record["baseline"][metric]),
                    f"\\textbf{{{fmt_metric(metric, record['corrected'][metric])}}}",
                    fmt_gain(record["mean_cell_gain_percent"][metric]),
                ]
            values.append(f"{record['strict_wins']}/{record['cells']}")
            lines.append(
                f"{dataset_label(dataset)} & {horizon} & "
                + " & ".join(values)
                + " \\\\"
            )
        if dataset_index + 1 != len(scope["datasets"]):
            lines.append("\\midrule")

    lines += ["\\bottomrule", "\\end{tabular}"]
    lines += ["\\end{table*}", ""]
    return "\n".join(lines)


def render_pems_horizon_tables(records: list[dict[str, Any]]) -> str:
    family = "pems"
    scope = SCOPE[family]
    by_horizon = index_by(records, ("task_family", "dataset", "pred_len"))
    tables: list[str] = [
        "% Generated by paper/tools/generate_main_tables.py; do not edit."
    ]

    for metric_index, metric in enumerate(scope["metrics"]):
        label = (
            "tab:pems-horizons"
            if metric_index == 0
            else f"tab:pems-horizons-{metric}"
        )
        lines = [
            "\\begin{table*}[p]",
            "\\centering",
            "\\footnotesize",
            "\\setlength{\\tabcolsep}{6pt}",
            (
                f"\\caption{{Full PEMS {metric.upper()} results by prediction "
                "length. Absolute metrics average the same 13 paired "
                "backbones; $\\Delta$ is the mean cell-wise relative reduction "
                "(\\%). Strict wins require all three PEMS metrics and therefore "
                "repeat across these tables.}"
            ),
            f"\\label{{{label}}}",
            "\\begin{tabular}{lrrrrr}",
            "\\toprule",
            (
                f"Dataset & Horizon & Base {metric.upper()} & "
                f"+\\method{{}} {metric.upper()} & $\\Delta$ {metric.upper()} "
                "& Strict wins \\\\"
            ),
            "\\midrule",
        ]
        for dataset_index, dataset in enumerate(scope["datasets"]):
            for horizon in scope["horizons"]:
                record = by_horizon[(family, dataset, horizon)]
                lines.append(
                    "{} & {} & {} & \\textbf{{{}}} & {} & {}/{} \\\\".format(
                        dataset_label(dataset),
                        horizon,
                        fmt_metric(metric, record["baseline"][metric]),
                        fmt_metric(metric, record["corrected"][metric]),
                        fmt_gain(record["mean_cell_gain_percent"][metric]),
                        record["strict_wins"],
                        record["cells"],
                    )
                )
            if dataset_index + 1 != len(scope["datasets"]):
                lines.append("\\midrule")
        lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""]
        tables.extend(lines)

    return "\n".join(tables)


def render_backbone_table(records: list[dict[str, Any]]) -> str:
    by_model = index_by(records, ("task_family", "model"))
    lines = [
        "% Generated by paper/tools/generate_main_tables.py; do not edit.",
        "\\begin{table}[h!]",
        "\\centering",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{2pt}",
        (
            "\\caption{Backbone-level paired results. Entries are strict wins "
            "and mean cell-wise relative reductions (\\%); positive values "
            "favor \\method{}. This view prevents weak backbones or large "
            "absolute errors from dominating the conclusion.}"
        ),
        "\\label{tab:backbone-results}",
        "\\begin{tabular}{lrrr|rrrr|rrr}",
        "\\toprule",
        (
            " & \\multicolumn{3}{c|}{Long-term (28 cells)} & "
            "\\multicolumn{4}{c|}{PEMS (12 cells)} & "
            "\\multicolumn{3}{c}{EPF (5 cells)} \\\\"
        ),
        "\\cmidrule(lr){2-4} \\cmidrule(lr){5-8} \\cmidrule(lr){9-11}",
        (
            "Backbone & Wins & MSE & MAE & Wins & MAE & RMSE & MAPE & "
            "Wins & MSE & MAE \\\\"
        ),
        "\\midrule",
    ]

    for model in MODELS:
        long_term = by_model[("long_term", model)]
        pems = by_model[("pems", model)]
        epf = by_model[("epf", model)]
        lines.append(
            "{} & {}/{} & {} & {} & {}/{} & {} & {} & {} & {}/{} & {} & {} "
            "\\\\".format(
                model_label(model),
                long_term["strict_wins"],
                long_term["cells"],
                fmt_gain(long_term["mean_cell_gain_percent"]["mse"]),
                fmt_gain(long_term["mean_cell_gain_percent"]["mae"]),
                pems["strict_wins"],
                pems["cells"],
                fmt_gain(pems["mean_cell_gain_percent"]["mae"]),
                fmt_gain(pems["mean_cell_gain_percent"]["rmse"]),
                fmt_gain(pems["mean_cell_gain_percent"]["mape"]),
                epf["strict_wins"],
                epf["cells"],
                fmt_gain(epf["mean_cell_gain_percent"]["mse"]),
                fmt_gain(epf["mean_cell_gain_percent"]["mae"]),
            )
        )

    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    source_bytes = args.input.read_bytes()
    source_hash = sha256_bytes(source_bytes)
    if source_hash != args.expected_sha256:
        raise ValueError(
            f"source SHA-256 mismatch: expected {args.expected_sha256}, "
            f"found {source_hash}"
        )

    document = json.loads(source_bytes)
    rows = validate_rows(document)
    dataset_records = aggregate(rows, ("task_family", "dataset"))
    horizon_records = aggregate(rows, ("task_family", "dataset", "pred_len"))
    backbone_records = aggregate(rows, ("task_family", "model"))

    table_dir = args.paper_root / "latex" / "tables"
    data_dir = args.paper_root / "data"
    table_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    for family, filename in (
        ("long_term", "longterm_horizons.tex"),
        ("pems", "pems_horizons.tex"),
    ):
        (table_dir / filename).write_text(
            render_horizon_table(family, horizon_records),
            encoding="utf-8",
        )

    (table_dir / "backbone_summary.tex").write_text(
        render_backbone_table(backbone_records),
        encoding="utf-8",
    )
    (table_dir / "full_cell_results.tex").write_text(
        render_full_cell_tables(rows),
        encoding="utf-8",
    )

    receipt = {
        "schema_version": 1,
        "source": {
            "filename": args.input.name,
            "sha256": source_hash,
            "size_bytes": len(source_bytes),
            "source_revision": document["source_revision"],
            "seed": document["seed"],
        },
        "validation": {
            "expected_cells": 585,
            "validated_cells": len(rows),
            "unique_cell_ids": len({row["cell_id"] for row in rows}),
            "all_completed_and_finite": True,
            "strict_wins": sum(
                bool(row["all_test_metrics_improve"]) for row in rows
            ),
        },
        "aggregation": {
            "absolute_metrics": (
                "arithmetic mean over equally weighted paired cells in each group"
            ),
            "relative_reduction": (
                "arithmetic mean of per-cell 100*(baseline-corrected)/baseline"
            ),
            "strict_win": "corrected is lower than baseline for every cell metric",
        },
        "dataset": dataset_records,
        "dataset_horizon": horizon_records,
        "backbone": backbone_records,
        "cell": rows,
    }
    (data_dir / "paired_table_results.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
