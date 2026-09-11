#!/usr/bin/env python3
"""Generate publication tables for the additional-system experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


EXPECTED_SHA256 = "e070233b7719277674f2cfc6a890da9396207cc699b7234420fda7a8f3c78098"

SYSTEMS = (
    "forward_selection",
    "portfolio_ensemble",
    "zeroshot_ensemble",
    "autogluon_high_quality",
    "chronos_bolt_finetuned",
    "chronos_bolt_zeroshot",
)

SYSTEM_LABELS = {
    "forward_selection": "Forward Selection",
    "portfolio_ensemble": "Portfolio Ensemble",
    "zeroshot_ensemble": "ZeroShot Ensemble",
    "autogluon_high_quality": "AutoGluon HQ",
    "chronos_bolt_finetuned": "Chronos-Bolt FT",
    "chronos_bolt_zeroshot": "Chronos-Bolt ZS",
}

SYSTEM_HEADERS = {
    "forward_selection": r"\shortstack{Forward\\Selection}",
    "portfolio_ensemble": r"\shortstack{Portfolio\\Ensemble}",
    "zeroshot_ensemble": r"\shortstack{ZeroShot\\Ensemble}",
    "autogluon_high_quality": r"\shortstack{AutoGluon\\HQ}",
    "chronos_bolt_finetuned": r"\shortstack{Chronos-Bolt\\FT}",
    "chronos_bolt_zeroshot": r"\shortstack{Chronos-Bolt\\ZS}",
}

TASKS = {
    "long_term": {
        "label": "Long-term",
        "horizon": 96,
        "datasets": ("ETTh1", "ETTh2", "ETTm1", "ETTm2", "weather", "electricity", "traffic"),
        "metrics": ("mse", "mae"),
    },
    "pems": {
        "label": "PEMS",
        "horizon": 24,
        "datasets": ("PEMS03", "PEMS04", "PEMS07", "PEMS08"),
        "metrics": ("mae", "rmse", "mape"),
    },
    "epf": {
        "label": "EPF",
        "horizon": 24,
        "datasets": ("NP", "PJM", "BE", "FR", "DE"),
        "metrics": ("mse", "mae"),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "latex" / "tables",
    )
    parser.add_argument("--expected-sha256", default=EXPECTED_SHA256)
    return parser.parse_args()


def finite_mapping(values: dict[str, Any], expected: Iterable[str], context: str) -> None:
    if set(values) != set(expected):
        raise ValueError(f"{context}: expected metrics {set(expected)}, found {set(values)}")
    for metric, value in values.items():
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{context}: non-finite {metric}={value!r}")


def load_rows(path: Path, expected_sha256: str) -> list[dict[str, Any]]:
    payload = path.read_bytes()
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"unexpected summary digest: expected {expected_sha256}, found {actual_sha256}"
        )

    summary = json.loads(payload)
    if summary.get("selector_policy") != "validation_argmin_v3":
        raise ValueError("unexpected additional-system selector")
    if summary.get("counts") != {
        "completed": 94,
        "evaluable": 94,
        "failed": 0,
        "improved": 81,
        "incomplete": 0,
        "not_improved": 13,
        "paper_oot": 2,
        "pending": 0,
        "reported": 96,
        "running": 0,
    }:
        raise ValueError("unexpected experiment counts")

    rows = summary.get("cell_states")
    if not isinstance(rows, list) or len(rows) != 96:
        raise ValueError("expected exactly 96 reported rows")

    seen: set[tuple[str, str, str]] = set()
    evaluable = strict = oot = 0
    for row in rows:
        task = row.get("task_family")
        dataset = row.get("dataset")
        system = row.get("baseline")
        if task not in TASKS or dataset not in TASKS[task]["datasets"]:
            raise ValueError(f"unexpected task/dataset: {task}/{dataset}")
        if system not in SYSTEMS:
            raise ValueError(f"unexpected system: {system}")
        if row.get("pred_len") != TASKS[task]["horizon"]:
            raise ValueError(f"unexpected horizon for {task}/{dataset}/{system}")
        key = (task, dataset, system)
        if key in seen:
            raise ValueError(f"duplicate row: {key}")
        seen.add(key)

        if row.get("paper_evaluable"):
            if row.get("state") != "completed":
                raise ValueError(f"evaluable row is not completed: {key}")
            metrics = TASKS[task]["metrics"]
            finite_mapping(row.get("test_baseline", {}), metrics, f"{key} baseline")
            finite_mapping(row.get("test_corrected", {}), metrics, f"{key} corrected")
            finite_mapping(row.get("metric_gain_percent", {}), metrics, f"{key} gains")
            recomputed = all(
                row["test_corrected"][metric] < row["test_baseline"][metric]
                for metric in metrics
            )
            if row.get("all_test_metrics_improve") is not recomputed:
                raise ValueError(f"strict-win flag mismatch: {key}")
            evaluable += 1
            strict += int(recomputed)
        else:
            if row.get("state") != "paper_oot":
                raise ValueError(f"unexpected non-evaluable state: {key}")
            if key not in {
                ("long_term", "electricity", "autogluon_high_quality"),
                ("long_term", "traffic", "autogluon_high_quality"),
            }:
                raise ValueError(f"unexpected OOT row: {key}")
            oot += 1

    if len(seen) != 96 or (evaluable, strict, oot) != (94, 81, 2):
        raise ValueError("row-level count mismatch")
    return rows


def grouped_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["baseline"]].append(row)
    return grouped


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def metric_mean(rows: list[dict[str, Any]], metric: str) -> tuple[float, int] | None:
    values = [
        row["metric_gain_percent"][metric]
        for row in rows
        if row.get("paper_evaluable") and metric in row["metric_gain_percent"]
    ]
    return (mean(values), len(values)) if values else None


def generate_summary_table(rows: list[dict[str, Any]]) -> str:
    grouped = grouped_rows(rows)
    lines = [
        r"\begin{table*}[h!]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\caption{\textbf{Correction of additional ensemble, AutoML, and "
        r"foundation-model systems.} Each system is evaluated once per dataset "
        r"at horizon 96 (long-term) or 24 (PEMS and EPF). ``Strict'' requires "
        r"every applicable metric to improve. Entries $\Delta$ are mean paired "
        r"error reductions in percent (higher is better). MSE uses 12 cells per "
        r"system except AutoGluon (10); MAE uses 16 except AutoGluon (14); "
        r"RMSE and MAPE use four.}",
        r"\label{tab:strong-systems}",
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r"System & Eval. & Strict & Rate & $\Delta$MSE & $\Delta$MAE & "
        r"$\Delta$RMSE & $\Delta$MAPE\\",
        r"\midrule",
    ]
    for system in SYSTEMS:
        system_rows = grouped[system]
        evaluable = [row for row in system_rows if row.get("paper_evaluable")]
        strict = sum(bool(row["all_test_metrics_improve"]) for row in evaluable)
        metric_values = []
        for metric in ("mse", "mae", "rmse", "mape"):
            result = metric_mean(system_rows, metric)
            metric_values.append("--" if result is None else f"{result[0]:.2f}")
        lines.append(
            f"{SYSTEM_LABELS[system]} & {len(evaluable)} & {strict} & "
            f"{100 * strict / len(evaluable):.1f}\\% & "
            + " & ".join(metric_values)
            + r"\\"
        )
    lines.extend(
        [
            r"\midrule",
            r"All systems & 94 & 81 & 86.2\% & 5.82 & 6.26 & 15.05 & 15.83\\",
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
            "",
        ]
    )
    return "\n".join(lines)


def format_value(value: float, decimals: int) -> str:
    return f"{value:.{decimals}f}"


def absolute_cell(row: dict[str, Any], metric: str, decimals: int) -> str:
    if not row.get("paper_evaluable"):
        return "OOT"
    baseline = row["test_baseline"][metric]
    corrected = row["test_corrected"][metric]
    corrected_text = format_value(corrected, decimals)
    if corrected < baseline:
        corrected_text = rf"\textbf{{{corrected_text}}}"
    return rf"{format_value(baseline, decimals)}$\rightarrow${corrected_text}"


def generate_absolute_table(
    task: str,
    rows_by_key: dict[tuple[str, str, str], dict[str, Any]],
) -> str:
    spec = TASKS[task]
    label = task.replace("_", "-")
    lines = []
    groups = (
        ("static ensembles", "ensembles", SYSTEMS[:3]),
        ("AutoML and foundation models", "external", SYSTEMS[3:]),
    )
    for group_name, suffix, systems in groups:
        lines.extend(
            [
                r"\begin{table*}[t]",
                r"\centering",
                r"\small",
                r"\setlength{\tabcolsep}{5pt}",
                rf"\caption{{Absolute {spec['label']} errors for {group_name} "
                rf"at horizon {spec['horizon']}. Each entry is "
                r"base$\rightarrow$\method{}; bold marks a lower corrected "
                r"error for that metric. OOT denotes an AutoGluon run that "
                r"exceeded the fixed evaluation-time budget.}",
                rf"\label{{tab:strong-{label}-absolute-{suffix}}}",
                r"\begin{tabular}{ll" + "r" * len(systems) + "}",
                r"\toprule",
                r"Dataset & Metric & "
                + " & ".join(SYSTEM_HEADERS[s] for s in systems)
                + r"\\",
                r"\midrule",
            ]
        )
        datasets = spec["datasets"]
        metrics = spec["metrics"]
        for dataset_index, dataset in enumerate(datasets):
            for metric_index, metric in enumerate(metrics):
                dataset_text = (
                    rf"\multirow{{{len(metrics)}}}{{*}}{{{dataset}}}"
                    if metric_index == 0
                    else ""
                )
                cells = [
                    absolute_cell(
                        rows_by_key[(task, dataset, system)],
                        metric,
                        decimals=4,
                    )
                    for system in systems
                ]
                lines.append(
                    f"{dataset_text} & {metric.upper()} & "
                    + " & ".join(cells)
                    + r"\\"
                )
            if dataset_index != len(datasets) - 1:
                lines.append(r"\addlinespace[1pt]")
        lines.extend(
            [
                r"\bottomrule",
                r"\end{tabular}",
                r"\end{table*}",
                "",
            ]
        )
    return "\n".join(lines)


def generate_worst_metric_table(
    rows_by_key: dict[tuple[str, str, str], dict[str, Any]]
) -> str:
    lines = [
        r"\begin{table*}[h!]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\caption{\textbf{Dataset-level robustness for additional systems.} "
        r"Each entry is the smallest paired error reduction across the metrics "
        r"reported for that cell; it is positive exactly when the cell is a "
        r"strict win. Higher is better.}",
        r"\label{tab:strong-system-worst-gain}",
        r"\begin{tabular}{ll" + "r" * len(SYSTEMS) + "}",
        r"\toprule",
        r"Family & Dataset & " + " & ".join(SYSTEM_HEADERS[s] for s in SYSTEMS) + r"\\",
        r"\midrule",
    ]
    task_items = list(TASKS.items())
    for task_index, (task, spec) in enumerate(task_items):
        for dataset_index, dataset in enumerate(spec["datasets"]):
            family_text = (
                rf"\multirow{{{len(spec['datasets'])}}}{{*}}{{{spec['label']}}}"
                if dataset_index == 0
                else ""
            )
            cells = []
            for system in SYSTEMS:
                row = rows_by_key[(task, dataset, system)]
                if not row.get("paper_evaluable"):
                    cells.append("OOT")
                else:
                    cells.append(f"{min(row['metric_gain_percent'].values()):.2f}")
            lines.append(
                f"{family_text} & {dataset} & " + " & ".join(cells) + r"\\"
            )
        if task_index != len(task_items) - 1:
            lines.append(r"\midrule")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    rows = load_rows(args.summary, args.expected_sha256)
    rows_by_key = {
        (row["task_family"], row["dataset"], row["baseline"]): row for row in rows
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "strong_system_summary.tex": generate_summary_table(rows),
        "strong_system_worst_gain.tex": generate_worst_metric_table(rows_by_key),
        "strong_long-term_absolute.tex": generate_absolute_table("long_term", rows_by_key),
        "strong_pems_absolute.tex": generate_absolute_table("pems", rows_by_key),
        "strong_epf_absolute.tex": generate_absolute_table("epf", rows_by_key),
    }
    for name, text in outputs.items():
        (args.output_dir / name).write_text(text, encoding="ascii")


if __name__ == "__main__":
    main()
