#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean, median

EXPECTED_CELLS = 585
EXPECTED_SYSTEMS = (
    "base",
    "analog_future",
    "raft_adapted",
    "saraf_adapted",
    "residual_retrieval",
    "timeraf",
)
MAIN_SYSTEMS = (
    "base",
    "raft_adapted",
    "saraf_adapted",
    "timeraf",
)
MODELS = (
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
)
SUMMARY_LABELS = {
    "base": "Base",
    "analog_future": r"\analogknn{}",
    "raft_adapted": r"\raft{}",
    "saraf_adapted": r"\saraf{}",
    "residual_retrieval": r"\residualknn{}",
    "timeraf": r"\method{}",
}
MODEL_COLUMN_LABELS = {
    "TimeXer": r"\textsc{TXer}",
    "TimeMixer": r"\textsc{TMix}",
    "PAttn": r"\textsc{PAttn}",
    "iTransformer": r"\textsc{iTrans.}",
    "TimesNet": r"\textsc{TNet}",
    "PatchTST": r"\textsc{PTST}",
    "DLinear": r"\textsc{DLin}",
    "FreTS": r"\textsc{FreTS}",
    "FEDformer": r"\textsc{FEDf.}",
    "Nonstationary_Transformer": r"\textsc{NonStat}",
    "LightTS": r"\textsc{LTS}",
    "Informer": r"\textsc{Inf.}",
    "Autoformer": r"\textsc{AutoF}",
}
SCOPE = {
    "long_term": {
        "datasets": (
            "ETTh1",
            "ETTh2",
            "ETTm1",
            "ETTm2",
            "weather",
            "electricity",
            "traffic",
        ),
        "horizons": (96, 192, 336, 720),
        "metrics": ("mse", "mae"),
        "label": "Long-term",
    },
    "pems": {
        "datasets": ("PEMS03", "PEMS04", "PEMS07", "PEMS08"),
        "horizons": (6, 12, 24),
        "metrics": ("mae", "rmse", "mape"),
        "label": "PEMS",
    },
    "epf": {
        "datasets": ("NP", "PJM", "BE", "FR", "DE"),
        "horizons": (24,),
        "metrics": ("mse", "mae"),
        "label": "EPF",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument(
        "--paper-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--expected-source-revision")
    parser.add_argument("--expected-launcher-revision")
    parser.add_argument("--expected-protocol-sha256")
    parser.add_argument("--expected-catalog-sha256")
    parser.add_argument(
        "--unified-summary",
        type=Path,
        help=(
            "Composed unified-portfolio summary whose timeraf row replaces the "
            "accepted one. Every other system stays with the accepted run."
        ),
    )
    parser.add_argument("--unified-sha256")
    parser.add_argument("--run-metadata", required=True, type=Path)
    parser.add_argument("--startup-topology", required=True, type=Path)
    parser.add_argument("--completion-receipt", required=True, type=Path)
    return parser.parse_args()


def expected_ids() -> set[str]:
    return {
        f"{family}/{dataset}/{model}/{horizon}"
        for family, scope in SCOPE.items()
        for dataset in scope["datasets"]
        for model in MODELS
        for horizon in scope["horizons"]
    }


def validate(document: dict, args: argparse.Namespace) -> list[dict]:
    if args.expected_source_revision and document.get("source_revision") != (
        args.expected_source_revision
    ):
        raise ValueError("Retrieval baseline source revision drifted")
    expected_launcher = getattr(args, "expected_launcher_revision", None)
    if expected_launcher and document.get("launcher_revision") != (
        expected_launcher
    ):
        raise ValueError("Retrieval baseline launcher revision drifted")
    if args.expected_protocol_sha256 and document.get("protocol_sha256") != (
        args.expected_protocol_sha256
    ):
        raise ValueError("Retrieval baseline protocol hash drifted")
    expected_catalog = getattr(args, "expected_catalog_sha256", None)
    if expected_catalog and document.get("catalog_sha256") != expected_catalog:
        raise ValueError("Retrieval baseline catalog hash drifted")
    if document.get("expected_cells") not in (None, EXPECTED_CELLS):
        raise ValueError("Retrieval baseline expected-cell count drifted")
    if document.get("expected_systems") not in (
        None,
        list(EXPECTED_SYSTEMS),
    ):
        raise ValueError("Retrieval baseline expected-system identity drifted")
    if document.get("counts") != {
        "completed": EXPECTED_CELLS,
        "failed": 0,
        "incomplete": 0,
        "pending": 0,
        "running": 0,
    }:
        raise ValueError("Retrieval baseline matrix is not complete")
    if document.get("all_completed") is not True:
        raise ValueError("Retrieval baseline all_completed gate is false")

    rows = document.get("cell_states")
    if not isinstance(rows, list) or len(rows) != EXPECTED_CELLS:
        raise ValueError(
            f"Retrieval baseline matrix must have {EXPECTED_CELLS} rows"
        )
    seen = set()
    for row in rows:
        cell_id = row["cell_id"]
        if cell_id in seen:
            raise ValueError(f"Duplicate retrieval baseline cell: {cell_id}")
        seen.add(cell_id)
        try:
            family_id, dataset_id, model_id, horizon_id = cell_id.split("/")
            horizon_id = int(horizon_id)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Malformed retrieval baseline cell ID: {cell_id}"
            ) from error
        identity = (
            row.get("task_family"),
            row.get("dataset"),
            row.get("model"),
            row.get("pred_len"),
        )
        if identity != (family_id, dataset_id, model_id, horizon_id):
            raise ValueError(
                f"Cell identity fields do not match cell ID: {cell_id}"
            )
        if row["state"] != "completed":
            raise ValueError(f"Non-completed retrieval cell: {cell_id}")
        systems = row.get("systems", {})
        if set(systems) != set(EXPECTED_SYSTEMS):
            raise ValueError(f"System coverage mismatch in {cell_id}")
        family = row["task_family"]
        metrics = SCOPE[family]["metrics"]
        base_metrics = systems["base"]["test_metrics"]
        for system_id, system in systems.items():
            values = system["test_metrics"]
            gains = system.get("metric_gain_percent", {})
            if set(values) != set(metrics):
                raise ValueError(
                    f"Metric coverage mismatch in {cell_id}/{system_id}"
                )
            if set(gains) != set(metrics):
                raise ValueError(
                    f"Metric-gain coverage mismatch in {cell_id}/{system_id}"
                )
            if not all(math.isfinite(float(value)) for value in values.values()):
                raise ValueError(
                    f"Non-finite metric in {cell_id}/{system_id}"
                )
            for name in metrics:
                base_value = float(base_metrics[name])
                value = float(values[name])
                gain = float(gains[name])
                if (
                    base_value <= 0.0
                    or not math.isfinite(gain)
                    or not math.isclose(
                        gain,
                        100.0 * (base_value - value) / base_value,
                        rel_tol=1e-10,
                        abs_tol=1e-10,
                    )
                ):
                    raise ValueError(
                        f"Metric gain mismatch in "
                        f"{cell_id}/{system_id}/{name}"
                    )
            expected_strict = (
                system_id != "base"
                and all(values[name] < base_metrics[name] for name in metrics)
            )
            if bool(system["strict_win"]) != expected_strict:
                raise ValueError(
                    f"Strict-win mismatch in {cell_id}/{system_id}"
                )
        if not row["base_metric_recomputation"]["passed"]:
            raise ValueError(f"Base recomputation gate failed in {cell_id}")
        if not row["no_lookahead"]["retrieval_systems"]["passed"]:
            raise ValueError(f"Retrieval lookahead gate failed in {cell_id}")
        if not row["no_lookahead"]["timeraf"]["passed"]:
            raise ValueError(f"TimeRAF lookahead gate failed in {cell_id}")

    if seen != expected_ids():
        raise ValueError(
            f"Retrieval cell identity mismatch: "
            f"{len(expected_ids() - seen)} missing, "
            f"{len(seen - expected_ids())} extra"
        )
    return rows


def validate_provenance(
    document: dict,
    run_metadata: dict,
    startup_topology: dict,
) -> None:
    identity_keys = (
        "source_revision",
        "launcher_revision",
        "protocol_sha256",
        "catalog_sha256",
    )
    for key in identity_keys:
        if run_metadata.get(key) != document.get(key):
            raise ValueError(f"Run metadata {key} drifted")
    if run_metadata.get("selected_cells") != EXPECTED_CELLS:
        raise ValueError(
            f"Run metadata does not cover {EXPECTED_CELLS} cells"
        )

    topology = run_metadata.get("topology", {})
    expected_topology = {
        "instance_type": "ml.p5.48xlarge",
        "instance_count": 1,
        "reserved_gpus_per_host": 8,
        "visible_cuda_devices": 8,
        "processes_per_host": 8,
        "worker_gpu_ids": list(range(8)),
        "total_gpus": 8,
        "world_size": 8,
        "inactive_reserved_gpus": 0,
        "child_device": "cuda:0",
    }
    for key, expected in expected_topology.items():
        if topology.get(key) != expected:
            raise ValueError(f"Run topology {key} drifted")

    bindings = startup_topology.get("worker_bindings")
    if (
        startup_topology.get("all_bindings_observed") is not True
        or startup_topology.get("expected_worker_count") != 8
        or startup_topology.get("distinct_positive_pids") is not True
        or startup_topology.get("gpu_ids") != list(range(8))
        or not isinstance(bindings, list)
        or len(bindings) != 8
    ):
        raise ValueError("Startup topology does not prove eight GPU bindings")
    worker_gpus = [binding.get("worker_gpu") for binding in bindings]
    pids = [binding.get("pid") for binding in bindings]
    gpu_uuids = [binding.get("gpu_uuid") for binding in bindings]
    if (
        worker_gpus != list(range(8))
        or len(set(pids)) != 8
        or any(not isinstance(pid, int) or pid <= 0 for pid in pids)
        or len(set(gpu_uuids)) != 8
        or any(not value for value in gpu_uuids)
    ):
        raise ValueError("Startup worker bindings are invalid")


def validate_completion_receipt(
    receipt: dict,
    document: dict,
    paths: dict[str, Path],
) -> None:
    if (
        receipt.get("schema_version") not in (1, 2)
        or receipt.get("status") != "completed"
        or receipt.get("expected_cells") != EXPECTED_CELLS
        or receipt.get("source_revision") != document.get("source_revision")
    ):
        raise ValueError("Full-horizon completion receipt identity is invalid")
    if (
        receipt.get("schema_version") == 2
        and receipt.get("launcher_revision")
        != document.get("launcher_revision")
    ):
        raise ValueError("Full-horizon completion receipt launcher drifted")
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("Full-horizon completion receipt has no artifacts")

    for suffix, path in paths.items():
        matches = [
            record
            for key, record in artifacts.items()
            if key == suffix or key.endswith("/" + suffix)
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Completion receipt does not uniquely bind {suffix}"
            )
        record = matches[0]
        if (
            record.get("size_bytes") != path.stat().st_size
            or record.get("sha256") != _sha256(path)
        ):
            raise ValueError(
                f"Completion receipt hash mismatch for {suffix}"
            )

    catalog_matches = [
        record
        for key, record in artifacts.items()
        if key == "prediction_bundle_catalog.json"
        or key.endswith("/prediction_bundle_catalog.json")
    ]
    if (
        len(catalog_matches) != 1
        or catalog_matches[0].get("sha256") != document.get("catalog_sha256")
    ):
        raise ValueError("Completion receipt catalog identity drifted")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def aggregate_model_metric(
    rows: list[dict],
    family: str,
    model: str,
    system_id: str,
    metric: str,
) -> tuple[float, float]:
    members = [
        row
        for row in rows
        if row["task_family"] == family and row["model"] == model
    ]
    expected = (
        len(SCOPE[family]["datasets"])
        * len(SCOPE[family]["horizons"])
    )
    if len(members) != expected:
        raise ValueError(
            f"Expected {expected} {family}/{model} cells, found {len(members)}"
        )
    absolute = fmean(
        float(row["systems"][system_id]["test_metrics"][metric])
        for row in members
    )
    base_absolute = fmean(
        float(row["systems"]["base"]["test_metrics"][metric])
        for row in members
    )
    return absolute, absolute - base_absolute


def _fmt_aggregate_absolute(family: str, metric: str, value: float) -> str:
    if family == "pems" and metric in ("mae", "rmse"):
        return f"{value:.2f}"
    return f"{value:.3f}"


def _fmt_delta(family: str, metric: str, value: float) -> str:
    decimals = 2 if family == "pems" and metric in ("mae", "rmse") else 3
    rounded = round(value, decimals)
    if rounded == 0.0:
        return f"{0.0:.{decimals}f}"
    return f"{rounded:+.{decimals}f}"


def _render_value_tables(
    rows: list[dict],
    families: tuple[str, ...],
    table_label: str,
    caption: str,
) -> str:
    lines = [
        "% Generated by paper/tools/generate_retrieval_baseline_tables.py.",
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{" + caption + "}",
        f"\\label{{{table_label}}}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{1.2pt}",
        r"\renewcommand{\arraystretch}{1.03}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{@{}l*{13}{c}@{}}",
        r"\toprule",
        "System & "
        + " & ".join(MODEL_COLUMN_LABELS[model] for model in MODELS)
        + r" \\",
    ]
    for family in families:
        scope = SCOPE[family]
        for metric in scope["metrics"]:
            aggregate = {
                (model, system_id): aggregate_model_metric(
                    rows,
                    family,
                    model,
                    system_id,
                    metric,
                )
                for model in MODELS
                for system_id in MAIN_SYSTEMS
            }
            best = {
                model: min(
                    aggregate[(model, system_id)][0]
                    for system_id in MAIN_SYSTEMS
                )
                for model in MODELS
            }
            lines.extend(
                [
                    r"\midrule",
                    (
                        rf"\multicolumn{{14}}{{@{{}}l}}{{\textit{{"
                        f"{scope['label']}, {metric.upper()}"
                        " (all datasets and horizons)"
                        r"}} \\"
                    ),
                ]
            )
            for system_id in MAIN_SYSTEMS:
                value_cells = []
                delta_cells = []
                for model in MODELS:
                    absolute, delta = aggregate[(model, system_id)]
                    is_best = absolute == best[model]
                    value_text = _fmt_aggregate_absolute(
                        family, metric, absolute
                    )
                    delta_text = _fmt_delta(family, metric, delta)
                    if is_best:
                        value_text = rf"\mathbf{{{value_text}}}"
                        delta_text = rf"\mathbf{{{delta_text}}}"
                    value_cells.append(rf"${value_text}$")
                    delta_cells.append(rf"${delta_text}$")
                lines.append(
                    f"{SUMMARY_LABELS[system_id]} & "
                    + " & ".join(value_cells)
                    + r" \\"
                )
                if system_id != "base":
                    lines.append(
                        rf"\quad $\Delta$ & "
                        + " & ".join(delta_cells)
                        + r" \\"
                    )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            r"\end{table*}",
            "",
        ]
    )
    return "\n".join(lines)


def render_value_table(rows: list[dict], family: str) -> str:
    scope = SCOPE[family]
    labels = {
        "long_term": ("Long-term", "tab:retrieval-long-term-values"),
        "pems": ("PEMS", "tab:retrieval-pems-values"),
        "epf": ("EPF", "tab:retrieval-epf-values"),
    }
    family_label, table_label = labels[family]
    dataset_count = len(scope["datasets"])
    horizon_count = len(scope["horizons"])
    caption = (
        rf"\textbf{{{family_label} fixed-forecast comparison.}} "
        f"Mean over all {dataset_count} datasets and {horizon_count} "
        "horizons for each base model. "
        r"$\Delta=\mathrm{method}-\mathrm{Base}$; lower is better. "
        r"Bold marks the lowest error and corresponding delta."
    )
    return _render_value_tables(rows, (family,), table_label, caption)


def render_short_value_table(rows: list[dict]) -> str:
    return _render_value_tables(
        rows,
        ("pems", "epf"),
        "tab:retrieval-short-values",
        (
            r"\textbf{PEMS and EPF fixed-forecast comparison.} "
            r"Dataset--horizon mean by base model within each task. "
            r"$\Delta=\mathrm{method}-\mathrm{Base}$; lower is better. "
            r"Bold marks the lowest error and corresponding delta."
        ),
    )


def build_data(
    document: dict,
    rows: list[dict],
    provenance: dict | None = None,
) -> dict:
    compact_rows = []
    for row in rows:
        compact_rows.append(
            {
                key: row[key]
                for key in (
                    "cell_id",
                    "task_family",
                    "dataset",
                    "model",
                    "pred_len",
                    "systems",
                    "a10g_metric_reference",
                )
            }
        )
    system_statistics = {}
    for system_id in EXPECTED_SYSTEMS:
        members = [
            row["systems"][system_id]
            for row in rows
        ]
        system_statistics[system_id] = {
            "strict_wins": sum(
                bool(member["strict_win"]) for member in members
            ),
            "cells": len(rows),
            "median_paired_gain_percent": {
                metric: median(
                    float(member["metric_gain_percent"][metric])
                    for member in members
                    if metric in member["metric_gain_percent"]
                )
                for metric in ("mse", "mae", "rmse", "mape")
                if any(
                    metric in member["metric_gain_percent"]
                    for member in members
                )
            },
        }
    return {
        "schema_version": 1,
        "source_revision": document["source_revision"],
        "launcher_revision": document["launcher_revision"],
        "protocol_sha256": document["protocol_sha256"],
        "catalog_sha256": document["catalog_sha256"],
        "provenance": provenance or {},
        "cells": compact_rows,
        "system_statistics": system_statistics,
    }


def apply_unified_method(document: dict, args: argparse.Namespace) -> dict:
    """Replace the method row with the composed unified-portfolio result.

    The accepted run remains the source for the base forecast and for every
    comparison system, so the appendix tables stay paired cell by cell.
    """

    payload = args.unified_summary.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if args.unified_sha256 and digest != args.unified_sha256:
        raise ValueError(f"unified summary SHA-256 mismatch: {digest}")
    unified = json.loads(payload)
    provenance = unified["provenance"]
    if provenance["accepted_summary_sha256"] != _sha256(args.input):
        raise ValueError("unified summary was composed from a different run")
    if not provenance["base_forecasts_identical"]:
        raise ValueError("unified summary does not share the base forecasts")
    replacement = {
        row["cell_id"]: row["systems"]["timeraf"] for row in unified["cell_states"]
    }
    for row in document["cell_states"]:
        row["systems"]["timeraf"] = replacement[row["cell_id"]]
    return {
        "unified_summary_sha256": digest,
        "unified_selected_policy": provenance["selected_policy"],
        "unified_method_revision": provenance["v2_method_revision"],
        "unified_protocol_sha256": provenance["v2_protocol_sha256"],
        "accepted_method_reproduction": provenance["accepted_method_reproduction"],
    }


def main() -> None:
    args = parse_args()
    document = json.loads(args.input.read_text(encoding="utf-8"))
    unified_provenance = (
        apply_unified_method(document, args) if args.unified_summary else {}
    )
    rows = validate(document, args)
    run_metadata = json.loads(args.run_metadata.read_text(encoding="utf-8"))
    startup_topology = json.loads(
        args.startup_topology.read_text(encoding="utf-8")
    )
    validate_provenance(document, run_metadata, startup_topology)
    completion_receipt = json.loads(
        args.completion_receipt.read_text(encoding="utf-8")
    )
    validate_completion_receipt(
        completion_receipt,
        document,
        {
            "retrieval_matrix/matrix_summary.json": args.input,
            "retrieval_matrix/run_metadata.json": args.run_metadata,
            "retrieval_matrix/startup_topology.json": args.startup_topology,
        },
    )
    protocol_path = (
        args.paper_root.parent / "docs" / "retrieval_baseline_protocol.json"
    )
    if _sha256(protocol_path) != document.get("protocol_sha256"):
        raise ValueError("Local frozen retrieval protocol hash drifted")
    latex_root = args.paper_root / "latex"
    table_root = latex_root / "tables"
    data_root = args.paper_root / "data"
    table_root.mkdir(parents=True, exist_ok=True)
    data_root.mkdir(parents=True, exist_ok=True)
    (table_root / "retrieval_long_term_model_values.tex").write_text(
        render_value_table(rows, "long_term"),
        encoding="utf-8",
    )
    (table_root / "retrieval_short_model_values.tex").write_text(
        render_short_value_table(rows),
        encoding="utf-8",
    )
    (data_root / "retrieval_baseline_results.json").write_text(
        json.dumps(
            build_data(
                document,
                rows,
                provenance={
                    **unified_provenance,
                    "run_metadata_sha256": _sha256(args.run_metadata),
                    "startup_topology_sha256": _sha256(
                        args.startup_topology
                    ),
                },
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
