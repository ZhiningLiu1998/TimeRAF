#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import fmean, median

try:
    from scripts.generate_native_retrieval_baseline_manifest import (
        build_manifest,
    )
except ModuleNotFoundError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.generate_native_retrieval_baseline_manifest import (
        build_manifest,
    )


EXPECTED_CELLS = 585
NATIVE_PROTOCOL_PATH = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "native_retrieval_baseline_protocol.json"
)
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
ABLATION_SYSTEMS = (
    "base",
    "analog_future",
    "residual_retrieval",
    "timeraf",
)
NATIVE_METHODS = {
    "RAF": {
        "pairs": 44,
        "metrics": ("wql", "mase"),
        "citation": "tire2024raf",
    },
    "TS-RAG": {
        "pairs": 7,
        "metrics": ("mse", "mae"),
        "citation": "ning2025tsrag",
    },
    "RATD": {
        "pairs": 1,
        "metrics": ("rmse", "mae"),
        "citation": "liu2024ratd",
    },
}
NATIVE_SOURCE_PAIRS = {
    "raf": 44,
    "ts_rag": 7,
    "ratd": 1,
}
SYSTEM_LABELS = {
    "base": "Base",
    "analog_future": "Analog-kNN",
    "raft_adapted": "RAFT",
    "saraf_adapted": "SARAF",
    "residual_retrieval": "Residual-kNN",
    "timeraf": r"\method{}",
}
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
MODEL_LABELS = {
    "Nonstationary_Transformer": "Non-stat. Transformer",
}
SUMMARY_LABELS = {
    "base": "Base",
    "analog_future": "Analog-kNN",
    "raft_adapted": "RAFT",
    "saraf_adapted": "SARAF",
    "residual_retrieval": "Residual-kNN",
    "timeraf": r"\method{}",
}
MODEL_COLUMN_LABELS = {
    "TimeXer": "TXer",
    "TimeMixer": "TMix",
    "PAttn": "PAttn",
    "iTransformer": "iTrans.",
    "TimesNet": "TNet",
    "PatchTST": "PTST",
    "DLinear": "DLin",
    "FreTS": "FreTS",
    "FEDformer": "FEDf.",
    "Nonstationary_Transformer": "NonStat",
    "LightTS": "LTS",
    "Informer": "Inf.",
    "Autoformer": "AutoF",
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
    parser.add_argument("--native-input", required=True, type=Path)
    parser.add_argument(
        "--paper-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--expected-source-revision")
    parser.add_argument("--expected-launcher-revision")
    parser.add_argument("--expected-protocol-sha256")
    parser.add_argument("--expected-catalog-sha256")
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


def validate_native(document: dict) -> list[dict]:
    def require_path(value: object, label: str) -> None:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Native {label} path is missing")

    def require_sha256(value: object, label: str) -> None:
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"Native {label} hash is invalid")

    if document.get("schema_version") != 2:
        raise ValueError("Native retrieval baseline schema must be 2")
    require_sha256(document.get("protocol_sha256"), "protocol")
    protocol = json.loads(NATIVE_PROTOCOL_PATH.read_text(encoding="utf-8"))
    canonical_manifest = build_manifest(protocol)
    canonical_by_id = {row["id"]: row for row in canonical_manifest}

    execution_record = document.get("execution_record")
    if not isinstance(execution_record, dict):
        raise ValueError("Native execution record is missing")
    require_path(execution_record.get("path"), "execution record")
    require_sha256(execution_record.get("sha256"), "execution record")

    source_summaries = document.get("source_summaries")
    if (
        not isinstance(source_summaries, dict)
        or set(source_summaries) != set(NATIVE_SOURCE_PAIRS)
    ):
        raise ValueError("Native source summaries are missing or invalid")
    for source_key, expected_pairs in NATIVE_SOURCE_PAIRS.items():
        source = source_summaries[source_key]
        if not isinstance(source, dict):
            raise ValueError(f"Native {source_key} source is invalid")
        require_path(source.get("path"), f"{source_key} source")
        require_sha256(source.get("sha256"), f"{source_key} source")
        source_revision = source.get("source_revision")
        if not isinstance(source_revision, str) or not source_revision.strip():
            raise ValueError(
                f"Native {source_key} source revision is missing"
            )
        require_path(source.get("topology_path"), f"{source_key} topology")
        require_sha256(
            source.get("topology_sha256"),
            f"{source_key} topology",
        )
        if source.get("pairs") != expected_pairs:
            raise ValueError(f"Native {source_key} source pair count drifted")

    if (
        document.get("expected_pairs") != 52
        or document.get("completed_pairs") != 52
        or document.get("all_completed") is not True
    ):
        raise ValueError("Native retrieval baseline matrix is incomplete")
    pairs = document.get("pairs")
    if not isinstance(pairs, list) or len(pairs) != 52:
        raise ValueError("Native retrieval baseline matrix must have 52 pairs")
    counts = defaultdict(int)
    seen = set()
    for pair in pairs:
        cell_id = pair.get("cell_id")
        method = pair.get("method")
        if not isinstance(cell_id, str) or not cell_id or cell_id in seen:
            raise ValueError("Native baseline cell IDs are invalid")
        seen.add(cell_id)
        canonical = canonical_by_id.get(cell_id)
        if canonical is None:
            raise ValueError(
                f"{cell_id}: native baseline cell ID is not canonical"
            )
        system = protocol["systems"][canonical["method"]]
        canonical_fields = {
            "method": system["retrieval_system"],
            "base_method": system["base_system"],
            "dataset": canonical["dataset"],
            "backbone": canonical["backbone"],
            "context_length": canonical["context_length"],
            "prediction_length": canonical["prediction_length"],
            "metrics": canonical["metrics"],
        }
        for field, expected_value in canonical_fields.items():
            if pair.get(field) != expected_value:
                raise ValueError(
                    f"{cell_id}: native canonical {field} mismatch"
                )
        if method not in NATIVE_METHODS:
            raise ValueError(f"Unknown native baseline method: {method}")
        counts[method] += 1
        expected_metrics = set(NATIVE_METHODS[method]["metrics"])
        if set(pair.get("metrics", ())) != expected_metrics:
            raise ValueError(f"{cell_id}: native metric coverage drifted")
        base = pair.get("base", {})
        retrieval = pair.get("retrieval", {})
        delta = pair.get("delta", {})
        if (
            set(base) != expected_metrics
            or set(retrieval) != expected_metrics
            or set(delta) != expected_metrics
        ):
            raise ValueError(f"{cell_id}: native paired metrics are missing")
        for metric in expected_metrics:
            values = (
                float(base[metric]),
                float(retrieval[metric]),
                float(delta[metric]),
            )
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"{cell_id}: non-finite native metric")
            if not math.isclose(
                values[2],
                values[1] - values[0],
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise ValueError(f"{cell_id}: native delta mismatch")
    expected_counts = {
        method: specification["pairs"]
        for method, specification in NATIVE_METHODS.items()
    }
    if dict(counts) != expected_counts:
        raise ValueError("Native method pair counts drifted")
    if seen != set(canonical_by_id):
        raise ValueError("Native baseline canonical cell coverage drifted")
    return pairs


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


def _escape(value: str) -> str:
    return value.replace("_", r"\_")


def _model_label(model: str) -> str:
    return MODEL_LABELS.get(model, model)


def _fmt_metric(family: str, metric: str, value: float) -> str:
    decimals = 3 if family == "pems" and metric != "mape" else 4
    return f"{value:.{decimals}f}"


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


def render_summary(rows: list[dict]) -> str:
    lines = [
        "% Generated by paper/tools/generate_retrieval_baseline_tables.py.",
        r"\begin{center}",
        r"\refstepcounter{table}",
        r"\label{tab:retrieval-baseline-summary}",
        r"\begin{minipage}[t]{\linewidth}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.5pt}",
        (
            r"\textbf{Table~\thetable: Base and retrieval-method results in "
            r"the 585-cell checkpoint replay. LT, PEMS, EPF, and All are "
            r"strict wins over Base. Metric columns are median paired error "
            r"reductions (\%); higher is better.}\par"
        ),
        r"\centering",
        r"\begin{tabular}{lrrrrrrrr}",
        r"\toprule",
        (
            r"System & LT & PEMS & EPF & All & "
            r"$\Delta$MSE & $\Delta$MAE & $\Delta$RMSE & $\Delta$MAPE \\"
        ),
        r"\midrule",
    ]
    for system_id in EXPECTED_SYSTEMS:
        counts = []
        for family, scope in SCOPE.items():
            members = [row for row in rows if row["task_family"] == family]
            wins = sum(
                bool(row["systems"][system_id]["strict_win"])
                for row in members
            )
            counts.append(f"{wins}/{len(members)}")
        overall = sum(
            bool(row["systems"][system_id]["strict_win"]) for row in rows
        )
        metric_medians = []
        for metric in ("mse", "mae", "rmse", "mape"):
            gains = [
                float(row["systems"][system_id]["metric_gain_percent"][metric])
                for row in rows
                if metric
                in row["systems"][system_id]["metric_gain_percent"]
            ]
            metric_medians.append(f"{median(gains):.1f}")
        lines.append(
            f"{SUMMARY_LABELS[system_id]} & "
            + " & ".join(counts)
            + f" & {overall}/{len(rows)} & "
            + " & ".join(metric_medians)
            + r" \\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{minipage}%",
            r"\end{center}",
            "",
        ]
    )
    return "\n".join(lines)


def aggregate_native_metric(
    pairs: list[dict],
    method: str,
    metric: str,
) -> tuple[float, float, float]:
    members = [pair for pair in pairs if pair["method"] == method]
    expected = NATIVE_METHODS[method]["pairs"]
    if len(members) != expected:
        raise ValueError(
            f"Expected {expected} native {method} pairs, found {len(members)}"
        )
    base = fmean(float(pair["base"][metric]) for pair in members)
    retrieval = fmean(
        float(pair["retrieval"][metric]) for pair in members
    )
    return base, retrieval, retrieval - base


def _fmt_native(value: float) -> str:
    return f"{value:.4f}"


def render_native_values(pairs: list[dict]) -> str:
    lines = [
        "% Generated by paper/tools/generate_retrieval_baseline_tables.py.",
        r"\begin{table}[t]",
        r"\centering",
        (
            r"\caption{\textbf{Published end-to-end retrieval systems in "
            r"their native protocols.} Base and retrieval are evaluated on "
            r"the same examples; $\Delta=\mathrm{retrieval}-\mathrm{base}$. "
            r"Negative is better. Values are averaged only within one "
            r"method/metric, and methods are not ranked across protocols. "
            r"Bold marks the lower paired error and its delta.}"
        ),
        r"\label{tab:native-retrieval-values}",
        r"\small",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\begin{tabular}{@{}llrrr@{}}",
        r"\toprule",
        r"Method & Metric & Base & Retrieval & $\Delta$ \\",
        r"\midrule",
    ]
    for method, specification in NATIVE_METHODS.items():
        citation = specification["citation"]
        for metric_index, metric in enumerate(specification["metrics"]):
            base, retrieval, delta = aggregate_native_metric(
                pairs, method, metric
            )
            base_text = _fmt_native(base)
            retrieval_text = _fmt_native(retrieval)
            delta_text = f"{delta:+.4f}" if delta else "0.0000"
            if retrieval < base:
                retrieval_text = rf"\textbf{{{retrieval_text}}}"
                delta_text = rf"\textbf{{{delta_text}}}"
            elif base < retrieval:
                base_text = rf"\textbf{{{base_text}}}"
            else:
                base_text = rf"\textbf{{{base_text}}}"
                retrieval_text = rf"\textbf{{{retrieval_text}}}"
            method_text = (
                rf"{method}~\citep{{{citation}}}"
                if metric_index == 0
                else ""
            )
            lines.append(
                f"{method_text} & {metric.upper()} & {base_text} & "
                f"{retrieval_text} & {delta_text} \\\\"
            )
        if method != tuple(NATIVE_METHODS)[-1]:
            lines.append(r"\addlinespace")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def render_native_full(pairs: list[dict]) -> str:
    lines = [
        "% Generated by paper/tools/generate_retrieval_baseline_tables.py.",
        r"\begingroup",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.5pt}",
        r"\begin{longtable}{@{}lllrrrr@{}}",
        (
            r"\caption{All native end-to-end retrieval pairs. "
            r"$\Delta=\mathrm{retrieval}-\mathrm{base}$; negative is better. "
            r"Bold marks the lower paired error and its delta. Methods use "
            r"different protocols and are not ranked against one another.}"
            r"\label{tab:native-retrieval-full}\\"
        ),
        r"\toprule",
        r"Method & Dataset & Metric & C/H & Base & Retrieval & $\Delta$ \\",
        r"\midrule",
        r"\endfirsthead",
        r"\multicolumn{7}{c}{\tablename\ \thetable\ continued} \\",
        r"\toprule",
        r"Method & Dataset & Metric & C/H & Base & Retrieval & $\Delta$ \\",
        r"\midrule",
        r"\endhead",
        r"\midrule",
        r"\multicolumn{7}{r}{Continued on next page} \\",
        r"\endfoot",
        r"\bottomrule",
        r"\endlastfoot",
    ]
    ordered_pairs = sorted(
        pairs,
        key=lambda pair: (
            tuple(NATIVE_METHODS).index(pair["method"]),
            pair["cell_id"],
        ),
    )
    previous_method = None
    for pair in ordered_pairs:
        method = pair["method"]
        if previous_method is not None and method != previous_method:
            lines.append(r"\midrule")
        for metric_index, metric in enumerate(pair["metrics"]):
            base = float(pair["base"][metric])
            retrieval = float(pair["retrieval"][metric])
            delta = retrieval - base
            base_text = _fmt_native(base)
            retrieval_text = _fmt_native(retrieval)
            delta_text = f"{delta:+.4f}" if delta else "0.0000"
            if retrieval < base:
                retrieval_text = rf"\textbf{{{retrieval_text}}}"
                delta_text = rf"\textbf{{{delta_text}}}"
            elif base < retrieval:
                base_text = rf"\textbf{{{base_text}}}"
            else:
                base_text = rf"\textbf{{{base_text}}}"
                retrieval_text = rf"\textbf{{{retrieval_text}}}"
            method_text = method if metric_index == 0 else ""
            dataset_text = (
                _escape(str(pair["dataset"])) if metric_index == 0 else ""
            )
            context_horizon_text = (
                f"{pair['context_length']}/{pair['prediction_length']}"
                if metric_index == 0
                else ""
            )
            lines.append(
                f"{method_text} & {dataset_text} & {metric.upper()} & "
                f"{context_horizon_text} & {base_text} & "
                f"{retrieval_text} & {delta_text} \\\\"
            )
        previous_method = method
    lines.extend([r"\end{longtable}", r"\endgroup", ""])
    return "\n".join(lines)


def _format_win_count(value: int, denominator: int, *, best: bool) -> str:
    text = f"{value}/{denominator}"
    return rf"\textbf{{{text}}}" if best else text


def render_model_columns(
    rows: list[dict], native_pairs: list[dict] | None = None
) -> str:
    native_pairs = native_pairs or []
    main_win_systems = ("raft_adapted", "saraf_adapted", "timeraf")
    ablation_win_systems = ("analog_future", "residual_retrieval")
    family_counts = {
        (system_id, family): sum(
            bool(row["systems"][system_id]["strict_win"])
            for row in rows
            if row["task_family"] == family
        )
        for system_id in EXPECTED_SYSTEMS
        for family in SCOPE
    }
    family_denominators = {
        family: (
            len(scope["datasets"])
            * len(scope["horizons"])
            * len(MODELS)
        )
        for family, scope in SCOPE.items()
    }
    overall_counts = {
        system_id: sum(
            bool(row["systems"][system_id]["strict_win"]) for row in rows
        )
        for system_id in EXPECTED_SYSTEMS
    }
    lines = [
        "% Generated by paper/tools/generate_retrieval_baseline_tables.py.",
        r"\begin{table*}[t]",
        r"\centering",
        (
            r"\caption{\textbf{Unified strict-win counts.} In the fixed-"
            r"forecast panels, a win lowers every metric; LT, PEMS, EPF, "
            r"and All contain 364, 156, 65, and 585 cells, respectively. "
            r"Native end-to-end rows use their own paired protocols and are "
            r"not ranked across methods. Bold marks the largest count within "
            r"each comparable fixed-forecast panel.}"
        ),
        r"\label{tab:retrieval-win-summary}",
        r"\setlength{\tabcolsep}{5pt}",
        r"\begin{tabular}{@{}lrrrr@{}}",
        r"\toprule",
        r"System & LT & PEMS & EPF & All \\",
        r"\midrule",
        r"\multicolumn{5}{@{}l}{Published retrieval operators and proposed method} \\",
    ]
    for system_id in main_win_systems:
        counts = []
        for family in SCOPE:
            value = family_counts[(system_id, family)]
            best = value == max(
                family_counts[(candidate, family)]
                for candidate in main_win_systems
            )
            counts.append(
                _format_win_count(
                    value,
                    family_denominators[family],
                    best=best,
                )
            )
        overall = overall_counts[system_id]
        overall_best = overall == max(
            overall_counts[candidate] for candidate in main_win_systems
        )
        lines.append(
            f"{SUMMARY_LABELS[system_id]} & "
            + " & ".join(counts)
            + " & "
            + _format_win_count(overall, len(rows), best=overall_best)
            + r" \\"
        )
    lines.extend(
        [
            r"\midrule",
            r"\multicolumn{5}{@{}l}{Controls and ablations} \\",
        ]
    )
    for system_id in ablation_win_systems:
        counts = []
        for family in SCOPE:
            value = family_counts[(system_id, family)]
            best = value == max(
                family_counts[(candidate, family)]
                for candidate in ablation_win_systems
            )
            counts.append(
                _format_win_count(
                    value,
                    family_denominators[family],
                    best=best,
                )
            )
        overall = overall_counts[system_id]
        overall_best = overall == max(
            overall_counts[candidate] for candidate in ablation_win_systems
        )
        lines.append(
            f"{SUMMARY_LABELS[system_id]} & "
            + " & ".join(counts)
            + " & "
            + _format_win_count(overall, len(rows), best=overall_best)
            + r" \\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )
    if native_pairs:
        lines.extend(
            [
                r"\par\vspace{3pt}",
                r"\small",
                r"\begin{tabular}{@{}lrrr@{}}",
                r"\toprule",
                r"Native method & Paired configurations & Strict wins & Non-strict \\",
                r"\midrule",
            ]
        )
        for method, specification in NATIVE_METHODS.items():
            members = [
                pair for pair in native_pairs if pair["method"] == method
            ]
            wins = sum(
                all(float(value) < 0.0 for value in pair["delta"].values())
                for pair in members
            )
            total = specification["pairs"]
            lines.append(
                rf"{method}~\citep{{{specification['citation']}}} & "
                f"{total} & {wins} & {total - wins} \\\\"
            )
        lines.extend([r"\bottomrule", r"\end{tabular}"])
    lines.extend([r"\end{table*}", ""])
    return "\n".join(lines)


def render_backbones(rows: list[dict]) -> str:
    systems = MAIN_SYSTEMS[1:]
    cells_per_model = sum(
        len(scope["datasets"]) * len(scope["horizons"])
        for scope in SCOPE.values()
    )
    lines = [
        "% Generated by paper/tools/generate_retrieval_baseline_tables.py.",
        r"\begin{table*}[t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        (
            r"\caption{Strict wins by frozen backbone in the full "
            f"{EXPECTED_CELLS}-cell checkpoint-replay comparison. Every "
            f"entry is a count out of {cells_per_model}; "
            r"Appendix~\ref{app:retrieval-baselines} reports all absolute "
            r"metrics.}"
        ),
        r"\label{tab:retrieval-baseline-backbones}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        (
            "Backbone & "
            + " & ".join(SUMMARY_LABELS[system] for system in systems)
            + r" \\"
        ),
        r"\midrule",
    ]
    for model in MODELS:
        members = [row for row in rows if row["model"] == model]
        values = [
            sum(
                bool(row["systems"][system_id]["strict_win"])
                for row in members
            )
            for system_id in systems
        ]
        lines.append(
            f"{_model_label(model)} & "
            + " & ".join(f"{value}/{cells_per_model}" for value in values)
            + r" \\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
            "",
        ]
    )
    return "\n".join(lines)


def render_full(
    rows: list[dict],
    systems: tuple[str, ...] = MAIN_SYSTEMS,
    *,
    table_role: str = "Published-operator comparison",
    label_prefix: str = "retrieval-full",
) -> str:
    indexed = {row["cell_id"]: row for row in rows}
    lines = [
        "% Generated by paper/tools/generate_retrieval_baseline_tables.py.",
        "% Every checkpoint-replay system/cell result appears exactly once.",
    ]
    for family, scope in SCOPE.items():
        for dataset in scope["datasets"]:
            for horizon in scope["horizons"]:
                metrics = scope["metrics"]
                columns = "ll" + "r" * len(systems)
                header = (
                    "Metric & Backbone & "
                    + " & ".join(
                        SUMMARY_LABELS[system] for system in systems
                    )
                    + r" \\"
                )
                lines.extend(
                    [
                        r"\begingroup",
                        r"\footnotesize",
                        r"\setlength{\tabcolsep}{1.5pt}",
                        r"\renewcommand{\arraystretch}{0.90}",
                        f"\\begin{{longtable}}{{{columns}}}",
                        (
                            f"\\caption{{{table_role} on {_escape(dataset)} "
                            f"at horizon {horizon}. Each row reports "
                            "absolute errors for one backbone and metric; "
                            "bold marks the best result within each "
                            "backbone/metric row.}"
                            f"\\label{{tab:{label_prefix}-{family}-"
                            f"{dataset.lower()}-h{horizon}}}\\\\"
                        ),
                        r"\toprule",
                        header,
                        r"\midrule",
                        r"\endfirsthead",
                        r"\multicolumn{"
                        + str(2 + len(systems))
                        + r"}{c}{\tablename\ \thetable\ continued} \\",
                        r"\toprule",
                        header,
                        r"\midrule",
                        r"\endhead",
                        r"\midrule",
                        r"\multicolumn{"
                        + str(2 + len(systems))
                        + r"}{r}{Continued on next page} \\",
                        r"\endfoot",
                        r"\bottomrule",
                        r"\endlastfoot",
                    ]
                )
                for metric_index, metric in enumerate(metrics):
                    if metric_index:
                        lines.append(r"\midrule")
                    for model in MODELS:
                        cell_id = (
                            f"{family}/{dataset}/{model}/{horizon}"
                        )
                        row = indexed[cell_id]
                        if metric_index == 0:
                            lines.append(f"% cell {cell_id}")
                        best = min(
                            float(
                                row["systems"][system_id]["test_metrics"][
                                    metric
                                ]
                            )
                            for system_id in systems
                        )
                        system_values = []
                        for system_id in systems:
                            value = float(
                                row["systems"][system_id]["test_metrics"][
                                    metric
                                ]
                            )
                            formatted = _fmt_metric(family, metric, value)
                            if value == best:
                                formatted = rf"\textbf{{{formatted}}}"
                            system_values.append(formatted)
                        lines.append(
                            f"{metric.upper()} & {_model_label(model)} & "
                            + " & ".join(system_values)
                            + r" \\"
                        )
                lines.extend(
                    [
                        r"\end{longtable}",
                        r"\endgroup",
                        "",
                    ]
                )
    return "\n".join(lines)


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


def main() -> None:
    args = parse_args()
    document = json.loads(args.input.read_text(encoding="utf-8"))
    rows = validate(document, args)
    native_document = json.loads(
        args.native_input.read_text(encoding="utf-8")
    )
    native_pairs = validate_native(native_document)
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
    (table_root / "retrieval_baseline_model_columns.tex").write_text(
        render_model_columns(rows, native_pairs),
        encoding="utf-8",
    )
    (table_root / "retrieval_long_term_model_values.tex").write_text(
        render_value_table(rows, "long_term"),
        encoding="utf-8",
    )
    (table_root / "retrieval_short_model_values.tex").write_text(
        render_short_value_table(rows),
        encoding="utf-8",
    )
    (table_root / "native_retrieval_baseline_values.tex").write_text(
        render_native_values(native_pairs),
        encoding="utf-8",
    )
    (table_root / "native_retrieval_baseline_full.tex").write_text(
        render_native_full(native_pairs),
        encoding="utf-8",
    )
    (table_root / "retrieval_baseline_full.tex").write_text(
        render_full(rows),
        encoding="utf-8",
    )
    (table_root / "retrieval_ablation_full.tex").write_text(
        render_full(
            rows,
            ABLATION_SYSTEMS,
            table_role="Control and ablation comparison",
            label_prefix="retrieval-ablation-full",
        ),
        encoding="utf-8",
    )
    (data_root / "retrieval_baseline_results.json").write_text(
        json.dumps(
            build_data(
                document,
                rows,
                provenance={
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
    (data_root / "native_retrieval_baseline_results.json").write_text(
        json.dumps(native_document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
