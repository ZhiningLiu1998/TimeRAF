import hashlib
import json
from argparse import Namespace
from pathlib import Path

from paper.tools.generate_retrieval_baseline_tables import (
    ABLATION_SYSTEMS,
    EXPECTED_CELLS,
    EXPECTED_SYSTEMS,
    NATIVE_METHODS,
    MODEL_COLUMN_LABELS,
    MODELS,
    SCOPE,
    _fmt_delta,
    aggregate_model_metric,
    aggregate_native_metric,
    build_data,
    expected_ids,
    render_full,
    render_model_columns,
    render_native_full,
    render_native_values,
    render_short_value_table,
    render_summary,
    render_value_table,
    validate,
    validate_completion_receipt,
    validate_native,
    validate_provenance,
)
from scripts.generate_native_retrieval_baseline_manifest import build_manifest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _document():
    rows = []
    for family, scope in SCOPE.items():
        for dataset in scope["datasets"]:
            for model in MODELS:
                for horizon in scope["horizons"]:
                    metrics = scope["metrics"]
                    systems = {}
                    for system_index, system_id in enumerate(EXPECTED_SYSTEMS):
                        value = 1.0 - 0.01 * system_index
                        systems[system_id] = {
                            "test_metrics": {
                                metric: value for metric in metrics
                            },
                            "metric_gain_percent": {
                                metric: 100.0 * (1.0 - value)
                                for metric in metrics
                            },
                            "strict_win": system_id != "base",
                        }
                    rows.append(
                        {
                            "cell_id": (
                                f"{family}/{dataset}/{model}/{horizon}"
                            ),
                            "task_family": family,
                            "dataset": dataset,
                            "model": model,
                            "pred_len": horizon,
                            "state": "completed",
                            "systems": systems,
                            "base_metric_recomputation": {"passed": True},
                            "a10g_metric_reference": {
                                "comparison_only": True,
                            },
                            "no_lookahead": {
                                "retrieval_systems": {"passed": True},
                                "timeraf": {"passed": True},
                            },
                        }
                    )
    return {
        "source_revision": "source",
        "launcher_revision": "launcher",
        "protocol_sha256": "protocol",
        "catalog_sha256": "catalog",
        "expected_cells": EXPECTED_CELLS,
        "expected_systems": list(EXPECTED_SYSTEMS),
        "counts": {
            "completed": EXPECTED_CELLS,
            "failed": 0,
            "incomplete": 0,
            "pending": 0,
            "running": 0,
        },
        "all_completed": True,
        "cell_states": rows,
    }


def _native_document():
    protocol = json.loads(
        (
            PROJECT_ROOT / "docs/native_retrieval_baseline_protocol.json"
        ).read_text(encoding="utf-8")
    )
    manifest = build_manifest(protocol)
    pairs = []
    method_indices = {method: 0 for method in protocol["systems"]}
    for row in manifest:
        system = protocol["systems"][row["method"]]
        metrics = row["metrics"]
        index = method_indices[row["method"]]
        method_indices[row["method"]] += 1
        base = {metric: 2.0 + index for metric in metrics}
        retrieval = {metric: 1.5 + index for metric in metrics}
        pairs.append(
            {
                "cell_id": row["id"],
                "method": system["retrieval_system"],
                "base_method": system["base_system"],
                "backbone": row["backbone"],
                "dataset": row["dataset"],
                "context_length": row["context_length"],
                "prediction_length": row["prediction_length"],
                "metrics": list(metrics),
                "base": base,
                "retrieval": retrieval,
                "delta": {
                    metric: retrieval[metric] - base[metric]
                    for metric in metrics
                },
            }
        )
    return {
        "schema_version": 2,
        "protocol_sha256": "a" * 64,
        "execution_record": {
            "path": "docs/native_retrieval_baseline_p5_execution.json",
            "sha256": "b" * 64,
        },
        "source_summaries": {
            "raf": {
                "path": "outputs/raf/summary.json",
                "sha256": "c" * 64,
                "source_revision": "raf-source",
                "launcher_revision": "raf-launcher",
                "topology_path": "outputs/raf/topology.json",
                "topology_sha256": "d" * 64,
                "pairs": 44,
            },
            "ts_rag": {
                "path": "outputs/ts-rag/summary.json",
                "sha256": "e" * 64,
                "source_revision": "ts-rag-source",
                "topology_path": "outputs/ts-rag/topology.json",
                "topology_sha256": "f" * 64,
                "pairs": 7,
            },
            "ratd": {
                "path": "outputs/ratd/summary.json",
                "sha256": "0" * 64,
                "source_revision": "ratd-source",
                "topology_path": "outputs/ratd/topology.json",
                "topology_sha256": "1" * 64,
                "pairs": 1,
            },
        },
        "expected_pairs": 52,
        "completed_pairs": 52,
        "all_completed": True,
        "pairs": pairs,
    }


def test_complete_retrieval_matrix_renders_all_dataset_tables():
    document = _document()
    args = Namespace(
        expected_source_revision="source",
        expected_launcher_revision="launcher",
        expected_protocol_sha256="protocol",
        expected_catalog_sha256="catalog",
    )

    rows = validate(document, args)
    latex = render_full(rows)

    assert len(rows) == 585
    assert len(expected_ids()) == 585
    assert latex.count(r"\begin{longtable}") == 45
    assert latex.count("% cell ") == 585
    assert "fixed forecasts.}}\\label" not in latex


def test_summary_reports_base_and_all_retrieval_systems():
    document = _document()
    rows = validate(
        document,
        Namespace(
            expected_source_revision="source",
            expected_launcher_revision="launcher",
            expected_protocol_sha256="protocol",
            expected_catalog_sha256="catalog",
        ),
    )

    latex = render_summary(rows)

    assert "Base & 0/364 & 0/156 & 0/65 & 0/585" in latex
    assert (
        r"\method{} & 364/364 & 156/156 & 65/65 & 585/585"
        in latex
    )
    assert r"$\Delta$MSE" in latex


def test_unified_win_table_uses_family_denominators():
    document = _document()
    rows = validate(
        document,
        Namespace(
            expected_source_revision="source",
            expected_launcher_revision="launcher",
            expected_protocol_sha256="protocol",
            expected_catalog_sha256="catalog",
        ),
    )

    native_pairs = validate_native(_native_document())
    latex = render_model_columns(rows, native_pairs)

    assert r"\begin{tabular}{@{}lrrrr@{}}" in latex
    assert "System & LT & PEMS & EPF & All" in latex
    assert (
        r"RAFT & \textbf{364/364} & \textbf{156/156} & "
        r"\textbf{65/65} & \textbf{585/585}"
    ) in latex
    assert latex.count(r"\method{} &") == 1
    assert "Native method & Paired configurations" in latex
    assert r"RAF~\citep{tire2024raf} & 44 & 44 & 0" in latex
    assert "adapted" not in latex.lower()


def test_value_tables_report_absolute_error_and_delta_by_model():
    document = _document()
    rows = validate(
        document,
        Namespace(
            expected_source_revision="source",
            expected_launcher_revision="launcher",
            expected_protocol_sha256="protocol",
            expected_catalog_sha256="catalog",
        ),
    )

    absolute, delta = aggregate_model_metric(
        rows,
        "long_term",
        "TimeXer",
        "timeraf",
        "mse",
    )
    latex = render_value_table(rows, "long_term")

    assert absolute == 0.95
    assert abs(delta + 0.05) < 1e-12
    expected_header = (
        "System & "
        + " & ".join(MODEL_COLUMN_LABELS[model] for model in MODELS)
    )
    assert expected_header in latex
    assert latex.count(r"\begin{tabular}{@{}l*{13}{c}@{}}") == 1
    assert expected_header.count(" & ") == 13
    assert r"\scriptsize" in latex
    assert r"\resizebox{\textwidth}{!}" in latex
    assert "all datasets and horizons" in latex
    assert (
        r"\method{} & $\mathbf{0.950}$"
        in latex
    )
    assert r"\quad $\Delta$ & $\mathbf{-0.050}$" in latex
    assert latex.count(r"\mathbf{0.950}") == 2 * len(MODEL_COLUMN_LABELS)
    assert "Analog-kNN" not in latex
    assert "Residual-kNN" not in latex
    assert "panel" not in latex.lower()
    assert "adapted" not in latex.lower()
    assert "Strict" not in latex
    assert "win" not in latex.lower()


def test_value_table_average_includes_every_family_horizon():
    rows = validate(
        _document(),
        Namespace(
            expected_source_revision="source",
            expected_launcher_revision="launcher",
            expected_protocol_sha256="protocol",
            expected_catalog_sha256="catalog",
        ),
    )
    target = next(
        row
        for row in rows
        if row["cell_id"] == "long_term/ETTh1/TimeXer/720"
    )
    target["systems"]["timeraf"]["test_metrics"]["mse"] = 28.95

    absolute, delta = aggregate_model_metric(
        rows,
        "long_term",
        "TimeXer",
        "timeraf",
        "mse",
    )

    assert abs(absolute - 1.95) < 1e-12
    assert abs(delta - 0.95) < 1e-12


def test_delta_format_normalizes_rounded_zero():
    assert _fmt_delta("long_term", "mse", -0.0001) == "0.000"
    assert _fmt_delta("long_term", "mse", 0.0001) == "0.000"
    assert _fmt_delta("long_term", "mse", -0.011) == "-0.011"
    assert _fmt_delta("pems", "mae", 0.111) == "+0.11"


def test_short_table_combines_pems_and_epf_without_controls():
    rows = validate(
        _document(),
        Namespace(
            expected_source_revision="source",
            expected_launcher_revision="launcher",
            expected_protocol_sha256="protocol",
            expected_catalog_sha256="catalog",
        ),
    )

    latex = render_short_value_table(rows)

    assert r"\label{tab:retrieval-short-values}" in latex
    assert "PEMS, MAE" in latex
    assert "EPF, MSE" in latex
    assert "Analog-kNN" not in latex
    assert "Residual-kNN" not in latex


def test_value_table_bolds_only_the_raw_best_before_rounding():
    rows = validate(
        _document(),
        Namespace(
            expected_source_revision="source",
            expected_launcher_revision="launcher",
            expected_protocol_sha256="protocol",
            expected_catalog_sha256="catalog",
        ),
    )
    values = {
        "base": 1.0000000000004,
        "raft_adapted": 1.0000000000003,
        "saraf_adapted": 1.0000000000002,
        "timeraf": 1.0000000000001,
    }
    for row in rows:
        if row["task_family"] == "pems" and row["model"] == "LightTS":
            for system_id, value in values.items():
                row["systems"][system_id]["test_metrics"]["mae"] = value

    latex = render_short_value_table(rows)
    value_rows = [
        line
        for line in latex.splitlines()
        if line.startswith(("Base &", "RAFT &", "SARAF &", r"\method{} &"))
    ][:4]
    model_column = MODELS.index("LightTS") + 1
    cells = [line.split(" & ")[model_column] for line in value_rows]

    assert cells == ["$1.00$", "$1.00$", "$1.00$", r"$\mathbf{1.00}$"]


def test_full_and_ablation_tables_use_distinct_labels():
    rows = validate(
        _document(),
        Namespace(
            expected_source_revision="source",
            expected_launcher_revision="launcher",
            expected_protocol_sha256="protocol",
            expected_catalog_sha256="catalog",
        ),
    )

    published = render_full(rows)
    ablation = render_full(
        rows,
        ABLATION_SYSTEMS,
        table_role="Control and ablation comparison",
        label_prefix="retrieval-ablation-full",
    )

    assert r"\label{tab:retrieval-full-long_term-etth1-h96}" in published
    assert r"\label{tab:retrieval-full-long_term-etth1-h720}" in published
    assert (
        r"\label{tab:retrieval-ablation-full-long_term-etth1-h96}"
        in ablation
    )
    published_labels = {
        line for line in published.splitlines() if r"\label{" in line
    }
    ablation_labels = {
        line for line in ablation.splitlines() if r"\label{" in line
    }
    assert published_labels.isdisjoint(ablation_labels)


def test_native_table_reports_base_retrieval_and_absolute_delta():
    pairs = validate_native(_native_document())

    base, retrieval, delta = aggregate_native_metric(
        pairs, "RATD", "rmse"
    )
    latex = render_native_values(pairs)

    assert base == 2.0
    assert retrieval == 1.5
    assert delta == -0.5
    assert r"RATD~\citep{liu2024ratd}" in latex
    assert r"\textbf{1.5000}" in latex
    assert r"\textbf{-0.5000}" in latex
    assert "win" not in latex.lower()

    full_latex = render_native_full(pairs)
    assert r"\label{tab:native-retrieval-full}" in full_latex
    assert r"\begin{longtable}{@{}lllrrrr@{}}" in full_latex
    assert "Method & Dataset & Metric & C/H & Base & Retrieval" in full_latex
    assert "RATD & electricity & RMSE & 96/168" in full_latex
    assert full_latex.count(r"\textbf{-0.5000}") == 104


def test_native_table_bolds_both_values_when_they_tie():
    document = _native_document()
    for pair in document["pairs"]:
        if pair["method"] == "RAF":
            for metric in pair["metrics"]:
                pair["retrieval"][metric] = pair["base"][metric]
                pair["delta"][metric] = 0.0

    latex = render_native_values(validate_native(document))

    assert (
        r"RAF~\citep{tire2024raf} & WQL & "
        r"\textbf{23.5000} & \textbf{23.5000} & 0.0000"
    ) in latex


def test_native_validation_rejects_missing_provenance():
    document = _native_document()
    del document["execution_record"]

    try:
        validate_native(document)
    except ValueError as error:
        assert str(error) == "Native execution record is missing"
    else:
        raise AssertionError("Missing native provenance was accepted")


def test_native_validation_rejects_invalid_provenance_hash():
    document = _native_document()
    document["source_summaries"]["ts_rag"]["topology_sha256"] = "A" * 64

    try:
        validate_native(document)
    except ValueError as error:
        assert str(error) == "Native ts_rag topology hash is invalid"
    else:
        raise AssertionError("Invalid native provenance hash was accepted")


def test_native_validation_rejects_source_pair_count_drift():
    document = _native_document()
    document["source_summaries"]["raf"]["pairs"] = 43

    try:
        validate_native(document)
    except ValueError as error:
        assert str(error) == "Native raf source pair count drifted"
    else:
        raise AssertionError("Native source pair count drift was accepted")


def test_native_validation_rejects_noncanonical_cell_id():
    document = _native_document()
    document["pairs"][0]["cell_id"] = "raf/not-a-canonical-cell"

    try:
        validate_native(document)
    except ValueError as error:
        assert "native baseline cell ID is not canonical" in str(error)
    else:
        raise AssertionError("Noncanonical native cell ID was accepted")


def test_native_validation_rejects_canonical_dataset_drift():
    document = _native_document()
    document["pairs"][0]["dataset"] = "wrong-dataset"

    try:
        validate_native(document)
    except ValueError as error:
        assert "native canonical dataset mismatch" in str(error)
    else:
        raise AssertionError("Native canonical dataset drift was accepted")


def test_native_validation_rejects_canonical_backbone_drift():
    document = _native_document()
    document["pairs"][0]["backbone"] = "wrong-backbone"

    try:
        validate_native(document)
    except ValueError as error:
        assert "native canonical backbone mismatch" in str(error)
    else:
        raise AssertionError("Native canonical backbone drift was accepted")


def test_compact_data_includes_base_statistics():
    document = _document()
    rows = validate(
        document,
        Namespace(
            expected_source_revision="source",
            expected_launcher_revision="launcher",
            expected_protocol_sha256="protocol",
            expected_catalog_sha256="catalog",
        ),
    )

    data = build_data(document, rows)

    assert data["system_statistics"]["base"]["cells"] == 585
    assert data["system_statistics"]["base"]["strict_wins"] == 0
    assert data["system_statistics"]["base"]["median_paired_gain_percent"] == {
        "mae": 0.0,
        "mape": 0.0,
        "mse": 0.0,
        "rmse": 0.0,
    }


def test_validation_rejects_catalog_identity_drift():
    document = _document()

    try:
        validate(
            document,
            Namespace(
                expected_source_revision="source",
                expected_launcher_revision="launcher",
                expected_protocol_sha256="protocol",
                expected_catalog_sha256="different-catalog",
            ),
        )
    except ValueError as error:
        assert str(error) == "Retrieval baseline catalog hash drifted"
    else:
        raise AssertionError("Catalog identity drift was accepted")


def test_validation_rejects_cell_identity_field_drift():
    document = _document()
    document["cell_states"][0]["pred_len"] = 999

    try:
        validate(
            document,
            Namespace(
                expected_source_revision="source",
                expected_launcher_revision="launcher",
                expected_protocol_sha256="protocol",
                expected_catalog_sha256="catalog",
            ),
        )
    except ValueError as error:
        assert "Cell identity fields do not match cell ID" in str(error)
    else:
        raise AssertionError("Cell identity field drift was accepted")


def test_validation_rejects_metric_gain_drift():
    document = _document()
    document["cell_states"][0]["systems"]["timeraf"][
        "metric_gain_percent"
    ]["mse"] = 999.0

    try:
        validate(
            document,
            Namespace(
                expected_source_revision="source",
                expected_launcher_revision="launcher",
                expected_protocol_sha256="protocol",
                expected_catalog_sha256="catalog",
            ),
        )
    except ValueError as error:
        assert "Metric gain mismatch" in str(error)
    else:
        raise AssertionError("Metric-gain drift was accepted")


def test_provenance_requires_distinct_eight_gpu_bindings():
    document = _document()
    run_metadata = {
        "source_revision": "source",
        "launcher_revision": "launcher",
        "protocol_sha256": "protocol",
        "catalog_sha256": "catalog",
        "selected_cells": 585,
        "topology": {
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
        },
    }
    startup_topology = {
        "all_bindings_observed": True,
        "expected_worker_count": 8,
        "distinct_positive_pids": True,
        "gpu_ids": list(range(8)),
        "worker_bindings": [
            {
                "worker_gpu": gpu,
                "pid": 100 + gpu,
                "gpu_uuid": f"gpu-{gpu}",
            }
            for gpu in range(8)
        ],
    }

    validate_provenance(document, run_metadata, startup_topology)

    startup_topology["worker_bindings"][7]["pid"] = 100
    try:
        validate_provenance(document, run_metadata, startup_topology)
    except ValueError as error:
        assert str(error) == "Startup worker bindings are invalid"
    else:
        raise AssertionError("Duplicate GPU process binding was accepted")


def test_completion_receipt_rehashes_publication_inputs(tmp_path):
    document = _document()
    paths = {}
    artifacts = {}
    for suffix, content in (
        ("retrieval_matrix/matrix_summary.json", b"summary\n"),
        ("retrieval_matrix/run_metadata.json", b"metadata\n"),
        ("retrieval_matrix/startup_topology.json", b"topology\n"),
    ):
        path = tmp_path / Path(suffix).name
        path.write_bytes(content)
        paths[suffix] = path
        artifacts[suffix] = {
            "path": f"/remote/run/{suffix}",
            "size_bytes": len(content),
            "sha256": __import__("hashlib").sha256(content).hexdigest(),
        }
    artifacts["prediction_bundle_catalog.json"] = {
        "path": "/remote/run/prediction_bundle_catalog.json",
        "size_bytes": 123,
        "sha256": "catalog",
    }
    receipt = {
        "schema_version": 1,
        "status": "completed",
        "expected_cells": EXPECTED_CELLS,
        "source_revision": "source",
        "artifacts": artifacts,
    }

    validate_completion_receipt(receipt, document, paths)

    paths["retrieval_matrix/matrix_summary.json"].write_bytes(b"changed\n")
    try:
        validate_completion_receipt(receipt, document, paths)
    except ValueError as error:
        assert "hash mismatch" in str(error)
    else:
        raise AssertionError("Changed publication input was accepted")


def test_committed_full_horizon_publication_artifacts_are_fresh():
    publication_root = PROJECT_ROOT / "docs/publication_results"
    paper_root = PROJECT_ROOT / "paper"
    summary_path = publication_root / "retrieval_baseline_summary.json"
    metadata_path = (
        publication_root / "retrieval_baseline_run_metadata.json"
    )
    topology_path = (
        publication_root / "retrieval_baseline_startup_topology.json"
    )
    receipt_path = (
        publication_root / "retrieval_baseline_completion_receipt.json"
    )
    catalog_path = (
        publication_root / "retrieval_baseline_bundle_catalog.json"
    )
    native_path = (
        publication_root / "native_retrieval/native-52-schema-v2.json"
    )
    document = json.loads(summary_path.read_text(encoding="utf-8"))
    args = Namespace(
        expected_source_revision=(
            "e97746ed847f25d1f11f47242bc04db2b48b83ef"
        ),
        expected_launcher_revision=(
            "cc80cabd363ac136c5d2e9284529e912b601544e"
        ),
        expected_protocol_sha256=(
            "2d3a9a6c66ae80808c0d734d48996341cb2d1fcb123cf04e81ab8dc1713a0bc6"
        ),
        expected_catalog_sha256=(
            "cd92b7e1a35341c3175a8e8c3468c41cc79cf0c9e508c5a39d0cdaa3255de1b8"
        ),
    )
    rows = validate(document, args)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    topology = json.loads(topology_path.read_text(encoding="utf-8"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    native_pairs = validate_native(
        json.loads(native_path.read_text(encoding="utf-8"))
    )
    validate_provenance(document, metadata, topology)
    validate_completion_receipt(
        receipt,
        document,
        {
            "retrieval_matrix/matrix_summary.json": summary_path,
            "retrieval_matrix/run_metadata.json": metadata_path,
            "retrieval_matrix/startup_topology.json": topology_path,
        },
    )
    assert hashlib.sha256(catalog_path.read_bytes()).hexdigest() == (
        document["catalog_sha256"]
    )

    expected_tables = {
        "retrieval_baseline_model_columns.tex": render_model_columns(
            rows, native_pairs
        ),
        "retrieval_long_term_model_values.tex": render_value_table(
            rows, "long_term"
        ),
        "retrieval_short_model_values.tex": render_short_value_table(rows),
        "retrieval_baseline_full.tex": render_full(rows),
        "retrieval_ablation_full.tex": render_full(
            rows,
            ABLATION_SYSTEMS,
            table_role="Control and ablation comparison",
            label_prefix="retrieval-ablation-full",
        ),
    }
    for filename, expected in expected_tables.items():
        assert (
            paper_root / "latex/tables" / filename
        ).read_text(encoding="utf-8") == expected

    expected_data = build_data(
        document,
        rows,
        provenance={
            "run_metadata_sha256": hashlib.sha256(
                metadata_path.read_bytes()
            ).hexdigest(),
            "startup_topology_sha256": hashlib.sha256(
                topology_path.read_bytes()
            ).hexdigest(),
        },
    )
    assert json.loads(
        (paper_root / "data/retrieval_baseline_results.json").read_text(
            encoding="utf-8"
        )
    ) == expected_data
