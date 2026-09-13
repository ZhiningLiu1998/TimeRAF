import hashlib
import json
from argparse import Namespace
from pathlib import Path

from paper.tools.generate_retrieval_baseline_tables import (
    EXPECTED_CELLS,
    EXPECTED_SYSTEMS,
    MODEL_COLUMN_LABELS,
    MODELS,
    SCOPE,
    _fmt_delta,
    aggregate_model_metric,
    build_data,
    expected_ids,
    render_short_value_table,
    render_value_table,
    validate,
    validate_completion_receipt,
    validate_provenance,
)


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
        "retrieval_long_term_model_values.tex": render_value_table(
            rows, "long_term"
        ),
        "retrieval_short_model_values.tex": render_short_value_table(rows),
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
