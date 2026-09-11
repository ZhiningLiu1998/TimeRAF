import hashlib
import json
from pathlib import Path

import pytest

from ts_rag.appendix_benchmark import (
    autogluon_checkpoint_compatibility,
)
from ts_rag.appendix_matrix import (
    build_appendix_summary,
    load_bundle_catalog,
    validate_catalog_scope,
)
from ts_rag.appendix_rag import load_appendix_manifest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
APPENDIX_MANIFEST = (
    REPOSITORY_ROOT / "docs" / "timefuse_appendix_experiment_manifest.jsonl"
)


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _cell(cell_id, family, baseline, evaluable=True):
    metrics = ["mae", "rmse"] if family == "pems" else ["mse", "mae"]
    return {
        "id": cell_id,
        "task_family": family,
        "dataset": cell_id.split("/")[1],
        "baseline": baseline,
        "pred_len": 24 if family == "pems" else 96,
        "metrics": metrics,
        "paper_status": {
            "evaluable": evaluable,
            "status": "reported" if evaluable else "paper_oot",
        },
    }


def _write_attempt(
    root,
    name,
    cell,
    improved=True,
    revision="current",
    method_revision=None,
):
    artifact = root / name
    artifact.mkdir()
    status = "completed" if cell["paper_status"]["evaluable"] else "paper_oot"
    status_payload = {
        "cell_id": cell["id"],
        "source_revision": revision,
        "started_unix": 1,
        "status": status,
    }
    if method_revision is not None:
        status_payload["method_revision"] = method_revision
    (artifact / "status.json").write_text(
        json.dumps(status_payload)
    )
    result = {
        "evaluation_status": status,
        "cell": cell,
        "paper_status": cell["paper_status"],
    }
    if cell["paper_status"]["evaluable"]:
        result.update(
            {
                "all_test_metrics_improve": improved,
                "test_baseline": {
                    metric: 100.0 for metric in cell["metrics"]
                },
                "test_corrected": {
                    metric: 90.0 if improved else 101.0
                    for metric in cell["metrics"]
                },
                "validation": {"selected": {"method": "test"}},
            }
        )
    (artifact / "result.json").write_text(json.dumps(result))


@pytest.fixture
def full_bundle_catalog(tmp_path, monkeypatch):
    manifest = tmp_path / "docs" / APPENDIX_MANIFEST.name
    manifest.parent.mkdir(parents=True)
    manifest.write_bytes(APPENDIX_MANIFEST.read_bytes())
    rows = load_appendix_manifest(manifest)
    replay_catalog = (
        tmp_path
        / "outputs"
        / "greenland_runs"
        / "replay"
        / "replay_bundle_catalog.json"
    )
    replay_catalog.parent.mkdir(parents=True)
    replay_catalog.write_text(
        json.dumps({"schema_version": 1, "bundles": {}}),
        encoding="utf-8",
    )
    replay_catalog_sha256 = _sha256(replay_catalog)
    bundles = {}
    for index, cell in enumerate(rows):
        if not cell["paper_status"]["evaluable"]:
            continue
        bundle = (
            tmp_path
            / "outputs"
            / "appendix_bundles"
            / f"{index:03d}"
            / "prediction_bundle.npz"
        )
        bundle.parent.mkdir(parents=True)
        bundle.write_bytes(f"bundle:{cell['id']}".encode())
        bundle_sha256 = _sha256(bundle)
        metadata = bundle.with_suffix(".npz.json")
        run_identity = {"source_revision": "source-revision"}
        if cell["baseline"] in {
            "forward_selection",
            "portfolio_ensemble",
            "zeroshot_ensemble",
        }:
            run_identity["replay_catalog_sha256"] = (
                replay_catalog_sha256
            )
        predictor = None
        if cell["baseline"] in {
            "autogluon_high_quality",
            "chronos_bolt_finetuned",
            "chronos_bolt_zeroshot",
        }:
            run_identity["autogluon_timeseries_version"] = "1.4.0"
            predictor = {
                "autogluon_timeseries_version": "1.4.0",
            }
        compatibility = None
        if cell["baseline"] == "autogluon_high_quality":
            compatibility = autogluon_checkpoint_compatibility(
                cell["baseline"]
            )
            run_identity["checkpoint_compatibility"] = compatibility
            predictor["checkpoint_compatibility"] = compatibility
        metadata.write_text(
            json.dumps(
                {
                    "cell_id": cell["id"],
                    "prediction_bundle_sha256": bundle_sha256,
                    "run_identity": run_identity,
                    **(
                        {"checkpoint_compatibility": compatibility}
                        if compatibility is not None
                        else {}
                    ),
                    **({"predictor": predictor} if predictor else {}),
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        bundles[cell["id"]] = {
            "path": bundle.relative_to(tmp_path).as_posix(),
            "sha256": bundle_sha256,
            "size_bytes": bundle.stat().st_size,
            "metadata_path": metadata.relative_to(tmp_path).as_posix(),
            "metadata_sha256": _sha256(metadata),
            "baseline": cell["baseline"],
            "dataset": cell["dataset"],
            "task_family": cell["task_family"],
            "pred_len": cell["pred_len"],
            **(
                {"autogluon_timeseries_version": "1.4.0"}
                if predictor
                else {}
            ),
            **(
                {"checkpoint_compatibility": compatibility}
                if compatibility is not None
                else {}
            ),
        }
    payload = {
        "schema_version": 2,
        "source_revision": "source-revision",
        "project_root": str(tmp_path),
        "manifest": manifest.relative_to(tmp_path).as_posix(),
        "manifest_sha256": _sha256(manifest),
        "replay_catalog": replay_catalog.relative_to(tmp_path).as_posix(),
        "replay_catalog_sha256": replay_catalog_sha256,
        "hashes_verified": True,
        "counts": {
            "reported": 96,
            "evaluable": 94,
            "cataloged": 94,
            "missing_evaluable": 0,
        },
        "bundles": bundles,
    }
    path = tmp_path / "docs" / "appendix_bundle_catalog.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    return {
        "path": path,
        "manifest": manifest,
        "replay_catalog": replay_catalog,
        "rows": rows,
        "payload": payload,
    }


def test_appendix_summary_uses_evaluable_denominator_and_oot_state(tmp_path):
    manifest = [
        _cell("appendix/D1/A", "long_term", "A"),
        _cell("appendix/D2/A", "long_term", "A"),
        _cell("appendix/D3/B", "pems", "B"),
        _cell("appendix/D4/B", "pems", "B", evaluable=False),
    ]
    for index, cell in enumerate(manifest):
        _write_attempt(tmp_path, str(index), cell, improved=index != 1)

    summary = build_appendix_summary(
        manifest,
        tmp_path,
        source_revision="current",
        expected_total=4,
        expected_evaluable=3,
    )

    assert summary["counts"]["reported"] == 4
    assert summary["counts"]["evaluable"] == 3
    assert summary["counts"]["paper_oot"] == 1
    assert summary["publication_gate"]["overall"][
        "improvement_rate"
    ] == pytest.approx(2 / 3)
    assert not summary["publication_gate"]["development_gate_passed"]


def test_appendix_gate_accepts_frozen_thresholds_per_family_and_baseline(
    tmp_path,
):
    manifest = []
    index = 0
    # Each family and each baseline receives 10 cells with 8 improvements.
    for family in ("long_term", "pems"):
        for baseline in ("A", "B"):
            for offset in range(10):
                cell = _cell(
                    f"appendix/D{index}/{baseline}",
                    family,
                    baseline,
                )
                manifest.append(cell)
                _write_attempt(
                    tmp_path,
                    str(index),
                    cell,
                    improved=offset < 8,
                )
                index += 1

    gate = build_appendix_summary(
        manifest,
        tmp_path,
        source_revision="current",
        expected_total=40,
        expected_evaluable=40,
    )["publication_gate"]

    assert gate["overall"]["improvement_rate"] == pytest.approx(0.8)
    assert all(
        group["improvement_rate"] == pytest.approx(0.8)
        for group in gate["task_families"].values()
    )
    assert all(
        group["improvement_rate"] == pytest.approx(0.8)
        for group in gate["baselines"].values()
    )
    assert gate["development_gate_passed"]


def test_appendix_summary_filters_frozen_method_revision(tmp_path):
    cell = _cell("appendix/D1/A", "long_term", "A")
    _write_attempt(
        tmp_path,
        "wrong",
        cell,
        method_revision="wrong",
    )
    _write_attempt(
        tmp_path,
        "frozen",
        cell,
        method_revision="d9be338",
    )

    summary = build_appendix_summary(
        [cell],
        tmp_path,
        source_revision="current",
        method_revision="d9be338",
        expected_total=1,
        expected_evaluable=1,
    )

    assert summary["counts"]["completed"] == 1
    assert summary["cell_states"][0]["artifact_dir"].endswith("frozen")
    assert summary["cell_states"][0]["method_revision"] == "d9be338"


def test_catalog_validation_accepts_filtered_scope_but_verifies_full_catalog(
    full_bundle_catalog,
):
    catalog = load_bundle_catalog(full_bundle_catalog["path"])
    selected = [
        full_bundle_catalog["rows"][0],
        next(
            row
            for row in full_bundle_catalog["rows"]
            if not row["paper_status"]["evaluable"]
        ),
    ]

    validate_catalog_scope(selected, catalog, allow_extra=True)

    removed = next(iter(catalog["bundles"]))
    del catalog["bundles"][removed]
    with pytest.raises(ValueError, match="must contain 94 bundles"):
        validate_catalog_scope(selected, catalog, allow_extra=True)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema_version", 1, "schema_version=2"),
        ("hashes_verified", False, "hashes_verified"),
        (
            "counts",
            {
                "reported": 96,
                "evaluable": 94,
                "cataloged": 93,
                "missing_evaluable": 1,
            },
            "counts must be",
        ),
    ],
)
def test_catalog_load_rejects_header_mismatch(
    full_bundle_catalog,
    field,
    value,
    message,
):
    payload = dict(full_bundle_catalog["payload"])
    payload[field] = value
    full_bundle_catalog["path"].write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        load_bundle_catalog(full_bundle_catalog["path"])


def test_catalog_rejects_manifest_and_selected_cell_drift(full_bundle_catalog):
    catalog = load_bundle_catalog(full_bundle_catalog["path"])
    selected = [dict(full_bundle_catalog["rows"][0])]
    selected[0]["baseline"] = "drifted"
    with pytest.raises(ValueError, match="manifest cell drifted"):
        validate_catalog_scope(selected, catalog)

    full_bundle_catalog["manifest"].write_bytes(
        full_bundle_catalog["manifest"].read_bytes() + b"\n"
    )
    with pytest.raises(ValueError, match="manifest SHA-256 mismatch"):
        validate_catalog_scope(full_bundle_catalog["rows"], catalog)


def test_catalog_rejects_replay_catalog_path_or_hash_drift(
    full_bundle_catalog,
):
    catalog = load_bundle_catalog(full_bundle_catalog["path"])
    catalog["replay_catalog"] = "../replay.json"
    with pytest.raises(ValueError, match="safe relative path"):
        validate_catalog_scope(full_bundle_catalog["rows"][:1], catalog)

    catalog = load_bundle_catalog(full_bundle_catalog["path"])
    full_bundle_catalog["replay_catalog"].write_bytes(b"drifted")
    with pytest.raises(ValueError, match="Replay catalog SHA-256 mismatch"):
        validate_catalog_scope(full_bundle_catalog["rows"][:1], catalog)


@pytest.mark.parametrize(
    ("target", "message"),
    [
        ("bundle", "bundle SHA-256 mismatch"),
        ("metadata", "metadata SHA-256 mismatch"),
    ],
)
def test_catalog_rejects_bundle_or_metadata_drift(
    full_bundle_catalog,
    target,
    message,
):
    catalog = load_bundle_catalog(full_bundle_catalog["path"])
    record = next(iter(catalog["bundles"].values()))
    path = Path(
        record["path"] if target == "bundle" else record["metadata_path"]
    )
    original = path.read_bytes()
    path.write_bytes(bytes([original[0] ^ 1]) + original[1:])

    with pytest.raises(ValueError, match=message):
        validate_catalog_scope(full_bundle_catalog["rows"][:1], catalog)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("cell_id", "wrong-cell", "metadata cell_id mismatch"),
        (
            "prediction_bundle_sha256",
            "0" * 64,
            "metadata prediction bundle SHA-256 mismatch",
        ),
        (
            "source_revision",
            "wrong-revision",
            "metadata source revision mismatch",
        ),
        (
            "replay_catalog_sha256",
            "0" * 64,
            "metadata replay catalog SHA-256 mismatch",
        ),
    ],
)
def test_catalog_rejects_metadata_provenance_mismatch(
    full_bundle_catalog,
    field,
    value,
    message,
):
    catalog = load_bundle_catalog(full_bundle_catalog["path"])
    cell_id, record = next(iter(catalog["bundles"].items()))
    metadata_path = Path(record["metadata_path"])
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if field in {"source_revision", "replay_catalog_sha256"}:
        metadata["run_identity"][field] = value
    else:
        metadata[field] = value
    metadata_path.write_text(
        json.dumps(metadata, sort_keys=True),
        encoding="utf-8",
    )
    record["metadata_sha256"] = _sha256(metadata_path)

    selected = [
        row for row in full_bundle_catalog["rows"] if row["id"] == cell_id
    ]
    with pytest.raises(ValueError, match=message):
        validate_catalog_scope(selected, catalog)


@pytest.mark.parametrize(
    ("target", "message"),
    [
        ("catalog", "catalog AutoGluon version mismatch"),
        ("run_identity", "run identity AutoGluon version mismatch"),
        ("predictor", "predictor AutoGluon version mismatch"),
    ],
)
def test_catalog_rejects_external_runtime_version_drift(
    full_bundle_catalog,
    target,
    message,
):
    catalog = load_bundle_catalog(full_bundle_catalog["path"])
    cell_id = next(
        cell_id
        for cell_id, record in catalog["bundles"].items()
        if record["baseline"] == "chronos_bolt_zeroshot"
    )
    record = catalog["bundles"][cell_id]
    if target == "catalog":
        record["autogluon_timeseries_version"] = "1.5.0"
    else:
        metadata_path = Path(record["metadata_path"])
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata[target]["autogluon_timeseries_version"] = "1.5.0"
        metadata_path.write_text(
            json.dumps(metadata, sort_keys=True),
            encoding="utf-8",
        )
        record["metadata_sha256"] = _sha256(metadata_path)

    selected = [
        row for row in full_bundle_catalog["rows"] if row["id"] == cell_id
    ]
    with pytest.raises(ValueError, match=message):
        validate_catalog_scope(selected, catalog)


@pytest.mark.parametrize(
    ("target", "message"),
    [
        ("catalog", "catalog checkpoint compatibility mismatch"),
        ("run_identity", "metadata checkpoint compatibility mismatch"),
        ("predictor", "metadata checkpoint compatibility mismatch"),
        ("metadata", "metadata checkpoint compatibility mismatch"),
    ],
)
def test_catalog_rejects_checkpoint_compatibility_drift(
    full_bundle_catalog,
    target,
    message,
):
    catalog = load_bundle_catalog(full_bundle_catalog["path"])
    cell_id = next(
        cell_id
        for cell_id, record in catalog["bundles"].items()
        if record["baseline"] == "autogluon_high_quality"
    )
    record = catalog["bundles"][cell_id]
    if target == "catalog":
        record["checkpoint_compatibility"] = {"enabled": False}
    else:
        metadata_path = Path(record["metadata_path"])
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if target == "metadata":
            metadata["checkpoint_compatibility"] = {"enabled": False}
        else:
            metadata[target]["checkpoint_compatibility"] = {
                "enabled": False,
            }
        metadata_path.write_text(
            json.dumps(metadata, sort_keys=True),
            encoding="utf-8",
        )
        record["metadata_sha256"] = _sha256(metadata_path)

    selected = [
        row for row in full_bundle_catalog["rows"] if row["id"] == cell_id
    ]
    with pytest.raises(ValueError, match=message):
        validate_catalog_scope(selected, catalog)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("path", "../outside.npz", "safe relative path"),
        ("metadata_path", "/tmp/metadata.json", "safe relative path"),
        ("baseline", "wrong-baseline", "identity mismatch"),
        ("dataset", "wrong-dataset", "identity mismatch"),
        ("task_family", "wrong-family", "identity mismatch"),
        ("pred_len", 999, "identity mismatch"),
    ],
)
def test_catalog_rejects_unsafe_paths_and_record_identity_mismatch(
    full_bundle_catalog,
    field,
    value,
    message,
):
    catalog = load_bundle_catalog(full_bundle_catalog["path"])
    record = next(iter(catalog["bundles"].values()))
    record[field] = value

    with pytest.raises(ValueError, match=message):
        validate_catalog_scope(
            full_bundle_catalog["rows"][:1],
            catalog,
            require_files=False,
        )
