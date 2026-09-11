import hashlib
import json

import pytest

from ts_rag.prediction_bundles import build_prediction_bundle_catalog


def _cell(cell_id):
    return {
        "id": cell_id,
        "task_family": "long_term",
        "dataset": "ETTh1",
        "model": "TimeXer",
        "pred_len": 96,
    }


def _write_export(root, cell_id, revision, content=b"bundle"):
    artifact = root / "outputs" / cell_id.replace("/", "_")
    artifact.mkdir(parents=True)
    bundle = artifact / "prediction_bundle.npz"
    bundle.write_bytes(content)
    (artifact / "status.json").write_text(
        json.dumps(
            {
                "cell_id": cell_id,
                "source_revision": revision,
                "status": "completed",
                "export_only": True,
                "started_unix": 1,
            }
        )
    )
    (artifact / "result.json").write_text(
        json.dumps(
            {
                "export_only": True,
                "prediction_bundle": str(bundle.relative_to(root)),
                "prediction_bundle_sha256": None,
            }
        )
    )
    return bundle


def test_prediction_bundle_catalog_indexes_export_only_artifacts(tmp_path):
    cell = _cell("long/ETTh1/TimeXer/96")
    _write_export(tmp_path, cell["id"], "revision")

    catalog = build_prediction_bundle_catalog(
        [cell],
        tmp_path / "outputs",
        tmp_path,
        source_revision="revision",
        compute_hashes=True,
    )

    record = catalog["bundles"][cell["id"]]
    assert catalog["usable_count"] == 1
    assert catalog["missing_count"] == 0
    assert record["usable"]
    assert record["path"].endswith("prediction_bundle.npz")
    assert record["sha256"]


def test_prediction_bundle_catalog_ignores_other_revisions(tmp_path):
    cell = _cell("long/ETTh1/TimeXer/96")
    _write_export(tmp_path, cell["id"], "old")

    catalog = build_prediction_bundle_catalog(
        [cell],
        tmp_path / "outputs",
        tmp_path,
        source_revision="current",
    )

    assert catalog["cataloged_cells"] == 0
    assert catalog["missing_count"] == 1


def test_prediction_bundle_catalog_detects_recorded_hash_drift(tmp_path):
    cell = _cell("long/ETTh1/TimeXer/96")
    _write_export(tmp_path, cell["id"], "revision")
    result = next((tmp_path / "outputs").glob("**/result.json"))
    payload = json.loads(result.read_text())
    payload["prediction_bundle_sha256"] = "0" * 64
    result.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="hash mismatch"):
        build_prediction_bundle_catalog(
            [cell],
            tmp_path / "outputs",
            tmp_path,
            source_revision="revision",
            compute_hashes=True,
        )


def test_prediction_bundle_catalog_rebases_relocated_absolute_path(tmp_path):
    cell = _cell("long/ETTh1/TimeXer/96")
    bundle = _write_export(tmp_path, cell["id"], "revision")
    result = next((tmp_path / "outputs").glob("**/result.json"))
    payload = json.loads(result.read_text())
    payload["prediction_bundle"] = (
        "/workspace/timeraf/outputs/checkpoint_replay/"
        "long/ETTh1/TimeXer/96/prediction_bundle.npz"
    )
    payload["prediction_bundle_sha256"] = hashlib.sha256(
        bundle.read_bytes()
    ).hexdigest()
    result.write_text(json.dumps(payload))

    catalog = build_prediction_bundle_catalog(
        [cell],
        tmp_path / "outputs",
        tmp_path,
        source_revision="revision",
        compute_hashes=True,
    )

    record = catalog["bundles"][cell["id"]]
    assert record["usable"]
    assert record["relocated"]
    assert record["recorded_path"].startswith("/workspace/timeraf/")
    assert tmp_path / record["path"] == bundle
