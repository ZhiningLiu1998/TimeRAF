import hashlib
import json

import pytest

from ts_rag.matrix_checkpoints import (
    build_matrix_checkpoint_catalog,
    validate_selected_checkpoint_catalog,
)


def _write_attempt(root, cell_id, name, checkpoint, source_revision):
    artifact = root / "outputs" / name
    artifact.mkdir(parents=True)
    (artifact / "status.json").write_text(
        json.dumps(
            {
                "cell_id": cell_id,
                "source_revision": source_revision,
                "status": "completed",
            }
        )
    )
    (artifact / "result.json").write_text(
        json.dumps(
            {
                "checkpoint": {
                    "mode": "trained",
                    "checkpoint": str(checkpoint.relative_to(root)),
                }
            }
        )
    )
    return artifact


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_matrix_checkpoint_catalog_indexes_reusable_checkpoints(tmp_path):
    checkpoint = tmp_path / "checkpoints" / "cell" / "checkpoint.pth"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    artifact = _write_attempt(
        tmp_path,
        "long/A/M/96",
        "cell",
        checkpoint,
        "method-revision",
    )
    summary = {
        "source_revision": "method-revision",
        "generated_unix": 123,
        "cell_states": [
            {
                "cell_id": "long/A/M/96",
                "state": "completed",
                "artifact_dir": str(artifact.relative_to(tmp_path)),
            },
            {
                "cell_id": "long/B/M/96",
                "state": "pending",
            },
        ],
    }

    catalog = build_matrix_checkpoint_catalog(
        summary,
        tmp_path,
        compute_hashes=True,
    )

    record = catalog["checkpoints"]["long/A/M/96"]
    assert catalog["checkpoint_count"] == 1
    assert catalog["usable_count"] == 1
    assert catalog["missing_count"] == 0
    assert record["usable"]
    assert record["relative_path"] == "checkpoints/cell/checkpoint.pth"
    assert record["source_revision"] == "method-revision"
    assert record["sha256"]


def test_matrix_checkpoint_catalog_marks_missing_checkpoint(tmp_path):
    checkpoint = tmp_path / "checkpoints" / "missing" / "checkpoint.pth"
    artifact = _write_attempt(
        tmp_path,
        "long/A/M/96",
        "missing",
        checkpoint,
        "method-revision",
    )
    summary = {
        "source_revision": "method-revision",
        "cell_states": [
            {
                "cell_id": "long/A/M/96",
                "state": "completed",
                "artifact_dir": str(artifact.relative_to(tmp_path)),
            }
        ],
    }

    catalog = build_matrix_checkpoint_catalog(summary, tmp_path)

    record = catalog["checkpoints"]["long/A/M/96"]
    assert not record["usable"]
    assert record["reason"] == "checkpoint_missing"
    assert catalog["missing_count"] == 1


def test_matrix_checkpoint_catalog_can_be_relative_to_checkpoint_root(
    tmp_path,
):
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint = checkpoint_root / "cell" / "checkpoint.pth"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    artifact = _write_attempt(
        tmp_path,
        "long/A/M/96",
        "cell",
        checkpoint,
        "method-revision",
    )
    summary = {
        "source_revision": "method-revision",
        "cell_states": [
            {
                "cell_id": "long/A/M/96",
                "state": "completed",
                "artifact_dir": str(artifact.relative_to(tmp_path)),
            }
        ],
    }

    catalog = build_matrix_checkpoint_catalog(
        summary,
        tmp_path,
        compute_hashes=True,
        checkpoint_path_root=checkpoint_root,
    )

    record = catalog["checkpoints"]["long/A/M/96"]
    assert record["relative_path"] == "cell/checkpoint.pth"
    assert catalog["checkpoint_path_root"] == str(checkpoint_root)


def test_matrix_checkpoint_catalog_stages_selected_multi_root_files(
    tmp_path,
):
    first = tmp_path / "checkpoints" / "release" / "first.pth"
    second = tmp_path / "checkpoints" / "resume" / "second.pth"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    first_artifact = _write_attempt(
        tmp_path,
        "long/A/M/96",
        "first",
        first,
        "method-revision",
    )
    second_artifact = _write_attempt(
        tmp_path,
        "long/B/M/96",
        "second",
        second,
        "method-revision",
    )
    summary = {
        "source_revision": "method-revision",
        "cell_states": [
            {
                "cell_id": "long/A/M/96",
                "state": "completed",
                "artifact_dir": str(first_artifact.relative_to(tmp_path)),
            },
            {
                "cell_id": "long/B/M/96",
                "state": "completed",
                "artifact_dir": str(second_artifact.relative_to(tmp_path)),
            },
        ],
    }
    checkpoint_root = tmp_path / "checkpoints" / "replay"
    stage_root = checkpoint_root / "staging" / "controller-revision"

    catalog = build_matrix_checkpoint_catalog(
        summary,
        tmp_path,
        compute_hashes=True,
        checkpoint_path_root=checkpoint_root,
        checkpoint_stage_root=stage_root,
        selected_cell_ids={
            "long/A/M/96",
            "long/B/M/96",
            "long/C/M/96",
        },
    )

    assert catalog["selected_count"] == 3
    assert catalog["checkpoint_count"] == 2
    assert catalog["usable_count"] == 2
    assert catalog["missing_selected"] == ["long/C/M/96"]
    assert catalog["checkpoint_path_root"] == str(checkpoint_root)
    assert catalog["checkpoint_stage_root"] == str(stage_root)
    for cell_id, expected in (
        ("long/A/M/96", b"first"),
        ("long/B/M/96", b"second"),
    ):
        record = catalog["checkpoints"][cell_id]
        assert record["relative_path"].startswith(
            "staging/controller-revision/"
        )
        staged = checkpoint_root / record["relative_path"]
        assert staged.read_bytes() == expected
        assert record["sha256"]
        assert record["staging_mode"] in {"hardlink", "copy"}

    repeated = build_matrix_checkpoint_catalog(
        summary,
        tmp_path,
        compute_hashes=True,
        checkpoint_path_root=checkpoint_root,
        checkpoint_stage_root=stage_root,
        selected_cell_ids={"long/A/M/96", "long/B/M/96"},
    )
    assert {
        record["staging_mode"]
        for record in repeated["checkpoints"].values()
    } == {"reused"}

    with pytest.raises(ValueError, match="must contain"):
        build_matrix_checkpoint_catalog(
            summary,
            tmp_path,
            compute_hashes=True,
            checkpoint_path_root=tmp_path / "checkpoints" / "other",
            checkpoint_stage_root=stage_root,
            selected_cell_ids={"long/A/M/96", "long/B/M/96"},
        )


def test_selected_checkpoint_catalog_requires_exact_hashed_revision_scope(
    tmp_path,
):
    checkpoint = tmp_path / "checkpoints" / "cell" / "checkpoint.pth"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    artifact = _write_attempt(
        tmp_path,
        "long/A/M/96",
        "cell",
        checkpoint,
        "method-revision",
    )
    summary = {
        "source_revision": "method-revision",
        "cell_states": [
            {
                "cell_id": "long/A/M/96",
                "state": "completed",
                "artifact_dir": str(artifact.relative_to(tmp_path)),
            }
        ],
    }
    catalog = build_matrix_checkpoint_catalog(
        summary,
        tmp_path,
        compute_hashes=True,
        checkpoint_stage_root=tmp_path / "staged",
        selected_cell_ids=["long/A/M/96"],
    )

    evidence = validate_selected_checkpoint_catalog(
        catalog,
        ["long/A/M/96"],
        expected_source_revision="method-revision",
        require_hashes=True,
    )

    assert evidence["selected_count"] == 1
    assert evidence["unique_relative_paths"] == 1
    assert evidence["hashes_verified"]


def test_selected_checkpoint_catalog_rejects_incomplete_or_drifted_scope(
    tmp_path,
):
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"checkpoint")
    artifact = _write_attempt(
        tmp_path,
        "long/A/M/96",
        "cell",
        checkpoint,
        "method-revision",
    )
    summary = {
        "source_revision": "method-revision",
        "cell_states": [
            {
                "cell_id": "long/A/M/96",
                "state": "completed",
                "artifact_dir": str(artifact.relative_to(tmp_path)),
            }
        ],
    }
    catalog = build_matrix_checkpoint_catalog(
        summary,
        tmp_path,
        compute_hashes=True,
        selected_cell_ids=["long/A/M/96", "long/B/M/96"],
    )

    with pytest.raises(ValueError, match="does not exactly match"):
        validate_selected_checkpoint_catalog(
            catalog,
            ["long/A/M/96", "long/B/M/96"],
            expected_source_revision="method-revision",
            require_hashes=True,
        )

    complete = build_matrix_checkpoint_catalog(
        summary,
        tmp_path,
        compute_hashes=True,
        selected_cell_ids=["long/A/M/96"],
    )
    complete["checkpoints"]["long/A/M/96"]["source_revision"] = "other"
    with pytest.raises(ValueError, match="source revision drifted"):
        validate_selected_checkpoint_catalog(
            complete,
            ["long/A/M/96"],
            expected_source_revision="method-revision",
            require_hashes=True,
        )


def _recovery_catalog(tmp_path):
    cell_id = "long/A/M/96"
    retained_id = "long/B/M/96"
    checkpoint = tmp_path / "checkpoints" / "recovery" / "checkpoint.pth"
    retained_checkpoint = (
        tmp_path / "checkpoints" / "first-pass" / "checkpoint.pth"
    )
    checkpoint.parent.mkdir(parents=True)
    retained_checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"recovered")
    retained_checkpoint.write_bytes(b"first-pass")
    artifact = _write_attempt(
        tmp_path,
        cell_id,
        "recovery",
        checkpoint,
        "method-revision",
    )
    retained_artifact = _write_attempt(
        tmp_path,
        retained_id,
        "first-pass",
        retained_checkpoint,
        "method-revision",
    )
    marker = {
        "cohort_id": "recovery-v1",
        "affected_cell_ids": [cell_id],
        "selected_profiles": {cell_id: "exact"},
        "selection_uses_metric_quality": False,
    }
    summary = {
        "source_revision": "method-revision",
        "generated_unix": 123,
        "numerical_recovery": marker,
        "cell_states": [
            {
                "cell_id": cell_id,
                "state": "completed",
                "artifact_dir": str(artifact),
                "started_unix": 10.0,
                "numerical_recovery": {
                    "cohort_id": "recovery-v1",
                    "profile": "exact",
                    "selection_uses_metric_quality": False,
                    "original_artifact_dir": str(
                        tmp_path / "outputs" / "primary"
                    ),
                    "selected_artifact_dir": str(artifact),
                    "first_pass_started_unix": 10.0,
                    "recovery_started_unix": 20.0,
                },
            },
            {
                "cell_id": retained_id,
                "state": "completed",
                "artifact_dir": str(retained_artifact),
                "started_unix": 11.0,
                "numerical_recovery": None,
            },
        ],
    }
    summary_path = tmp_path / "operations" / "composed_a10g.json"
    summary_path.parent.mkdir(parents=True)
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    receipt = {
        **marker,
        "outputs": {
            "a10g": {
                "path": str(summary_path),
                "sha256": _sha256(summary_path),
            }
        },
    }
    receipt_path = tmp_path / "operations" / "composition_receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    protocol_path = tmp_path / "operations" / "recovery_protocol.json"
    protocol_path.write_text(
        json.dumps(
            {
                "method_revision": "method-revision",
                "cohort_id": "recovery-v1",
                "cell_ids": [cell_id, retained_id],
            }
        ),
        encoding="utf-8",
    )
    catalog = build_matrix_checkpoint_catalog(
        summary_path,
        tmp_path,
        compute_hashes=True,
        checkpoint_stage_root=tmp_path / "staged",
        selected_cell_ids=[cell_id, retained_id],
        recovery_receipt=receipt_path,
        recovery_protocol=protocol_path,
    )
    return cell_id, retained_id, catalog


def test_checkpoint_catalog_binds_recovered_checkpoint_and_receipt(tmp_path):
    cell_id, retained_id, catalog = _recovery_catalog(tmp_path)

    evidence = validate_selected_checkpoint_catalog(
        catalog,
        [cell_id, retained_id],
        expected_source_revision="method-revision",
        require_hashes=True,
    )

    assert catalog["checkpoints"][cell_id]["numerical_recovery"][
        "profile"
    ] == "exact"
    assert catalog["numerical_recovery_receipt"]["sha256"]
    assert catalog["checkpoints"][retained_id]["numerical_recovery"] is None
    assert catalog["checkpoints"][retained_id][
        "numerical_recovery_cohort"
    ]["profile"] == "first-pass"
    assert evidence["recovered_checkpoint_count"] == 1
    assert evidence["recovery_cohort_checkpoint_count"] == 2
    assert evidence["first_pass_recovery_cohort_checkpoint_count"] == 1

    catalog["checkpoints"][retained_id]["numerical_recovery_cohort"][
        "profile"
    ] = "exact"
    with pytest.raises(ValueError, match="cohort provenance drifted"):
        validate_selected_checkpoint_catalog(
            catalog,
            [cell_id, retained_id],
            expected_source_revision="method-revision",
            require_hashes=True,
        )


def test_checkpoint_catalog_requires_and_validates_recovery_receipt(tmp_path):
    cell_id, retained_id, catalog = _recovery_catalog(tmp_path)
    catalog["checkpoints"][cell_id]["numerical_recovery"][
        "profile"
    ] = "fallback-v1"

    with pytest.raises(ValueError, match="profile drifted"):
        validate_selected_checkpoint_catalog(
            catalog,
            [cell_id, retained_id],
            expected_source_revision="method-revision",
            require_hashes=True,
        )

    summary_path = tmp_path / "operations" / "composed_a10g.json"
    with pytest.raises(ValueError, match="requires its receipt"):
        build_matrix_checkpoint_catalog(
            summary_path,
            tmp_path,
            compute_hashes=True,
            selected_cell_ids=[cell_id],
        )

    receipt_path = tmp_path / "operations" / "composition_receipt.json"
    with pytest.raises(ValueError, match="requires its recovery protocol"):
        build_matrix_checkpoint_catalog(
            summary_path,
            tmp_path,
            compute_hashes=True,
            selected_cell_ids=[cell_id],
            recovery_receipt=receipt_path,
        )
