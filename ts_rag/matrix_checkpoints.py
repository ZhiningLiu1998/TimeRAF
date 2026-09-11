import hashlib
import json
import os
import shutil
import time
from pathlib import Path, PurePosixPath


def _revision_matches(actual, expected):
    return bool(
        actual
        and expected
        and (actual.startswith(expected) or expected.startswith(actual))
    )


def _load_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value):
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _resolve(project_root, path):
    path = Path(path)
    return path if path.is_absolute() else project_root / path


def _catalog_path(path_root, path):
    path = Path(path)
    try:
        return str(path.resolve().relative_to(path_root.resolve()))
    except ValueError as error:
        raise ValueError(
            f"Checkpoint is outside its catalog root: {path}"
        ) from error


def _stage_relative_path(cell_id):
    path = PurePosixPath(str(cell_id))
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError(f"Unsafe checkpoint cell ID: {cell_id!r}")
    return Path(*path.parts) / "checkpoint.pth"


def _stage_checkpoint(source, stage_root, cell_id):
    source = Path(source)
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"Checkpoint is not a regular file: {source}")
    destination = Path(stage_root) / _stage_relative_path(cell_id)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.is_symlink() or not destination.is_file():
            raise ValueError(
                f"Staged checkpoint is not a regular file: {destination}"
            )
        source_digest = _sha256(source)
        if (
            destination.stat().st_size != source.stat().st_size
            or _sha256(destination) != source_digest
        ):
            raise ValueError(
                f"Staged checkpoint drifted for {cell_id}: {destination}"
            )
        return destination, source_digest, "reused"
    try:
        os.link(source, destination)
        mode = "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        mode = "copy"
    return destination, None, mode


def _recovery_evidence(summary, summary_path, project_root, receipt_path):
    marker = summary.get("numerical_recovery")
    recovered_states = {
        row.get("cell_id"): row.get("numerical_recovery")
        for row in summary.get("cell_states", [])
        if row.get("numerical_recovery") is not None
    }
    if marker is None:
        if recovered_states or receipt_path is not None:
            raise ValueError(
                "Recovery state or receipt requires a composed summary marker"
            )
        return None, None
    if summary_path is None:
        raise ValueError("A composed recovery summary must be supplied by path")
    if receipt_path is None:
        raise ValueError("A composed recovery summary requires its receipt")

    affected = marker.get("affected_cell_ids")
    profiles = marker.get("selected_profiles")
    if (
        not isinstance(affected, list)
        or len(set(affected)) != len(affected)
        or not isinstance(profiles, dict)
        or set(profiles) != set(affected)
        or any(
            profile not in {"exact", "fallback-v1"}
            for profile in profiles.values()
        )
        or marker.get("selection_uses_metric_quality") is not False
        or not isinstance(marker.get("cohort_id"), str)
        or not marker["cohort_id"]
    ):
        raise ValueError("Composed recovery summary marker drifted")
    states = {
        row.get("cell_id"): row for row in summary.get("cell_states", [])
    }
    if set(recovered_states) != set(affected):
        raise ValueError(
            "Recovered cell states do not match the affected cell set"
        )
    for cell_id in affected:
        state = states[cell_id]
        recovery = recovered_states[cell_id]
        artifact_dir = _resolve(project_root, state.get("artifact_dir"))
        selected_dir = _resolve(
            project_root,
            recovery.get("selected_artifact_dir"),
        )
        if (
            recovery.get("cohort_id") != marker["cohort_id"]
            or recovery.get("profile") != profiles[cell_id]
            or recovery.get("selection_uses_metric_quality") is not False
            or selected_dir.resolve() != artifact_dir.resolve()
            or recovery.get("first_pass_started_unix")
            != state.get("started_unix")
            or not isinstance(
                recovery.get("recovery_started_unix"),
                (int, float),
            )
        ):
            raise ValueError(
                f"Recovered checkpoint state drifted for {cell_id}"
            )

    receipt_path = _resolve(project_root, receipt_path).resolve()
    project_root = Path(project_root).resolve()
    if (
        project_root not in receipt_path.parents
        or receipt_path.is_symlink()
        or not receipt_path.is_file()
    ):
        raise ValueError("Recovery receipt must be a project regular file")
    receipt = _load_json(receipt_path)
    output = (
        receipt.get("outputs", {}).get("a10g")
        if isinstance(receipt, dict)
        else None
    )
    if (
        not isinstance(output, dict)
        or _resolve(project_root, output.get("path")).resolve()
        != summary_path.resolve()
        or output.get("sha256") != _sha256(summary_path)
        or receipt.get("cohort_id") != marker["cohort_id"]
        or receipt.get("affected_cell_ids") != affected
        or receipt.get("selected_profiles") != profiles
        or receipt.get("selection_uses_metric_quality") is not False
    ):
        raise ValueError("Recovery receipt does not bind the composed summary")
    return dict(marker), {
        "path": _catalog_path(project_root, receipt_path),
        "size_bytes": receipt_path.stat().st_size,
        "sha256": _sha256(receipt_path),
    }


def _recovery_cohort_evidence(
    recovery_marker,
    recovery_receipt,
    protocol_path,
    project_root,
    source_revision,
):
    if recovery_marker is None:
        if protocol_path is not None:
            raise ValueError(
                "Recovery protocol requires a composed recovery summary"
            )
        return None, {}
    if protocol_path is None:
        raise ValueError(
            "A composed recovery summary requires its recovery protocol"
        )

    project_root = Path(project_root).resolve()
    protocol_path = _resolve(project_root, protocol_path).resolve()
    if (
        protocol_path == project_root
        or project_root not in protocol_path.parents
        or protocol_path.is_symlink()
        or not protocol_path.is_file()
    ):
        raise ValueError("Recovery protocol must be a project regular file")
    protocol = _load_json(protocol_path)
    cell_ids = protocol.get("cell_ids") if isinstance(protocol, dict) else None
    affected = recovery_marker["affected_cell_ids"]
    selected_profiles = recovery_marker["selected_profiles"]
    if (
        not isinstance(protocol, dict)
        or protocol.get("cohort_id") != recovery_marker["cohort_id"]
        or not _revision_matches(
            protocol.get("method_revision"),
            source_revision,
        )
        or not isinstance(cell_ids, list)
        or not cell_ids
        or any(
            not isinstance(cell_id, str) or not cell_id
            for cell_id in cell_ids
        )
        or len(cell_ids) != len(set(cell_ids))
        or not set(affected).issubset(cell_ids)
    ):
        raise ValueError("Recovery protocol does not bind the composed cohort")

    profiles = {
        cell_id: selected_profiles.get(cell_id, "first-pass")
        for cell_id in cell_ids
    }
    protocol_record = {
        "path": _catalog_path(project_root, protocol_path),
        "size_bytes": protocol_path.stat().st_size,
        "sha256": _sha256(protocol_path),
    }
    return {
        "cohort_id": recovery_marker["cohort_id"],
        "cell_ids": cell_ids,
        "profiles": profiles,
        "selection_uses_metric_quality": False,
        "protocol": protocol_record,
        "composition_receipt_sha256": recovery_receipt["sha256"],
    }, profiles


def build_matrix_checkpoint_catalog(
    summary,
    project_root,
    compute_hashes=False,
    checkpoint_path_root=None,
    checkpoint_stage_root=None,
    selected_cell_ids=None,
    recovery_receipt=None,
    recovery_protocol=None,
):
    project_root = Path(project_root)
    if checkpoint_stage_root is not None:
        checkpoint_stage_root = _resolve(
            project_root, checkpoint_stage_root
        )
        checkpoint_path_root = (
            checkpoint_stage_root
            if checkpoint_path_root is None
            else _resolve(project_root, checkpoint_path_root)
        )
        resolved_stage_root = checkpoint_stage_root.resolve()
        resolved_path_root = checkpoint_path_root.resolve()
        if (
            resolved_stage_root != resolved_path_root
            and resolved_path_root not in resolved_stage_root.parents
        ):
            raise ValueError(
                "checkpoint_path_root must contain checkpoint_stage_root "
                "when staging"
            )
    checkpoint_path_root = (
        project_root
        if checkpoint_path_root is None
        else _resolve(project_root, checkpoint_path_root)
    )
    selected_cell_ids = (
        None if selected_cell_ids is None else set(selected_cell_ids)
    )
    summary_path = None
    if not isinstance(summary, dict):
        summary_path = _resolve(project_root, summary).resolve()
        summary = _load_json(summary_path)
    if not summary:
        raise ValueError("Matrix summary is missing or invalid")
    recovery_marker, recovery_receipt_evidence = _recovery_evidence(
        summary,
        summary_path,
        project_root,
        recovery_receipt,
    )
    recovery_cohort, recovery_cohort_profiles = _recovery_cohort_evidence(
        recovery_marker,
        recovery_receipt_evidence,
        recovery_protocol,
        project_root,
        summary.get("source_revision"),
    )

    checkpoints = {}
    for cell in summary.get("cell_states", []):
        if cell.get("state") != "completed":
            continue
        cell_id = cell["cell_id"]
        if (
            selected_cell_ids is not None
            and cell_id not in selected_cell_ids
        ):
            continue
        artifact_dir = _resolve(project_root, cell["artifact_dir"])
        result_path = artifact_dir / "result.json"
        status_path = artifact_dir / "status.json"
        result = _load_json(result_path)
        status = _load_json(status_path)
        checkpoint_value = (
            result.get("checkpoint", {}).get("checkpoint")
            if result
            else None
        )
        checkpoint_path = (
            _resolve(project_root, checkpoint_value)
            if checkpoint_value
            else None
        )
        usable = bool(checkpoint_path and checkpoint_path.is_file())
        staged_digest = None
        staging_mode = None
        if usable and checkpoint_stage_root is not None:
            checkpoint_path, staged_digest, staging_mode = _stage_checkpoint(
                checkpoint_path,
                checkpoint_stage_root,
                cell_id,
            )
        record = {
            "usable": usable,
            "reason": None if usable else "checkpoint_missing",
            "relative_path": (
                _catalog_path(checkpoint_path_root, checkpoint_path)
                if checkpoint_path
                else None
            ),
            "file_size": (
                checkpoint_path.stat().st_size if usable else None
            ),
            "artifact_dir": _catalog_path(project_root, artifact_dir),
            "result_path": _catalog_path(project_root, result_path),
            "source_revision": (
                status.get("source_revision") if status else None
            ),
            "checkpoint_mode": (
                result.get("checkpoint", {}).get("mode") if result else None
            ),
            "staging_mode": staging_mode,
            "numerical_recovery": cell.get("numerical_recovery"),
            "numerical_recovery_cohort": None,
        }
        cohort_profile = recovery_cohort_profiles.get(cell_id)
        if cohort_profile is not None:
            record["numerical_recovery_cohort"] = {
                "cohort_id": recovery_cohort["cohort_id"],
                "profile": cohort_profile,
                "affected": cell_id in recovery_marker["affected_cell_ids"],
                "selection_uses_metric_quality": False,
                "composition_receipt_sha256": recovery_receipt_evidence[
                    "sha256"
                ],
            }
        if compute_hashes and usable:
            record["sha256"] = staged_digest or _sha256(checkpoint_path)
        checkpoints[cell_id] = record

    usable_count = sum(
        record["usable"] for record in checkpoints.values()
    )
    missing_selected = (
        []
        if selected_cell_ids is None
        else sorted(selected_cell_ids - set(checkpoints))
    )
    return {
        "schema_version": 1,
        "generated_unix": time.time(),
        "source_revision": summary.get("source_revision"),
        "summary_generated_unix": summary.get("generated_unix"),
        "project_root": str(project_root.resolve()),
        "checkpoint_path_root": str(checkpoint_path_root.resolve()),
        "checkpoint_stage_root": (
            str(checkpoint_stage_root.resolve())
            if checkpoint_stage_root is not None
            else None
        ),
        "selected_count": (
            None
            if selected_cell_ids is None
            else len(selected_cell_ids)
        ),
        "missing_selected_count": len(missing_selected),
        "missing_selected": missing_selected,
        "checkpoint_count": len(checkpoints),
        "usable_count": usable_count,
        "missing_count": len(checkpoints) - usable_count,
        "hashes_included": bool(compute_hashes),
        "numerical_recovery": recovery_marker,
        "numerical_recovery_cohort": recovery_cohort,
        "numerical_recovery_receipt": recovery_receipt_evidence,
        "checkpoints": dict(sorted(checkpoints.items())),
    }


def validate_selected_checkpoint_catalog(
    catalog,
    selected_cell_ids,
    expected_source_revision,
    require_hashes=False,
):
    """Require an exact, usable, archive-safe selected checkpoint catalog."""
    selected_cell_ids = list(selected_cell_ids)
    selected = set(selected_cell_ids)
    if len(selected) != len(selected_cell_ids):
        raise ValueError("Selected checkpoint cell IDs must be unique")
    records = catalog.get("checkpoints")
    if not isinstance(records, dict):
        raise ValueError("Checkpoint catalog requires a checkpoints object")
    if set(records) != selected:
        raise ValueError(
            "Checkpoint catalog does not exactly match selected cells: "
            f"missing={len(selected - set(records))}, "
            f"extra={len(set(records) - selected)}"
        )
    if catalog.get("selected_count") != len(selected):
        raise ValueError("Checkpoint catalog selected_count is inconsistent")
    if (
        catalog.get("checkpoint_count") != len(selected)
        or catalog.get("usable_count") != len(selected)
        or catalog.get("missing_count") != 0
        or catalog.get("missing_selected_count") != 0
    ):
        raise ValueError("Every selected checkpoint must be usable")
    if not _revision_matches(
        catalog.get("source_revision"),
        expected_source_revision,
    ):
        raise ValueError("Checkpoint catalog source revision drifted")
    if require_hashes and catalog.get("hashes_included") is not True:
        raise ValueError("Selected checkpoint catalog must include hashes")

    recovery_marker = catalog.get("numerical_recovery")
    recovery_cohort = catalog.get("numerical_recovery_cohort")
    recovery_receipt = catalog.get("numerical_recovery_receipt")
    if recovery_marker is None:
        if (
            recovery_cohort is not None
            or recovery_receipt is not None
            or any(
                record.get("numerical_recovery") is not None
                or record.get("numerical_recovery_cohort") is not None
                for record in records.values()
            )
        ):
            raise ValueError("Checkpoint recovery provenance is inconsistent")
        recovery_profiles = {}
        recovery_cohort_profiles = {}
    else:
        affected = recovery_marker.get("affected_cell_ids")
        recovery_profiles = recovery_marker.get("selected_profiles")
        cohort_ids = (
            recovery_cohort.get("cell_ids")
            if isinstance(recovery_cohort, dict)
            else None
        )
        recovery_cohort_profiles = (
            recovery_cohort.get("profiles")
            if isinstance(recovery_cohort, dict)
            else None
        )
        protocol = (
            recovery_cohort.get("protocol")
            if isinstance(recovery_cohort, dict)
            else None
        )
        if (
            not isinstance(affected, list)
            or any(
                not isinstance(cell_id, str) or not cell_id
                for cell_id in affected
            )
            or len(affected) != len(set(affected))
            or not isinstance(recovery_profiles, dict)
            or set(recovery_profiles) != set(affected)
            or any(
                profile not in {"exact", "fallback-v1"}
                for profile in recovery_profiles.values()
            )
            or recovery_marker.get("selection_uses_metric_quality") is not False
            or not isinstance(recovery_receipt, dict)
            or not isinstance(recovery_receipt.get("path"), str)
            or not isinstance(recovery_receipt.get("size_bytes"), int)
            or recovery_receipt["size_bytes"] <= 0
            or not _is_sha256(recovery_receipt.get("sha256"))
            or not isinstance(cohort_ids, list)
            or not cohort_ids
            or any(
                not isinstance(cell_id, str) or not cell_id
                for cell_id in cohort_ids
            )
            or len(cohort_ids) != len(set(cohort_ids))
            or not isinstance(recovery_cohort_profiles, dict)
            or set(recovery_cohort_profiles) != set(cohort_ids)
            or not set(affected).issubset(cohort_ids)
            or recovery_cohort.get("cohort_id")
            != recovery_marker.get("cohort_id")
            or recovery_cohort.get("selection_uses_metric_quality") is not False
            or recovery_cohort.get("composition_receipt_sha256")
            != recovery_receipt.get("sha256")
            or not isinstance(protocol, dict)
            or not isinstance(protocol.get("path"), str)
            or not isinstance(protocol.get("size_bytes"), int)
            or protocol["size_bytes"] <= 0
            or not _is_sha256(protocol.get("sha256"))
            or any(
                recovery_cohort_profiles[cell_id]
                != recovery_profiles.get(cell_id, "first-pass")
                for cell_id in cohort_ids
            )
        ):
            raise ValueError("Checkpoint recovery provenance is incomplete")

    relative_paths = set()
    for cell_id, record in records.items():
        if record.get("usable") is not True:
            raise ValueError(f"Checkpoint is not usable: {cell_id}")
        if not _revision_matches(
            record.get("source_revision"),
            expected_source_revision,
        ):
            raise ValueError(
                f"Checkpoint source revision drifted: {cell_id}"
            )
        relative = record.get("relative_path")
        path = PurePosixPath(relative) if relative else None
        if (
            path is None
            or path.is_absolute()
            or not path.parts
            or ".." in path.parts
        ):
            raise ValueError(f"Unsafe checkpoint path for {cell_id}")
        if relative in relative_paths:
            raise ValueError(f"Duplicate checkpoint path: {relative}")
        relative_paths.add(relative)
        if not isinstance(record.get("file_size"), int) or (
            record["file_size"] <= 0
        ):
            raise ValueError(f"Invalid checkpoint size for {cell_id}")
        if require_hashes:
            digest = record.get("sha256")
            if not _is_sha256(digest):
                raise ValueError(f"Invalid checkpoint SHA-256 for {cell_id}")
        expected_profile = recovery_profiles.get(cell_id)
        recovery = record.get("numerical_recovery")
        cohort_profile = recovery_cohort_profiles.get(cell_id)
        cohort = record.get("numerical_recovery_cohort")
        if expected_profile is None:
            if recovery is not None:
                raise ValueError(
                    f"Unexpected checkpoint recovery state for {cell_id}"
                )
        elif (
            not isinstance(recovery, dict)
            or recovery.get("profile") != expected_profile
            or recovery.get("selection_uses_metric_quality") is not False
        ):
            raise ValueError(
                f"Checkpoint recovery profile drifted for {cell_id}"
            )
        if cohort_profile is None:
            if cohort is not None:
                raise ValueError(
                    f"Unexpected recovery cohort state for {cell_id}"
                )
        elif (
            not isinstance(cohort, dict)
            or cohort.get("cohort_id") != recovery_cohort["cohort_id"]
            or cohort.get("profile") != cohort_profile
            or cohort.get("affected") is not (expected_profile is not None)
            or cohort.get("selection_uses_metric_quality") is not False
            or cohort.get("composition_receipt_sha256")
            != recovery_receipt["sha256"]
        ):
            raise ValueError(
                f"Checkpoint recovery cohort provenance drifted for {cell_id}"
            )
    return {
        "selected_count": len(selected),
        "usable_count": len(records),
        "source_revision": expected_source_revision,
        "hashes_verified": bool(require_hashes),
        "unique_relative_paths": len(relative_paths),
        "recovered_checkpoint_count": sum(
            cell_id in recovery_profiles for cell_id in records
        ),
        "recovery_cohort_checkpoint_count": sum(
            cell_id in recovery_cohort_profiles for cell_id in records
        ),
        "first_pass_recovery_cohort_checkpoint_count": sum(
            recovery_cohort_profiles.get(cell_id) == "first-pass"
            for cell_id in records
        ),
    }
