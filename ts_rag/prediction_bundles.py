import json
import time
from pathlib import Path

from ts_rag.appendix_rag import sha256_file


def _load_json(path):
    try:
        with Path(path).open("r", encoding="utf-8") as source:
            return json.load(source)
    except (OSError, json.JSONDecodeError):
        return None


def _resolve(project_root, path):
    path = Path(path)
    return path if path.is_absolute() else Path(project_root) / path


def _catalog_path(project_root, path):
    path = Path(path)
    try:
        return str(path.resolve().relative_to(Path(project_root).resolve()))
    except ValueError:
        return str(path.resolve())


def build_prediction_bundle_catalog(
    manifest,
    output_root,
    project_root,
    source_revision=None,
    compute_hashes=False,
):
    manifest = list(manifest)
    known = {cell["id"]: cell for cell in manifest}
    attempts = {}
    for status_path in Path(output_root).glob("**/status.json"):
        status = _load_json(status_path)
        if (
            not status
            or status.get("status") != "completed"
            or status.get("cell_id") not in known
            or (
                source_revision is not None
                and status.get("source_revision") != source_revision
            )
        ):
            continue
        result_path = status_path.with_name("result.json")
        result = _load_json(result_path)
        if not result or not result.get("export_only"):
            continue
        current = attempts.get(status["cell_id"])
        if current is None or float(status.get("started_unix", 0.0)) > float(
            current["status"].get("started_unix", 0.0)
        ):
            attempts[status["cell_id"]] = {
                "status": status,
                "result": result,
                "result_path": result_path,
            }

    bundles = {}
    for cell_id, attempt in sorted(attempts.items()):
        result = attempt["result"]
        recorded_bundle = result.get("prediction_bundle")
        bundle_path = _resolve(
            project_root,
            recorded_bundle
            or attempt["result_path"].with_name("prediction_bundle.npz"),
        )
        relocated = False
        relocated_bundle = attempt["result_path"].with_name(
            "prediction_bundle.npz"
        )
        if (
            recorded_bundle
            and Path(recorded_bundle).is_absolute()
            and result.get("prediction_bundle_sha256")
            and not bundle_path.exists()
            and relocated_bundle.is_file()
        ):
            bundle_path = relocated_bundle
            relocated = True
        usable = bundle_path.is_file()
        digest = result.get("prediction_bundle_sha256")
        if compute_hashes and usable:
            actual = sha256_file(bundle_path)
            if digest is not None and digest != actual:
                raise ValueError(
                    f"Prediction bundle hash mismatch for {cell_id}"
                )
            digest = actual
        cell = known[cell_id]
        bundles[cell_id] = {
            "usable": usable,
            "reason": None if usable else "prediction_bundle_missing",
            "path": (
                _catalog_path(project_root, bundle_path)
                if usable
                else None
            ),
            "recorded_path": recorded_bundle,
            "relocated": relocated,
            "sha256": digest if usable else None,
            "size_bytes": bundle_path.stat().st_size if usable else None,
            "result_path": _catalog_path(
                project_root,
                attempt["result_path"],
            ),
            "source_revision": attempt["status"].get("source_revision"),
            "task_family": cell["task_family"],
            "dataset": cell["dataset"],
            "model": cell["model"],
            "pred_len": cell["pred_len"],
        }
    usable = sum(record["usable"] for record in bundles.values())
    return {
        "schema_version": 1,
        "generated_unix": time.time(),
        "source_revision": source_revision,
        "output_root": str(output_root),
        "project_root": str(Path(project_root).resolve()),
        "expected_cells": len(manifest),
        "cataloged_cells": len(bundles),
        "usable_count": usable,
        "missing_count": len(manifest) - usable,
        "hashes_verified": bool(compute_hashes),
        "bundles": bundles,
    }
