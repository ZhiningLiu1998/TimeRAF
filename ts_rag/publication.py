import hashlib
import json
import time
from collections import Counter
from pathlib import Path, PurePosixPath

from scripts.compose_timefuse_numerical_recovery import (
    compose_numerical_recovery,
    validate_composition_provenance,
)
from scripts.greenland_common import (
    METHOD_REVISION,
    NUMERICAL_RECOVERY_COHORT_ID,
    NUMERICAL_RECOVERY_PROTOCOL_SHA256,
    REPLAY_CELLS,
    validate_gpu_topology_evidence,
    validate_job_spec,
    validate_numerical_recovery_protocol,
    validate_replay_manifest,
)
from ts_rag.appendix_matrix import (
    APPENDIX_EXECUTION_TOPOLOGY,
    APPENDIX_EXPECTED_EVALUABLE,
    APPENDIX_EXPECTED_TOTAL,
    APPENDIX_METHOD_REVISION,
    APPENDIX_SELECTOR_POLICY,
    APPENDIX_SELECTOR_PROTOCOL_SHA256,
    _publication_gate as _appendix_publication_gate,
    load_bundle_catalog,
    validate_catalog_scope,
)
from ts_rag.appendix_rag import load_appendix_manifest
from ts_rag.matrix import (
    PUBLICATION_EXPECTED_CELLS,
    _publication_gate as _matrix_publication_gate,
    build_confirmation_summary,
    load_manifest,
)


PRIMARY_MANIFEST_SHA256 = (
    "45830d23f3b017c15d3680c88f837f84"
    "a9441eef9a6ceaaa5370c14872660c2a"
)
CONFIRMATION_PROTOCOL_SHA256 = (
    "62f6eaebe3da3993b13c414bebfee6786"
    "5afd4345cce3251ed5ee3eeb75e6f61"
)
APPENDIX_MANIFEST_SHA256 = (
    "877a672feb3555cabfff6e1157e0b245c"
    "7f18397b740cd29bb99da7869e58d06"
)
CONFIRMATION_EXPECTED_CELLS = 462
DEVELOPMENT_EXPECTED_CELLS = 123


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path, description):
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"{description} is missing or invalid: {path}"
        ) from error
    if not isinstance(payload, dict):
        raise ValueError(f"{description} must be a JSON object")
    return payload


def _revision_matches(actual, expected):
    return bool(
        actual
        and expected
        and (
            str(actual).startswith(str(expected))
            or str(expected).startswith(str(actual))
        )
    )


def _safe_relative(value, description):
    if not isinstance(value, str) or "\\" in value:
        raise ValueError(f"{description} must be a safe relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError(f"{description} must be a safe relative path")
    return Path(*path.parts)


def _resolve_under(root, value, description, require_file=True):
    root = Path(root).resolve()
    path = Path(value)
    path = path.resolve() if path.is_absolute() else (root / path).resolve()
    if path == root or root not in path.parents:
        raise ValueError(f"{description} must live below the project root")
    if require_file and (
        not path.is_file()
        or path.is_symlink()
    ):
        raise ValueError(f"{description} is not a regular file: {path}")
    return path


class _FileVerifier:
    def __init__(self, project_root):
        self.project_root = Path(project_root).resolve()
        self._hashes = {}

    def path(self, value, description):
        return _resolve_under(self.project_root, value, description)

    def sha256(self, path):
        path = Path(path).resolve()
        cached = self._hashes.get(path)
        stat = path.stat()
        identity = (stat.st_size, stat.st_mtime_ns)
        if cached is None or cached[0] != identity:
            cached = (identity, _sha256_file(path))
            self._hashes[path] = cached
        return cached[1]

    def verify(self, value, description, sha256=None, size_bytes=None):
        path = self.path(value, description)
        if size_bytes is not None and path.stat().st_size != size_bytes:
            raise ValueError(f"{description} size mismatch")
        actual = self.sha256(path)
        if sha256 is not None and actual != sha256:
            raise ValueError(f"{description} SHA-256 mismatch")
        return path

    def evidence(self, path):
        path = Path(path).resolve()
        return {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": self.sha256(path),
        }


def _assert_file_hash(verifier, path, expected, description):
    path = verifier.verify(path, description)
    actual = verifier.sha256(path)
    if actual != expected:
        raise ValueError(
            f"{description} drifted: expected {expected}, found {actual}"
        )
    return path


def _validate_state_identity(manifest, states, model_field):
    if not isinstance(states, list):
        raise ValueError("Summary requires a cell_states array")
    manifest_by_id = {row.get("id"): row for row in manifest}
    states_by_id = {row.get("cell_id"): row for row in states}
    if (
        len(manifest_by_id) != len(manifest)
        or None in manifest_by_id
        or len(states_by_id) != len(states)
        or set(states_by_id) != set(manifest_by_id)
    ):
        raise ValueError("Summary cell IDs do not exactly match its manifest")
    identity_fields = (
        "task_family",
        "dataset",
        model_field,
        "pred_len",
    )
    for cell_id, cell in manifest_by_id.items():
        state = states_by_id[cell_id]
        for field in identity_fields:
            if state.get(field) != cell.get(field):
                raise ValueError(
                    f"Summary cell identity drifted for {cell_id}: {field}"
                )
    return states_by_id


def _primary_counts(states):
    counts = Counter(row.get("state") for row in states)
    completed = [row for row in states if row.get("state") == "completed"]
    return {
        "expected": len(states),
        "completed": len(completed),
        "improved": sum(
            bool(row.get("all_test_metrics_improve")) for row in completed
        ),
        "not_improved": sum(
            not bool(row.get("all_test_metrics_improve"))
            for row in completed
        ),
        "failed": counts["failed"],
        "running": counts["running"],
        "pending": counts["pending"],
        "incomplete": counts["incomplete"],
    }


def _validate_primary(manifest, summary, method_revision):
    if len(manifest) != PUBLICATION_EXPECTED_CELLS:
        raise ValueError("Primary manifest must contain exactly 585 cells")
    if not _revision_matches(summary.get("source_revision"), method_revision):
        raise ValueError("Primary summary method revision drifted")
    if (
        summary.get("smoke") is not False
        or summary.get("seed") != 2021
        or summary.get("export_only") is not False
    ):
        raise ValueError("Primary summary does not describe the frozen run")
    states = summary.get("cell_states")
    _validate_state_identity(manifest, states, "model")
    actual_counts = _primary_counts(states)
    if summary.get("counts") != actual_counts:
        raise ValueError("Primary summary counts contradict its cell states")
    if summary.get("all_completed") is not (
        actual_counts["completed"] == PUBLICATION_EXPECTED_CELLS
    ):
        raise ValueError("Primary all_completed flag contradicts cell states")
    recomputed = _matrix_publication_gate(
        manifest,
        states,
        required_cells=PUBLICATION_EXPECTED_CELLS,
    )
    if summary.get("publication_gate") != recomputed:
        raise ValueError("Primary publication gate does not recompute exactly")
    return recomputed["development_gate_passed"]


def _verify_recorded_json(verifier, record, description):
    if not isinstance(record, dict):
        raise ValueError(f"{description} evidence is missing")
    path = verifier.verify(
        record.get("path"),
        description,
        sha256=record.get("sha256"),
        size_bytes=record.get("size_bytes"),
    )
    return path, _load_json(path, description)


def _validate_recovery_composition(
    verifier,
    manifest,
    primary_summary_path,
    primary_summary,
    receipt_path,
    protocol_path,
    method_revision,
):
    marker = primary_summary.get("numerical_recovery")
    if marker is None:
        if receipt_path is not None:
            raise ValueError(
                "A recovery receipt was supplied for a non-recovery summary"
            )
        return {"required": False, "verified": True}
    if receipt_path is None:
        raise ValueError(
            "Composed numerical-recovery primary requires its receipt"
        )
    if protocol_path is None:
        raise ValueError(
            "Composed numerical-recovery primary requires its protocol"
        )

    receipt_path = verifier.verify(
        receipt_path,
        "Numerical recovery composition receipt",
    )
    receipt = _load_json(
        receipt_path,
        "Numerical recovery composition receipt",
    )
    protocol_path = _assert_file_hash(
        verifier,
        protocol_path,
        NUMERICAL_RECOVERY_PROTOCOL_SHA256,
        "Numerical recovery protocol",
    )
    validate_numerical_recovery_protocol(protocol_path)
    protocol = _load_json(protocol_path, "Numerical recovery protocol")
    if (
        receipt.get("schema_version") != 1
        or receipt.get("method_revision") != method_revision
        or receipt.get("cohort_id") != NUMERICAL_RECOVERY_COHORT_ID
        or receipt.get("protocol_sha256")
        != NUMERICAL_RECOVERY_PROTOCOL_SHA256
        or receipt.get("selection_uses_metric_quality") is not False
    ):
        raise ValueError("Numerical recovery composition identity drifted")

    input_names = (
        "a10g_primary",
        "a100_primary",
        "a10g_exact",
        "a100_exact",
        "a10g_fallback",
        "a100_fallback",
    )
    input_records = receipt.get("inputs")
    if not isinstance(input_records, dict) or set(input_records) != set(
        input_names
    ):
        raise ValueError("Numerical recovery composition inputs drifted")
    input_payloads = {}
    input_paths = {}
    for name in input_names:
        input_paths[name], input_payloads[name] = _verify_recorded_json(
            verifier,
            input_records[name],
            f"Numerical recovery input {name}",
        )

    provenance = receipt.get("provenance")
    provenance_records = (
        provenance.get("inputs") if isinstance(provenance, dict) else None
    )
    provenance_names = (
        "a10g_exact_metadata",
        "a10g_fallback_metadata",
        "a100_exact_receipt",
        "a100_fallback_receipt",
    )
    if (
        not isinstance(provenance_records, dict)
        or set(provenance_records) != set(provenance_names)
    ):
        raise ValueError("Numerical recovery provenance inputs drifted")
    provenance_payloads = {}
    provenance_paths = {}
    for name in provenance_names:
        (
            provenance_paths[name],
            provenance_payloads[name],
        ) = _verify_recorded_json(
            verifier,
            provenance_records[name],
            f"Numerical recovery provenance {name}",
        )
    rebuilt_provenance = validate_composition_provenance(
        **provenance_payloads,
        a10g_exact_summary=input_paths["a10g_exact"],
        a10g_fallback_summary=input_paths["a10g_fallback"],
        a100_exact_summary=input_paths["a100_exact"],
        a100_fallback_summary=input_paths["a100_fallback"],
    )
    recorded_provenance = {
        key: value for key, value in provenance.items() if key != "inputs"
    }
    if rebuilt_provenance != recorded_provenance:
        raise ValueError("Numerical recovery provenance does not recompute")

    rebuilt = compose_numerical_recovery(
        manifest,
        protocol,
        **input_payloads,
    )
    output_records = receipt.get("outputs")
    if (
        not isinstance(output_records, dict)
        or set(output_records) != {"a10g", "a100"}
    ):
        raise ValueError("Numerical recovery composition outputs drifted")
    output_payloads = {}
    output_paths = {}
    for hardware in ("a10g", "a100"):
        output_paths[hardware], output_payloads[hardware] = (
            _verify_recorded_json(
                verifier,
                output_records[hardware],
                f"Numerical recovery output {hardware}",
            )
        )
        expected = rebuilt["composed"][hardware]
        expected["generated_unix"] = output_payloads[hardware].get(
            "generated_unix"
        )
        if expected != output_payloads[hardware]:
            raise ValueError(
                f"Numerical recovery {hardware} output does not recompute"
            )
    if output_paths["a10g"].resolve() != Path(primary_summary_path).resolve():
        raise ValueError(
            "Publication primary is not the composed A10G recovery output"
        )

    rebuilt_receipt = {
        key: value for key, value in rebuilt.items() if key != "composed"
    }
    for key, value in rebuilt_receipt.items():
        if receipt.get(key) != value:
            raise ValueError(
                f"Numerical recovery receipt field does not recompute: {key}"
            )
    if marker != output_payloads["a10g"].get("numerical_recovery"):
        raise ValueError(
            "Primary numerical recovery marker contradicts its output"
        )
    return {
        "required": True,
        "verified": True,
        "receipt": verifier.evidence(receipt_path),
        "protocol": verifier.evidence(protocol_path),
        "affected_cells": len(rebuilt["affected_cell_ids"]),
        "exact_selected": rebuilt["exact_selected_count"],
        "fallback_selected": rebuilt["fallback_selected_count"],
        "a10g_recovery_revision": rebuilt_provenance["a10g"][
            "recovery_revision"
        ],
        "a100_recovery_revision": rebuilt_provenance["a100"][
            "recovery_revision"
        ],
    }


def _confirmation_evidence(summary):
    fields = (
        "schema_version",
        "method_source_revision",
        "selection",
        "scope",
        "development_counts",
        "confirmatory_groups",
        "confirmatory_gate",
        "confirmatory_cell_ids",
    )
    return {field: summary.get(field) for field in fields}


def _validate_confirmation(
    manifest,
    primary_summary,
    protocol,
    summary,
    method_revision,
):
    if protocol.get("schema_version") != 1:
        raise ValueError("Confirmation protocol schema drifted")
    if not _revision_matches(
        protocol.get("method_source_revision"),
        method_revision,
    ):
        raise ValueError("Confirmation protocol method revision drifted")
    if (
        protocol.get("full_matrix_cell_count")
        != PUBLICATION_EXPECTED_CELLS
        or protocol.get("development_cell_count")
        != DEVELOPMENT_EXPECTED_CELLS
        or protocol.get("confirmatory_cell_count")
        != CONFIRMATION_EXPECTED_CELLS
    ):
        raise ValueError("Confirmation protocol scope drifted")
    rebuilt = build_confirmation_summary(
        manifest,
        primary_summary,
        protocol,
    )
    if _confirmation_evidence(summary) != _confirmation_evidence(rebuilt):
        raise ValueError(
            "Confirmation summary does not rebuild from primary evidence"
        )
    return rebuilt["confirmatory_gate"]["confirmatory_gate_passed"]


def _validate_topology(topology):
    validate_gpu_topology_evidence(topology)


def _output_object_path(value, description):
    path = _safe_relative(value, description)
    if len(path.parts) < 2 or path.parts[0] != "outputs":
        raise ValueError(f"{description} must start with outputs/")
    return Path(*path.parts[1:])


def _validate_replay_catalog(
    verifier,
    catalog,
    catalog_path,
    manifest,
    source_revision,
    destination,
):
    manifest_by_id = {row["id"]: row for row in manifest}
    bundles = catalog.get("bundles")
    if (
        catalog.get("schema_version") != 1
        or catalog.get("hashes_verified") is not True
        or catalog.get("expected_cells") != REPLAY_CELLS
        or catalog.get("cataloged_cells") != REPLAY_CELLS
        or catalog.get("usable_count") != REPLAY_CELLS
        or catalog.get("missing_count") != 0
        or not isinstance(bundles, dict)
        or set(bundles) != set(manifest_by_id)
        or not _revision_matches(
            catalog.get("source_revision"),
            source_revision,
        )
        or Path(catalog.get("project_root", "")).resolve()
        != verifier.project_root
    ):
        raise ValueError("Greenland replay bundle catalog is inconsistent")
    for cell_id, cell in manifest_by_id.items():
        record = bundles[cell_id]
        if (
            record.get("usable") is not True
            or record.get("reason") is not None
            or not _revision_matches(
                record.get("source_revision"),
                source_revision,
            )
        ):
            raise ValueError(f"Replay bundle is unusable for {cell_id}")
        for field in ("task_family", "dataset", "model", "pred_len"):
            if record.get(field) != cell.get(field):
                raise ValueError(
                    f"Replay bundle identity drifted for {cell_id}: {field}"
                )
        bundle_path = verifier.verify(
            record.get("path"),
            f"Replay bundle for {cell_id}",
            sha256=record.get("sha256"),
            size_bytes=record.get("size_bytes"),
        )
        if destination not in bundle_path.parents:
            raise ValueError(
                f"Replay bundle escaped the imported run for {cell_id}"
            )
    return {
        "catalog": verifier.evidence(catalog_path),
        "verified_bundles": len(bundles),
    }


def _validate_replay(
    verifier,
    manifest_path,
    receipt_path,
    method_revision,
):
    validate_replay_manifest(manifest_path)
    manifest = load_manifest(manifest_path)
    receipt = _load_json(receipt_path, "Greenland fetch receipt")
    if receipt.get("schema_version") != 1:
        raise ValueError("Greenland fetch receipt schema drifted")
    if Path(receipt.get("project_root", "")).resolve() != verifier.project_root:
        raise ValueError("Greenland receipt project root drifted")
    destination = _resolve_under(
        verifier.project_root,
        receipt.get("destination"),
        "Greenland destination",
        require_file=False,
    )
    if not destination.is_dir() or destination.is_symlink():
        raise ValueError("Greenland destination is not a regular directory")

    objects = receipt.get("objects")
    if not isinstance(objects, dict) or not objects:
        raise ValueError("Greenland receipt has no verified objects")
    total_size = 0
    for relative, record in objects.items():
        relative_path = _safe_relative(
            relative,
            "Greenland object key",
        )
        expected_path = (destination / relative_path).resolve()
        if expected_path != Path(record.get("path", "")).resolve():
            raise ValueError(f"Greenland object path drifted: {relative}")
        verifier.verify(
            expected_path,
            f"Greenland object {relative}",
            sha256=record.get("sha256"),
            size_bytes=record.get("size_bytes"),
        )
        total_size += record["size_bytes"]
    if (
        receipt.get("object_count") != len(objects)
        or receipt.get("total_size_bytes") != total_size
        or receipt.get("downloaded_count", 0)
        + receipt.get("reused_count", 0)
        != len(objects)
    ):
        raise ValueError("Greenland receipt object counts are inconsistent")

    def object_json(relative, description):
        record = objects.get(relative)
        if record is None:
            raise ValueError(f"{description} is absent from the receipt")
        return _load_json(record["path"], description)

    spec = validate_job_spec(
        object_json(
            "evidence/accepted_job_spec.json",
            "Accepted Greenland job spec",
        )
    )
    if spec.get("method_revision") != method_revision:
        raise ValueError("Greenland method revision drifted")
    summary_relative = _output_object_path(
        spec["matrix"]["summary"],
        "Greenland matrix summary",
    ).as_posix()
    final_status = object_json(
        "final_status.json",
        "Greenland final status",
    )
    topology = object_json(
        "evidence/topology_summary.json",
        "Greenland topology evidence",
    )
    replay_summary = object_json(
        summary_relative,
        "Greenland replay summary",
    )
    _validate_topology(topology)
    if receipt.get("final_status") != final_status:
        raise ValueError("Greenland receipt final status drifted")
    if (
        final_status.get("run_id") != receipt.get("run_id")
        or final_status.get("status") != "succeeded"
        or final_status.get("exit_code") != 0
        or final_status.get("topology_gate_passed") is not True
        or final_status.get("method_revision") != method_revision
        or final_status.get("source_revision") != spec["source_revision"]
        or final_status.get("launcher_revision") != spec["launcher_revision"]
        or receipt.get("method_revision") != method_revision
        or receipt.get("source_revision") != spec["source_revision"]
        or receipt.get("launcher_revision") != spec["launcher_revision"]
        or receipt.get("topology_gate_passed") is not True
    ):
        raise ValueError("Greenland run identity or final status drifted")
    expected_counts = {
        "expected": REPLAY_CELLS,
        "completed": REPLAY_CELLS,
        "failed": 0,
        "pending": 0,
        "running": 0,
        "incomplete": 0,
    }
    actual_counts = replay_summary.get("counts", {})
    if (
        replay_summary.get("source_revision") != spec["source_revision"]
        or replay_summary.get("export_only") is not True
        or replay_summary.get("all_completed") is not True
        or any(
            actual_counts.get(key) != value
            for key, value in expected_counts.items()
        )
        or any(
            receipt.get("replay_counts", {}).get(key) != value
            for key, value in expected_counts.items()
        )
    ):
        raise ValueError("Greenland replay summary is incomplete")

    catalog_path = verifier.verify(
        receipt.get("replay_bundle_catalog"),
        "Greenland replay bundle catalog",
        sha256=receipt.get("replay_bundle_catalog_sha256"),
    )
    catalog = _load_json(
        catalog_path,
        "Greenland replay bundle catalog",
    )
    catalog_evidence = _validate_replay_catalog(
        verifier,
        catalog,
        catalog_path,
        manifest,
        spec["source_revision"],
        destination,
    )
    return {
        "run_id": receipt["run_id"],
        "method_revision": method_revision,
        "source_revision": spec["source_revision"],
        "launcher_revision": spec["launcher_revision"],
        "object_count": len(objects),
        "topology_gate_passed": True,
        **catalog_evidence,
    }


def _appendix_counts(states):
    counts = Counter(row.get("state") for row in states)
    evaluable = [row for row in states if row.get("paper_evaluable")]
    completed = [row for row in evaluable if row.get("state") == "completed"]
    return {
        "completed": len(completed),
        "paper_oot": counts["paper_oot"],
        "failed": counts["failed"],
        "running": counts["running"],
        "pending": counts["pending"],
        "incomplete": counts["incomplete"],
        "reported": len(states),
        "evaluable": len(evaluable),
        "improved": sum(
            bool(row.get("all_test_metrics_improve")) for row in completed
        ),
        "not_improved": sum(
            not bool(row.get("all_test_metrics_improve"))
            for row in completed
        ),
    }


def _validate_appendix(
    verifier,
    manifest,
    catalog_path,
    summary,
    method_revision,
    launcher_revision,
):
    if (
        len(manifest) != APPENDIX_EXPECTED_TOTAL
        or sum(
            row.get("paper_status", {}).get("evaluable") is True
            for row in manifest
        )
        != APPENDIX_EXPECTED_EVALUABLE
    ):
        raise ValueError("Appendix manifest scope drifted")
    if summary.get("method_revision") != method_revision:
        raise ValueError("Appendix method revision drifted")
    if method_revision != APPENDIX_METHOD_REVISION:
        raise ValueError("Publication method revision is not the frozen method")
    if summary.get("launcher_revision") != launcher_revision:
        raise ValueError("Appendix launcher revision drifted")
    if (
        summary.get("selector_policy") != APPENDIX_SELECTOR_POLICY
        or summary.get("selector_protocol_sha256")
        != APPENDIX_SELECTOR_PROTOCOL_SHA256
    ):
        raise ValueError("Appendix selector protocol drifted")
    if summary.get("execution_topology") != APPENDIX_EXECUTION_TOPOLOGY:
        raise ValueError("Appendix execution topology drifted")
    states = summary.get("cell_states")
    _validate_state_identity(manifest, states, "baseline")
    if any(
        row.get("method_revision") != method_revision for row in states
    ):
        raise ValueError("Appendix cell method revision drifted")
    actual_counts = _appendix_counts(states)
    if summary.get("counts") != actual_counts:
        raise ValueError("Appendix summary counts contradict its cell states")
    if summary.get("all_evaluable_completed") is not (
        actual_counts["completed"] == APPENDIX_EXPECTED_EVALUABLE
    ):
        raise ValueError(
            "Appendix all_evaluable_completed contradicts cell states"
        )
    recomputed = _appendix_publication_gate(
        manifest,
        states,
        expected_total=APPENDIX_EXPECTED_TOTAL,
        expected_evaluable=APPENDIX_EXPECTED_EVALUABLE,
    )
    if summary.get("publication_gate") != recomputed:
        raise ValueError("Appendix publication gate does not recompute exactly")

    catalog = load_bundle_catalog(
        catalog_path,
        project_root=verifier.project_root,
    )
    validate_catalog_scope(manifest, catalog, require_files=True)
    if catalog.get("source_revision") != summary.get("source_revision"):
        raise ValueError(
            "Appendix result and bundle source revisions do not match"
        )
    return {
        "gate_passed": recomputed["development_gate_passed"],
        "source_revision": summary.get("source_revision"),
        "catalog": verifier.evidence(catalog_path),
        "verified_bundles": len(catalog["bundles"]),
    }


def audit_timefuse_publication(
    *,
    project_root,
    primary_manifest,
    primary_summary,
    confirmation_protocol,
    confirmation_summary,
    replay_manifest,
    replay_receipt,
    appendix_manifest,
    appendix_catalog,
    appendix_summary,
    appendix_launcher_revision,
    recovery_receipt=None,
    recovery_protocol=None,
    method_revision=METHOD_REVISION,
    appendix_method_revision=APPENDIX_METHOD_REVISION,
):
    project_root = Path(project_root).resolve()
    verifier = _FileVerifier(project_root)
    primary_manifest = _assert_file_hash(
        verifier,
        primary_manifest,
        PRIMARY_MANIFEST_SHA256,
        "Primary manifest",
    )
    confirmation_protocol = _assert_file_hash(
        verifier,
        confirmation_protocol,
        CONFIRMATION_PROTOCOL_SHA256,
        "Confirmation protocol",
    )
    appendix_manifest = _assert_file_hash(
        verifier,
        appendix_manifest,
        APPENDIX_MANIFEST_SHA256,
        "Appendix manifest",
    )
    replay_manifest = verifier.verify(
        replay_manifest,
        "Replay manifest",
    )
    primary_summary = verifier.verify(
        primary_summary,
        "Primary summary",
    )
    confirmation_summary = verifier.verify(
        confirmation_summary,
        "Confirmation summary",
    )
    replay_receipt = verifier.verify(
        replay_receipt,
        "Greenland replay receipt",
    )
    appendix_catalog = verifier.verify(
        appendix_catalog,
        "Appendix bundle catalog",
    )
    appendix_summary = verifier.verify(
        appendix_summary,
        "Appendix summary",
    )

    primary_rows = load_manifest(primary_manifest)
    primary_payload = _load_json(primary_summary, "Primary summary")
    primary_passed = _validate_primary(
        primary_rows,
        primary_payload,
        method_revision,
    )
    recovery = _validate_recovery_composition(
        verifier,
        primary_rows,
        primary_summary,
        primary_payload,
        recovery_receipt,
        recovery_protocol,
        method_revision,
    )
    protocol_payload = _load_json(
        confirmation_protocol,
        "Confirmation protocol",
    )
    confirmation_payload = _load_json(
        confirmation_summary,
        "Confirmation summary",
    )
    confirmation_passed = _validate_confirmation(
        primary_rows,
        primary_payload,
        protocol_payload,
        confirmation_payload,
        method_revision,
    )
    replay = _validate_replay(
        verifier,
        replay_manifest,
        replay_receipt,
        method_revision,
    )
    appendix_rows = load_appendix_manifest(appendix_manifest)
    appendix_payload = _load_json(appendix_summary, "Appendix summary")
    appendix = _validate_appendix(
        verifier,
        appendix_rows,
        appendix_catalog,
        appendix_payload,
        appendix_method_revision,
        appendix_launcher_revision,
    )

    criteria = {
        "primary_585_gate_passed": primary_passed,
        "numerical_recovery_composition_verified": recovery["verified"],
        "frozen_confirmation_462_gate_passed": confirmation_passed,
        "greenland_replay_208_complete": True,
        "greenland_eight_a100_topology_passed": (
            replay["topology_gate_passed"]
        ),
        "appendix_94_gate_passed": appendix["gate_passed"],
    }
    evidence_paths = {
        "primary_manifest": primary_manifest,
        "primary_summary": primary_summary,
        "confirmation_protocol": confirmation_protocol,
        "confirmation_summary": confirmation_summary,
        "replay_manifest": replay_manifest,
        "replay_receipt": replay_receipt,
        "appendix_manifest": appendix_manifest,
        "appendix_catalog": appendix_catalog,
        "appendix_summary": appendix_summary,
    }
    if recovery_receipt is not None:
        evidence_paths["recovery_receipt"] = Path(recovery_receipt)
    return {
        "schema_version": 1,
        "generated_unix": time.time(),
        "project_root": str(project_root),
        "method_revision": method_revision,
        "appendix_method_revision": appendix_method_revision,
        "appendix_launcher_revision": appendix_launcher_revision,
        "scope": {
            "primary_cells": PUBLICATION_EXPECTED_CELLS,
            "development_cells": DEVELOPMENT_EXPECTED_CELLS,
            "confirmatory_cells": CONFIRMATION_EXPECTED_CELLS,
            "replay_cells": REPLAY_CELLS,
            "appendix_reported_cells": APPENDIX_EXPECTED_TOTAL,
            "appendix_evaluable_cells": APPENDIX_EXPECTED_EVALUABLE,
        },
        "criteria": criteria,
        "publication_ready": all(criteria.values()),
        "replay": replay,
        "appendix": appendix,
        "numerical_recovery": recovery,
        "evidence": {
            name: verifier.evidence(path)
            for name, path in evidence_paths.items()
        },
    }
