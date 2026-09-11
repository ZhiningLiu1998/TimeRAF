"""Compose immutable primary and numerical-recovery matrix summaries."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from scripts.greenland_common import (
    METHOD_REVISION,
    NUMERICAL_RECOVERY_CELL_IDS,
    NUMERICAL_RECOVERY_CELL_IDS_SHA256,
    NUMERICAL_RECOVERY_COHORT_ID,
    NUMERICAL_RECOVERY_JOB,
    NUMERICAL_RECOVERY_PROFILES,
    NUMERICAL_RECOVERY_PROTOCOL_SHA256,
    validate_numerical_recovery_protocol,
)
from ts_rag.matrix import (
    load_manifest,
    save_json_atomic,
    summarize_cell_states,
)


def _load_json(path, label):
    try:
        with Path(path).open("r", encoding="utf-8") as source:
            payload = json.load(source)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is missing or invalid: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _validate_full_revision(value, label):
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a full lowercase Git commit")
    return value


def _validate_exact_failure_types(summary, label):
    states = summary.get("cell_states")
    if not isinstance(states, list):
        raise ValueError(f"{label} has no cell states")
    invalid = [
        row.get("cell_id")
        for row in states
        if row.get("state") == "failed"
        and row.get("error_type") != "NumericalIntegrityError"
    ]
    if invalid:
        raise ValueError(
            f"{label} exact failures must be explicit "
            f"NumericalIntegrityError: {invalid}"
        )


def _validate_a10g_metadata(metadata, summary_path, profile):
    summary_path = Path(summary_path).resolve()
    output_root = Path(metadata.get("output_root", "")).resolve()
    expected = {
        "schema_version": 1,
        "cohort_id": NUMERICAL_RECOVERY_COHORT_ID,
        "profile": profile,
        "overrides": NUMERICAL_RECOVERY_PROFILES[profile],
        "method_revision": METHOD_REVISION,
        "protocol_sha256": NUMERICAL_RECOVERY_PROTOCOL_SHA256,
        "cell_ids": list(NUMERICAL_RECOVERY_CELL_IDS),
        "cell_ids_sha256": NUMERICAL_RECOVERY_CELL_IDS_SHA256,
        "seed": 2021,
        "instance_type": "ml.g5.12xlarge",
        "instance_count": 1,
        "reserved_gpus_per_host": 4,
        "processes_per_host": 4,
        "total_gpus": 4,
        "world_size": 4,
        "inactive_reserved_gpus": 0,
    }
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError(f"A10G {profile} recovery metadata drifted")
    if output_root != summary_path.parent:
        raise ValueError(
            f"A10G {profile} summary is outside its recorded output root"
        )
    if profile == "exact":
        _validate_exact_failure_types(
            _load_json(summary_path, f"A10G {profile} summary"),
            f"A10G {profile}",
        )
    return {
        "recovery_revision": _validate_full_revision(
            metadata.get("recovery_revision"),
            f"A10G {profile} recovery revision",
        ),
        "parent_run_id": metadata.get("parent_run_id"),
    }


def _validate_a100_receipt(receipt, summary_path, profile):
    summary_path = Path(summary_path).resolve()
    summary_record = receipt.get("recomputed_matrix_summary")
    counts = receipt.get("matrix_counts", {})
    if (
        receipt.get("schema_version") != 1
        or receipt.get("job_kind") != NUMERICAL_RECOVERY_JOB
        or receipt.get("profile") != profile
        or receipt.get("method_revision") != METHOD_REVISION
        or receipt.get("protocol_sha256")
        != NUMERICAL_RECOVERY_PROTOCOL_SHA256
        or receipt.get("cell_ids_sha256")
        != NUMERICAL_RECOVERY_CELL_IDS_SHA256
        or receipt.get("topology_gate_passed") is not True
        or not isinstance(summary_record, dict)
        or Path(summary_record.get("path", "")).resolve() != summary_path
        or summary_record.get("sha256") != _sha256(summary_path)
        or summary_record.get("size_bytes") != summary_path.stat().st_size
        or counts.get("expected") != len(NUMERICAL_RECOVERY_CELL_IDS)
        or counts.get("pending") != 0
        or counts.get("running") != 0
        or counts.get("incomplete") != 0
        or counts.get("completed", 0) + counts.get("failed", 0)
        != len(NUMERICAL_RECOVERY_CELL_IDS)
        or (
            profile == "fallback-v1"
            and (
                counts.get("completed") != len(NUMERICAL_RECOVERY_CELL_IDS)
                or counts.get("failed") != 0
            )
        )
    ):
        raise ValueError(f"A100 {profile} recovery receipt drifted")
    source_revision = _validate_full_revision(
        receipt.get("source_revision"),
        f"A100 {profile} source revision",
    )
    recovery_revision = _validate_full_revision(
        receipt.get("recovery_revision"),
        f"A100 {profile} recovery revision",
    )
    if source_revision != recovery_revision:
        raise ValueError(
            f"A100 {profile} source and recovery revisions differ"
        )
    if profile == "exact":
        _validate_exact_failure_types(
            _load_json(summary_path, f"A100 {profile} summary"),
            f"A100 {profile}",
        )
    return {
        "run_id": receipt.get("run_id"),
        "source_revision": source_revision,
        "launcher_revision": _validate_full_revision(
            receipt.get("launcher_revision"),
            f"A100 {profile} launcher revision",
        ),
        "recovery_revision": recovery_revision,
    }


def validate_composition_provenance(
    *,
    a10g_exact_metadata,
    a10g_fallback_metadata,
    a100_exact_receipt,
    a100_fallback_receipt,
    a10g_exact_summary,
    a10g_fallback_summary,
    a100_exact_summary,
    a100_fallback_summary,
):
    a10g = {
        "exact": _validate_a10g_metadata(
            a10g_exact_metadata,
            a10g_exact_summary,
            "exact",
        ),
        "fallback-v1": _validate_a10g_metadata(
            a10g_fallback_metadata,
            a10g_fallback_summary,
            "fallback-v1",
        ),
    }
    a100 = {
        "exact": _validate_a100_receipt(
            a100_exact_receipt,
            a100_exact_summary,
            "exact",
        ),
        "fallback-v1": _validate_a100_receipt(
            a100_fallback_receipt,
            a100_fallback_summary,
            "fallback-v1",
        ),
    }
    if (
        a10g["exact"]["recovery_revision"]
        != a10g["fallback-v1"]["recovery_revision"]
    ):
        raise ValueError("A10G recovery profile revisions differ")
    if (
        a10g["exact"]["parent_run_id"]
        != a10g["fallback-v1"]["parent_run_id"]
    ):
        raise ValueError("A10G recovery parent runs differ")
    if (
        a100["exact"]["recovery_revision"]
        != a100["fallback-v1"]["recovery_revision"]
    ):
        raise ValueError("A100 recovery profile revisions differ")
    if (
        a100["exact"]["source_revision"]
        != a100["fallback-v1"]["source_revision"]
        or a100["exact"]["launcher_revision"]
        != a100["fallback-v1"]["launcher_revision"]
    ):
        raise ValueError("A100 recovery profile source identity differs")
    if a100["exact"]["run_id"] == a100["fallback-v1"]["run_id"]:
        raise ValueError("A100 recovery profiles must use distinct run IDs")
    return {
        "a10g": {
            "recovery_revision": a10g["exact"]["recovery_revision"],
            "parent_run_id": a10g["exact"]["parent_run_id"],
        },
        "a100": {
            "recovery_revision": a100["exact"]["recovery_revision"],
            "source_revision": a100["exact"]["source_revision"],
            "launcher_revision": a100["exact"]["launcher_revision"],
            "run_ids": {
                profile: a100[profile]["run_id"] for profile in a100
            },
        },
        "cross_hardware_recovery_revision_match_required": False,
    }


def _revision_matches(actual, expected):
    return bool(
        actual
        and expected
        and (actual.startswith(expected) or expected.startswith(actual))
    )


def _index_summary(summary, expected_ids, label):
    if not _revision_matches(summary.get("source_revision"), METHOD_REVISION):
        raise ValueError(f"{label} method revision drifted")
    states = summary.get("cell_states")
    if not isinstance(states, list):
        raise ValueError(f"{label} has no cell_states")
    indexed = {row.get("cell_id"): row for row in states}
    if (
        len(indexed) != len(states)
        or set(indexed) != set(expected_ids)
    ):
        raise ValueError(f"{label} cell scope drifted")
    return indexed


def _is_numerically_invalid(row):
    return bool(row.get("numerical_integrity_error"))


def compose_numerical_recovery(
    manifest,
    protocol,
    a10g_primary,
    a100_primary,
    a10g_exact,
    a100_exact,
    a10g_fallback,
    a100_fallback,
):
    manifest = list(manifest)
    manifest_ids = [cell["id"] for cell in manifest]
    cohort_ids = list(protocol["cell_ids"])
    if cohort_ids != list(NUMERICAL_RECOVERY_CELL_IDS):
        raise ValueError("Recovery protocol cohort drifted")

    primary = {
        "a10g": _index_summary(a10g_primary, manifest_ids, "A10G primary"),
        "a100": _index_summary(a100_primary, manifest_ids, "A100 primary"),
    }
    exact = {
        "a10g": _index_summary(a10g_exact, cohort_ids, "A10G exact"),
        "a100": _index_summary(a100_exact, cohort_ids, "A100 exact"),
    }
    _validate_exact_failure_types(a10g_exact, "A10G exact")
    _validate_exact_failure_types(a100_exact, "A100 exact")
    fallback = {
        "a10g": _index_summary(
            a10g_fallback,
            cohort_ids,
            "A10G fallback",
        ),
        "a100": _index_summary(
            a100_fallback,
            cohort_ids,
            "A100 fallback",
        ),
    }

    affected = sorted(
        {
            cell_id
            for hardware in ("a10g", "a100")
            for cell_id, row in primary[hardware].items()
            if _is_numerically_invalid(row)
        }
    )
    outside = set(affected) - set(cohort_ids)
    if outside:
        raise ValueError(
            "Non-finite primary cells fall outside the frozen recovery "
            f"cohort: {sorted(outside)}"
        )
    for hardware in ("a10g", "a100"):
        unresolved = [
            cell_id
            for cell_id, row in primary[hardware].items()
            if cell_id not in affected and row.get("state") != "completed"
        ]
        if unresolved:
            raise ValueError(
                f"{hardware} primary has unresolved non-recovery cells: "
                f"{unresolved}"
            )

    selected_profiles = {}
    for cell_id in affected:
        if all(
            exact[hardware][cell_id].get("state") == "completed"
            for hardware in ("a10g", "a100")
        ):
            selected_profiles[cell_id] = "exact"
        elif all(
            fallback[hardware][cell_id].get("state") == "completed"
            for hardware in ("a10g", "a100")
        ):
            selected_profiles[cell_id] = "fallback-v1"
        else:
            raise ValueError(
                "No cross-hardware finite recovery profile for "
                f"{cell_id}"
            )

    composed = {}
    for hardware in ("a10g", "a100"):
        states = []
        for cell in manifest:
            cell_id = cell["id"]
            original = primary[hardware][cell_id]
            profile = selected_profiles.get(cell_id)
            if profile is None:
                states.append(copy.deepcopy(original))
                continue
            source = (
                exact[hardware][cell_id]
                if profile == "exact"
                else fallback[hardware][cell_id]
            )
            selected = copy.deepcopy(source)
            selected["started_unix"] = original.get("started_unix")
            selected["numerical_recovery"] = {
                "cohort_id": NUMERICAL_RECOVERY_COHORT_ID,
                "profile": profile,
                "selection_uses_metric_quality": False,
                "original_artifact_dir": original.get("artifact_dir"),
                "selected_artifact_dir": source.get("artifact_dir"),
                "first_pass_started_unix": original.get("started_unix"),
                "recovery_started_unix": source.get("started_unix"),
            }
            states.append(selected)
        summary = summarize_cell_states(
            manifest,
            states,
            output_root=f"composed:{hardware}",
            source_revision=METHOD_REVISION,
        )
        summary["numerical_recovery"] = {
            "cohort_id": NUMERICAL_RECOVERY_COHORT_ID,
            "affected_cell_ids": affected,
            "selected_profiles": selected_profiles,
            "selection_uses_metric_quality": False,
        }
        composed[hardware] = summary

    return {
        "schema_version": 1,
        "method_revision": METHOD_REVISION,
        "cohort_id": NUMERICAL_RECOVERY_COHORT_ID,
        "affected_cell_ids": affected,
        "selected_profiles": selected_profiles,
        "exact_selected_count": sum(
            profile == "exact" for profile in selected_profiles.values()
        ),
        "fallback_selected_count": sum(
            profile == "fallback-v1"
            for profile in selected_profiles.values()
        ),
        "selection_uses_metric_quality": False,
        "composed": composed,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Compose TimeRAF primary and numerical recovery summaries"
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--a10g-primary", required=True)
    parser.add_argument("--a100-primary", required=True)
    parser.add_argument("--a10g-exact", required=True)
    parser.add_argument("--a100-exact", required=True)
    parser.add_argument("--a10g-fallback", required=True)
    parser.add_argument("--a100-fallback", required=True)
    parser.add_argument("--a10g-exact-metadata", required=True)
    parser.add_argument("--a10g-fallback-metadata", required=True)
    parser.add_argument("--a100-exact-receipt", required=True)
    parser.add_argument("--a100-fallback-receipt", required=True)
    parser.add_argument("--a10g-output", required=True)
    parser.add_argument("--a100-output", required=True)
    parser.add_argument("--receipt", required=True)
    args = parser.parse_args(argv)

    protocol_evidence = validate_numerical_recovery_protocol(args.protocol)
    protocol = _load_json(args.protocol, "Recovery protocol")
    paths = {
        "a10g_primary": args.a10g_primary,
        "a100_primary": args.a100_primary,
        "a10g_exact": args.a10g_exact,
        "a100_exact": args.a100_exact,
        "a10g_fallback": args.a10g_fallback,
        "a100_fallback": args.a100_fallback,
    }
    payloads = {
        name: _load_json(path, name)
        for name, path in paths.items()
    }
    provenance_paths = {
        "a10g_exact_metadata": args.a10g_exact_metadata,
        "a10g_fallback_metadata": args.a10g_fallback_metadata,
        "a100_exact_receipt": args.a100_exact_receipt,
        "a100_fallback_receipt": args.a100_fallback_receipt,
    }
    provenance_payloads = {
        name: _load_json(path, name)
        for name, path in provenance_paths.items()
    }
    provenance = validate_composition_provenance(
        **provenance_payloads,
        a10g_exact_summary=args.a10g_exact,
        a10g_fallback_summary=args.a10g_fallback,
        a100_exact_summary=args.a100_exact,
        a100_fallback_summary=args.a100_fallback,
    )
    result = compose_numerical_recovery(
        load_manifest(args.manifest),
        protocol,
        **payloads,
    )
    save_json_atomic(result["composed"]["a10g"], args.a10g_output)
    save_json_atomic(result["composed"]["a100"], args.a100_output)
    receipt = {
        key: value
        for key, value in result.items()
        if key != "composed"
    }
    receipt.update(
        {
            "protocol_sha256": protocol_evidence["sha256"],
            "inputs": {
                name: {"path": str(path), "sha256": _sha256(path)}
                for name, path in paths.items()
            },
            "provenance": {
                **provenance,
                "inputs": {
                    name: {"path": str(path), "sha256": _sha256(path)}
                    for name, path in provenance_paths.items()
                },
            },
            "outputs": {
                "a10g": {
                    "path": str(args.a10g_output),
                    "sha256": _sha256(args.a10g_output),
                },
                "a100": {
                    "path": str(args.a100_output),
                    "sha256": _sha256(args.a100_output),
                },
            },
        }
    )
    save_json_atomic(receipt, args.receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
