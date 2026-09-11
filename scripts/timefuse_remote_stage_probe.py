"""Emit explicit remote artifact state without relying on SSH exit status."""

import argparse
import hashlib
import json
from pathlib import Path


EXPECTED_AUDIT_CRITERIA = {
    "primary_585_gate_passed",
    "numerical_recovery_composition_verified",
    "frozen_confirmation_462_gate_passed",
    "greenland_replay_208_complete",
    "greenland_eight_a100_topology_passed",
    "appendix_94_gate_passed",
}
EXPECTED_APPENDIX_TOPOLOGY = {
    "instance_type": "ml.g5.12xlarge",
    "instance_count": 1,
    "reserved_gpus_per_host": 4,
    "active_gpu_workers": 0,
    "inactive_reserved_gpus": 4,
    "cpu_workers": 4,
    "gpu_inactivity_reason": (
        "Appendix RAG replays numerical corrections over existing prediction "
        "bundles and performs no model inference."
    ),
}
EXPECTED_APPENDIX_SELECTOR_POLICY = "validation_argmin_v3"
EXPECTED_APPENDIX_SELECTOR_PROTOCOL_SHA256 = (
    "6c7de88b2d10e19b04ad56259088ec51"
    "d0b14dc3cd5f092f5cc14d66878f53f1"
)


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _marker(state, reason, **details):
    return {
        "marker": state,
        "reason": reason,
        **details,
    }


def _load_artifact(artifact_root, artifact):
    root = Path(artifact_root).resolve()
    path = Path(artifact)
    path = path.resolve() if path.is_absolute() else (root / path).resolve()
    if path == root or root not in path.parents:
        return None, _marker("invalid", "artifact escapes artifact root")
    if not path.exists():
        return None, _marker("absent", "artifact does not exist")
    if path.is_symlink() or not path.is_file():
        return None, _marker("invalid", "artifact is not a regular file")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, _marker("invalid", "artifact is not valid JSON")
    if not isinstance(payload, dict):
        return None, _marker("invalid", "artifact is not a JSON object")
    return (path, payload), None


def probe_appendix_summary(
    artifact_root,
    artifact,
    *,
    source_revision,
    appendix_method_revision,
    appendix_launcher_revision,
):
    loaded, marker = _load_artifact(artifact_root, artifact)
    if marker is not None:
        return marker
    path, payload = loaded
    counts = payload.get("counts")
    states = payload.get("cell_states")
    if (
        payload.get("source_revision") != source_revision
        or payload.get("method_revision") != appendix_method_revision
        or payload.get("launcher_revision") != appendix_launcher_revision
        or payload.get("selector_policy")
        != EXPECTED_APPENDIX_SELECTOR_POLICY
        or payload.get("selector_protocol_sha256")
        != EXPECTED_APPENDIX_SELECTOR_PROTOCOL_SHA256
        or payload.get("execution_topology") != EXPECTED_APPENDIX_TOPOLOGY
        or not isinstance(counts, dict)
        or counts.get("reported") != 96
        or counts.get("evaluable") != 94
        or counts.get("completed") != 94
        or counts.get("paper_oot") != 2
        or any(
            counts.get(name) != 0
            for name in ("failed", "running", "pending", "incomplete")
        )
        or payload.get("all_evaluable_completed") is not True
        or not isinstance(states, list)
        or len(states) != 96
        or len({row.get("cell_id") for row in states}) != 96
        or not isinstance(payload.get("publication_gate"), dict)
        or not isinstance(
            payload["publication_gate"].get("development_gate_passed"),
            bool,
        )
    ):
        return _marker("invalid", "Appendix summary identity or scope drifted")
    return _marker(
        "valid",
        "Appendix summary is complete and structurally valid",
        artifact=str(path),
        sha256=_sha256_file(path),
        publication_gate_passed=payload["publication_gate"][
            "development_gate_passed"
        ],
    )


def probe_publication_audit(
    artifact_root,
    artifact,
    *,
    method_revision,
    appendix_method_revision,
    appendix_launcher_revision,
):
    loaded, marker = _load_artifact(artifact_root, artifact)
    if marker is not None:
        return marker
    path, payload = loaded
    criteria = payload.get("criteria")
    publication_ready = payload.get("publication_ready")
    if (
        payload.get("schema_version") != 1
        or payload.get("method_revision") != method_revision
        or payload.get("appendix_method_revision")
        != appendix_method_revision
        or payload.get("appendix_launcher_revision")
        != appendix_launcher_revision
        or "error" in payload
        or not isinstance(criteria, dict)
        or set(criteria) != EXPECTED_AUDIT_CRITERIA
        or any(not isinstance(value, bool) for value in criteria.values())
        or not isinstance(publication_ready, bool)
        or publication_ready is not all(criteria.values())
    ):
        return _marker("invalid", "publication audit identity or result drifted")
    return _marker(
        "valid",
        "publication audit is structurally valid",
        artifact=str(path),
        sha256=_sha256_file(path),
        publication_ready=publication_ready,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Probe a TimeFuse remote stage with an explicit marker"
    )
    parser.add_argument(
        "--kind",
        choices=("appendix-summary", "publication-audit"),
        required=True,
    )
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--source-revision")
    parser.add_argument("--method-revision")
    parser.add_argument("--appendix-method-revision", required=True)
    parser.add_argument("--appendix-launcher-revision", required=True)
    args = parser.parse_args(argv)

    if args.kind == "appendix-summary":
        if not args.source_revision:
            parser.error("--source-revision is required for appendix-summary")
        result = probe_appendix_summary(
            args.artifact_root,
            args.artifact,
            source_revision=args.source_revision,
            appendix_method_revision=args.appendix_method_revision,
            appendix_launcher_revision=args.appendix_launcher_revision,
        )
    else:
        if not args.method_revision:
            parser.error("--method-revision is required for publication-audit")
        result = probe_publication_audit(
            args.artifact_root,
            args.artifact,
            method_revision=args.method_revision,
            appendix_method_revision=args.appendix_method_revision,
            appendix_launcher_revision=args.appendix_launcher_revision,
        )
    print("TIMERAF_STAGE_PROBE=" + json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
