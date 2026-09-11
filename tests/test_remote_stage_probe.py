import json

from scripts.timefuse_remote_stage_probe import (
    EXPECTED_APPENDIX_TOPOLOGY,
    EXPECTED_APPENDIX_SELECTOR_POLICY,
    EXPECTED_APPENDIX_SELECTOR_PROTOCOL_SHA256,
    EXPECTED_AUDIT_CRITERIA,
    probe_appendix_summary,
    probe_publication_audit,
)


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_probe_reports_absent_without_using_process_status(tmp_path):
    result = probe_appendix_summary(
        tmp_path,
        tmp_path / "missing.json",
        source_revision="source",
        appendix_method_revision="appendix",
        appendix_launcher_revision="launcher",
    )

    assert result["marker"] == "absent"


def test_probe_accepts_complete_appendix_summary(tmp_path):
    summary = _write_json(
        tmp_path / "outputs" / "appendix.json",
        {
            "source_revision": "source",
            "method_revision": "appendix",
            "launcher_revision": "launcher",
            "selector_policy": EXPECTED_APPENDIX_SELECTOR_POLICY,
            "selector_protocol_sha256": (
                EXPECTED_APPENDIX_SELECTOR_PROTOCOL_SHA256
            ),
            "execution_topology": EXPECTED_APPENDIX_TOPOLOGY,
            "counts": {
                "reported": 96,
                "evaluable": 94,
                "completed": 94,
                "paper_oot": 2,
                "failed": 0,
                "running": 0,
                "pending": 0,
                "incomplete": 0,
            },
            "all_evaluable_completed": True,
            "cell_states": [
                {"cell_id": f"cell-{index}"} for index in range(96)
            ],
            "publication_gate": {"development_gate_passed": True},
        },
    )

    result = probe_appendix_summary(
        tmp_path,
        summary,
        source_revision="source",
        appendix_method_revision="appendix",
        appendix_launcher_revision="launcher",
    )

    assert result["marker"] == "valid"
    assert result["publication_gate_passed"]
    assert len(result["sha256"]) == 64


def test_probe_rejects_incomplete_appendix_summary(tmp_path):
    summary = _write_json(
        tmp_path / "appendix.json",
        {
            "source_revision": "source",
            "method_revision": "appendix",
            "counts": {"reported": 96, "completed": 93},
        },
    )

    result = probe_appendix_summary(
        tmp_path,
        summary,
        source_revision="source",
        appendix_method_revision="appendix",
        appendix_launcher_revision="launcher",
    )

    assert result["marker"] == "invalid"


def test_probe_accepts_below_threshold_audit_as_valid(tmp_path):
    criteria = {name: True for name in EXPECTED_AUDIT_CRITERIA}
    criteria["appendix_94_gate_passed"] = False
    audit = _write_json(
        tmp_path / "audit.json",
        {
            "schema_version": 1,
            "method_revision": "primary",
            "appendix_method_revision": "appendix",
            "appendix_launcher_revision": "launcher",
            "criteria": criteria,
            "publication_ready": False,
        },
    )

    result = probe_publication_audit(
        tmp_path,
        audit,
        method_revision="primary",
        appendix_method_revision="appendix",
        appendix_launcher_revision="launcher",
    )

    assert result["marker"] == "valid"
    assert not result["publication_ready"]


def test_probe_rejects_audit_identity_drift(tmp_path):
    audit = _write_json(
        tmp_path / "audit.json",
        {
            "schema_version": 1,
            "method_revision": "primary",
            "appendix_method_revision": "wrong",
            "appendix_launcher_revision": "launcher",
            "criteria": {
                name: True for name in EXPECTED_AUDIT_CRITERIA
            },
            "publication_ready": True,
        },
    )

    result = probe_publication_audit(
        tmp_path,
        audit,
        method_revision="primary",
        appendix_method_revision="appendix",
        appendix_launcher_revision="launcher",
    )

    assert result["marker"] == "invalid"
