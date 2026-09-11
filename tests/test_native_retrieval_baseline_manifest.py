import json
from pathlib import Path

from scripts.generate_native_retrieval_baseline_manifest import build_manifest


ROOT = Path(__file__).resolve().parents[1]


def test_native_manifest_has_exact_frozen_scope():
    protocol = json.loads(
        (ROOT / "docs/native_retrieval_baseline_protocol.json").read_text(
            encoding="utf-8"
        )
    )
    rows = build_manifest(protocol)

    assert len(rows) == 52
    assert len({row["id"] for row in rows}) == 52
    assert sum(row["method"] == "ts_rag" for row in rows) == 7
    assert sum(row["method"] == "raf" for row in rows) == 44
    assert sum(row["method"] == "ratd" for row in rows) == 1


def test_raf_etth_scope_contains_context_50_once():
    protocol = json.loads(
        (ROOT / "docs/native_retrieval_baseline_protocol.json").read_text(
            encoding="utf-8"
        )
    )
    rows = build_manifest(protocol)
    contexts = [
        row["context_length"]
        for row in rows
        if row["method"] == "raf" and row["dataset"] == "ETTh"
    ]

    assert contexts == [50, 75, 100, 150]
