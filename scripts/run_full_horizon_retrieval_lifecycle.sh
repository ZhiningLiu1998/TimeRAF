#!/usr/bin/env bash

set -euo pipefail

ROOT="${TIMERAF_ROOT:-/home/sagemaker-user/user-default-efs/workspace/TimeRAF}"
REVISION="${1:?usage: $0 FULL_SOURCE_REVISION}"
SOURCE="$ROOT/operations/full-horizon-retrieval-$REVISION/source"
PYTHON="$ROOT/.venv-gpu312/bin/python"
RUN_ROOT="$ROOT/outputs/full_horizon_retrieval/$REVISION"
REPLAY_ROOT="$RUN_ROOT/checkpoint_replay"
RETRIEVAL_ROOT="$RUN_ROOT/retrieval_matrix"
CATALOG="$RUN_ROOT/prediction_bundle_catalog.json"
STATE="$RUN_ROOT/lifecycle.state"
LOG="$RUN_ROOT/lifecycle.log"
MANIFEST="$SOURCE/docs/timefuse_experiment_manifest.jsonl"
PROTOCOL="$SOURCE/docs/retrieval_baseline_protocol.json"
CHECKPOINT_CATALOG="$SOURCE/docs/timefuse_full_horizon_checkpoint_catalog.json"
A10G_SUMMARY="$SOURCE/docs/publication_results/a10g_composed_summary.json"

mkdir -p "$RUN_ROOT"
exec >>"$LOG" 2>&1

record_state() {
    local stage="$1"
    local status="$2"
    local temporary="$STATE.tmp.$$"
    printf '%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "$stage" "$status" \
        >"$temporary"
    mv "$temporary" "$STATE"
}

fail() {
    local status=$?
    record_state "${CURRENT_STAGE:-startup}" "FAILED(exit=$status)"
    exit "$status"
}
trap fail ERR

CURRENT_STAGE="preflight"
record_state "$CURRENT_STAGE" "RUNNING"
test -x "$PYTHON"
test -d "$SOURCE"
test -f "$MANIFEST"
test -f "$PROTOCOL"
test -f "$CHECKPOINT_CATALOG"
test -f "$A10G_SUMMARY"
test "$(
    git -C "$SOURCE" rev-parse --verify "$REVISION^{commit}"
)" = "$REVISION"
test -z "$(git -C "$SOURCE" for-each-ref --format='%(refname)')"
test "$(findmnt -T "$ROOT" -o FSTYPE -n)" = "nfs4"
test "$(nvidia-smi -L | wc -l)" -eq 8
"$PYTHON" - "$MANIFEST" "$PROTOCOL" "$CHECKPOINT_CATALOG" <<'PY'
import json
import sys

manifest_path, protocol_path, catalog_path = sys.argv[1:]
with open(manifest_path, encoding="utf-8") as source:
    rows = [json.loads(line) for line in source if line.strip()]
with open(protocol_path, encoding="utf-8") as source:
    protocol = json.load(source)
with open(catalog_path, encoding="utf-8") as source:
    catalog = json.load(source)
ids = {row["id"] for row in rows}
assert len(rows) == len(ids) == 585
assert protocol["scope"]["expected_cells"] == 585
assert set(catalog["checkpoints"]) == ids
assert catalog["usable_count"] == 585
assert catalog["missing_count"] == 0
assert catalog["hashes_included"] is True
assert all(record["usable"] for record in catalog["checkpoints"].values())
PY
record_state "$CURRENT_STAGE" "COMPLETE"

CURRENT_STAGE="checkpoint_replay"
record_state "$CURRENT_STAGE" "RUNNING"
PYTHONPATH="$SOURCE" "$PYTHON" "$SOURCE/scripts/run_benchmark_matrix.py" \
    --manifest "$MANIFEST" \
    --output-root "$REPLAY_ROOT" \
    --checkpoint-root "$RUN_ROOT/unused_checkpoint_root" \
    --summary "$REPLAY_ROOT/matrix_summary.json" \
    --source-revision "$REVISION" \
    --launcher-revision "$REVISION" \
    --num-workers 4 \
    --no-train \
    --export-only \
    --max-failures 1 \
    --instance-type ml.p5.48xlarge \
    --reserved-gpus-per-host 8 \
    --processes-per-host 8 \
    --checkpoint-catalog "$CHECKPOINT_CATALOG" \
    --release-checkpoint-root "$ROOT" \
    --release-only \
    --working-root "$ROOT"
"$PYTHON" - "$REPLAY_ROOT/matrix_summary.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as source:
    summary = json.load(source)
assert summary["counts"] == {
    "expected": 585,
    "completed": 585,
    "failed": 0,
    "running": 0,
    "pending": 0,
    "incomplete": 0,
    "improved": 0,
    "not_improved": 585,
}
assert summary["all_completed"] is True
assert summary["export_only"] is True
PY
record_state "$CURRENT_STAGE" "COMPLETE"

CURRENT_STAGE="bundle_catalog"
record_state "$CURRENT_STAGE" "RUNNING"
if [[ ! -f "$CATALOG" ]]; then
    PYTHONPATH="$SOURCE" "$PYTHON" \
        "$SOURCE/scripts/index_prediction_bundles.py" \
        --manifest "$MANIFEST" \
        --output-root "$REPLAY_ROOT" \
        --project-root "$ROOT" \
        --source-revision "$REVISION" \
        --output "$CATALOG" \
        --hash
fi
"$PYTHON" - "$CATALOG" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as source:
    catalog = json.load(source)
assert catalog["expected_cells"] == 585
assert catalog["cataloged_cells"] == 585
assert catalog["usable_count"] == 585
assert catalog["missing_count"] == 0
assert catalog["hashes_verified"] is True
assert len(catalog["bundles"]) == 585
assert all(record["usable"] for record in catalog["bundles"].values())
PY
record_state "$CURRENT_STAGE" "COMPLETE"

CURRENT_STAGE="retrieval_matrix"
record_state "$CURRENT_STAGE" "RUNNING"
PYTHONPATH="$SOURCE" "$PYTHON" \
    "$SOURCE/scripts/run_retrieval_baseline_matrix.py" \
    --catalog "$CATALOG" \
    --manifest "$MANIFEST" \
    --protocol "$PROTOCOL" \
    --a10g-summary "$A10G_SUMMARY" \
    --output-root "$RETRIEVAL_ROOT" \
    --project-root "$ROOT" \
    --source-root "$SOURCE" \
    --source-revision "$REVISION" \
    --launcher-revision "$REVISION" \
    --instance-type ml.p5.48xlarge \
    --reserved-gpus-per-host 8 \
    --processes-per-host 8
"$PYTHON" - "$RETRIEVAL_ROOT/matrix_summary.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as source:
    summary = json.load(source)
assert summary["counts"] == {
    "completed": 585,
    "failed": 0,
    "running": 0,
    "pending": 0,
    "incomplete": 0,
}
assert summary["all_completed"] is True
assert len(summary["cell_states"]) == 585
PY
record_state "$CURRENT_STAGE" "COMPLETE"

CURRENT_STAGE="finalize"
record_state "$CURRENT_STAGE" "RUNNING"
"$PYTHON" - \
    "$RUN_ROOT" \
    "$REVISION" \
    "$CHECKPOINT_CATALOG" \
    "$CATALOG" \
    "$REPLAY_ROOT/matrix_summary.json" \
    "$REPLAY_ROOT/run_metadata.json" \
    "$REPLAY_ROOT/startup_topology.json" \
    "$RETRIEVAL_ROOT/matrix_summary.json" \
    "$RETRIEVAL_ROOT/run_metadata.json" \
    "$RETRIEVAL_ROOT/startup_topology.json" <<'PY'
import hashlib
import json
import os
import sys
import time

root, revision, *paths = sys.argv[1:]

def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

artifacts = {
    os.path.relpath(path, root): {
        "path": path,
        "size_bytes": os.path.getsize(path),
        "sha256": sha256(path),
    }
    for path in paths
}
payload = {
    "schema_version": 1,
    "status": "completed",
    "recorded_unix": time.time(),
    "source_revision": revision,
    "expected_cells": 585,
    "artifacts": artifacts,
}
output = os.path.join(root, "completion_receipt.json")
temporary = output + ".tmp"
with open(temporary, "w", encoding="utf-8") as destination:
    json.dump(payload, destination, indent=2, sort_keys=True)
    destination.write("\n")
os.replace(temporary, output)
print(json.dumps(payload, sort_keys=True))
PY
record_state "$CURRENT_STAGE" "COMPLETE"
printf 'COMPLETE %s\n' "$RUN_ROOT/completion_receipt.json"
