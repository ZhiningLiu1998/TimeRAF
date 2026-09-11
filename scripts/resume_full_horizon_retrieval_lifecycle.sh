#!/usr/bin/env bash

set -euo pipefail

ROOT="${TIMERAF_ROOT:-/home/sagemaker-user/user-default-efs/workspace/TimeRAF}"
METHOD_REVISION="${1:?usage: $0 METHOD_REVISION LAUNCHER_REVISION}"
LAUNCHER_REVISION="${2:?usage: $0 METHOD_REVISION LAUNCHER_REVISION}"
SOURCE="$ROOT/operations/full-horizon-retrieval-$LAUNCHER_REVISION/source"
PYTHON="$ROOT/.venv-gpu312/bin/python"
RUN_ROOT="$ROOT/outputs/full_horizon_retrieval/$METHOD_REVISION"
REPLAY_ROOT="$RUN_ROOT/checkpoint_replay"
RETRIEVAL_ROOT="$RUN_ROOT/retrieval_matrix"
CATALOG="$RUN_ROOT/prediction_bundle_catalog.json"
STATE="$RUN_ROOT/lifecycle.state"
LOG="$RUN_ROOT/lifecycle.log"
PID_FILE="$RUN_ROOT/lifecycle.pid"
MANIFEST="$SOURCE/docs/timefuse_experiment_manifest.jsonl"
PROTOCOL="$SOURCE/docs/retrieval_baseline_protocol.json"
CHECKPOINT_CATALOG="$SOURCE/docs/timefuse_full_horizon_checkpoint_catalog.json"
A10G_SUMMARY="$SOURCE/docs/publication_results/a10g_composed_summary.json"
RECOVERY_PROTOCOL="$SOURCE/docs/retrieval_baseline_horizon_compatibility_recovery.json"
EVIDENCE_ROOT="$RUN_ROOT/recovery/horizon_compatibility"

mkdir -p "$RUN_ROOT"
printf '%s\n' "$$" >"$PID_FILE"
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
    record_state "${CURRENT_STAGE:-recovery_startup}" "FAILED(exit=$status)"
    exit "$status"
}
trap fail ERR

CURRENT_STAGE="compatibility_recovery_preflight"
record_state "$CURRENT_STAGE" "RUNNING"
test -x "$PYTHON"
test -d "$SOURCE"
test -d "$RUN_ROOT"
test -f "$MANIFEST"
test -f "$PROTOCOL"
test -f "$CHECKPOINT_CATALOG"
test -f "$A10G_SUMMARY"
test -f "$RECOVERY_PROTOCOL"
test -f "$CATALOG"
test -f "$RETRIEVAL_ROOT/matrix_summary.json"
test -f "$RETRIEVAL_ROOT/run_metadata.json"
test -f "$RETRIEVAL_ROOT/startup_topology.json"
test "$(
    git -C "$SOURCE" rev-parse --verify "$LAUNCHER_REVISION^{commit}"
)" = "$LAUNCHER_REVISION"
test "$(
    git -C "$SOURCE" rev-parse --verify "$METHOD_REVISION^{commit}"
)" = "$METHOD_REVISION"
test -z "$(git -C "$SOURCE" for-each-ref --format='%(refname)')"
test "$(findmnt -T "$ROOT" -o FSTYPE -n)" = "nfs4"
test "$(nvidia-smi -L | wc -l)" -eq 8

for relative_path in \
    scripts/resume_full_horizon_retrieval_lifecycle.sh \
    scripts/run_retrieval_baseline_matrix.py \
    scripts/run_retrieval_baseline_cell.py \
    ts_rag/retrieval_baselines.py \
    docs/retrieval_baseline_horizon_compatibility_recovery.json \
    docs/retrieval_baseline_protocol.json; do
    git -C "$SOURCE" show "$LAUNCHER_REVISION:$relative_path" |
        cmp - "$SOURCE/$relative_path"
done

"$PYTHON" - \
    "$METHOD_REVISION" \
    "$LAUNCHER_REVISION" \
    "$PROTOCOL" \
    "$RECOVERY_PROTOCOL" \
    "$CATALOG" \
    "$RETRIEVAL_ROOT/matrix_summary.json" \
    "$RETRIEVAL_ROOT/run_metadata.json" \
    "$EVIDENCE_ROOT/initial_failure_receipt.json" <<'PY'
import hashlib
import json
import os
import sys

(
    method_revision,
    launcher_revision,
    protocol_path,
    recovery_path,
    catalog_path,
    summary_path,
    metadata_path,
    evidence_receipt_path,
) = sys.argv[1:]

def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

with open(recovery_path, encoding="utf-8") as source:
    recovery = json.load(source)
with open(summary_path, encoding="utf-8") as source:
    summary = json.load(source)
with open(metadata_path, encoding="utf-8") as source:
    metadata = json.load(source)

assert recovery["status"] == "frozen_before_recovery_execution"
assert recovery["method_identity"]["source_revision"] == method_revision
assert recovery["method_identity"]["protocol_sha256"] == sha256(protocol_path)
assert recovery["method_identity"]["catalog_sha256"] == sha256(catalog_path)
assert recovery["recovery_rule"]["selection_uses_metric_quality"] is False
assert recovery["affected_shape_cohort"]["expected_cells"] == 52
assert method_revision != launcher_revision
assert summary["source_revision"] == method_revision
assert summary["protocol_sha256"] == sha256(protocol_path)
assert summary["catalog_sha256"] == sha256(catalog_path)
assert metadata["source_revision"] == method_revision
assert metadata["topology"]["worker_gpu_ids"] == list(range(8))
if not os.path.exists(evidence_receipt_path):
    assert summary["launcher_revision"] == method_revision
    assert summary["counts"] == {
        "completed": 432,
        "failed": 1,
        "running": 0,
        "pending": 152,
        "incomplete": 0,
    }
    failed = [
        row for row in summary["cell_states"] if row["state"] == "failed"
    ]
    assert len(failed) == 1
    assert failed[0]["cell_id"] == recovery["trigger"]["cell_id"]
    assert failed[0]["error_type"] == recovery["trigger"]["error_type"]
    assert failed[0]["error"] == recovery["trigger"]["error"]
    assert metadata["launcher_revision"] == method_revision
else:
    assert summary["launcher_revision"] == launcher_revision
    assert metadata["launcher_revision"] == launcher_revision
    assert summary["counts"]["failed"] == 0
    assert summary["counts"]["incomplete"] == 0
    assert sum(summary["counts"].values()) == 585
PY

if [[ ! -f "$EVIDENCE_ROOT/initial_failure_receipt.json" ]]; then
    mkdir -p "$EVIDENCE_ROOT"
    cp -p "$STATE" "$EVIDENCE_ROOT/initial_lifecycle.state"
    cp -p "$RETRIEVAL_ROOT/matrix_summary.json" \
        "$EVIDENCE_ROOT/initial_matrix_summary.json"
    cp -p "$RETRIEVAL_ROOT/run_metadata.json" \
        "$EVIDENCE_ROOT/initial_run_metadata.json"
    mv "$RETRIEVAL_ROOT/startup_topology.json" \
        "$EVIDENCE_ROOT/initial_startup_topology.json"
    cp -p \
        "$RETRIEVAL_ROOT/cells/pems/PEMS03/Autoformer/pl6/status.json" \
        "$EVIDENCE_ROOT/failed_status.json"
    cp -p "$RETRIEVAL_ROOT/logs/pems__PEMS03__Autoformer__6.log" \
        "$EVIDENCE_ROOT/failed_cell.log"
    "$PYTHON" - "$EVIDENCE_ROOT" <<'PY'
import hashlib
import json
import os
import sys
import time

root = sys.argv[1]

def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

artifacts = {}
for name in sorted(os.listdir(root)):
    path = os.path.join(root, name)
    if os.path.isfile(path):
        artifacts[name] = {
            "size_bytes": os.path.getsize(path),
            "sha256": sha256(path),
        }
payload = {
    "schema_version": 1,
    "recorded_unix": time.time(),
    "failure_cell": "pems/PEMS03/Autoformer/6",
    "artifacts": artifacts,
}
output = os.path.join(root, "initial_failure_receipt.json")
with open(output + ".tmp", "w", encoding="utf-8") as destination:
    json.dump(payload, destination, indent=2, sort_keys=True)
    destination.write("\n")
os.replace(output + ".tmp", output)
PY
fi
record_state "$CURRENT_STAGE" "COMPLETE"

CURRENT_STAGE="retrieval_matrix_compatibility_recovery"
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
    --source-revision "$METHOD_REVISION" \
    --launcher-revision "$LAUNCHER_REVISION" \
    --instance-type ml.p5.48xlarge \
    --reserved-gpus-per-host 8 \
    --processes-per-host 8
"$PYTHON" - \
    "$METHOD_REVISION" \
    "$LAUNCHER_REVISION" \
    "$RETRIEVAL_ROOT/matrix_summary.json" \
    "$RETRIEVAL_ROOT/run_metadata.json" \
    "$RETRIEVAL_ROOT/startup_topology.json" <<'PY'
import json
import sys

method_revision, launcher_revision, summary_path, metadata_path, topology_path = (
    sys.argv[1:]
)
with open(summary_path, encoding="utf-8") as source:
    summary = json.load(source)
with open(metadata_path, encoding="utf-8") as source:
    metadata = json.load(source)
with open(topology_path, encoding="utf-8") as source:
    topology = json.load(source)
assert summary["source_revision"] == method_revision
assert summary["launcher_revision"] == launcher_revision
assert summary["counts"] == {
    "completed": 585,
    "failed": 0,
    "running": 0,
    "pending": 0,
    "incomplete": 0,
}
assert summary["all_completed"] is True
assert len(summary["cell_states"]) == 585
assert metadata["source_revision"] == method_revision
assert metadata["launcher_revision"] == launcher_revision
assert metadata["selected_cells"] == 585
assert topology["all_bindings_observed"] is True
assert topology["expected_worker_count"] == 8
assert topology["gpu_ids"] == list(range(8))
assert topology["distinct_positive_pids"] is True
assert len(topology["worker_bindings"]) == 8
PY
record_state "$CURRENT_STAGE" "COMPLETE"

CURRENT_STAGE="finalize_compatibility_recovery"
record_state "$CURRENT_STAGE" "RUNNING"
"$PYTHON" - \
    "$RUN_ROOT" \
    "$METHOD_REVISION" \
    "$LAUNCHER_REVISION" \
    "$SOURCE/../materialization_receipt.json" \
    "$MANIFEST" \
    "$PROTOCOL" \
    "$RECOVERY_PROTOCOL" \
    "$CHECKPOINT_CATALOG" \
    "$CATALOG" \
    "$REPLAY_ROOT/matrix_summary.json" \
    "$REPLAY_ROOT/run_metadata.json" \
    "$REPLAY_ROOT/startup_topology.json" \
    "$EVIDENCE_ROOT/initial_failure_receipt.json" \
    "$RETRIEVAL_ROOT/matrix_summary.json" \
    "$RETRIEVAL_ROOT/run_metadata.json" \
    "$RETRIEVAL_ROOT/startup_topology.json" <<'PY'
import hashlib
import json
import os
import sys
import time

root, method_revision, launcher_revision, *paths = sys.argv[1:]

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
    "schema_version": 2,
    "status": "completed",
    "recorded_unix": time.time(),
    "source_revision": method_revision,
    "launcher_revision": launcher_revision,
    "initial_launcher_revision": method_revision,
    "expected_cells": 585,
    "retained_completed_cells": 432,
    "compatibility_recovery_cells": 153,
    "selection_uses_metric_quality": False,
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
