#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REVISION="${TIMERAF_POST_PRIMARY_REVISION:-$(git -C "$REPO_ROOT" rev-parse HEAD)}"
METHOD_REVISION="${TIMERAF_METHOD_REVISION:-d9be338}"
REGION="${TIMERAF_REGION:-us-east-1}"
export AWS_PROFILE="${TIMERAF_AWS_PROFILE:-timeraf-modeldev}"
DOMAIN_ID="${TIMERAF_DOMAIN_ID:-<SAGEMAKER_DOMAIN_ID>}"
SPACE_NAME="${TIMERAF_SPACE_NAME:-<DEV_SPACE>}"
SPACE_ARN="arn:aws:sagemaker:${REGION}:<AWS_ACCOUNT_ID>:space/${DOMAIN_ID}/${SPACE_NAME}"
CONNECT_SCRIPT="$REPO_ROOT/scripts/sagemaker_connect.sh"
KNOWN_HOSTS="/tmp/timeraf-sagemaker-known-hosts"
MONITOR_EVENTS="/tmp/timeraf-matrix-monitor-events.log"
LOG="/tmp/timeraf-post-primary-worker.log"
EVENTS="/tmp/timeraf-post-primary-events.log"
REMOTE_ROOT="/home/sagemaker-user/user-default-efs/workspace/TimeRAF"
REMOTE_OPS="$REMOTE_ROOT/operations/post-primary-$REVISION"
POLL_SECONDS="${TIMERAF_POST_PRIMARY_POLL_SECONDS:-60}"

SSH=(
    ssh -T
    -o "ProxyCommand=$CONNECT_SCRIPT %h"
    -o StrictHostKeyChecking=no
    -o "UserKnownHostsFile=$KNOWN_HOSTS"
    -o ConnectTimeout=45
    -l sagemaker-user
    "$SPACE_ARN"
)

log() {
    printf '%s %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$*" >>"$LOG"
}

event() {
    local line
    line="$(date -u +'%Y-%m-%dT%H:%M:%SZ') $*"
    printf '%s\n' "$line" >>"$LOG"
    printf '%s\n' "$line" >>"$EVENTS"
}

failed=1
finish() {
    local rc=$?
    if ((failed)); then
        if ((rc == 0)); then
            event "STOPPED revision=$REVISION rc=$rc"
        else
            event "FAILED revision=$REVISION rc=$rc"
        fi
    fi
}
trap finish EXIT

git -C "$REPO_ROOT" cat-file -e "$REVISION^{commit}"
log "worker started revision=$REVISION method_revision=$METHOD_REVISION aws_profile=$AWS_PROFILE"

while ! rg -q ' COMPLETE completed=585 ' "$MONITOR_EVENTS" 2>/dev/null; do
    screens="$(screen -list 2>/dev/null || true)"
    if ! rg -q '[.]timeraf-matrix-monitor[[:space:]]' <<<"$screens"; then
        event "FAILED matrix monitor exited without a 585-cell completion event"
        exit 2
    fi
    sleep "$POLL_SECONDS"
done
log "matrix completion event observed"

while true; do
    readiness="$(
        "${SSH[@]}" "
ROOT='$REMOTE_ROOT'
SUMMARY=\"\$ROOT/ts_rag_outputs/timefuse_matrix_full/matrix_d9be338_summary.json\"
LOCK=\"\$ROOT/ts_rag_outputs/timefuse_matrix_full/matrix.lock\"
test \"\$(findmnt -T \"\$ROOT\" -o FSTYPE -n)\" = nfs4 || {
    printf storage-not-efs
    exit 0
}
test \"\$(jq -r '.counts.completed' \"\$SUMMARY\")\" = 585 || {
    printf summary-incomplete
    exit 0
}
if pgrep -f '[s]cripts/run_benchmark_matrix.py' >/dev/null; then
    printf runner-active
elif flock -n \"\$LOCK\" -c true; then
    printf ready
else
    printf matrix-locked
fi
" 2>>"$LOG" || true
    )"
    if [[ "$readiness" == "ready" ]]; then
        break
    fi
    log "waiting for finalization readiness: ${readiness:-ssh-error}"
    sleep "$POLL_SECONDS"
done

log "streaming pinned source revision to EFS operation directory"
git -C "$REPO_ROOT" archive "$REVISION" |
    "${SSH[@]}" "
set -eu
OPS='$REMOTE_OPS'
mkdir -p \"\$OPS/source\"
tar -xf - -C \"\$OPS/source\"
" >>"$LOG" 2>&1

log "building publication, confirmation, and checkpoint evidence"
"${SSH[@]}" bash -s -- \
    "$REMOTE_ROOT" "$REMOTE_OPS" "$METHOD_REVISION" >>"$LOG" 2>&1 <<'REMOTE'
set -euo pipefail

ROOT="$1"
OPS="$2"
METHOD_REVISION="$3"
SOURCE="$OPS/source"
PYTHON="$ROOT/.venv-gpu312/bin/python"
PRIMARY="$OPS/timefuse_matrix_publication_summary.json"
CONFIRMATION="$OPS/timefuse_confirmation_summary.json"
CATALOG="$OPS/timefuse_matrix_checkpoint_catalog.json"

test "$(findmnt -T "$ROOT" -o FSTYPE -n)" = nfs4
test -x "$PYTHON"
cd "$SOURCE"
export PYTHONPATH="$SOURCE"

"$PYTHON" scripts/summarize_timefuse_matrix.py \
    --manifest "$SOURCE/docs/timefuse_experiment_manifest.jsonl" \
    --output-root "$ROOT/ts_rag_outputs/timefuse_matrix_full" \
    --summary "$PRIMARY" \
    --source-revision "$METHOD_REVISION" \
    --require-publication-scope

"$PYTHON" scripts/summarize_timefuse_confirmation.py \
    --manifest "$SOURCE/docs/timefuse_experiment_manifest.jsonl" \
    --matrix-summary "$PRIMARY" \
    --protocol "$SOURCE/docs/timefuse_confirmation_protocol.json" \
    --output "$CONFIRMATION"

"$PYTHON" scripts/index_matrix_checkpoints.py \
    "$PRIMARY" \
    --project-root "$ROOT" \
    --manifest "$SOURCE/docs/timefuse_checkpoint_replay_manifest.jsonl" \
    --stage-root checkpoints/replay \
    --checkpoint-path-root checkpoints/replay \
    --output "$CATALOG" \
    --hash

"$PYTHON" - "$OPS" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

ops = Path(sys.argv[1])
names = (
    "timefuse_matrix_publication_summary.json",
    "timefuse_confirmation_summary.json",
    "timefuse_matrix_checkpoint_catalog.json",
)
artifacts = {}
for name in names:
    path = ops / name
    artifacts[name] = {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
primary = json.loads((ops / names[0]).read_text(encoding="utf-8"))
confirmation = json.loads((ops / names[1]).read_text(encoding="utf-8"))
catalog = json.loads((ops / names[2]).read_text(encoding="utf-8"))
receipt = {
    "schema_version": 1,
    "artifacts": artifacts,
    "primary_counts": primary["counts"],
    "primary_gate_passed": primary["publication_gate"][
        "development_gate_passed"
    ],
    "confirmation_gate_passed": confirmation["confirmatory_gate"][
        "confirmatory_gate_passed"
    ],
    "checkpoint_count": catalog["checkpoint_count"],
    "usable_checkpoint_count": catalog["usable_count"],
}
temporary = ops / "finalization_receipt.json.tmp"
temporary.write_text(
    json.dumps(receipt, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
temporary.replace(ops / "finalization_receipt.json")
print(json.dumps(receipt, sort_keys=True))
PY
REMOTE

receipt="$(
    "${SSH[@]}" "jq -c . '$REMOTE_OPS/finalization_receipt.json'"
)"
event "COMPLETE revision=$REVISION remote_ops=$REMOTE_OPS receipt=$receipt"
failed=0
