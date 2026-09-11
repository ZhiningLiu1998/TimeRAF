#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ID="${TIMERAF_GREENLAND_RUN_ID:?set TIMERAF_GREENLAND_RUN_ID}"
REVISION="${TIMERAF_GREENLAND_REVISION:-$(git -C "$REPO_ROOT" rev-parse HEAD)}"
CONTROLLER_REVISION="$(
    git -C "$REPO_ROOT" rev-parse \
        "${TIMERAF_GREENLAND_CONTROLLER_REVISION:-HEAD}^{commit}"
)"
REVISION="$(git -C "$REPO_ROOT" rev-parse "$REVISION^{commit}")"
METHOD_REVISION="${TIMERAF_METHOD_REVISION:-d9be338}"
REGION="${TIMERAF_REGION:-us-east-1}"
export AWS_PROFILE="${TIMERAF_AWS_PROFILE:-timeraf-modeldev}"
DOMAIN_ID="${TIMERAF_DOMAIN_ID:-<SAGEMAKER_DOMAIN_ID>}"
SPACE_NAME="${TIMERAF_SPACE_NAME:-<DEV_SPACE>}"
SPACE_ARN="arn:aws:sagemaker:${REGION}:<AWS_ACCOUNT_ID>:space/${DOMAIN_ID}/${SPACE_NAME}"
CONNECT_SCRIPT="$REPO_ROOT/scripts/sagemaker_connect.sh"
KNOWN_HOSTS="/tmp/timeraf-sagemaker-known-hosts"
LOG="/tmp/timeraf-greenland-replication-${RUN_ID}.log"
EVENTS="/tmp/timeraf-greenland-replication-events.log"
REMOTE_ROOT="/home/sagemaker-user/user-default-efs/workspace/TimeRAF"
REMOTE_OPS="$REMOTE_ROOT/operations/greenland-replication-$REVISION"
REMOTE_SOURCE="$REMOTE_OPS/source"
REMOTE_CONTROLLER_OPS="$REMOTE_ROOT/operations/greenland-replication-controller-$CONTROLLER_REVISION"
REMOTE_CONTROLLER_SOURCE="$REMOTE_CONTROLLER_OPS/source"
DESTINATION="$REMOTE_ROOT/outputs/greenland_runs/$RUN_ID"
COMPARISON="$REMOTE_ROOT/outputs/greenland_comparisons/$RUN_ID"
POLL_SECONDS="${TIMERAF_GREENLAND_POLL_SECONDS:-60}"
POST_IMPORT_ACTION="${TIMERAF_GREENLAND_POST_IMPORT_ACTION:-compare}"
IMPORT_MODE_ARG=""
REMOTE_MARKER="__TIMERAF_REMOTE_OK_${RUN_ID//-/_}__"

case "$POST_IMPORT_ACTION" in
    compare) ;;
    import-only)
        IMPORT_MODE_ARG="--accept-terminal-nonfinite-first-pass"
        ;;
    *)
        printf 'unsupported TIMERAF_GREENLAND_POST_IMPORT_ACTION: %s\n' \
            "$POST_IMPORT_ACTION" >&2
        exit 2
        ;;
esac

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
            event "STOPPED run_id=$RUN_ID revision=$REVISION rc=$rc"
        else
            event "FAILED run_id=$RUN_ID revision=$REVISION rc=$rc"
        fi
    fi
}
trap finish EXIT

git -C "$REPO_ROOT" cat-file -e "$CONTROLLER_REVISION^{commit}"
log "worker started run_id=$RUN_ID revision=$REVISION controller_revision=$CONTROLLER_REVISION method_revision=$METHOD_REVISION post_import_action=$POST_IMPORT_ACTION"

while true; do
    materialized="$(
        git -C "$REPO_ROOT" archive "$REVISION" |
            "${SSH[@]}" "
set -eu
ROOT='$REMOTE_ROOT'
OPS='$REMOTE_OPS'
test \"\$(findmnt -T \"\$ROOT\" -o FSTYPE -n)\" = nfs4
mkdir -p \"\$OPS/source\"
tar -xf - -C \"\$OPS/source\"
printf '%s\n' '$REVISION' >\"\$OPS/source_revision\"
printf '%s\n' '$REMOTE_MARKER'
" 2>>"$LOG" || true
    )"
    if [[ "$(tail -n 1 <<<"$materialized")" == "$REMOTE_MARKER" ]]; then
        break
    fi
    log "waiting to materialize pinned source in canonical EFS"
    sleep "$POLL_SECONDS"
done

while true; do
    controller_materialized="$(
        git -C "$REPO_ROOT" archive "$CONTROLLER_REVISION" |
            "${SSH[@]}" "
set -eu
ROOT='$REMOTE_ROOT'
OPS='$REMOTE_CONTROLLER_OPS'
test \"\$(findmnt -T \"\$ROOT\" -o FSTYPE -n)\" = nfs4
mkdir -p \"\$OPS/source\"
tar -xf - -C \"\$OPS/source\"
printf '%s\n' '$CONTROLLER_REVISION' >\"\$OPS/source_revision\"
printf '%s\n' '$REMOTE_MARKER'
" 2>>"$LOG" || true
    )"
    if [[ "$(tail -n 1 <<<"$controller_materialized")" == "$REMOTE_MARKER" ]]; then
        break
    fi
    log "waiting to materialize controller source in canonical EFS"
    sleep "$POLL_SECONDS"
done

while true; do
    status="$(
        "${SSH[@]}" "
set -eu
'$REMOTE_ROOT/.venv-greenland-control/bin/python' \
    '$REMOTE_SOURCE/scripts/greenland_run_status.py' \
    --run-id '$RUN_ID' \
    --region '$REGION' \
    --expected-job-kind full_matrix_replication \
    --expected-method-revision '$METHOD_REVISION' \
    --expected-source-revision '$REVISION' \
    --expected-launcher-revision '$REVISION'
" 2>>"$LOG" || true
    )"
    if [[ -z "$status" ]] || ! jq -e --arg run_id "$RUN_ID" '
        type == "object"
        and .run_id == $run_id
        and (.upload_complete | type == "boolean")
        and (.upload_failed | type == "boolean")
        and ((.upload_complete and .upload_failed) | not)
    ' >/dev/null 2>&1 <<<"$status"; then
        log "waiting for valid Greenland status JSON"
        sleep "$POLL_SECONDS"
        continue
    fi
    if jq -e '.upload_complete == true' >/dev/null <<<"$status"; then
        event "UPLOAD_COMPLETE run_id=$RUN_ID marker=$(jq -c . <<<"$status")"
        break
    fi
    if jq -e '.upload_failed == true' >/dev/null <<<"$status"; then
        event "UPLOAD_FAILED run_id=$RUN_ID marker=$(jq -c . <<<"$status")"
        exit 3
    fi
    log "waiting for Greenland upload completion"
    sleep "$POLL_SECONDS"
done

log "importing hash-verified Greenland result artifacts into EFS"
imported="$(
    "${SSH[@]}" "
set -euo pipefail
ROOT='$REMOTE_ROOT'
PY=\"\$ROOT/.venv-gpu312/bin/python\"
\"\$PY\" '$REMOTE_CONTROLLER_SOURCE/scripts/fetch_greenland_full_matrix.py' \
    --run-id '$RUN_ID' \
    --project-root \"\$ROOT\" \
    --destination '$DESTINATION' \
    --manifest '$REMOTE_CONTROLLER_SOURCE/docs/timefuse_experiment_manifest.jsonl' \
    --region '$REGION' $IMPORT_MODE_ARG
SUMMARY='$DESTINATION/recomputed_matrix_summary.json'
RECEIPT='$DESTINATION/fetch_receipt.json'
test -s \"\$SUMMARY\"
test \"\$(jq -r '.recomputed_matrix_summary.sha256' \"\$RECEIPT\")\" = \
    \"\$(sha256sum \"\$SUMMARY\" | cut -d' ' -f1)\"
test \"\$(jq -r '.recomputed_matrix_summary.size_bytes' \"\$RECEIPT\")\" = \
    \"\$(stat -c %s \"\$SUMMARY\")\"
if [[ '$POST_IMPORT_ACTION' == import-only ]]; then
    jq -e --slurpfile summary \"\$SUMMARY\" '
        .import_acceptance.mode == \"terminal-nonfinite-first-pass\"
        and .import_acceptance.terminal_attempt_count == 585
        and .matrix_counts.expected == 585
        and ((.matrix_counts.completed + .matrix_counts.incomplete) == 585)
        and .matrix_counts.failed == 0
        and .matrix_counts.pending == 0
        and .matrix_counts.running == 0
        and .matrix_counts == \$summary[0].counts
        and .import_acceptance.nonfinite_cell_ids ==
            ([\$summary[0].cell_states[]
              | select(.state == \"incomplete\")
              | .cell_id] | sort)
    ' \"\$RECEIPT\" >/dev/null
fi
printf '%s\n' '$REMOTE_MARKER'
" 2>>"$LOG" || true
)"
printf '%s\n' "$imported" >>"$LOG"
if [[ "$(tail -n 1 <<<"$imported")" != "$REMOTE_MARKER" ]]; then
    event "IMPORT_FAILED run_id=$RUN_ID destination=$DESTINATION"
    exit 4
fi
summary_sha="$(
    "${SSH[@]}" "jq -r '.recomputed_matrix_summary.sha256' '$DESTINATION/fetch_receipt.json'"
)"
if [[ ! "$summary_sha" =~ ^[0-9a-f]{64}$ ]]; then
    event "IMPORT_RECEIPT_FAILED run_id=$RUN_ID destination=$DESTINATION"
    exit 5
fi
event "IMPORTED run_id=$RUN_ID destination=$DESTINATION controller_revision=$CONTROLLER_REVISION recomputed_summary_sha256=$summary_sha"

if [[ "$POST_IMPORT_ACTION" == "import-only" ]]; then
    event "COMPLETE_IMPORT_ONLY run_id=$RUN_ID destination=$DESTINATION"
    failed=0
    exit 0
fi

while true; do
    readiness="$(
        "${SSH[@]}" "
ROOT='$REMOTE_ROOT'
LOCK=\"\$ROOT/ts_rag_outputs/timefuse_matrix_full/matrix.lock\"
if pgrep -f '[s]cripts/run_benchmark_matrix.py' >/dev/null; then
    printf runner-active
elif ! flock -n \"\$LOCK\" -c true; then
    printf matrix-locked
else
    '$REMOTE_ROOT/.venv-gpu312/bin/python' - \
        '$REMOTE_CONTROLLER_SOURCE' \"\$ROOT\" '$METHOD_REVISION' <<'PY'
import sys
source, root, revision = sys.argv[1:]
sys.path.insert(0, source)
from scripts.compare_timefuse_matrix_runs import discover_completed_results
from ts_rag.matrix import load_manifest
manifest = load_manifest(source + '/docs/timefuse_experiment_manifest.jsonl')
records = discover_completed_results(
    root + '/ts_rag_outputs/timefuse_matrix_full',
    manifest,
    revision,
    2021,
)
print('ready' if len(records) == 585 else f'a10g-completed-{len(records)}')
PY
fi
" 2>>"$LOG" || true
    )"
    if [[ "$readiness" == "ready" ]]; then
        break
    fi
    log "waiting for A10G comparison readiness: ${readiness:-ssh-error}"
    sleep "$POLL_SECONDS"
done

log "building A10G versus A100 consistency and runtime report"
compared="$(
    "${SSH[@]}" "
set -euo pipefail
ROOT='$REMOTE_ROOT'
PY=\"\$ROOT/.venv-gpu312/bin/python\"
mkdir -p '$COMPARISON'
\"\$PY\" '$REMOTE_CONTROLLER_SOURCE/scripts/compare_timefuse_matrix_runs.py' \
    --manifest '$REMOTE_CONTROLLER_SOURCE/docs/timefuse_experiment_manifest.jsonl' \
    --a10g-output-root \"\$ROOT/ts_rag_outputs/timefuse_matrix_full\" \
    --a100-output-root '$DESTINATION/full_matrix_replication' \
    --a100-final-status '$DESTINATION/final_status.json' \
    --method-revision '$METHOD_REVISION' \
    --output '$COMPARISON/comparison.json' \
    --markdown '$COMPARISON/comparison.md'
jq -e '.schema_version == 1' '$COMPARISON/comparison.json' >/dev/null
printf '%s\n' '$REMOTE_MARKER'
" 2>>"$LOG" || true
)"
printf '%s\n' "$compared" >>"$LOG"
if [[ "$(tail -n 1 <<<"$compared")" != "$REMOTE_MARKER" ]]; then
    event "COMPARISON_FAILED run_id=$RUN_ID comparison=$COMPARISON"
    exit 6
fi

receipt="$(
    "${SSH[@]}" "jq -c '{consistency_gates,runtime:{observed_matrix_makespan_speedup:.runtime.observed_matrix_makespan_speedup,both_trained_comparable_subset:.runtime.both_trained_comparable_subset}}' '$COMPARISON/comparison.json'"
)"
if [[ -z "$receipt" ]] || ! jq -e 'type == "object"' >/dev/null <<<"$receipt"; then
    event "COMPARISON_RECEIPT_FAILED run_id=$RUN_ID comparison=$COMPARISON"
    exit 7
fi
event "COMPLETE run_id=$RUN_ID comparison=$COMPARISON receipt=$receipt"
failed=0
