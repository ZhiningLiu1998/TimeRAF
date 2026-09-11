#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${TIMERAF_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONTROLLER_REVISION="${TIMERAF_FINALIZER_CONTROLLER_REVISION:?set TIMERAF_FINALIZER_CONTROLLER_REVISION}"
A10G_RECOVERY_REVISION="${TIMERAF_A10G_RECOVERY_REVISION:-cbd4481e94cee8c6ae21edf5258ac29c58eceeea}"
A100_RECOVERY_REVISION="${TIMERAF_A100_RECOVERY_REVISION:-3cf7b3559c88dfa57bf2f62aba4f5cf5ed61fe94}"
METHOD_REVISION="${TIMERAF_METHOD_REVISION:-d9be338}"
PRIMARY_RUN_ID="${TIMERAF_PRIMARY_GREENLAND_RUN_ID:-full-matrix-a100-9974eac-20260731}"
EXACT_RUN_ID="${TIMERAF_EXACT_RUN_ID:-numerical-recovery-exact-3cf7b35-20260801}"
FALLBACK_RUN_ID="${TIMERAF_FALLBACK_RUN_ID:-numerical-recovery-fallback-3cf7b35-20260801}"
export AWS_PROFILE="${TIMERAF_AWS_PROFILE:-timeraf-modeldev}"
REGION="${TIMERAF_REGION:-us-east-1}"
DOMAIN_ID="${TIMERAF_DOMAIN_ID:-<SAGEMAKER_DOMAIN_ID>}"
SPACE_NAME="${TIMERAF_SPACE_NAME:-<DEV_SPACE>}"
SPACE_ARN="arn:aws:sagemaker:${REGION}:<AWS_ACCOUNT_ID>:space/${DOMAIN_ID}/${SPACE_NAME}"
CONNECT_SCRIPT="$REPO_ROOT/scripts/sagemaker_connect.sh"
KNOWN_HOSTS="/tmp/timeraf-sagemaker-known-hosts"
LOG="/tmp/timeraf-recovery-finalization-worker.log"
EVENTS="/tmp/timeraf-recovery-finalization-events.log"
REMOTE_ROOT="/home/sagemaker-user/user-default-efs/workspace/TimeRAF"
REMOTE_OPS="$REMOTE_ROOT/operations/numerical-recovery-finalizer-$CONTROLLER_REVISION"
REMOTE_SOURCE="$REMOTE_OPS/source"
REMOTE_OUTPUT="$REMOTE_ROOT/outputs/numerical_recovery/finalized/$CONTROLLER_REVISION"
REMOTE_PRIMARY="$REMOTE_OUTPUT/a10g_primary_authoritative.json"
REMOTE_A10G="$REMOTE_ROOT/outputs/numerical_recovery/a10g/$A10G_RECOVERY_REVISION"
REMOTE_A100_PRIMARY="$REMOTE_ROOT/outputs/greenland_runs/$PRIMARY_RUN_ID"
REMOTE_A100_RECOVERY="$REMOTE_ROOT/outputs/greenland_recovery_runs"
REMOTE_EXACT="$REMOTE_A100_RECOVERY/$EXACT_RUN_ID"
REMOTE_FALLBACK="$REMOTE_A100_RECOVERY/$FALLBACK_RUN_ID"
REMOTE_CHECKPOINT_ROOT="$REMOTE_ROOT/checkpoints/replay"
REMOTE_STAGE="$REMOTE_CHECKPOINT_ROOT/staging/$CONTROLLER_REVISION"
POLL_SECONDS="${TIMERAF_FINALIZER_POLL_SECONDS:-60}"
PREFLIGHT_ONLY="${TIMERAF_FINALIZER_PREFLIGHT_ONLY:-0}"
REMOTE_MARKER="__TIMERAF_RECOVERY_FINALIZER_OK__"

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
            event "STOPPED controller_revision=$CONTROLLER_REVISION rc=$rc"
        else
            event "FAILED controller_revision=$CONTROLLER_REVISION rc=$rc"
        fi
    fi
}
trap finish EXIT

git -C "$REPO_ROOT" cat-file -e "$CONTROLLER_REVISION^{commit}"
test "$(git -C "$REPO_ROOT" rev-parse "$CONTROLLER_REVISION^{commit}")" = \
    "$CONTROLLER_REVISION"
case "$PREFLIGHT_ONLY" in
    0 | 1) ;;
    *) printf 'TIMERAF_FINALIZER_PREFLIGHT_ONLY must be 0 or 1\n' >&2; exit 2 ;;
esac
worker_sha="$(
    git -C "$REPO_ROOT" show \
        "$CONTROLLER_REVISION:scripts/timefuse_recovery_finalization_worker.sh" |
        sha256sum |
        cut -d' ' -f1
)"
test "$worker_sha" = "$(sha256sum "${BASH_SOURCE[0]}" | cut -d' ' -f1)"
log "worker started controller_revision=$CONTROLLER_REVISION a10g_recovery_revision=$A10G_RECOVERY_REVISION a100_recovery_revision=$A100_RECOVERY_REVISION"

while true; do
    materialized="$(
        git -C "$REPO_ROOT" archive "$CONTROLLER_REVISION" |
            "${SSH[@]}" "
set -eu
ROOT='$REMOTE_ROOT'
OPS='$REMOTE_OPS'
test \"\$(findmnt -T \"\$ROOT\" -o FSTYPE -n)\" = nfs4
mkdir -p \"\$OPS/source\"
tar -xf - -C \"\$OPS/source\"
printf '%s\n' '$CONTROLLER_REVISION' >\"\$OPS/source_revision\"
printf '%s\n' '$REMOTE_MARKER'
" 2>>"$LOG" || true
    )"
    if [[ "$(tail -n 1 <<<"$materialized")" == "$REMOTE_MARKER" ]]; then
        break
    fi
    log "waiting to materialize finalizer source in canonical EFS"
    sleep "$POLL_SECONDS"
done

while true; do
    readiness="$(
        "${SSH[@]}" "
set -u
ROOT='$REMOTE_ROOT'
A10G='$REMOTE_A10G'
A100_PRIMARY='$REMOTE_A100_PRIMARY'
EXACT='$REMOTE_EXACT'
FALLBACK='$REMOTE_FALLBACK'
LOCK=\"\$ROOT/ts_rag_outputs/timefuse_matrix_full/matrix.lock\"
test \"\$(findmnt -T \"\$ROOT\" -o FSTYPE -n)\" = nfs4 || {
    printf storage-not-efs
    exit 0
}
if pgrep -f '[s]cripts/run_benchmark_matrix.py' >/dev/null; then
    printf a10g-primary-runner-active
    exit 0
fi
flock -n \"\$LOCK\" -c true || {
    printf a10g-primary-locked
    exit 0
}
test \"\$(jq -r '.counts.completed' \"\$ROOT/ts_rag_outputs/timefuse_matrix_full/matrix_d9be338_summary.json\" 2>/dev/null)\" = 585 || {
    printf a10g-primary-incomplete
    exit 0
}
for path in \
    \"\$A100_PRIMARY/fetch_receipt.json\" \
    \"\$A100_PRIMARY/recomputed_matrix_summary.json\" \
    \"\$A100_PRIMARY/final_status.json\" \
    \"\$A10G/exact/matrix_summary.json\" \
    \"\$A10G/exact/recovery_metadata.json\" \
    \"\$A10G/fallback-v1/matrix_summary.json\" \
    \"\$A10G/fallback-v1/recovery_metadata.json\" \
    \"\$EXACT/fetch_receipt.json\" \
    \"\$EXACT/recomputed_matrix_summary.json\" \
    \"\$FALLBACK/fetch_receipt.json\" \
    \"\$FALLBACK/recomputed_matrix_summary.json\"; do
    test -s \"\$path\" || {
        printf 'missing-%s' \"\${path##*/}\"
        exit 0
    }
done
jq -e --slurpfile summary \"\$A100_PRIMARY/recomputed_matrix_summary.json\" '
    .schema_version == 1
    and .job_kind == \"full_matrix_replication\"
    and .topology_gate_passed == true
    and .import_acceptance.mode == \"terminal-nonfinite-first-pass\"
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
' \"\$A100_PRIMARY/fetch_receipt.json\" >/dev/null 2>&1 || {
    printf a100-primary-import-invalid
    exit 0
}
test \"\$(jq -r '.recomputed_matrix_summary.sha256' \"\$A100_PRIMARY/fetch_receipt.json\")\" = \
    \"\$(sha256sum \"\$A100_PRIMARY/recomputed_matrix_summary.json\" | cut -d' ' -f1)\" || {
    printf a100-primary-summary-drift
    exit 0
}
test \"\$(jq -r '.recomputed_matrix_summary.size_bytes' \"\$A100_PRIMARY/fetch_receipt.json\")\" = \
    \"\$(stat -c %s \"\$A100_PRIMARY/recomputed_matrix_summary.json\")\" || {
    printf a100-primary-summary-drift
    exit 0
}
jq -e '
    .matrix_counts.expected == 9
    and ((.matrix_counts.completed + .matrix_counts.failed) == 9)
    and .matrix_counts.pending == 0
    and .matrix_counts.running == 0
    and .matrix_counts.incomplete == 0
    and .topology_gate_passed == true
    and .source_revision == \"$A100_RECOVERY_REVISION\"
    and .recovery_revision == \"$A100_RECOVERY_REVISION\"
' \"\$EXACT/fetch_receipt.json\" >/dev/null 2>&1 || {
    printf a100-exact-import-invalid
    exit 0
}
jq -e '
    .matrix_counts.expected == 9
    and .matrix_counts.completed == 9
    and .matrix_counts.failed == 0
    and .matrix_counts.pending == 0
    and .matrix_counts.running == 0
    and .matrix_counts.incomplete == 0
    and .topology_gate_passed == true
    and .source_revision == \"$A100_RECOVERY_REVISION\"
    and .recovery_revision == \"$A100_RECOVERY_REVISION\"
' \"\$FALLBACK/fetch_receipt.json\" >/dev/null 2>&1 || {
    printf a100-fallback-import-invalid
    exit 0
}
printf ready
" 2>>"$LOG" || true
    )"
    if [[ "$PREFLIGHT_ONLY" == "1" ]]; then
        event "PREFLIGHT_ONLY readiness=${readiness:-ssh-error} controller_revision=$CONTROLLER_REVISION"
        failed=0
        exit 0
    fi
    if [[ "$readiness" == "ready" ]]; then
        break
    fi
    log "waiting for finalization prerequisites: ${readiness:-ssh-error}"
    sleep "$POLL_SECONDS"
done
event "PREREQUISITES_READY controller_revision=$CONTROLLER_REVISION"

finalized="$(
    "${SSH[@]}" "
set -euo pipefail
ROOT='$REMOTE_ROOT'
SOURCE='$REMOTE_SOURCE'
OUTPUT='$REMOTE_OUTPUT'
PRIMARY='$REMOTE_PRIMARY'
PY=\"\$ROOT/.venv-gpu312/bin/python\"
RECEIPT=\"\$OUTPUT/finalization_receipt.json\"
test \"\$(findmnt -T \"\$ROOT\" -o FSTYPE -n)\" = nfs4
test \"\$(cat '$REMOTE_OPS/source_revision')\" = '$CONTROLLER_REVISION'
test -x \"\$PY\"
mkdir -p \"\$OUTPUT\"
export PYTHONPATH=\"\$SOURCE\"
cd \"\$SOURCE\"

if [[ ! -s \"\$PRIMARY\" ]]; then
    if [[ -e \"\$RECEIPT\" ]]; then
        printf '%s\n' 'finalization receipt exists without primary input' >&2
        exit 3
    fi
    \"\$PY\" scripts/summarize_timefuse_matrix.py \
        --manifest \"\$SOURCE/docs/timefuse_experiment_manifest.jsonl\" \
        --output-root \"\$ROOT/ts_rag_outputs/timefuse_matrix_full\" \
        --summary \"\$PRIMARY\" \
        --source-revision '$METHOD_REVISION' \
        --require-publication-scope
fi
jq -e '
    .source_revision == \"$METHOD_REVISION\"
    and .counts.expected == 585
    and ((.counts.completed + .counts.incomplete) == 585)
    and .counts.failed == 0
    and .counts.pending == 0
    and .counts.running == 0
    and (.cell_states | length) == 585
' \"\$PRIMARY\" >/dev/null

\"\$PY\" scripts/finalize_timefuse_numerical_recovery.py \
    --project-root \"\$ROOT\" \
    --source-root \"\$SOURCE\" \
    --python \"\$PY\" \
    --controller-revision '$CONTROLLER_REVISION' \
    --manifest \"\$SOURCE/docs/timefuse_experiment_manifest.jsonl\" \
    --replay-manifest \"\$SOURCE/docs/timefuse_checkpoint_replay_manifest.jsonl\" \
    --confirmation-protocol \"\$SOURCE/docs/timefuse_confirmation_protocol.json\" \
    --protocol \"\$SOURCE/docs/timefuse_numerical_recovery_protocol.json\" \
    --a10g-primary \"\$PRIMARY\" \
    --a100-primary '$REMOTE_A100_PRIMARY/recomputed_matrix_summary.json' \
    --a10g-exact '$REMOTE_A10G/exact/matrix_summary.json' \
    --a100-exact '$REMOTE_EXACT/recomputed_matrix_summary.json' \
    --a10g-fallback '$REMOTE_A10G/fallback-v1/matrix_summary.json' \
    --a100-fallback '$REMOTE_FALLBACK/recomputed_matrix_summary.json' \
    --a10g-exact-metadata '$REMOTE_A10G/exact/recovery_metadata.json' \
    --a10g-fallback-metadata '$REMOTE_A10G/fallback-v1/recovery_metadata.json' \
    --a100-exact-receipt '$REMOTE_EXACT/fetch_receipt.json' \
    --a100-fallback-receipt '$REMOTE_FALLBACK/fetch_receipt.json' \
    --a100-final-status '$REMOTE_A100_PRIMARY/final_status.json' \
    --output-root \"\$OUTPUT\" \
    --checkpoint-stage-root '$REMOTE_STAGE' \
    --checkpoint-path-root '$REMOTE_CHECKPOINT_ROOT' \
    --required-filesystem nfs4 \
    >\"\$OUTPUT/finalizer.stdout.json.tmp\"
mv \"\$OUTPUT/finalizer.stdout.json.tmp\" \"\$OUTPUT/finalizer.stdout.json\"
jq -e '
    .schema_version == 1
    and .controller_revision == \"$CONTROLLER_REVISION\"
    and .method_revision == \"$METHOD_REVISION\"
    and .evidence.checkpoint_count == 208
    and (.artifacts.checkpoint_catalog.sha256 | test(\"^[0-9a-f]{64}$\"))
' \"\$RECEIPT\" >/dev/null
catalog=\"\$(jq -r '.artifacts.checkpoint_catalog.path' \"\$RECEIPT\")\"
test \"\$(jq -r '.artifacts.checkpoint_catalog.sha256' \"\$RECEIPT\")\" = \
    \"\$(sha256sum \"\$catalog\" | cut -d' ' -f1)\"
jq -c '{
    controller_revision,
    method_revision,
    evidence,
    checkpoint_catalog:.artifacts.checkpoint_catalog,
    finalization_receipt:\"'$REMOTE_OUTPUT'/finalization_receipt.json\"
}' \"\$RECEIPT\"
printf '%s\n' '$REMOTE_MARKER'
" 2>>"$LOG" || true
)"
if [[ "$(tail -n 1 <<<"$finalized")" != "$REMOTE_MARKER" ]]; then
    event "FINALIZATION_FAILED controller_revision=$CONTROLLER_REVISION"
    exit 4
fi
receipt="$(tail -n 2 <<<"$finalized" | head -n 1)"
if ! jq -e '.evidence.checkpoint_count == 208' >/dev/null <<<"$receipt"; then
    event "FINALIZATION_RECEIPT_FAILED controller_revision=$CONTROLLER_REVISION"
    exit 5
fi
event "CATALOG_READY controller_revision=$CONTROLLER_REVISION receipt=$receipt"
event "COMPLETE controller_revision=$CONTROLLER_REVISION output=$REMOTE_OUTPUT"
failed=0
