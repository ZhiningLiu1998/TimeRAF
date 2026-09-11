#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REVISION="${TIMERAF_RECOVERY_REVISION:?set TIMERAF_RECOVERY_REVISION}"
METHOD_REVISION="${TIMERAF_METHOD_REVISION:-d9be338}"
export AWS_PROFILE="${TIMERAF_AWS_PROFILE:-timeraf-modeldev}"
REGION="${TIMERAF_REGION:-us-east-1}"
DOMAIN_ID="${TIMERAF_DOMAIN_ID:-<SAGEMAKER_DOMAIN_ID>}"
SPACE_NAME="${TIMERAF_SPACE_NAME:-<DEV_SPACE>}"
SPACE_ARN="arn:aws:sagemaker:${REGION}:<AWS_ACCOUNT_ID>:space/${DOMAIN_ID}/${SPACE_NAME}"
CONNECT_SCRIPT="$REPO_ROOT/scripts/sagemaker_connect.sh"
KNOWN_HOSTS="/tmp/timeraf-sagemaker-known-hosts"
LOG="/tmp/timeraf-numerical-recovery-worker.log"
EVENTS="/tmp/timeraf-numerical-recovery-events.log"
REMOTE_ROOT="/home/sagemaker-user/user-default-efs/workspace/TimeRAF"
REMOTE_OPS="$REMOTE_ROOT/operations/numerical-recovery-$REVISION"
REMOTE_SOURCE="$REMOTE_OPS/source"
POLL_SECONDS="${TIMERAF_RECOVERY_POLL_SECONDS:-90}"
MAX_CONTROL_RECONNECTS="${TIMERAF_RECOVERY_MAX_CONTROL_RECONNECTS:-16}"

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
test "$(git -C "$REPO_ROOT" rev-parse "$REVISION^{commit}")" = "$REVISION"
log "worker started revision=$REVISION method_revision=$METHOD_REVISION"

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
    printf raw-matrix-incomplete
    exit 0
}
if pgrep -f '[s]cripts/run_timefuse_numerical_recovery.py' >/dev/null ||
    pgrep -f '[s]cripts/run_benchmark_matrix.py' >/dev/null; then
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
    log "waiting for primary matrix: ${readiness:-ssh-error}"
    sleep "$POLL_SECONDS"
done

log "streaming pinned recovery source to EFS"
git -C "$REPO_ROOT" archive "$REVISION" |
    "${SSH[@]}" "
set -eu
ROOT='$REMOTE_ROOT'
OPS='$REMOTE_OPS'
test \"\$(findmnt -T \"\$ROOT\" -o FSTYPE -n)\" = nfs4
mkdir -p \"\$OPS/source\"
tar -xf - -C \"\$OPS/source\"
printf '%s\n' '$REVISION' >\"\$OPS/source_revision\"
" >>"$LOG" 2>&1

profile_runner_state() {
    local profile="$1"
    local state
    state="$(
        "${SSH[@]}" "
ROOT='$REMOTE_ROOT'
BASE=\"\$ROOT/outputs/numerical_recovery/a10g/$REVISION/$profile\"
LOCK=\"\$BASE/matrix.lock\"
if ps -eo comm=,args= | awk '
    \$1 ~ /^python/ && (
        (
            index(\$0, \"scripts/run_timefuse_numerical_recovery.py\")
            && index(\$0, \"--profile $profile\")
            && index(\$0, \"--recovery-revision $REVISION\")
        )
        || (
            index(\$0, \"scripts/run_benchmark_matrix.py\")
            && index(\$0, \"/a10g/$REVISION/$profile\")
        )
    ) {
        found = 1
    }
    END {
        exit !found
    }
'; then
    printf profile-active
elif test -e \"\$LOCK\" && ! flock -n \"\$LOCK\" -c true; then
    printf profile-locked
else
    printf idle
fi
" 2>>"$LOG" || true
    )"
    printf '%s\n' "${state:-ssh-error}"
}

run_profile() {
    local profile="$1"
    local output
    local marker
    local reconnects=0
    local ssh_rc
    local state
    while true; do
        ssh_rc=0
        output="$(
            "${SSH[@]}" "
set -uo pipefail
ROOT='$REMOTE_ROOT'
SOURCE='$REMOTE_SOURCE'
PY=\"\$ROOT/.venv-gpu312/bin/python\"
SUMMARY=\"\$ROOT/outputs/numerical_recovery/a10g/$REVISION/$profile/matrix_summary.json\"
resume_args=()
if [[ '$profile' == exact ]] && jq -e '
    .counts.expected == 9
    and ((.counts.completed + .counts.failed) == 9)
    and .counts.pending == 0
    and .counts.running == 0
    and .counts.incomplete == 0
' \"\$SUMMARY\" >/dev/null 2>&1; then
    resume_args=(--dry-run)
fi
export PYTHONPATH=\"\$SOURCE\"
cd \"\$SOURCE\"
\"\$PY\" scripts/run_timefuse_numerical_recovery.py \
    --source-root \"\$SOURCE\" \
    --artifact-root \"\$ROOT\" \
    --profile '$profile' \
    --parent-run-id a10g-primary-d9be338 \
    --recovery-revision '$REVISION' \
    --materialized-revision \"\$(cat '$REMOTE_OPS/source_revision')\" \
    \"\${resume_args[@]}\"
rc=\$?
if [[ '$profile' == exact ]]; then
    jq -e '
        .counts.expected == 9
        and ((.counts.completed + .counts.failed) == 9)
        and .counts.pending == 0
        and .counts.running == 0
        and .counts.incomplete == 0
        and all(
            .cell_states[];
            .state != \"failed\"
            or .error_type == \"NumericalIntegrityError\"
        )
    ' \"\$SUMMARY\" >/dev/null 2>&1 || rc=5
else
    jq -e '
        .all_completed == true
        and .counts.expected == 9
        and .counts.completed == 9
        and .counts.failed == 0
        and .counts.pending == 0
        and .counts.running == 0
        and .counts.incomplete == 0
    ' \"\$SUMMARY\" >/dev/null 2>&1 || rc=5
fi
printf '%s%s\n' '__TIMERAF_RECOVERY_RC__=' \"\$rc\"
" 2>>"$LOG"
        )" || ssh_rc=$?
        printf '%s\n' "$output" >>"$LOG"
        marker="$(
            tail -n 1 <<<"$output" |
                sed -n 's/^__TIMERAF_RECOVERY_RC__=//p'
        )"
        if [[ "$marker" =~ ^[0-9]+$ ]]; then
            printf '%s\n' "$marker"
            return 0
        fi

        ((reconnects += 1))
        event "PROFILE_CONTROL_CHANNEL_LOST profile=$profile ssh_rc=$ssh_rc reconnect=$reconnects"
        if ((reconnects > MAX_CONTROL_RECONNECTS)); then
            printf '%s\n' control-channel-lost
            return 0
        fi
        while true; do
            state="$(profile_runner_state "$profile")"
            if [[ "$state" == "idle" ]]; then
                event "PROFILE_CONTROL_RECONNECT profile=$profile reconnect=$reconnects"
                break
            fi
            log "waiting after profile control loss: profile=$profile state=$state reconnect=$reconnects"
            sleep "$POLL_SECONDS"
        done
    done
}

exact_rc="$(run_profile exact)"
if [[ "$exact_rc" != "0" && "$exact_rc" != "1" ]]; then
    event "EXACT_FAILED revision=$REVISION rc=$exact_rc"
    exit 3
fi
event "EXACT_FINISHED revision=$REVISION rc=$exact_rc"

fallback_rc="$(run_profile fallback-v1)"
if [[ "$fallback_rc" != "0" ]]; then
    event "FALLBACK_FAILED revision=$REVISION rc=$fallback_rc"
    exit 4
fi
event "FALLBACK_FINISHED revision=$REVISION rc=$fallback_rc"

receipt="$(
    "${SSH[@]}" "
ROOT='$REMOTE_ROOT'
BASE=\"\$ROOT/outputs/numerical_recovery/a10g/$REVISION\"
jq -n \
    --slurpfile exact \"\$BASE/exact/matrix_summary.json\" \
    --slurpfile fallback \"\$BASE/fallback-v1/matrix_summary.json\" \
    '{exact:\$exact[0].counts,fallback:\$fallback[0].counts}'
"
)"
event "COMPLETE revision=$REVISION receipt=$(jq -c . <<<"$receipt")"
failed=0
