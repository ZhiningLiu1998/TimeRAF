#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LIFECYCLE_REVISION="${TIMERAF_LIFECYCLE_REVISION:?set TIMERAF_LIFECYCLE_REVISION}"
A10G_REVISION="${TIMERAF_A10G_RECOVERY_REVISION:?set TIMERAF_A10G_RECOVERY_REVISION}"
A100_REVISION="${TIMERAF_A100_RECOVERY_REVISION:-3cf7b3559c88dfa57bf2f62aba4f5cf5ed61fe94}"
CONTROLLER_REVISION="${TIMERAF_RECOVERY_CONTROLLER_REVISION:-f6350bb2dbc6691e640fb0f074aa2f7864784c1d}"
METHOD_REVISION="${TIMERAF_METHOD_REVISION:-d9be338}"
EXACT_PREPARATION_SHA="${TIMERAF_EXACT_PREPARATION_SHA256:?set TIMERAF_EXACT_PREPARATION_SHA256}"
FALLBACK_PREPARATION_SHA="${TIMERAF_FALLBACK_PREPARATION_SHA256:?set TIMERAF_FALLBACK_PREPARATION_SHA256}"
DRY_RUN_AUDIT_SHA="${TIMERAF_DRY_RUN_AUDIT_SHA256:?set TIMERAF_DRY_RUN_AUDIT_SHA256}"
PREDECESSOR_PID="${TIMERAF_RECOVERY_PREDECESSOR_PID:-}"
CONTROLLER_ROOT="/tmp/timeraf-recovery-controller-$CONTROLLER_REVISION"
LOG="/tmp/timeraf-recovery-pipeline-worker.log"

log() {
    printf '%s %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$*" >>"$LOG"
}

failed=1
finish() {
    local rc=$?
    if ((failed)); then
        log "FAILED lifecycle_revision=$LIFECYCLE_REVISION rc=$rc"
    fi
}
trap finish EXIT

git -C "$REPO_ROOT" cat-file -e "$LIFECYCLE_REVISION^{commit}"
test "$(git -C "$REPO_ROOT" rev-parse "$LIFECYCLE_REVISION^{commit}")" = \
    "$LIFECYCLE_REVISION"
git -C "$REPO_ROOT" diff --quiet "$LIFECYCLE_REVISION" -- \
    scripts/timefuse_numerical_recovery_worker.sh \
    scripts/timefuse_recovery_pipeline_worker.sh
mkdir -p "$CONTROLLER_ROOT"
git -C "$REPO_ROOT" archive "$CONTROLLER_REVISION" \
    scripts/timefuse_greenland_recovery_worker.sh \
    scripts/timefuse_recovery_finalization_worker.sh |
    tar -xf - -C "$CONTROLLER_ROOT"

if [[ -n "$PREDECESSOR_PID" ]]; then
    [[ "$PREDECESSOR_PID" =~ ^[1-9][0-9]*$ ]]
    log "waiting for predecessor_pid=$PREDECESSOR_PID"
    while kill -0 "$PREDECESSOR_PID" 2>/dev/null; do
        sleep 30
    done
    log "predecessor exited pid=$PREDECESSOR_PID"
fi

test -f "$CONTROLLER_ROOT/scripts/timefuse_greenland_recovery_worker.sh"
test -f "$CONTROLLER_ROOT/scripts/timefuse_recovery_finalization_worker.sh"
log "starting A10G recovery revision=$A10G_REVISION method_revision=$METHOD_REVISION"

export TIMERAF_RECOVERY_REVISION="$A10G_REVISION"
export TIMERAF_METHOD_REVISION="$METHOD_REVISION"
export TIMERAF_AWS_PROFILE="${TIMERAF_AWS_PROFILE:-timeraf-modeldev}"
export TIMERAF_GREENLAND_LOG_PROFILE="${TIMERAF_GREENLAND_LOG_PROFILE:-timeraf-greenland-console}"
bash "$REPO_ROOT/scripts/timefuse_numerical_recovery_worker.sh"

log "starting Greenland recovery revision=$A100_REVISION controller=$CONTROLLER_REVISION"
env \
    TIMERAF_RECOVERY_CONTROLLER_REVISION="$CONTROLLER_REVISION" \
    TIMERAF_GREENLAND_RECOVERY_REVISION="$A100_REVISION" \
    TIMERAF_A10G_RECOVERY_REVISION="$A10G_REVISION" \
    TIMERAF_EXACT_PREPARATION_SHA256="$EXACT_PREPARATION_SHA" \
    TIMERAF_FALLBACK_PREPARATION_SHA256="$FALLBACK_PREPARATION_SHA" \
    TIMERAF_DRY_RUN_AUDIT_SHA256="$DRY_RUN_AUDIT_SHA" \
    TIMERAF_RECOVERY_PREFLIGHT_ONLY=0 \
    TIMERAF_REPO_ROOT="$REPO_ROOT" \
    bash "$CONTROLLER_ROOT/scripts/timefuse_greenland_recovery_worker.sh"

log "starting recovery finalization controller=$CONTROLLER_REVISION"
env \
    TIMERAF_FINALIZER_CONTROLLER_REVISION="$CONTROLLER_REVISION" \
    TIMERAF_FINALIZER_PREFLIGHT_ONLY=0 \
    TIMERAF_REPO_ROOT="$REPO_ROOT" \
    TIMERAF_A10G_RECOVERY_REVISION="$A10G_REVISION" \
    TIMERAF_A100_RECOVERY_REVISION="$A100_REVISION" \
    bash "$CONTROLLER_ROOT/scripts/timefuse_recovery_finalization_worker.sh"

failed=0
log "COMPLETE lifecycle_revision=$LIFECYCLE_REVISION"
