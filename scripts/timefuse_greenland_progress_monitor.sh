#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
JOB_UUID="${TIMERAF_GREENLAND_JOB_UUID:?set TIMERAF_GREENLAND_JOB_UUID}"
JOB_NAME="${TIMERAF_GREENLAND_JOB_NAME:?set TIMERAF_GREENLAND_JOB_NAME}"
SUBMITTED_AT="${TIMERAF_GREENLAND_SUBMITTED_AT:?set TIMERAF_GREENLAND_SUBMITTED_AT}"
EXPECTED_CELLS="${TIMERAF_GREENLAND_EXPECTED_CELLS:-585}"
POLL_SECONDS="${TIMERAF_GREENLAND_PROGRESS_POLL_SECONDS:-180}"
PROBE_TIMEOUT_SECONDS="${TIMERAF_GREENLAND_PROGRESS_TIMEOUT_SECONDS:-180}"
MAX_ACTIVITY_AGE_SECONDS="${TIMERAF_GREENLAND_MAX_ACTIVITY_AGE_SECONDS:-1800}"
PYTHON="${TIMERAF_LOCAL_PYTHON:-$REPO_ROOT/.venv/bin/python}"
STATUS="${TIMERAF_GREENLAND_PROGRESS_STATUS:-/tmp/timeraf-greenland-current-status.json}"
PROBE_LOG="${TIMERAF_GREENLAND_PROGRESS_PROBE_LOG:-/tmp/timeraf-greenland-progress-probe.log}"
LOG="${TIMERAF_GREENLAND_PROGRESS_LOG:-/tmp/timeraf-greenland-progress-monitor.log}"
EVENTS="${TIMERAF_GREENLAND_PROGRESS_EVENTS:-/tmp/timeraf-greenland-progress-events.log}"

log() {
    printf '%s %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$*" >>"$LOG"
}

event() {
    local line
    line="$(date -u +'%Y-%m-%dT%H:%M:%SZ') $*"
    printf '%s\n' "$line" >>"$LOG"
    printf '%s\n' "$line" >>"$EVENTS"
}

finished=0
finish() {
    local rc=$?
    if ((finished == 0)); then
        if ((rc == 0)); then
            event "STOPPED job_uuid=$JOB_UUID job_name=$JOB_NAME rc=$rc"
        else
            event "FAILED job_uuid=$JOB_UUID job_name=$JOB_NAME rc=$rc"
        fi
    fi
}
trap finish EXIT

test -x "$PYTHON"
[[ "$EXPECTED_CELLS" =~ ^[1-9][0-9]*$ ]]
[[ "$POLL_SECONDS" =~ ^[1-9][0-9]*$ ]]
[[ "$PROBE_TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ ]]
[[ "$MAX_ACTIVITY_AGE_SECONDS" =~ ^[1-9][0-9]*$ ]]

log "monitor started job_uuid=$JOB_UUID job_name=$JOB_NAME expected_cells=$EXPECTED_CELLS interval=${POLL_SECONDS}s"
last_error_total=-1
activity_alerted=0

while true; do
    if ! "$PYTHON" "$REPO_ROOT/scripts/verify_greenland_startup.py" \
        --job-uuid "$JOB_UUID" \
        --job-name "$JOB_NAME" \
        --submitted-at "$SUBMITTED_AT" \
        --poll-seconds 1 \
        --timeout-seconds "$PROBE_TIMEOUT_SECONDS" \
        --output "$STATUS" \
        >"$PROBE_LOG" 2>&1; then
        log "probe_failed details=$(tail -n 1 "$PROBE_LOG" 2>/dev/null || true)"
        sleep "$POLL_SECONDS"
        continue
    fi

    if ! jq -e \
        --arg job_uuid "$JOB_UUID" \
        --arg job_name "$JOB_NAME" \
        --argjson expected "$EXPECTED_CELLS" '
        .schema_version == 1
        and .job_uuid == $job_uuid
        and .job_name == $job_name
        and .startup_topology_gate_passed == true
        and (.worker_bindings | length) == 8
        and ([.worker_bindings[].worker_gpu] | sort) == [0,1,2,3,4,5,6,7]
        and (.progress.scheduled_cells | type == "number")
        and .progress.scheduled_cells <= $expected
        and (.progress.completed_result_lines | type == "number")
        and .progress.completed_result_lines <= $expected
        and (.progress.error_counts | type == "object")
        and (.activity.latest_log_at | type == "string")
        and (.activity.latest_log_message | type == "string")
        and (.activity.age_seconds | type == "number")
        and .activity.age_seconds >= 0
    ' "$STATUS" >/dev/null; then
        event "ALERT invalid_status job_uuid=$JOB_UUID status=$STATUS"
        sleep "$POLL_SECONDS"
        continue
    fi

    snapshot="$(
        jq -c \
            '{observed_at,progress,activity,startup_topology_gate_passed}' \
            "$STATUS"
    )"
    log "$snapshot"
    completed="$(jq -r '.progress.completed_result_lines' "$STATUS")"
    error_total="$(
        jq '[.progress.error_counts[]] | add // 0' "$STATUS"
    )"
    if ((error_total > 0 && error_total != last_error_total)); then
        event "ALERT runtime_errors=$error_total snapshot=$snapshot"
    fi
    last_error_total="$error_total"

    activity_age="$(
        jq -r '.activity.age_seconds | floor' "$STATUS"
    )"
    if ((completed < EXPECTED_CELLS &&
        activity_age > MAX_ACTIVITY_AGE_SECONDS)); then
        if ((activity_alerted == 0)); then
            event "ALERT stale_log_activity age_seconds=$activity_age \
threshold_seconds=$MAX_ACTIVITY_AGE_SECONDS snapshot=$snapshot"
        fi
        activity_alerted=1
    else
        if ((activity_alerted == 1)); then
            event "RECOVERY log_activity_resumed age_seconds=$activity_age \
snapshot=$snapshot"
        fi
        activity_alerted=0
    fi

    if ((completed >= EXPECTED_CELLS)); then
        event "COMPLETE job_uuid=$JOB_UUID completed=$completed snapshot=$snapshot"
        finished=1
        exit 0
    fi
    sleep "$POLL_SECONDS"
done
