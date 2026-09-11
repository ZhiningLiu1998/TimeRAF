#!/usr/bin/env bash
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export AWS_PROFILE="${TIMERAF_AWS_PROFILE:-timeraf-modeldev}"
REGION="${TIMERAF_REGION:-us-east-1}"
DOMAIN_ID="${TIMERAF_DOMAIN_ID:-<SAGEMAKER_DOMAIN_ID>}"
SPACE_NAME="${TIMERAF_SPACE_NAME:-<DEV_SPACE>}"
APP_TYPE="${TIMERAF_APP_TYPE:-JupyterLab}"
APP_NAME="${TIMERAF_APP_NAME:-default}"
SPACE_ARN="arn:aws:sagemaker:${REGION}:<AWS_ACCOUNT_ID>:space/${DOMAIN_ID}/${SPACE_NAME}"
CONNECT_SCRIPT="$REPO_ROOT/scripts/sagemaker_connect.sh"
KNOWN_HOSTS="${TIMERAF_KNOWN_HOSTS:-/tmp/timeraf-sagemaker-known-hosts}"
LOG="${TIMERAF_MONITOR_LOG:-/tmp/timeraf-matrix-monitor.log}"
EVENTS="${TIMERAF_MONITOR_EVENTS:-/tmp/timeraf-matrix-monitor-events.log}"
REMOTE_ROOT="/home/sagemaker-user/user-default-efs/workspace/TimeRAF"
SUMMARY="$REMOTE_ROOT/ts_rag_outputs/timefuse_matrix_full/matrix_d9be338_summary.json"
EXPECTED_CELLS="${TIMERAF_EXPECTED_CELLS:-585}"
EXPECTED_WORKERS="${TIMERAF_EXPECTED_WORKERS:-4}"
POLL_SECONDS="${TIMERAF_MONITOR_POLL_SECONDS:-90}"
PROGRESS_STEP="${TIMERAF_MONITOR_PROGRESS_STEP:-25}"

baseline_completed="${TIMERAF_MONITOR_BASELINE_COMPLETED:-0}"
next_progress=$((baseline_completed + PROGRESS_STEP))
last_failed=-1
last_app_status=""
last_completed=-1
last_nonfinite_completed=-1
health_bad_streak=0
health_alerted=0
poll_count=0

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
    local line
    line="$(date -u +'%Y-%m-%dT%H:%M:%SZ') $*"
    printf '%s\n' "$line" >>"$LOG"
}

event() {
    local line
    line="$(date -u +'%Y-%m-%dT%H:%M:%SZ') $*"
    printf '%s\n' "$line" >>"$LOG"
    printf '%s\n' "$line" >>"$EVENTS"
}

stop() {
    log "monitor stop requested"
    exit 0
}

trap stop INT TERM

log "monitor started: baseline_completed=$baseline_completed \
next_progress=$next_progress expected_cells=$EXPECTED_CELLS \
expected_workers=$EXPECTED_WORKERS aws_profile=$AWS_PROFILE \
interval=${POLL_SECONDS}s"

while true; do
    poll_count=$((poll_count + 1))
    app_status="$(
        aws sagemaker describe-app \
            --region "$REGION" \
            --domain-id "$DOMAIN_ID" \
            --space-name "$SPACE_NAME" \
            --app-type "$APP_TYPE" \
            --app-name "$APP_NAME" \
            --query Status \
            --output text 2>>"$LOG"
    )"
    app_rc=$?

    if [[ $app_rc -ne 0 ]]; then
        app_status="describe-error"
    fi
    if [[ "$app_status" != "$last_app_status" ]]; then
        log "app status: ${last_app_status:-unknown} -> $app_status"
        last_app_status="$app_status"
    fi
    if [[ "$app_status" != "InService" ]]; then
        event "ALERT app_status=$app_status; app supervisor owns recovery"
        sleep "$POLL_SECONDS"
        continue
    fi

    snapshot="$(
        "${SSH[@]}" bash -s -- \
            "$REMOTE_ROOT" "$SUMMARY" <<'REMOTE' 2>>"$LOG"
set -uo pipefail

root="$1"
summary="$2"
test -r "$summary" || exit 20
summary_fields="$(
    jq -r '
        [
            .counts.completed,
            .counts.improved,
            .counts.not_improved,
            .counts.failed,
            .counts.running,
            .counts.pending,
            (.counts.incomplete // 0),
            ([
                .cell_states[]
                | select(.state == "completed")
                | select(any(.. | numbers; isnan or isinfinite))
            ] | length)
        ]
        | @tsv
    ' "$summary"
)" || exit 21
fstype="$(findmnt -T "$root" -o FSTYPE -n 2>/dev/null || true)"
runner_pid="$(pgrep -f '[s]cripts/run_benchmark_matrix.py' | head -n 1)"
runner_cwd=""
if [[ -n "$runner_pid" ]]; then
    runner_cwd="$(readlink -f "/proc/$runner_pid/cwd" 2>/dev/null || true)"
fi
runner_pid="${runner_pid:-missing}"
runner_cwd="${runner_cwd:-missing}"
fstype="${fstype:-unknown}"
apps="$(
    nvidia-smi \
        --query-compute-apps=gpu_uuid,pid \
        --format=csv,noheader 2>/dev/null || true
)"
if [[ -n "$apps" ]]; then
    app_count="$(printf '%s\n' "$apps" | sed '/^[[:space:]]*$/d' | wc -l)"
    uuid_count="$(
        printf '%s\n' "$apps" |
            cut -d, -f1 |
            sed 's/[[:space:]]//g' |
            sort -u |
            wc -l
    )"
else
    app_count=0
    uuid_count=0
fi
cvds=""
cells=""
bad_cell_cwd=0
for pid in $(printf '%s\n' "$apps" | cut -d, -f2 | tr -d ' '); do
    [[ -n "$pid" ]] || continue
    cvd="$(
        tr '\0' '\n' <"/proc/$pid/environ" 2>/dev/null |
            sed -n 's/^CUDA_VISIBLE_DEVICES=//p'
    )"
    cell="$(
        tr '\0' ' ' <"/proc/$pid/cmdline" 2>/dev/null |
            sed -n 's/.*--cell-id \([^ ]*\).*/\1/p'
    )"
    cwd="$(readlink -f "/proc/$pid/cwd" 2>/dev/null || true)"
    cvds="${cvds}${cvd},"
    cells="${cells}${pid}:${cvd}:${cell}|"
    case "$cwd" in
        "$root"|/mnt/custom-file-systems/efs/*/workspace/TimeRAF) ;;
        *) bad_cell_cwd=$((bad_cell_cwd + 1)) ;;
    esac
done
cvd_count="$(
    printf '%s' "$cvds" |
        tr ',' '\n' |
        sed '/^$/d' |
        sort -u |
        wc -l
)"
cells="${cells:-none}"
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$summary_fields" "$fstype" "$runner_pid" "$runner_cwd" \
    "$app_count" "$uuid_count" "$cvd_count" "$bad_cell_cwd" "$cells"
REMOTE
    )"
    snapshot_rc=$?

    if [[ $snapshot_rc -ne 0 ]]; then
        event "ALERT snapshot_failed rc=$snapshot_rc"
        sleep "$POLL_SECONDS"
        continue
    fi

    IFS=$'\t' read -r \
        completed improved not_improved failed running pending incomplete \
        nonfinite_completed \
        fstype runner_pid runner_cwd app_count uuid_count cvd_count \
        bad_cell_cwd cells <<<"$snapshot"

    health_reason=""
    [[ "$fstype" == "nfs4" ]] || health_reason+=" fstype=$fstype"
    [[ "$runner_pid" != "missing" ]] || health_reason+=" runner=missing"
    [[ "$runner_cwd" == /mnt/custom-file-systems/efs/*/workspace/TimeRAF ]] ||
        health_reason+=" runner_cwd=$runner_cwd"
    if ((pending > 0)); then
        [[ "$app_count" == "$EXPECTED_WORKERS" ]] ||
            health_reason+=" gpu_processes=$app_count"
        [[ "$uuid_count" == "$EXPECTED_WORKERS" ]] ||
            health_reason+=" gpu_uuids=$uuid_count"
        [[ "$cvd_count" == "$EXPECTED_WORKERS" ]] ||
            health_reason+=" cuda_visible_ids=$cvd_count"
    fi
    [[ "$bad_cell_cwd" == "0" ]] ||
        health_reason+=" bad_cell_cwd=$bad_cell_cwd"

    if [[ -n "$health_reason" ]]; then
        health_bad_streak=$((health_bad_streak + 1))
        if ((health_bad_streak >= 2 && health_alerted == 0)); then
            event "ALERT health_anomaly:${health_reason} \
completed=$completed cells=$cells"
            health_alerted=1
        fi
    else
        if ((health_alerted == 1)); then
            event "RECOVERY health_ok completed=$completed cells=$cells"
        fi
        health_bad_streak=0
        health_alerted=0
    fi

    if ((failed > 0 && failed != last_failed)); then
        event "ALERT failures=$failed completed=$completed cells=$cells"
    fi
    last_failed=$failed

    if ((nonfinite_completed > 0 &&
        nonfinite_completed != last_nonfinite_completed)); then
        event "ALERT numerical_integrity nonfinite_completed=$nonfinite_completed \
completed=$completed incomplete=$incomplete"
    fi
    last_nonfinite_completed=$nonfinite_completed

    if ((completed >= next_progress)); then
        event "PROGRESS completed=$completed improved=$improved \
not_improved=$not_improved failed=$failed running=$running pending=$pending \
cells=$cells"
        while ((next_progress <= completed)); do
            next_progress=$((next_progress + PROGRESS_STEP))
        done
    fi

    terminal_count=$((completed + failed + incomplete))
    if ((terminal_count >= EXPECTED_CELLS && running == 0 && pending == 0)); then
        if ((failed == 0 && incomplete == 0 &&
            nonfinite_completed == 0)); then
            event "COMPLETE completed=$completed improved=$improved \
not_improved=$not_improved failed=$failed running=$running pending=$pending"
            exit 0
        fi
        if ((failed == 0)); then
            event "COMPLETE_REQUIRES_NUMERICAL_RECOVERY \
completed=$completed improved=$improved not_improved=$not_improved \
failed=$failed incomplete=$incomplete \
nonfinite_completed=$nonfinite_completed"
            exit 0
        fi
        event "TERMINAL_WITH_FAILURES completed=$completed failed=$failed \
incomplete=$incomplete nonfinite_completed=$nonfinite_completed"
        exit 1
    fi

    if ((poll_count == 1 || poll_count % 10 == 0 ||
        completed != last_completed)); then
        log "snapshot completed=$completed improved=$improved \
not_improved=$not_improved failed=$failed running=$running pending=$pending \
incomplete=$incomplete nonfinite_completed=$nonfinite_completed \
runner=$runner_pid gpu_processes=$app_count cells=$cells"
    fi
    last_completed=$completed
    sleep "$POLL_SECONDS"
done
