#!/usr/bin/env bash

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STATE="${TIMERAF_NATIVE_MONITOR_STATE:-/tmp/timeraf-native-retrieval.state}"
LOG="${TIMERAF_NATIVE_MONITOR_LOG:-/tmp/timeraf-native-retrieval.log}"
POLL_SECONDS="${TIMERAF_NATIVE_MONITOR_POLL_SECONDS:-600}"
SPACE_ARN="arn:aws:sagemaker:us-east-1:<AWS_ACCOUNT_ID>:space/<SAGEMAKER_DOMAIN_ID>/<P5_SPACE>"
PROFILE="${TIMERAF_AWS_PROFILE:-timeraf-modeldev}"
REMOTE_ROOT="/home/sagemaker-user/user-default-efs/workspace/TimeRAF"
REMOTE_FINAL="$REMOTE_ROOT/outputs/native_retrieval_baselines/d614258/ratd-full-p5/summary.json"
REMOTE_LIFECYCLE="run_native_ratd_sharded_lifecycle.sh"

SSH=(
  ssh
  -o StrictHostKeyChecking=no
  -o UserKnownHostsFile=/dev/null
  -o "ProxyCommand=$ROOT/scripts/sagemaker_connect.sh $SPACE_ARN $PROFILE"
  sagemaker-user@timeraf-p5
)

record_state() {
  local status="$1"
  local detail="$2"
  local temporary="${STATE}.tmp.$$"
  printf '%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "$status" "$detail" \
    > "$temporary"
  mv "$temporary" "$STATE"
}

mkdir -p "$(dirname "$STATE")" "$(dirname "$LOG")"
record_state "STARTING" "$REMOTE_FINAL"
printf '%s monitor started; poll_seconds=%s\n' \
  "$(date -u +%FT%TZ)" "$POLL_SECONDS" >> "$LOG"

while true; do
  remote_status="$(
    "${SSH[@]}" \
      "if [[ -f '$REMOTE_FINAL' ]]; then
         printf 'COMPLETE'
       elif pgrep -f '$REMOTE_LIFECYCLE' >/dev/null; then
         printf 'RUNNING'
       else
         printf 'FAILED'
       fi" 2>> "$LOG"
  )"
  ssh_status=$?
  recorded_at="$(date -u +%FT%TZ)"

  if [[ $ssh_status -ne 0 ]]; then
    record_state "CONNECTION_RETRY" "ssh_exit=$ssh_status"
    printf '%s connection retry; ssh_exit=%s\n' \
      "$recorded_at" "$ssh_status" >> "$LOG"
    sleep "$POLL_SECONDS"
    continue
  fi

  case "$remote_status" in
    COMPLETE)
      record_state "COMPLETE" "$REMOTE_FINAL"
      printf '%s COMPLETE %s\n' "$recorded_at" "$REMOTE_FINAL" >> "$LOG"
      exit 0
      ;;
    RUNNING)
      record_state "RUNNING" "$REMOTE_FINAL"
      ;;
    FAILED)
      record_state "FAILED" "lifecycle exited without $REMOTE_FINAL"
      printf '%s FAILED lifecycle exited without summary\n' \
        "$recorded_at" >> "$LOG"
      exit 1
      ;;
    *)
      record_state "CONNECTION_RETRY" "unexpected_status=$remote_status"
      printf '%s unexpected remote status: %s\n' \
        "$recorded_at" "$remote_status" >> "$LOG"
      ;;
  esac
  sleep "$POLL_SECONDS"
done
