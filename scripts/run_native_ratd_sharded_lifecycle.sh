#!/usr/bin/env bash

set -euo pipefail

ROOT="${TIMERAF_ROOT:-/home/sagemaker-user/user-default-efs/workspace/TimeRAF}"
PYTHON="$ROOT/.venv-native-ratd/bin/python"
LAUNCHER_SOURCE="$ROOT/operations/native-ratd-sharded-813cbcc/source"
FROZEN_SOURCE="$ROOT/operations/native-retrieval-baselines-d614258/source"
UPSTREAM="$ROOT/operations/native-retrieval-baselines-161c835/upstream/ratd"
DATA="$ROOT/operations/native-retrieval-baselines-161c835/ratd-assets/data/electricity.csv"
ORIGINAL_ROOT="$ROOT/outputs/native_retrieval_baselines/d614258/ratd-full-p5"
SHARDED_ROOT="$ROOT/outputs/native_retrieval_baselines/813cbcc/ratd-sharded-eval-p5"
OPERATION_ROOT="$ROOT/operations/native-ratd-sharded-813cbcc"
RUNNER="$LAUNCHER_SOURCE/scripts/run_native_ratd_sharded_eval.py"
FROZEN_RUNNER="$FROZEN_SOURCE/scripts/run_native_ratd_baseline.py"
PROTOCOL="$FROZEN_SOURCE/docs/native_retrieval_baseline_protocol.json"
CONFIG="$UPSTREAM/config/base_forecasting.yaml"
REFERENCE_MAP="$ORIGINAL_ROOT/reference_map.npz"
SOURCE_REVISION="d614258b6e5fd4ca71c4d814ab2fe10e2503399d"
LAUNCHER_REVISION="813cbcc0b4d962f5ddeeafe30a0f687eb82e1245"
UPSTREAM_COMMIT="719e4008f72e89544a621d97b5d1a69164fc14d3"
POLL_SECONDS="${TIMERAF_RATD_LIFECYCLE_POLL_SECONDS:-120}"

checkpoint_epoch() {
  "$PYTHON" -c \
    'import sys, torch; print(torch.load(sys.argv[1], map_location="cpu", weights_only=False).get("epoch", -1))' \
    "$1"
}

wait_for_epoch_100() {
  local method="$1"
  local checkpoint="$ORIGINAL_ROOT/$method/training_checkpoint.pt"
  while true; do
    local epoch
    epoch="$(checkpoint_epoch "$checkpoint" 2>/dev/null || printf '%s' -1)"
    printf '%s %s checkpoint epoch=%s\n' "$(date -u +%FT%TZ)" "$method" "$epoch"
    if [[ "$epoch" == "100" ]]; then
      return
    fi
    sleep "$POLL_SECONDS"
  done
}

terminate_original_worker() {
  local method="$1"
  local status="$ORIGINAL_ROOT/$method/status.json"
  local pid
  pid="$("$PYTHON" -c \
    'import json, sys; print(json.load(open(sys.argv[1]))["pid"])' \
    "$status")"
  if ! kill -0 "$pid" 2>/dev/null; then
    printf '%s original %s worker %s already exited\n' \
      "$(date -u +%FT%TZ)" "$method" "$pid"
    return
  fi
  printf '%s terminating original serial %s evaluation pid=%s\n' \
    "$(date -u +%FT%TZ)" "$method" "$pid"
  kill -TERM -- "-$pid"
  for _ in $(seq 1 60); do
    if ! kill -0 "$pid" 2>/dev/null; then
      return
    fi
    sleep 1
  done
  kill -KILL -- "-$pid"
}

wait_for_process_exit() {
  local pid="$1"
  while kill -0 "$pid" 2>/dev/null; do
    sleep 5
  done
}

common_args=(
  --runner "$FROZEN_RUNNER"
  --upstream-root "$UPSTREAM"
  --config "$CONFIG"
  --data "$DATA"
  --reference-map "$REFERENCE_MAP"
  --protocol "$PROTOCOL"
  --source-revision "$SOURCE_REVISION"
  --launcher-revision "$LAUNCHER_REVISION"
  --upstream-commit "$UPSTREAM_COMMIT"
)

mkdir -p "$OPERATION_ROOT/equivalence-final"

wait_for_epoch_100 csdi
terminate_original_worker csdi

"$PYTHON" "$RUNNER" verify \
  --method csdi \
  "${common_args[@]}" \
  --checkpoint "$ORIGINAL_ROOT/csdi/training_checkpoint.pt" \
  --origin-start 0 \
  --oracle-gpu 1 \
  --comparison-gpu 2 \
  --output "$OPERATION_ROOT/equivalence-final/csdi.json"

"$PYTHON" "$RUNNER" run \
  --method csdi \
  "${common_args[@]}" \
  --checkpoint "$ORIGINAL_ROOT/csdi/training_checkpoint.pt" \
  --output-root "$SHARDED_ROOT" \
  --gpu-ids 1,2,3,4,5,6,7 \
  --planner-gpu 1

wait_for_epoch_100 ratd
terminate_original_worker ratd

controller_pid="$(cat "$ORIGINAL_ROOT/controller.pid")"
wait_for_process_exit "$controller_pid"

if [[ -f "$ORIGINAL_ROOT/summary.json" ]]; then
  mv "$ORIGINAL_ROOT/summary.json" \
    "$ORIGINAL_ROOT/original-serial-evaluation-interrupted-summary.json"
fi

"$PYTHON" "$RUNNER" verify \
  --method ratd \
  "${common_args[@]}" \
  --checkpoint "$ORIGINAL_ROOT/ratd/training_checkpoint.pt" \
  --origin-start 0 \
  --oracle-gpu 0 \
  --comparison-gpu 1 \
  --output "$OPERATION_ROOT/equivalence-final/ratd.json"

"$PYTHON" "$RUNNER" run \
  --method ratd \
  "${common_args[@]}" \
  --checkpoint "$ORIGINAL_ROOT/ratd/training_checkpoint.pt" \
  --output-root "$SHARDED_ROOT" \
  --gpu-ids 0,1,2,3,4,5,6,7 \
  --planner-gpu 0

"$PYTHON" "$RUNNER" finalize \
  --ratd-result "$SHARDED_ROOT/ratd/result.json" \
  --ratd-receipt "$SHARDED_ROOT/ratd/receipt.json" \
  --csdi-result "$SHARDED_ROOT/csdi/result.json" \
  --csdi-receipt "$SHARDED_ROOT/csdi/receipt.json" \
  --reference-metadata "$ORIGINAL_ROOT/reference_map.json" \
  --training-topology "$ORIGINAL_ROOT/startup_topology.json" \
  --topology-output "$ORIGINAL_ROOT/sharded_startup_topology.json" \
  --output "$ORIGINAL_ROOT/summary.json" \
  --receipt-output "$ORIGINAL_ROOT/sharded_finalizer_receipt.json" \
  --training-eval-exit-policy intentional-post-epoch100-termination

printf '%s COMPLETE %s\n' \
  "$(date -u +%FT%TZ)" "$ORIGINAL_ROOT/summary.json"
