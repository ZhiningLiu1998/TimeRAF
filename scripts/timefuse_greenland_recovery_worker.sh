#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${TIMERAF_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONTROLLER_REVISION="${TIMERAF_RECOVERY_CONTROLLER_REVISION:?set TIMERAF_RECOVERY_CONTROLLER_REVISION}"
RECOVERY_REVISION="${TIMERAF_GREENLAND_RECOVERY_REVISION:-3cf7b3559c88dfa57bf2f62aba4f5cf5ed61fe94}"
A10G_RECOVERY_REVISION="${TIMERAF_A10G_RECOVERY_REVISION:-cbd4481e94cee8c6ae21edf5258ac29c58eceeea}"
METHOD_REVISION="${TIMERAF_METHOD_REVISION:-d9be338}"
PRIMARY_RUN_ID="${TIMERAF_PRIMARY_GREENLAND_RUN_ID:-full-matrix-a100-9974eac-20260731}"
EXACT_RUN_ID="${TIMERAF_EXACT_RUN_ID:-numerical-recovery-exact-3cf7b35-20260801}"
FALLBACK_RUN_ID="${TIMERAF_FALLBACK_RUN_ID:-numerical-recovery-fallback-3cf7b35-20260801}"
EXACT_SPEC_SHA="${TIMERAF_EXACT_SPEC_SHA:-5bc82aa56da7683fd5021d25f6ec6ecd93a1a4843097dffe55e4f0d93e1739f2}"
FALLBACK_SPEC_SHA="${TIMERAF_FALLBACK_SPEC_SHA:-a676be6076d700b90c78eadbcc1a7dd5a08ba3f8a8100ac9f0d9f1f2f979fb9f}"
EXACT_PREPARATION_SHA="${TIMERAF_EXACT_PREPARATION_SHA256:?set TIMERAF_EXACT_PREPARATION_SHA256}"
FALLBACK_PREPARATION_SHA="${TIMERAF_FALLBACK_PREPARATION_SHA256:?set TIMERAF_FALLBACK_PREPARATION_SHA256}"
DRY_RUN_AUDIT_SHA="${TIMERAF_DRY_RUN_AUDIT_SHA256:?set TIMERAF_DRY_RUN_AUDIT_SHA256}"
IMAGE_URI="${TIMERAF_RECOVERY_IMAGE_URI:-<ECR_REGISTRY>/timeraf-greenland@sha256:2417e94636d74e455f8bfdc97d7f7e4ff9022107e6ea7b2ebc4a4295e994d25d}"
export AWS_PROFILE="${TIMERAF_AWS_PROFILE:-timeraf-modeldev}"
GREENLAND_LOG_PROFILE="${TIMERAF_GREENLAND_LOG_PROFILE:-timeraf-greenland-console}"
REGION="${TIMERAF_REGION:-us-east-1}"
DOMAIN_ID="${TIMERAF_DOMAIN_ID:-<SAGEMAKER_DOMAIN_ID>}"
SPACE_NAME="${TIMERAF_SPACE_NAME:-<DEV_SPACE>}"
SPACE_ARN="arn:aws:sagemaker:${REGION}:<AWS_ACCOUNT_ID>:space/${DOMAIN_ID}/${SPACE_NAME}"
CONNECT_SCRIPT="$REPO_ROOT/scripts/sagemaker_connect.sh"
KNOWN_HOSTS="/tmp/timeraf-sagemaker-known-hosts"
LOCAL_PYTHON="${TIMERAF_LOCAL_PYTHON:-$REPO_ROOT/.venv/bin/python}"
LOCAL_CONTROLLER_ROOT="/tmp/timeraf-greenland-recovery-controller-$CONTROLLER_REVISION"
LOG="/tmp/timeraf-greenland-recovery-worker.log"
EVENTS="/tmp/timeraf-greenland-recovery-events.log"
REMOTE_ROOT="/home/sagemaker-user/user-default-efs/workspace/TimeRAF"
REMOTE_PREP="$REMOTE_ROOT/operations/numerical-recovery-greenland-$RECOVERY_REVISION"
REMOTE_CONTROLLER_OPS="$REMOTE_ROOT/operations/greenland-recovery-controller-$CONTROLLER_REVISION"
REMOTE_CONTROLLER_SOURCE="$REMOTE_CONTROLLER_OPS/source"
REMOTE_A10G="$REMOTE_ROOT/outputs/numerical_recovery/a10g/$A10G_RECOVERY_REVISION"
REMOTE_PRIMARY_IMPORT="$REMOTE_ROOT/outputs/greenland_runs/$PRIMARY_RUN_ID"
REMOTE_IMPORT_ROOT="$REMOTE_ROOT/outputs/greenland_recovery_runs"
POLL_SECONDS="${TIMERAF_RECOVERY_GREENLAND_POLL_SECONDS:-60}"
PREFLIGHT_ONLY="${TIMERAF_RECOVERY_PREFLIGHT_ONLY:-0}"
REMOTE_MARKER="__TIMERAF_GREENLAND_RECOVERY_OK__"

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

for revision in "$CONTROLLER_REVISION" "$RECOVERY_REVISION"; do
    git -C "$REPO_ROOT" cat-file -e "$revision^{commit}"
    test "$(git -C "$REPO_ROOT" rev-parse "$revision^{commit}")" = "$revision"
done
test -x "$LOCAL_PYTHON"
case "$PREFLIGHT_ONLY" in
    0 | 1) ;;
    *) printf 'TIMERAF_RECOVERY_PREFLIGHT_ONLY must be 0 or 1\n' >&2; exit 2 ;;
esac
for digest in \
    "$EXACT_PREPARATION_SHA" \
    "$FALLBACK_PREPARATION_SHA" \
    "$DRY_RUN_AUDIT_SHA"; do
    [[ "$digest" =~ ^[0-9a-f]{64}$ ]] || {
        printf 'recovery evidence SHA-256 is invalid: %s\n' "$digest" >&2
        exit 2
    }
done
worker_sha="$(
    git -C "$REPO_ROOT" show \
        "$CONTROLLER_REVISION:scripts/timefuse_greenland_recovery_worker.sh" |
        sha256sum |
        cut -d' ' -f1
)"
test "$worker_sha" = "$(sha256sum "${BASH_SOURCE[0]}" | cut -d' ' -f1)"
mkdir -p "$LOCAL_CONTROLLER_ROOT"
git -C "$REPO_ROOT" archive "$CONTROLLER_REVISION" \
    scripts/check_greenland_capacity.py \
    scripts/verify_greenland_startup.py |
    tar -xf - -C "$LOCAL_CONTROLLER_ROOT"
log "worker started controller_revision=$CONTROLLER_REVISION recovery_revision=$RECOVERY_REVISION a10g_recovery_revision=$A10G_RECOVERY_REVISION"

while true; do
    materialized="$(
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
    if [[ "$(tail -n 1 <<<"$materialized")" == "$REMOTE_MARKER" ]]; then
        break
    fi
    log "waiting to materialize pinned controller source in canonical EFS"
    sleep "$POLL_SECONDS"
done

control_identity="$(
    "${SSH[@]}" "
set -euo pipefail
ROOT='$REMOTE_ROOT'
PY=\"\$ROOT/.venv-greenland-control/bin/python\"
test \"\$(findmnt -T \"\$ROOT\" -o FSTYPE -n)\" = nfs4
test -x \"\$PY\"
case \"\$PY\" in
    \"\$ROOT\"/*) ;;
    *) exit 2 ;;
esac
\"\$PY\" - <<'PY'
import importlib.metadata as metadata
import json
from torchx.runner import get_runner

payload = {
    'launcher': metadata.version('amzn-greenland-torchx-launcher'),
    'torchx': metadata.version('torchx-nightly'),
    'schedulers': get_runner().scheduler_backends(),
}
if (
    payload['launcher'] != '1.0.47'
    or payload['torchx'] != '2026.7.30'
    or 'greenland' not in payload['schedulers']
):
    raise SystemExit('Greenland control environment drifted')
print(json.dumps(payload, sort_keys=True))
PY
" 2>>"$LOG"
)"
if ! jq -e '
    .launcher == "1.0.47"
    and .torchx == "2026.7.30"
    and (.schedulers | index("greenland") != null)
' >/dev/null <<<"$control_identity"; then
    event "CONTROL_IDENTITY_FAILED controller_revision=$CONTROLLER_REVISION"
    exit 2
fi
printf '%s\n' "$control_identity" |
    "${SSH[@]}" "
set -eu
cat >'$REMOTE_CONTROLLER_OPS/control-identity.json.tmp'
mv '$REMOTE_CONTROLLER_OPS/control-identity.json.tmp' \
   '$REMOTE_CONTROLLER_OPS/control-identity.json'
"

audit_state="$(
    "${SSH[@]}" "
if test \"\$(sha256sum '$REMOTE_PREP/scheduler-dry-run-audit.json' 2>/dev/null |
        cut -d' ' -f1)\" = '$DRY_RUN_AUDIT_SHA'; then
    printf '%s\n' __TIMERAF_DRY_RUN_AUDIT_VALID__
else
    printf '%s\n' __TIMERAF_DRY_RUN_AUDIT_INVALID__
fi
" 2>>"$LOG" || true
)"
audit_state="$(tail -n 1 <<<"$audit_state")"
if [[ "$audit_state" != "__TIMERAF_DRY_RUN_AUDIT_VALID__" ]]; then
    event "DRY_RUN_AUDIT_INVALID recovery_revision=$RECOVERY_REVISION"
    exit 2
fi

while true; do
    readiness="$(
        "${SSH[@]}" "
set -u
ROOT='$REMOTE_ROOT'
PRIMARY='$REMOTE_PRIMARY_IMPORT'
A10G='$REMOTE_A10G'
LOCK=\"\$ROOT/ts_rag_outputs/timefuse_matrix_full/matrix.lock\"
test \"\$(findmnt -T \"\$ROOT\" -o FSTYPE -n)\" = nfs4 || {
    printf storage-not-efs
    exit 0
}
test -s \"\$PRIMARY/fetch_receipt.json\" || {
    printf a100-primary-not-imported
    exit 0
}
jq -e --slurpfile summary \"\$PRIMARY/recomputed_matrix_summary.json\" '
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
' \"\$PRIMARY/fetch_receipt.json\" >/dev/null 2>&1 || {
    printf a100-primary-incomplete
    exit 0
}
test \"\$(jq -r '.recomputed_matrix_summary.sha256' \"\$PRIMARY/fetch_receipt.json\")\" = \
    \"\$(sha256sum \"\$PRIMARY/recomputed_matrix_summary.json\" | cut -d' ' -f1)\" || {
    printf a100-primary-summary-drift
    exit 0
}
test \"\$(jq -r '.recomputed_matrix_summary.size_bytes' \"\$PRIMARY/fetch_receipt.json\")\" = \
    \"\$(stat -c %s \"\$PRIMARY/recomputed_matrix_summary.json\")\" || {
    printf a100-primary-summary-drift
    exit 0
}
test \"\$(jq -r '.counts.completed' \"\$ROOT/ts_rag_outputs/timefuse_matrix_full/matrix_d9be338_summary.json\" 2>/dev/null)\" = 585 || {
    printf a10g-primary-incomplete
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
jq -e '
    .counts.expected == 9
    and ((.counts.completed + .counts.failed) == 9)
    and .counts.pending == 0
    and .counts.running == 0
    and .counts.incomplete == 0
' \"\$A10G/exact/matrix_summary.json\" >/dev/null 2>&1 || {
    printf a10g-exact-incomplete
    exit 0
}
jq -e '
    .all_completed == true
    and .counts.expected == 9
    and .counts.completed == 9
    and .counts.failed == 0
    and .counts.pending == 0
    and .counts.running == 0
    and .counts.incomplete == 0
' \"\$A10G/fallback-v1/matrix_summary.json\" >/dev/null 2>&1 || {
    printf a10g-fallback-incomplete
    exit 0
}
test \"\$(jq -r '.recovery_revision' \"\$A10G/exact/recovery_metadata.json\")\" = '$A10G_RECOVERY_REVISION' || {
    printf a10g-exact-identity-drift
    exit 0
}
test \"\$(jq -r '.recovery_revision' \"\$A10G/fallback-v1/recovery_metadata.json\")\" = '$A10G_RECOVERY_REVISION' || {
    printf a10g-fallback-identity-drift
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
    log "waiting for recovery submission prerequisites: ${readiness:-ssh-error}"
    sleep "$POLL_SECONDS"
done
event "PREREQUISITES_READY primary_run_id=$PRIMARY_RUN_ID"

submit_profile() {
    local profile="$1"
    local run_id="$2"
    local spec_sha="$3"
    local staging_name="$4"
    local receipt="$REMOTE_PREP/submission-$profile.json"
    local output="$REMOTE_PREP/submission-$profile.stdout.json"
    local spec="$REMOTE_PREP/staging-$staging_name/job-spec-$spec_sha.json"
    local spec_uri="s3://<DEV_BUCKET>/timeraf/greenland/specs/$spec_sha.json"
    local s3_receipt_key="timeraf/greenland/runs/$run_id/control/submission_receipt.json"
    local capacity_local="/tmp/timeraf-greenland-recovery-capacity-$profile.json"
    local capacity_remote="$REMOTE_PREP/capacity-pre-submit-$profile.json"
    local expected_s3_receipt_sha
    local observed_s3_receipt_sha
    local capacity_sha
    local preparation_sha
    local preparation_state
    local receipt_state
    local submitted

    case "$profile" in
        exact) preparation_sha="$EXACT_PREPARATION_SHA" ;;
        fallback-v1) preparation_sha="$FALLBACK_PREPARATION_SHA" ;;
        *) event "PREPARATION_PROFILE_INVALID profile=$profile"; return 3 ;;
    esac
    preparation_state="$(
        "${SSH[@]}" "
if test \"\$(sha256sum '$REMOTE_PREP/prepare-$profile.json' 2>/dev/null |
        cut -d' ' -f1)\" = '$preparation_sha' &&
    jq -e '
        .job_spec_sha256 == \"$spec_sha\"
        and .numerical_integrity_source_evidence.contract ==
            \"explicit-numerical-integrity-error-v1\"
        and .numerical_integrity_source_evidence.source_revision ==
            \"$RECOVERY_REVISION\"
        and .numerical_integrity_source_evidence.entrypoint ==
            \"scripts/run_benchmark_cell.py\"
        and (.numerical_integrity_source_evidence.entrypoint_sha256
             | test(\"^[0-9a-f]{64}$\"))
        and .numerical_integrity_source_evidence.passed == true
    ' '$REMOTE_PREP/prepare-$profile.json' >/dev/null 2>&1; then
    printf '%s\n' __TIMERAF_PREPARATION_VALID__
else
    printf '%s\n' __TIMERAF_PREPARATION_INVALID__
fi
" 2>>"$LOG" || true
    )"
    preparation_state="$(tail -n 1 <<<"$preparation_state")"
    if [[ "$preparation_state" != "__TIMERAF_PREPARATION_VALID__" ]]; then
        event "PREPARATION_INVALID profile=$profile run_id=$run_id"
        return 3
    fi

    receipt_state="$(
        "${SSH[@]}" "
if test ! -e '$receipt'; then
    printf '%s\n' __TIMERAF_SUBMISSION_RECEIPT_ABSENT__
elif jq -e '
    .dry_run == false
    and .run_id == \"$run_id\"
    and .job_kind == \"supplemental_numerical_recovery\"
    and .instance_type == \"p4de.24xlarge\"
    and .instance_count == 1
    and .reserved_gpus_per_host == 8
    and .processes_per_host == 8
    and .total_gpus == 8
    and .world_size == 8
    and .inactive_reserved_gpus == 0
    and (.job_uuid | type == \"string\" and length > 0)
' '$receipt' >/dev/null 2>&1; then
    printf '%s\n' __TIMERAF_SUBMISSION_RECEIPT_VALID__
else
    printf '%s\n' __TIMERAF_SUBMISSION_RECEIPT_INVALID__
fi
" 2>>"$LOG" || true
    )"
    receipt_state="$(tail -n 1 <<<"$receipt_state")"
    if [[ "$receipt_state" == "__TIMERAF_SUBMISSION_RECEIPT_VALID__" ]]; then
        expected_s3_receipt_sha="$(
            "${SSH[@]}" "jq -r '
                .control_records[]
                | select(.s3_uri == \"s3://<DEV_BUCKET>/$s3_receipt_key\")
                | .sha256
            ' '$receipt'"
        )"
        observed_s3_receipt_sha="$(
            aws --profile "$AWS_PROFILE" --region "$REGION" s3api head-object \
                --bucket <DEV_BUCKET> \
                --key "$s3_receipt_key" \
                --query 'Metadata.sha256' \
                --output text 2>>"$LOG" || true
        )"
        if [[ ! "$expected_s3_receipt_sha" =~ ^[0-9a-f]{64}$ ||
            "$observed_s3_receipt_sha" != "$expected_s3_receipt_sha" ]]; then
            event "SUBMISSION_RECEIPT_DRIFT profile=$profile run_id=$run_id"
            return 3
        fi
        event "SUBMISSION_REUSED profile=$profile run_id=$run_id receipt=$receipt"
        return 0
    fi
    if [[ "$receipt_state" == "__TIMERAF_SUBMISSION_RECEIPT_INVALID__" ]]; then
        event "SUBMISSION_RECEIPT_INVALID profile=$profile run_id=$run_id"
        return 3
    fi
    if [[ "$receipt_state" != "__TIMERAF_SUBMISSION_RECEIPT_ABSENT__" ]]; then
        event "SUBMISSION_RECEIPT_STATUS_FAILED profile=$profile run_id=$run_id"
        return 3
    fi
    if aws --profile "$AWS_PROFILE" --region "$REGION" s3api head-object \
        --bucket <DEV_BUCKET> \
        --key "$s3_receipt_key" >/dev/null 2>&1; then
        event "SUBMISSION_RECONCILIATION_REQUIRED profile=$profile run_id=$run_id"
        return 3
    fi

    while true; do
        if "$LOCAL_PYTHON" "$LOCAL_CONTROLLER_ROOT/scripts/check_greenland_capacity.py" \
            --requested-hosts 1 \
            --output "$capacity_local" >>"$LOG" 2>&1 &&
            jq -e '
                .submission_eligible == true
                and .initiative == "feedml-sp-os"
                and .region == "us-east-1"
                and .instance_type == "p4de.24xlarge"
                and .runtime == "EKS"
                and .requested_hosts == 1
            ' "$capacity_local" >/dev/null; then
            break
        fi
        log "capacity preflight failed for profile=$profile; retrying"
        sleep 300
    done
    "${SSH[@]}" "
set -eu
cat >'$capacity_remote.tmp'
mv '$capacity_remote.tmp' '$capacity_remote'
" <"$capacity_local"
    capacity_sha="$(sha256sum "$capacity_local" | cut -d' ' -f1)"
    event "CAPACITY_VERIFIED profile=$profile run_id=$run_id sha256=$capacity_sha"

    submitted="$(
        "${SSH[@]}" "
set -euo pipefail
ROOT='$REMOTE_ROOT'
PY=\"\$ROOT/.venv-greenland-control/bin/python\"
unset AWS_PROFILE
test \"\$(sha256sum '$spec' | cut -d' ' -f1)\" = '$spec_sha'
jq -e '
    .submission_eligible == true
    and .initiative == \"feedml-sp-os\"
    and .instance_type == \"p4de.24xlarge\"
    and .runtime == \"EKS\"
    and .requested_hosts == 1
' '$capacity_remote' >/dev/null
jq -e --arg profile '$profile' --arg run_id '$run_id' \
    '.all_passed == true
     and any(.records[];
         .profile == \$profile
         and .run_id == \$run_id
         and .instance_count == 1
         and .reserved_gpus_per_host == 8
         and .processes_per_host == 8
         and .total_gpus == 8
         and .world_size == 8
         and .inactive_reserved_gpus == 0
         and .manifest_verified == true
         and .s3_metadata_verified == true)' \
    '$REMOTE_PREP/scheduler-dry-run-audit.json' >/dev/null
\"\$PY\" '$REMOTE_CONTROLLER_SOURCE/scripts/submit_greenland_job.py' \
    --job-spec '$spec' \
    --job-spec-s3-uri '$spec_uri' \
    --image-uri '$IMAGE_URI' \
    --receipt '$receipt' \
    --submit \
    --confirm SUBMIT \
    >'$output.tmp'
mv '$output.tmp' '$output'
jq -e '
    .dry_run == false
    and .run_id == \"$run_id\"
    and .job_kind == \"supplemental_numerical_recovery\"
    and .job_spec_sha256 == \"$spec_sha\"
    and .image_uri == \"$IMAGE_URI\"
    and .instance_type == \"p4de.24xlarge\"
    and .instance_count == 1
    and .reserved_gpus_per_host == 8
    and .processes_per_host == 8
    and .total_gpus == 8
    and .world_size == 8
    and .inactive_reserved_gpus == 0
    and (.job_uuid | type == \"string\" and length > 0)
    and (.app_handle | type == \"string\" and length > 0)
' '$receipt' >/dev/null
cat '$receipt'
printf '%s\n' '$REMOTE_MARKER'
" 2>>"$LOG" || true
    )"
    if [[ "$(tail -n 1 <<<"$submitted")" != "$REMOTE_MARKER" ]]; then
        event "SUBMISSION_FAILED profile=$profile run_id=$run_id"
        return 4
    fi
    event "SUBMITTED profile=$profile run_id=$run_id receipt=$receipt"
}

submit_profile exact "$EXACT_RUN_ID" "$EXACT_SPEC_SHA" exact
submit_profile fallback-v1 "$FALLBACK_RUN_ID" "$FALLBACK_SPEC_SHA" fallback-v1

verify_startup() {
    local profile="$1"
    local run_id="$2"
    local receipt="$REMOTE_PREP/submission-$profile.json"
    local evidence_local="/tmp/timeraf-greenland-recovery-startup-$profile.json"
    local evidence_remote="$REMOTE_PREP/startup-$profile.json"
    local identity
    local job_uuid
    local job_name
    local submitted_unix
    local submitted_at

    identity="$("${SSH[@]}" "jq -r '[.job_uuid,.job_name,(.submitted_unix|tostring)] | @tsv' '$receipt'")"
    IFS=$'\t' read -r job_uuid job_name submitted_unix <<<"$identity"
    submitted_at="$(
        "$LOCAL_PYTHON" -c \
            'import datetime,sys; print(datetime.datetime.fromtimestamp(float(sys.argv[1]), datetime.timezone.utc).isoformat().replace("+00:00","Z"))' \
            "$submitted_unix"
    )"
    while true; do
        if "$LOCAL_PYTHON" "$LOCAL_CONTROLLER_ROOT/scripts/verify_greenland_startup.py" \
            --job-uuid "$job_uuid" \
            --job-name "$job_name" \
            --submitted-at "$submitted_at" \
            --aws-profile "$GREENLAND_LOG_PROFILE" \
            --poll-seconds 10 \
            --timeout-seconds 180 \
            --output "$evidence_local" >>"$LOG" 2>&1; then
            break
        fi
        if aws --profile "$AWS_PROFILE" --region "$REGION" s3api head-object \
            --bucket <DEV_BUCKET> \
            --key "timeraf/greenland/runs/$run_id/upload_failed.json" \
            >/dev/null 2>&1; then
            event "STARTUP_FAILED profile=$profile run_id=$run_id"
            return 5
        fi
        log "waiting for eight-GPU startup evidence profile=$profile run_id=$run_id"
        sleep "$POLL_SECONDS"
    done
    "${SSH[@]}" "
set -eu
cat >'$evidence_remote.tmp'
mv '$evidence_remote.tmp' '$evidence_remote'
" <"$evidence_local"
    event "STARTUP_VERIFIED profile=$profile run_id=$run_id job_uuid=$job_uuid"
}

verify_startup exact "$EXACT_RUN_ID"
verify_startup fallback-v1 "$FALLBACK_RUN_ID"

import_profile() {
    local profile="$1"
    local run_id="$2"
    local destination="$REMOTE_IMPORT_ROOT/$run_id"
    local receipt="$destination/fetch_receipt.json"
    local status
    local imported
    local receipt_state

    receipt_state="$(
        "${SSH[@]}" "
if test ! -e '$receipt'; then
    printf '%s\n' __TIMERAF_IMPORT_RECEIPT_ABSENT__
elif jq -e '
    .run_id == \"$run_id\"
    and .profile == \"$profile\"
    and .job_kind == \"supplemental_numerical_recovery\"
    and .method_revision == \"$METHOD_REVISION\"
    and .source_revision == \"$RECOVERY_REVISION\"
    and .launcher_revision == \"$RECOVERY_REVISION\"
    and .recovery_revision == \"$RECOVERY_REVISION\"
    and .topology_gate_passed == true
    and .matrix_counts.expected == 9
    and ((.matrix_counts.completed + .matrix_counts.failed) == 9)
    and .matrix_counts.pending == 0
    and .matrix_counts.running == 0
    and .matrix_counts.incomplete == 0
    and (.failed_cell_errors | type == \"array\")
    and ((.failed_cell_errors | length) == .matrix_counts.failed)
    and (\"$profile\" != \"exact\"
         or all(.failed_cell_errors[];
                .error_type == \"NumericalIntegrityError\"))
' '$receipt' >/dev/null 2>&1; then
    printf '%s\n' __TIMERAF_IMPORT_RECEIPT_VALID__
else
    printf '%s\n' __TIMERAF_IMPORT_RECEIPT_INVALID__
fi
" 2>>"$LOG" || true
    )"
    receipt_state="$(tail -n 1 <<<"$receipt_state")"
    if [[ "$receipt_state" == "__TIMERAF_IMPORT_RECEIPT_VALID__" ]]; then
        event "IMPORT_REUSED profile=$profile run_id=$run_id destination=$destination"
        return 0
    fi
    if [[ "$receipt_state" == "__TIMERAF_IMPORT_RECEIPT_INVALID__" ]]; then
        event "IMPORT_RECEIPT_INVALID profile=$profile run_id=$run_id"
        return 7
    fi
    if [[ "$receipt_state" != "__TIMERAF_IMPORT_RECEIPT_ABSENT__" ]]; then
        event "IMPORT_RECEIPT_STATUS_FAILED profile=$profile run_id=$run_id"
        return 7
    fi

    while true; do
        status="$(
            "${SSH[@]}" "
set -eu
unset AWS_PROFILE
'$REMOTE_ROOT/.venv-greenland-control/bin/python' \
    '$REMOTE_CONTROLLER_SOURCE/scripts/greenland_run_status.py' \
    --run-id '$run_id' \
    --region '$REGION' \
    --expected-job-kind supplemental_numerical_recovery \
    --expected-method-revision '$METHOD_REVISION' \
    --expected-source-revision '$RECOVERY_REVISION' \
    --expected-launcher-revision '$RECOVERY_REVISION' \
    --expected-recovery-revision '$RECOVERY_REVISION' \
    --expected-recovery-profile '$profile'
" 2>>"$LOG" || true
        )"
        if [[ -z "$status" ]] || ! jq -e --arg run_id "$run_id" '
            type == "object"
            and .run_id == $run_id
            and (.upload_complete | type == "boolean")
            and (.upload_failed | type == "boolean")
            and ((.upload_complete and .upload_failed) | not)
        ' >/dev/null 2>&1 <<<"$status"; then
            log "waiting for valid recovery status profile=$profile"
            sleep "$POLL_SECONDS"
            continue
        fi
        if jq -e '.upload_complete == true' >/dev/null <<<"$status"; then
            break
        fi
        if jq -e '.upload_failed == true' >/dev/null <<<"$status"; then
            event "UPLOAD_FAILED profile=$profile run_id=$run_id marker=$(jq -c . <<<"$status")"
            return 6
        fi
        log "waiting for recovery upload profile=$profile run_id=$run_id"
        sleep "$POLL_SECONDS"
    done

    imported="$(
        "${SSH[@]}" "
set -euo pipefail
ROOT='$REMOTE_ROOT'
PY=\"\$ROOT/.venv-gpu312/bin/python\"
unset AWS_PROFILE
test \"\$(sha256sum '$REMOTE_CONTROLLER_SOURCE/docs/timefuse_numerical_recovery_protocol.json' | cut -d' ' -f1)\" = \
    '1320f3eca815a93a6500caadbf7e526610379b2af5f6976f1db94f91ed259d14'
\"\$PY\" '$REMOTE_CONTROLLER_SOURCE/scripts/fetch_greenland_numerical_recovery.py' \
    --run-id '$run_id' \
    --project-root \"\$ROOT\" \
    --destination '$destination' \
    --manifest '$REMOTE_CONTROLLER_SOURCE/docs/timefuse_experiment_manifest.jsonl' \
    --protocol '$REMOTE_CONTROLLER_SOURCE/docs/timefuse_numerical_recovery_protocol.json' \
    --receipt '$receipt' \
    --region '$REGION'
jq -e '
    .run_id == \"$run_id\"
    and .profile == \"$profile\"
    and .job_kind == \"supplemental_numerical_recovery\"
    and .method_revision == \"$METHOD_REVISION\"
    and .source_revision == \"$RECOVERY_REVISION\"
    and .launcher_revision == \"$RECOVERY_REVISION\"
    and .recovery_revision == \"$RECOVERY_REVISION\"
    and .topology_gate_passed == true
    and .matrix_counts.expected == 9
    and ((.matrix_counts.completed + .matrix_counts.failed) == 9)
    and .matrix_counts.pending == 0
    and .matrix_counts.running == 0
    and .matrix_counts.incomplete == 0
    and (.failed_cell_errors | type == \"array\")
    and ((.failed_cell_errors | length) == .matrix_counts.failed)
    and (\"$profile\" != \"exact\"
         or all(.failed_cell_errors[];
                .error_type == \"NumericalIntegrityError\"))
' '$receipt' >/dev/null
test \"\$(jq -r '.recomputed_matrix_summary.sha256' '$receipt')\" = \
    \"\$(sha256sum '$destination/recomputed_matrix_summary.json' | cut -d' ' -f1)\"
printf '%s\n' '$REMOTE_MARKER'
" 2>>"$LOG" || true
    )"
    if [[ "$(tail -n 1 <<<"$imported")" != "$REMOTE_MARKER" ]]; then
        event "IMPORT_FAILED profile=$profile run_id=$run_id destination=$destination"
        return 7
    fi
    event "IMPORTED profile=$profile run_id=$run_id destination=$destination"
}

import_profile exact "$EXACT_RUN_ID"
import_profile fallback-v1 "$FALLBACK_RUN_ID"
event "COMPLETE controller_revision=$CONTROLLER_REVISION exact_run_id=$EXACT_RUN_ID fallback_run_id=$FALLBACK_RUN_ID"
failed=0
