#!/usr/bin/env bash
set -uo pipefail

REGION="${AWS_REGION:-us-east-1}"
export AWS_PROFILE="${TIMERAF_AWS_PROFILE:-timeraf-modeldev}"
DOMAIN_ID="${TIMERAF_DOMAIN_ID:-<SAGEMAKER_DOMAIN_ID>}"
SPACE_NAME="${TIMERAF_SPACE_NAME:-<DEV_SPACE>}"
APP_TYPE="${TIMERAF_APP_TYPE:-JupyterLab}"
APP_NAME="${TIMERAF_APP_NAME:-default}"
INSTANCE_TYPE="${TIMERAF_INSTANCE_TYPE:-ml.g5.12xlarge}"
IMAGE_ARN="${TIMERAF_IMAGE_ARN:-arn:aws:sagemaker:us-east-1:<AWS_ACCOUNT_ID_IMAGES>:image/sagemaker-distribution-gpu}"
IMAGE_ALIAS="${TIMERAF_IMAGE_ALIAS:-4.2.2}"
LIFECYCLE_CONFIG_ARN="${TIMERAF_LIFECYCLE_CONFIG_ARN:-arn:aws:sagemaker:us-east-1:<AWS_ACCOUNT_ID>:studio-lifecycle-config/<STUDIO_LIFECYCLE_CONFIG>}"
POLL_SECONDS="${TIMERAF_SUPERVISOR_POLL_SECONDS:-90}"
HEARTBEAT_POLLS="${TIMERAF_SUPERVISOR_HEARTBEAT_POLLS:-10}"

log() {
    printf '%s %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$*"
}

stop() {
    log "supervisor stop requested"
    exit 0
}

trap stop INT TERM

command -v aws >/dev/null
command -v jq >/dev/null

resource_spec="$(
    printf \
        'SageMakerImageArn=%s,SageMakerImageVersionAlias=%s,InstanceType=%s,LifecycleConfigArn=%s' \
        "$IMAGE_ARN" \
        "$IMAGE_ALIAS" \
        "$INSTANCE_TYPE" \
        "$LIFECYCLE_CONFIG_ARN"
)"

create_app() {
    local output
    if output="$(
        aws sagemaker create-app \
            --region "$REGION" \
            --domain-id "$DOMAIN_ID" \
            --space-name "$SPACE_NAME" \
            --app-type "$APP_TYPE" \
            --app-name "$APP_NAME" \
            --resource-spec "$resource_spec" \
            --output json 2>&1
    )"; then
        log "create-app accepted: ${output//$'\n'/ }"
        return 0
    fi
    log "create-app will retry: ${output//$'\n'/ }"
    return 1
}

delete_failed_app() {
    local output
    if output="$(
        aws sagemaker delete-app \
            --region "$REGION" \
            --domain-id "$DOMAIN_ID" \
            --space-name "$SPACE_NAME" \
            --app-type "$APP_TYPE" \
            --app-name "$APP_NAME" 2>&1
    )"; then
        log "delete-app accepted for failed App"
        return 0
    fi
    log "delete-app will retry: ${output//$'\n'/ }"
    return 1
}

last_status=""
poll_count=0
log "supervisor started: domain=$DOMAIN_ID space=$SPACE_NAME \
app=$APP_TYPE/$APP_NAME instance=$INSTANCE_TYPE image_alias=$IMAGE_ALIAS \
aws_profile=$AWS_PROFILE interval=${POLL_SECONDS}s"

while true; do
    describe_output="$(
        aws sagemaker describe-app \
            --region "$REGION" \
            --domain-id "$DOMAIN_ID" \
            --space-name "$SPACE_NAME" \
            --app-type "$APP_TYPE" \
            --app-name "$APP_NAME" \
            --output json 2>&1
    )"
    describe_rc=$?
    poll_count=$((poll_count + 1))

    if [[ $describe_rc -ne 0 ]]; then
        if [[ "$describe_output" == *"ResourceNotFound"* || "$describe_output" == *"not found"* ]]; then
            if [[ "$last_status" != "not-found" ]]; then
                log "state transition: ${last_status:-unknown} -> not-found"
                last_status="not-found"
            fi
            create_app
        else
            log "describe-app will retry: ${describe_output//$'\n'/ }"
        fi
        sleep "$POLL_SECONDS"
        continue
    fi

    status="$(jq -r '.Status' <<<"$describe_output")"
    if [[ "$status" != "$last_status" ]]; then
        log "state transition: ${last_status:-unknown} -> $status"
        last_status="$status"
    elif ((poll_count % HEARTBEAT_POLLS == 0)); then
        log "heartbeat: status=$status"
    fi

    case "$status" in
        InService)
            actual_instance="$(jq -r '.ResourceSpec.InstanceType // "unknown"' <<<"$describe_output")"
            actual_image="$(jq -r '.ResourceSpec.SageMakerImageArn // "unknown"' <<<"$describe_output")"
            actual_alias="$(jq -r '.ResourceSpec.SageMakerImageVersionAlias // "unknown"' <<<"$describe_output")"
            if [[ "$actual_instance" != "$INSTANCE_TYPE" || "$actual_image" != "$IMAGE_ARN" ]]; then
                log "configuration drift detected; leaving live App untouched: instance=$actual_instance image=$actual_image"
            elif [[ "$actual_alias" != "unknown" && "$actual_alias" != "$IMAGE_ALIAS" ]]; then
                log "image alias drift detected; leaving live App untouched: alias=$actual_alias"
            fi
            ;;
        Pending|Deleting)
            ;;
        Failed)
            delete_failed_app
            ;;
        Deleted)
            create_app
            ;;
        *)
            log "unrecognized state; leaving App untouched: status=$status"
            ;;
    esac

    sleep "$POLL_SECONDS"
done
